from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .contracts import EndpointClass, EndpointPrecision, OperatingState, OperatingStateSource
from .demo import DEFAULT_DEMO_DATABASE, DEFAULT_START_TIME, DemoConfig, generate_demo_database
from .evaluation import evaluate_plant
from .golden import verify_golden_replay
from .manifest import verify_manifest, write_manifest
from .runtime import FrozenRuntimeRouter
from .service import PlantShadowService
from .source import PlantPostgresSource, load_source_configs
from .storage import EvidenceStore


DEFAULT_DATABASE = "output/plant_shadow/plant_shadow.db"
DEFAULT_MANIFEST = "output/plant_shadow/plant_shadow_runtime_manifest.json"
DEFAULT_MODEL = "models/rul_v2_7_full_cadence.joblib"
DEFAULT_GOLDEN = "config/plant_shadow_golden_replay.json"


def add_plant_shadow_parser(subparsers) -> None:
    shadow = subparsers.add_parser("plant-shadow", help="Read-only plant-shadow evidence and monitoring operations")
    actions = shadow.add_subparsers(dest="shadow_action", required=True)

    manifest = actions.add_parser("create-manifest", help="Create the code/model runtime manifest")
    manifest.add_argument("--output", default=DEFAULT_MANIFEST)
    manifest.add_argument("--deployment-id", required=True)

    golden = actions.add_parser("verify-golden", help="Verify frozen v2.7 golden replay")
    golden.add_argument("--model", default=DEFAULT_MODEL)
    golden.add_argument("--config", default="config/vvb001.json")
    golden.add_argument("--fixture", default=DEFAULT_GOLDEN)

    source_save = actions.add_parser("source-save", help="Save inactive, versioned source configuration locally")
    source_save.add_argument("--database", default=DEFAULT_DATABASE)
    source_save.add_argument("--sources", required=True)
    source_save.add_argument("--actor", required=True)

    source_list = actions.add_parser("source-list", help="List redacted local source configuration versions")
    source_list.add_argument("--database", default=DEFAULT_DATABASE)
    source_list.add_argument("--include-archived", action="store_true")

    ingest = actions.add_parser("ingest", help="Run frozen v2.7 shadow ingestion against read-only sources")
    ingest.add_argument("--database", default=DEFAULT_DATABASE)
    ingest.add_argument("--sources", required=True)
    ingest.add_argument("--config", default="config/vvb001.json")
    ingest.add_argument("--model", default=DEFAULT_MODEL)
    ingest.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ingest.add_argument("--golden", default=DEFAULT_GOLDEN)
    ingest.add_argument("--once", action="store_true")

    status = actions.add_parser("status", help="Print local plant-shadow evidence status")
    status.add_argument("--database", default=DEFAULT_DATABASE)

    demo = actions.add_parser(
        "generate-demo",
        help="Rebuild an isolated deterministic local demo through the frozen plant-shadow runtime",
    )
    demo.add_argument("--database", default=DEFAULT_DEMO_DATABASE)
    demo.add_argument("--machines", type=int, default=8)
    demo.add_argument("--hours", type=float, default=72.0)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--start-time", default=DEFAULT_START_TIME.isoformat())
    demo.add_argument("--cadence-minutes", type=int, default=10)
    demo.add_argument("--actor", default="local-demo-generator")
    demo.add_argument("--config", default="config/vvb001.json")
    demo.add_argument("--model", default=DEFAULT_MODEL)

    operating = actions.add_parser(
        "operating-state-add",
        help="Add a bounded PLC/CMMS/operator operating-state interval to the local evidence ledger",
    )
    operating.add_argument("--database", default=DEFAULT_DATABASE)
    operating.add_argument("--source-system", required=True)
    operating.add_argument("--external-event-id", required=True)
    operating.add_argument("--machine-uid", required=True)
    operating.add_argument("--operating-state", choices=[item.value for item in OperatingState], required=True)
    operating.add_argument(
        "--operating-state-source",
        choices=[item.value for item in OperatingStateSource],
        required=True,
    )
    operating.add_argument("--confidence", type=float, required=True)
    operating.add_argument("--effective-from", required=True)
    operating.add_argument("--effective-to")
    operating.add_argument("--maintenance-event-id")
    operating.add_argument("--details-json", default="{}")
    operating.add_argument("--actor", required=True)

    evidence = actions.add_parser("evidence-add", help="Add independent endpoint evidence locally")
    evidence.add_argument("--database", default=DEFAULT_DATABASE)
    evidence.add_argument("--source-system", required=True)
    evidence.add_argument("--external-event-id", required=True)
    evidence.add_argument("--machine-uid", required=True)
    evidence.add_argument("--lifecycle-id", required=True)
    evidence.add_argument("--precision", choices=[item.value for item in EndpointPrecision], required=True)
    evidence.add_argument("--time-lower")
    evidence.add_argument("--time-upper")
    evidence.add_argument("--details-json", default="{}")
    evidence.add_argument("--actor", required=True)

    classify = actions.add_parser("evidence-classify", help="Confirm endpoint class and target-specific eligibility")
    classify.add_argument("--database", default=DEFAULT_DATABASE)
    classify.add_argument("--evidence-id", required=True)
    classify.add_argument("--endpoint-class", choices=[item.value for item in EndpointClass], required=True)
    classify.add_argument("--actor", required=True)

    evaluate = actions.add_parser("evaluate", help="Compute support-gated plant metrics")
    evaluate.add_argument("--database", default=DEFAULT_DATABASE)

    api = actions.add_parser("serve-api", help="Serve the read-only local FastAPI backend")
    api.add_argument("--database", default=DEFAULT_DATABASE)
    api.add_argument("--manifest", default=DEFAULT_MANIFEST)
    api.add_argument("--model", default=DEFAULT_MODEL)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("endpoint timestamps must include a timezone offset")
    return parsed


def _watermark(store: EvidenceStore, source_key: str) -> tuple[datetime, str] | None:
    row = store.db.execute(
        "SELECT event_timestamp,source_row_id FROM source_watermarks WHERE source_key=?",
        (source_key,),
    ).fetchone()
    return (datetime.fromisoformat(str(row[0])), str(row[1])) if row else None


def _run_ingestion(args: argparse.Namespace) -> None:
    root = Path.cwd()
    manifest = verify_manifest(root, args.manifest)
    sensor = AppConfig.load(args.config).sensor
    verify_golden_replay(args.golden, args.model, sensor)
    configs = load_source_configs(args.sources)
    with EvidenceStore(args.database) as store:
        router = FrozenRuntimeRouter(
            args.model,
            sensor,
            deployment_id=str(manifest["deployment_id"]),
            runtime_manifest_id=str(manifest["deployment_id"]),
        )
        service = PlantShadowService(store, router)
        for config in configs:
            if any(True for _ in store.committed_observations(config.source_key)):
                router.rebuild_source(config.source_key, store.committed_observations(config.source_key))
        while True:
            activity = False
            for config in configs:
                started = time.perf_counter()
                try:
                    with PlantPostgresSource(config) as source:
                        source.validate_schema()
                        batch = source.fetch_after(_watermark(store, config.source_key))
                        duplicate = late = invalid = 0
                        committed_activity = False
                        for observation in batch:
                            result = service.process(observation)
                            duplicate += int(result.duplicate)
                            late += int(result.disposition == "LATE_QUARANTINED")
                            invalid += int(result.disposition == "INVALID")
                            committed_activity = committed_activity or not result.duplicate
                        activity = activity or committed_activity
                        store.record_source_health(
                            config.source_key,
                            "ONLINE",
                            latest_source_timestamp=batch[-1].event_timestamp.isoformat() if batch else None,
                            query_latency_ms=(time.perf_counter() - started) * 1000.0,
                            malformed_rows=invalid,
                            duplicate_rows=duplicate,
                            late_rows=late,
                            message=(
                                json.dumps(source.bootstrap_window, sort_keys=True)
                                if source.bootstrap_window is not None else None
                            ),
                        )
                except Exception as exc:
                    store.record_source_health(
                        config.source_key,
                        "ERROR",
                        query_latency_ms=(time.perf_counter() - started) * 1000.0,
                        error_code=type(exc).__name__,
                        message=str(exc),
                    )
                    if args.once:
                        raise
            if args.once:
                break
            if not activity:
                time.sleep(min(config.poll_seconds for config in configs))


def run_plant_shadow_command(args: argparse.Namespace) -> None:
    action = args.shadow_action
    if action == "create-manifest":
        payload = write_manifest(Path.cwd(), args.output, deployment_id=args.deployment_id)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif action == "verify-golden":
        sensor = AppConfig.load(args.config).sensor
        print(json.dumps(verify_golden_replay(args.fixture, args.model, sensor), indent=2, sort_keys=True))
    elif action == "source-save":
        configs = load_source_configs(args.sources)
        with EvidenceStore(args.database) as store:
            values = [
                store.save_source_config(item.source_key, item.redacted_dict(), actor=args.actor)
                for item in configs
            ]
        print(json.dumps({"saved_versions": values, "enabled": False}, indent=2))
    elif action == "source-list":
        with EvidenceStore(args.database) as store:
            values = store.latest_source_configs(include_archived=args.include_archived)
        for value in values:
            config = value.get("config") or {}
            config.pop("dsn_env", None)
        print(json.dumps(values, indent=2, sort_keys=True))
    elif action == "ingest":
        _run_ingestion(args)
    elif action == "status":
        with EvidenceStore(args.database) as store:
            print(json.dumps(store.overview(), indent=2, sort_keys=True))
    elif action == "generate-demo":
        start_time = _parse_time(args.start_time)
        if start_time is None:
            raise ValueError("start-time is required")
        result = generate_demo_database(
            args.database,
            config=DemoConfig(
                machines=args.machines,
                hours=args.hours,
                seed=args.seed,
                start_time=start_time,
                cadence_minutes=args.cadence_minutes,
                actor=args.actor,
            ),
            app_config_path=args.config,
            model_path=args.model,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif action == "operating-state-add":
        effective_from = _parse_time(args.effective_from)
        if effective_from is None:
            raise ValueError("effective-from is required")
        with EvidenceStore(args.database) as store:
            evidence_id = store.add_operating_state_evidence(
                source_system=args.source_system,
                external_event_id=args.external_event_id,
                machine_uid=args.machine_uid,
                operating_state=OperatingState(args.operating_state),
                operating_state_source=OperatingStateSource(args.operating_state_source),
                confidence=args.confidence,
                effective_from=effective_from,
                effective_to=_parse_time(args.effective_to),
                maintenance_event_id=args.maintenance_event_id,
                details=json.loads(args.details_json),
                actor=args.actor,
            )
        print(json.dumps({"operating_state_evidence_id": evidence_id}, indent=2))
    elif action == "evidence-add":
        with EvidenceStore(args.database) as store:
            evidence_id = store.add_endpoint_evidence(
                source_system=args.source_system,
                external_event_id=args.external_event_id,
                machine_uid=args.machine_uid,
                lifecycle_id=args.lifecycle_id,
                precision=EndpointPrecision(args.precision),
                event_time_lower=_parse_time(args.time_lower),
                event_time_upper=_parse_time(args.time_upper),
                details=json.loads(args.details_json),
                actor=args.actor,
            )
        print(json.dumps({"evidence_id": evidence_id}, indent=2))
    elif action == "evidence-classify":
        with EvidenceStore(args.database) as store:
            result = store.confirm_endpoint_classification(
                args.evidence_id,
                EndpointClass(args.endpoint_class),
                actor=args.actor,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif action == "evaluate":
        with EvidenceStore(args.database) as store:
            print(json.dumps(evaluate_plant(store.db), indent=2, sort_keys=True, default=str))
    elif action == "serve-api":
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("initial plant-shadow API is restricted to localhost")
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install requirements-dev.txt before serving the FastAPI backend") from exc
        from .api import create_app
        uvicorn.run(create_app(args.database, manifest_path=args.manifest, model_path=args.model), host=args.host, port=args.port)
    else:  # pragma: no cover - argparse enforces choices
        raise RuntimeError(action)
