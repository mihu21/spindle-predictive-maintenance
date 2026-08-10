from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .data_profile import file_sha256, profile_data, write_profile
from .display import print_result
from .domain import validate_domain, validate_registry_domain
from .io import read_csv_records
from .lifecycle_store import LifecycleCSVStore
from .ml_forecaster import MLForecaster
from .monitor import ConditionMonitor
from .model_registry import ModelRegistry
from .offline import prepare_offline_replay
from .retraining import build_training_examples, evaluate_candidate, train_candidate
from .realistic_generator import generate_realistic_mock
from .simulator import DegradationSimulator
from .storage import CSVResultStore, InvalidCSVStore, SQLiteResultStore
from .anomaly_fixtures import evaluate_anomaly_fixtures


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config() -> Path:
    return _project_root() / "config" / "thresholds.json"


def _default_dataset() -> Path:
    return _project_root() / "data" / "spindle_predictive_maintenance_10000_unlabeled.csv"


def _default_models() -> Path:
    return _project_root() / "models" / "mock"


def _count_csv_rows(path: str | Path) -> int:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if row)


def _add_output_arguments(parser: argparse.ArgumentParser, *, prefix: str) -> None:
    parser.add_argument("--database", default="output/monitor.db")
    parser.add_argument("--csv", default=f"output/{prefix}_results.csv")
    parser.add_argument("--detailed-csv", default=None)
    parser.add_argument("--lifecycles-csv", default="output/lifecycles.csv")
    parser.add_argument("--invalid-csv", default="output/invalid_rows.csv")
    parser.add_argument("--models-root", default=str(_default_models()))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--no-database", action="store_true",
        help="Write CSV audit outputs without duplicating every row into SQLite.",
    )
    parser.add_argument("--data-domain", choices=["accelerated_mock", "realistic_synthetic", "plant"], default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manufacturer safety + anomaly handling + v6 probabilistic degradation prognostics"
    )
    parser.add_argument("--config", default=str(_default_config()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="Process the complete input CSV")
    replay.add_argument("--input", default=str(_default_dataset()))
    replay.add_argument("--limit", type=int, default=None)
    replay.add_argument("--delay", type=float, default=0.0)
    replay.add_argument(
        "--anomaly-audit", action="store_true",
        help="Enable the conservative causal anomaly and model-usability layer",
    )
    replay.add_argument(
        "--legacy-ml-audit", action="store_true",
        help="Also compute the retired supervised ML forecast for research/audit only; it cannot override v6.",
    )
    _add_output_arguments(replay, prefix="replay")

    simulate = subparsers.add_parser("simulate", help="Generate degradation and optional resets")
    simulate.add_argument("--samples", type=int, default=None)
    simulate.add_argument("--reference-data", default=str(_default_dataset()))
    simulate.add_argument("--interval", type=float, default=0.0)
    simulate.add_argument("--sample-seconds", type=float, default=60.0)
    simulate.add_argument("--degradation-per-sample", type=float, default=0.0025)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument(
        "--reset-at-sample",
        action="append",
        type=int,
        default=[],
        help="Apply an inferred-maintenance-style reset before this sample; repeatable",
    )
    _add_output_arguments(simulate, prefix="monitor")

    detect = subparsers.add_parser("detect-lifecycles", help="Replay and report inferred lifecycle resets")
    detect.add_argument("--input", default=str(_default_dataset()))
    detect.add_argument("--lifecycles-csv", default="output/lifecycles.csv")
    detect.add_argument("--invalid-csv", default="output/invalid_rows.csv")
    detect.add_argument("--models-root", default=str(_default_models()))

    def add_train_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--input", default=str(_default_dataset()))
        command.add_argument("--data-domain", choices=sorted({"accelerated_mock", "realistic_synthetic", "plant"}), default="accelerated_mock")
        command.add_argument("--models-root", default=str(_default_models()))
        command.add_argument("--metrics", default="output/mock/model_metrics.json")
        command.add_argument("--database", default="output/mock/training.db")
        mode = command.add_mutually_exclusive_group()
        mode.add_argument(
            "--anomaly-screening", action="store_true",
            help="Screen anomaly-contaminated rows; required for plant-domain training",
        )
        mode.add_argument(
            "--reproduction-mode", action="store_true",
            help="Intentionally disable screening only to reproduce existing synthetic artifacts",
        )

    train = subparsers.add_parser("train", help="Legacy supervised research training; not used by v6 primary prognostics")
    add_train_arguments(train)
    train_model = subparsers.add_parser("train-model", help="Legacy supervised research candidate training; not used by v6 primary prognostics")
    add_train_arguments(train_model)

    screen_training = subparsers.add_parser(
        "audit-training-screening",
        help="Prepare and count anomaly-screened training rows without fitting or saving models",
    )
    screen_training.add_argument("--input", default=str(_default_dataset()))
    screen_training.add_argument("--models-root", default=str(_default_models()))
    screen_training.add_argument("--output", default="output/anomaly_training_screening.json")

    evaluate = subparsers.add_parser(
        "evaluate-model",
        help="Compare candidate with production and baseline; promotion is opt-in",
    )
    evaluate.add_argument("--models-root", default=str(_default_models()))
    evaluate.add_argument("--promote", action="store_true")
    evaluate.add_argument("--output", default="output/model_evaluation.json")
    evaluate.add_argument("--database", default="output/monitor.db")

    guardrails = subparsers.add_parser(
        "evaluate-guardrails",
        help="Select ML/statistical disagreement thresholds on validation lifecycles",
    )
    guardrails.add_argument("--input", default=str(_default_dataset()))
    guardrails.add_argument("--models-root", default=str(_default_models()))
    guardrails.add_argument("--output", default="output/guardrail_evaluation.json")

    validate = subparsers.add_parser("validate", help="Audit input and threshold-label agreement")
    validate.add_argument("--input", default=str(_default_dataset()))
    validate.add_argument("--report", default="output/validation_report.txt")
    validate.add_argument("--models-root", default=str(_default_models()))

    profile = subparsers.add_parser("profile-data", help="Create a read-only data-readiness profile")
    profile.add_argument("--input", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--models-root", default=None)

    realistic = subparsers.add_parser("generate-realistic-mock", help="Generate deterministic realistic-duration synthetic lifecycles")
    realistic.add_argument("--profile", required=True)
    realistic.add_argument("--lifecycles", type=int, default=30)
    realistic.add_argument("--output", required=True)
    realistic.add_argument("--metadata-output", required=True)
    realistic.add_argument("--seed", type=int, default=42)
    realistic.add_argument("--suite-role", choices=["development", "acceptance"], default="development")

    anomaly_eval = subparsers.add_parser(
        "evaluate-anomalies", help="Run deterministic conservative anomaly fixtures"
    )
    anomaly_eval.add_argument("--output", default="output/anomaly_evaluation.json")

    anomaly_summary = subparsers.add_parser(
        "summarize-anomalies", help="Summarize anomaly audit counts from SQLite"
    )
    anomaly_summary.add_argument("--database", default="output/monitor.db")
    anomaly_summary.add_argument("--output", default=None)
    anomaly_summary.add_argument("--replay-csv", default=None)

    anomaly_export = subparsers.add_parser(
        "export-anomaly-intervals", help="Export machine-readable anomaly event intervals"
    )
    anomaly_export.add_argument("--database", default="output/monitor.db")
    anomaly_export.add_argument("--output", default="output/anomaly_intervals.json")
    anomaly_export.add_argument("--csv", default=None)

    anomaly_events = subparsers.add_parser(
        "export-anomaly-events", help="Export raw causal anomaly event rows without consolidation"
    )
    anomaly_events.add_argument("--database", default="output/monitor.db")
    anomaly_events.add_argument("--output", default="output/anomaly_events.json")

    anomaly_state = subparsers.add_parser(
        "inspect-anomaly-state", help="Inspect the most recent causal anomaly state"
    )
    anomaly_state.add_argument("--database", default="output/monitor.db")

    explain = subparsers.add_parser(
        "explain-prediction", help="Explain the latest reduced, suspended, or refused prediction"
    )
    explain.add_argument("--database", default="output/monitor.db")
    explain.add_argument("--timestamp", default=None)
    return parser


def _show_result_or_progress(result, count: int, verbose: bool, progress_every: int) -> None:
    if verbose:
        print_result(result)
    elif progress_every > 0 and count % progress_every == 0:
        print(
            f"Processed {count} valid readings; status={result.raw_status.name}; "
            f"lifecycle={result.lifecycle_id}/{result.lifecycle_state}."
        )


def _open_stores(args):
    if args.no_save:
        return None, None, None, None, None
    database = None if args.no_database else SQLiteResultStore(args.database)
    if database is not None:
        database.clear_monitoring_data()
    concise = CSVResultStore(args.csv)
    detailed = CSVResultStore(args.detailed_csv, detailed=True) if args.detailed_csv else None
    lifecycle = LifecycleCSVStore(args.lifecycles_csv)
    invalid = InvalidCSVStore(args.invalid_csv)
    return database, concise, detailed, lifecycle, invalid


def _close_stores(*stores) -> None:
    for store in stores:
        if store is not None:
            store.close()


def run_replay(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")
    config = load_config(args.config)
    config = replace(config, anomaly=replace(config.anomaly, enabled=args.anomaly_audit))
    inferred_domain = args.data_domain or ({"mock": "accelerated_mock", "realistic": "realistic_synthetic", "plant": "plant"}.get(Path(args.models_root).name.lower(), "plant"))
    validations, lifecycle_plan = prepare_offline_replay(
        args.input,
        config,
        args.models_root,
        valid_limit=args.limit,
    )
    input_path = Path(args.input)
    metadata_path = input_path.with_name(f"{input_path.stem}_metadata.json")
    generator_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {}
    )
    monitor = ConditionMonitor(
        config,
        models_root=args.models_root,
        lifecycle_plan=lifecycle_plan,
        enable_ml=bool(args.legacy_ml_audit),
        data_domain=inferred_domain,
        dataset_hash=file_sha256(input_path),
        generator_version=str(generator_metadata.get("generator_version") or ""),
    )
    database, concise, detailed, lifecycle_store, invalid_store = _open_stores(args)
    valid_count = 0
    invalid_count = 0
    lifecycle_count = 0
    try:
        for validation in validations:
            if not validation.valid or validation.reading is None:
                invalid_count += 1
                if database:
                    database.save_invalid(validation)
                if invalid_store:
                    invalid_store.save(validation)
                continue
            result = monitor.process(
                validation.reading,
                interpolated=validation.interpolated,
                source_sampling_interval_seconds=validation.source_sampling_interval_seconds,
                effective_resampling_interval_seconds=validation.effective_resampling_interval_seconds,
                input_features_available=validation.features_available,
            )
            valid_count += 1
            _show_result_or_progress(result, valid_count, args.verbose, args.progress_every)
            if database:
                database.save(result, validation.source_label)
            if concise:
                concise.save(result, validation.source_label)
            if detailed:
                detailed.save(result, validation.source_label)
            if result.completed_lifecycle:
                lifecycle_count += 1
                if lifecycle_store:
                    lifecycle_store.save(result.completed_lifecycle)
            if args.delay > 0:
                time.sleep(args.delay)
            if args.limit not in (None, 0) and valid_count >= args.limit:
                break
    finally:
        _close_stores(database, concise, detailed, lifecycle_store, invalid_store)

    print(
        f"Replay completed: {valid_count} valid row(s), {invalid_count} invalid row(s), "
        f"{lifecycle_count} completed inferred lifecycle(s)."
    )
    if concise:
        print(f"Readings CSV: {Path(args.csv).resolve()}")
        print(f"Lifecycles CSV: {Path(args.lifecycles_csv).resolve()}")
        print(f"SQLite database: {Path(args.database).resolve()}")

def run_simulation(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sample_count = args.samples
    if sample_count is None:
        sample_count = _count_csv_rows(args.reference_data)
    if sample_count < 0:
        raise ValueError("--samples cannot be negative")
    monitor = ConditionMonitor(config, models_root=args.models_root)
    simulator = DegradationSimulator(
        config,
        seed=args.seed,
        degradation_per_sample=args.degradation_per_sample,
        sample_interval_seconds=args.sample_seconds,
    )
    database, concise, detailed, lifecycle_store, invalid_store = _open_stores(args)
    reset_samples = set(args.reset_at_sample)
    count = 0
    lifecycle_count = 0
    try:
        while sample_count == 0 or count < sample_count:
            next_sample = count + 1
            if next_sample in reset_samples:
                simulator.maintenance_reset()
            result = monitor.process(simulator.read())
            count += 1
            _show_result_or_progress(result, count, args.verbose, args.progress_every)
            if database:
                database.save(result)
            if concise:
                concise.save(result)
            if detailed:
                detailed.save(result)
            if result.completed_lifecycle:
                lifecycle_count += 1
                if lifecycle_store:
                    lifecycle_store.save(result.completed_lifecycle)
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        _close_stores(database, concise, detailed, lifecycle_store, invalid_store)
    print(f"Simulation completed: {count} readings, {lifecycle_count} inferred resets.")


def run_detect_lifecycles(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validations, lifecycle_plan = prepare_offline_replay(
        args.input, config, args.models_root
    )
    lifecycle_store = LifecycleCSVStore(args.lifecycles_csv)
    invalid_store = InvalidCSVStore(args.invalid_csv)
    valid = invalid = completed = 0
    try:
        for validation in validations:
            if not validation.valid or validation.reading is None:
                invalid += 1
                invalid_store.save(validation)
                continue
            valid += 1
        for record in lifecycle_plan.records:
            completed += 1
            lifecycle_store.save(record)
    finally:
        lifecycle_store.close()
        invalid_store.close()
    print(
        f"Lifecycle detection completed: {valid} valid rows, {invalid} invalid rows, "
        f"{completed} completed inferred lifecycle(s)."
    )
    if completed == 0:
        print("No reset boundary was confirmed; the data remains one open lifecycle.")


def run_train(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    config = load_config(args.config)
    validate_domain(args.data_domain)
    screening_requested = bool(getattr(args, "anomaly_screening", False))
    if args.data_domain == "plant" and not screening_requested:
        raise ValueError(
            "Plant-domain training is refused without --anomaly-screening. "
            "Reproduction mode is restricted to existing synthetic artifacts."
        )
    screening_enabled = screening_requested
    config = replace(
        config, anomaly=replace(config.anomaly, enabled=screening_enabled)
    )
    if screening_enabled:
        print(
            "ANOMALY SCREENING ENABLED: unsuitable rows will be excluded and counted by reason."
        )
    else:
        print(
            "REPRODUCTION MODE: anomaly-contaminated rows are not screened; "
            "use only for deterministic reproduction of existing synthetic artifacts."
        )
    validate_registry_domain(args.models_root, args.data_domain)
    report = train_candidate(
        args.input, config, args.models_root, args.metrics, data_domain=args.data_domain
    )
    store = SQLiteResultStore(args.database)
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        store.save_training_run(timestamp, str(report.get("status", "unknown")), report)
        if report.get("status") == "candidate_trained":
            store.save_model_metadata(report, "candidate")
            store.save_model_target_records(report, "candidate")
            for target, target_data in report.get("targets", {}).items():
                store.save_validation_result(
                    timestamp,
                    target,
                    {
                        "validation_metrics": target_data.get("validation_metrics", {}),
                        "test_metrics": target_data.get("test_metrics", {}),
                    },
                )
    finally:
        store.close()
    print(json.dumps(report, indent=2))


def run_audit_training_screening(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config = replace(config, anomaly=replace(config.anomaly, enabled=True))
    feature_names, examples, scan = build_training_examples(
        args.input, config, args.models_root
    )
    report = {
        "status": "screening_audit_only_no_model_training_performed",
        "feature_count": len(feature_names),
        "usable_lifecycle_count": len(examples),
        **scan,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def run_profile_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = profile_data(args.input, config, models_root=args.models_root)
    write_profile(report, args.output)
    print(json.dumps(report, indent=2))


def run_generate_realistic(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = generate_realistic_mock(
        args.profile, args.lifecycles, args.output, args.metadata_output,
        config, seed=args.seed, suite_role=args.suite_role,
    )
    print(json.dumps(report, indent=2))


def run_evaluate_model(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    config = load_config(args.config)
    report = evaluate_candidate(config, args.models_root, promote=args.promote)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store = SQLiteResultStore(args.database)
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        store.save_validation_result(timestamp, "__model_evaluation__", report)
        for target, target_data in report.get("targets", {}).items():
            store.save_validation_result(timestamp, target, target_data)
        if report.get("promoted_targets"):
            store.archive_production_model_records(report["promoted_targets"], timestamp)
            production = ModelRegistry(args.models_root).load_metadata("production")
            if production is not None:
                store.save_model_metadata(production, "production")
                store.save_model_target_records(
                    production, "production", report["promoted_targets"]
                )
    finally:
        store.close()
    print(json.dumps(report, indent=2))


def run_evaluate_guardrails(args: argparse.Namespace) -> None:
    from .guardrail_evaluation import evaluate_guardrails

    config = load_config(args.config)
    report = evaluate_guardrails(args.input, config, args.models_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ModelRegistry(args.models_root).record_audit(
        "guardrail_configuration_selected",
        {
            "training_lifecycle_ids": report["training_lifecycle_ids"],
            "validation_lifecycle_ids": report["validation_lifecycle_ids"],
            "test_lifecycle_ids": report["test_lifecycle_ids"],
            "selection": report["selection"],
            "untouched_test_evaluation_count": report["untouched_test"]["evaluation_count"],
        },
    )
    print(json.dumps(report, indent=2))


def run_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    monitor = ConditionMonitor(config, models_root=args.models_root)
    total = invalid = labelled = correct = 0
    status_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    for validation in read_csv_records(args.input, config):
        total += 1
        if not validation.valid or validation.reading is None:
            invalid += 1
            invalid_reasons[validation.reason] += 1
            continue
        result = monitor.process(validation.reading)
        status_counts[result.raw_status.name] += 1
        if validation.source_label:
            labelled += 1
            correct += int(validation.source_label.upper() == result.raw_status.name)
    lines = [
        "SPINDLE MONITOR VALIDATION",
        f"Input rows: {total}",
        f"Valid rows: {total - invalid}",
        f"Invalid rows: {invalid}",
        f"Immediate status counts: {dict(status_counts)}",
        f"Labelled rows: {labelled}",
        f"Manufacturer-rule label accuracy: {(correct / labelled):.4%}" if labelled else "Manufacturer-rule label accuracy: unavailable",
        f"Completed inferred lifecycles: {len(monitor.lifecycle.completed)}",
        "ML validation status: unavailable until multiple completed lifecycles exist.",
    ]
    if invalid_reasons:
        lines.append(f"Invalid reasons: {dict(invalid_reasons)}")
    report = "\n".join(lines) + "\n"
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report, end="")


def run_evaluate_anomalies(args: argparse.Namespace) -> None:
    report = evaluate_anomaly_fixtures(load_config(args.config), args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


def _database_rows(path: str | Path, query: str, parameters: tuple = ()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return list(connection.execute(query, parameters))
    finally:
        connection.close()


def run_summarize_anomalies(args: argparse.Namespace) -> None:
    event_rows = _database_rows(
        args.database,
        "SELECT quality_status, anomaly_type, model_action, COUNT(*) AS count "
        "FROM anomaly_events GROUP BY quality_status, anomaly_type, model_action "
        "ORDER BY count DESC, anomaly_type",
    )
    reading_rows = _database_rows(
        args.database,
        "SELECT anomaly_type, quality_status, model_action, feature_history_action, "
        "training_eligible, COUNT(*) AS count FROM readings GROUP BY anomaly_type, "
        "quality_status, model_action, feature_history_action, training_eligible "
        "ORDER BY count DESC",
    )
    scalar = _database_rows(
        args.database,
        "SELECT (SELECT COUNT(*) FROM readings) AS reading_count, "
        "(SELECT COUNT(*) FROM anomaly_events) AS event_count, "
        "(SELECT COUNT(*) FROM anomaly_intervals) AS interval_count, "
        "(SELECT COUNT(*) FROM anomaly_intervals WHERE active=1) AS active_interval_count, "
        "(SELECT COUNT(*) FROM anomaly_state WHERE active=1) AS active_sensor_count",
    )[0]
    stuck = _database_rows(
        args.database,
        "SELECT anomaly_type, COUNT(*) AS count FROM anomaly_events "
        "WHERE anomaly_type LIKE '%STUCK_SENSOR%' GROUP BY anomaly_type",
    )
    report = {
        **dict(scalar),
        "reading_decision_counts": [dict(row) for row in reading_rows],
        "raw_event_counts": [dict(row) for row in event_rows],
        "stuck_event_counts": [dict(row) for row in stuck],
        "stuck_confirmed_count": sum(
            row["count"] for row in stuck if "STUCK_SENSOR_CONFIRMED" in row["anomaly_type"]
        ),
        "stuck_suspected_count": sum(
            row["count"] for row in stuck if "STUCK_SENSOR_SUSPECTED" in row["anomaly_type"]
        ),
    }
    if args.replay_csv:
        with Path(args.replay_csv).open(encoding="utf-8-sig", newline="") as source:
            csv_rows = list(csv.DictReader(source))
        report["csv_reading_count"] = len(csv_rows)
        report["csv_sqlite_reading_count_match"] = len(csv_rows) == scalar["reading_count"]
        report["csv_sqlite_anomaly_type_counts_match"] = (
            Counter(row["anomaly_type"] for row in csv_rows)
            == Counter({
                row["anomaly_type"]: sum(
                    item["count"] for item in reading_rows
                    if item["anomaly_type"] == row["anomaly_type"]
                )
                for row in reading_rows
            })
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def run_export_anomaly_intervals(args: argparse.Namespace) -> None:
    rows = _database_rows(
        args.database,
        "SELECT interval_id, anomaly_type, affected_sensors, first_observed_timestamp, "
        "decision_or_confirmation_timestamp, end_or_resolution_timestamp, duration_seconds, "
        "maximum_severity, maximum_confidence, safety_actions_json, model_actions_json, "
        "active, final_offline_classification, event_row_count, supporting_evidence_json "
        "FROM anomaly_intervals ORDER BY first_observed_timestamp, interval_id",
    )
    intervals = []
    for row in rows:
        intervals.append({
            "interval_id": row["interval_id"], "anomaly_type": row["anomaly_type"],
            "affected_sensors": row["affected_sensors"].split("|") if row["affected_sensors"] else [],
            "first_observed_timestamp": row["first_observed_timestamp"],
            "decision_or_confirmation_timestamp": row["decision_or_confirmation_timestamp"],
            "end_or_resolution_timestamp": row["end_or_resolution_timestamp"],
            "duration_seconds": row["duration_seconds"],
            "maximum_severity": row["maximum_severity"],
            "maximum_confidence": row["maximum_confidence"],
            "safety_actions_observed": json.loads(row["safety_actions_json"]),
            "model_actions_observed": json.loads(row["model_actions_json"]),
            "active": bool(row["active"]),
            "final_offline_classification": row["final_offline_classification"],
            "event_rows_consolidated": row["event_row_count"],
            "supporting_evidence_summary": json.loads(row["supporting_evidence_json"]),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(intervals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as destination:
            fieldnames = list(intervals[0]) if intervals else [
                "interval_id", "anomaly_type", "affected_sensors",
                "first_observed_timestamp", "decision_or_confirmation_timestamp",
                "end_or_resolution_timestamp", "duration_seconds", "active",
            ]
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for interval in intervals:
                writer.writerow({
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in interval.items()
                })
    print(f"Exported {len(intervals)} consolidated anomaly interval(s) to {output.resolve()}")


def run_export_anomaly_events(args: argparse.Namespace) -> None:
    rows = _database_rows(
        args.database,
        "SELECT id, timestamp, row_number, quality_status, anomaly_type, model_action, details_json "
        "FROM anomaly_events ORDER BY id",
    )
    events = [{**dict(row), "details_json": json.loads(row["details_json"])} for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {len(events)} raw anomaly event row(s) to {output.resolve()}")


def run_inspect_anomaly_state(args: argparse.Namespace) -> None:
    rows = _database_rows(
        args.database,
        "SELECT sensor, current_anomaly_type, quality_status, active, "
        "first_observed_timestamp, latest_update_timestamp, resolution_timestamp, "
        "safety_action, model_action FROM anomaly_state ORDER BY sensor",
    )
    print(json.dumps({
        "sensors": [
            {**dict(row), "active": bool(row["active"])} for row in rows
        ],
        "active_sensor_count": sum(bool(row["active"]) for row in rows),
        "source": "maintained_anomaly_state_table",
    }, indent=2, sort_keys=True))


def run_explain_prediction(args: argparse.Namespace) -> None:
    where = "WHERE timestamp = ?" if args.timestamp else ""
    parameters = (args.timestamp,) if args.timestamp else ()
    rows = _database_rows(
        args.database,
        f"SELECT details_json FROM readings {where} ORDER BY id DESC LIMIT 1",
        parameters,
    )
    if not rows:
        raise ValueError("No matching prediction audit record exists")
    details = json.loads(rows[0]["details_json"])
    explanation = {
        "timestamp": details.get("timestamp"),
        "raw_model_prediction_hours": details.get("prediction_critical_hours_raw"),
        "operational_prediction_hours": details.get("final_time_to_critical_hours"),
        "forecast_confidence": details.get("forecast_confidence"),
        "model_action": details.get("model_action"),
        "quality_status": details.get("quality_status"),
        "anomaly_type": details.get("anomaly_type"),
        "reason": details.get("refusal_or_fallback_reason") or details.get("forecast_reason"),
        "safety_action": details.get("safety_action"),
    }
    print(json.dumps(explanation, indent=2, sort_keys=True))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "replay": run_replay,
        "simulate": run_simulation,
        "detect-lifecycles": run_detect_lifecycles,
        "train": run_train,
        "train-model": run_train,
        "audit-training-screening": run_audit_training_screening,
        "evaluate-model": run_evaluate_model,
        "evaluate-guardrails": run_evaluate_guardrails,
        "validate": run_validate,
        "profile-data": run_profile_data,
        "generate-realistic-mock": run_generate_realistic,
        "evaluate-anomalies": run_evaluate_anomalies,
        "summarize-anomalies": run_summarize_anomalies,
        "export-anomaly-intervals": run_export_anomaly_intervals,
        "export-anomaly-events": run_export_anomaly_events,
        "inspect-anomaly-state": run_inspect_anomaly_state,
        "explain-prediction": run_explain_prediction,
    }
    commands[args.command](args)
