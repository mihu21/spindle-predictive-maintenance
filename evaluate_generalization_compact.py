#!/usr/bin/env python3
"""Fast, narrow-column evaluation for very wide replay audit CSVs.

This evaluator is intentionally model-quality focused.  It validates every
compact audit payload through distinct-payload coverage, aligns labels exactly
by timestamp, and then computes the same row/event/false-alert semantics used
by ``evaluate_monitoring.py`` without repeatedly re-running timestamp and JSON
validation for every target.

Operational trust remains fail-closed in this compact mode.  Full operational
policy/trust validation continues to belong to ``evaluate_monitoring.py``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from evaluate_monitoring import (
    _binary_metrics,
    report_csv_rows,
    status_layer_metrics,
    timestamp_validation,
)

TARGETS = [
    f"probability_{kind}_{hours}h"
    for kind in ("warning", "critical")
    for hours in (6, 12, 24)
]
PROBABILITY_COLUMNS = [
    f"{prefix}{target}"
    for prefix in ("", "raw_", "reconciled_")
    for target in TARGETS
]
AUDIT_COLUMNS = [
    "timestamp", "lifecycle_id", "status", "raw_status", "raw_safety_status",
    "stabilized_status", "event_status", "selected_probability_thresholds",
    "probability_target_eligibility", "probability_threshold_crossings",
    "loaded_model_targets", "recommendation_actionable",
]
JSON_AUDIT_COLUMNS = (
    "selected_probability_thresholds",
    "probability_target_eligibility",
    "probability_threshold_crossings",
    "loaded_model_targets",
)
CATEGORICAL_COLUMNS = JSON_AUDIT_COLUMNS


def _duration_bucket(hours: float) -> str:
    for lower, upper in ((0, 72), (72, 168), (168, 336), (336, 720)):
        if lower <= hours < upper:
            return f"[{lower},{upper})h"
    return "[720,inf)h"


def _payload_key(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    return str(raw)


def _parse_payload(raw: object, column: str, expected: type) -> Any:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:  # pragma: no cover - exact exception type is parser dependent
        raise ValueError(f"malformed compact audit JSON: {column}") from exc
    if not isinstance(value, expected):
        raise ValueError(f"invalid compact audit type: {column}")
    return value


def _parse_unique_json(frame: pd.DataFrame, column: str, expected: type) -> dict[str, Any]:
    """Parse every distinct payload exactly once and retain a row lookup cache."""
    if column not in frame:
        raise ValueError(f"missing compact audit column: {column}")
    parsed: dict[str, Any] = {}
    for raw in frame[column].drop_duplicates():
        key = _payload_key(raw)
        parsed[key] = _parse_payload(raw, column, expected)
    return parsed


@dataclass(frozen=True)
class CompactAudit:
    thresholds: dict[str, dict[str, Any]]
    eligibility: dict[str, dict[str, Any]]
    crossings: dict[str, list[Any]]
    loaded: dict[str, list[Any]]

    def threshold(self, raw: object, target: str) -> float | None:
        value = self.thresholds[_payload_key(raw)].get(target)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return value if np.isfinite(value) else None
        return None

    def eligible(self, raw: object, target: str) -> bool:
        return self.eligibility[_payload_key(raw)].get(target) is True

    def crossed(self, raw: object, target: str) -> bool:
        return target in self.crossings[_payload_key(raw)]

    def is_loaded(self, raw: object, target: str) -> bool:
        return target in self.loaded[_payload_key(raw)]


def _build_compact_audit(frame: pd.DataFrame) -> CompactAudit:
    return CompactAudit(
        thresholds=_parse_unique_json(frame, "selected_probability_thresholds", dict),
        eligibility=_parse_unique_json(frame, "probability_target_eligibility", dict),
        crossings=_parse_unique_json(frame, "probability_threshold_crossings", list),
        loaded=_parse_unique_json(frame, "loaded_model_targets", list),
    )


def validate_compact_audit(frame: pd.DataFrame, audit: CompactAudit | None = None) -> dict[str, object]:
    """Validate every compact audit payload without row-wise JSON parsing.

    Validation is done over all distinct payloads, so repeated rows cannot hide
    malformed structures while the 500k+ row acceptance file avoids hundreds
    of thousands of identical ``json.loads`` calls.
    """
    audit = audit or _build_compact_audit(frame)
    invalid_thresholds = 0
    invalid_eligibility = 0
    invalid_crossings = 0
    invalid_loaded = 0
    for mapping in audit.thresholds.values():
        for target, value in mapping.items():
            if (
                not isinstance(target, str)
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                invalid_thresholds += 1
    for mapping in audit.eligibility.values():
        invalid_eligibility += sum(
            not isinstance(target, str) or not isinstance(value, bool)
            for target, value in mapping.items()
        )
    for values in audit.crossings.values():
        invalid_crossings += sum(not isinstance(value, str) for value in values)
    for values in audit.loaded.values():
        invalid_loaded += sum(not isinstance(value, str) for value in values)
    if invalid_thresholds:
        raise ValueError("compact audit contains invalid selected probability threshold")
    if invalid_eligibility:
        raise ValueError("compact audit contains invalid probability target eligibility")
    if invalid_crossings:
        raise ValueError("compact audit contains invalid threshold crossing target")
    if invalid_loaded:
        raise ValueError("compact audit contains invalid loaded model target")
    return {
        "validated": True,
        "validation_scope": "all rows through complete distinct-payload coverage",
        "selected_threshold_payload_count": len(audit.thresholds),
        "eligibility_payload_count": len(audit.eligibility),
        "crossing_payload_count": len(audit.crossings),
        "loaded_target_payload_count": len(audit.loaded),
        "invalid_threshold_count": 0,
        "invalid_eligibility_count": 0,
        "invalid_crossing_count": 0,
        "invalid_loaded_target_count": 0,
        "operational_trust_evaluated": False,
        "operational_trust_reason": "compact model-quality mode intentionally fails trust closed",
    }


def load_compact_frames(
    replay_path: str | Path, labels_path: str | Path, *, chunksize: int = 50_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load only evaluator-required columns and enforce exact chunk alignment."""
    header = pd.read_csv(replay_path, nrows=0).columns
    columns = [value for value in [*AUDIT_COLUMNS, *PROBABILITY_COLUMNS] if value in header]
    missing = {"timestamp", *JSON_AUDIT_COLUMNS} - set(columns)
    if missing:
        raise ValueError(f"missing compact replay columns: {sorted(missing)}")
    replay_reader = pd.read_csv(
        replay_path, usecols=columns, chunksize=chunksize, low_memory=False
    )
    label_reader = pd.read_csv(
        labels_path, usecols=["timestamp", "health_status"], chunksize=chunksize, low_memory=False
    )
    replay_chunks: list[pd.DataFrame] = []
    actual_chunks: list[pd.Series] = []
    label_iterator = iter(label_reader)
    for number, replay_chunk in enumerate(replay_reader, 1):
        try:
            label_chunk = next(label_iterator)
        except StopIteration as exc:
            raise ValueError("label/replay row count alignment failed") from exc
        replay_time = pd.to_datetime(replay_chunk["timestamp"], errors="coerce").reset_index(drop=True)
        label_time = pd.to_datetime(label_chunk["timestamp"], errors="coerce").reset_index(drop=True)
        if (
            len(replay_chunk) != len(label_chunk)
            or replay_time.isna().any()
            or label_time.isna().any()
            or not replay_time.equals(label_time)
        ):
            raise ValueError(f"label/replay timestamp alignment failed in chunk {number}")
        replay_chunks.append(replay_chunk)
        actual_chunks.append(label_chunk["health_status"].astype(str).str.upper())
    try:
        next(label_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("label/replay row count alignment failed")
    frame = pd.concat(replay_chunks, ignore_index=True) if replay_chunks else pd.DataFrame(columns=columns)
    actual = pd.concat(actual_chunks, ignore_index=True) if actual_chunks else pd.Series(dtype=str)
    # Repeated audit payloads dominate memory on wide acceptance replays.  A
    # categorical representation keeps the narrow evaluator memory bounded
    # without changing values or metric definitions.
    for column in CATEGORICAL_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype("category")
    return frame, actual.reset_index(drop=True)


@dataclass(frozen=True)
class EvaluationContext:
    times: pd.Series
    segments: pd.Series
    lifecycle: pd.Series
    segment_hours: float
    events: dict[str, list[dict[str, Any]]]


def _build_events(
    frame: pd.DataFrame, actual: pd.Series, times: pd.Series, segments: pd.Series,
) -> dict[str, list[dict[str, Any]]]:
    lifecycle = frame.get("lifecycle_id", pd.Series("default", index=frame.index)).fillna("unknown").astype(str)
    previous = actual.shift()
    same_segment = segments.eq(segments.shift())
    starts_by_kind = {
        "warning": actual.eq("WARNING") & previous.eq("NORMAL") & same_segment,
        "critical": actual.eq("CRITICAL") & previous.isin(["NORMAL", "WARNING"]) & same_segment,
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for kind, starts in starts_by_kind.items():
        events: list[dict[str, Any]] = []
        for number, position in enumerate(np.flatnonzero(starts.to_numpy(dtype=bool)), 1):
            events.append({
                "episode_id": f"{kind}_{number:04d}",
                "lifecycle_id": str(lifecycle.iloc[position]),
                "onset_timestamp": times.iloc[position],
                "segment_id": int(segments.iloc[position]),
                "onset_position": int(position),
            })
        output[kind] = events
    return output


def build_evaluation_context(
    frame: pd.DataFrame, actual: pd.Series, maximum_gap_minutes: float,
) -> EvaluationContext:
    times, segments, _, _ = timestamp_validation(frame, maximum_gap_minutes)
    lifecycle = frame.get("lifecycle_id", pd.Series("default", index=frame.index)).fillna("unknown").astype(str)
    valid = pd.DataFrame({"time": times, "segment": segments}).dropna(subset=["time"])
    if valid.empty:
        segment_hours = 0.0
    else:
        bounds = valid.groupby("segment", sort=False)["time"].agg(["min", "max"])
        segment_hours = float(((bounds["max"] - bounds["min"]).dt.total_seconds().clip(lower=0).sum()) / 3600.0)
    return EvaluationContext(
        times=times,
        segments=segments,
        lifecycle=lifecycle,
        segment_hours=segment_hours,
        events=_build_events(frame, actual, times, segments),
    )


def _future_event_labels_fast(
    context: EvaluationContext, kind: str, hours: int, row_count: int,
) -> pd.Series:
    labels = np.zeros(row_count, dtype=bool)
    times = context.times.to_numpy()
    segments = context.segments.to_numpy()
    lifecycle = context.lifecycle.to_numpy(dtype=str)
    horizon = pd.Timedelta(hours=hours)
    for event in context.events[kind]:
        onset = event["onset_timestamp"]
        if pd.isna(onset):
            continue
        mask = (
            (segments == event["segment_id"])
            & (lifecycle == event["lifecycle_id"])
            & (times >= (onset - horizon).to_datetime64())
            & (times < onset.to_datetime64())
        )
        labels |= mask
    return pd.Series(labels)


def _target_event_metrics_fast(
    context: EvaluationContext, kind: str, predicted: pd.Series, hours: int,
) -> dict[str, Any]:
    predicted_values = predicted.to_numpy(dtype=bool)
    times = context.times.to_numpy()
    segments = context.segments.to_numpy()
    lifecycle = context.lifecycle.to_numpy(dtype=str)
    leads: list[float] = []
    missed = 0
    detail: list[dict[str, Any]] = []
    horizon = pd.Timedelta(hours=hours)
    for event in context.events[kind]:
        onset = event["onset_timestamp"]
        mask = (
            predicted_values
            & (segments == event["segment_id"])
            & (lifecycle == event["lifecycle_id"])
            & (times >= (onset - horizon).to_datetime64())
            & (times < onset.to_datetime64())
        )
        indexes = np.flatnonzero(mask)
        lead = None
        if len(indexes):
            first = context.times.iloc[int(indexes[0])]
            lead = float((onset - first).total_seconds() / 3600.0)
            leads.append(lead)
        else:
            missed += 1
        detail.append({
            "episode_id": event["episode_id"],
            "onset_timestamp": onset.isoformat() if pd.notna(onset) else None,
            "warned_before_onset": lead is not None,
            "first_alert_lead_time_hours": lead,
        })
    events = context.events[kind]
    return {
        "event_onsets": len(events),
        "warned_before_onset": len(events) - missed,
        "missed": missed,
        "event_fn_rate": (missed / len(events) if events else None),
        "median_lead_time_hours": median(leads) if leads else None,
        "minimum_lead_time_hours": min(leads) if leads else None,
        "maximum_lead_time_hours": max(leads) if leads else None,
        **{
            f"warned_at_least_{lead}h_early_rate": (
                sum(value >= lead for value in leads) / len(events) if events else None
            )
            for lead in (1, 3, 6, 12, 24)
        },
        "episodes": detail,
    }


def _false_episode_detail_fast(
    mask: pd.Series, context: EvaluationContext,
) -> dict[str, Any]:
    values = mask.to_numpy(dtype=bool)
    indexes = np.flatnonzero(values)
    durations: list[float] = []
    episode_count = 0
    crossing = 0
    if len(indexes):
        segments = context.segments.to_numpy()
        breaks = np.ones(len(indexes), dtype=bool)
        if len(indexes) > 1:
            breaks[1:] = (
                (indexes[1:] != indexes[:-1] + 1)
                | (segments[indexes[1:]] != segments[indexes[:-1]])
            )
        starts = np.flatnonzero(breaks)
        ends = np.r_[starts[1:] - 1, len(indexes) - 1]
        episode_count = int(len(starts))
        for left, right in zip(starts, ends):
            first_position = int(indexes[left])
            last_position = int(indexes[right])
            first = context.times.iloc[first_position]
            last = context.times.iloc[last_position]
            if pd.notna(first) and pd.notna(last):
                durations.append(max(0.0, float((last - first).total_seconds() / 3600.0)))
            if context.lifecycle.iloc[first_position] != context.lifecycle.iloc[last_position]:
                crossing += 1
    return {
        "false_positive_rows": int(values.sum()),
        "false_alert_episodes": episode_count,
        "false_alert_episode_rate_per_operating_day": (
            episode_count / (context.segment_hours / 24.0) if context.segment_hours else None
        ),
        "median_false_alert_duration_hours": median(durations) if durations else None,
        "maximum_false_alert_duration_hours": max(durations) if durations else None,
        "normal_time_under_false_alert": (
            sum(durations) / context.segment_hours if context.segment_hours else None
        ),
        "alerts_crossing_lifecycle_boundaries": crossing,
    }


def _target_audit_arrays(
    frame: pd.DataFrame, audit: CompactAudit, target: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    threshold_cache = {
        key: audit.threshold(key, target) for key in audit.thresholds
    }
    eligibility_cache = {
        key: audit.eligibility[key].get(target) is True for key in audit.eligibility
    }
    crossing_cache = {
        key: target in values for key, values in audit.crossings.items()
    }
    loaded_cache = {
        key: target in values for key, values in audit.loaded.items()
    }
    threshold = frame["selected_probability_thresholds"].map(
        lambda value: threshold_cache[_payload_key(value)]
    ).astype("float64")
    eligible = frame["probability_target_eligibility"].map(
        lambda value: eligibility_cache[_payload_key(value)]
    ).astype(bool)
    crossed = frame["probability_threshold_crossings"].map(
        lambda value: crossing_cache[_payload_key(value)]
    ).astype(bool)
    loaded = frame["loaded_model_targets"].map(
        lambda value: loaded_cache[_payload_key(value)]
    ).astype(bool)
    return threshold, eligible, crossed, loaded


def fast_per_target_metrics(
    frame: pd.DataFrame,
    actual: pd.Series,
    trusted: pd.Series,
    maximum_gap_minutes: float,
    *,
    context: EvaluationContext | None = None,
    audit: CompactAudit | None = None,
) -> dict[str, Any]:
    """Metric-equivalent fast path for ``evaluate_monitoring.per_target_metrics``."""
    context = context or build_evaluation_context(frame, actual, maximum_gap_minutes)
    audit = audit or _build_compact_audit(frame)
    output: dict[str, Any] = {}
    for kind in ("warning", "critical"):
        for hours in (6, 12, 24):
            target = f"probability_{kind}_{hours}h"
            labels = _future_event_labels_fast(context, kind, hours, len(frame))
            reconciled = pd.to_numeric(
                frame.get(f"reconciled_{target}", frame.get(target, pd.Series(np.nan, index=frame.index))),
                errors="coerce",
            )
            raw = pd.to_numeric(frame.get(f"raw_{target}", reconciled), errors="coerce")
            threshold, eligible, crossed, loaded = _target_audit_arrays(frame, audit, target)
            valid_raw = raw.notna() & np.isfinite(raw) & raw.between(0, 1) & threshold.notna() & threshold.between(0, 1, inclusive="neither")
            valid_reconciled = reconciled.notna() & np.isfinite(reconciled) & reconciled.between(0, 1) & threshold.notna() & threshold.between(0, 1, inclusive="neither")
            model_raw = valid_raw & (raw >= threshold)
            model_reconciled = valid_reconciled & (reconciled >= threshold)
            strict = crossed & trusted & eligible & loaded
            normal_negative = ~labels & actual.eq("NORMAL")
            output[target] = {
                "target": target,
                "horizon_hours": hours,
                "actual_label_source": f"next {kind.upper()} onset within elapsed {hours}h in same lifecycle/continuous segment",
                "threshold_source": "serialized validation-selected per-target threshold",
                "model_only_raw": _binary_metrics(labels, model_raw, raw),
                "model_only_reconciled": _binary_metrics(labels, model_reconciled, reconciled),
                "strict_operational": _binary_metrics(labels, strict, reconciled),
                "event_level_model_only": _target_event_metrics_fast(context, kind, model_reconciled, hours),
                "event_level_strict_operational": _target_event_metrics_fast(context, kind, strict, hours),
                "false_alerts_model_only": _false_episode_detail_fast(model_reconciled & normal_negative, context),
                "false_alerts_strict_operational": _false_episode_detail_fast(strict & normal_negative, context),
                "availability": {
                    "raw_probability_rows": int(valid_raw.sum()),
                    "reconciled_probability_rows": int(valid_reconciled.sum()),
                    "loaded_rows": int(loaded.sum()),
                    "eligible_rows": int(eligible.sum()),
                    "strict_crossing_rows": int(strict.sum()),
                    "withheld_or_ineligible_positive_rows": int((labels & ~strict).sum()),
                },
                "reconciliation": {
                    "changed_probability_rows": int((valid_raw & valid_reconciled & ~np.isclose(raw, reconciled)).sum()),
                    "changed_classification_rows": int((model_raw != model_reconciled).sum()),
                },
            }
    return output


def _evaluate_group(
    frame: pd.DataFrame,
    actual: pd.Series,
    maximum_gap_minutes: float,
) -> dict[str, Any]:
    subset = frame.reset_index(drop=True)
    subset_actual = actual.reset_index(drop=True)
    subset_trusted = pd.Series(False, index=subset.index)
    audit = _build_compact_audit(subset)
    context = build_evaluation_context(subset, subset_actual, maximum_gap_minutes)
    return fast_per_target_metrics(
        subset, subset_actual, subset_trusted, maximum_gap_minutes,
        context=context, audit=audit,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--maximum-gap-minutes", type=float, default=10.0)
    args = parser.parse_args()
    frame, actual = load_compact_frames(args.replay, args.labels_file, chunksize=args.chunksize)
    audit = _build_compact_audit(frame)
    audit_validation = validate_compact_audit(frame, audit)
    trusted = pd.Series(False, index=frame.index)
    context = build_evaluation_context(frame, actual, args.maximum_gap_minutes)
    report = {
        "evaluation_input": {
            "replay": str(Path(args.replay)),
            "labels_file": str(Path(args.labels_file)),
            "row_count": len(frame),
            "timestamp_alignment": "exact",
            "mode": "compact_model_quality; operational trust intentionally false",
            "maximum_gap_minutes": args.maximum_gap_minutes,
            "wide_columns_skipped": True,
            "json_validation_strategy": "parse each distinct compact payload once",
            "timestamp_segmentation_passes": 1,
        },
        "status_layers": status_layer_metrics(frame, actual),
        "compact_audit_validation": audit_validation,
        "per_probability_target": fast_per_target_metrics(
            frame, actual, trusted, args.maximum_gap_minutes,
            context=context, audit=audit,
        ),
        "group_metrics": {},
    }
    if args.metadata:
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        lifecycle_meta = {value["lifecycle_id"]: value for value in metadata.get("lifecycles", [])}
        dimensions = {
            "duration_bucket": {
                key: _duration_bucket(float(value["duration_hours"])) for key, value in lifecycle_meta.items()
            },
            "operating_regime": {
                key: str(value.get("operating_regime", "unknown")) for key, value in lifecycle_meta.items()
            },
        }
        lifecycle_values = frame["lifecycle_id"].astype(str)
        for dimension, mapping in dimensions.items():
            report["group_metrics"][dimension] = {}
            for group in sorted(set(mapping.values())):
                ids = {key for key, value in mapping.items() if value == group}
                mask = lifecycle_values.isin(ids)
                subset = frame.loc[mask].reset_index(drop=True)
                subset_actual = actual.loc[mask].reset_index(drop=True)
                report["group_metrics"][dimension][group] = {
                    "lifecycle_count": len(ids),
                    "row_count": int(mask.sum()),
                    "per_probability_target": _evaluate_group(
                        subset, subset_actual, args.maximum_gap_minutes
                    ) if len(subset) else {},
                }
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_output:
        report_csv_rows(report).to_csv(args.csv_output, index=False)
    print(json.dumps({
        "status": "completed",
        "row_count": len(frame),
        "json_output": str(output),
        "group_dimensions": sorted(report["group_metrics"]),
        "timestamp_segmentation_passes": 1,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
