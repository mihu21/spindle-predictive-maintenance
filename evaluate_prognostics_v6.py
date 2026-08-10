from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_RANK = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if labels.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(labels.size)
    value = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities >= low) & (probabilities < high if index < bins - 1 else probabilities <= high)
        if not np.any(mask):
            continue
        weight = float(np.sum(mask)) / total
        value += weight * abs(float(np.mean(probabilities[mask])) - float(np.mean(labels[mask])))
    return value


def _first_target_onset(group: pd.DataFrame, target_rank: int) -> pd.Timestamp | None:
    """Return the first confirmed target onset in one lifecycle.

    Prognostics evaluates the first WARNING/CRITICAL onset of a lifecycle. A later
    re-entry after the confirmed status temporarily falls back is not a new machine
    failure event and must not create another positive prediction window.
    """
    ranks = group["event_status"].map(STATUS_RANK).fillna(0).astype(int).to_numpy()
    indices = np.flatnonzero(ranks >= target_rank)
    if indices.size == 0:
        return None
    return pd.Timestamp(group.iloc[int(indices[0])]["timestamp"])


def _labels_to_first_event(
    group: pd.DataFrame, target_rank: int, horizon_hours: float
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp | None]:
    timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
    onset = _first_target_onset(group, target_rank)
    labels = np.zeros(len(group), dtype=bool)
    if onset is None:
        return np.ones(len(group), dtype=bool), labels, None

    onset64 = np.datetime64(onset.to_datetime64())
    eligible = timestamps < onset64
    delta_hours = (onset64 - timestamps) / np.timedelta64(1, "h")
    labels = eligible & (delta_hours > 0) & (delta_hours <= horizon_hours)
    return eligible, labels, onset


def _episode_count(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(np.sum(mask & ~np.r_[False, mask[:-1]]))


def _episode_durations_hours(mask: np.ndarray, cadence_hours: float) -> list[float]:
    """Return contiguous true-run durations using the lifecycle cadence."""
    if mask.size == 0 or cadence_hours <= 0:
        return []
    padded = np.r_[False, mask.astype(bool), False].astype(int)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [float(end - start) * cadence_hours for start, end in zip(starts, ends)]


def _contains_target(value: object, target: str) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return target in parsed
    except Exception:
        pass
    return target in {part.strip() for part in text.replace("|", ",").split(",") if part.strip()}


def evaluate_target(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon_hours: int,
    threshold: float,
) -> dict:
    target_rank = STATUS_RANK[target.upper()]
    probability_col = f"probability_{target.lower()}_{horizon_hours}h"
    eta_col = f"prognostic_time_to_{target.lower()}_hours"
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    false_episode_count = 0
    operational_false_episode_count = 0
    raw_false_alert_durations: list[float] = []
    operational_false_alert_durations: list[float] = []
    raw_false_alert_hours = 0.0
    operational_false_alert_hours = 0.0
    eligible_negative_hours = 0.0
    event_rows: list[dict] = []
    eta_errors: list[float] = []

    for lifecycle_id, group in frame.groupby("lifecycle_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        eligible, labels, onset = _labels_to_first_event(group, target_rank, float(horizon_hours))
        probs = pd.to_numeric(group[probability_col], errors="coerce").to_numpy(dtype=float)
        valid = eligible & np.isfinite(probs)
        predictions = probs >= threshold
        all_labels.append(labels[valid].astype(int))
        all_probs.append(probs[valid])
        all_predictions.append(predictions[valid].astype(int))

        negative_alert = valid & ~labels & predictions
        operational = np.asarray([
            _contains_target(value, probability_col)
            for value in group.get("probability_threshold_crossings", pd.Series([""] * len(group)))
        ], dtype=bool)
        operational_negative_alert = valid & ~labels & operational
        if len(group) > 1:
            cadence_hours = float(np.median(np.diff(group["timestamp"].to_numpy(dtype="datetime64[ns]")).astype("timedelta64[s]").astype(float))) / 3600.0
        else:
            cadence_hours = 0.0
        false_episode_count += _episode_count(negative_alert)
        operational_false_episode_count += _episode_count(operational_negative_alert)
        raw_durations = _episode_durations_hours(negative_alert, cadence_hours)
        operational_durations = _episode_durations_hours(operational_negative_alert, cadence_hours)
        raw_false_alert_durations.extend(raw_durations)
        operational_false_alert_durations.extend(operational_durations)
        raw_false_alert_hours += float(np.sum(negative_alert)) * cadence_hours
        operational_false_alert_hours += float(np.sum(operational_negative_alert)) * cadence_hours
        eligible_negative_hours += float(np.sum(valid & ~labels)) * cadence_hours

        if onset is not None:
            prior_mask = (group["timestamp"] < onset) & (group["timestamp"] >= onset - pd.Timedelta(hours=horizon_hours))
            prior = group.loc[prior_mask]
            prior_probs = pd.to_numeric(prior[probability_col], errors="coerce")
            alerted = prior.loc[prior_probs >= threshold]
            if len(alerted):
                first = alerted.iloc[0]["timestamp"]
                lead = (onset - first).total_seconds() / 3600.0
                detected = True
            else:
                lead = None
                detected = False
            prior_indices = np.flatnonzero(prior_mask.to_numpy())
            operational_prior = operational[prior_indices] if prior_indices.size else np.asarray([], dtype=bool)
            if np.any(operational_prior):
                first_operational_index = int(prior_indices[np.flatnonzero(operational_prior)[0]])
                operational_first = pd.Timestamp(group.iloc[first_operational_index]["timestamp"])
                operational_lead = (onset - operational_first).total_seconds() / 3600.0
                operational_detected = True
            else:
                operational_lead = None
                operational_detected = False
            event_rows.append({
                "lifecycle_id": str(lifecycle_id),
                "event_timestamp": onset.isoformat(),
                "detected_within_horizon": detected,
                "first_alert_lead_hours": lead,
                "operational_detected_within_horizon": operational_detected,
                "operational_first_alert_lead_hours": operational_lead,
            })

            # ETA error is evaluated only before the first confirmed target onset.
            timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
            onset64 = np.datetime64(onset.to_datetime64())
            actual = (onset64 - timestamps) / np.timedelta64(1, "h")
            eta = pd.to_numeric(group[eta_col], errors="coerce").to_numpy(dtype=float)
            eta_valid = eligible & np.isfinite(eta) & (actual > 0) & (actual <= horizon_hours)
            eta_errors.extend(np.abs(eta[eta_valid] - actual[eta_valid]).tolist())

    labels = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=int)
    probs = np.concatenate(all_probs) if all_probs else np.asarray([], dtype=float)
    predictions = np.concatenate(all_predictions) if all_predictions else np.asarray([], dtype=int)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    fn_rate = fn / (tp + fn) if tp + fn else None
    fp_rate = fp / (fp + tn) if fp + tn else None
    event_count = len(event_rows)
    event_detected = sum(bool(row["detected_within_horizon"]) for row in event_rows)
    operational_event_detected = sum(bool(row["operational_detected_within_horizon"]) for row in event_rows)
    leads = [float(row["first_alert_lead_hours"]) for row in event_rows if row["first_alert_lead_hours"] is not None]
    operational_leads = [float(row["operational_first_alert_lead_hours"]) for row in event_rows if row["operational_first_alert_lead_hours"] is not None]
    return {
        "target": target.lower(),
        "horizon_hours": horizon_hours,
        "probability_threshold": threshold,
        "eligible_row_count": int(labels.size),
        "positive_row_count": int(np.sum(labels)),
        "negative_row_count": int(np.sum(labels == 0)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "row_false_negative_rate": fn_rate,
        "row_false_positive_rate": fp_rate,
        "brier_score": float(np.mean((probs - labels) ** 2)) if labels.size else None,
        "expected_calibration_error": _ece(labels, probs),
        "event_count": event_count,
        "detected_event_count": event_detected,
        "event_false_negative_rate": ((event_count - event_detected) / event_count) if event_count else None,
        "event_lead_hours": {
            "minimum": min(leads) if leads else None,
            "median": float(np.median(leads)) if leads else None,
            "maximum": max(leads) if leads else None,
        },
        "operational_detected_event_count": operational_event_detected,
        "operational_event_false_negative_rate": ((event_count - operational_event_detected) / event_count) if event_count else None,
        "operational_event_lead_hours": {
            "minimum": min(operational_leads) if operational_leads else None,
            "median": float(np.median(operational_leads)) if operational_leads else None,
            "maximum": max(operational_leads) if operational_leads else None,
        },
        "raw_probability_false_alert_episode_count": false_episode_count,
        "operational_false_alert_episode_count": operational_false_episode_count,
        "raw_probability_false_alert_hours": raw_false_alert_hours,
        "operational_false_alert_hours": operational_false_alert_hours,
        "operational_false_alert_time_fraction": (operational_false_alert_hours / eligible_negative_hours) if eligible_negative_hours > 0 else None,
        "raw_false_alert_episode_duration_hours": {
            "median": float(np.median(raw_false_alert_durations)) if raw_false_alert_durations else None,
            "p90": float(np.percentile(raw_false_alert_durations, 90)) if raw_false_alert_durations else None,
            "maximum": max(raw_false_alert_durations) if raw_false_alert_durations else None,
        },
        "operational_false_alert_episode_duration_hours": {
            "median": float(np.median(operational_false_alert_durations)) if operational_false_alert_durations else None,
            "p90": float(np.percentile(operational_false_alert_durations, 90)) if operational_false_alert_durations else None,
            "maximum": max(operational_false_alert_durations) if operational_false_alert_durations else None,
        },
        "eligible_negative_hours": eligible_negative_hours,
        "raw_probability_false_alert_episodes_per_100_hours": (100.0 * false_episode_count / eligible_negative_hours) if eligible_negative_hours > 0 else None,
        "operational_false_alert_episodes_per_100_hours": (100.0 * operational_false_episode_count / eligible_negative_hours) if eligible_negative_hours > 0 else None,
        "eta_absolute_error_hours": {
            "sample_count": len(eta_errors),
            "mae": float(np.mean(eta_errors)) if eta_errors else None,
            "median": float(np.median(eta_errors)) if eta_errors else None,
            "p90": float(np.percentile(eta_errors, 90)) if eta_errors else None,
        },
        "events": event_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v6 model-based prognostics with event-oriented metrics")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warning-threshold", type=float, default=0.5)
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    args = parser.parse_args()
    frame = pd.read_csv(args.replay, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    required = {
        "timestamp", "lifecycle_id", "event_status",
        "probability_warning_12h", "probability_critical_24h",
        "prognostic_time_to_warning_hours", "prognostic_time_to_critical_hours",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Replay is missing required v6 columns: {missing}")
    report = {
        "schema_version": "1.0",
        "method": "probabilistic_degradation_v6",
        "evaluation_semantics": {
            "event_status": "confirmed time-based machine event status, not one-sample raw threshold spikes",
            "row_label": "current row is before the first confirmed target onset in its lifecycle and that first onset occurs within the configured elapsed-time horizon",
            "false_alert_episode": "contiguous probability-threshold crossing before the first confirmed target onset during eligible negative time",
            "operational_detection": "probability_threshold_crossings after runtime activation persistence and release hysteresis; this is the alert-policy detection metric",
            "event_definition": "at most one event per lifecycle per target: the first confirmed WARNING/CRITICAL onset; later status re-entries are not independent failures",
            "note": "Synthetic or previously inspected trajectories are development evidence, not independent production validation.",
        },
        "targets": {
            "probability_warning_12h": evaluate_target(frame, target="warning", horizon_hours=12, threshold=args.warning_threshold),
            "probability_critical_24h": evaluate_target(frame, target="critical", horizon_hours=24, threshold=args.critical_threshold),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
