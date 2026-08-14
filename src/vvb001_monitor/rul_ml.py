from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

from .rul import RULPrediction
from .rul_features_v2_4 import V24CausalFeatureBuilder, sensor_values_from_runtime_features

RUL_MODEL_VERSION_V2 = "supervised_quantile_rul_v2"
RUL_MODEL_VERSION_V2_1 = "supervised_quantile_rul_v2_1"
RUL_MODEL_VERSION = RUL_MODEL_VERSION_V2_1
RUL_MODEL_VERSION_V2_2 = "supervised_quantile_rul_v2_2"
RUL_MODEL_VERSION_V2_3 = "supervised_quantile_rul_v2_3_forecastability_aware"
RUL_MODEL_VERSION_V2_4 = "supervised_quantile_rul_v2_4_state_selective"
RUL_MODEL_VERSION_V2_5 = "supervised_quantile_rul_v2_5_active_population_calibrated"
RUL_MODEL_VERSION_V2_6 = "supervised_quantile_rul_v2_6_target_specific_corrected"
RUL_MODEL_VERSION_V2_7 = "supervised_quantile_rul_v2_7_identity_first_critical"
TARGET_SPECIFIC_RUL_MODEL_VERSIONS = frozenset({
    RUL_MODEL_VERSION_V2_6, RUL_MODEL_VERSION_V2_7,
})
SUPPORTED_RUL_MODEL_VERSIONS = frozenset({
    RUL_MODEL_VERSION_V2, RUL_MODEL_VERSION_V2_1, RUL_MODEL_VERSION_V2_2,
    RUL_MODEL_VERSION_V2_3, RUL_MODEL_VERSION_V2_4, RUL_MODEL_VERSION_V2_5,
    RUL_MODEL_VERSION_V2_6, RUL_MODEL_VERSION_V2_7,
})
RUL_CALIBRATION_METHOD_V2_1 = "lifecycle_hierarchical_post_stabilization_split_conformal_v2_1"
RUL_CALIBRATION_METHOD_V2_5 = "final_active_hierarchical_lifecycle_conformal_v2_5"
RUL_CALIBRATION_METHOD_V2_6 = "target_specific_corrected_asymmetric_conformal_v2_6"
RUL_BUCKETS = ("gt_48h", "24_48h", "12_24h", "6_12h", "le_6h")
RUL_DIAGNOSTIC_BUCKETS = ("le_6h", "6_12h", "12_24h", "24_48h", "48_72h", "gt_72h")
RUL_V2_2_CANDIDATES = ("extra_trees_v2_1", "random_forest_v2_2", "hist_gradient_boosting_v2_2")
RUL_TARGET_THRESHOLDS = {"warning": 0.35, "critical": 0.75}


def resolve_v2_5_serviceability(
    state: dict[str, int | bool],
    *,
    hard_eligible: bool,
    hard_reasons: Iterable[str] = (),
    selector_score: float | None,
    activation_threshold: float,
    deactivation_threshold: float,
    minimum_confirmations: int = 3,
    minimum_active_dwell: int = 3,
) -> dict[str, Any]:
    """Resolve the single pre-inference v2.5 serviceability intent.

    `serviceable_intent` is deliberately distinct from exported `RUL_ACTIVE`: the former is the
    frozen policy's decision before point/interval construction, while the latter is emitted only
    after a complete finite output contract has been produced.
    """
    reasons = tuple(dict.fromkeys(str(value) for value in hard_reasons if value))
    minimum_confirmations = max(1, int(minimum_confirmations))
    minimum_active_dwell = max(1, int(minimum_active_dwell))
    if not hard_eligible:
        state.update(active=False, confirmation_count=0, active_dwell=0)
        return {
            "serviceable_intent": False,
            "selector_active": False,
            "hysteresis_confirmed": False,
            "reason_code": reasons[0] if reasons else "INVALID_RUNTIME_STATE",
            "reasons": reasons or ("INVALID_RUNTIME_STATE",),
        }

    score = float(selector_score) if selector_score is not None else math.nan
    was_active = bool(state.get("active", False))
    raw_threshold = deactivation_threshold if was_active else activation_threshold
    selector_active = bool(math.isfinite(score) and score >= raw_threshold)
    if was_active:
        dwell = int(state.get("active_dwell", 0)) + 1
        if selector_active or dwell < minimum_active_dwell:
            state.update(active=True, confirmation_count=minimum_confirmations, active_dwell=dwell)
            return {
                "serviceable_intent": True,
                "selector_active": selector_active,
                "hysteresis_confirmed": True,
                "reason_code": None,
                "reasons": (),
            }
        state.update(active=False, confirmation_count=0, active_dwell=0)
        return {
            "serviceable_intent": False,
            "selector_active": False,
            "hysteresis_confirmed": False,
            "reason_code": "SELECTOR_BELOW_THRESHOLD",
            "reasons": ("SELECTOR_BELOW_THRESHOLD",),
        }

    if not selector_active:
        state.update(active=False, confirmation_count=0, active_dwell=0)
        return {
            "serviceable_intent": False,
            "selector_active": False,
            "hysteresis_confirmed": False,
            "reason_code": "SELECTOR_BELOW_THRESHOLD",
            "reasons": ("SELECTOR_BELOW_THRESHOLD",),
        }
    confirmations = int(state.get("confirmation_count", 0)) + 1
    if confirmations < minimum_confirmations:
        state.update(active=False, confirmation_count=confirmations, active_dwell=0)
        return {
            "serviceable_intent": False,
            "selector_active": True,
            "hysteresis_confirmed": False,
            "reason_code": "HYSTERESIS_NOT_CONFIRMED",
            "reasons": ("HYSTERESIS_NOT_CONFIRMED",),
        }
    state.update(active=True, confirmation_count=minimum_confirmations, active_dwell=1)
    return {
        "serviceable_intent": True,
        "selector_active": True,
        "hysteresis_confirmed": True,
        "reason_code": None,
        "reasons": (),
    }


def resolve_v2_6_target_serviceability(
    state: dict[str, int | str],
    *,
    hard_eligible: bool,
    hard_reasons: Iterable[str] = (),
    selector_score: float | None,
    activation_threshold: float,
    deactivation_threshold: float,
    confirmations: int = 3,
) -> dict[str, Any]:
    """Target-local fail-closed ACTIVE/LOW_CONFIDENCE/UNAVAILABLE transition.

    Hard invalidation clears exact RUL immediately. A soft selector failure also clears exact
    output immediately by entering LOW_CONFIDENCE; only the state label is debounced.
    """
    confirmations = max(1, int(confirmations))
    reasons = tuple(dict.fromkeys(str(value) for value in hard_reasons if value))
    if not hard_eligible:
        state.update(mode="RUL_UNAVAILABLE", valid_count=0, soft_count=0)
        return {
            "state": "RUL_UNAVAILABLE", "serviceable_intent": False,
            "selector_active": False, "reason_code": reasons[0] if reasons else "INVALID_RUNTIME_STATE",
            "reasons": reasons or ("INVALID_RUNTIME_STATE",),
        }
    score = float(selector_score) if selector_score is not None else math.nan
    mode = str(state.get("mode", "RUL_UNAVAILABLE"))
    threshold = deactivation_threshold if mode == "RUL_ACTIVE" else activation_threshold
    selector_active = bool(math.isfinite(score) and score >= threshold)
    if selector_active:
        if mode == "RUL_ACTIVE":
            state.update(mode="RUL_ACTIVE", valid_count=confirmations, soft_count=0)
            return {"state": "RUL_ACTIVE", "serviceable_intent": True, "selector_active": True,
                    "reason_code": None, "reasons": ()}
        valid_count = int(state.get("valid_count", 0)) + 1
        if valid_count >= confirmations:
            state.update(mode="RUL_ACTIVE", valid_count=confirmations, soft_count=0)
            return {"state": "RUL_ACTIVE", "serviceable_intent": True, "selector_active": True,
                    "reason_code": None, "reasons": ()}
        state.update(mode="RUL_LOW_CONFIDENCE", valid_count=valid_count, soft_count=0)
        return {"state": "RUL_LOW_CONFIDENCE", "serviceable_intent": False,
                "selector_active": True, "reason_code": "HYSTERESIS_NOT_CONFIRMED",
                "reasons": ("HYSTERESIS_NOT_CONFIRMED",)}
    if mode == "RUL_ACTIVE":
        soft_count = 1
        next_mode = "RUL_LOW_CONFIDENCE"
    elif mode == "RUL_LOW_CONFIDENCE":
        soft_count = int(state.get("soft_count", 0)) + 1
        next_mode = "RUL_UNAVAILABLE" if soft_count >= confirmations else "RUL_LOW_CONFIDENCE"
    else:
        soft_count = confirmations
        next_mode = "RUL_UNAVAILABLE"
    state.update(mode=next_mode, valid_count=0, soft_count=soft_count)
    return {"state": next_mode, "serviceable_intent": False, "selector_active": False,
            "reason_code": "SELECTOR_BELOW_THRESHOLD", "reasons": ("SELECTOR_BELOW_THRESHOLD",)}


def apply_v2_6_bias_correction(
    point: float,
    correction: dict[str, Any] | None,
) -> tuple[float, str]:
    """Apply a deliberately low-complexity banded or piecewise-linear correction."""
    config = dict(correction or {})
    raw = float(point)
    knots = list(config.get("knots") or [])
    if knots:
        ordered = sorted(knots, key=lambda row: float(row["point_hours"]))
        knot_points = np.asarray([float(row["point_hours"]) for row in ordered], dtype=float)
        knot_corrections = np.asarray(
            [float(row["correction_hours"]) for row in ordered], dtype=float,
        )
        correction_value = float(np.interp(
            raw, knot_points, knot_corrections,
            left=knot_corrections[0], right=knot_corrections[-1],
        ))
        horizon = float(config.get("max_forecast_hours", 720.0))
        return float(np.clip(raw + correction_value, 0.0, horizon)), "LINEAR_KNOTS"
    bands = list(config.get("bands") or [])
    selected = None
    for band in bands:
        lower = float(band.get("lower_hours", -math.inf))
        upper = float(band.get("upper_hours", math.inf))
        if lower <= raw < upper and bool(band.get("supported", True)):
            selected = band
            break
    if selected is None:
        selected = dict(config.get("global") or {"correction_hours": 0.0})
        name = "GLOBAL"
    else:
        name = str(selected.get("name") or "HORIZON_BAND")
    value = raw + float(selected.get("correction_hours", 0.0))
    horizon = float(config.get("max_forecast_hours", 720.0))
    return float(np.clip(value, 0.0, horizon)), name
RUL_DYNAMIC_FEATURES = (
    "degradation_score",
    "predicted_status_severity",
    "elapsed_lifecycle_hours",
    "score_rate_1h",
    "score_rate_6h",
    "score_rate_12h",
    "score_delta_1h",
    "score_delta_6h",
    "score_delta_12h",
    "score_std_6h",
    "score_std_12h",
)
FORBIDDEN_RUL_FEATURE_TOKENS = (
    "latent_damage",
    "true_damage",
    "true_state",
    "fault_mode",
    "future_critical",
    "future_warning",
    "critical_timestamp",
    "warning_timestamp",
    "hours_to_critical",
    "hours_to_warning",
    "lifecycle_progress",
    "actual_rul",
    "true_rul",
    "remaining_life",
    "target_rul",
    "rul_target",
)


def validate_rul_feature_names(feature_names: Iterable[str]) -> list[str]:
    """Reject simulator/future-only fields from the learned RUL input contract."""
    names = [str(name) for name in feature_names]
    unsafe = [
        name for name in names
        if any(token in name.lower() for token in FORBIDDEN_RUL_FEATURE_TOKENS)
    ]
    if unsafe:
        raise ValueError(f"Forbidden simulator/future-only RUL feature(s): {sorted(unsafe)}")
    if len(set(names)) != len(names):
        raise ValueError("RUL feature names must be unique")
    return names


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _window_stats(
    history: deque[tuple[datetime, float]] | list[tuple[datetime, float]],
    timestamp: datetime,
    hours: float,
) -> tuple[float | None, float | None, float | None]:
    cutoff = timestamp - timedelta(hours=hours)
    points = [(ts, value) for ts, value in history if ts >= cutoff and ts <= timestamp]
    if len(points) < 2:
        return None, None, None
    t0 = points[0][0]
    x = np.asarray([(ts - t0).total_seconds() / 3600.0 for ts, _ in points], dtype=float)
    y = np.asarray([value for _, value in points], dtype=float)
    if float(x[-1] - x[0]) <= 1e-12:
        return None, None, float(np.std(y))
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 1e-12 else None
    delta = float(y[-1] - y[0])
    return slope, delta, float(np.std(y))


def build_dynamic_rul_features(
    history: deque[tuple[datetime, float]] | list[tuple[datetime, float]],
    *,
    timestamp: datetime,
    lifecycle_started_at: datetime,
    degradation_score: float,
    predicted_status: str,
) -> dict[str, float]:
    rate_1, delta_1, _ = _window_stats(history, timestamp, 1.0)
    rate_6, delta_6, std_6 = _window_stats(history, timestamp, 6.0)
    rate_12, delta_12, std_12 = _window_stats(history, timestamp, 12.0)
    severity = {"NORMAL": 0.0, "WARNING": 1.0, "CRITICAL": 2.0}.get(str(predicted_status).upper(), 0.0)
    return {
        "degradation_score": float(degradation_score),
        "predicted_status_severity": severity,
        "elapsed_lifecycle_hours": max(0.0, (timestamp - lifecycle_started_at).total_seconds() / 3600.0),
        "score_rate_1h": float(rate_1) if rate_1 is not None else np.nan,
        "score_rate_6h": float(rate_6) if rate_6 is not None else np.nan,
        "score_rate_12h": float(rate_12) if rate_12 is not None else np.nan,
        "score_delta_1h": float(delta_1) if delta_1 is not None else np.nan,
        "score_delta_6h": float(delta_6) if delta_6 is not None else np.nan,
        "score_delta_12h": float(delta_12) if delta_12 is not None else np.nan,
        "score_std_6h": float(std_6) if std_6 is not None else np.nan,
        "score_std_12h": float(std_12) if std_12 is not None else np.nan,
    }


def causal_status_signals(
    model: Any,
    X: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    *,
    smoothing_tau_hours: float = 0.25,
    hysteresis_margin: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the frozen regime model causally within each lifecycle."""
    raw_scores = np.asarray(model.degradation_score(X), dtype=float)
    scores = np.full(len(X), np.nan, dtype=float)
    statuses = np.empty(len(X), dtype=object)
    for lifecycle in sorted(set(groups.tolist())):
        positions = np.flatnonzero(groups == lifecycle)
        positions = positions[np.argsort(np.asarray([timestamps[p].timestamp() for p in positions]))]
        previous_score: float | None = None
        previous_status: str | None = None
        previous_ts: datetime | None = None
        for pos in positions:
            value = float(raw_scores[pos])
            ts = timestamps[pos]
            if previous_score is None or previous_ts is None:
                smoothed = value
            else:
                dt_h = max(0.0, (ts - previous_ts).total_seconds() / 3600.0)
                alpha = 1.0 - math.exp(-dt_h / max(smoothing_tau_hours, 1e-6)) if dt_h > 0 else 0.0
                smoothed = float(alpha * value + (1.0 - alpha) * previous_score)
            status = str(model.status_from_score(smoothed, previous_status=previous_status, hysteresis=hysteresis_margin))
            scores[pos] = smoothed
            statuses[pos] = status
            previous_score = smoothed
            previous_status = status
            previous_ts = ts
    return scores, statuses


def build_rul_matrix(
    X: np.ndarray,
    base_feature_names: list[str],
    groups: np.ndarray,
    timestamps: np.ndarray,
    degradation_scores: np.ndarray,
    statuses: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    feature_names = validate_rul_feature_names(list(base_feature_names) + list(RUL_DYNAMIC_FEATURES))
    rows = np.full((len(X), len(feature_names)), np.nan, dtype=float)
    rows[:, : len(base_feature_names)] = np.asarray(X, dtype=float)
    offset = len(base_feature_names)
    for lifecycle in sorted(set(groups.tolist())):
        positions = np.flatnonzero(groups == lifecycle)
        positions = positions[np.argsort(np.asarray([timestamps[p].timestamp() for p in positions]))]
        started_at = timestamps[positions[0]]
        history: deque[tuple[datetime, float]] = deque()
        for pos in positions:
            ts = timestamps[pos]
            score = float(degradation_scores[pos])
            history.append((ts, score))
            cutoff = ts - timedelta(hours=12.0)
            while history and history[0][0] < cutoff:
                history.popleft()
            dynamic = build_dynamic_rul_features(
                history,
                timestamp=ts,
                lifecycle_started_at=started_at,
                degradation_score=score,
                predicted_status=str(statuses[pos]),
            )
            rows[pos, offset:] = [dynamic[name] for name in RUL_DYNAMIC_FEATURES]
    return rows, feature_names


def derive_time_to_onset_targets(
    groups: np.ndarray,
    timestamps: np.ndarray,
    latent_damage: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Build supervised labels from hidden simulator truth; never used as an input feature."""
    targets = np.full(len(groups), np.nan, dtype=float)
    for lifecycle in sorted(set(groups.tolist())):
        positions = np.flatnonzero(groups == lifecycle)
        positions = positions[np.argsort(np.asarray([timestamps[p].timestamp() for p in positions]))]
        onset_pos = next(
            (
                int(pos) for pos in positions
                if math.isfinite(float(latent_damage[pos])) and float(latent_damage[pos]) >= threshold
            ),
            None,
        )
        if onset_pos is None:
            continue
        onset_ts = timestamps[onset_pos]
        for pos in positions:
            delta_h = (onset_ts - timestamps[pos]).total_seconds() / 3600.0
            if delta_h < -1e-9:
                break
            targets[pos] = max(0.0, float(delta_h))
    return targets


def rul_sample_weights(groups: np.ndarray, targets: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Equalize lifecycle mass and emphasize the safety-important final 48/24/12/6 h."""
    gi = groups[indices]
    yi = targets[indices]
    weights = np.ones(len(indices), dtype=float)
    for lifecycle in set(gi.tolist()):
        mask = gi == lifecycle
        count = max(1, int(np.sum(mask)))
        weights[mask] *= 1.0 / count
    multipliers = np.ones(len(indices), dtype=float)
    multipliers[yi <= 48.0] = 2.0
    multipliers[yi <= 24.0] = 3.0
    multipliers[yi <= 12.0] = 4.5
    multipliers[yi <= 6.0] = 6.0
    weights *= multipliers
    weights *= len(weights) / max(float(np.sum(weights)), 1e-12)
    return weights


def diagnostic_horizon_bucket(true_rul_hours: float) -> str:
    value = float(true_rul_hours)
    if value <= 6.0:
        return "le_6h"
    if value <= 12.0:
        return "6_12h"
    if value <= 24.0:
        return "12_24h"
    if value <= 48.0:
        return "24_48h"
    if value <= 72.0:
        return "48_72h"
    return "gt_72h"


def rul_sample_weights_v2_2(
    groups: np.ndarray, targets: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Balance both lifecycle and true-horizon mass using deterministic iterative scaling."""
    gi = groups[indices]
    yi = targets[indices]
    bucket_names = np.asarray([diagnostic_horizon_bucket(value) for value in yi], dtype=object)
    lifecycles = sorted(set(gi.tolist()))
    buckets = sorted(set(bucket_names.tolist()))
    weights = np.ones(len(indices), dtype=float)
    # Damped iterative scaling is deterministic and training-only. Exact equal horizon margins
    # are not always mathematically compatible with exact equal lifecycle margins when short
    # lifecycles do not occupy the long buckets, so lifecycle equality is the hard constraint and
    # horizon equality is approached conservatively as a soft constraint.
    for _ in range(50):
        for lifecycle in lifecycles:
            mask = gi == lifecycle
            weights[mask] *= 1.0 / max(float(np.sum(weights[mask])), 1e-12)
        target_bucket_mass = len(lifecycles) / max(len(buckets), 1)
        for bucket in buckets:
            mask = bucket_names == bucket
            ratio = target_bucket_mass / max(float(np.sum(weights[mask])), 1e-12)
            weights[mask] *= math.sqrt(ratio)
    for lifecycle in lifecycles:
        mask = gi == lifecycle
        weights[mask] *= 1.0 / max(float(np.sum(weights[mask])), 1e-12)
    weights *= len(weights) / max(float(np.sum(weights)), 1e-12)
    return weights


def rul_sample_weights_v2_4(
    groups: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Lifecycle-equal weights with only moderate near-failure emphasis."""
    idx = np.asarray(indices, dtype=int)
    local_groups = groups[idx]
    local_targets = targets[idx]
    weights = np.ones(len(idx), dtype=float)
    multipliers = np.asarray([
        1.50 if value <= 6.0 else
        1.40 if value <= 12.0 else
        1.25 if value <= 24.0 else
        1.10 if value <= 48.0 else
        1.0
        for value in local_targets
    ], dtype=float)
    weights *= multipliers
    for lifecycle in sorted(set(local_groups.tolist())):
        mask = local_groups == lifecycle
        weights[mask] *= 1.0 / max(float(np.sum(weights[mask])), 1e-12)
    weights *= len(weights) / max(float(np.sum(weights)), 1e-12)
    return weights


class Log1pExtraTreesRegressor:
    """Fixed log1p-target Extra Trees candidate with sklearn-compatible prediction."""

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=32,
                max_depth=18,
                min_samples_leaf=4,
                max_features=0.8,
                n_jobs=1,
                random_state=self.seed,
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        final_step = self.model.steps[-1][0]
        self.model.fit(X, np.log1p(np.maximum(0.0, y)), **{
            f"{final_step}__sample_weight": sample_weight
        })
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.expm1(np.asarray(self.model.predict(X), dtype=float)))


class ColumnSubsetRegressor:
    """Persisted adapter allowing one runtime matrix to serve fixed feature ablations."""

    def __init__(self, model: Any, indices: Iterable[int]) -> None:
        self.model = model
        self.indices = np.asarray(list(indices), dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.asarray(self.model.predict(values[:, self.indices]), dtype=float)


def _fit_point_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
    *,
    estimator_name: str = "extra_trees_v2_1",
    shared_v2_2_preprocessing: bool = False,
):
    # ExtraTrees is fast enough for row-by-row online inference while still learning nonlinear
    # interactions between sensor dynamics, degradation state, and elapsed lifecycle time.
    if estimator_name == "extra_trees_v2_1":
        model = ExtraTreesRegressor(
            n_estimators=32, max_depth=18, min_samples_leaf=4, max_features=0.8,
            n_jobs=1, random_state=seed,
        )
    elif estimator_name == "random_forest_v2_2":
        model = RandomForestRegressor(
            n_estimators=64, max_depth=20, min_samples_leaf=4, max_features=0.8,
            n_jobs=1, random_state=seed,
        )
    elif estimator_name == "hist_gradient_boosting_v2_2":
        model = HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
        )
    elif estimator_name == "extra_trees_log1p_v2_4":
        model = Log1pExtraTreesRegressor(seed=seed)
    else:
        raise ValueError(f"Unknown predetermined RUL estimator: {estimator_name}")
    if estimator_name == "extra_trees_log1p_v2_4":
        model.fit(X, y, sample_weight)
    elif shared_v2_2_preprocessing:
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            model,
        )
        final_step = model.steps[-1][0]
        model.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _empirical_quantile(values: np.ndarray, q: float, method: str) -> float:
    if len(values) == 0:
        return 0.0
    try:
        return float(np.quantile(values, q, method=method))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, q, interpolation=method))


def _finite_sample_conformal_quantile(
    values: np.ndarray,
    target_coverage: float,
) -> tuple[float, int, bool]:
    """Return the split-conformal order statistic ceil((n + 1) * coverage).

    When the requested rank is n + 1, no finite sample can provide the nominal guarantee.  The
    maximum observed score is used and the rank-clipped flag is persisted for auditability.
    """
    scores = np.asarray(values, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0, 0, False
    if not 0.0 < float(target_coverage) < 1.0:
        raise ValueError("target_coverage must be strictly between 0 and 1")
    requested_rank = int(math.ceil((len(scores) + 1) * float(target_coverage)))
    rank = min(len(scores), requested_rank)
    ordered = np.sort(scores)
    return float(ordered[rank - 1]), requested_rank, requested_rank > len(scores)


def conformal_quantile(values: np.ndarray, target_coverage: float = 0.80) -> float:
    """Public deterministic finite-sample conformal quantile helper."""
    return _finite_sample_conformal_quantile(values, target_coverage)[0]


def rul_region_bucket(predicted_median_hours: float) -> str:
    """Select the fixed inference bucket from a causal predicted median, never true RUL."""
    value = float(predicted_median_hours)
    if not math.isfinite(value):
        raise ValueError("Cannot select an RUL calibration bucket from a non-finite median")
    if value > 48.0:
        return "gt_48h"
    if value > 24.0:
        return "24_48h"
    if value > 12.0:
        return "12_24h"
    if value > 6.0:
        return "6_12h"
    return "le_6h"


def _lifecycle_conformal_margin(
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    within_lifecycle_coverage: float,
    lifecycle_coverage: float,
) -> tuple[float, dict[str, Any]]:
    """Hierarchical conformal margin with complete lifecycles as exchangeability units.

    Each lifecycle first contributes one finite-sample score: the margin needed to cover the
    requested fraction of its rows.  A second finite-sample conformal quantile is then taken over
    those lifecycle scores, so long trajectories cannot dominate merely by containing more rows.
    """
    lifecycle_scores: list[float] = []
    lifecycle_details: dict[str, Any] = {}
    for lifecycle in sorted(set(groups.tolist())):
        local = np.asarray(scores[groups == lifecycle], dtype=float)
        margin, rank, clipped = _finite_sample_conformal_quantile(local, within_lifecycle_coverage)
        lifecycle_scores.append(margin)
        lifecycle_details[str(lifecycle)] = {
            "rows": int(len(local)),
            "required_margin_hours": float(margin),
            "within_lifecycle_rank": int(rank),
            "rank_clipped": bool(clipped),
        }
    if not lifecycle_scores:
        return 0.0, {
            "lifecycle_count": 0,
            "finite_sample_rank": 0,
            "rank_clipped": False,
            "per_lifecycle": {},
        }
    margin, rank, clipped = _finite_sample_conformal_quantile(
        np.asarray(lifecycle_scores, dtype=float), lifecycle_coverage
    )
    return margin, {
        "lifecycle_count": len(lifecycle_scores),
        "finite_sample_rank": int(rank),
        "rank_clipped": bool(clipped),
        "per_lifecycle": lifecycle_details,
    }


def _enforce_minimum_width(
    lower: float,
    median: float,
    upper: float,
    minimum_width: float,
    horizon: float,
) -> tuple[float, float, float]:
    lo, mid, hi = sorted([float(lower), float(median), float(upper)])
    mid = float(np.clip(mid, 0.0, horizon))
    lo = min(lo, mid)
    hi = max(hi, mid)
    half = max(0.0, float(minimum_width)) / 2.0
    lo = min(lo, mid - half)
    hi = max(hi, mid + half)
    lo = max(0.0, lo)
    hi = min(horizon, hi)
    if hi - lo < minimum_width and hi < horizon:
        hi = min(horizon, lo + minimum_width)
    if hi - lo < minimum_width and lo > 0.0:
        lo = max(0.0, hi - minimum_width)
    return float(lo), float(mid), float(max(mid, hi))


def _interval_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    hits = (truth >= lower) & (truth <= upper)
    widths = upper - lower
    lifecycle_coverage = [
        float(np.mean(hits[groups == lifecycle])) for lifecycle in sorted(set(groups.tolist()))
    ]
    return {
        "coverage": float(np.mean(hits)) if len(hits) else None,
        "macro_lifecycle_coverage": float(np.mean(lifecycle_coverage)) if lifecycle_coverage else None,
        "median_lifecycle_coverage": float(np.median(lifecycle_coverage)) if lifecycle_coverage else None,
        "worst_lifecycle_coverage": float(np.min(lifecycle_coverage)) if lifecycle_coverage else None,
        "best_lifecycle_coverage": float(np.max(lifecycle_coverage)) if lifecycle_coverage else None,
        "mean_interval_width_hours": float(np.mean(widths)) if len(widths) else None,
        "median_interval_width_hours": float(np.median(widths)) if len(widths) else None,
        "p90_interval_width_hours": _empirical_quantile(widths, 0.90, "higher") if len(widths) else None,
    }


def horizon_point_diagnostics(
    truth: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    minimum_lifecycles: int = 8,
) -> dict[str, Any]:
    """Return fixed true-RUL bucket diagnostics without hiding unsupported buckets."""
    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    groups = np.asarray(groups, dtype=object)
    finite = np.isfinite(truth) & np.isfinite(predicted)
    result: dict[str, Any] = {}
    for bucket in RUL_DIAGNOSTIC_BUCKETS:
        mask = finite & np.asarray(
            [diagnostic_horizon_bucket(value) == bucket for value in truth], dtype=bool
        )
        rows = int(np.sum(mask))
        lifecycle_values = sorted(set(groups[mask].tolist()))
        errors = predicted[mask] - truth[mask]
        lifecycle_mae = [
            float(np.mean(np.abs(errors[groups[mask] == lifecycle])))
            for lifecycle in lifecycle_values
        ]
        entry: dict[str, Any] = {
            "rows": rows,
            "lifecycles": len(lifecycle_values),
            "supported": len(lifecycle_values) >= minimum_lifecycles,
            "mae_hours": float(np.mean(np.abs(errors))) if rows else None,
            "median_abs_error_hours": float(np.median(np.abs(errors))) if rows else None,
            "p90_abs_error_hours": _empirical_quantile(np.abs(errors), 0.90, "higher") if rows else None,
            "mean_signed_error_hours": float(np.mean(errors)) if rows else None,
            "median_signed_error_hours": float(np.median(errors)) if rows else None,
            "mean_true_rul_hours": float(np.mean(truth[mask])) if rows else None,
            "mean_predicted_rul_hours": float(np.mean(predicted[mask])) if rows else None,
            "predicted_p10_hours": _empirical_quantile(predicted[mask], 0.10, "lower") if rows else None,
            "predicted_p50_hours": _empirical_quantile(predicted[mask], 0.50, "linear") if rows else None,
            "predicted_p90_hours": _empirical_quantile(predicted[mask], 0.90, "higher") if rows else None,
            "macro_lifecycle_mae_hours": float(np.mean(lifecycle_mae)) if lifecycle_mae else None,
        }
        if lower is not None and upper is not None and rows:
            interval = _interval_metrics(truth[mask], lower[mask], upper[mask], groups[mask])
            entry.update({
                "interval_coverage": interval["coverage"],
                "macro_lifecycle_coverage": interval["macro_lifecycle_coverage"],
                "mean_interval_width_hours": interval["mean_interval_width_hours"],
                "median_interval_width_hours": interval["median_interval_width_hours"],
                "p90_interval_width_hours": interval["p90_interval_width_hours"],
            })
        else:
            entry.update({
                "interval_coverage": None,
                "macro_lifecycle_coverage": None,
                "mean_interval_width_hours": None,
                "median_interval_width_hours": None,
                "p90_interval_width_hours": None,
            })
        result[bucket] = entry
    return result


def _group_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_idx: np.ndarray,
    *,
    seed: int,
    estimator_name: str = "extra_trees_v2_1",
    weighting_version: str = "v2_1",
    row_weight_multipliers: np.ndarray | None = None,
    split_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    local_groups = (np.asarray(split_groups, dtype=object) if split_groups is not None else groups)[fit_idx]
    unique_lifecycles = sorted(set(local_groups.tolist()))
    if len(unique_lifecycles) < 2:
        raise ValueError("At least two fit lifecycles are required for grouped OOF RUL residuals")
    folds = min(5, len(unique_lifecycles))
    predictions = np.full(len(fit_idx), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_local, holdout_local) in enumerate(
        splitter.split(X[fit_idx], y[fit_idx], groups=local_groups)
    ):
        train_indices = fit_idx[train_local]
        weights = (
            rul_sample_weights_v2_2(groups, y, train_indices)
            if weighting_version == "v2_2"
            else rul_sample_weights_v2_4(groups, y, train_indices)
            if weighting_version == "v2_4_flat"
            else rul_sample_weights(groups, y, train_indices)
        )
        if row_weight_multipliers is not None:
            weights *= np.asarray(row_weight_multipliers[train_indices], dtype=float)
            weights *= len(weights) / max(float(np.sum(weights)), 1e-12)
        fold_model = _fit_point_model(
            X[train_indices], y[train_indices], weights, seed + fold + 1,
            estimator_name=estimator_name,
            shared_v2_2_preprocessing=weighting_version in {"v2_2", "v2_4_flat"},
        )
        predictions[holdout_local] = np.asarray(fold_model.predict(X[fit_idx[holdout_local]]), dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Grouped OOF RUL prediction did not cover every fit row")
    return predictions, folds


def compare_rul_point_candidates(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    *,
    seed: int,
    minimum_long_horizon_lifecycles: int = 8,
) -> tuple[str, dict[str, Any]]:
    """Compare the frozen v2.2 candidate set using fit-lifecycle grouped OOF evidence only."""
    fit_idx = np.asarray([i for i in fit_indices if math.isfinite(float(y[i]))], dtype=int)
    comparisons: dict[str, Any] = {}
    for offset, name in enumerate(RUL_V2_2_CANDIDATES):
        raw, folds = _group_oof_predictions(
            X, y, groups, fit_idx, seed=seed + 100 * offset,
            estimator_name=name, weighting_version="v2_2",
        )
        bias = float(np.median(y[fit_idx] - raw))
        predicted = np.clip(raw + bias, 0.0, 720.0)
        diagnostics = horizon_point_diagnostics(
            y[fit_idx], predicted, groups[fit_idx],
            minimum_lifecycles=minimum_long_horizon_lifecycles,
        )
        lifecycle_mae = [
            float(np.mean(np.abs(predicted[groups[fit_idx] == lifecycle] - y[fit_idx][groups[fit_idx] == lifecycle])))
            for lifecycle in sorted(set(groups[fit_idx].tolist()))
        ]
        long_rows = y[fit_idx] > 48.0
        long_lifecycles = len(set(groups[fit_idx][long_rows].tolist()))
        comparisons[name] = {
            "oof_folds": folds,
            "oof_rows": int(len(fit_idx)),
            "train_oof_bias_correction_hours": bias,
            "overall_mae_hours": float(np.mean(np.abs(predicted - y[fit_idx]))),
            "overall_macro_lifecycle_mae_hours": float(np.mean(lifecycle_mae)),
            "long_horizon_rows": int(np.sum(long_rows)),
            "long_horizon_lifecycles": long_lifecycles,
            "long_horizon_supported": long_lifecycles >= minimum_long_horizon_lifecycles,
            "long_horizon_mae_hours": float(np.mean(np.abs(predicted[long_rows] - y[fit_idx][long_rows]))) if np.any(long_rows) else None,
            "long_horizon_signed_bias_hours": float(np.mean(predicted[long_rows] - y[fit_idx][long_rows])) if np.any(long_rows) else None,
            "horizon_diagnostics": diagnostics,
        }

    def selection_key(name: str) -> tuple[float, float, float, str]:
        row = comparisons[name]
        long_mae = row["long_horizon_mae_hours"] if row["long_horizon_supported"] else math.inf
        return (
            float(long_mae),
            float(row["overall_macro_lifecycle_mae_hours"]),
            float(row["overall_mae_hours"]),
            name,
        )

    selected = min(RUL_V2_2_CANDIDATES, key=selection_key)
    report = {
        "candidate_list_frozen_before_evaluation": list(RUL_V2_2_CANDIDATES),
        "target_transform": "none_predeclared",
        "weighting": "equal_lifecycle_hard_constraint_with_damped_horizon_balance_v2_2",
        "selection_rule": (
            "minimum supported >48h OOF MAE; tie-break lifecycle-macro OOF MAE, overall OOF MAE, then stable candidate name"
        ),
        "minimum_long_horizon_lifecycles": int(minimum_long_horizon_lifecycles),
        "selected": selected,
        "candidates": comparisons,
    }
    return selected, report


def _minimum_width_metadata(
    medians: np.ndarray,
    centered_residuals: np.ndarray,
    groups: np.ndarray,
    *,
    min_rows: int,
    min_lifecycles: int,
) -> dict[str, Any]:
    global_width = max(0.0, 2.0 * float(np.median(np.abs(centered_residuals))))
    result: dict[str, Any] = {
        "method": "train_group_oof_median_absolute_residual_v1",
        "global_width_hours": global_width,
        "buckets": {},
    }
    buckets = np.asarray([rul_region_bucket(value) for value in medians], dtype=object)
    for bucket in RUL_BUCKETS:
        mask = buckets == bucket
        lifecycle_count = len(set(groups[mask].tolist()))
        supported = int(np.sum(mask)) >= min_rows and lifecycle_count >= min_lifecycles
        width = max(0.0, 2.0 * float(np.median(np.abs(centered_residuals[mask])))) if supported else global_width
        result["buckets"][bucket] = {
            "width_hours": width,
            "rows": int(np.sum(mask)),
            "lifecycles": lifecycle_count,
            "fallback_used": not supported,
            "fallback_reason": None if supported else "insufficient_training_oof_support",
        }
    return result


def _minimum_width_for_bucket(calibration: dict[str, Any], bucket: str) -> float:
    metadata = dict(calibration.get("minimum_interval_widths") or {})
    global_width = float(metadata.get("global_width_hours", 0.0))
    bucket_meta = dict((metadata.get("buckets") or {}).get(bucket) or {})
    return float(bucket_meta.get("width_hours", global_width))


def _apply_v2_1_calibration(
    target_artifact: dict[str, Any],
    values: tuple[float, float, float],
    *,
    ordering_corrected: bool = False,
    calibration_context: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    calibration = dict(target_artifact.get("calibration") or {})
    horizon = float(calibration.get("max_forecast_hours", 720.0))
    raw_lo, mid, raw_hi = sorted(float(value) for value in values)
    mid = float(np.clip(mid, 0.0, horizon))
    if calibration.get("method") in {RUL_CALIBRATION_METHOD_V2_5, RUL_CALIBRATION_METHOD_V2_6}:
        context = calibration_context or {}
        candidate = str(calibration.get("candidate") or "global_active")
        if candidate == "selector_confidence_stratified":
            threshold = float(calibration.get("selector_high_threshold", 0.95))
            score = context.get("selector_score")
            stratum = "SELECTOR_HIGH" if score is not None and float(score) >= threshold else "SELECTOR_MODERATE"
        elif candidate == "support_stratified":
            threshold = float(calibration.get("support_mad_threshold_hours", 12.0))
            dispersion = context.get("neighbor_dispersion_hours")
            stratum = "SUPPORT_HIGH" if dispersion is not None and float(dispersion) <= threshold else "SUPPORT_MARGINAL"
        else:
            stratum = "GLOBAL"
        strata = dict(calibration.get("strata") or {})
        selected = dict(strata.get(stratum) or strata.get("GLOBAL") or {})
        fallback_used = stratum not in strata or bool(selected.get("fallback_used", False))
        lower_margin = max(0.0, float(selected.get("lower_margin_hours", selected.get("margin_hours", 0.0))))
        upper_margin = max(0.0, float(selected.get("upper_margin_hours", selected.get("margin_hours", 0.0))))
        lo = max(0.0, raw_lo - lower_margin)
        hi = min(horizon, raw_hi + upper_margin)
        return {
            "lower": lo,
            "median": mid,
            "upper": hi,
            "raw_lower": raw_lo,
            "raw_upper": raw_hi,
            "bucket": stratum if not fallback_used else f"{stratum}->GLOBAL",
            "method": calibration.get("method"),
            "margin_hours": max(lower_margin, upper_margin),
            "lower_margin_hours": lower_margin,
            "upper_margin_hours": upper_margin,
            "minimum_width_hours": 0.0,
            "ordering_corrected": ordering_corrected,
            "fallback_used": fallback_used,
        }
    bucket = rul_region_bucket(mid)
    minimum_width = _minimum_width_for_bucket(calibration, bucket)
    raw_lo, mid, raw_hi = _enforce_minimum_width(raw_lo, mid, raw_hi, minimum_width, horizon)
    bucket_meta = dict((calibration.get("buckets") or {}).get(bucket) or {})
    margin = float(bucket_meta.get("margin_hours", calibration.get("global_margin_hours", 0.0)))
    lo = max(0.0, raw_lo - margin)
    hi = min(horizon, raw_hi + margin)
    lo, mid, hi = _enforce_minimum_width(lo, mid, hi, minimum_width, horizon)
    return {
        "lower": lo,
        "median": mid,
        "upper": hi,
        "raw_lower": raw_lo,
        "raw_upper": raw_hi,
        "bucket": bucket,
        "method": calibration.get("method"),
        "margin_hours": margin,
        "minimum_width_hours": minimum_width,
        "ordering_corrected": ordering_corrected,
        "fallback_used": bool(bucket_meta.get("fallback_used", False)),
    }


def _prediction_from_point(
    target_artifact: dict[str, Any],
    point: float,
    *,
    apply_calibration: bool = True,
) -> dict[str, Any]:
    calibration = dict(target_artifact.get("calibration") or {})
    horizon = float(calibration.get("max_forecast_hours", 720.0))
    mid_unclipped = float(point) + float(calibration.get("point_bias_hours", 0.0))
    raw_values = (
        mid_unclipped + float(calibration.get("residual_p10_hours", 0.0)),
        mid_unclipped,
        mid_unclipped + float(calibration.get("residual_p90_hours", 0.0)),
    )
    if not all(math.isfinite(value) for value in raw_values):
        return {
            "lower": math.nan, "median": math.nan, "upper": math.nan,
            "raw_lower": math.nan, "raw_upper": math.nan,
            "bucket": None, "method": calibration.get("method"), "ordering_corrected": False,
        }
    ordering_corrected = not (raw_values[0] <= raw_values[1] <= raw_values[2])
    raw_lo, mid, raw_hi = sorted(raw_values)
    mid = float(np.clip(mid, 0.0, horizon))
    method = str(calibration.get("method") or "")
    if method not in {
        RUL_CALIBRATION_METHOD_V2_1, RUL_CALIBRATION_METHOD_V2_5,
        RUL_CALIBRATION_METHOD_V2_6,
    }:
        expansion = float(calibration.get("interval_expansion_hours", 0.0))
        lo, mid, hi = sorted([raw_lo - expansion, mid, raw_hi + expansion])
        return {
            "lower": float(np.clip(lo, 0.0, horizon)),
            "median": mid,
            "upper": float(np.clip(hi, 0.0, horizon)),
            "raw_lower": float(np.clip(raw_lo, 0.0, horizon)),
            "raw_upper": float(np.clip(raw_hi, 0.0, horizon)),
            "bucket": None,
            "method": method or "validation_only_asymmetric_residual_quantiles_plus_conformal_v2",
            "ordering_corrected": ordering_corrected,
        }

    base = _apply_v2_1_calibration(
        target_artifact,
        (raw_lo, mid, raw_hi),
        ordering_corrected=ordering_corrected,
    )
    if apply_calibration:
        return base
    return {
        **base,
        "lower": base["raw_lower"],
        "upper": base["raw_upper"],
        "margin_hours": 0.0,
    }


def fit_quantile_rul_target(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    seed: int,
    target_name: str,
    desired_interval_coverage: float = 0.80,
    desired_macro_lifecycle_coverage: float = 0.75,
    minimum_bucket_rows: int = 50,
    minimum_bucket_lifecycles: int = 8,
    max_forecast_hours: float = 720.0,
    estimator_name: str = "extra_trees_v2_1",
    weighting_version: str = "v2_1",
    row_weight_multipliers: np.ndarray | None = None,
    oof_split_groups: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a point model and calibrate a frozen train-only raw interval on validation lifecycles."""
    fit_idx = np.asarray([i for i in fit_indices if math.isfinite(float(y[i]))], dtype=int)
    val_idx = np.asarray([i for i in validation_indices if math.isfinite(float(y[i]))], dtype=int)
    if len(fit_idx) < 100:
        raise ValueError(f"Not enough completed-lifecycle RUL rows to train {target_name}: {len(fit_idx)}")
    if len(val_idx) < 20:
        raise ValueError(f"Not enough validation RUL rows to calibrate {target_name}: {len(val_idx)}")

    if minimum_bucket_rows < 1 or minimum_bucket_lifecycles < 1:
        raise ValueError("Bucket support requirements must be positive")
    oof_raw, oof_folds = _group_oof_predictions(
        X, y, groups, fit_idx, seed=seed + 100,
        estimator_name=estimator_name, weighting_version=weighting_version,
        row_weight_multipliers=row_weight_multipliers,
        split_groups=oof_split_groups,
    )
    fit_truth = y[fit_idx]
    point_bias = float(np.median(fit_truth - oof_raw))
    oof_median = np.clip(oof_raw + point_bias, 0.0, max_forecast_hours)
    oof_residual = fit_truth - oof_median
    residual_lo = _empirical_quantile(oof_residual, 0.10, "lower")
    residual_hi = _empirical_quantile(oof_residual, 0.90, "higher")
    minimum_widths = _minimum_width_metadata(
        oof_median,
        oof_residual,
        groups[fit_idx],
        min_rows=minimum_bucket_rows,
        min_lifecycles=minimum_bucket_lifecycles,
    )

    weights = (
        rul_sample_weights_v2_2(groups, y, fit_idx)
        if weighting_version == "v2_2"
        else rul_sample_weights_v2_4(groups, y, fit_idx)
        if weighting_version == "v2_4_flat"
        else rul_sample_weights(groups, y, fit_idx)
    )
    if row_weight_multipliers is not None:
        weights *= np.asarray(row_weight_multipliers[fit_idx], dtype=float)
        weights *= len(weights) / max(float(np.sum(weights)), 1e-12)
    model = _fit_point_model(
        X[fit_idx], y[fit_idx], weights, seed, estimator_name=estimator_name,
        shared_v2_2_preprocessing=weighting_version in {"v2_2", "v2_4_flat"},
    )
    validation_points = np.asarray(model.predict(X[val_idx]), dtype=float)
    truth = y[val_idx]

    def point_audit(predictions: np.ndarray) -> dict[str, Any]:
        clipped = np.clip(predictions, 0.0, max_forecast_hours)
        result: dict[str, Any] = {
            "mae_hours": float(np.mean(np.abs(clipped - truth))),
            "median_abs_error_hours": float(np.median(np.abs(clipped - truth))),
        }
        for horizon in (24.0, 12.0, 6.0):
            mask = (truth > 0.0) & (truth <= horizon)
            result[f"within_{int(horizon)}h_mae_hours"] = (
                float(np.mean(np.abs(clipped[mask] - truth[mask]))) if np.any(mask) else None
            )
        return result

    point_median_audit = {
        "unchanged_extra_trees_point_model": point_audit(validation_points),
        "train_group_oof_bias_corrected": point_audit(validation_points + point_bias),
        "selected": "train_group_oof_bias_corrected",
        "selection_policy": (
            "Precommitted train-only leakage-safe bias correction; validation comparison is audit-only "
            "and does not select or refit the predictor."
        ),
    }
    temporary_calibration = {
        "method": RUL_CALIBRATION_METHOD_V2_1,
        "point_bias_hours": point_bias,
        "residual_p10_hours": residual_lo,
        "residual_p90_hours": residual_hi,
        "minimum_interval_widths": minimum_widths,
        "max_forecast_hours": float(max_forecast_hours),
        "global_margin_hours": 0.0,
        "buckets": {},
    }
    temporary_artifact = {"calibration": temporary_calibration}
    base = [_prediction_from_point(temporary_artifact, value) for value in validation_points]
    med = np.asarray([row["median"] for row in base], dtype=float)
    raw_lo = np.asarray([row["raw_lower"] for row in base], dtype=float)
    raw_hi = np.asarray([row["raw_upper"] for row in base], dtype=float)
    predicted_buckets = np.asarray([row["bucket"] for row in base], dtype=object)
    nonconformity = np.maximum.reduce(
        [raw_lo - truth, truth - raw_hi, np.zeros(len(truth), dtype=float)]
    )

    micro_margin, global_rank, global_rank_clipped = _finite_sample_conformal_quantile(
        nonconformity, desired_interval_coverage
    )
    macro_margin, global_lifecycle_conformal = _lifecycle_conformal_margin(
        nonconformity,
        groups[val_idx],
        within_lifecycle_coverage=desired_interval_coverage,
        lifecycle_coverage=desired_macro_lifecycle_coverage,
    )
    global_margin = max(micro_margin, macro_margin)
    buckets: dict[str, Any] = {}
    lo_cal = np.empty(len(val_idx), dtype=float)
    hi_cal = np.empty(len(val_idx), dtype=float)
    for bucket in RUL_BUCKETS:
        mask = predicted_buckets == bucket
        bucket_groups = groups[val_idx][mask]
        rows = int(np.sum(mask))
        lifecycle_count = len(set(bucket_groups.tolist()))
        supported = rows >= minimum_bucket_rows and lifecycle_count >= minimum_bucket_lifecycles
        if supported:
            bucket_micro, bucket_rank, bucket_rank_clipped = _finite_sample_conformal_quantile(
                nonconformity[mask], desired_interval_coverage
            )
            bucket_macro, bucket_lifecycle_conformal = _lifecycle_conformal_margin(
                nonconformity[mask],
                bucket_groups,
                within_lifecycle_coverage=desired_interval_coverage,
                lifecycle_coverage=desired_macro_lifecycle_coverage,
            )
            margin = max(bucket_micro, bucket_macro)
            fallback_reason = None
        else:
            bucket_micro = bucket_macro = None
            bucket_lifecycle_conformal = None
            bucket_rank = None
            bucket_rank_clipped = False
            margin = global_margin
            fallback_reason = "insufficient_validation_support"
        lo_bucket = np.maximum(0.0, raw_lo[mask] - margin)
        hi_bucket = np.minimum(max_forecast_hours, raw_hi[mask] + margin)
        lo_cal[mask] = lo_bucket
        hi_cal[mask] = hi_bucket
        raw_metrics = _interval_metrics(truth[mask], raw_lo[mask], raw_hi[mask], bucket_groups) if rows else {}
        calibrated_metrics = _interval_metrics(truth[mask], lo_bucket, hi_bucket, bucket_groups) if rows else {}
        buckets[bucket] = {
            "margin_hours": float(margin),
            "rows": rows,
            "lifecycles": lifecycle_count,
            "raw_coverage": raw_metrics.get("coverage"),
            "calibrated_coverage": calibrated_metrics.get("coverage"),
            "macro_lifecycle_coverage": calibrated_metrics.get("macro_lifecycle_coverage"),
            "mean_interval_width_hours": calibrated_metrics.get("mean_interval_width_hours"),
            "median_interval_width_hours": calibrated_metrics.get("median_interval_width_hours"),
            "finite_sample_rank": bucket_rank,
            "finite_sample_rank_clipped": bucket_rank_clipped,
            "micro_margin_hours": bucket_micro,
            "macro_margin_hours": bucket_macro,
            "lifecycle_conformal": bucket_lifecycle_conformal,
            "fallback_used": not supported,
            "fallback_reason": fallback_reason,
        }

    raw_metrics = _interval_metrics(truth, raw_lo, raw_hi, groups[val_idx])
    calibrated_metrics = _interval_metrics(truth, lo_cal, hi_cal, groups[val_idx])

    artifact = {
        "model": model,
        "calibration": {
            "method": RUL_CALIBRATION_METHOD_V2_1,
            "target_coverage": float(desired_interval_coverage),
            "target_macro_lifecycle_coverage": float(desired_macro_lifecycle_coverage),
            "raw_interval_method": "train_group_oof_asymmetric_residuals_v1",
            "raw_interval_oof_folds": int(oof_folds),
            "point_estimator": estimator_name,
            "sample_weighting_version": weighting_version,
            "point_bias_hours": point_bias,
            "point_median_audit": point_median_audit,
            "residual_p10_hours": residual_lo,
            "residual_p90_hours": residual_hi,
            "global_margin_hours": float(global_margin),
            "global_micro_margin_hours": float(micro_margin),
            "global_macro_margin_hours": float(macro_margin),
            "global_lifecycle_conformal": global_lifecycle_conformal,
            "global_finite_sample_rank": int(global_rank),
            "global_finite_sample_rank_clipped": bool(global_rank_clipped),
            "buckets": buckets,
            "minimum_interval_widths": minimum_widths,
            "fallback_rules": {
                "minimum_validation_rows_per_bucket": int(minimum_bucket_rows),
                "minimum_validation_lifecycles_per_bucket": int(minimum_bucket_lifecycles),
                "fallback": "global_validation_conformal_margin",
            },
            "validation_rows": int(len(val_idx)),
            "validation_lifecycles": sorted(set(groups[val_idx].tolist())),
            "validation_raw_interval": raw_metrics,
            "validation_calibrated_interval": calibrated_metrics,
            "max_forecast_hours": float(max_forecast_hours),
            "upper_bound_truncation": "clip_to_persisted_max_forecast_hours",
        },
    }
    metrics = {
        "fit_rows": int(len(fit_idx)),
        "validation_rows": int(len(val_idx)),
        "fit_lifecycles": sorted(set(groups[fit_idx].tolist())),
        "validation_lifecycles": sorted(set(groups[val_idx].tolist())),
        "validation_mae_hours": float(np.mean(np.abs(np.clip(med, 0.0, max_forecast_hours) - truth))),
        "validation_median_abs_error_hours": float(np.median(np.abs(np.clip(med, 0.0, max_forecast_hours) - truth))),
        "point_median_audit": point_median_audit,
        "validation_raw_interval": raw_metrics,
        "validation_calibrated_interval": calibrated_metrics,
        "validation_interval_coverage": calibrated_metrics["coverage"],
        "validation_macro_lifecycle_coverage": calibrated_metrics["macro_lifecycle_coverage"],
        "bucket_calibration": buckets,
        "sample_weighting": {
            "version": weighting_version,
            "lifecycle_equalization": True,
            "strategy": (
                "equal_lifecycle_hard_constraint_with_damped_horizon_balance"
                if weighting_version == "v2_2"
                else "near_failure_multipliers_v2_1"
            ),
        },
    }
    return artifact, metrics


def predict_quantiles(target_artifact: dict[str, Any], row: np.ndarray) -> tuple[float, float, float]:
    X = np.asarray(row, dtype=float).reshape(1, -1)
    point = float(target_artifact["model"].predict(X)[0])
    prediction = _prediction_from_point(target_artifact, point)
    return prediction["lower"], prediction["median"], prediction["upper"]


def predict_quantiles_with_metadata(target_artifact: dict[str, Any], row: np.ndarray) -> dict[str, Any]:
    X = np.asarray(row, dtype=float).reshape(1, -1)
    point = float(target_artifact["model"].predict(X)[0])
    return _prediction_from_point(target_artifact, point)


def evaluate_quantile_rul_target(
    target_artifact: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    idx = np.asarray([i for i in indices if math.isfinite(float(y[i]))], dtype=int)
    if len(idx) == 0:
        return {
            "rows": 0,
            "lifecycles": [],
            "mae_hours": None,
            "median_abs_error_hours": None,
            "interval_coverage": None,
        }
    point = np.asarray(target_artifact["model"].predict(X[idx]), dtype=float)
    predictions = [_prediction_from_point(target_artifact, value) for value in point]
    med = np.asarray([row["median"] for row in predictions], dtype=float)
    lo = np.asarray([row["lower"] for row in predictions], dtype=float)
    hi = np.asarray([row["upper"] for row in predictions], dtype=float)
    raw_lo = np.asarray([row["raw_lower"] for row in predictions], dtype=float)
    raw_hi = np.asarray([row["raw_upper"] for row in predictions], dtype=float)
    truth = y[idx]
    abs_error = np.abs(med - truth)
    return {
        "rows": int(len(idx)),
        "lifecycles": sorted(set(groups[idx].tolist())),
        "mae_hours": float(np.mean(abs_error)),
        "median_abs_error_hours": float(np.median(abs_error)),
        "raw_interval": _interval_metrics(truth, raw_lo, raw_hi, groups[idx]),
        "calibrated_interval": _interval_metrics(truth, lo, hi, groups[idx]),
        "interval_coverage": float(np.mean((truth >= lo) & (truth <= hi))),
        "macro_lifecycle_interval_coverage": _interval_metrics(truth, lo, hi, groups[idx])["macro_lifecycle_coverage"],
        "mean_interval_width_hours": float(np.mean(hi - lo)),
        "median_interval_width_hours": float(np.median(hi - lo)),
    }


def _stabilize_prediction_pair_values(
    warning: tuple[float, float, float],
    critical: tuple[float, float, float],
    *,
    previous_warning_mid: float | None,
    previous_critical_mid: float | None,
    timestamp: datetime,
    elapsed_h: float,
    last_revision_at: datetime | None,
    max_upward_jump_floor_hours: float,
    max_upward_jump_per_elapsed_hour: float,
    upward_revision_cooldown_hours: float,
    upward_revision_trigger_hours: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], datetime | None]:
    allowed = max(
        max_upward_jump_floor_hours,
        max_upward_jump_per_elapsed_hour * max(0.0, elapsed_h),
    )
    cooldown_ready = (
        last_revision_at is None
        or (timestamp - last_revision_at).total_seconds() / 3600.0 >= upward_revision_cooldown_hours
    )
    upward_revision_used = False

    def stabilize(
        values: tuple[float, float, float], prev_mid: float | None
    ) -> tuple[float, float, float]:
        nonlocal upward_revision_used
        lo, mid, hi = values
        if prev_mid is None or mid <= prev_mid:
            return values
        requested = mid - prev_mid
        if requested < upward_revision_trigger_hours or not cooldown_ready or upward_revision_used:
            shift = requested
            return max(0.0, lo - shift), prev_mid, max(prev_mid, hi - shift)
        revised_mid = prev_mid + min(requested, allowed)
        shift = mid - revised_mid
        upward_revision_used = True
        return max(0.0, lo - shift), revised_mid, max(revised_mid, hi - shift)

    stabilized_warning = stabilize(warning, previous_warning_mid)
    stabilized_critical = stabilize(critical, previous_critical_mid)
    return (
        stabilized_warning,
        stabilized_critical,
        timestamp if upward_revision_used else last_revision_at,
    )


def _calibrate_stabilized_target(
    target_artifact: dict[str, Any],
    truth: np.ndarray,
    groups: np.ndarray,
    medians: np.ndarray,
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    *,
    target_coverage: float,
    target_macro_lifecycle_coverage: float,
    minimum_bucket_rows: int,
    minimum_bucket_lifecycles: int,
) -> dict[str, Any]:
    calibration = dict(target_artifact.get("calibration") or {})
    horizon = float(calibration.get("max_forecast_hours", 720.0))
    predicted_buckets = np.asarray([rul_region_bucket(value) for value in medians], dtype=object)
    scores = np.maximum.reduce(
        [raw_lower - truth, truth - raw_upper, np.zeros(len(truth), dtype=float)]
    )
    micro_margin, global_rank, global_rank_clipped = _finite_sample_conformal_quantile(
        scores, target_coverage
    )
    lifecycle_margin, lifecycle_metadata = _lifecycle_conformal_margin(
        scores,
        groups,
        within_lifecycle_coverage=target_coverage,
        lifecycle_coverage=target_macro_lifecycle_coverage,
    )
    global_margin = max(micro_margin, lifecycle_margin)
    calibrated_lower = np.empty(len(truth), dtype=float)
    calibrated_upper = np.empty(len(truth), dtype=float)
    bucket_metadata: dict[str, Any] = {}
    for bucket in RUL_BUCKETS:
        mask = predicted_buckets == bucket
        local_groups = groups[mask]
        rows = int(np.sum(mask))
        lifecycle_count = len(set(local_groups.tolist()))
        supported = rows >= minimum_bucket_rows and lifecycle_count >= minimum_bucket_lifecycles
        if supported:
            local_micro, local_rank, local_clipped = _finite_sample_conformal_quantile(
                scores[mask], target_coverage
            )
            local_lifecycle, local_lifecycle_metadata = _lifecycle_conformal_margin(
                scores[mask],
                local_groups,
                within_lifecycle_coverage=target_coverage,
                lifecycle_coverage=target_macro_lifecycle_coverage,
            )
            margin = max(local_micro, local_lifecycle)
            fallback_reason = None
        else:
            local_micro = local_lifecycle = None
            local_rank = None
            local_clipped = False
            local_lifecycle_metadata = None
            margin = global_margin
            fallback_reason = "insufficient_validation_lifecycle_support"
        lo = np.maximum(0.0, raw_lower[mask] - margin)
        hi = np.minimum(horizon, raw_upper[mask] + margin)
        calibrated_lower[mask] = lo
        calibrated_upper[mask] = hi
        raw_metrics = _interval_metrics(truth[mask], raw_lower[mask], raw_upper[mask], local_groups) if rows else {}
        final_metrics = _interval_metrics(truth[mask], lo, hi, local_groups) if rows else {}
        bucket_metadata[bucket] = {
            "margin_hours": float(margin),
            "rows": rows,
            "lifecycles": lifecycle_count,
            "raw_coverage": raw_metrics.get("coverage"),
            "calibrated_coverage": final_metrics.get("coverage"),
            "macro_lifecycle_coverage": final_metrics.get("macro_lifecycle_coverage"),
            "mean_interval_width_hours": final_metrics.get("mean_interval_width_hours"),
            "median_interval_width_hours": final_metrics.get("median_interval_width_hours"),
            "finite_sample_rank": local_rank,
            "finite_sample_rank_clipped": local_clipped,
            "micro_margin_hours": local_micro,
            "lifecycle_margin_hours": local_lifecycle,
            "lifecycle_conformal": local_lifecycle_metadata,
            "fallback_used": not supported,
            "fallback_reason": fallback_reason,
        }
    raw_metrics = _interval_metrics(truth, raw_lower, raw_upper, groups)
    final_metrics = _interval_metrics(truth, calibrated_lower, calibrated_upper, groups)
    calibration.update({
        "method": RUL_CALIBRATION_METHOD_V2_1,
        "calibration_prediction_stage": "after_causal_temporal_stabilization",
        "target_coverage": float(target_coverage),
        "target_macro_lifecycle_coverage": float(target_macro_lifecycle_coverage),
        "global_margin_hours": float(global_margin),
        "global_micro_margin_hours": float(micro_margin),
        "global_lifecycle_margin_hours": float(lifecycle_margin),
        "global_finite_sample_rank": int(global_rank),
        "global_finite_sample_rank_clipped": bool(global_rank_clipped),
        "global_lifecycle_conformal": lifecycle_metadata,
        "buckets": bucket_metadata,
        "validation_rows": int(len(truth)),
        "validation_lifecycles": sorted(set(groups.tolist())),
        "validation_raw_interval": raw_metrics,
        "validation_calibrated_interval": final_metrics,
        "fallback_rules": {
            "minimum_validation_rows_per_bucket": int(minimum_bucket_rows),
            "minimum_validation_lifecycles_per_bucket": int(minimum_bucket_lifecycles),
            "fallback": "global_hierarchical_lifecycle_conformal_margin",
        },
    })
    target_artifact["calibration"] = calibration
    abs_error = np.abs(medians - truth)
    return {
        "validation_rows": int(len(truth)),
        "validation_lifecycles": sorted(set(groups.tolist())),
        "validation_mae_hours": float(np.mean(abs_error)),
        "validation_median_abs_error_hours": float(np.median(abs_error)),
        "validation_raw_interval": raw_metrics,
        "validation_calibrated_interval": final_metrics,
        "validation_interval_coverage": final_metrics["coverage"],
        "validation_macro_lifecycle_coverage": final_metrics["macro_lifecycle_coverage"],
        "bucket_calibration": bucket_metadata,
        "calibration_prediction_stage": "after_causal_temporal_stabilization",
    }


def calibrate_rul_targets_temporally(
    targets: dict[str, dict[str, Any]],
    X: np.ndarray,
    y_targets: dict[str, np.ndarray],
    groups: np.ndarray,
    timestamps: np.ndarray,
    validation_indices: np.ndarray,
    *,
    target_coverage: float = 0.80,
    target_macro_lifecycle_coverage: float = 0.75,
    minimum_bucket_rows: int = 50,
    minimum_bucket_lifecycles: int = 8,
    min_history_hours: float = 0.5,
    min_points: int = 6,
    max_upward_jump_floor_hours: float = 0.5,
    max_upward_jump_per_elapsed_hour: float = 1.5,
    upward_revision_cooldown_hours: float = 6.0,
    upward_revision_trigger_hours: float = 4.0,
    activation_scores: np.ndarray | None = None,
    activation_statuses: np.ndarray | None = None,
    minimum_activation_score: float | None = None,
    activation_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Calibrate the exact stateful interval path emitted at runtime."""
    validation_idx = np.asarray(validation_indices, dtype=int)
    points_by_target = {
        name: np.asarray(artifact["model"].predict(X[validation_idx]), dtype=float)
        for name, artifact in targets.items()
    }
    raw_by_target = {
        name: [_prediction_from_point(targets[name], value, apply_calibration=False) for value in values]
        for name, values in points_by_target.items()
    }
    stabilized: dict[str, list[tuple[float, float, float] | None]] = {
        name: [None] * len(validation_idx) for name in targets
    }
    runtime_eligible = np.zeros(len(validation_idx), dtype=bool)
    validation_groups = groups[validation_idx]
    active_contract = np.ones(len(validation_idx), dtype=bool)
    if activation_mask is not None:
        supplied = np.asarray(activation_mask, dtype=bool)
        if len(supplied) != len(groups):
            raise ValueError("Forecastability activation mask must align with the full corpus")
        active_contract = supplied[validation_idx]
    elif minimum_activation_score is not None:
        if activation_scores is None or activation_statuses is None:
            raise ValueError("Forecastability-aware calibration requires activation scores and statuses")
        active_contract = (
            np.asarray(activation_scores[validation_idx], dtype=float) >= float(minimum_activation_score)
        ) | (np.asarray(activation_statuses[validation_idx], dtype=object) != "NORMAL")
    for lifecycle in sorted(set(validation_groups.tolist())):
        local_positions = np.flatnonzero(validation_groups == lifecycle)
        local_positions = local_positions[np.argsort(np.asarray([
            timestamps[validation_idx[pos]].timestamp() for pos in local_positions
        ]))]
        started_at = timestamps[validation_idx[local_positions[0]]]
        recent_timestamps: deque[datetime] = deque()
        previous_warning_mid: float | None = None
        previous_critical_mid: float | None = None
        previous_ts: datetime | None = None
        last_revision_at: datetime | None = None
        for local_pos in local_positions:
            ts = timestamps[validation_idx[local_pos]]
            recent_timestamps.append(ts)
            cutoff = ts - timedelta(hours=12.0)
            while recent_timestamps and recent_timestamps[0] < cutoff:
                recent_timestamps.popleft()
            history_hours = max(0.0, (ts - started_at).total_seconds() / 3600.0)
            eligible = len(recent_timestamps) >= min_points and history_hours >= min_history_hours
            runtime_eligible[local_pos] = eligible
            if not eligible:
                previous_ts = ts
                continue
            warning_raw = raw_by_target["warning"][local_pos]
            critical_raw = raw_by_target["critical"][local_pos]
            warning_values = (warning_raw["lower"], warning_raw["median"], warning_raw["upper"])
            critical_values = (critical_raw["lower"], critical_raw["median"], critical_raw["upper"])
            elapsed_h = max(0.0, (ts - previous_ts).total_seconds() / 3600.0) if previous_ts else 0.0
            warning_values, critical_values, last_revision_at = _stabilize_prediction_pair_values(
                warning_values,
                critical_values,
                previous_warning_mid=previous_warning_mid,
                previous_critical_mid=previous_critical_mid,
                timestamp=ts,
                elapsed_h=elapsed_h,
                last_revision_at=last_revision_at,
                max_upward_jump_floor_hours=max_upward_jump_floor_hours,
                max_upward_jump_per_elapsed_hour=max_upward_jump_per_elapsed_hour,
                upward_revision_cooldown_hours=upward_revision_cooldown_hours,
                upward_revision_trigger_hours=upward_revision_trigger_hours,
            )
            stabilized["warning"][local_pos] = warning_values
            stabilized["critical"][local_pos] = critical_values
            previous_warning_mid = warning_values[1]
            previous_critical_mid = critical_values[1]
            previous_ts = ts

    metrics: dict[str, dict[str, Any]] = {}
    for name, artifact in targets.items():
        truth_all = y_targets[name][validation_idx]
        eligible = runtime_eligible & active_contract & np.isfinite(truth_all) & (truth_all > 0.0)
        values = [stabilized[name][pos] for pos in np.flatnonzero(eligible)]
        if not values or any(value is None for value in values):
            raise RuntimeError(f"No eligible stabilized validation predictions for {name}")
        intervals = np.asarray(values, dtype=float)
        metrics[name] = _calibrate_stabilized_target(
            artifact,
            truth_all[eligible],
            validation_groups[eligible],
            intervals[:, 1],
            intervals[:, 0],
            intervals[:, 2],
            target_coverage=target_coverage,
            target_macro_lifecycle_coverage=target_macro_lifecycle_coverage,
            minimum_bucket_rows=minimum_bucket_rows,
            minimum_bucket_lifecycles=minimum_bucket_lifecycles,
        )
    return metrics


def evaluate_rul_targets_temporally(
    targets: dict[str, dict[str, Any]],
    X: np.ndarray,
    y_targets: dict[str, np.ndarray],
    groups: np.ndarray,
    timestamps: np.ndarray,
    indices: np.ndarray,
    *,
    minimum_bucket_lifecycles: int = 8,
    min_history_hours: float = 0.5,
    min_points: int = 6,
    max_upward_jump_floor_hours: float = 0.5,
    max_upward_jump_per_elapsed_hour: float = 1.5,
    upward_revision_cooldown_hours: float = 6.0,
    upward_revision_trigger_hours: float = 4.0,
    activation_scores: np.ndarray | None = None,
    activation_statuses: np.ndarray | None = None,
    minimum_activation_score: float | None = None,
    batch_labels: np.ndarray | None = None,
    activation_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate the frozen calibrated artifacts through the runtime temporal policy."""
    idx = np.asarray(indices, dtype=int)
    local_groups = groups[idx]
    active_contract = np.ones(len(idx), dtype=bool)
    if activation_mask is not None:
        supplied = np.asarray(activation_mask, dtype=bool)
        if len(supplied) != len(groups):
            raise ValueError("Forecastability activation mask must align with the full corpus")
        active_contract = supplied[idx]
    elif minimum_activation_score is not None:
        if activation_scores is None or activation_statuses is None:
            raise ValueError("Forecastability-aware evaluation requires activation scores and statuses")
        active_contract = (
            np.asarray(activation_scores[idx], dtype=float) >= float(minimum_activation_score)
        ) | (np.asarray(activation_statuses[idx], dtype=object) != "NORMAL")
    points = {
        name: np.asarray(artifact["model"].predict(X[idx]), dtype=float)
        for name, artifact in targets.items()
    }
    raw = {
        name: [_prediction_from_point(targets[name], value, apply_calibration=False) for value in values]
        for name, values in points.items()
    }
    emitted: dict[str, list[dict[str, Any] | None]] = {
        name: [None] * len(idx) for name in targets
    }
    runtime_eligible = np.zeros(len(idx), dtype=bool)
    for lifecycle in sorted(set(local_groups.tolist())):
        positions = np.flatnonzero(local_groups == lifecycle)
        positions = positions[np.argsort(np.asarray([timestamps[idx[pos]].timestamp() for pos in positions]))]
        started_at = timestamps[idx[positions[0]]]
        recent: deque[datetime] = deque()
        previous_warning_mid: float | None = None
        previous_critical_mid: float | None = None
        previous_ts: datetime | None = None
        last_revision_at: datetime | None = None
        for pos in positions:
            ts = timestamps[idx[pos]]
            recent.append(ts)
            cutoff = ts - timedelta(hours=12.0)
            while recent and recent[0] < cutoff:
                recent.popleft()
            history_hours = max(0.0, (ts - started_at).total_seconds() / 3600.0)
            eligible = len(recent) >= min_points and history_hours >= min_history_hours
            runtime_eligible[pos] = eligible
            if not eligible:
                previous_ts = ts
                continue
            warning_values = tuple(raw["warning"][pos][key] for key in ("lower", "median", "upper"))
            critical_values = tuple(raw["critical"][pos][key] for key in ("lower", "median", "upper"))
            elapsed_h = max(0.0, (ts - previous_ts).total_seconds() / 3600.0) if previous_ts else 0.0
            warning_values, critical_values, last_revision_at = _stabilize_prediction_pair_values(
                warning_values,
                critical_values,
                previous_warning_mid=previous_warning_mid,
                previous_critical_mid=previous_critical_mid,
                timestamp=ts,
                elapsed_h=elapsed_h,
                last_revision_at=last_revision_at,
                max_upward_jump_floor_hours=max_upward_jump_floor_hours,
                max_upward_jump_per_elapsed_hour=max_upward_jump_per_elapsed_hour,
                upward_revision_cooldown_hours=upward_revision_cooldown_hours,
                upward_revision_trigger_hours=upward_revision_trigger_hours,
            )
            emitted["warning"][pos] = _apply_v2_1_calibration(targets["warning"], warning_values)
            emitted["critical"][pos] = _apply_v2_1_calibration(targets["critical"], critical_values)
            previous_warning_mid = warning_values[1]
            previous_critical_mid = critical_values[1]
            previous_ts = ts

    results: dict[str, dict[str, Any]] = {}
    for name in targets:
        truth_all = y_targets[name][idx]
        truth_eligible = np.isfinite(truth_all) & (truth_all > 0.0)
        forecastability_by_horizon: dict[str, Any] = {}
        for bucket in RUL_DIAGNOSTIC_BUCKETS:
            bucket_mask = truth_eligible & np.asarray([
                np.isfinite(value) and diagnostic_horizon_bucket(value) == bucket
                for value in truth_all
            ])
            active_rows = int(np.sum(bucket_mask & active_contract))
            eligible_rows = int(np.sum(bucket_mask))
            forecastability_by_horizon[bucket] = {
                "eligible_rows": eligible_rows,
                "contract_active_rows": active_rows,
                "contract_withheld_rows": eligible_rows - active_rows,
                "contract_active_rate": float(active_rows / eligible_rows) if eligible_rows else None,
                "lifecycles": len(set(local_groups[bucket_mask].tolist())),
                "batches": (
                    len(set(np.asarray(batch_labels[idx], dtype=object)[bucket_mask].tolist()))
                    if batch_labels is not None else None
                ),
            }
        available = truth_eligible & runtime_eligible & active_contract & np.asarray(
            [row is not None for row in emitted[name]], dtype=bool
        )
        positions = np.flatnonzero(available)
        prediction = np.asarray([emitted[name][pos]["median"] for pos in positions], dtype=float)
        lower = np.asarray([emitted[name][pos]["lower"] for pos in positions], dtype=float)
        upper = np.asarray([emitted[name][pos]["upper"] for pos in positions], dtype=float)
        truth = truth_all[positions]
        local_available_groups = local_groups[positions]
        errors = prediction - truth
        interval = _interval_metrics(truth, lower, upper, local_available_groups) if len(positions) else {}
        lifecycle_mae = [
            float(np.mean(np.abs(errors[local_available_groups == lifecycle])))
            for lifecycle in sorted(set(local_available_groups.tolist()))
        ]
        horizon_diagnostics = horizon_point_diagnostics(
            truth, prediction, local_available_groups, lower=lower, upper=upper,
            minimum_lifecycles=minimum_bucket_lifecycles,
        ) if len(positions) else {}
        if batch_labels is not None:
            available_batches = np.asarray(batch_labels[idx], dtype=object)[positions]
            for bucket, row in horizon_diagnostics.items():
                bucket_mask = np.asarray([diagnostic_horizon_bucket(value) == bucket for value in truth])
                row["batches"] = len(set(available_batches[bucket_mask].tolist()))
        long_mask = truth > 48.0
        long_interval = (
            _interval_metrics(
                truth[long_mask], lower[long_mask], upper[long_mask], local_available_groups[long_mask]
            )
            if np.any(long_mask)
            else {}
        )
        long_lifecycles = len(set(local_available_groups[long_mask].tolist()))
        gt48_combined = {
            "rows": int(np.sum(long_mask)),
            "lifecycles": long_lifecycles,
            "batches": (
                len(set(np.asarray(batch_labels[idx], dtype=object)[positions][long_mask].tolist()))
                if batch_labels is not None else None
            ),
            "supported": long_lifecycles >= minimum_bucket_lifecycles,
            "mae_hours": float(np.mean(np.abs(errors[long_mask]))) if np.any(long_mask) else None,
            "mean_signed_error_hours": float(np.mean(errors[long_mask])) if np.any(long_mask) else None,
            "interval_coverage": long_interval.get("coverage"),
            "macro_lifecycle_coverage": long_interval.get("macro_lifecycle_coverage"),
            "mean_interval_width_hours": long_interval.get("mean_interval_width_hours"),
        }
        monotonic_good = monotonic_pairs = 0
        for lifecycle in sorted(set(local_available_groups.tolist())):
            lp = positions[local_available_groups == lifecycle]
            lp = lp[np.argsort(np.asarray([timestamps[idx[pos]].timestamp() for pos in lp]))]
            seq = [float(emitted[name][pos]["median"]) for pos in lp]
            for before, after in zip(seq, seq[1:]):
                monotonic_pairs += 1
                monotonic_good += int(after <= before + 1e-9)
        results[name] = {
            "eligible_rows": int(np.sum(truth_eligible)),
            "estimated_rows": int(len(positions)),
            "availability": float(len(positions) / np.sum(truth_eligible)) if np.sum(truth_eligible) else None,
            "forecastability_contract_active_rows": int(np.sum(truth_eligible & active_contract)),
            "forecastability_contract_withheld_rows": int(np.sum(truth_eligible & ~active_contract)),
            "availability_inside_active_region": (
                float(len(positions) / np.sum(truth_eligible & active_contract))
                if np.sum(truth_eligible & active_contract)
                else None
            ),
            "forecastability_contract_by_true_horizon": forecastability_by_horizon,
            "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
            "median_abs_error_hours": float(np.median(np.abs(errors))) if len(errors) else None,
            "mean_signed_error_hours": float(np.mean(errors)) if len(errors) else None,
            "median_signed_error_hours": float(np.median(errors)) if len(errors) else None,
            "macro_lifecycle_mae_hours": float(np.mean(lifecycle_mae)) if lifecycle_mae else None,
            "interval_coverage": interval.get("coverage"),
            "macro_lifecycle_coverage": interval.get("macro_lifecycle_coverage"),
            "mean_interval_width_hours": interval.get("mean_interval_width_hours"),
            "median_interval_width_hours": interval.get("median_interval_width_hours"),
            "monotonicity": float(monotonic_good / monotonic_pairs) if monotonic_pairs else None,
            "horizon_diagnostics": horizon_diagnostics,
            "gt48_combined_diagnostic": gt48_combined,
            "evaluation_stage": "causal_runtime_history_then_temporal_stabilization_then_frozen_calibration",
        }
    return results


def v2_4_support_signals(bank: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    """Return training-reference-only distance and target-dispersion signals."""
    values = np.asarray(rows, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    transformed = bank["scaler"].transform(bank["imputer"].transform(values))
    reference = np.asarray(bank["reference_X"], dtype=float)
    targets = np.asarray(bank["reference_targets"], dtype=float)
    batches = np.asarray(bank["reference_batches"], dtype=object)
    k = min(int(bank.get("neighbors", 10)), len(reference))
    output = np.empty((len(transformed), 4), dtype=float)
    neighbor_model = bank.get("neighbor_model")
    if neighbor_model is None:
        neighbor_model = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1).fit(reference)
    batch_distances, batch_neighbors = neighbor_model.kneighbors(transformed, n_neighbors=k)
    for pos, (distances, nearest) in enumerate(zip(batch_distances, batch_neighbors)):
        local_targets = targets[nearest]
        median = float(np.median(local_targets))
        output[pos] = (
            float(np.min(distances)),
            float(np.std(local_targets)),
            float(np.median(np.abs(local_targets - median))),
            float(len(set(batches[nearest].tolist())) / max(k, 1)),
        )
    return output


def v2_4_selector_matrix(
    X: np.ndarray,
    anchor_prediction: np.ndarray,
    alternate_prediction: np.ndarray,
    support: np.ndarray,
    base_indices: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(anchor_prediction, dtype=float)
    alternate = np.asarray(alternate_prediction, dtype=float)
    disagreement = np.abs(anchor - alternate)
    relative = disagreement / np.maximum(np.maximum(np.abs(anchor), np.abs(alternate)), 6.0)
    return np.column_stack([
        np.asarray(X, dtype=float)[:, np.asarray(base_indices, dtype=int)],
        anchor,
        disagreement,
        relative,
        np.asarray(support, dtype=float),
    ])


class VVB001LearnedRULEstimator:
    """Online learned RUL estimator using only causal runtime features.

    A nonlinear point regressor and train-OOF raw bounds are learned from completed synthetic
    lifecycle labels. Hierarchical conformal margins are calibrated only on lifecycle-disjoint
    validation data after causal temporal stabilization, then embedded in the same joblib bundle
    as the frozen regime model. This class never receives simulator truth at runtime.
    """

    def __init__(self, artifact: dict[str, Any]) -> None:
        version = str(artifact.get("version"))
        if version not in SUPPORTED_RUL_MODEL_VERSIONS:
            raise ValueError(f"Unsupported learned RUL artifact version: {artifact.get('version')!r}")
        self.artifact = artifact
        self.version = version
        self.method = (
            "supervised_learned_rul_v2_7_identity_first_critical"
            if version == RUL_MODEL_VERSION_V2_7
            else
            "supervised_learned_rul_v2_6_target_specific_corrected"
            if version == RUL_MODEL_VERSION_V2_6
            else "supervised_learned_rul_v2_5_active_population_calibrated"
            if version == RUL_MODEL_VERSION_V2_5
            else "supervised_learned_rul_v2_4_state_selective"
            if version == RUL_MODEL_VERSION_V2_4
            else "supervised_learned_rul_v2_3_forecastability_aware"
            if version == RUL_MODEL_VERSION_V2_3
            else "supervised_learned_rul_v2_2"
            if version == RUL_MODEL_VERSION_V2_2
            else "supervised_learned_rul_v2_1"
            if version == RUL_MODEL_VERSION_V2_1
            else "supervised_learned_rul_v2"
        )
        self.feature_names = validate_rul_feature_names(artifact["feature_names"])
        self.base_feature_names = list(artifact["base_feature_names"])
        self.targets = dict(artifact["targets"])
        critical_calibration = dict(self.targets.get("critical", {}).get("calibration") or {})
        self.calibration_method = str(critical_calibration.get("method") or "") or None
        self.min_history_hours = float(artifact.get("min_history_hours", 0.5))
        self.min_points = int(artifact.get("min_points", 6))
        self.max_upward_jump_floor_hours = float(artifact.get("max_upward_jump_floor_hours", 0.5))
        self.max_upward_jump_per_elapsed_hour = float(artifact.get("max_upward_jump_per_elapsed_hour", 1.5))
        self._history: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        self._started_at: dict[str, datetime] = {}
        self._last_prediction: dict[str, RULPrediction] = {}
        self._last_timestamp: dict[str, datetime] = {}
        self._last_upward_revision_at: dict[str, datetime] = {}
        self.upward_revision_cooldown_hours = float(artifact.get("upward_revision_cooldown_hours", 6.0))
        self.upward_revision_trigger_hours = float(artifact.get("upward_revision_trigger_hours", 4.0))
        forecastability = dict(artifact.get("forecastability_contract") or {})
        self.forecastability_aware = version in {
            RUL_MODEL_VERSION_V2_3, RUL_MODEL_VERSION_V2_4, RUL_MODEL_VERSION_V2_5,
            RUL_MODEL_VERSION_V2_6, RUL_MODEL_VERSION_V2_7,
        }
        self.activation_score_threshold = float(
            forecastability.get("minimum_degradation_score", 0.0)
        )
        self.low_confidence_score_threshold = float(
            forecastability.get("low_confidence_score")
            if version in {RUL_MODEL_VERSION_V2_4, RUL_MODEL_VERSION_V2_5, RUL_MODEL_VERSION_V2_6, RUL_MODEL_VERSION_V2_7}
            and forecastability.get("low_confidence_score") is not None
            else forecastability.get(
                "low_confidence_degradation_score",
                max(0.0, 0.5 * self.activation_score_threshold),
            )
        )
        self.v2_4_selector = dict(forecastability.get("selector") or {})
        if version in {RUL_MODEL_VERSION_V2_4, RUL_MODEL_VERSION_V2_5, RUL_MODEL_VERSION_V2_6, RUL_MODEL_VERSION_V2_7} and self.v2_4_selector:
            support_bank = self.v2_4_selector["support_bank"]
            reference = np.asarray(support_bank["reference_X"], dtype=float)
            neighbors = min(int(support_bank.get("neighbors", 10)), len(reference))
            support_bank["neighbor_model"] = NearestNeighbors(
                n_neighbors=neighbors, algorithm="auto", n_jobs=1
            ).fit(reference)
        self.v2_4_feature_builder = (
            V24CausalFeatureBuilder()
            if version in {RUL_MODEL_VERSION_V2_4, RUL_MODEL_VERSION_V2_5, RUL_MODEL_VERSION_V2_6, RUL_MODEL_VERSION_V2_7}
            else None
        )
        self.v2_4_activation_threshold = float(forecastability.get("activation_threshold", 1.0))
        self.v2_4_deactivation_threshold = float(
            forecastability.get("deactivation_threshold", self.v2_4_activation_threshold)
        )
        self._v2_4_selector_active: dict[str, bool] = {}
        self._v2_5_serviceability: dict[str, dict[str, int | bool]] = defaultdict(dict)
        self.v2_6_target_contracts = dict(artifact.get("target_contracts") or {})
        self._v2_6_serviceability: dict[str, dict[str, dict[str, int | str]]] = {
            target: defaultdict(dict) for target in ("warning", "critical")
        }
        self.v2_5_minimum_confirmations = int(forecastability.get("minimum_active_confirmations", 3))
        self.v2_5_minimum_active_dwell = int(forecastability.get("minimum_active_dwell", 3))
        self.v2_5_maximum_exact_rul_horizon = float(
            forecastability.get("maximum_exact_rul_horizon_hours", math.inf)
        )

    def reset_machine(self, machine_key: str) -> None:
        self._history.pop(machine_key, None)
        self._started_at.pop(machine_key, None)
        self._last_prediction.pop(machine_key, None)
        self._last_timestamp.pop(machine_key, None)
        self._last_upward_revision_at.pop(machine_key, None)
        self._v2_4_selector_active.pop(machine_key, None)
        self._v2_5_serviceability.pop(machine_key, None)
        for target_states in self._v2_6_serviceability.values():
            target_states.pop(machine_key, None)
        if self.v2_4_feature_builder is not None:
            self.v2_4_feature_builder.reset_machine(machine_key)

    def _unavailable(
        self,
        machine_key: str,
        reason: str,
        *,
        history_hours: float,
        points: int,
        predicted_status: str = "NORMAL",
        reason_code: str | None = None,
        reasons: Iterable[str] = (),
        serviceable_intent: bool = False,
        hard_eligible: bool = False,
        selector_active: bool = False,
        forecastability_score: float | None = None,
        support_distance: float | None = None,
        neighbor_dispersion_hours: float | None = None,
        model_disagreement_hours: float | None = None,
    ) -> RULPrediction:
        warning_observed = str(predicted_status).upper() == "WARNING"
        structured = tuple(dict.fromkeys(str(value) for value in reasons if value))
        result = RULPrediction(
            estimated_hours_to_warning=0.0 if warning_observed else None,
            warning_lower_hours=0.0 if warning_observed else None,
            warning_upper_hours=0.0 if warning_observed else None,
            estimated_hours_to_critical=None,
            critical_lower_hours=None,
            critical_upper_hours=None,
            reliability="UNAVAILABLE",
            reason=reason,
            trend_score_per_hour=None,
            trend_r2=None,
            history_hours=float(history_hours),
            trusted_points=int(points),
            state_source=(
                "LEARNED_RUL_V2_7_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_7
                else "LEARNED_RUL_V2_6_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_6
                else "LEARNED_RUL_V2_5_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_5
                else "LEARNED_RUL_V2_4_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_4
                else "LEARNED_RUL_V2_3_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_3
                else "LEARNED_RUL_V2_2_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_2
                else "LEARNED_RUL_V2_1_UNAVAILABLE"
                if self.version == RUL_MODEL_VERSION_V2_1
                else "LEARNED_RUL_V2_UNAVAILABLE"
            ),
            method=self.method,
            calibration_method=self.calibration_method,
            forecastability_state="RUL_UNAVAILABLE",
            forecastability_score=forecastability_score,
            serviceable_intent=serviceable_intent,
            hard_eligible=hard_eligible,
            selector_active=selector_active,
            withholding_reason_code=reason_code,
            withholding_reasons=structured or ((reason_code,) if reason_code else ()),
            support_distance=support_distance,
            neighbor_dispersion_hours=neighbor_dispersion_hours,
            model_disagreement_hours=model_disagreement_hours,
        )
        self._last_prediction[machine_key] = result
        return result

    def _withheld_forecast(
        self,
        machine_key: str,
        *,
        score: float,
        history_hours: float,
        points: int,
        predicted_status: str = "NORMAL",
        reason_code: str | None = None,
        forecastability_score: float | None = None,
        forced_state: str | None = None,
        hard_eligible: bool = False,
        selector_active: bool = False,
        withholding_reasons: Iterable[str] = (),
        support_distance: float | None = None,
        neighbor_dispersion_hours: float | None = None,
        model_disagreement_hours: float | None = None,
    ) -> RULPrediction:
        state = forced_state or (
            "RUL_LOW_CONFIDENCE"
            if score >= self.low_confidence_score_threshold
            else "RUL_UNAVAILABLE_NO_DEGRADATION_EVIDENCE"
        )
        if self.version == RUL_MODEL_VERSION_V2_5 and state == "RUL_UNAVAILABLE_NO_DEGRADATION_EVIDENCE":
            state = "RUL_UNAVAILABLE"
        warning_observed = str(predicted_status).upper() == "WARNING"
        reason = reason_code or (
            "exact_rul_withheld_weak_degradation_evidence"
            if state == "RUL_LOW_CONFIDENCE"
            else "exact_rul_unidentifiable_without_degradation_evidence"
        )
        result = RULPrediction(
            estimated_hours_to_warning=0.0 if warning_observed else None,
            warning_lower_hours=0.0 if warning_observed else None,
            warning_upper_hours=0.0 if warning_observed else None,
            estimated_hours_to_critical=None,
            critical_lower_hours=None,
            critical_upper_hours=None,
            reliability="LOW" if state == "RUL_LOW_CONFIDENCE" else "UNAVAILABLE",
            reason=(
                f"current_warning_immediate_future_critical_withheld:{reason}"
                if warning_observed else reason
            ),
            trend_score_per_hour=None,
            trend_r2=None,
            history_hours=float(history_hours),
            trusted_points=int(points),
            state_source=state,
            method=self.method,
            calibration_method=self.calibration_method,
            calibration_bucket=None,
            forecastability_state=state,
            forecastability_score=forecastability_score,
            serviceable_intent=False,
            hard_eligible=hard_eligible,
            selector_active=selector_active,
            withholding_reason_code=reason_code,
            withholding_reasons=tuple(dict.fromkeys(
                str(value) for value in withholding_reasons if value
            )) or ((reason_code,) if reason_code else ()),
            support_distance=support_distance,
            neighbor_dispersion_hours=neighbor_dispersion_hours,
            model_disagreement_hours=model_disagreement_hours,
        )
        self._last_prediction[machine_key] = result
        return result

    @staticmethod
    def _hold(previous: RULPrediction) -> RULPrediction:
        return RULPrediction(
            **{
                **previous.__dict__,
                "reason": "sensor_quality_quarantine_held_last_trusted_rul",
                "state_source": "HELD_LAST_TRUSTED_RUL",
            }
        )

    def _row(
        self,
        features: dict[str, Any],
        dynamic: dict[str, float],
        v2_4_features: dict[str, float] | None = None,
    ) -> np.ndarray:
        merged = dict(features)
        merged.update(dynamic)
        merged.update(v2_4_features or {})
        values: list[float] = []
        for name in self.feature_names:
            value = _finite(merged.get(name))
            values.append(value if value is not None else np.nan)
        return np.asarray(values, dtype=float)

    def _v2_4_forecastability_details(self, row: np.ndarray) -> dict[str, float]:
        selector = self.v2_4_selector
        anchor_row = row[np.asarray(selector["anchor_feature_indices"], dtype=int)]
        alternate_row = row[np.asarray(selector["alternate_feature_indices"], dtype=int)]
        support_row = row[np.asarray(selector["support_feature_indices"], dtype=int)]
        anchor = float(selector["anchor_model"].predict(anchor_row.reshape(1, -1))[0])
        alternate = float(selector["alternate_model"].predict(alternate_row.reshape(1, -1))[0])
        support = v2_4_support_signals(selector["support_bank"], support_row.reshape(1, -1))
        matrix = v2_4_selector_matrix(
            row.reshape(1, -1),
            np.asarray([anchor]),
            np.asarray([alternate]),
            support,
            np.asarray(selector["base_feature_indices"], dtype=int),
        )
        probabilities = selector["model"].predict_proba(matrix)
        classes = list(selector["model"].classes_)
        return {
            "score": float(probabilities[0, classes.index(1)]) if 1 in classes else 0.0,
            "anchor_prediction_hours": anchor,
            "support_distance": float(support[0, 0]),
            "neighbor_dispersion_hours": float(support[0, 2]),
            "model_disagreement_hours": abs(anchor - alternate),
        }

    def _v2_4_forecastability_score(self, row: np.ndarray) -> float:
        return self._v2_4_forecastability_details(row)["score"]

    def _v2_6_target_selector_score(
        self, target: str, details: dict[str, float], degradation_score: float,
    ) -> float:
        contract = dict(self.v2_6_target_contracts.get(target) or {})
        model = contract.get("selector_model")
        values = np.asarray([[
            details["score"], details["anchor_prediction_hours"],
            details["support_distance"], details["neighbor_dispersion_hours"],
            details["model_disagreement_hours"], float(degradation_score),
        ]], dtype=float)
        if model is not None:
            probability = model.predict_proba(values)
            classes = list(model.classes_)
            return float(probability[0, classes.index(1)]) if 1 in classes else 0.0
        scale = float(contract.get("selector_score_scale", 1.0))
        offset = float(contract.get("selector_score_offset", 0.0))
        return float(np.clip(scale * details["score"] + offset, 0.0, 1.0))

    def _v2_5_hard_withhold(
        self,
        machine_key: str,
        *,
        reason_code: str,
        reason: str,
        history_hours: float,
        points: int,
        predicted_status: str,
    ) -> RULPrediction:
        resolution = resolve_v2_5_serviceability(
            self._v2_5_serviceability[machine_key],
            hard_eligible=False,
            hard_reasons=(reason_code,),
            selector_score=None,
            activation_threshold=self.v2_4_activation_threshold,
            deactivation_threshold=self.v2_4_deactivation_threshold,
            minimum_confirmations=self.v2_5_minimum_confirmations,
            minimum_active_dwell=self.v2_5_minimum_active_dwell,
        )
        return self._unavailable(
            machine_key,
            reason,
            history_hours=history_hours,
            points=points,
            predicted_status=predicted_status,
            reason_code=reason_code,
            reasons=resolution["reasons"],
        )

    def _v2_6_hard_withhold(
        self,
        machine_key: str,
        *,
        reason_code: str,
        reason: str,
        history_hours: float,
        points: int,
        predicted_status: str,
    ) -> RULPrediction:
        resolutions = {
            target: resolve_v2_6_target_serviceability(
                self._v2_6_serviceability[target][machine_key],
                hard_eligible=False,
                hard_reasons=(reason_code,),
                selector_score=None,
                activation_threshold=1.0,
                deactivation_threshold=1.0,
            )
            for target in ("warning", "critical")
        }
        result = self._unavailable(
            machine_key, reason, history_hours=history_hours, points=points,
            predicted_status=predicted_status, reason_code=reason_code, reasons=(reason_code,),
        )
        enriched = RULPrediction(**{
            **result.__dict__,
            **{
                f"{target}_forecastability_state": resolutions[target]["state"]
                for target in ("warning", "critical")
            },
            **{
                f"{target}_withholding_reason_code": reason_code
                for target in ("warning", "critical")
            },
            **{
                f"{target}_withholding_reasons": (reason_code,)
                for target in ("warning", "critical")
            },
        })
        self._last_prediction[machine_key] = enriched
        return enriched

    def _versioned_hard_withhold(self, machine_key: str, **kwargs: Any) -> RULPrediction:
        if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS:
            return self._v2_6_hard_withhold(machine_key, **kwargs)
        return self._v2_5_hard_withhold(machine_key, **kwargs)

    def _stabilize(
        self,
        machine_key: str,
        timestamp: datetime,
        previous: RULPrediction | None,
        *,
        elapsed_h: float,
        warning: tuple[float, float, float],
        critical: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        last_revision = self._last_upward_revision_at.get(machine_key)
        stabilized_warning, stabilized_critical, revised_at = _stabilize_prediction_pair_values(
            warning,
            critical,
            previous_warning_mid=previous.estimated_hours_to_warning if previous else None,
            previous_critical_mid=previous.estimated_hours_to_critical if previous else None,
            timestamp=timestamp,
            elapsed_h=elapsed_h,
            last_revision_at=last_revision,
            max_upward_jump_floor_hours=self.max_upward_jump_floor_hours,
            max_upward_jump_per_elapsed_hour=self.max_upward_jump_per_elapsed_hour,
            upward_revision_cooldown_hours=self.upward_revision_cooldown_hours,
            upward_revision_trigger_hours=self.upward_revision_trigger_hours,
        )
        if revised_at is not None and revised_at != last_revision:
            self._last_upward_revision_at[machine_key] = revised_at
        return stabilized_warning, stabilized_critical

    def update(
        self,
        *,
        machine_key: str,
        timestamp: datetime,
        features: dict[str, Any],
        degradation_score: float,
        predicted_status: str,
        trusted: bool = True,
        state_reset: bool = False,
        raw_safety_override: bool = False,
    ) -> RULPrediction:
        if state_reset:
            self.reset_machine(machine_key)

        if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS} and str(predicted_status).upper() == "CRITICAL":
            result = RULPrediction(
                estimated_hours_to_warning=0.0,
                warning_lower_hours=0.0,
                warning_upper_hours=0.0,
                estimated_hours_to_critical=0.0,
                critical_lower_hours=0.0,
                critical_upper_hours=0.0,
                reliability="SAFETY_OVERRIDE" if raw_safety_override else "CURRENTLY_CRITICAL",
                reason="raw_safety_override_currently_critical" if raw_safety_override else "current_status_is_critical",
                trend_score_per_hour=None,
                trend_r2=None,
                history_hours=0.0,
                trusted_points=0,
                state_source="RAW_SAFETY_OVERRIDE" if raw_safety_override else "CURRENT_STATUS",
                method=self.method,
                calibration_method=self.calibration_method,
                calibration_bucket="current_critical_override",
                forecastability_state="RUL_ACTIVE",
                serviceable_intent=True,
                hard_eligible=True,
                selector_active=True,
                warning_forecastability_state="RUL_ACTIVE",
                warning_serviceable_intent=True,
                warning_hard_eligible=True,
                warning_selector_active=True,
                warning_raw_point_hours=0.0,
                warning_corrected_point_hours=0.0,
                warning_calibration_stratum="current_critical_override",
                critical_forecastability_state="RUL_ACTIVE",
                critical_serviceable_intent=True,
                critical_hard_eligible=True,
                critical_selector_active=True,
                critical_raw_point_hours=0.0,
                critical_corrected_point_hours=0.0,
                critical_calibration_stratum="current_critical_override",
            )
            self._last_prediction[machine_key] = result
            return result

        if not trusted:
            if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}:
                return self._versioned_hard_withhold(
                    machine_key,
                    reason_code="ANOMALY_WITHHOLD",
                    reason="sensor_quality_quarantine_exact_rul_cleared",
                    history_hours=0.0,
                    points=0,
                    predicted_status=predicted_status,
                )
            previous = self._last_prediction.get(machine_key)
            if previous is None:
                return self._unavailable(
                    machine_key,
                    "sensor_quality_quarantine_no_trusted_rul_yet",
                    history_hours=0.0,
                    points=0,
                )
            return self._hold(previous)

        score = _finite(degradation_score)
        if score is None:
            if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}:
                return self._versioned_hard_withhold(
                    machine_key,
                    reason_code="INVALID_RUNTIME_STATE",
                    reason="non_finite_degradation_score",
                    history_hours=0.0,
                    points=0,
                    predicted_status=predicted_status,
                )
            return self._unavailable(machine_key, "non_finite_degradation_score", history_hours=0.0, points=0)

        previous_ts = self._last_timestamp.get(machine_key)
        if previous_ts is not None and timestamp < previous_ts:
            history = self._history[machine_key]
            history_hours = (history[-1][0] - history[0][0]).total_seconds() / 3600.0 if len(history) > 1 else 0.0
            if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}:
                return self._versioned_hard_withhold(
                    machine_key,
                    reason_code="INVALID_RUNTIME_STATE",
                    reason="out_of_order_timestamp_not_used_for_rul",
                    history_hours=history_hours,
                    points=len(history),
                    predicted_status=predicted_status,
                )
            return self._unavailable(
                machine_key,
                "out_of_order_timestamp_not_used_for_rul",
                history_hours=history_hours,
                points=len(history),
            )

        started_at = self._started_at.setdefault(machine_key, timestamp)
        history = self._history[machine_key]
        if history and history[-1][0] == timestamp:
            history[-1] = (timestamp, score)
        else:
            history.append((timestamp, score))
        cutoff = timestamp - timedelta(hours=12.0)
        while history and history[0][0] < cutoff:
            history.popleft()
        self._last_timestamp[machine_key] = timestamp
        history_hours = max(0.0, (timestamp - started_at).total_seconds() / 3600.0)
        points = len(history)

        if str(predicted_status).upper() == "CRITICAL":
            result = RULPrediction(
                estimated_hours_to_warning=0.0,
                warning_lower_hours=0.0,
                warning_upper_hours=0.0,
                estimated_hours_to_critical=0.0,
                critical_lower_hours=0.0,
                critical_upper_hours=0.0,
                reliability="SAFETY_OVERRIDE" if raw_safety_override else "CURRENTLY_CRITICAL",
                reason="raw_safety_override_currently_critical" if raw_safety_override else "current_status_is_critical",
                trend_score_per_hour=None,
                trend_r2=None,
                history_hours=history_hours,
                trusted_points=points,
                state_source="RAW_SAFETY_OVERRIDE" if raw_safety_override else "CURRENT_STATUS",
                method=self.method,
                calibration_method=self.calibration_method,
                calibration_bucket="current_critical_override",
                forecastability_state="RUL_ACTIVE",
            )
            self._last_prediction[machine_key] = result
            return result

        v2_4_features: dict[str, float] = {}
        if self.v2_4_feature_builder is not None:
            try:
                v2_4_features = self.v2_4_feature_builder.update(
                    machine_key,
                    timestamp,
                    sensor_values_from_runtime_features(features),
                    state_reset=False,
                )
            except ValueError:
                if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}:
                    return self._versioned_hard_withhold(
                        machine_key,
                        reason_code="MISSING_REQUIRED_FEATURE",
                        reason="missing_runtime_sensor_values_for_v2_5_forecastability",
                        history_hours=history_hours,
                        points=points,
                        predicted_status=predicted_status,
                    )
                return self._unavailable(
                    machine_key,
                    "missing_runtime_sensor_values_for_v2_4_forecastability",
                    history_hours=history_hours,
                    points=points,
                )

        if points < self.min_points or history_hours < self.min_history_hours:
            if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}:
                return self._versioned_hard_withhold(
                    machine_key,
                    reason_code="INSUFFICIENT_HISTORY",
                    reason="insufficient_causal_history_for_learned_rul",
                    history_hours=history_hours,
                    points=points,
                    predicted_status=predicted_status,
                )
            return self._unavailable(
                machine_key,
                "insufficient_causal_history_for_learned_rul",
                history_hours=history_hours,
                points=points,
            )

        if (
            self.version == RUL_MODEL_VERSION_V2_3
            and str(predicted_status).upper() == "NORMAL"
            and score < self.activation_score_threshold
        ):
            return self._withheld_forecast(
                machine_key,
                score=score,
                history_hours=history_hours,
                points=points,
                predicted_status=predicted_status,
            )

        dynamic = build_dynamic_rul_features(
            history,
            timestamp=timestamp,
            lifecycle_started_at=started_at,
            degradation_score=score,
            predicted_status=predicted_status,
        )
        row = self._row(features, dynamic, v2_4_features)
        forecastability_score: float | None = None
        forecastability_details: dict[str, float] = {}
        serviceable_intent = False
        selector_active_for_output = False
        target_resolutions: dict[str, dict[str, Any]] = {}
        target_scores: dict[str, float] = {}
        target_raw_points: dict[str, float] = {}
        if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS:
            forecastability_details = self._v2_4_forecastability_details(row)
            for target in ("warning", "critical"):
                contract = dict(self.v2_6_target_contracts.get(target) or {})
                raw_point = float(self.targets[target]["model"].predict(row.reshape(1, -1))[0])
                target_raw_points[target] = raw_point
                target_scores[target] = self._v2_6_target_selector_score(
                    target, forecastability_details, score,
                )
                maximum_horizon = float(contract.get(
                    "maximum_exact_rul_horizon_hours", 24.0 if target == "warning" else 48.0,
                ))
                inside_horizon = math.isfinite(raw_point) and raw_point <= maximum_horizon
                target_resolutions[target] = resolve_v2_6_target_serviceability(
                    self._v2_6_serviceability[target][machine_key],
                    hard_eligible=inside_horizon,
                    hard_reasons=(() if inside_horizon else ("OUT_OF_SUPPORT",)),
                    selector_score=target_scores[target],
                    activation_threshold=float(contract.get("activation_threshold", self.v2_4_activation_threshold)),
                    deactivation_threshold=float(contract.get("deactivation_threshold", self.v2_4_deactivation_threshold)),
                    confirmations=int(contract.get("confirmations", self.v2_5_minimum_confirmations)),
                )
            forecastability_score = target_scores["critical"]
            serviceable_intent = bool(target_resolutions["critical"]["serviceable_intent"])
            selector_active_for_output = bool(target_resolutions["critical"]["selector_active"])
            if not any(value["serviceable_intent"] for value in target_resolutions.values()):
                aggregate_state = (
                    "RUL_LOW_CONFIDENCE"
                    if any(value["state"] == "RUL_LOW_CONFIDENCE" for value in target_resolutions.values())
                    else "RUL_UNAVAILABLE"
                )
                base = self._withheld_forecast(
                    machine_key, score=score, history_hours=history_hours, points=points,
                    predicted_status=predicted_status,
                    reason_code=str(target_resolutions["critical"]["reason_code"]),
                    forecastability_score=forecastability_score, forced_state=aggregate_state,
                    hard_eligible=bool(target_resolutions["critical"]["state"] != "RUL_UNAVAILABLE"),
                    selector_active=selector_active_for_output,
                    withholding_reasons=target_resolutions["critical"]["reasons"],
                    support_distance=forecastability_details["support_distance"],
                    neighbor_dispersion_hours=forecastability_details["neighbor_dispersion_hours"],
                    model_disagreement_hours=forecastability_details["model_disagreement_hours"],
                )
                enriched = RULPrediction(**{
                    **base.__dict__,
                    **{f"{target}_forecastability_state": target_resolutions[target]["state"] for target in ("warning", "critical")},
                    **{f"{target}_forecastability_score": target_scores[target] for target in ("warning", "critical")},
                    **{f"{target}_serviceable_intent": False for target in ("warning", "critical")},
                    **{f"{target}_hard_eligible": target_resolutions[target]["reason_code"] != "OUT_OF_SUPPORT" for target in ("warning", "critical")},
                    **{f"{target}_selector_active": target_resolutions[target]["selector_active"] for target in ("warning", "critical")},
                    **{f"{target}_withholding_reason_code": target_resolutions[target]["reason_code"] for target in ("warning", "critical")},
                    **{f"{target}_withholding_reasons": target_resolutions[target]["reasons"] for target in ("warning", "critical")},
                    **{f"{target}_raw_point_hours": target_raw_points[target] for target in ("warning", "critical")},
                })
                self._last_prediction[machine_key] = enriched
                return enriched
        elif self.version == RUL_MODEL_VERSION_V2_5:
            forecastability_details = self._v2_4_forecastability_details(row)
            forecastability_score = forecastability_details["score"]
            inside_horizon_contract = (
                math.isfinite(forecastability_details["anchor_prediction_hours"])
                and forecastability_details["anchor_prediction_hours"]
                <= self.v2_5_maximum_exact_rul_horizon
            )
            resolution = resolve_v2_5_serviceability(
                self._v2_5_serviceability[machine_key],
                hard_eligible=inside_horizon_contract,
                hard_reasons=(() if inside_horizon_contract else ("OUT_OF_SUPPORT",)),
                selector_score=forecastability_score,
                activation_threshold=self.v2_4_activation_threshold,
                deactivation_threshold=self.v2_4_deactivation_threshold,
                minimum_confirmations=self.v2_5_minimum_confirmations,
                minimum_active_dwell=self.v2_5_minimum_active_dwell,
            )
            serviceable_intent = bool(resolution["serviceable_intent"])
            selector_active_for_output = bool(resolution["selector_active"])
            if not serviceable_intent:
                return self._withheld_forecast(
                    machine_key,
                    score=forecastability_score,
                    history_hours=history_hours,
                    points=points,
                    predicted_status=predicted_status,
                    reason_code=str(resolution["reason_code"]),
                    forecastability_score=forecastability_score,
                    forced_state=(
                        "RUL_LOW_CONFIDENCE"
                        if resolution["reason_code"] in {
                            "HYSTERESIS_NOT_CONFIRMED", "SELECTOR_BELOW_THRESHOLD",
                        }
                        and forecastability_score >= self.low_confidence_score_threshold
                        else "RUL_UNAVAILABLE"
                    ),
                    hard_eligible=inside_horizon_contract,
                    selector_active=selector_active_for_output,
                    withholding_reasons=resolution["reasons"],
                    support_distance=forecastability_details["support_distance"],
                    neighbor_dispersion_hours=forecastability_details["neighbor_dispersion_hours"],
                    model_disagreement_hours=forecastability_details["model_disagreement_hours"],
                )
        elif self.version == RUL_MODEL_VERSION_V2_4:
            forecastability_score = self._v2_4_forecastability_score(row)
            was_active = bool(self._v2_4_selector_active.get(machine_key, False))
            required = (
                self.v2_4_deactivation_threshold if was_active
                else self.v2_4_activation_threshold
            )
            selector_active = forecastability_score >= required
            self._v2_4_selector_active[machine_key] = selector_active
            if not selector_active:
                return self._withheld_forecast(
                    machine_key,
                    score=forecastability_score,
                    history_hours=history_hours,
                    points=points,
                    predicted_status=predicted_status,
                    reason_code=(
                        "exact_rul_withheld_out_of_supported_forecastability_state"
                        if forecastability_score >= self.low_confidence_score_threshold
                        else "exact_rul_withheld_no_reliable_degradation_evidence"
                    ),
                    forecastability_score=forecastability_score,
                )
        post_stabilization_calibration = self.calibration_method in {
            RUL_CALIBRATION_METHOD_V2_1, RUL_CALIBRATION_METHOD_V2_5,
            RUL_CALIBRATION_METHOD_V2_6,
        }
        target_corrected_points: dict[str, float] = {}
        target_bias_strata: dict[str, str] = {}
        if post_stabilization_calibration:
            if not target_raw_points:
                target_raw_points = {
                    target: float(self.targets[target]["model"].predict(row.reshape(1, -1))[0])
                    for target in ("warning", "critical")
                }
            for target in ("warning", "critical"):
                if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS:
                    target_corrected_points[target], target_bias_strata[target] = apply_v2_6_bias_correction(
                        target_raw_points[target],
                        dict(self.v2_6_target_contracts.get(target) or {}).get("bias_correction"),
                    )
                else:
                    target_corrected_points[target] = target_raw_points[target]
                    target_bias_strata[target] = "NONE"
            warning_details = _prediction_from_point(
                self.targets["warning"],
                target_corrected_points["warning"],
                apply_calibration=False,
            )
            critical_details = _prediction_from_point(
                self.targets["critical"],
                target_corrected_points["critical"],
                apply_calibration=False,
            )
        else:
            warning_details = predict_quantiles_with_metadata(self.targets["warning"], row)
            critical_details = predict_quantiles_with_metadata(self.targets["critical"], row)
        warning = (warning_details["lower"], warning_details["median"], warning_details["upper"])
        critical = (critical_details["lower"], critical_details["median"], critical_details["upper"])
        previous = self._last_prediction.get(machine_key)
        elapsed_h = max(0.0, (timestamp - previous_ts).total_seconds() / 3600.0) if previous_ts is not None else 0.0
        warning, critical = self._stabilize(
            machine_key, timestamp, previous, elapsed_h=elapsed_h, warning=warning, critical=critical
        )
        if post_stabilization_calibration:
            calibration_context = {
                "selector_score": forecastability_score,
                "support_distance": forecastability_details.get("support_distance"),
                "neighbor_dispersion_hours": forecastability_details.get("neighbor_dispersion_hours"),
                "model_disagreement_hours": forecastability_details.get("model_disagreement_hours"),
            }
            warning_details = _apply_v2_1_calibration(
                self.targets["warning"], warning, calibration_context=calibration_context,
            )
            critical_details = _apply_v2_1_calibration(
                self.targets["critical"], critical, calibration_context=calibration_context,
            )
            warning = (warning_details["lower"], warning_details["median"], warning_details["upper"])
            critical = (critical_details["lower"], critical_details["median"], critical_details["upper"])

        warning_lo, warning_mid, warning_hi = warning
        critical_lo, critical_mid, critical_hi = critical
        if str(predicted_status).upper() == "WARNING":
            warning_lo = warning_mid = warning_hi = 0.0

        # Preserve logical ordering: time to WARNING cannot exceed time to CRITICAL.
        if warning_mid > critical_mid:
            warning_mid = critical_mid
            warning_lo = min(warning_lo, warning_mid)
            warning_hi = min(warning_hi, critical_hi)
        warning_lo, warning_mid, warning_hi = sorted([warning_lo, warning_mid, warning_hi])
        critical_lo, critical_mid, critical_hi = sorted([critical_lo, critical_mid, critical_hi])

        interval_width = critical_hi - critical_lo
        if not all(math.isfinite(x) for x in (warning_lo, warning_mid, warning_hi, critical_lo, critical_mid, critical_hi)):
            if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}:
                return self._unavailable(
                    machine_key,
                    "learned_rul_non_finite_prediction",
                    history_hours=history_hours,
                    points=points,
                    predicted_status=predicted_status,
                    reason_code="INVALID_RUNTIME_STATE",
                    reasons=("INVALID_RUNTIME_STATE",),
                    serviceable_intent=True,
                    hard_eligible=True,
                    selector_active=selector_active_for_output,
                    forecastability_score=forecastability_score,
                    support_distance=forecastability_details.get("support_distance"),
                    neighbor_dispersion_hours=forecastability_details.get("neighbor_dispersion_hours"),
                    model_disagreement_hours=forecastability_details.get("model_disagreement_hours"),
                )
            return self._unavailable(
                machine_key,
                "learned_rul_non_finite_prediction",
                history_hours=history_hours,
                points=points,
            )

        if interval_width <= 24.0 and history_hours >= 3.0:
            reliability = "HIGH"
        elif interval_width <= 72.0:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"

        warning_emit = True
        critical_emit = True
        if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS:
            warning_emit = bool(target_resolutions["warning"]["serviceable_intent"])
            critical_emit = bool(target_resolutions["critical"]["serviceable_intent"])
            if str(predicted_status).upper() == "WARNING":
                warning_emit = True
                target_resolutions["warning"] = {
                    **target_resolutions["warning"], "state": "RUL_ACTIVE",
                    "serviceable_intent": True, "reason_code": None, "reasons": (),
                }
            reliability = reliability if critical_emit else ("MEDIUM" if warning_emit else "LOW")

        result = RULPrediction(
            estimated_hours_to_warning=float(warning_mid) if warning_emit else None,
            warning_lower_hours=float(warning_lo) if warning_emit else None,
            warning_upper_hours=float(warning_hi) if warning_emit else None,
            estimated_hours_to_critical=float(critical_mid) if critical_emit else None,
            critical_lower_hours=float(critical_lo) if critical_emit else None,
            critical_upper_hours=float(critical_hi) if critical_emit else None,
            reliability=reliability,
            reason="learned_quantile_rul_synthetic_only_not_plant_validated",
            trend_score_per_hour=_finite(dynamic["score_rate_6h"]),
            trend_r2=None,
            history_hours=history_hours,
            trusted_points=points,
            state_source=(
                "LEARNED_RUL_V2_7_IDENTITY_FIRST_CRITICAL"
                if self.version == RUL_MODEL_VERSION_V2_7
                else "LEARNED_RUL_V2_6_TARGET_SPECIFIC"
                if self.version == RUL_MODEL_VERSION_V2_6
                else "LEARNED_RUL_V2_5_ACTIVE"
                if self.version == RUL_MODEL_VERSION_V2_5
                else "LEARNED_RUL_V2_4_ACTIVE"
                if self.version == RUL_MODEL_VERSION_V2_4
                else "LEARNED_RUL_V2_3_ACTIVE"
                if self.version == RUL_MODEL_VERSION_V2_3
                else "LEARNED_RUL_V2_2"
                if self.version == RUL_MODEL_VERSION_V2_2
                else "LEARNED_RUL_V2_1"
                if self.version == RUL_MODEL_VERSION_V2_1
                else "LEARNED_RUL_V2"
            ),
            method=self.method,
            calibration_method=self.calibration_method,
            calibration_bucket=critical_details.get("bucket"),
            forecastability_state=(
                "RUL_ACTIVE" if warning_emit or critical_emit else "RUL_UNAVAILABLE"
            ),
            forecastability_score=forecastability_score,
            serviceable_intent=(
                serviceable_intent
                if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}
                else True
            ),
            hard_eligible=(self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}),
            selector_active=(
                selector_active_for_output
                if self.version in {RUL_MODEL_VERSION_V2_5, *TARGET_SPECIFIC_RUL_MODEL_VERSIONS}
                else False
            ),
            support_distance=forecastability_details.get("support_distance"),
            neighbor_dispersion_hours=forecastability_details.get("neighbor_dispersion_hours"),
            model_disagreement_hours=forecastability_details.get("model_disagreement_hours"),
            warning_forecastability_state=(
                target_resolutions["warning"]["state"]
                if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else "RUL_ACTIVE"
            ),
            warning_forecastability_score=(target_scores.get("warning") if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else forecastability_score),
            warning_serviceable_intent=(warning_emit if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else True),
            warning_hard_eligible=(target_resolutions.get("warning", {}).get("reason_code") != "OUT_OF_SUPPORT" if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else True),
            warning_selector_active=(bool(target_resolutions.get("warning", {}).get("selector_active")) if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else selector_active_for_output),
            warning_withholding_reason_code=(None if warning_emit else target_resolutions["warning"]["reason_code"]) if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else None,
            warning_withholding_reasons=(() if warning_emit else target_resolutions["warning"]["reasons"]) if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else (),
            warning_raw_point_hours=target_raw_points.get("warning"),
            warning_corrected_point_hours=target_corrected_points.get("warning"),
            warning_calibration_stratum=warning_details.get("bucket"),
            critical_forecastability_state=(
                target_resolutions["critical"]["state"]
                if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else "RUL_ACTIVE"
            ),
            critical_forecastability_score=(target_scores.get("critical") if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else forecastability_score),
            critical_serviceable_intent=(critical_emit if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else True),
            critical_hard_eligible=(target_resolutions.get("critical", {}).get("reason_code") != "OUT_OF_SUPPORT" if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else True),
            critical_selector_active=(bool(target_resolutions.get("critical", {}).get("selector_active")) if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else selector_active_for_output),
            critical_withholding_reason_code=(None if critical_emit else target_resolutions["critical"]["reason_code"]) if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else None,
            critical_withholding_reasons=(() if critical_emit else target_resolutions["critical"]["reasons"]) if self.version in TARGET_SPECIFIC_RUL_MODEL_VERSIONS else (),
            critical_raw_point_hours=target_raw_points.get("critical"),
            critical_corrected_point_hours=target_corrected_points.get("critical"),
            critical_calibration_stratum=critical_details.get("bucket"),
        )
        self._last_prediction[machine_key] = result
        return result
