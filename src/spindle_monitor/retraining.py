from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

from .config import ProjectConfig
from .data_profile import file_sha256
from .domain import candidate_stage, promotion_refusal
from .duration_evaluation import duration_bucket_evaluation
from .environment import environment_compatibility, runtime_environment
from .model_registry import ModelRegistry
from .monitor import ConditionMonitor
from .offline import prepare_offline_replay
from .policy_contract import (
    CONTRACT,
    MODEL_METADATA_SCHEMA_VERSION,
    FORECAST_POLICY_CONTRACT_VERSION,
    FORECAST_POLICY_VERSION,
)


PROBABILITY_FEATURE_POLICY_VERSION = "relative_state_plus_trend_v4_normalized_hard_negative"


@dataclass
class LifecycleExamples:
    lifecycle_id: str
    timestamps: list[datetime]
    feature_rows: list[list[float]]
    baseline_warning: list[float | None]
    baseline_critical: list[float | None]
    first_warning_timestamp: datetime | None
    first_critical_timestamp: datetime | None
    duration_hours: float = 0.0
    operating_regime: str = "unknown"
    censored: bool = False
    degradation_family: str = "unknown"

    @property
    def warning_reached(self) -> bool:
        return self.first_warning_timestamp is not None

    @property
    def critical_reached(self) -> bool:
        return self.first_critical_timestamp is not None

    def target_rows(
        self, target: str, maximum: int
    ) -> tuple[list[list[float]], list[float], list[float | None], list[datetime]]:
        event = getattr(self, f"first_{target}_timestamp")
        if event is None:
            return [], [], [], []
        baseline = getattr(self, f"baseline_{target}")
        eligible = [index for index, timestamp in enumerate(self.timestamps) if timestamp < event]
        selected = _subsample_indices(len(eligible), maximum)
        indices = [eligible[int(index)] for index in selected]
        return (
            [self.feature_rows[index] for index in indices],
            [(event - self.timestamps[index]).total_seconds() / 3600.0 for index in indices],
            [baseline[index] for index in indices],
            [self.timestamps[index] for index in indices],
        )

    def probability_rows(
        self, target: str, horizon_hours: float
    ) -> tuple[list[list[float]], np.ndarray, list[datetime]]:
        """Return natural pre-onset labels, including event-free lifecycles.

        Event onset itself and every post-onset row are excluded. A lifecycle
        without the target event contributes only legitimate negatives.
        """
        event = getattr(self, f"first_{target}_timestamp")
        indices = [
            index for index, timestamp in enumerate(self.timestamps)
            if event is None or timestamp < event
        ]
        labels = np.asarray([
            int(event is not None and 0.0 < (event - self.timestamps[index]).total_seconds() / 3600.0 <= horizon_hours)
            for index in indices
        ], dtype=int)
        return (
            [self.feature_rows[index] for index in indices],
            labels,
            [self.timestamps[index] for index in indices],
        )


@dataclass
class ConstantProbabilityClassifier:
    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = float(np.clip(self.probability, 0.0, 1.0))
        return np.tile(np.asarray([1.0 - probability, probability]), (len(x), 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), int(self.probability >= 0.5), dtype=int)


@dataclass
class CalibratedFeatureClassifier:
    """Serializable estimator with causal direct features and relative-state residuals.

    ``residual_pairs`` are indexes into the unchanged runtime feature vector.
    Each derived feature is ``x[left] - x[right]``.  This lets a v5 research
    model use local/rolling baseline deviations without making an absolute
    sensor level a direct predictive input or changing the runtime CSV/schema.
    ``getattr`` is intentionally used so older serialized v3/v4 wrappers that
    predate ``residual_pairs`` remain loadable.
    """

    estimator: Any
    feature_indices: tuple[int, ...]
    calibration_method: str = "none"
    calibrator: Any | None = None
    residual_pairs: tuple[tuple[int, int], ...] = ()
    standardized_residual_specs: tuple[tuple[int, int, int, float], ...] = ()
    trend_snr_specs: tuple[tuple[int, int, float, float], ...] = ()
    derived_feature_clip: float = 8.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=float)
        direct = rows[:, self.feature_indices]
        residual_pairs = getattr(self, "residual_pairs", ())
        derived: list[np.ndarray] = []
        if residual_pairs:
            derived.append(np.column_stack([
                rows[:, left] - rows[:, right]
                for left, right in residual_pairs
            ]))
        clip = float(getattr(self, "derived_feature_clip", 8.0))
        standardized_specs = getattr(self, "standardized_residual_specs", ())
        if standardized_specs:
            standardized = []
            for left, right, scale, floor in standardized_specs:
                denominator = np.maximum(np.abs(rows[:, scale]), float(floor))
                value = (rows[:, left] - rows[:, right]) / denominator
                standardized.append(np.clip(value, -clip, clip))
            derived.append(np.column_stack(standardized))
        trend_specs = getattr(self, "trend_snr_specs", ())
        if trend_specs:
            trend_snr = []
            for slope, scale, window_hours, floor in trend_specs:
                denominator = np.maximum(np.abs(rows[:, scale]), float(floor))
                value = rows[:, slope] * float(window_hours) / denominator
                trend_snr.append(np.clip(value, -clip, clip))
            derived.append(np.column_stack(trend_snr))
        if not derived:
            return direct
        return np.column_stack((direct, *derived))

    def raw_predict_proba(self, x: np.ndarray) -> np.ndarray:
        selected = self.transform(x)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(selected)[:, 1], dtype=float)
        return np.asarray(self.estimator.predict(selected), dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = np.clip(self.raw_predict_proba(x), 1e-9, 1.0 - 1e-9)
        if self.calibration_method == "sigmoid" and self.calibrator is not None:
            calibrated = self.calibrator.predict_proba(
                np.log(raw / (1.0 - raw)).reshape(-1, 1)
            )[:, 1]
        elif self.calibration_method == "isotonic" and self.calibrator is not None:
            calibrated = self.calibrator.predict(raw)
        else:
            calibrated = raw
        calibrated = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
        return np.column_stack((1.0 - calibrated, calibrated))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= .5).astype(int)


@dataclass
class FeatureSelectedRegressor:
    estimator: Any
    feature_indices: tuple[int, ...]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(np.asarray(x, dtype=float)[:, self.feature_indices]),
            dtype=float,
        )


def _regressor(config: ProjectConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.ml.max_iter,
        learning_rate=config.ml.learning_rate,
        max_leaf_nodes=config.ml.max_leaf_nodes,
        l2_regularization=config.ml.l2_regularization,
        random_state=config.ml.random_state,
        loss="absolute_error",
    )


def _classifier(config: ProjectConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config.ml.max_iter,
        learning_rate=config.ml.learning_rate,
        max_leaf_nodes=config.ml.max_leaf_nodes,
        l2_regularization=config.ml.l2_regularization,
        random_state=config.ml.random_state,
        loss="log_loss",
    )


def _fit_probability_model(
    x: np.ndarray, y: np.ndarray, config: ProjectConfig, sample_weight: np.ndarray | None = None
):
    if len(np.unique(y)) < 2:
        return ConstantProbabilityClassifier(float(np.mean(y)))
    model = _classifier(config)
    model.fit(x, y, sample_weight=sample_weight)
    return model


def probability_sample_weights(
    labels: np.ndarray,
    lifecycle_ids: list[str],
    lead_hours: np.ndarray | None = None,
    horizon_hours: float | None = None,
    early_positive_weight_multiplier: float = 1.0,
    positive_class_balance_strength: float = 1.0,
    boundary_negative_weight_multiplier: float = 1.0,
) -> np.ndarray:
    """Equalize lifecycle influence and gently favor earlier positives.

    V5 fully equalized positive and negative class mass.  Combined with the new
    relative-state residuals, that made the classifier far too alarm-prone.  V5.1
    therefore applies only *partial* class balancing, controlled by an exponent:
    0 disables class balancing and 1 reproduces the old full balancing behavior.

    Earlier positives are still emphasized, but only by redistributing weight
    within the positive class.  Negatives immediately outside the requested
    prediction horizon may also be emphasized.  Those boundary negatives teach
    the classifier that "degradation exists" is not the same as "event is inside
    the 6/12/24-hour forecast window".
    """
    labels = np.asarray(labels, dtype=int)
    weights = lifecycle_sample_weights(lifecycle_ids)
    positive = labels == 1
    negative = ~positive
    positive_total = float(np.sum(weights[positive]))
    negative_total = float(np.sum(weights[negative]))
    if positive_total > 0 and negative_total > 0 and positive_class_balance_strength > 0:
        ratio = negative_total / positive_total
        weights[positive] *= ratio ** float(positive_class_balance_strength)
    if (
        lead_hours is not None
        and horizon_hours is not None
        and horizon_hours > 0
        and early_positive_weight_multiplier > 1
        and np.any(positive)
    ):
        before = float(np.sum(weights[positive]))
        leads = np.asarray(lead_hours, dtype=float)
        if len(leads) != len(labels):
            raise ValueError("lead_hours length must match probability labels")
        normalized = np.clip(leads / float(horizon_hours), 0.0, 1.0)
        emphasis = 1.0 + (float(early_positive_weight_multiplier) - 1.0) * normalized**2
        emphasis = np.where(np.isfinite(emphasis), emphasis, 1.0)
        weights[positive] *= emphasis[positive]
        after = float(np.sum(weights[positive]))
        if before > 0 and after > 0:
            # Keep class balance unchanged; only redistribute positive influence.
            weights[positive] *= before / after
    if (
        lead_hours is not None
        and horizon_hours is not None
        and horizon_hours > 0
        and boundary_negative_weight_multiplier > 1
        and np.any(negative)
    ):
        leads = np.asarray(lead_hours, dtype=float)
        if len(leads) != len(labels):
            raise ValueError("lead_hours length must match probability labels")
        boundary_negative = (
            negative
            & np.isfinite(leads)
            & (leads > float(horizon_hours))
            & (leads <= 2.0 * float(horizon_hours))
        )
        weights[boundary_negative] *= float(boundary_negative_weight_multiplier)
    return weights * (len(weights) / max(float(np.sum(weights)), 1e-12))


def predictive_feature_indices(names: list[str]) -> tuple[int, ...]:
    """Keep raw age for OOD evidence but prevent it from becoming a risk proxy."""
    audit_only = {
        "elapsed_lifecycle_hours",
        "consecutive_warning_hours",
        "consecutive_critical_hours",
    }
    return tuple(index for index, name in enumerate(names) if name not in audit_only)


def probability_feature_indices(names: list[str]) -> tuple[int, ...]:
    """Use causal change evidence for forecast probabilities only.

    The runtime feature schema is intentionally unchanged so the installed v3
    production candidate remains loadable.  New probability estimators receive
    only rate/change features; absolute levels, rolling levels, severity,
    variability, threshold-history counts, and lifecycle-age proxies remain
    available to manufacturer safety/audit logic but cannot become ML risk
    shortcuts.
    """
    allowed_tokens = (
        "kalman_rate_per_hour",
        "fast_slow_difference",
        "slope_per_hour",
        "__change",
        "slope_acceleration_per_hour2",
    )
    selected = tuple(
        index for index, name in enumerate(names)
        if any(token in name for token in allowed_tokens)
    )
    if not selected:
        raise ValueError("probability feature policy selected no causal change features")
    return selected


def probability_residual_feature_pairs(names: list[str]) -> tuple[tuple[int, int], ...]:
    """Build causal local-baseline residuals without exposing absolute level directly.

    Each pair computes ``short/current state - longer rolling baseline`` using
    features that already exist in the 515-column runtime schema.  These
    residuals restore degradation-state context that v4 lost, while remaining
    substantially more invariant to a legitimately high but stable load regime
    than raw/rolling absolute level features.
    """
    by_name = {name: index for index, name in enumerate(names)}
    sensor_prefixes = sorted({
        name.split("__", 1)[0]
        for name in names
        if "__" in name and name.split("__", 1)[0] in {
            "vibration_mps2", "temperature_c", "current_ampere"
        }
    })
    suffix_pairs = (
        ("__kalman_level", "__w1440m__median"),
        ("__slow_ewma", "__w1440m__median"),
        ("__w15m__mean", "__w360m__median"),
        ("__w60m__mean", "__w360m__median"),
        ("__w60m__mean", "__w720m__median"),
        ("__w180m__mean", "__w720m__median"),
        ("__w180m__mean", "__w1440m__median"),
        ("__w360m__mean", "__w1440m__median"),
    )
    pairs: list[tuple[int, int]] = []
    for sensor in sensor_prefixes:
        for left_suffix, right_suffix in suffix_pairs:
            left = by_name.get(sensor + left_suffix)
            right = by_name.get(sensor + right_suffix)
            if left is not None and right is not None:
                pairs.append((left, right))
    return tuple(pairs)


def probability_residual_feature_names(
    names: list[str], residual_pairs: tuple[tuple[int, int], ...]
) -> list[str]:
    return [f"residual::{names[left]}-minus-{names[right]}" for left, right in residual_pairs]


def _robust_scale_floor(values: np.ndarray) -> float:
    """Return a training-only denominator floor for normalized derived features.

    A pure division by rolling standard deviation is unstable when a healthy
    period is nearly flat.  The lower-quartile positive scale keeps residuals
    dimensionless while preventing tiny denominators from turning harmless
    sensor noise into an extreme degradation score.
    """
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 1e-12)]
    if not len(finite):
        return 1e-6
    return max(float(np.quantile(finite, 0.25)), 1e-6)


def probability_standardized_residual_specs(
    names: list[str], training_x: np.ndarray
) -> tuple[tuple[int, int, int, float], ...]:
    """Create load/noise-normalized relative-state features from existing columns.

    V5/V5.1 used absolute *differences* between short and long local baselines.
    Those differences still vary with operating regime and sensor noise scale.
    V5.2 additionally divides each difference by a long-window variability
    estimate, with a training-only robust floor.  This should make a 0.2-unit
    shift under a quiet regime more meaningful than the same shift under a
    naturally noisy/high-load regime without exposing raw absolute level.
    """
    rows = np.asarray(training_x, dtype=float)
    by_name = {name: index for index, name in enumerate(names)}
    sensor_prefixes = ("vibration_mps2", "temperature_c", "current_ampere")
    suffix_specs = (
        ("__kalman_level", "__w1440m__median", "__w1440m__std"),
        ("__slow_ewma", "__w1440m__median", "__w1440m__std"),
        ("__w15m__mean", "__w360m__median", "__w360m__std"),
        ("__w60m__mean", "__w360m__median", "__w360m__std"),
        ("__w60m__mean", "__w720m__median", "__w720m__std"),
        ("__w180m__mean", "__w720m__median", "__w720m__std"),
        ("__w180m__mean", "__w1440m__median", "__w1440m__std"),
        ("__w360m__mean", "__w1440m__median", "__w1440m__std"),
    )
    specs: list[tuple[int, int, int, float]] = []
    for sensor in sensor_prefixes:
        for left_suffix, right_suffix, scale_suffix in suffix_specs:
            left = by_name.get(sensor + left_suffix)
            right = by_name.get(sensor + right_suffix)
            scale = by_name.get(sensor + scale_suffix)
            if left is None or right is None or scale is None:
                continue
            floor = _robust_scale_floor(rows[:, scale])
            specs.append((left, right, scale, floor))
    return tuple(specs)


def probability_standardized_residual_feature_names(
    names: list[str], specs: tuple[tuple[int, int, int, float], ...]
) -> list[str]:
    return [
        f"normalized_residual::{names[left]}-minus-{names[right]}-over-{names[scale]}"
        for left, right, scale, _ in specs
    ]


def probability_trend_snr_specs(
    names: list[str], training_x: np.ndarray
) -> tuple[tuple[int, int, float, float], ...]:
    """Create persistent-trend signal-to-noise features for slow degradation.

    ``slope_per_hour * window_hours / rolling_std`` approximates how much of the
    recent movement is directional drift rather than ordinary variability.
    Long windows are intentionally included because v5.1 performed worst on
    720h+ lifecycles where degradation evolves slowly.
    """
    rows = np.asarray(training_x, dtype=float)
    by_name = {name: index for index, name in enumerate(names)}
    specs: list[tuple[int, int, float, float]] = []
    for sensor in ("vibration_mps2", "temperature_c", "current_ampere"):
        for minutes in (60, 180, 360, 720, 1440):
            slope = by_name.get(f"{sensor}__w{minutes}m__slope_per_hour")
            scale = by_name.get(f"{sensor}__w{minutes}m__std")
            if slope is None or scale is None:
                continue
            floor = _robust_scale_floor(rows[:, scale])
            specs.append((slope, scale, minutes / 60.0, floor))
    return tuple(specs)


def probability_trend_snr_feature_names(
    names: list[str], specs: tuple[tuple[int, int, float, float], ...]
) -> list[str]:
    return [
        f"trend_snr::{names[slope]}-over-{names[scale]}"
        for slope, scale, _, _ in specs
    ]


def split_probability_validation_lifecycles(
    labels: np.ndarray,
    lifecycle_ids: list[str] | np.ndarray,
    minimum_lifecycles_for_calibration: int = 8,
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Split validation lifecycles into calibration and threshold/model-selection subsets.

    Lifecycle IDs, not rows, are the unit of separation. If both subsets cannot
    retain both classes, calibration is disabled and the complete validation set
    is used only for model/threshold selection. Test lifecycles are never an
    input to this function.
    """
    labels = np.asarray(labels, dtype=int)
    ids = np.asarray(lifecycle_ids, dtype=str)
    if len(labels) != len(ids):
        raise ValueError("validation labels/lifecycle IDs length mismatch")
    unique_ids = sorted(set(ids.tolist()))
    if len(unique_ids) < minimum_lifecycles_for_calibration:
        # With only four v4 validation lifecycles, a calibration split left
        # threshold/model selection dependent on only two lifecycles.  That is
        # too unstable.  Prefer the full validation cohort and no calibrator.
        return (
            np.zeros(len(labels), dtype=bool),
            np.ones(len(labels), dtype=bool),
            set(),
        )
    positive_ids = sorted({
        lifecycle_id for lifecycle_id in unique_ids
        if np.any(labels[ids == lifecycle_id] == 1)
    })
    negative_ids = sorted(set(unique_ids) - set(positive_ids))
    calibration_ids = set(positive_ids[::2] + negative_ids[::2])
    calibration_mask = np.asarray([value in calibration_ids for value in ids], dtype=bool)
    selection_mask = ~calibration_mask
    split_calibration = bool(
        np.any(calibration_mask)
        and np.any(selection_mask)
        and len(np.unique(labels[calibration_mask])) == 2
        and len(np.unique(labels[selection_mask])) == 2
    )
    if not split_calibration:
        calibration_mask = np.zeros(len(labels), dtype=bool)
        selection_mask = np.ones(len(labels), dtype=bool)
        calibration_ids = set()
    if calibration_ids & set(ids[selection_mask]):
        raise RuntimeError("calibration and threshold-selection lifecycle overlap")
    return calibration_mask, selection_mask, calibration_ids


def _fit_calibrator(method: str, probabilities: np.ndarray, labels: np.ndarray) -> Any | None:
    if method == "none" or len(np.unique(labels)) < 2:
        return None
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    if method == "sigmoid":
        calibrator = LogisticRegression(random_state=0, solver="lbfgs")
        calibrator.fit(np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1), labels)
        return calibrator
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, labels)
    return calibrator


def _probability_dataset(
    examples: Iterable[LifecycleExamples], target: str, horizon_hours: float
) -> tuple[np.ndarray, np.ndarray, list[str], list[datetime], np.ndarray]:
    x: list[list[float]] = []
    labels: list[int] = []
    lifecycle_ids: list[str] = []
    timestamps: list[datetime] = []
    lead_hours: list[float] = []
    for example in examples:
        rows, values, times = example.probability_rows(target, horizon_hours)
        event = getattr(example, f"first_{target}_timestamp")
        x.extend(rows)
        labels.extend(values.tolist())
        lifecycle_ids.extend([example.lifecycle_id] * len(rows))
        timestamps.extend(times)
        lead_hours.extend([
            ((event - timestamp).total_seconds() / 3600.0) if event is not None else float("nan")
            for timestamp in times
        ])
    if not x:
        return (
            np.empty((0, 0), dtype=float), np.asarray(labels, dtype=int), lifecycle_ids,
            timestamps, np.asarray(lead_hours, dtype=float),
        )
    return (
        np.asarray(x, dtype=float), np.asarray(labels, dtype=int), lifecycle_ids,
        timestamps, np.asarray(lead_hours, dtype=float),
    )


def _probability_estimators(config: ProjectConfig) -> list[tuple[str, Any]]:
    return [
        ("HistGradientBoostingClassifier", _classifier(config)),
        ("ExtraTreesClassifier", ExtraTreesClassifier(
            n_estimators=120, min_samples_leaf=8, max_features="sqrt",
            n_jobs=2, random_state=config.ml.random_state, class_weight=None,
        )),
        ("ExtraTreesStableClassifier", ExtraTreesClassifier(
            n_estimators=180, min_samples_leaf=16, max_features=0.35,
            max_depth=18, n_jobs=2, random_state=config.ml.random_state + 17,
            class_weight=None,
        )),
    ]


def _cross_fitted_probability_scores(
    estimator: Any,
    x: np.ndarray,
    labels: np.ndarray,
    lifecycle_ids: list[str],
    sample_weight: np.ndarray | None,
    feature_indices: tuple[int, ...],
    residual_pairs: tuple[tuple[int, int], ...],
    standardized_specs: tuple[tuple[int, int, int, float], ...],
    trend_specs: tuple[tuple[int, int, float, float], ...],
    folds: int,
) -> np.ndarray | None:
    """Return lifecycle-held-out training probabilities for hard-example mining.

    These scores never use validation or test data.  Their purpose is to find
    training negatives that look fault-like *to a model that did not train on
    that lifecycle*, which is much safer than mining hard examples from fitted
    in-sample predictions.
    """
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(lifecycle_ids, dtype=str)
    unique_groups = sorted(set(groups.tolist()))
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2 or len(np.unique(labels)) < 2:
        return None
    scores = np.full(len(labels), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, held_out_index in splitter.split(x, labels, groups):
        fold_labels = labels[train_index]
        if len(np.unique(fold_labels)) < 2:
            scores[held_out_index] = float(np.mean(fold_labels))
            continue
        fold_estimator = clone(estimator)
        fold_wrapper = CalibratedFeatureClassifier(
            fold_estimator,
            feature_indices,
            residual_pairs=residual_pairs,
            standardized_residual_specs=standardized_specs,
            trend_snr_specs=trend_specs,
        )
        fold_weight = sample_weight[train_index] if sample_weight is not None else None
        fold_estimator.fit(
            fold_wrapper.transform(x[train_index]),
            fold_labels,
            sample_weight=fold_weight,
        )
        scores[held_out_index] = fold_wrapper.predict_proba(x[held_out_index])[:, 1]
    return scores if np.all(np.isfinite(scores)) else None


def probability_hard_example_multipliers(
    labels: np.ndarray,
    oof_probabilities: np.ndarray,
    lead_hours: np.ndarray,
    horizon_hours: float,
    hard_negative_quantile: float,
    hard_negative_multiplier: float,
    hard_early_positive_quantile: float,
    hard_early_positive_multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Upweight lifecycle-held-out confusions without changing class labels.

    High-scoring negatives are the most informative false-alarm examples.  Low-
    scoring positives in the *early half* of the horizon are the examples most
    likely to recreate v4's late-detection failure.  Mining both sides at once
    improves separation rather than merely shifting the decision threshold.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(oof_probabilities, dtype=float)
    leads = np.asarray(lead_hours, dtype=float)
    if len(labels) != len(probabilities) or len(labels) != len(leads):
        raise ValueError("hard-example inputs must have matching lengths")
    multipliers = np.ones(len(labels), dtype=float)
    negative = labels == 0
    positive = labels == 1
    finite_negative = negative & np.isfinite(probabilities)
    negative_cutoff = None
    hard_negative = np.zeros(len(labels), dtype=bool)
    if np.any(finite_negative):
        negative_cutoff = float(np.quantile(
            probabilities[finite_negative], float(hard_negative_quantile)
        ))
        hard_negative = finite_negative & (probabilities >= negative_cutoff)
        multipliers[hard_negative] *= float(hard_negative_multiplier)
    early_positive = (
        positive
        & np.isfinite(probabilities)
        & np.isfinite(leads)
        & (leads >= 0.5 * float(horizon_hours))
        & (leads <= float(horizon_hours))
    )
    early_positive_cutoff = None
    hard_early_positive = np.zeros(len(labels), dtype=bool)
    if np.any(early_positive):
        early_positive_cutoff = float(np.quantile(
            probabilities[early_positive], float(hard_early_positive_quantile)
        ))
        hard_early_positive = early_positive & (probabilities <= early_positive_cutoff)
        multipliers[hard_early_positive] *= float(hard_early_positive_multiplier)
    return multipliers, {
        "hard_negative_count": int(np.sum(hard_negative)),
        "hard_negative_probability_cutoff": negative_cutoff,
        "hard_early_positive_count": int(np.sum(hard_early_positive)),
        "hard_early_positive_probability_cutoff": early_positive_cutoff,
        "negative_quantile": float(hard_negative_quantile),
        "negative_multiplier": float(hard_negative_multiplier),
        "early_positive_quantile": float(hard_early_positive_quantile),
        "early_positive_multiplier": float(hard_early_positive_multiplier),
    }


def _subsample_indices(count: int, maximum: int) -> np.ndarray:
    """Deterministically cover early, middle, late, and near-event stages."""
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def build_training_examples(
    input_path: str | Path,
    config: ProjectConfig,
    models_root: str | Path,
) -> tuple[list[str], list[LifecycleExamples], dict[str, Any]]:
    validations, plan = prepare_offline_replay(input_path, config, models_root)
    monitor = ConditionMonitor(
        config, models_root=models_root, lifecycle_plan=plan, enable_ml=False
    )
    selected_timestamps: set[datetime] = set()
    for record in plan.records:
        lifecycle_rows = [
            value.reading.timestamp for value in validations
            if value.valid and value.reading is not None
            and record.start_timestamp <= value.reading.timestamp < record.end_timestamp
        ]
        for event in (record.first_warning_timestamp, record.first_critical_timestamp):
            eligible = [value for value in lifecycle_rows if event is None or value < event]
            selected_timestamps.update(
                eligible[int(index)] for index in _subsample_indices(len(eligible), config.ml.maximum_training_rows_per_lifecycle)
            )
            if event is not None:
                # Dense near-onset coverage prevents a 1,000-hour lifecycle
                # from contributing only a handful of 6-hour positives.
                for horizon in config.ml.forecast_horizons_hours:
                    near = [
                        value for value in eligible
                        if 0.0 < (event - value).total_seconds() / 3600.0 <= horizon
                    ]
                    selected_timestamps.update(
                        near[int(index)] for index in _subsample_indices(
                            len(near), max(60, config.ml.maximum_training_rows_per_lifecycle // 3)
                        )
                    )
                    # Also sample the immediately preceding negative band.
                    # These examples are especially valuable for learning the
                    # difference between "degrading" and "inside the forecast
                    # horizon", and cost little memory because they are drawn
                    # from the existing lifecycle CSV rather than generated.
                    boundary_negative = [
                        value for value in eligible
                        if horizon < (event - value).total_seconds() / 3600.0 <= 2 * horizon
                    ]
                    selected_timestamps.update(
                        boundary_negative[int(index)] for index in _subsample_indices(
                            len(boundary_negative), max(30, config.ml.maximum_training_rows_per_lifecycle // 10)
                        )
                    )
    rows_by_lifecycle: dict[str, list[Any]] = {}
    valid_rows = invalid_rows = 0
    eligible_rows = human_review_rows = interpolated_eligible_rows = 0
    exclusion_counts: Counter[str] = Counter()
    for validation in validations:
        if not validation.valid or validation.reading is None:
            invalid_rows += 1
            exclusion_counts["invalid_sensor_data"] += int(config.anomaly.enabled)
            continue
        keep_features = validation.reading.timestamp in selected_timestamps
        result = monitor.process(
            validation.reading,
            compute_features=keep_features,
            interpolated=validation.interpolated,
            source_sampling_interval_seconds=validation.source_sampling_interval_seconds,
            effective_resampling_interval_seconds=validation.effective_resampling_interval_seconds,
            input_features_available=validation.features_available,
        )
        valid_rows += 1
        if result.anomaly.training_eligible:
            eligible_rows += 1
            interpolated_eligible_rows += int(result.interpolated)
        else:
            exclusion_counts[
                result.anomaly.training_exclusion_reason or "anomaly_policy_exclusion"
            ] += 1
        human_review_rows += int(result.anomaly.requires_human_review)
        if keep_features and result.anomaly.training_eligible:
            rows_by_lifecycle.setdefault(result.lifecycle_id, []).append(result)

    examples: list[LifecycleExamples] = []
    for record in plan.records:
        rows = [
            result for result in rows_by_lifecycle.get(record.lifecycle_id, [])
            if result.timestamp < record.end_timestamp
        ]
        if len(rows) < config.ml.minimum_samples_per_lifecycle:
            continue
        examples.append(
            LifecycleExamples(
                lifecycle_id=record.lifecycle_id,
                timestamps=[result.timestamp for result in rows],
                feature_rows=[
                    [result.features[name] for name in monitor.features.names] for result in rows
                ],
                baseline_warning=[
                    result.statistical_warning_forecast.estimated_hours
                    if result.statistical_warning_forecast else None for result in rows
                ],
                baseline_critical=[
                    result.statistical_critical_forecast.estimated_hours
                    if result.statistical_critical_forecast else None for result in rows
                ],
                first_warning_timestamp=record.first_warning_timestamp,
                first_critical_timestamp=record.first_critical_timestamp,
                duration_hours=record.duration_hours,
                operating_regime=record.operating_regime,
                censored=record.censored,
                degradation_family=record.degradation_family,
            )
        )
    raw_timestamps = [
        value.reading.timestamp for value in validations
        if value.valid and value.reading is not None and not value.interpolated
    ]
    raw_intervals = np.asarray([
        (right - left).total_seconds() for left, right in zip(raw_timestamps, raw_timestamps[1:])
        if right > left
    ], dtype=float)
    anomaly_snapshot = asdict(config.anomaly)
    anomaly_config_hash = hashlib.sha256(
        json.dumps(anomaly_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return monitor.features.names, examples, {
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "completed_lifecycles": len(plan.records),
        "usable_lifecycles": len(examples),
        "confirmed_boundary_timestamps": [value.isoformat() for value in plan.boundary_timestamps],
        "raw_sampling_interval_distribution_seconds": {
            "minimum": float(np.min(raw_intervals)) if len(raw_intervals) else None,
            "p50": float(np.median(raw_intervals)) if len(raw_intervals) else None,
            "p95": float(np.quantile(raw_intervals, .95)) if len(raw_intervals) else None,
            "maximum": float(np.max(raw_intervals)) if len(raw_intervals) else None,
        },
        "effective_resampling_interval_seconds": config.target_sampling_interval_seconds,
        "interpolation_enabled": config.interpolation_enabled,
        "interpolated_row_count": sum(value.interpolated for value in validations if value.valid),
        "large_gap_policy": config.large_gap_policy,
        "anomaly_screening_enabled": config.anomaly.enabled,
        "anomaly_detector_version": config.anomaly.detector_version,
        "anomaly_configuration_sha256": anomaly_config_hash,
        "anomaly_training_eligible_rows": eligible_rows,
        "anomaly_training_excluded_rows": sum(exclusion_counts.values()),
        "anomaly_training_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "anomaly_rows_requiring_human_review": human_review_rows,
        "interpolated_training_eligible_rows": interpolated_eligible_rows,
        "interpolated_training_policy": (
            "excluded_pending_policy_review" if config.anomaly.enabled
            else "legacy_reproduction_policy"
        ),
    }


def _lifecycle_duration_bucket(example: LifecycleExamples, config: ProjectConfig) -> str:
    lower = 0.0
    for upper in config.duration_bucket_boundaries_hours:
        if example.duration_hours < upper:
            return f"[{lower:g},{upper:g})h"
        lower = upper
    return f"[{lower:g},inf)h"


def _split_once(
    ordered: list[LifecycleExamples], config: ProjectConfig, seed: int
) -> tuple[list[LifecycleExamples], list[LifecycleExamples], list[LifecycleExamples]]:
    labels = np.asarray([int(example.critical_reached) for example in ordered], dtype=int)
    indices = np.arange(len(ordered))
    train_count = max(1, int(round(len(ordered) * config.ml.training_fraction)))
    train_count = min(train_count, len(ordered) - 2)
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    train_indices, remaining = train_test_split(
        indices,
        train_size=train_count,
        random_state=seed,
        stratify=stratify,
    )
    remaining_labels = labels[remaining]
    validation_fraction = config.ml.validation_fraction / (
        config.ml.validation_fraction + config.ml.test_fraction
    )
    validation_count = max(1, int(round(len(remaining) * validation_fraction)))
    validation_count = min(validation_count, len(remaining) - 1)
    remaining_stratify = (
        remaining_labels
        if len(np.unique(remaining_labels)) == 2
        and min(np.bincount(remaining_labels)) >= 2
        and validation_count >= 2
        and len(remaining) - validation_count >= 2
        else None
    )
    validation_indices, test_indices = train_test_split(
        remaining,
        train_size=validation_count,
        random_state=seed + 1,
        stratify=remaining_stratify,
    )
    pick = lambda values: [ordered[int(index)] for index in sorted(values)]
    return pick(train_indices), pick(validation_indices), pick(test_indices)


def _distribution_distance(
    group: list[LifecycleExamples], all_values: list[LifecycleExamples], attribute,
) -> float:
    categories = sorted({attribute(value) for value in all_values})
    if not group or not categories:
        return float("inf")
    total_counts = Counter(attribute(value) for value in all_values)
    group_counts = Counter(attribute(value) for value in group)
    return float(sum(
        abs(group_counts[category] / len(group) - total_counts[category] / len(all_values))
        for category in categories
    ))


def _split_quality_score(
    groups: tuple[list[LifecycleExamples], list[LifecycleExamples], list[LifecycleExamples]],
    config: ProjectConfig,
) -> tuple[float, ...]:
    training, validation, test = groups
    all_values = [*training, *validation, *test]
    dimensions = (
        (lambda value: str(value.critical_reached).lower(), 4.0),
        (lambda value: value.operating_regime, 3.0),
        (lambda value: _lifecycle_duration_bucket(value, config), 1.5),
        (lambda value: value.degradation_family, 1.0),
    )
    weighted_distance = 0.0
    for group in (validation, test):
        for attribute, weight in dimensions:
            weighted_distance += weight * _distribution_distance(group, all_values, attribute)
    global_regimes = {value.operating_regime for value in all_values}
    regime_missing = sum(
        len(global_regimes - {value.operating_regime for value in group})
        for group in (validation, test)
    )
    outcome_missing = sum(
        len({False, True} - {value.critical_reached for value in group})
        for group in (validation, test)
    )
    # Prioritize representation coverage, then distribution distance.  The
    # deterministic seed tie-break keeps repeated runs byte-for-byte stable.
    return float(outcome_missing), float(regime_missing), float(weighted_distance)


def split_lifecycles(
    examples: list[LifecycleExamples], config: ProjectConfig
) -> tuple[list[LifecycleExamples], list[LifecycleExamples], list[LifecycleExamples]]:
    """Create deterministic lifecycle groups with representation-aware seed search."""
    ordered = sorted(examples, key=lambda example: example.lifecycle_id)
    if len(ordered) < 3:
        return ordered, [], []
    candidates: list[tuple[tuple[float, ...], int, tuple[list[LifecycleExamples], list[LifecycleExamples], list[LifecycleExamples]]]] = []
    for offset in range(config.ml.lifecycle_split_candidate_attempts):
        seed = config.ml.split_random_seed + offset
        groups = _split_once(ordered, config, seed)
        candidates.append((_split_quality_score(groups, config), seed, groups))
    _, _, best = min(candidates, key=lambda value: (value[0], value[1]))
    return best


def _split_representation(
    groups: dict[str, list[LifecycleExamples]], config: ProjectConfig
) -> dict[str, Any]:
    boundaries = config.duration_bucket_boundaries_hours
    def duration_bucket(hours: float) -> str:
        lower = 0.0
        for upper in boundaries:
            if hours < upper:
                return f"[{lower:g},{upper:g})h"
            lower = upper
        return f"[{lower:g},inf)h"
    report: dict[str, Any] = {}
    for name, values in groups.items():
        dimensions = {
            "duration_bucket": [duration_bucket(value.duration_hours) for value in values],
            "operating_regime": [value.operating_regime for value in values],
            "warning_reached": [str(value.warning_reached).lower() for value in values],
            "critical_reached": [str(value.critical_reached).lower() for value in values],
            "censoring_state": ["censored" if value.censored else "uncensored" for value in values],
            "degradation_family": [value.degradation_family for value in values],
        }
        report[name] = {
            key: {category: categories.count(category) for category in sorted(set(categories))}
            for key, categories in dimensions.items()
        }
    all_values = [value for values in groups.values() for value in values]
    critical_counts = [sum(value.critical_reached == state for value in all_values) for state in (False, True)]
    report["stratification"] = {
        "attempted_dimensions": [
            "critical_reached", "operating_regime", "duration_bucket", "degradation_family"
        ],
        "applied": len(all_values) >= 3 and min(critical_counts) >= 2,
        "limitations": [
            "Candidate lifecycle splits remain critical-stratified, then a deterministic representation-aware search minimizes validation/test mismatch across operating regime, duration bucket, and degradation family. Sparse category combinations still cannot be guaranteed in every split."
        ],
    }
    return report


def _target_dataset(
    examples: Iterable[LifecycleExamples], target: str, maximum: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[float | None], list[datetime]]:
    x: list[list[float]] = []
    y: list[float] = []
    lifecycle_ids: list[str] = []
    baselines: list[float | None] = []
    timestamps: list[datetime] = []
    for example in examples:
        rows, targets, baseline, sample_times = example.target_rows(target, maximum)
        x.extend(rows)
        y.extend(targets)
        lifecycle_ids.extend([example.lifecycle_id] * len(rows))
        baselines.extend(baseline)
        timestamps.extend(sample_times)
    if not x:
        return np.empty((0, 0), dtype=float), np.asarray(y, dtype=float), lifecycle_ids, baselines, timestamps
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), lifecycle_ids, baselines, timestamps


def lifecycle_sample_weights(lifecycle_ids: list[str]) -> np.ndarray:
    counts = {lifecycle_id: lifecycle_ids.count(lifecycle_id) for lifecycle_id in set(lifecycle_ids)}
    weights = np.asarray([1.0 / counts[value] for value in lifecycle_ids], dtype=float)
    return weights * (len(weights) / max(float(np.sum(weights)), 1e-12))


def regression_metrics(
    actual: Iterable[float], predicted: Iterable[float], lifecycle_ids: list[str]
) -> dict[str, Any]:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    errors = predicted_array - actual_array
    absolute = np.abs(errors)
    per_lifecycle: dict[str, dict[str, Any]] = {}
    for lifecycle_id in sorted(set(lifecycle_ids)):
        mask = np.asarray([value == lifecycle_id for value in lifecycle_ids])
        lifecycle_errors = errors[mask]
        per_lifecycle[lifecycle_id] = {
            "mae_hours": float(np.mean(np.abs(lifecycle_errors))),
            "bias_hours": float(np.mean(lifecycle_errors)),
            "sample_count": int(np.sum(mask)),
        }
    lifecycle_maes = [value["mae_hours"] for value in per_lifecycle.values()]
    lifecycle_biases = [value["bias_hours"] for value in per_lifecycle.values()]
    micro_mae = float(mean_absolute_error(actual_array, predicted_array))
    metrics: dict[str, Any] = {
        "micro_mae_hours": micro_mae,
        "micro_median_absolute_error_hours": float(median_absolute_error(actual_array, predicted_array)),
        "micro_bias_hours": float(np.mean(errors)),
        "micro_p90_absolute_error_hours": float(np.percentile(absolute, 90)),
        "macro_lifecycle_mae_hours": float(np.mean(lifecycle_maes)),
        "median_lifecycle_mae_hours": float(np.median(lifecycle_maes)),
        "worst_lifecycle_mae_hours": float(np.max(lifecycle_maes)),
        "macro_lifecycle_bias_hours": float(np.mean(lifecycle_biases)),
        "sample_count": int(len(actual_array)),
        "lifecycle_count": len(per_lifecycle),
        "per_lifecycle_metrics": per_lifecycle,
        # Backward-compatible aliases used by older audit consumers.
        "mae_hours": micro_mae,
        "median_absolute_error_hours": float(median_absolute_error(actual_array, predicted_array)),
        "bias_hours": float(np.mean(errors)),
        "p90_absolute_error_hours": float(np.percentile(absolute, 90)),
    }
    return metrics


def _baseline_metrics(
    actual: np.ndarray, baselines: list[float | None], lifecycle_ids: list[str]
) -> dict[str, Any] | None:
    valid = [index for index, value in enumerate(baselines) if value is not None]
    if not valid:
        return None
    return regression_metrics(
        actual[valid],
        [max(0.0, float(baselines[index])) for index in valid],
        [lifecycle_ids[index] for index in valid],
    )


def _paired_baseline_metrics(
    actual: np.ndarray,
    candidate: np.ndarray,
    baselines: list[float | None],
    lifecycle_ids: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    valid = [index for index, value in enumerate(baselines) if value is not None]
    if not valid:
        return None, None
    ids = [lifecycle_ids[index] for index in valid]
    return (
        regression_metrics(actual[valid], candidate[valid], ids),
        regression_metrics(
            actual[valid],
            [max(0.0, float(baselines[index])) for index in valid],
            ids,
        ),
    )


def _calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if np.any(mask):
            error += float(np.mean(mask)) * abs(float(np.mean(labels[mask])) - float(np.mean(probabilities[mask])))
    return float(error)


def probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    predicted = probabilities >= threshold
    positives = labels == 1
    negatives = labels == 0
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & negatives))
    false_negative = int(np.sum(~predicted & positives))
    true_negative = int(np.sum(~predicted & negatives))
    both_classes = len(np.unique(labels)) == 2
    metrics: dict[str, Any] = {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(true_negative / max(false_positive + true_negative, 1)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "false_positive_rate": float(false_positive / max(false_positive + true_negative, 1)),
        "false_negative_rate": float(false_negative / max(false_negative + true_positive, 1)),
        "observed_event_rate": float(np.mean(labels)),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "calibration_error": _calibration_error(labels, probabilities),
        "classification_threshold": float(threshold),
        "sample_count": int(len(labels)),
        "class_count": int(len(np.unique(labels))),
        "positive_class_count": int(np.sum(positives)),
        "negative_class_count": int(np.sum(negatives)),
        "probability_coverage": float(np.mean(np.isfinite(probabilities))),
    }
    if not both_classes:
        metrics["invalid_metric_reason"] = "Only one class exists in this split; ROC AUC is undefined."
    return metrics


def _event_detection_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    lifecycle_ids: list[str] | np.ndarray | None,
) -> dict[str, Any]:
    if lifecycle_ids is None or len(lifecycle_ids) != len(labels):
        return {"event_count": None, "detected_event_count": None, "missed_event_count": None, "event_false_negative_rate": None}
    ids = np.asarray(lifecycle_ids, dtype=str)
    predicted = np.asarray(probabilities, dtype=float) >= threshold
    events = detected = 0
    for lifecycle_id in sorted(set(ids)):
        mask = ids == lifecycle_id
        positives = mask & (np.asarray(labels, dtype=int) == 1)
        if np.any(positives):
            events += 1
            detected += int(np.any(predicted & positives))
    return {
        "event_count": events,
        "detected_event_count": detected,
        "missed_event_count": events - detected,
        "event_false_negative_rate": ((events - detected) / events if events else None),
    }


def _lead_time_detection_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    lead_hours: np.ndarray,
    horizon_hours: float,
) -> dict[str, Any]:
    """Measure whether positives are recognized early, not merely near onset."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    leads = np.asarray(lead_hours, dtype=float)
    predicted = probabilities >= threshold
    positive = labels == 1
    early = positive & np.isfinite(leads) & (leads >= float(horizon_hours) / 2.0)
    late = positive & np.isfinite(leads) & (leads < float(horizon_hours) / 2.0)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.sum(mask))
        if count == 0:
            return {"positive_count": 0, "detected_count": 0, "false_negative_rate": None}
        detected = int(np.sum(predicted & mask))
        return {
            "positive_count": count,
            "detected_count": detected,
            "false_negative_rate": float((count - detected) / count),
        }

    return {
        "horizon_hours": float(horizon_hours),
        "early_half": summarize(early),
        "late_half": summarize(late),
    }


def _probability_group_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    lifecycle_ids: list[str],
    examples: Iterable[LifecycleExamples],
    config: ProjectConfig,
) -> dict[str, Any]:
    by_id = {value.lifecycle_id: value for value in examples}
    boundaries = tuple(config.duration_bucket_boundaries_hours)

    def duration_bucket(hours: float) -> str:
        lower = 0.0
        for upper in boundaries:
            if hours < upper:
                return f"[{lower:g},{upper:g})h"
            lower = upper
        return f"[{lower:g},inf)h"

    dimensions = {
        "duration_bucket": {
            lifecycle_id: duration_bucket(by_id[lifecycle_id].duration_hours)
            for lifecycle_id in set(lifecycle_ids)
        },
        "operating_regime": {
            lifecycle_id: by_id[lifecycle_id].operating_regime
            for lifecycle_id in set(lifecycle_ids)
        },
    }
    report: dict[str, Any] = {}
    ids = np.asarray(lifecycle_ids, dtype=str)
    for dimension, mapping in dimensions.items():
        report[dimension] = {}
        for group in sorted(set(mapping.values())):
            lifecycle_group = {key for key, value in mapping.items() if value == group}
            mask = np.asarray([value in lifecycle_group for value in ids])
            metrics = probability_metrics(labels[mask], probabilities[mask], threshold)
            metrics["lifecycle_count"] = len(lifecycle_group)
            metrics["event_metrics"] = _event_detection_metrics(
                labels[mask], probabilities[mask], threshold, ids[mask]
            )
            report[dimension][group] = metrics
    return report


def _select_probability_threshold(labels: np.ndarray, probabilities: np.ndarray, default: float,
                                  maximum_fn_rate: float = 0.10,
                                  lifecycle_ids: list[str] | np.ndarray | None = None,
                                  maximum_fp_rate: float | None = None) -> dict[str, Any]:
    """Select per-target validation thresholds with FN constrained before FP.

    Each horizon calls this independently.  We first retain thresholds meeting
    the configured FN ceiling, then minimize FP; if none meet it, choose the
    lowest FN and retain the default-distance tie-break for reproducibility.
    """
    if len(np.unique(labels)) < 2:
        metrics = probability_metrics(labels, probabilities, default)
        return {"selected_threshold": float(default), "eligible": False,
                "reason": "single_class_in_split", "best_achievable_metrics": metrics,
                "comparison": "false_negative_rate <= maximum_probability_false_negative_rate"}
    finite = np.asarray(probabilities, dtype=float)
    finite = finite[np.isfinite(finite)]
    candidates = sorted({
        float(value) for value in np.concatenate((np.linspace(0.01, 0.99, 99), finite, np.asarray([default])))
        if 0.0 < float(value) < 1.0
    })
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    candidate_array = np.asarray(candidates, dtype=float)
    predicted = probabilities_array[None, :] >= candidate_array[:, None]
    positives = labels_array == 1; negatives = labels_array == 0
    fn_rates = np.sum(~predicted & positives[None, :], axis=1) / max(int(np.sum(positives)), 1)
    fp_rates = np.sum(predicted & negatives[None, :], axis=1) / max(int(np.sum(negatives)), 1)
    event_rates = np.zeros(len(candidates), dtype=float)
    if lifecycle_ids is not None and len(lifecycle_ids) == len(labels_array):
        ids = np.asarray(lifecycle_ids, dtype=str)
        event_maxima = [float(np.max(probabilities_array[(ids == lifecycle_id) & positives]))
                        for lifecycle_id in sorted(set(ids)) if np.any((ids == lifecycle_id) & positives)]
        if event_maxima:
            maxima = np.asarray(event_maxima, dtype=float)
            event_rates = np.mean(maxima[None, :] < candidate_array[:, None], axis=1)
    scored = []
    for index, threshold in enumerate(candidates):
        scored.append((float(fn_rates[index]), float(fp_rates[index]), float(event_rates[index]),
                       abs(threshold - default), threshold))
    eligible = [
        value for value in scored
        if value[0] <= maximum_fn_rate
        and value[2] <= maximum_fn_rate
        and (maximum_fp_rate is None or value[1] <= maximum_fp_rate)
    ]
    fn_feasible = [
        value for value in scored
        if value[0] <= maximum_fn_rate and value[2] <= maximum_fn_rate
    ]

    def normalized_excess(value: float, ceiling: float | None) -> float:
        if ceiling is None:
            return 0.0
        if ceiling <= 0:
            return 0.0 if value <= 0 else 1e6 + value
        return max(0.0, value - ceiling) / ceiling

    def fallback_score(value: tuple[float, float, float, float, float]) -> tuple[float, ...]:
        row_fn, fp, event_fn, default_distance, threshold = value
        fn_excess = normalized_excess(row_fn, maximum_fn_rate)
        event_excess = normalized_excess(event_fn, maximum_fn_rate)
        fp_excess = normalized_excess(fp, maximum_fp_rate)
        # Minimize the worst policy violation first, then total violation.  This
        # prevents an infeasible model from being represented by an
        # alarm-everywhere threshold merely because it achieves zero FN.
        return (
            max(fn_excess, event_excess, fp_excess),
            fn_excess + event_excess + fp_excess,
            fp,
            row_fn,
            event_fn,
            default_distance,
            threshold,
        )

    # Deterministic: among truly feasible thresholds minimize FP.  If the
    # constraints are jointly impossible, retain a balanced diagnostic
    # threshold rather than the lowest-FN/alarm-everywhere operating point.
    chosen = (
        min(eligible, key=lambda value: (value[1], value[2], value[3], value[4]))
        if eligible else min(scored, key=fallback_score)
    )
    if eligible:
        reason = None
    elif fn_feasible and maximum_fp_rate is not None:
        reason = "no_threshold_satisfies_joint_fn_fp_constraints"
    else:
        reason = "fn_ceiling_not_met"
    metrics = probability_metrics(labels, probabilities, float(chosen[4]))
    metrics["event_metrics"] = _event_detection_metrics(labels, probabilities, float(chosen[4]), lifecycle_ids)
    return {"selected_threshold": float(chosen[4]), "eligible": bool(eligible),
            "reason": reason,
            "best_achievable_metrics": metrics,
            "maximum_false_negative_rate": float(maximum_fn_rate),
            "maximum_false_positive_rate": (float(maximum_fp_rate) if maximum_fp_rate is not None else None),
            "fn_constraint_feasible": bool(fn_feasible),
            "joint_constraint_feasible": bool(eligible),
            "fallback_constraint_violation_score": (None if eligible else float(fallback_score(chosen)[0])),
            "comparison": "row/event FN must satisfy the selection ceiling; validation FP must satisfy its research ceiling when configured",
            "candidate_count": len(candidates),
            "selection_dataset": "validation_lifecycles",
            "tie_breaking": "feasible: lowest_fp; infeasible: smallest_normalized_joint_constraint_violation"}


def _event_balance_eligible(rate: float, config: ProjectConfig) -> bool:
    return config.ml.minimum_probability_positive_rate <= rate <= config.ml.maximum_probability_positive_rate


def probability_target_eligibility(
    split_rates: dict[str, float], class_counts: dict[str, int], config: ProjectConfig
) -> tuple[bool, str | None]:
    if not all(_event_balance_eligible(rate, config) for rate in split_rates.values()):
        return False, "target_is_nearly_constant"
    if not all(count >= 2 for count in class_counts.values()):
        return False, "single_class_in_split"
    return True, None


def _write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


EXTERNAL_ACCEPTANCE_FILENAMES = {
    "spindle_fp_rate_stress_test_100000.csv",
    "spindle_fn_rate_stress_test_100000.csv",
}


def validate_training_input(input_path: str | Path) -> dict[str, Any]:
    """Fail closed if external/acceptance evidence reaches model fitting."""
    path = Path(input_path).resolve()
    if path.name.lower() in EXTERNAL_ACCEPTANCE_FILENAMES:
        raise ValueError(f"External acceptance dataset cannot be used for training: {path.name}")
    sidecar_candidates = [
        path.with_name(f"{path.stem}_metadata.json"),
        path.with_suffix(".metadata.json"),
    ]
    sidecar = next((value for value in sidecar_candidates if value.exists()), None)
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar else {}
    if metadata.get("dataset_role") == "acceptance":
        raise ValueError("Held-out acceptance suite cannot be used for training")
    data_hash = file_sha256(path)
    project_data = Path(__file__).resolve().parents[2] / "data"
    external_hashes = {
        file_sha256(candidate)
        for name in EXTERNAL_ACCEPTANCE_FILENAMES
        for candidate in [project_data / name]
        if candidate.exists()
    }
    if data_hash in external_hashes:
        raise ValueError("Training dataset hash matches an external acceptance dataset")
    return {
        "training_dataset_path": str(path),
        "training_dataset_sha256": data_hash,
        "dataset_role": metadata.get("dataset_role", "development"),
        "external_acceptance_filenames_forbidden": sorted(EXTERNAL_ACCEPTANCE_FILENAMES),
        "external_acceptance_hash_overlap": False,
    }


def train_candidate(
    input_path: str | Path,
    config: ProjectConfig,
    models_root: str | Path,
    metrics_path: str | Path | None = None,
    *,
    data_domain: str = "accelerated_mock",
) -> dict[str, Any]:
    input_guard = validate_training_input(input_path)
    feature_names, examples, scan = build_training_examples(input_path, config, models_root)
    required_lifecycles = max(
        config.ml.minimum_candidate_lifecycles,
        config.ml.minimum_completed_lifecycles_for_training,
    )
    if len(examples) < required_lifecycles:
        report = {
            "status": "insufficient_lifecycles",
            "message": f"Dataset has only {len(examples)} completed lifecycle(s); minimum required is {required_lifecycles}.",
            "data_domain": data_domain,
            **scan, "leakage_checks": input_guard,
        }
        _write_report(metrics_path, report)
        return report

    training, validation, test = split_lifecycles(examples, config)
    split_groups = {
        "training": [example.lifecycle_id for example in training],
        "validation": [example.lifecycle_id for example in validation],
        "test": [example.lifecycle_id for example in test],
    }
    split_representation = _split_representation(
        {"training": training, "validation": validation, "test": test}, config
    )
    if len(validation) < config.ml.minimum_validation_lifecycles or len(test) < config.ml.minimum_test_lifecycles:
        report = {
            "status": "insufficient_split_lifecycles",
            "message": (
                f"Lifecycle split has {len(validation)} validation and {len(test)} test lifecycle(s); "
                f"minimums are {config.ml.minimum_validation_lifecycles} and {config.ml.minimum_test_lifecycles}."
            ),
            "data_domain": data_domain, **scan, **split_groups, "leakage_checks": input_guard,
        }
        _write_report(metrics_path, report)
        return report
    if any(set(split_groups[left]) & set(split_groups[right]) for left, right in (("training", "validation"), ("training", "test"), ("validation", "test"))):
        raise RuntimeError("Lifecycle split groups overlap")

    models: dict[str, Any] = {}
    targets_meta: dict[str, Any] = {}
    probability_meta: dict[str, Any] = {}
    baseline_meta: dict[str, Any] = {}
    maximum = config.ml.maximum_training_rows_per_lifecycle
    for target in ("warning", "critical"):
        train_x, train_y, train_ids, _, train_times = _target_dataset(training, target, maximum)
        validation_x, validation_y, validation_ids, validation_baseline, validation_times = _target_dataset(validation, target, maximum)
        test_x, test_y, test_ids, test_baseline, test_times = _target_dataset(test, target, maximum)
        if not len(train_y) or not len(validation_y) or not len(test_y):
            continue
        weights = lifecycle_sample_weights(train_ids) if config.ml.sample_weighting_enabled else None
        regression_feature_indices = predictive_feature_indices(feature_names)
        estimator = _regressor(config)
        estimator.fit(train_x[:, regression_feature_indices], train_y, sample_weight=weights)
        model = FeatureSelectedRegressor(estimator, regression_feature_indices)
        validation_prediction = model.predict(validation_x)
        test_prediction = model.predict(test_x)
        validation_metrics = regression_metrics(validation_y, validation_prediction, validation_ids)
        test_metrics = regression_metrics(test_y, test_prediction, test_ids)
        validation_metrics["duration_bucket_evaluation"] = duration_bucket_evaluation(
            validation_y, validation_prediction, validation_ids,
            {value.lifecycle_id: value.duration_hours for value in validation},
            config.duration_bucket_boundaries_hours,
        )
        test_metrics["duration_bucket_evaluation"] = duration_bucket_evaluation(
            test_y, test_prediction, test_ids,
            {value.lifecycle_id: value.duration_hours for value in test},
            config.duration_bucket_boundaries_hours,
        )
        validation_candidate_paired, validation_baseline_metrics = _paired_baseline_metrics(
            validation_y, validation_prediction, validation_baseline, validation_ids
        )
        test_candidate_paired, test_baseline_metrics = _paired_baseline_metrics(
            test_y, test_prediction, test_baseline, test_ids
        )
        registry_target = f"time_to_{target}"
        models[registry_target] = model
        targets_meta[registry_target] = {
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "baseline_metrics": {
                "validation": validation_baseline_metrics,
                "test": test_baseline_metrics,
            },
            "baseline_comparison_candidate_metrics": {
                "validation": validation_candidate_paired,
                "test": test_candidate_paired,
            },
            "training_sample_count": int(len(train_y)),
            "validation_sample_count": int(len(validation_y)),
            "test_sample_count": int(len(test_y)),
            "training_lifecycle_ids": sorted(set(train_ids)),
            "validation_lifecycle_ids": sorted(set(validation_ids)),
            "test_lifecycle_ids": sorted(set(test_ids)),
            "minimum_training_target_hours": float(np.min(train_y)),
            "minimum_validation_target_hours": float(np.min(validation_y)),
            "minimum_test_target_hours": float(np.min(test_y)),
            "all_samples_pre_event": bool(all(value > 0 for value in train_y) and all(value > 0 for value in validation_y) and all(value > 0 for value in test_y)),
            "primary_promotion_metric": "macro_lifecycle_mae_hours",
            "sample_weight_total_by_lifecycle": {
                lifecycle_id: float(np.sum(weights[np.asarray(train_ids) == lifecycle_id])) if weights is not None else float(train_ids.count(lifecycle_id))
                for lifecycle_id in sorted(set(train_ids))
            },
        }
        baseline_meta[registry_target] = test_baseline_metrics

        for horizon in config.ml.forecast_horizons_hours:
            probability_target = f"probability_{target}_{horizon}h"
            p_train_x, train_labels, p_train_ids, _, train_lead_hours = _probability_dataset(training, target, horizon)
            p_validation_x, validation_labels_all, p_validation_ids_all, _, validation_lead_hours_all = _probability_dataset(validation, target, horizon)
            p_test_x, test_labels, p_test_ids, _, test_lead_hours = _probability_dataset(test, target, horizon)
            if not len(train_labels) or not len(validation_labels_all) or not len(test_labels):
                continue
            feature_indices = probability_feature_indices(feature_names)
            residual_pairs = probability_residual_feature_pairs(feature_names)
            standardized_specs = probability_standardized_residual_specs(feature_names, p_train_x)
            trend_specs = probability_trend_snr_specs(feature_names, p_train_x)
            derived_feature_names = (
                probability_residual_feature_names(feature_names, residual_pairs)
                + probability_standardized_residual_feature_names(feature_names, standardized_specs)
                + probability_trend_snr_feature_names(feature_names, trend_specs)
            )
            probability_weights = (
                probability_sample_weights(
                    train_labels,
                    p_train_ids,
                    train_lead_hours,
                    horizon,
                    config.ml.probability_early_positive_weight_multiplier,
                    config.ml.probability_positive_class_balance_strength,
                    config.ml.probability_boundary_negative_weight_multiplier,
                )
                if config.ml.sample_weighting_enabled else None
            )
            hard_mining_oof: np.ndarray | None = None
            hard_mining_report: dict[str, Any] = {
                "enabled": False,
                "reason": "not_required_target",
            }
            if (
                config.ml.probability_hard_negative_mining_enabled
                and probability_target in config.ml.required_probability_targets
                and len(np.unique(train_labels)) == 2
            ):
                miner = ExtraTreesClassifier(
                    n_estimators=140,
                    min_samples_leaf=16,
                    max_features=0.35,
                    max_depth=18,
                    n_jobs=2,
                    random_state=config.ml.random_state + 101,
                    class_weight=None,
                )
                hard_mining_oof = _cross_fitted_probability_scores(
                    miner,
                    p_train_x,
                    train_labels,
                    p_train_ids,
                    probability_weights,
                    feature_indices,
                    residual_pairs,
                    standardized_specs,
                    trend_specs,
                    config.ml.probability_hard_example_cv_folds,
                )
                if hard_mining_oof is not None:
                    hard_multipliers, mined = probability_hard_example_multipliers(
                        train_labels,
                        hard_mining_oof,
                        train_lead_hours,
                        horizon,
                        config.ml.probability_hard_negative_quantile,
                        config.ml.probability_hard_negative_weight_multiplier,
                        config.ml.probability_hard_early_positive_quantile,
                        config.ml.probability_hard_early_positive_weight_multiplier,
                    )
                    if probability_weights is None:
                        probability_weights = hard_multipliers
                    else:
                        probability_weights = probability_weights * hard_multipliers
                    probability_weights = probability_weights * (
                        len(probability_weights) / max(float(np.sum(probability_weights)), 1e-12)
                    )
                    hard_mining_report = {
                        "enabled": True,
                        "source": "training_lifecycle_group_kfold",
                        "folds": min(
                            config.ml.probability_hard_example_cv_folds,
                            len(set(p_train_ids)),
                        ),
                        "oof_roc_auc": float(roc_auc_score(train_labels, hard_mining_oof)),
                        "oof_pr_auc": float(average_precision_score(train_labels, hard_mining_oof)),
                        **mined,
                    }
                else:
                    hard_mining_report = {
                        "enabled": False,
                        "reason": "insufficient_training_groups_for_cross_fitted_scores",
                    }
            calibration_mask, selection_mask, calibration_ids = split_probability_validation_lifecycles(
                validation_labels_all,
                p_validation_ids_all,
                config.ml.minimum_validation_lifecycles_for_calibration,
            )
            split_calibration = bool(np.any(calibration_mask))
            validation_labels = validation_labels_all[selection_mask]
            validation_ids = [value for value, keep in zip(p_validation_ids_all, selection_mask) if keep]
            validation_x_selected = p_validation_x[selection_mask]
            validation_lead_hours = validation_lead_hours_all[selection_mask]
            candidate_comparison: list[dict[str, Any]] = []
            fitted_candidates: list[tuple[tuple[Any, ...], CalibratedFeatureClassifier, dict[str, Any], np.ndarray]] = []
            estimators = (
                [("constant", ConstantProbabilityClassifier(float(np.mean(train_labels))))]
                if len(np.unique(train_labels)) < 2 else _probability_estimators(config)
            )
            for estimator_name, estimator in estimators:
                base_wrapper = CalibratedFeatureClassifier(
                    estimator,
                    feature_indices,
                    residual_pairs=residual_pairs,
                    standardized_residual_specs=standardized_specs,
                    trend_snr_specs=trend_specs,
                )
                if not isinstance(estimator, ConstantProbabilityClassifier):
                    estimator.fit(base_wrapper.transform(p_train_x), train_labels, sample_weight=probability_weights)
                raw_validation_all = base_wrapper.predict_proba(p_validation_x)[:, 1]
                calibration_results: list[tuple[float, str, Any | None, str]] = []
                methods = ("none", "sigmoid", "isotonic") if split_calibration else ("none",)
                for calibration_method in methods:
                    calibrator = (
                        _fit_calibrator(calibration_method, raw_validation_all[calibration_mask], validation_labels_all[calibration_mask])
                        if split_calibration else None
                    )
                    wrapper = CalibratedFeatureClassifier(
                        estimator,
                        feature_indices,
                        calibration_method,
                        calibrator,
                        residual_pairs,
                        standardized_specs,
                        trend_specs,
                    )
                    selection_probability = wrapper.predict_proba(validation_x_selected)[:, 1]
                    calibration_results.append((
                        float(brier_score_loss(validation_labels, selection_probability)),
                        calibration_method,
                        calibrator,
                        "validation_lifecycle_calibration" if split_calibration and calibration_method != "none" else "none",
                    ))
                if (
                    not split_calibration
                    and estimator_name == "ExtraTreesStableClassifier"
                    and hard_mining_oof is not None
                ):
                    oof_calibrator = _fit_calibrator("sigmoid", hard_mining_oof, train_labels)
                    oof_wrapper = CalibratedFeatureClassifier(
                        estimator,
                        feature_indices,
                        "sigmoid",
                        oof_calibrator,
                        residual_pairs,
                        standardized_specs,
                        trend_specs,
                    )
                    oof_selection_probability = oof_wrapper.predict_proba(validation_x_selected)[:, 1]
                    calibration_results.append((
                        float(brier_score_loss(validation_labels, oof_selection_probability)),
                        "sigmoid",
                        oof_calibrator,
                        "training_lifecycle_oof",
                    ))
                _, calibration_method, calibrator, calibration_source = min(
                    calibration_results,
                    key=lambda value: (value[0], ("none", "sigmoid", "isotonic").index(value[1])),
                )
                wrapper = CalibratedFeatureClassifier(
                    estimator,
                    feature_indices,
                    calibration_method,
                    calibrator,
                    residual_pairs,
                    standardized_specs,
                    trend_specs,
                )
                candidate_probability = wrapper.predict_proba(validation_x_selected)[:, 1]
                selection_fn_ceiling = (
                    config.ml.maximum_probability_false_negative_rate
                    * config.ml.probability_validation_fn_safety_factor
                )
                candidate_threshold = _select_probability_threshold(
                    validation_labels, candidate_probability, config.ml.probability_classification_threshold,
                    selection_fn_ceiling, validation_ids,
                    config.ml.maximum_probability_validation_false_positive_rate,
                )
                threshold = candidate_threshold["selected_threshold"]
                candidate_metrics = probability_metrics(validation_labels, candidate_probability, threshold)
                candidate_event = _event_detection_metrics(validation_labels, candidate_probability, threshold, validation_ids)
                candidate_lead = _lead_time_detection_metrics(
                    validation_labels, candidate_probability, threshold, validation_lead_hours, horizon
                )
                event_fn = candidate_event["event_false_negative_rate"]
                early_fn = candidate_lead["early_half"]["false_negative_rate"]
                if candidate_threshold["eligible"]:
                    score = (
                        0,
                        float(early_fn if early_fn is not None else 1.0),
                        candidate_metrics["false_negative_rate"],
                        candidate_metrics["false_positive_rate"],
                        candidate_metrics["brier_score"],
                        float(event_fn if event_fn is not None else 1.0),
                        estimator_name,
                    )
                else:
                    # Never prefer an alarm-everywhere failed candidate merely
                    # because it has zero FN.  Prefer the estimator/threshold
                    # closest to satisfying the joint FN/FP policy.
                    score = (
                        1,
                        float(candidate_threshold.get("fallback_constraint_violation_score") or 1e9),
                        candidate_metrics["false_positive_rate"],
                        candidate_metrics["false_negative_rate"],
                        float(early_fn if early_fn is not None else 1.0),
                        candidate_metrics["brier_score"],
                        estimator_name,
                    )
                comparison = {
                    "estimator": estimator_name, "calibration": calibration_method,
                    "calibration_source": calibration_source,
                    "selected_on": "validation_threshold_subset", "selected_threshold": candidate_threshold,
                    "validation_metrics": candidate_metrics,
                    "validation_event_metrics": candidate_event,
                    "validation_lead_time_metrics": candidate_lead,
                }
                candidate_comparison.append(comparison)
                fitted_candidates.append((score, wrapper, comparison, candidate_probability))
            _, probability_model, selected_candidate, validation_probability = min(fitted_candidates, key=lambda value: value[0])
            threshold_selection = selected_candidate["selected_threshold"]
            selected_threshold = threshold_selection["selected_threshold"]
            test_probability = probability_model.predict_proba(p_test_x)[:, 1]
            validation_probability_metrics = probability_metrics(validation_labels, validation_probability, selected_threshold)
            test_probability_metrics = probability_metrics(test_labels, test_probability, selected_threshold)
            validation_probability_metrics["event_metrics"] = _event_detection_metrics(
                validation_labels, validation_probability, selected_threshold, validation_ids
            )
            test_probability_metrics["event_metrics"] = _event_detection_metrics(
                test_labels, test_probability, selected_threshold, p_test_ids
            )
            validation_probability_metrics["lead_time_metrics"] = _lead_time_detection_metrics(
                validation_labels, validation_probability, selected_threshold, validation_lead_hours, horizon
            )
            test_probability_metrics["lead_time_metrics"] = _lead_time_detection_metrics(
                test_labels, test_probability, selected_threshold, test_lead_hours, horizon
            )
            validation_probability_metrics["group_metrics"] = _probability_group_metrics(
                validation_labels, validation_probability, selected_threshold, validation_ids,
                validation, config,
            )
            test_probability_metrics["group_metrics"] = _probability_group_metrics(
                test_labels, test_probability, selected_threshold, p_test_ids, test, config,
            )
            split_rates = {
                "training": float(np.mean(train_labels)), "validation": float(np.mean(validation_labels)),
                "test": float(np.mean(test_labels)),
            }
            class_counts = {
                "training": len(np.unique(train_labels)), "validation": len(np.unique(validation_labels)),
                "test": len(np.unique(test_labels)),
            }
            eligible, initial_reason = probability_target_eligibility(split_rates, class_counts, config)
            reasons = [] if initial_reason is None else [initial_reason]
            if not threshold_selection["eligible"]:
                reasons.append(threshold_selection["reason"])
            validation_fn_met = validation_probability_metrics["false_negative_rate"] <= config.ml.maximum_probability_false_negative_rate
            test_fn_met = test_probability_metrics["false_negative_rate"] <= config.ml.maximum_probability_false_negative_rate
            if not validation_fn_met:
                reasons.append("validation_fn_ceiling_not_met")
            if not test_fn_met:
                reasons.append("test_fn_ceiling_not_met")
            training_event_rate = float(np.mean(train_labels))
            probability_baselines = {
                "validation": probability_metrics(validation_labels, np.full(len(validation_labels), training_event_rate), selected_threshold),
                "test": probability_metrics(test_labels, np.full(len(test_labels), training_event_rate), selected_threshold),
            }
            beats_probability_baseline = (
                float(brier_score_loss(validation_labels, validation_probability)) < probability_baselines["validation"]["brier_score"]
                and float(brier_score_loss(test_labels, test_probability)) < probability_baselines["test"]["brier_score"]
            )
            if not beats_probability_baseline:
                reasons.append("does_not_beat_constant_rate_baseline")
            if (
                validation_probability_metrics["calibration_error"] > config.ml.maximum_probability_calibration_error
                or test_probability_metrics["calibration_error"] > config.ml.maximum_probability_calibration_error
            ):
                reasons.append("calibration_error_exceeds_limit")
            if selected_candidate["estimator"] == "constant":
                reasons.append("constant_training_target")
            reasons = sorted(set(reason for reason in reasons if reason))
            eligible = not reasons
            reason = reasons[0] if reasons else None
            models[probability_target] = probability_model
            probability_meta[probability_target] = {
                "metadata_schema_version": MODEL_METADATA_SCHEMA_VERSION,
                "forecast_policy_contract_version": FORECAST_POLICY_CONTRACT_VERSION,
                "forecast_policy_version": FORECAST_POLICY_VERSION,
                "target": target, "horizon_hours": horizon,
                "validation_metrics": validation_probability_metrics, "test_metrics": test_probability_metrics,
                "baseline_metrics": probability_baselines, "beats_constant_rate_baseline": beats_probability_baseline,
                "event_rates": split_rates,
                "event_rate_limits": {"minimum": config.ml.minimum_probability_positive_rate, "maximum": config.ml.maximum_probability_positive_rate},
                "maximum_calibration_error": config.ml.maximum_probability_calibration_error,
                "eligible": eligible, "reason": reason,
                "constant_model": selected_candidate["estimator"] == "constant",
                "estimator": selected_candidate["estimator"], "calibration_method": selected_candidate["calibration"],
                "calibration_source": selected_candidate.get("calibration_source", "none"),
                "model_selection_comparison": candidate_comparison,
                "probability_feature_policy_version": PROBABILITY_FEATURE_POLICY_VERSION,
                "predictive_feature_count": (
                    len(feature_indices)
                    + len(residual_pairs)
                    + len(standardized_specs)
                    + len(trend_specs)
                ),
                "predictive_feature_names": [feature_names[index] for index in feature_indices] + derived_feature_names,
                "direct_predictive_feature_names": [feature_names[index] for index in feature_indices],
                "derived_predictive_feature_names": derived_feature_names,
                "derived_feature_source_names": sorted({
                    feature_names[index] for pair in residual_pairs for index in pair
                } | {
                    feature_names[index]
                    for left, right, scale, _ in standardized_specs
                    for index in (left, right, scale)
                } | {
                    feature_names[index]
                    for slope, scale, _, _ in trend_specs
                    for index in (slope, scale)
                }),
                "audit_only_feature_names": [
                    name for index, name in enumerate(feature_names)
                    if index not in set(feature_indices)
                    | {source for pair in residual_pairs for source in pair}
                    | {source for left, right, scale, _ in standardized_specs for source in (left, right, scale)}
                    | {source for slope, scale, _, _ in trend_specs for source in (slope, scale)}
                ],
                "early_positive_weight_multiplier": config.ml.probability_early_positive_weight_multiplier,
                "positive_class_balance_strength": config.ml.probability_positive_class_balance_strength,
                "boundary_negative_weight_multiplier": config.ml.probability_boundary_negative_weight_multiplier,
                "hard_example_mining": hard_mining_report,
                "threshold_selection_fn_ceiling": selection_fn_ceiling,
                "threshold_selection_fp_ceiling": config.ml.maximum_probability_validation_false_positive_rate,
                "production_fn_ceiling": config.ml.maximum_probability_false_negative_rate,
                "training_lifecycle_ids": sorted(set(p_train_ids)),
                "validation_lifecycle_ids": sorted(set(validation_ids)),
                "calibration_lifecycle_ids": sorted(calibration_ids),
                "threshold_selection_lifecycle_ids": sorted(set(validation_ids)),
                "calibration_threshold_lifecycle_overlap": sorted(calibration_ids & set(validation_ids)),
                "test_lifecycle_ids": sorted(set(p_test_ids)),
                "test_used_for_estimator_calibration_or_threshold_selection": False,
                "threshold_selected_on": "validation_lifecycles",
                "selected_threshold": selected_threshold, "threshold_selection": threshold_selection,
                "validation_threshold_selected": selected_threshold,
                "validation_fp_rate": validation_probability_metrics["false_positive_rate"],
                "validation_fn_rate": validation_probability_metrics["false_negative_rate"],
                "validation_fn_ceiling_met": validation_fn_met,
                "validation_fn_ceiling": config.ml.maximum_probability_false_negative_rate,
                "test_fp_rate": test_probability_metrics["false_positive_rate"],
                "test_fn_rate": test_probability_metrics["false_negative_rate"],
                "test_fn_ceiling_met": test_fn_met, "test_fn_ceiling": config.ml.maximum_probability_false_negative_rate,
                "baseline_passed": beats_probability_baseline,
                "calibration_passed": "calibration_error_exceeds_limit" not in reasons,
                "target_required": probability_target in config.ml.required_probability_targets,
                "target_eligible": eligible, "target_ineligibility_reasons": reasons, "trained_available": True,
                "training_sample_count": int(len(train_labels)), "validation_sample_count": int(len(validation_labels)),
                "test_sample_count": int(len(test_labels)),
                "training_positive_count": int(np.sum(train_labels == 1)), "training_negative_count": int(np.sum(train_labels == 0)),
                "validation_positive_count": int(np.sum(validation_labels == 1)), "validation_negative_count": int(np.sum(validation_labels == 0)),
                "test_positive_count": int(np.sum(test_labels == 1)), "test_negative_count": int(np.sum(test_labels == 0)),
                "feature_schema": list(feature_names), "data_domain": data_domain,
                "label_timing_evidence": {
                    "label_definition": f"0 < event onset - current timestamp <= {horizon} elapsed hours",
                    "elapsed_time_not_row_count": True, "cross_lifecycle_labels_forbidden": True,
                    "post_event_samples_in_training": 0, "event_onset_excluded": True,
                    "event_free_lifecycles_included_as_negatives": True, "all_samples_pre_event": True,
                },
            }

    if not targets_meta:
        report = {"status": "insufficient_target_lifecycles", **scan, **split_groups}
        _write_report(metrics_path, report)
        return report

    all_training_rows = [row for example in training for row in example.feature_rows]
    feature_array = np.asarray(all_training_rows, dtype=float)
    timestamp = datetime.now(timezone.utc)
    version = timestamp.strftime("ml_%Y%m%dT%H%M%SZ")
    target_support = {
        target: {
            "minimum": float(data["minimum_training_target_hours"]),
            "p50": None, "p90": None, "p95": None,
            "maximum": float(max(
                _target_dataset(training, target.removeprefix("time_to_"), maximum)[1]
            )),
        }
        for target, data in targets_meta.items()
    }
    for target in target_support:
        values = _target_dataset(training, target.removeprefix("time_to_"), maximum)[1]
        target_support[target].update(
            p50=float(np.quantile(values, .50)), p90=float(np.quantile(values, .90)),
            p95=float(np.quantile(values, .95)),
        )
    durations = np.asarray([value.duration_hours for value in training], dtype=float)
    sampling_intervals = [
        (right - left).total_seconds()
        for example in training for left, right in zip(example.timestamps, example.timestamps[1:])
        if right > left
    ]
    sidecar = Path(input_path).with_name(f"{Path(input_path).stem}_metadata.json")
    generator_metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    required_targets = [*config.ml.required_eta_targets, *config.ml.required_probability_targets]
    eta_target_evidence = {}
    for target, detail in targets_meta.items():
        validation_candidate = detail["baseline_comparison_candidate_metrics"]["validation"]
        test_candidate = detail["baseline_comparison_candidate_metrics"]["test"]
        validation_baseline = detail["baseline_metrics"]["validation"]
        test_baseline = detail["baseline_metrics"]["test"]
        baseline_passed = bool(
            validation_candidate["macro_lifecycle_mae_hours"] < validation_baseline["macro_lifecycle_mae_hours"]
            and test_candidate["macro_lifecycle_mae_hours"] < test_baseline["macro_lifecycle_mae_hours"]
        )
        support = target_support[target]
        eta_reasons = []
        if not detail["all_samples_pre_event"]:
            eta_reasons.append("post_event_training_sample_detected")
        if not baseline_passed:
            eta_reasons.append("does_not_beat_statistical_baseline")
        eta_target_evidence[target] = {
            "target": target,
            "target_required": target in config.ml.required_eta_targets,
            "trained_available": True,
            "target_eligible": not eta_reasons,
            "target_ineligibility_reasons": eta_reasons,
            "model_loaded": True,
            "model_schema_compatible": True,
            "feature_schema_compatible": True,
            "physical_validation_passed": bool(detail["all_samples_pre_event"]),
            "baseline_passed": baseline_passed,
            "support_min_hours": support["minimum"],
            "support_max_hours": support["maximum"],
            "training_sample_count": detail["training_sample_count"],
            "validation_sample_count": detail["validation_sample_count"],
            "test_sample_count": detail["test_sample_count"],
            "training_lifecycle_ids": detail["training_lifecycle_ids"],
            "validation_lifecycle_ids": detail["validation_lifecycle_ids"],
            "test_lifecycle_ids": detail["test_lifecycle_ids"],
            "data_domain": data_domain,
            "feature_schema": list(feature_names),
        }

    metadata: dict[str, Any] = {
        "model_metadata_schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "forecast_policy_contract_version": FORECAST_POLICY_CONTRACT_VERSION,
        "forecast_policy_version": FORECAST_POLICY_VERSION,
        "forecast_policy_contract": {
            **CONTRACT.to_dict(),
            "required_eta_targets": list(config.ml.required_eta_targets),
            "required_probability_targets": list(config.ml.required_probability_targets),
            "required_targets": list(required_targets),
        },
        "model_version": version,
        "training_timestamp": timestamp.isoformat(),
        "data_domain": data_domain,
        "model_stage": candidate_stage(data_domain),
        "production_eligible": data_domain == "plant" and all(
            (probability_meta.get(target, {}).get("eligible") is True if target.startswith("probability_")
             else eta_target_evidence.get(target, {}).get("target_eligible") is True)
            for target in required_targets
        ),
        "sampling_interval_seconds": int(scan["raw_sampling_interval_distribution_seconds"]["p50"]) if scan["raw_sampling_interval_distribution_seconds"]["p50"] is not None else None,
        "training_lifecycle_count": len(training),
        "minimum_training_lifecycle_hours": float(np.min(durations)),
        "median_training_lifecycle_hours": float(np.median(durations)),
        "maximum_training_lifecycle_hours": float(np.max(durations)),
        "lifecycle_duration_distribution_hours": {
            "minimum": float(np.min(durations)), "p50": float(np.quantile(durations, .5)),
            "p90": float(np.quantile(durations, .9)), "p95": float(np.quantile(durations, .95)),
            "maximum": float(np.max(durations)),
        },
        "sampling_interval_distribution_seconds": scan["raw_sampling_interval_distribution_seconds"],
        "target_support": target_support,
        "dataset_hashes": [file_sha256(input_path)],
        "generator_version": generator_metadata.get("generator_version"),
        "training_lifecycle_ids": split_groups["training"],
        "validation_lifecycle_ids": split_groups["validation"],
        "test_lifecycle_ids": split_groups["test"],
        "split_random_seed": config.ml.split_random_seed,
        "split_strategy": "deterministic_representation_aware_lifecycle_search",
        "split_representation": split_representation,
        "feature_names": feature_names,
        "feature_order": feature_names,
        "feature_role_evidence": {
            "predictive_features": [feature_names[index] for index in predictive_feature_indices(feature_names)],
            "audit_only_features": [
                name for index, name in enumerate(feature_names)
                if index not in predictive_feature_indices(feature_names)
            ],
            "absolute_lifecycle_age_is_predictive_input": False,
            "rationale": "Absolute age remains available to OOD/support checks but cannot drive probability or ETA recommendations.",
        },
        "probability_feature_role_evidence": {
            "policy_version": PROBABILITY_FEATURE_POLICY_VERSION,
            "direct_predictive_features": [feature_names[index] for index in probability_feature_indices(feature_names)],
            "derived_predictive_features": probability_residual_feature_names(
                feature_names, probability_residual_feature_pairs(feature_names)
            ),
            "excluded_features": [
                name for index, name in enumerate(feature_names)
                if index not in (
                    set(probability_feature_indices(feature_names))
                    | {source for pair in probability_residual_feature_pairs(feature_names) for source in pair}
                )
            ],
            "absolute_level_features_are_direct_predictive_inputs": False,
            "adaptive_relative_state_features_are_predictive_inputs": True,
            "variability_features_are_predictive_inputs": False,
            "lifecycle_accumulation_is_predictive_input": False,
            "rationale": (
                "Probability models combine causal rate/change evidence with causal deviations "
                "from longer local rolling baselines. Raw absolute level remains excluded; "
                "manufacturer rules remain authoritative for absolute thresholds."
            ),
        },
        "feature_distribution": {
            "lower_quantile_01": np.quantile(feature_array, 0.01, axis=0).tolist(),
            "upper_quantile_99": np.quantile(feature_array, 0.99, axis=0).tolist(),
        },
        "model_parameters": {
            "model_type": config.ml.model_type,
            "max_iter": config.ml.max_iter,
            "learning_rate": config.ml.learning_rate,
            "max_leaf_nodes": config.ml.max_leaf_nodes,
            "l2_regularization": config.ml.l2_regularization,
            "random_state": config.ml.random_state,
        },
        "sampling_strategy": "deterministic_lifecycle_stage_linspace",
        "maximum_samples_per_lifecycle": maximum,
        "sample_weighting_enabled": config.ml.sample_weighting_enabled,
        "sampling_random_seed": config.ml.sampling_random_seed,
        "probability_validation_fn_safety_factor": config.ml.probability_validation_fn_safety_factor,
        "maximum_probability_validation_false_positive_rate": config.ml.maximum_probability_validation_false_positive_rate,
        "probability_early_positive_weight_multiplier": config.ml.probability_early_positive_weight_multiplier,
        "probability_positive_class_balance_strength": config.ml.probability_positive_class_balance_strength,
        "probability_boundary_negative_weight_multiplier": config.ml.probability_boundary_negative_weight_multiplier,
        "probability_hard_negative_mining_enabled": config.ml.probability_hard_negative_mining_enabled,
        "probability_hard_negative_quantile": config.ml.probability_hard_negative_quantile,
        "probability_hard_negative_weight_multiplier": config.ml.probability_hard_negative_weight_multiplier,
        "probability_hard_early_positive_quantile": config.ml.probability_hard_early_positive_quantile,
        "probability_hard_early_positive_weight_multiplier": config.ml.probability_hard_early_positive_weight_multiplier,
        "probability_hard_example_cv_folds": config.ml.probability_hard_example_cv_folds,
        "minimum_validation_lifecycles_for_calibration": config.ml.minimum_validation_lifecycles_for_calibration,
        "lifecycle_split_candidate_attempts": config.ml.lifecycle_split_candidate_attempts,
        "targets": targets_meta,
        "probability_targets": probability_meta,
        "probability_thresholds": {"schema_version": "1.0", "targets": {
            name: {**data["threshold_selection"], "eligible": bool(data["eligible"]), "reason": data["reason"]}
            for name, data in probability_meta.items()
        }},
        "required_model_targets": list(required_targets),
        "artifact_targets": {name: name for name in ["time_to_warning", "time_to_critical", *probability_meta]},
        "eta_target_evidence": eta_target_evidence,
        "baseline_metrics": baseline_meta,
        "input_schema_version": config.ml.schema_version,
        "threshold_config_version": config.threshold_config_version,
        "lifecycle_config_version": config.lifecycle_config_version,
        "maturity_stage": "candidate",
        "synthetic_data_only": data_domain != "plant",
        "training_environment": runtime_environment(),
        "leakage_checks": {
            **input_guard,
            "lifecycle_split_overlap": False,
            "calibration_uses_test_data": False,
            "threshold_selection_uses_test_data": False,
            "external_acceptance_metrics_read_during_selection": False,
            "probability_feature_policy_version": PROBABILITY_FEATURE_POLICY_VERSION,
        },
        **scan,
    }
    registry = ModelRegistry(models_root)
    registry.save_candidate(models, metadata)
    report = {"status": "candidate_trained", "candidate_path": str(registry.candidate.resolve()), **metadata}
    _write_report(metrics_path, report)
    return report


def _primary_metric(metadata: dict[str, Any] | None, target: str, split: str = "test") -> float | None:
    try:
        return float(metadata["targets"][target][f"{split}_metrics"]["macro_lifecycle_mae_hours"])
    except (KeyError, TypeError, ValueError):
        try:
            return float(metadata["targets"][target]["validation_metrics"]["mae_hours"])
        except (KeyError, TypeError, ValueError):
            return None


def evaluate_candidate(
    config: ProjectConfig, models_root: str | Path, *, promote: bool = False
) -> dict[str, Any]:
    registry = ModelRegistry(models_root)
    candidate = registry.load_metadata("candidate")
    production = registry.load_metadata("production")
    if candidate is None:
        report = {"status": "no_candidate", "promoted": False, "targets": {}}
        registry.record_audit("candidate_evaluated", report)
        return report
    compatibility = environment_compatibility(candidate.get("training_environment"))
    refusal = (
        promotion_refusal(str(candidate.get("data_domain", "accelerated_mock")))
        if promote and Path(models_root).name.lower() in {"mock", "realistic", "plant"}
        else None
    )
    if refusal:
        report = {
            "status": "promotion_refused", "promoted": False, "promoted_targets": [],
            "targets": {}, "probability_targets": {}, "promotion_refusal_reason": refusal,
            "data_domain": candidate.get("data_domain", "accelerated_mock"),
            "training_environment": candidate.get("training_environment"),
            "runtime_environment": compatibility["runtime_environment"],
            "environment_compatibility": compatibility,
        }
        registry.record_audit("promotion_refused", report)
        return report

    decisions: dict[str, Any] = {}
    eligible_targets: list[str] = []
    for target, target_data in candidate.get("targets", {}).items():
        candidate_test = _primary_metric(candidate, target, "test")
        candidate_validation = _primary_metric(candidate, target, "validation")
        production_test = _primary_metric(production, target, "test")
        baseline = target_data.get("baseline_metrics", {})
        baseline_validation = (baseline.get("validation") or {}).get("macro_lifecycle_mae_hours") if isinstance(baseline, dict) else None
        baseline_test = (baseline.get("test") or {}).get("macro_lifecycle_mae_hours") if isinstance(baseline, dict) else None
        paired = target_data.get("baseline_comparison_candidate_metrics", {})
        paired_validation = (paired.get("validation") or {}).get("macro_lifecycle_mae_hours") if isinstance(paired, dict) else None
        paired_test = (paired.get("test") or {}).get("macro_lifecycle_mae_hours") if isinstance(paired, dict) else None
        legacy_baseline = candidate.get("baseline_metrics", {}).get(target)
        if baseline_validation is None and isinstance(legacy_baseline, dict):
            baseline_validation = legacy_baseline.get("macro_lifecycle_mae_hours", legacy_baseline.get("mae_hours"))
        if baseline_test is None:
            baseline_test = baseline_validation
        target_lifecycle_count = len(target_data.get("training_lifecycle_ids", [])) or int(candidate.get("usable_lifecycles", 0))
        count_eligible = target_lifecycle_count >= config.ml.minimum_deployment_lifecycles
        comparison_validation = paired_validation if paired_validation is not None else candidate_validation
        comparison_test = paired_test if paired_test is not None else candidate_test
        beats_validation = baseline_validation is not None and comparison_validation is not None and comparison_validation < float(baseline_validation)
        beats_test = baseline_test is not None and comparison_test is not None and comparison_test < float(baseline_test)
        beats_production = production_test is None or (candidate_test is not None and candidate_test < production_test)
        eligible = bool(count_eligible and beats_validation and beats_test and beats_production)
        if eligible:
            eligible_targets.append(target)
        decisions[target] = {
            "eligible": eligible,
            "recommended_source": "ml" if eligible else "statistical",
            "primary_metric": "macro_lifecycle_mae_hours",
            "candidate_validation_macro_lifecycle_mae_hours": candidate_validation,
            "candidate_test_macro_lifecycle_mae_hours": candidate_test,
            "baseline_validation_macro_lifecycle_mae_hours": baseline_validation,
            "baseline_test_macro_lifecycle_mae_hours": baseline_test,
            "paired_candidate_validation_macro_lifecycle_mae_hours": paired_validation,
            "paired_candidate_test_macro_lifecycle_mae_hours": paired_test,
            "production_test_macro_lifecycle_mae_hours": production_test,
            "minimum_lifecycle_count_satisfied": count_eligible,
            "beats_validation_baseline": beats_validation,
            "beats_test_baseline": beats_test,
            "better_than_baseline": bool(beats_validation and beats_test),
            "beats_production": beats_production,
        }

    probability_decisions: dict[str, Any] = {}
    eligible_probability_targets: list[str] = []
    for target, target_data in candidate.get("probability_targets", {}).items():
        validation_calibration = (target_data.get("validation_metrics") or {}).get("calibration_error")
        test_calibration = (target_data.get("test_metrics") or {}).get("calibration_error")
        calibration_ok = bool(
            validation_calibration is not None
            and test_calibration is not None
            and float(validation_calibration) <= config.ml.maximum_probability_calibration_error
            and float(test_calibration) <= config.ml.maximum_probability_calibration_error
        )
        eligible = bool(target_data.get("eligible") and calibration_ok)
        reason = target_data.get("reason")
        if not calibration_ok and reason is None:
            reason = "calibration_error_exceeds_limit"
        probability_decisions[target] = {
            "eligible": eligible,
            "reason": reason,
            "event_rates": target_data.get("event_rates"),
            "validation_metrics": target_data.get("validation_metrics"),
            "test_metrics": target_data.get("test_metrics"),
        }
        if eligible:
            eligible_probability_targets.append(target)

    promoted_targets: list[str] = []
    archived_targets: list[str] = []
    if promote and (eligible_targets or eligible_probability_targets):
        promotion = registry.promote_targets(
            [*eligible_targets, *eligible_probability_targets],
            maximum_fn_rate=config.ml.maximum_probability_false_negative_rate,
        )
        promoted_targets = promotion["promoted_targets"]
        archived_targets = promotion["archived_targets"]

    eligible_count = len(eligible_targets)
    if promote:
        status = "partially_promoted" if promoted_targets and eligible_count < len(decisions) else "promoted" if promoted_targets else "not_promoted"
    else:
        status = "partially_eligible" if 0 < eligible_count < len(decisions) else "eligible" if eligible_count else "not_eligible"
    for target, decision in decisions.items():
        decision["promoted"] = target in promoted_targets
        decision["active_source"] = "ml" if target in promoted_targets else (
            (production or {}).get("forecast_sources", {}).get(target, {}).get("source", "statistical")
        )
    for target, decision in probability_decisions.items():
        decision["promoted"] = target in promoted_targets
    report = {
        "status": status,
        "promoted": bool(promoted_targets),
        "promoted_targets": promoted_targets,
        "archived_targets": archived_targets,
        "targets": decisions,
        "probability_targets": probability_decisions,
        "partial_promotion_available": 0 < eligible_count < len(decisions),
        "lifecycle_count_eligible": int(candidate.get("usable_lifecycles", 0)) >= config.ml.minimum_deployment_lifecycles,
        "minimum_deployment_lifecycles": config.ml.minimum_deployment_lifecycles,
        "synthetic_data_only": bool(candidate.get("synthetic_data_only", False)),
        "training_environment": candidate.get("training_environment"),
        "runtime_environment": compatibility["runtime_environment"],
        "environment_compatibility": compatibility,
    }
    registry.record_audit("candidate_evaluated", {**report, "promotion_requested": promote})
    return report
