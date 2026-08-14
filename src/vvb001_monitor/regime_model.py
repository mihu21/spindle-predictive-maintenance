from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


STATUSES = ("NORMAL", "WARNING", "CRITICAL")


def _softmax_negative_distances(distances: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -distances / max(float(temperature), 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass
class LearnedRegimeModel:
    """Lifecycle-aware unsupervised three-regime model.

    K-means discovers operating regimes without NORMAL/WARNING/CRITICAL labels. Early-lifecycle
    samples act only as a healthy anchor. The discovered healthy cluster is the one with the
    greatest early-anchor purity; the other clusters are ordered by their median lifecycle
    position (with centroid distance as a deterministic tie-breaker).
    """

    seed: int = 42
    n_clusters: int = 3

    def fit(
        self,
        X: np.ndarray,
        lifecycle_progress: np.ndarray,
        baseline_mask: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "LearnedRegimeModel":
        if self.n_clusters != 3:
            raise ValueError("This runtime currently maps exactly three learned regimes to NORMAL/WARNING/CRITICAL")
        self.imputer = SimpleImputer(strategy="median")
        Xi = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (len(X),):
                raise ValueError("sample_weight must have one value per training row")
            if np.any(~np.isfinite(sample_weight)) or np.any(sample_weight <= 0):
                raise ValueError("sample_weight values must be finite and positive")
            self.scaler.fit(Xi, sample_weight=sample_weight)
            Xs = self.scaler.transform(Xi)
        else:
            Xs = self.scaler.fit_transform(Xi)
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            batch_size=4096,
            n_init=10,
            reassignment_ratio=0.01,
        )
        clusters = self.clusterer.fit_predict(Xs, sample_weight=sample_weight)
        if not np.any(baseline_mask):
            raise ValueError("At least one early-lifecycle baseline-anchor sample is required")

        baseline_scores: list[tuple[float, int]] = []
        for cluster in range(self.n_clusters):
            members = clusters == cluster
            member_count = max(1, int(np.sum(members)))
            anchored = int(np.sum(members & baseline_mask))
            # Prefer clusters that contain a large share of anchor data, while preventing a tiny
            # cluster from winning solely because every one of its few rows is early-lifecycle.
            score = anchored / member_count + anchored / max(1, int(np.sum(baseline_mask)))
            baseline_scores.append((score, cluster))
        healthy_cluster = max(baseline_scores)[1]

        healthy_centroid = self.clusterer.cluster_centers_[healthy_cluster]
        remaining = [c for c in range(self.n_clusters) if c != healthy_cluster]
        ranking: list[tuple[float, float, int]] = []
        for cluster in remaining:
            members = clusters == cluster
            median_progress = float(np.median(lifecycle_progress[members])) if np.any(members) else 1.0
            centroid_distance = float(np.linalg.norm(self.clusterer.cluster_centers_[cluster] - healthy_centroid))
            ranking.append((median_progress, centroid_distance, cluster))
        ranking.sort()
        ordered = [healthy_cluster, ranking[0][2], ranking[1][2]]
        self.cluster_to_status = {cluster: status for cluster, status in zip(ordered, STATUSES)}
        self.status_to_cluster = {status: cluster for cluster, status in self.cluster_to_status.items()}

        assigned_dist = np.min(self.clusterer.transform(Xs), axis=1)
        self.temperature = float(np.median(assigned_dist))
        if not np.isfinite(self.temperature) or self.temperature <= 1e-6:
            self.temperature = 1.0

        train_probs = self.predict_proba(X)
        train_scores = train_probs[:, 1] * 0.5 + train_probs[:, 2]
        status_medians: dict[str, float] = {}
        mapped_status = np.asarray([self.cluster_to_status[int(c)] for c in clusters], dtype=object)
        for status in STATUSES:
            members = mapped_status == status
            status_medians[status] = float(np.median(train_scores[members])) if np.any(members) else {"NORMAL": 0.0, "WARNING": 0.5, "CRITICAL": 1.0}[status]
        ordered_medians = [status_medians[s] for s in STATUSES]
        # Guard against pathological overlap while keeping boundaries data-derived.
        ordered_medians = list(np.maximum.accumulate(np.asarray(ordered_medians, dtype=float)))
        self.score_boundaries = (
            float((ordered_medians[0] + ordered_medians[1]) / 2.0),
            float((ordered_medians[1] + ordered_medians[2]) / 2.0),
        )
        self.score_boundary_calibration = {
            "method": "cluster_midpoints",
            "warning_boundary": self.score_boundaries[0],
            "critical_boundary": self.score_boundaries[1],
        }
        self.training_cluster_stats = {}
        for cluster in range(self.n_clusters):
            members = clusters == cluster
            self.training_cluster_stats[str(cluster)] = {
                "status": self.cluster_to_status[cluster],
                "rows": int(np.sum(members)),
                "median_lifecycle_progress": float(np.median(lifecycle_progress[members])) if np.any(members) else None,
                "baseline_anchor_fraction": float(np.mean(baseline_mask[members])) if np.any(members) else None,
                "distance_from_healthy_centroid": float(np.linalg.norm(self.clusterer.cluster_centers_[cluster] - healthy_centroid)),
                "median_degradation_score": float(np.median(train_scores[members])) if np.any(members) else None,
            }
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        clusters = self.clusterer.predict(self._transform(X))
        return np.asarray([self.cluster_to_status[int(c)] for c in clusters], dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = self.clusterer.transform(self._transform(X))
        cluster_probs = _softmax_negative_distances(distances, self.temperature)
        out = np.zeros((len(X), 3), dtype=float)
        for status_index, status in enumerate(STATUSES):
            out[:, status_index] = cluster_probs[:, self.status_to_cluster[status]]
        out /= np.sum(out, axis=1, keepdims=True)
        return out

    def degradation_score(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs[:, 1] * 0.5 + probs[:, 2]

    def set_score_boundaries(
        self,
        warning_boundary: float,
        critical_boundary: float,
        *,
        calibration: dict[str, Any] | None = None,
    ) -> None:
        warning = float(warning_boundary)
        critical = float(critical_boundary)
        if not (0.0 <= warning < critical <= 1.0):
            raise ValueError("score boundaries must satisfy 0 <= warning < critical <= 1")
        self.score_boundaries = (warning, critical)
        if calibration is not None:
            self.score_boundary_calibration = dict(calibration)

    def status_from_score(self, score: float, previous_status: str | None = None, hysteresis: float = 0.0) -> str:
        warning_boundary, critical_boundary = self.score_boundaries
        h = max(0.0, float(hysteresis))
        if previous_status == "NORMAL":
            if score >= critical_boundary + h:
                return "CRITICAL"
            if score >= warning_boundary + h:
                return "WARNING"
            return "NORMAL"
        if previous_status == "WARNING":
            if score >= critical_boundary + h:
                return "CRITICAL"
            if score < warning_boundary - h:
                return "NORMAL"
            return "WARNING"
        if previous_status == "CRITICAL":
            if score < warning_boundary - h:
                return "NORMAL"
            if score < critical_boundary - h:
                return "WARNING"
            return "CRITICAL"
        if score < warning_boundary:
            return "NORMAL"
        if score < critical_boundary:
            return "WARNING"
        return "CRITICAL"

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(STATUSES, dtype=object)

    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "standardized_minibatch_kmeans_lifecycle_ordered",
            "n_clusters": self.n_clusters,
            "temperature": self.temperature,
            "score_boundaries": list(self.score_boundaries),
            "score_boundary_calibration": getattr(self, "score_boundary_calibration", None),
            "cluster_to_status": {str(k): v for k, v in self.cluster_to_status.items()},
            "training_cluster_stats": self.training_cluster_stats,
        }
