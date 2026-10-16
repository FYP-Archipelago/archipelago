
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ...config import Config
from ...types import FeatureSet
from ..calibrate import neighbour_distance


class MicroCluster:
    __slots__ = ("weight", "cf1", "cf2", "created_at", "last_at", "label")

    def __init__(self, point: np.ndarray, weight: float, now: float, label: int):
        self.weight = weight
        self.cf1 = point * weight
        self.cf2 = float(point @ point) * weight
        self.created_at = now
        self.last_at = now
        self.label = label

    @property
    def centre(self) -> np.ndarray:
        return self.cf1 / self.weight

    def radius_if_added(self, point: np.ndarray, weight: float) -> float:
        w = self.weight + weight
        cf1 = self.cf1 + point * weight
        cf2 = self.cf2 + float(point @ point) * weight
        centre = cf1 / w
        variance = cf2 / w - float(centre @ centre)
        return math.sqrt(max(variance, 0.0))

    def add(self, point: np.ndarray, weight: float, now: float) -> None:
        self.weight += weight
        self.cf1 += point * weight
        self.cf2 += float(point @ point) * weight
        self.last_at = now


class DenStreamStage:
    name = "denstream"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.denstream
        self._diagnostics: dict[str, Any] = {}
        self.row_weights: np.ndarray | None = None
        self.row_macro: np.ndarray | None = None

    def _clock(self, features: FeatureSet, rows: np.ndarray) -> np.ndarray:
        choice = self.settings.clock
        if choice == "arrival":
            return np.arange(rows.size, dtype=np.float64)
        if choice == "generation" and features.generations is not None:
            return features.generations[rows].astype(np.float64)
        if features.times is not None:
            times = features.times[rows]
            return times - times.min()
        return np.arange(rows.size, dtype=np.float64)

    def _summarise(self, features: FeatureSet, groups: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collapse the incoming partition into weighted points."""
        clock = self._clock(features, rows)
        keys = groups[rows]
        unique, inverse = np.unique(keys, return_inverse=True)
        inverse = inverse.ravel()
        if unique.size <= 1 or unique.size == rows.size:
            order = np.argsort(clock, kind="stable")
            remap = np.empty(rows.size, dtype=np.int64)
            remap[order] = np.arange(rows.size)
            return (
                features.matrix[rows][order],
                np.ones(rows.size),
                clock[order],
                remap,
            )

        dimension = features.matrix.shape[1]
        centroids = np.zeros((unique.size, dimension))
        weights = np.zeros(unique.size)
        np.add.at(centroids, inverse, features.matrix[rows])
        np.add.at(weights, inverse, 1.0)
        centroids /= weights[:, None]
        # Use each group's latest member - fading is about how recently a region was active.
        latest = np.zeros(unique.size)
        np.maximum.at(latest, inverse, clock)
        order = np.argsort(latest, kind="stable")
        remap = np.empty(unique.size, dtype=np.int64)
        remap[order] = np.arange(unique.size)
        return centroids[order], weights[order], latest[order], remap[inverse]

    def fit_predict(self, features: FeatureSet, groups: np.ndarray) -> np.ndarray:
        rows = features.usable
        labels = np.arange(len(features), dtype=np.int64)
        if rows.size == 0:
            return labels

        points, weights, times, row_to_point = self._summarise(features, groups, rows)
        epsilon = self.settings.epsilon
        if epsilon == "auto":
            epsilon = neighbour_distance(
                points,
                self.config.calibration.sample_size,
                self.config.calibration.percentile,
                self.config.seed,
            )
        epsilon = max(float(epsilon), 1e-12)

        span = float(times.max() - times.min()) if times.size else 1.0
        half_life = self.settings.half_life
        if half_life == "auto":
            half_life = max(span / 4.0, 1e-9)
        lam = 1.0 / float(half_life)

        mu, beta = self.settings.mu, self.settings.beta
        potential_threshold = beta * mu
        if potential_threshold > 1.0:
            prune_period = math.ceil(
                math.log2(potential_threshold / (potential_threshold - 1.0)) / lam
            )
        else:
            prune_period = math.inf

        assignments, clusters, stats = self._stream(
            points, weights, times, epsilon, lam, potential_threshold, prune_period
        )
        macro = self._macro_clusters(clusters, epsilon, mu, stats)

        point_labels = np.asarray(assignments, dtype=np.int64)
        labels[rows] = point_labels[row_to_point]

        final_weights = np.array(
            [
                clusters[label].weight if label in clusters else 0.0
                for label in point_labels
            ]
        )
        self.row_weights = np.zeros(len(features))
        self.row_weights[rows] = final_weights[row_to_point]
        self.row_macro = np.array(
            [macro.get(label, -1) for label in point_labels], dtype=np.int64
        )[row_to_point]

        self._diagnostics.update(
            {
                "epsilon": epsilon,
                "half_life": float(half_life),
                "lambda": lam,
                "clock": self.settings.clock,
                "time_span": span,
                "prune_period": None if prune_period == math.inf else int(prune_period),
                "input_points": int(points.shape[0]),
                "summarised": bool(points.shape[0] != rows.size),
                **stats,
            }
        )
        return labels

    def _stream(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        times: np.ndarray,
        epsilon: float,
        lam: float,
        potential_threshold: float,
        prune_period: float,
    ) -> tuple[list[int], dict[int, MicroCluster], dict[str, Any]]:
        clusters: dict[int, MicroCluster] = {}
        assignments: list[int] = []
        next_label = 0
        reference = float(times[0]) if times.size else 0.0
        next_prune = reference + prune_period
        merged_potential = merged_outlier = created = pruned = 0

        dimension = points.shape[1]
        centres = np.zeros((max(64, points.shape[0] // 8), dimension))
        labels_at = np.zeros(centres.shape[0], dtype=np.int64)
        position_of: dict[int, int] = {}
        live = 0

        def append(cluster: MicroCluster) -> None:
            nonlocal centres, labels_at, live
            if live == centres.shape[0]:
                centres = np.vstack([centres, np.zeros_like(centres)])
                labels_at = np.concatenate([labels_at, np.zeros_like(labels_at)])
            centres[live] = cluster.centre
            labels_at[live] = cluster.label
            position_of[cluster.label] = live
            live += 1

        def compact() -> None:
            nonlocal live
            position_of.clear()
            keep = 0
            for index in range(live):
                label = int(labels_at[index])
                if label in clusters:
                    centres[keep] = clusters[label].centre
                    labels_at[keep] = label
                    position_of[label] = keep
                    keep += 1
            live = keep

        for index in range(points.shape[0]):
            now = float(times[index])
            point = points[index]
            inflated = weights[index] * (2.0 ** (lam * (now - reference)))

            target = None
            if live:
                distances = np.linalg.norm(centres[:live] - point, axis=1)
                # Only the nearest few can absorb the point without breaching
                # epsilon, and argpartition beats sorting the whole set.
                probe = min(8, live)
                nearest = np.argpartition(distances, probe - 1)[:probe]
                current_scale = 2.0 ** (-lam * (now - reference))
                for candidate_position in nearest[np.argsort(distances[nearest])]:
                    candidate = clusters[int(labels_at[candidate_position])]
                    if candidate.radius_if_added(point, inflated) <= epsilon:
                        merged_potential += int(
                            candidate.weight * current_scale >= potential_threshold
                        )
                        merged_outlier += int(
                            candidate.weight * current_scale < potential_threshold
                        )
                        target = candidate
                        break

            if target is None:
                target = MicroCluster(point, inflated, now, next_label)
                clusters[next_label] = target
                next_label += 1
                created += 1
                append(target)
            else:
                target.add(point, inflated, now)
                centres[position_of[target.label]] = target.centre

            assignments.append(target.label)

            if now >= next_prune:
                pruned += self._prune(
                    clusters, now, reference, lam, potential_threshold, prune_period
                )
                reference = self._rescale(clusters, now, reference, lam)
                next_prune = now + prune_period
                compact()

        # Put every survivor's weight on the same footing: the final instant.
        final = float(times[-1]) if times.size else reference
        self._rescale(clusters, final, reference, lam)
        return assignments, clusters, {
            "micro_clusters_created": created,
            "micro_clusters_pruned": pruned,
            "merges_into_potential": merged_potential,
            "merges_into_outlier": merged_outlier,
            "micro_clusters_surviving": len(clusters),
        }

    @staticmethod
    def _prune(
        clusters: dict[int, MicroCluster],
        now: float,
        reference: float,
        lam: float,
        potential_threshold: float,
        prune_period: float,
    ) -> int:
        """Drop faded clusters, using the decaying outlier threshold."""
        scale = 2.0 ** (-lam * (now - reference))
        doomed = []
        for label, cluster in clusters.items():
            weight = cluster.weight * scale
            if weight >= potential_threshold:
                continue
            if prune_period == math.inf:
                cutoff = potential_threshold
            else:
                age = now - cluster.created_at
                numerator = 2.0 ** (-lam * (age + prune_period)) - 1.0
                denominator = 2.0 ** (-lam * prune_period) - 1.0
                cutoff = numerator / denominator if denominator else potential_threshold
            if weight < cutoff:
                doomed.append(label)
        for label in doomed:
            del clusters[label]
        return len(doomed)

    @staticmethod
    def _rescale(clusters: dict[int, MicroCluster], now: float, reference: float, lam: float) -> float:
        scale = 2.0 ** (-lam * (now - reference))
        for cluster in clusters.values():
            cluster.weight *= scale
            cluster.cf1 *= scale
            cluster.cf2 *= scale
        return now

    @staticmethod
    def _macro_clusters(
        clusters: dict[int, MicroCluster],
        epsilon: float,
        mu: float,
        stats: dict[str, Any],
    ) -> dict[int, int]:
        potential = [c for c in clusters.values() if c.weight >= mu]
        stats["potential_micro_clusters"] = len(potential)
        if not potential:
            return {}
        centres = np.stack([c.centre for c in potential])
        weights = np.array([c.weight for c in potential])
        # sklearn compares min_samples against total sample_weight but types it int.
        model = DBSCAN(eps=2.0 * epsilon, min_samples=max(1, int(round(mu))))
        found = model.fit_predict(centres, sample_weight=weights)
        stats["macro_clusters"] = int(len({f for f in found if f >= 0}))
        return {
            cluster.label: int(macro)
            for cluster, macro in zip(potential, found)
            if macro >= 0
        }

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics
