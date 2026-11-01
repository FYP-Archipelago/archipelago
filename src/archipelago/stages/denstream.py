from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ..config import (
    CALIBRATION_SAMPLE_SIZE,
    HALF_LIFE_FRACTION,
    NEIGHBOUR_PERCENTILE,
    Config,
)
from ..encoding.features import FeatureSet
from .calibrate import neighbour_distance
from .cf import radius_if_added


class MicroCluster:
    __slots__ = ("cf1", "cf2", "created_at", "label", "weight")

    def __init__(self, point: np.ndarray, weight: float, now: float, label: int):
        self.weight = weight
        self.cf1 = point * weight
        self.cf2 = float(point @ point) * weight
        self.created_at = now
        self.label = label

    @property
    def centre(self) -> np.ndarray:
        return self.cf1 / self.weight

    def radius_if_added(self, point: np.ndarray, weight: float) -> float:
        return radius_if_added(self.weight, self.cf1, self.cf2, point, weight)

    def add(self, point: np.ndarray, weight: float) -> None:
        self.weight += weight
        self.cf1 += point * weight
        self.cf2 += float(point @ point) * weight


def prune_period_for(potential_threshold: float, lam: float) -> float:
    if potential_threshold > 1.0:
        return math.ceil(math.log2(potential_threshold / (potential_threshold - 1.0)) / lam)
    return math.inf


class BatchDenStream:
    name = "denstream"

    def __init__(self, config: Config):
        self.config = config
        self._diagnostics: dict[str, Any] = {}
        self.row_weights: np.ndarray | None = None

    # Clock-corrected t_wall, so points from different islands age along the same axis.
    @staticmethod
    def _clock(features: FeatureSet, rows: np.ndarray) -> np.ndarray:
        times = features.times[rows]
        return times - times.min()

    # Collapse the incoming partition into weighted points.
    def _summarise(self, features: FeatureSet, groups: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        epsilon = self.config.denstream_epsilon
        if epsilon is None:
            epsilon = (
                neighbour_distance(
                    points,
                    CALIBRATION_SAMPLE_SIZE,
                    NEIGHBOUR_PERCENTILE,
                    self.config.seed,
                )
                / self.config.detail
            )
        epsilon = max(float(epsilon), 1e-12)

        span = float(times.max() - times.min()) if times.size else 1.0
        half_life = self.config.denstream_half_life
        if half_life is None:
            half_life = max(span * HALF_LIFE_FRACTION, 1e-9)
        lam = 1.0 / float(half_life)

        mu, beta = self.config.denstream_mu, self.config.denstream_beta
        potential_threshold = beta * mu
        prune_period = prune_period_for(potential_threshold, lam)

        assignments, clusters, stats = self._stream(points, weights, times, epsilon, lam, potential_threshold, prune_period)
        self._macro_clusters(clusters, epsilon, mu, stats)

        point_labels = np.asarray(assignments, dtype=np.int64)
        labels[rows] = point_labels[row_to_point]

        final_weights = np.array([clusters[label].weight if label in clusters else 0.0 for label in point_labels])
        self.row_weights = np.zeros(len(features))
        self.row_weights[rows] = final_weights[row_to_point]

        self._diagnostics.update(
            {
                "epsilon": epsilon,
                "half_life": float(half_life),
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
                        merged_potential += int(candidate.weight * current_scale >= potential_threshold)
                        merged_outlier += int(candidate.weight * current_scale < potential_threshold)
                        target = candidate
                        break

            if target is None:
                target = MicroCluster(point, inflated, now, next_label)
                clusters[next_label] = target
                next_label += 1
                created += 1
                append(target)
            else:
                target.add(point, inflated)
                centres[position_of[target.label]] = target.centre

            assignments.append(target.label)

            if now >= next_prune:
                pruned += self._prune(clusters, now, reference, lam, potential_threshold, prune_period)
                reference = self._rescale(clusters, now, reference, lam)
                next_prune = now + prune_period
                compact()

        # Put every survivor's weight on the same footing: the final instant.
        final = float(times[-1]) if times.size else reference
        self._rescale(clusters, final, reference, lam)
        return (
            assignments,
            clusters,
            {
                "micro_clusters_created": created,
                "micro_clusters_pruned": pruned,
                "merges_into_potential": merged_potential,
                "merges_into_outlier": merged_outlier,
                "micro_clusters_surviving": len(clusters),
            },
        )

    # Drop faded clusters, using the decaying outlier threshold.
    @staticmethod
    def _prune(
        clusters: dict[int, MicroCluster],
        now: float,
        reference: float,
        lam: float,
        potential_threshold: float,
        prune_period: float,
    ) -> int:
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
    ) -> None:
        potential = [c for c in clusters.values() if c.weight >= mu]
        stats["potential_micro_clusters"] = len(potential)
        if not potential:
            return
        centres = np.stack([c.centre for c in potential])
        weights = np.array([c.weight for c in potential])
        # sklearn compares min_samples against total sample_weight but types it int.
        model = DBSCAN(eps=2.0 * epsilon, min_samples=max(1, round(mu)))
        found = model.fit_predict(centres, sample_weight=weights)
        stats["macro_clusters"] = len({f for f in found if f >= 0})

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics


# Streaming DenStream: the batch stream loop, one row at a time. Every row arrives with weight 1
# at its own clock time; micro-clusters fade and are pruned exactly as in batch, and a pruned
# label never comes back (a revisit starts a new micro-cluster).
class StreamDenStream:
    def __init__(self, config: Config, epsilon: float, half_life: float, dimension: int):
        self.mu = config.denstream_mu
        self.potential_threshold = config.denstream_beta * self.mu
        self.epsilon = epsilon
        self.half_life = half_life
        self.lam = 1.0 / half_life
        self.prune_period = prune_period_for(self.potential_threshold, self.lam)
        self.reference = 0.0
        self.next_prune = self.prune_period
        self.now = 0.0
        self.clusters: dict[int, MicroCluster] = {}
        self.next_label = 0
        self.centres = np.zeros((64, dimension))
        self.labels_at = np.zeros(64, dtype=np.int64)
        self.position_of: dict[int, int] = {}
        self.live = 0
        # The micro-cluster each upstream group last fed.
        self.fed: dict[int, int] = {}
        self.counters = dict.fromkeys(("micro_clusters_created", "micro_clusters_pruned", "merges_into_potential", "merges_into_outlier"), 0)
        self.input_points = 0
        self.epsilons = [epsilon]
        self.half_lives = [half_life]
        self.lambda_segments: list[tuple[float, float]] = [(0.0, self.lam)]

    def alive(self, label: int | None) -> bool:
        return label is not None and label in self.clusters

    def _append(self, cluster: MicroCluster) -> None:
        if self.live == self.centres.shape[0]:
            self.centres = np.vstack([self.centres, np.zeros_like(self.centres)])
            self.labels_at = np.concatenate([self.labels_at, np.zeros_like(self.labels_at)])
        self.centres[self.live] = cluster.centre
        self.labels_at[self.live] = cluster.label
        self.position_of[cluster.label] = self.live
        self.live += 1

    def _compact(self) -> None:
        self.position_of.clear()
        keep = 0
        for index in range(self.live):
            label = int(self.labels_at[index])
            if label in self.clusters:
                self.centres[keep] = self.clusters[label].centre
                self.labels_at[keep] = label
                self.position_of[label] = keep
                keep += 1
        self.live = keep

    def _search(self, point: np.ndarray, inflated: float) -> MicroCluster | None:
        if not self.live:
            return None
        distances = np.linalg.norm(self.centres[: self.live] - point, axis=1)
        # Only the nearest few can absorb the point without breaching epsilon.
        probe = min(8, self.live)
        nearest = np.argpartition(distances, probe - 1)[:probe]
        for position in nearest[np.lexsort((self.labels_at[nearest], distances[nearest]))]:
            candidate = self.clusters[int(self.labels_at[position])]
            if candidate.radius_if_added(point, inflated) <= self.epsilon:
                return candidate
        return None

    # Place one row. Returns its micro-cluster label, how it was placed (cache, group, search or
    # new), and the labels pruned right after it (None if no prune ran, [] if one removed nothing).
    def insert(self, point: np.ndarray, now: float, cached: int | None, group: int | None) -> tuple[int, str, list[int] | None]:
        self.now = now
        self.input_points += 1
        inflated = 2.0 ** (self.lam * (now - self.reference))
        scale = 2.0 ** (-self.lam * (now - self.reference))
        if self.alive(cached):
            target, how = self.clusters[cached], "cache"
        elif group is not None and self.alive(self.fed.get(group)):
            target, how = self.clusters[self.fed[group]], "group"
        else:
            target = self._search(point, inflated)
            how = "search" if target is not None else "new"

        if target is None:
            target = MicroCluster(point, inflated, now, self.next_label)
            self.clusters[self.next_label] = target
            self.next_label += 1
            self.counters["micro_clusters_created"] += 1
            self._append(target)
        else:
            potential = target.weight * scale >= self.potential_threshold
            self.counters["merges_into_potential" if potential else "merges_into_outlier"] += 1
            target.add(point, inflated)
            self.centres[self.position_of[target.label]] = target.centre
        if group is not None:
            self.fed[group] = target.label

        pruned = None
        if now >= self.next_prune:
            before = set(self.clusters)
            self.counters["micro_clusters_pruned"] += BatchDenStream._prune(
                self.clusters, now, self.reference, self.lam, self.potential_threshold, self.prune_period
            )
            pruned = sorted(before - set(self.clusters))
            self.reference = BatchDenStream._rescale(self.clusters, now, self.reference, self.lam)
            self.next_prune = now + self.prune_period
            self._compact()
        return target.label, how, pruned

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.epsilons.append(epsilon)

    # Weights so far were decayed with the old lambda; bring them to now first, then switch.
    def set_half_life(self, half_life: float) -> None:
        self.reference = BatchDenStream._rescale(self.clusters, self.now, self.reference, self.lam)
        self.half_life = half_life
        self.lam = 1.0 / half_life
        self.prune_period = prune_period_for(self.potential_threshold, self.lam)
        self.next_prune = self.now + self.prune_period
        self.half_lives.append(half_life)
        self.lambda_segments.append((self.now, self.lam))

    # Every live micro-cluster's weight at the latest clock time.
    def weights(self) -> dict[int, float]:
        scale = 2.0 ** (-self.lam * (self.now - self.reference))
        return {label: cluster.weight * scale for label, cluster in self.clusters.items()}

    def diagnostics(self, final: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "epsilon": self.epsilon,
            "epsilon_checkpoints": list(self.epsilons),
            "half_life": self.half_life,
            "half_life_checkpoints": list(self.half_lives),
            "time_span": self.now,
            "prune_period": None if self.prune_period == math.inf else int(self.prune_period),
            "input_points": self.input_points,
            **self.counters,
            "micro_clusters_surviving": len(self.clusters),
            "clusters_after": len(self.clusters),
        }
        if final:
            stats: dict[str, Any] = {"macro_clusters": 0}
            self.reference = BatchDenStream._rescale(self.clusters, self.now, self.reference, self.lam)
            BatchDenStream._macro_clusters(self.clusters, self.epsilon, self.mu, stats)
            out.update(stats)
        return out
