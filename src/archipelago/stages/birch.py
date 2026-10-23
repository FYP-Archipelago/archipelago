from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import Birch

from ..config import (
    BIRCH_BRANCHING_FACTOR,
    CALIBRATION_SAMPLE_SIZE,
    NEIGHBOUR_PERCENTILE,
    Config,
)
from ..encoding.features import FeatureSet
from .calibrate import neighbour_distance
from .cf import nearest, radius_if_added

MIN_BLOCK_TO_FIT = 3


class BatchBirch:
    name = "birch"

    def __init__(self, config: Config):
        self.config = config
        self._diagnostics: dict[str, Any] = {}

    def _threshold(self, features: FeatureSet, rows: np.ndarray) -> float:
        if self.config.birch_threshold is not None:
            return float(self.config.birch_threshold)
        measured = max(
            neighbour_distance(
                features.matrix[rows],
                CALIBRATION_SAMPLE_SIZE,
                NEIGHBOUR_PERCENTILE,
                self.config.seed,
            ),
            1e-12,
        )
        return measured / self.config.detail

    def fit_predict(self, features: FeatureSet, groups: np.ndarray) -> np.ndarray:
        rows = features.usable
        labels = np.arange(len(features), dtype=np.int64)
        if rows.size == 0:
            return labels

        threshold = self._threshold(features, rows)
        self._diagnostics["threshold"] = threshold

        next_label = 0
        fitted = skipped = 0
        subcluster_counts: list[int] = []
        order = np.argsort(groups[rows], kind="stable")
        ordered_rows = rows[order]
        ordered_groups = groups[ordered_rows]
        boundaries = np.flatnonzero(np.diff(ordered_groups)) + 1

        for block in np.split(ordered_rows, boundaries):
            if block.size < MIN_BLOCK_TO_FIT:
                labels[block] = next_label
                next_label += 1
                skipped += 1
                continue
            model = Birch(
                threshold=threshold,
                branching_factor=BIRCH_BRANCHING_FACTOR,
                n_clusters=None,
            )
            local = model.fit_predict(features.matrix[block])
            _, local = np.unique(local, return_inverse=True)
            labels[block] = next_label + local.ravel()
            next_label += int(local.max()) + 1
            fitted += 1
            subcluster_counts.append(int(local.max()) + 1)

        self._diagnostics.update(
            {
                "blocks_fitted": fitted,
                "blocks_too_small_to_fit": skipped,
                "mean_subclusters_per_fitted_block": (
                    float(np.mean(subcluster_counts)) if subcluster_counts else 0.0
                ),
                "rows_clustered": int(rows.size),
            }
        )
        return labels

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics


# One upstream group's subclusters as cluster features: count, linear sum, square sum.
@dataclass
class _Subclusters:
    labels: list[int] = field(default_factory=list)
    count: np.ndarray = field(default_factory=lambda: np.zeros(0))
    linear: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    square: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centre: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    norm: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def add(self, index: int, point: np.ndarray) -> None:
        self.count[index] += 1.0
        self.linear[index] += point
        self.square[index] += float(point @ point)
        self.centre[index] = self.linear[index] / self.count[index]
        self.norm[index] = float(self.centre[index] @ self.centre[index])

    def create(self, point: np.ndarray, label: int) -> None:
        size = len(self.labels)
        if size == self.count.shape[0]:
            capacity = max(8, 2 * size)
            for name in ("count", "square", "norm"):
                grown = np.zeros(capacity)
                grown[:size] = getattr(self, name)[:size]
                setattr(self, name, grown)
            for name in ("linear", "centre"):
                grown = np.zeros((capacity, point.size))
                if size:
                    grown[:size] = getattr(self, name)[:size]
                setattr(self, name, grown)
        self.labels.append(label)
        self.linear[size] = 0.0
        self.square[size] = 0.0
        self.add(size, point)


# Streaming BIRCH: a row joins the nearest subcluster of its upstream group (LSH block, or one
# group without LSH) if the subcluster's radius stays within the threshold, else starts a new
# one. This is BIRCH's leaf absorb rule; the CF tree itself only speeds up the nearest search.
class StreamBirch:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.thresholds = [threshold]
        self.groups: dict[int, _Subclusters] = {}
        self.where: list[tuple[int, int]] = []

    def insert(self, point: np.ndarray, group: int) -> int:
        subclusters = self.groups.setdefault(group, _Subclusters())
        size = len(subclusters.labels)
        if size:
            index = nearest(subclusters.centre[:size], subclusters.norm[:size], point)
            radius = radius_if_added(
                float(subclusters.count[index]), subclusters.linear[index], float(subclusters.square[index]), point, 1.0
            )
            if radius <= self.threshold:
                subclusters.add(index, point)
                return subclusters.labels[index]
        label = len(self.where)
        subclusters.create(point, label)
        self.where.append((group, size))
        return label

    # A repeat of a genome already in this subcluster: added without the radius test.
    def absorb(self, point: np.ndarray, label: int) -> None:
        group, index = self.where[label]
        self.groups[group].add(index, point)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self.thresholds.append(threshold)

    def diagnostics(self) -> dict[str, Any]:
        per_group = [len(g.labels) for g in self.groups.values()]
        return {
            "threshold": self.threshold,
            "threshold_checkpoints": list(self.thresholds),
            "groups": len(per_group),
            "subclusters": len(self.where),
            "mean_subclusters_per_group": float(np.mean(per_group)) if per_group else 0.0,
            "rows_clustered": int(sum(float(g.count[: len(g.labels)].sum()) for g in self.groups.values())),
            "clusters_after": len(self.where),
        }
