from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import Birch

from ...config import Config
from ...types import FeatureSet
from ..calibrate import neighbour_distance

MIN_BLOCK_TO_FIT = 3


class BirchStage:
    name = "birch"

    def __init__(self, config: Config):
        self.config = config
        self.settings = config.birch
        self._diagnostics: dict[str, Any] = {}

    def _threshold(self, features: FeatureSet, rows: np.ndarray) -> float:
        if self.settings.threshold != "auto":
            return float(self.settings.threshold)
        return max(
            neighbour_distance(
                features.matrix[rows],
                self.config.calibration.sample_size,
                self.config.calibration.percentile,
                self.config.seed,
            ),
            1e-12,
        )

    def fit_predict(self, features: FeatureSet, groups: np.ndarray) -> np.ndarray:
        rows = features.usable
        labels = np.arange(len(features), dtype=np.int64)
        if rows.size == 0:
            return labels

        threshold = self._threshold(features, rows)
        self._diagnostics["threshold"] = threshold
        self._diagnostics["branching_factor"] = self.settings.branching_factor

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
                branching_factor=self.settings.branching_factor,
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
                "subclusters": next_label,
                "mean_subclusters_per_fitted_block": (
                    float(np.mean(subcluster_counts)) if subcluster_counts else 0.0
                ),
                "rows_clustered": int(rows.size),
            }
        )
        return labels

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics
