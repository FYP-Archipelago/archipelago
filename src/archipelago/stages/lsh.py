from __future__ import annotations

from typing import Any

import numpy as np

from ..calibrate import pair_distance
from ..config import (
    CALIBRATION_SAMPLE_SIZE,
    HAMMING_BIT_MULTIPLIER,
    LSH_WIDTH_PERCENTILE,
    NUM_PERM,
    Config,
)
from ..features import FeatureSet, Metric
from ..hashing import MinHasher


class LSHStage:
    name = "lsh"

    def __init__(self, config: Config):
        self.config = config
        self._diagnostics: dict[str, Any] = {}

    def _hash_matrix(self, features: FeatureSet, rows: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        matrix = features.matrix[rows]
        count = self.config.lsh_hashes

        if features.metric is Metric.L2:
            width = pair_distance(
                matrix,
                CALIBRATION_SAMPLE_SIZE,
                LSH_WIDTH_PERCENTILE,
                self.config.seed,
            )
            width = max(float(width), 1e-12)
            self._diagnostics["family"] = "p_stable"
            self._diagnostics["bucket_width"] = width
            projections = rng.standard_normal((matrix.shape[1], count))
            offsets = rng.uniform(0.0, width, size=count)
            return np.floor((matrix @ projections + offsets) / width).astype(np.int64)

        if features.metric is Metric.HAMMING:
            bits = count * HAMMING_BIT_MULTIPLIER
            positions = rng.integers(0, max(matrix.shape[1], 1), size=bits)
            self._diagnostics["family"] = "bit_sampling"
            return matrix[:, positions].astype(np.int64)

        # MinHash over the element-id sets. P[collision] is exactly the Jaccard.
        hasher = MinHasher(max(NUM_PERM, count), self.config.seed)
        assert features.sets is not None
        signatures = np.empty((rows.size, count), dtype=np.int64)
        for out_row, source_row in enumerate(rows):
            signatures[out_row] = hasher.signature(features.sets[source_row])[
                :count
            ].astype(np.int64)
        self._diagnostics["family"] = "minhash"
        return signatures

    def fit_predict(self, features: FeatureSet, groups: np.ndarray) -> np.ndarray:
        rows = features.usable
        labels = np.arange(len(features), dtype=np.int64)
        if rows.size == 0:
            return labels

        hashes = self._hash_matrix(features, rows)
        keyed = np.column_stack([groups[rows], hashes])
        _, inverse = np.unique(keyed, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        labels[rows] = inverse

        sizes = np.bincount(inverse)
        self._diagnostics.update(
            {
                "blocks": int(sizes.size),
                "largest_block": int(sizes.max()),
                "mean_block_size": float(sizes.mean()),
                "median_block_size": float(np.median(sizes)),
                "singleton_blocks": int((sizes == 1).sum()),
                "rows_hashed": int(rows.size),
            }
        )
        return labels

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics
