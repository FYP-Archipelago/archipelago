from __future__ import annotations

from typing import Any

import numpy as np

from ..config import (
    CALIBRATION_SAMPLE_SIZE,
    HAMMING_BIT_MULTIPLIER,
    LSH_WIDTH_PERCENTILE,
    NUM_PERM,
    Config,
)
from ..encoding.features import FeatureSet, Metric
from ..encoding.hashing import MinHasher
from .calibrate import pair_distance


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


# Streaming LSH: the same hash families and generator order as LSHStage, keyed per row as rows
# arrive. The bucket width is fixed once (at warm-up), since keys hashed with different widths
# can never collide.
class OnlineLSH:
    name = "lsh"

    def __init__(self, config: Config, metric: Metric, dimension: int, width: float | None):
        rng = np.random.default_rng(config.seed)
        self.metric = metric
        self.count = config.lsh_hashes
        self.width = width
        if metric is Metric.L2:
            assert width is not None
            self.projections = rng.standard_normal((dimension, self.count))
            self.offsets = rng.uniform(0.0, width, size=self.count)
        elif metric is Metric.HAMMING:
            self.positions = rng.integers(0, max(dimension, 1), size=self.count * HAMMING_BIT_MULTIPLIER)
        else:
            # Same generator as the batch MinHasher; only the first lsh_hashes permutations are
            # used, so only those are computed.
            hasher = MinHasher(max(NUM_PERM, self.count), config.seed)
            self.a = hasher.a[: self.count].reshape(1, -1)
            self.b = hasher.b[: self.count].reshape(1, -1)
            self.prime = np.uint64(MinHasher.MERSENNE_PRIME)
        self.labels: dict[bytes, int] = {}
        self.sizes: list[int] = []

    def keys(self, matrix: np.ndarray, sets: list[np.ndarray] | None) -> np.ndarray:
        if self.metric is Metric.L2:
            # einsum (not BLAS) so each row's dot products don't depend on the batch shape.
            projected = np.einsum("nd,dc->nc", matrix, self.projections)
            return np.floor((projected + self.offsets) / self.width).astype(np.int64)
        if self.metric is Metric.HAMMING:
            return matrix[:, self.positions].astype(np.int64)
        assert sets is not None
        keys = np.full((len(sets), self.count), self.prime, dtype=np.uint64)
        for row, ids in enumerate(sets):
            if ids.size:
                keys[row] = ((ids.astype(np.uint64).reshape(-1, 1) * self.a + self.b) % self.prime).min(axis=0)
        return keys.astype(np.int64)

    def assign(self, key: np.ndarray) -> int:
        label = self.labels.setdefault(key.tobytes(), len(self.labels))
        if label == len(self.sizes):
            self.sizes.append(0)
        self.sizes[label] += 1
        return label

    # A repeat of a genome already in the given block.
    def absorb(self, label: int) -> None:
        self.sizes[label] += 1

    def diagnostics(self) -> dict[str, Any]:
        sizes = np.array(self.sizes) if self.sizes else np.zeros(1, dtype=np.int64)
        out: dict[str, Any] = {"family": {Metric.L2: "p_stable", Metric.HAMMING: "bit_sampling"}.get(self.metric, "minhash")}
        if self.metric is Metric.L2:
            out["bucket_width"] = self.width
        out.update(
            {
                "blocks": len(self.sizes),
                "largest_block": int(sizes.max()),
                "mean_block_size": float(sizes.mean()),
                "median_block_size": float(np.median(sizes)),
                "singleton_blocks": int((sizes == 1).sum()),
                "rows_hashed": int(sizes.sum()),
                "clusters_after": len(self.sizes),
            }
        )
        return out
