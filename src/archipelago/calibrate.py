from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _subsample(matrix: np.ndarray, size: int, seed: int) -> np.ndarray:
    n = matrix.shape[0]
    if n <= size:
        return matrix
    rng = np.random.default_rng(seed)
    return matrix[rng.choice(n, size=size, replace=False)]


def pair_distance(matrix: np.ndarray, sample_size: int, percentile: float, seed: int, pairs: int = 20000) -> float:
    """Percentile of distances between random pairs. The global spread."""
    if matrix.shape[0] < 2:
        return 1.0
    sample = _subsample(matrix, sample_size, seed)
    rng = np.random.default_rng(seed)
    take = sample.shape[0]
    left = rng.integers(0, take, size=pairs)
    right = rng.integers(0, take, size=pairs)
    keep = left != right
    if not keep.any():
        return 1.0
    distances = np.linalg.norm(sample[left[keep]] - sample[right[keep]], axis=1)
    positive = distances[distances > 0]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def neighbour_distance(matrix: np.ndarray, sample_size: int, percentile: float, seed: int) -> float:
    """Percentile of nearest-neighbour distances. The local density scale."""
    if matrix.shape[0] < 2:
        return 1.0
    sample = _subsample(matrix, sample_size, seed)
    if sample.shape[0] < 2:
        return 1.0
    # k=2 because a point's own nearest neighbour is itself.
    distances, _ = NearestNeighbors(n_neighbors=2).fit(sample).kneighbors(sample)
    nearest = distances[:, 1]
    positive = nearest[nearest > 0]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0
