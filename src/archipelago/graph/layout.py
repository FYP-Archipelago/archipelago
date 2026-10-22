from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from ..config import LAYOUT_BY_METRIC, MDS_LANDMARKS
from ..encoding.features import Metric

COMPONENTS = 3
# Rows transformed per batch by MDS; a full rows x landmarks distance matrix won't fit in memory.
MDS_CHUNK = 5000


# One projection of the run's usable rows into [-1, 1]^3.
@dataclass(frozen=True)
class Layout:
    method: str
    # Scaled position of every usable row, shape (rows, 3).
    coords: np.ndarray
    variance_kept_2d: float
    variance_kept_3d: float


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(
        (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0
    )


def _pad(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] >= COMPONENTS:
        return matrix[:, :COMPONENTS]
    padded = np.zeros((matrix.shape[0], COMPONENTS))
    padded[:, : matrix.shape[1]] = matrix
    return padded


def _scale(coords: np.ndarray) -> np.ndarray:
    low = coords.min(axis=0)
    span = coords.max(axis=0) - low
    safe = np.where(span > 0, span, 1.0)
    return np.where(span > 0, (coords - low) / safe * 2.0 - 1.0, 0.0)


def _pca(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, list[float]]:
    components = min(COMPONENTS, matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=components, random_state=seed).fit(matrix)
    return _pad(model.transform(matrix)), [float(v) for v in model.explained_variance_ratio_]


# Landmark (classical) MDS: embed a random sample exactly, then place every row from its
# squared distances to those landmarks.
def _mds(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    take = min(MDS_LANDMARKS, matrix.shape[0])
    index = rng.choice(matrix.shape[0], size=take, replace=False) if take < matrix.shape[0] else np.arange(take)
    anchors = matrix[index]

    squared = _squared_distances(anchors, anchors)
    row_means = squared.mean(axis=1)
    grand = row_means.mean()
    centred = -0.5 * (squared - row_means[:, None] - row_means[None, :] + grand)
    values, vectors = np.linalg.eigh(centred)
    order = np.argsort(values)[::-1][:COMPONENTS]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]

    total = float(np.clip(np.linalg.eigvalsh(centred), 0.0, None).sum())
    variance = [float(v / total) for v in values] if total > 0 else []
    scale = np.where(values > 0, 1.0 / np.sqrt(np.where(values > 0, values, 1.0)), 0.0)

    coords = np.empty((matrix.shape[0], COMPONENTS))
    for start in range(0, matrix.shape[0], MDS_CHUNK):
        chunk = _squared_distances(matrix[start : start + MDS_CHUNK], anchors)
        coords[start : start + MDS_CHUNK] = _pad(-0.5 * (chunk - row_means[None, :]) @ vectors * scale)
    return coords, variance


# Project the usable rows' feature matrix; PCA or MDS is picked from the metric.
def project(matrix: np.ndarray, metric: Metric, seed: int) -> Layout:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return Layout("none", np.zeros((matrix.shape[0], COMPONENTS)), 0.0, 0.0)

    method = LAYOUT_BY_METRIC[metric.value]
    coords, variance = _mds(matrix, seed) if method == "mds" else _pca(matrix, seed)
    return Layout(
        method=method,
        coords=_scale(coords),
        variance_kept_2d=round(sum(variance[:2]), 6),
        variance_kept_3d=round(sum(variance[:3]), 6),
    )
