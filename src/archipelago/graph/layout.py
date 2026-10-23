from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from ..config import LAYOUT_BY_METRIC, MDS_LANDMARKS
from ..encoding.features import Metric
from ..models import LayoutInfo

COMPONENTS = 3


# A fitted projection that can place rows it wasn't fitted on: raw components, then the
# min-max scaling of the rows it was fitted on.
@dataclass(frozen=True)
class Projection:
    method: str
    raw: Callable[[np.ndarray], np.ndarray]
    low: np.ndarray
    span: np.ndarray
    variance_kept_2d: float
    variance_kept_3d: float


def _pad(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] >= COMPONENTS:
        return matrix[:, :COMPONENTS]
    padded = np.zeros((matrix.shape[0], COMPONENTS))
    padded[:, : matrix.shape[1]] = matrix
    return padded


def _scale(coords: np.ndarray, low: np.ndarray, span: np.ndarray) -> np.ndarray:
    safe = np.where(span > 0, span, 1.0)
    return np.where(span > 0, (coords - low) / safe * 2.0 - 1.0, 0.0)


def _pca(matrix: np.ndarray, seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], list[float]]:
    components = min(COMPONENTS, matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=components, random_state=seed).fit(matrix)
    return lambda rows: _pad(model.transform(rows)), [float(v) for v in model.explained_variance_ratio_]


# Landmark (classical) MDS: embed a random sample exactly, then place every row relative to it.
# On Euclidean feature distances, classical MDS of the landmarks is PCA of the landmarks
# (Gower), so it is solved with an SVD of the centred landmarks: the same embedding (up to axis
# sign) and eigenvalue ratios, without the landmarks x landmarks eigenproblem, and placing a
# row is a projection instead of distances to every landmark.
def _mds(matrix: np.ndarray, seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    take = min(MDS_LANDMARKS, matrix.shape[0])
    index = rng.choice(matrix.shape[0], size=take, replace=False) if take < matrix.shape[0] else np.arange(take)
    anchors = matrix[index]
    mean = anchors.mean(axis=0)
    _, singular, rotation = np.linalg.svd(anchors - mean, full_matrices=False)
    values = singular**2
    total = float(values.sum())
    variance = [float(v / total) for v in values[:COMPONENTS]] if total > 0 else []
    basis = rotation[:COMPONENTS].T

    def raw(rows: np.ndarray) -> np.ndarray:
        return _pad((rows - mean) @ basis)

    return raw, variance


# Fit a projection on these rows (PCA or MDS, picked from the metric). Returns it with the rows'
# scaled coordinates.
def fit(matrix: np.ndarray, metric: Metric, seed: int) -> tuple[Projection, np.ndarray]:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        zeros = np.zeros(COMPONENTS)
        projection = Projection("none", lambda rows: np.zeros((rows.shape[0], COMPONENTS)), zeros, zeros, 0.0, 0.0)
        return projection, np.zeros((matrix.shape[0], COMPONENTS))

    method = LAYOUT_BY_METRIC[metric.value]
    raw, variance = _mds(matrix, seed) if method == "mds" else _pca(matrix, seed)
    coords = raw(matrix)
    low = coords.min(axis=0)
    span = coords.max(axis=0) - low
    projection = Projection(
        method=method,
        raw=raw,
        low=low,
        span=span,
        variance_kept_2d=round(sum(variance[:2]), 6),
        variance_kept_3d=round(sum(variance[:3]), 6),
    )
    return projection, _scale(coords, low, span)


# Place rows the projection wasn't fitted on; they can fall outside the fitted range, so clip.
def transform(projection: Projection, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, COMPONENTS))
    return np.clip(_scale(projection.raw(matrix), projection.low, projection.span), -1.0, 1.0)


# None when no projection was ever fitted (no usable rows).
def layout_info(projection: Projection | None) -> LayoutInfo:
    if projection is None:
        return LayoutInfo(method="none", variance_kept_2d=0.0, variance_kept_3d=0.0)
    return LayoutInfo(
        method=projection.method,
        variance_kept_2d=projection.variance_kept_2d,
        variance_kept_3d=projection.variance_kept_3d,
    )
