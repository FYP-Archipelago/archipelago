from __future__ import annotations

import numpy as np
import pytest

from archipelago.config import MDS_LANDMARKS
from archipelago.encoding.features import Metric
from archipelago.graph.layout import fit, transform


def matrix(rows: int, columns: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, columns)) * np.array([3.0, 2.0, 1.0, 0.5, 0.2, 0.1][:columns])


def aligned(a: np.ndarray, b: np.ndarray) -> bool:
    return all(np.allclose(a[:, k], b[:, k], atol=1e-9) or np.allclose(a[:, k], -b[:, k], atol=1e-9) for k in range(a.shape[1]))


def classical_mds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    centring = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centring @ d2 @ centring
    values, vectors = np.linalg.eigh(b)
    top = np.argsort(values)[::-1]
    return vectors[:, top[:3]] * np.sqrt(values[top[:3]]), values[top]


def minmax(coords: np.ndarray) -> np.ndarray:
    low, high = coords.min(axis=0), coords.max(axis=0)
    return (coords - low) / (high - low) * 2.0 - 1.0


@pytest.mark.parametrize("metric", [Metric.L2, Metric.JACCARD])
def test_transform_reproduces_fit_and_stays_in_unit_cube(metric: Metric) -> None:
    m = matrix(60, 5, seed=1)
    projection, coords = fit(m, metric, seed=0)
    assert projection.method == ("pca" if metric is Metric.L2 else "mds")
    assert np.allclose(transform(projection, m), coords, atol=1e-12)
    assert coords.min() >= -1.0 and coords.max() <= 1.0
    # Min-max scaling puts the extremes of every axis exactly on -1 and 1.
    assert np.allclose(coords.min(axis=0), -1.0) and np.allclose(coords.max(axis=0), 1.0)
    # Rows far outside the fitted range are clipped.
    far = transform(projection, m[:3] * 100.0)
    assert far.min() >= -1.0 and far.max() <= 1.0 and np.abs(far).max() == 1.0


def test_mds_equals_classical_mds_up_to_axis_sign() -> None:
    m = matrix(40, 6, seed=2)
    assert m.shape[0] < MDS_LANDMARKS
    projection, coords = fit(m, Metric.JACCARD, seed=0)
    reference, eigenvalues = classical_mds(m)
    assert aligned(coords, minmax(reference))
    # Variance kept is the share of the top eigenvalues among the positive ones.
    positive = eigenvalues[eigenvalues > 1e-9]
    assert projection.variance_kept_2d == pytest.approx(round(positive[:2].sum() / positive.sum(), 6), abs=1e-6)
    assert projection.variance_kept_3d == pytest.approx(round(positive[:3].sum() / positive.sum(), 6), abs=1e-6)
