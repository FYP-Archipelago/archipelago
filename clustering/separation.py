"""How separate the islands are: MMD between their genomes, via random Fourier features.

MMD (maximum mean discrepancy) is a distance between two sets of points that
assumes nothing about their shape. Exact MMD sums a kernel over every pair of
points, which is quadratic. With random Fourier features (Rahimi & Recht, 2007)
the Gaussian kernel exp(-|x - y|^2 / h) becomes a dot product of explicit
features, z(x) = [cos(w.x), sin(w.x)] / sqrt(D) with w ~ N(0, 2/h I), and MMD^2
becomes the squared distance between two mean feature vectors: linear in the
number of evaluations.

It reads the same FeatureSet the clustering stages do, so a genome means the
same thing here as everywhere else in the pipeline. Two views:

- the whole run, one bandwidth: every island pair, with a generation-swap
  p-value. Swapping whole generations between the two islands, not single
  evaluations, respects that a child sits next to its parent.
- over time, a bandwidth per window of generations. One whole-run bandwidth is
  set by the wide early search and makes a contracted late population look the
  same everywhere; per window, islands are compared at the scale they occupy then.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from . import ARTIFACT_SCHEMA_VERSION
from .types import FeatureSet

FREQUENCIES = 512
WINDOW_GENERATIONS = 10
SWAPS = 300


def median_bandwidth(x: np.ndarray, rng: np.random.Generator, sample: int = 2000) -> float:
    """The median heuristic: the median squared distance between points, from a sample."""
    if len(x) > sample:
        x = x[rng.choice(len(x), size=sample, replace=False)]
    sq = (x * x).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    d2 = d2[np.triu_indices(len(x), k=1)]
    d2 = d2[d2 > 1e-12]
    return float(np.median(d2)) if d2.size else 1.0


def fourier_features(
    x: np.ndarray, bandwidth: float, rng: np.random.Generator, frequencies: int = FREQUENCIES
) -> np.ndarray:
    """z(x) for every row, so that z(x) . z(y) approximates exp(-|x - y|^2 / bandwidth)."""
    w = rng.normal(0.0, np.sqrt(2.0 / bandwidth), size=(x.shape[1], frequencies))
    projection = x @ w
    return np.hstack([np.cos(projection), np.sin(projection)]) / np.sqrt(frequencies)


def _cells(z: np.ndarray, islands: np.ndarray, generations: np.ndarray, keys: list[int]):
    """Feature sums and counts per (island, generation): all the swap test needs."""
    first = int(generations.min())
    span = int(generations.max()) - first + 1
    sums = np.zeros((len(keys), span, z.shape[1]))
    counts = np.zeros((len(keys), span))
    for k, island in enumerate(keys):
        rows = islands == island
        np.add.at(sums[k], generations[rows] - first, z[rows])
        np.add.at(counts[k], generations[rows] - first, 1.0)
    return sums, counts


def _gap(sum_a: np.ndarray, n_a: float, sum_b: np.ndarray, n_b: float) -> float:
    return float(((sum_a / n_a - sum_b / n_b) ** 2).sum())


def _swap_test(
    sums_a: np.ndarray, counts_a: np.ndarray, sums_b: np.ndarray, counts_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """MMD^2 between two islands, and how often random generation swaps match or beat it."""
    total_a, total_b = sums_a.sum(axis=0), sums_b.sum(axis=0)
    n_a, n_b = counts_a.sum(), counts_b.sum()
    observed = _gap(total_a, n_a, total_b, n_b)
    delta = sums_b - sums_a
    delta_n = counts_b - counts_a
    at_least = 0
    for _ in range(SWAPS):
        swap = rng.random(len(delta)) < 0.5
        moved, moved_n = delta[swap].sum(axis=0), delta_n[swap].sum()
        if n_a + moved_n > 0 and n_b - moved_n > 0:
            at_least += _gap(total_a + moved, n_a + moved_n, total_b - moved, n_b - moved_n) >= observed
    return observed, (1 + at_least) / (1 + SWAPS)


def island_separation(features: FeatureSet, seed: int = 0) -> dict[str, Any]:
    """MMD between islands for one run: the whole run, and window by window."""
    # Rows with a readable genome and a known island (the reader marks a missing one -1).
    usable = features.usable[features.islands[features.usable] >= 0]
    x = features.matrix[usable]
    islands = features.islands[usable]
    generations = features.generations[usable]
    keys = sorted(int(i) for i in np.unique(islands))
    result: dict[str, Any] = {
        "method": "mmd-random-fourier-features",
        "frequencies": FREQUENCIES,
        "window_generations": WINDOW_GENERATIONS,
        "swaps": SWAPS,
        "rows_used": len(usable),
        "rows_skipped": len(features) - len(usable),
        "islands": keys,
        "typical_distance": None,
        "pairs": [],
        "windows": [],
    }
    if len(keys) < 2:
        return result

    rng = np.random.default_rng(seed)
    bandwidth = median_bandwidth(x, rng)
    z = fourier_features(x, bandwidth, rng)
    sums, counts = _cells(z, islands, generations, keys)
    result["typical_distance"] = float(np.sqrt(bandwidth))
    for i, j in combinations(range(len(keys)), 2):
        mmd2, p = _swap_test(sums[i], counts[i], sums[j], counts[j], rng)
        result["pairs"].append({"a": keys[i], "b": keys[j], "mmd2": mmd2, "p": p})

    window_of = generations // WINDOW_GENERATIONS
    for w in np.unique(window_of):
        rows = window_of == w
        present = [k for k in keys if np.any(islands[rows] == k)]
        if len(present) < 2:
            continue
        h = median_bandwidth(x[rows], rng)
        zw = fourier_features(x[rows], h, rng)
        means = {k: zw[islands[rows] == k].mean(axis=0) for k in present}
        gaps = [float(((means[a] - means[b]) ** 2).sum()) for a, b in combinations(present, 2)]
        result["windows"].append(
            {
                "from_generation": int(w) * WINDOW_GENERATIONS,
                "median_pair": float(np.median(gaps)),
                "typical_distance": float(np.sqrt(h)),
            }
        )
    return result


def write_separation(out_dir: str | Path, result: dict[str, Any], run_id: str) -> Path:
    """Write separation.json next to the run's other artifacts, in the same form they take."""
    path = Path(out_dir) / "separation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": ARTIFACT_SCHEMA_VERSION, "run_id": run_id, **result}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return path
