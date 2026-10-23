from __future__ import annotations

from itertools import combinations

import numpy as np

from .models import (
    Separation,
    SeparationPairs,
    SeparationWindowPairs,
    SeparationWindows,
)

# How separate the islands are: MMD (maximum mean discrepancy) between their genomes. Exact MMD
# sums a Gaussian kernel exp(-|x - y|^2 / h) over every pair of points, which is quadratic. With
# random Fourier features (Rahimi & Recht, 2007) the kernel becomes a dot product of explicit
# features z(x) = [cos(w.x), sin(w.x)] / sqrt(D), w ~ N(0, 2/h I), and MMD^2 becomes the squared
# distance between two mean feature vectors: linear in the number of evaluations.

# Random frequencies per feature map; more -> closer to exact MMD, slower.
FREQUENCIES = 512
# Generations per window in the over-time view.
WINDOW_GENERATIONS = 10
# Generation swaps per island pair in the permutation test; sets the smallest p (1 / (SWAPS + 1)).
SWAPS = 300
# Rows sampled for the median-heuristic bandwidth.
BANDWIDTH_SAMPLE = 2000
# Rows turned into Fourier features at a time, so 100k rows never need the full feature matrix.
CHUNK = 10000


# The median heuristic: the median squared distance between points, from a sample.
def median_bandwidth(x: np.ndarray, rng: np.random.Generator) -> float:
    if len(x) > BANDWIDTH_SAMPLE:
        x = x[rng.choice(len(x), size=BANDWIDTH_SAMPLE, replace=False)]
    sq = (x * x).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    d2 = d2[np.triu_indices(len(x), k=1)]
    d2 = d2[d2 > 1e-12]
    return float(np.median(d2)) if d2.size else 1.0


def frequencies(dimension: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / bandwidth), size=(dimension, FREQUENCIES))


# z(x) for every row, so that z(x) . z(y) approximates exp(-|x - y|^2 / bandwidth).
def fourier_features(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    projection = x @ w
    return np.hstack([np.cos(projection), np.sin(projection)]) / np.sqrt(w.shape[1])


# Feature sums and row counts per (group, cell), computed chunk by chunk.
def _sums(x: np.ndarray, w: np.ndarray, group: np.ndarray, cell: np.ndarray, groups: int, cells: int):
    sums = np.zeros((groups, cells, 2 * w.shape[1]))
    counts = np.zeros((groups, cells))
    for start in range(0, len(x), CHUNK):
        rows = slice(start, start + CHUNK)
        np.add.at(sums, (group[rows], cell[rows]), fourier_features(x[rows], w))
        np.add.at(counts, (group[rows], cell[rows]), 1.0)
    return sums, counts


def _gap(sum_a: np.ndarray, n_a: float, sum_b: np.ndarray, n_b: float) -> float:
    return float(((sum_a / n_a - sum_b / n_b) ** 2).sum())


# MMD^2 between two islands, and how often random generation swaps match or beat it. Whole
# generations move, not single evaluations, because a child sits next to its parent.
def swap_test(sums_a: np.ndarray, counts_a: np.ndarray, sums_b: np.ndarray, counts_b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
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


# MMD between islands for one run: every pair over the whole run with a generation-swap p-value,
# and every pair per window of generations at that window's own scale (one whole-run scale is set
# by the wide early search and makes a contracted late population look the same everywhere).
# Rows are put in (island, seq) order first, so batch and streaming give the same answer.
def island_separation(x: np.ndarray, islands: np.ndarray, generations: np.ndarray, seq: np.ndarray, seed: int) -> Separation:
    order = np.lexsort((seq, islands))
    x, islands, generations = x[order], islands[order], generations[order]
    keys = sorted(int(i) for i in np.unique(islands))
    pairs: dict[str, list] = {"a": [], "b": [], "mmd2": [], "p": []}
    windows: dict[str, list] = {"from_generation": [], "typical_distance": []}
    window_pairs: dict[str, list] = {"window": [], "a": [], "b": [], "mmd2": []}
    typical = None
    if len(keys) >= 2:
        rng = np.random.default_rng(seed)
        bandwidth = median_bandwidth(x, rng)
        typical = float(np.sqrt(bandwidth))
        group = np.searchsorted(keys, islands)
        first = int(generations.min())
        cells = int(generations.max()) - first + 1
        sums, counts = _sums(x, frequencies(x.shape[1], bandwidth, rng), group, generations - first, len(keys), cells)
        for i, j in combinations(range(len(keys)), 2):
            mmd2, p = swap_test(sums[i], counts[i], sums[j], counts[j], rng)
            for name, value in zip(("a", "b", "mmd2", "p"), (keys[i], keys[j], mmd2, p)):
                pairs[name].append(value)

        window_of = generations // WINDOW_GENERATIONS
        for window in np.unique(window_of):
            rows = window_of == window
            present = [k for k in range(len(keys)) if np.any(group[rows] == k)]
            if len(present) < 2:
                continue
            h = median_bandwidth(x[rows], rng)
            single = np.zeros(int(rows.sum()), dtype=np.int64)
            means, sizes = _sums(x[rows], frequencies(x.shape[1], h, rng), group[rows], single, len(keys), 1)
            index = len(windows["from_generation"])
            windows["from_generation"].append(int(window) * WINDOW_GENERATIONS)
            windows["typical_distance"].append(float(np.sqrt(h)))
            for a, b in combinations(present, 2):
                gap = _gap(means[a, 0], sizes[a, 0], means[b, 0], sizes[b, 0])
                for name, value in zip(("window", "a", "b", "mmd2"), (index, keys[a], keys[b], gap)):
                    window_pairs[name].append(value)

    return Separation.model_construct(
        method="mmd-random-fourier-features",
        typical_distance=typical,
        pairs=SeparationPairs.model_construct(**pairs),
        windows=SeparationWindows.model_construct(**windows),
        window_pairs=SeparationWindowPairs.model_construct(**window_pairs),
    )
