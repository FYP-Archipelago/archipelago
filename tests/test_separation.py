from __future__ import annotations

import numpy as np

from archipelago.separation import fourier_features, frequencies, island_separation, median_bandwidth


def populations(seed: int, shifts: dict[int, np.ndarray], per_island: int = 200) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x, islands, generations, seq = [], [], [], []
    for island, shift in shifts.items():
        x.append(rng.normal(size=(per_island, len(shift))) + shift)
        islands += [island] * per_island
        generations += [k // 4 for k in range(per_island)]
        seq += list(range(per_island))
    return np.vstack(x), np.array(islands), np.array(generations), np.array(seq)


def test_shifted_island_is_separated() -> None:
    zero = np.zeros(5)
    x, islands, generations, seq = populations(0, {0: zero, 1: zero, 2: zero + 1.5})
    pairs = island_separation(x, islands, generations, seq, seed=0).pairs
    p = {(a, b): v for a, b, v in zip(pairs.a, pairs.b, pairs.p)}
    mmd2 = {(a, b): v for a, b, v in zip(pairs.a, pairs.b, pairs.mmd2)}
    assert p[(0, 2)] < 0.05 and p[(1, 2)] < 0.05
    assert mmd2[(0, 2)] > 10 * mmd2[(0, 1)]


def test_fourier_mmd_matches_exact_gaussian_kernel_mmd() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(400, 5))
    b = rng.normal(size=(400, 5)) + np.array([0.6, 0, 0, 0, 0])
    both = np.vstack([a, b])
    h = median_bandwidth(both, np.random.default_rng(4))
    z = fourier_features(both, frequencies(5, h, np.random.default_rng(5)))
    approx = float(((z[:400].mean(axis=0) - z[400:].mean(axis=0)) ** 2).sum())

    def kernel(p: np.ndarray, q: np.ndarray) -> float:
        d2 = ((p[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)
        return float(np.exp(-d2 / h).mean())

    exact = kernel(a, a) + kernel(b, b) - 2 * kernel(a, b)
    assert exact > 0.005
    assert abs(approx - exact) < 0.01
    # The features themselves: z(x).z(y) approximates exp(-|x - y|^2 / h) entry by entry. With 512
    # frequencies the mean error is about 0.02; a wrong frequency scale (e.g. a kernel twice as wide)
    # gives about 0.2, which the MMD check above can miss when MMD^2 is small.
    sample = both[:300]
    gram = np.exp(-((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2) / h)
    assert float(np.abs(z[:300] @ z[:300].T - gram).mean()) < 0.05
