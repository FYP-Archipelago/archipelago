import json

import numpy as np

from clustering import ARTIFACT_SCHEMA_VERSION
from clustering.separation import (
    FREQUENCIES,
    _cells,
    _swap_test,
    fourier_features,
    island_separation,
    median_bandwidth,
    write_separation,
)
from clustering.types import FeatureSet, Metric


def islands_feature_set(shift: float, generations: int = 30, per_generation: int = 20, seed: int = 0) -> FeatureSet:
    """Three islands on a 5-D Gaussian; island 2 is moved by `shift`."""
    rng = np.random.default_rng(seed)
    rows, islands, gens = [], [], []
    for g in range(generations):
        for island in range(3):
            centre = np.full(5, shift if island == 2 else 0.0)
            rows.append(rng.normal(centre, 1.0, size=(per_generation, 5)))
            islands += [island] * per_generation
            gens += [g] * per_generation
    matrix = np.vstack(rows)
    return FeatureSet(
        matrix=matrix,
        metric=Metric.L2,
        hashes=[str(i) for i in range(len(matrix))],
        mask=np.ones(len(matrix), dtype=bool),
        generations=np.asarray(gens),
        islands=np.asarray(islands),
    )


def pair(result, a, b):
    return next(p for p in result["pairs"] if p["a"] == a and p["b"] == b)


def test_a_moved_island_stands_out():
    result = island_separation(islands_feature_set(shift=1.5))
    assert pair(result, 0, 2)["p"] < 0.05
    assert pair(result, 1, 2)["p"] < 0.05
    assert pair(result, 0, 2)["mmd2"] > 10 * pair(result, 0, 1)["mmd2"]


def test_agrees_with_exact_mmd():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=(400, 5))
    b = rng.normal(0.6, 1.0, size=(400, 5))
    h = median_bandwidth(np.vstack([a, b]), rng)

    def kernel_mean(p, q):
        d2 = (p * p).sum(1)[:, None] + (q * q).sum(1)[None, :] - 2 * p @ q.T
        return np.exp(-np.maximum(d2, 0) / h).mean()

    exact = kernel_mean(a, a) + kernel_mean(b, b) - 2 * kernel_mean(a, b)
    z = fourier_features(np.vstack([a, b]), h, np.random.default_rng(2))
    fourier = ((z[:400].mean(0) - z[400:].mean(0)) ** 2).sum()
    assert abs(fourier - exact) < 0.01
    assert z.shape[1] == 2 * FREQUENCIES


def test_swap_test_is_calibrated_when_islands_are_interchangeable():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2000, 5))
    generations = np.repeat(np.arange(50), 40)
    z = fourier_features(x, median_bandwidth(x, rng), rng)
    rejected = 0
    for _ in range(100):
        halves = (rng.random(len(x)) < 0.5).astype(np.int64)
        sums, counts = _cells(z, halves, generations, [0, 1])
        rejected += _swap_test(sums[0], counts[0], sums[1], counts[1], rng)[1] < 0.05
    assert rejected <= 12


def test_windows_and_determinism():
    features = islands_feature_set(shift=1.0)
    first = island_separation(features, seed=7)
    assert first == island_separation(features, seed=7)
    assert [w["from_generation"] for w in first["windows"]] == [0, 10, 20]
    assert all(w["typical_distance"] > 0 for w in first["windows"])


def test_one_island_has_nothing_to_compare():
    features = islands_feature_set(shift=0.0)
    keep = features.islands == 0
    single = FeatureSet(
        matrix=features.matrix[keep],
        metric=features.metric,
        hashes=[h for h, k in zip(features.hashes, keep) if k],
        mask=features.mask[keep],
        generations=features.generations[keep],
        islands=features.islands[keep],
    )
    result = island_separation(single)
    assert result["pairs"] == [] and result["windows"] == []


def test_rows_without_an_island_are_left_out():
    features = islands_feature_set(shift=1.0)
    features.islands = features.islands.copy()
    features.islands[:25] = -1
    result = island_separation(features)
    assert result["islands"] == [0, 1, 2]
    assert result["rows_skipped"] == 25
    assert result["rows_used"] == len(features) - 25


def test_written_file_takes_the_form_of_the_other_artifacts(tmp_path):
    result = island_separation(islands_feature_set(shift=1.0))
    path = write_separation(tmp_path, result, run_id="run-test")
    text = path.read_text()
    payload = json.loads(text)
    assert path.name == "separation.json"
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["run_id"] == "run-test"
    assert payload["pairs"] == result["pairs"]
    assert text == json.dumps(payload, indent=2, allow_nan=False)
