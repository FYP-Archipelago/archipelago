from __future__ import annotations

import numpy as np

from archipelago.config import NUM_PERM, Config
from archipelago.encoding.features import FeatureSet, Metric
from archipelago.encoding.hashing import MinHasher
from archipelago.stages.birch import StreamBirch
from archipelago.stages.cf import nearest, radius_if_added
from archipelago.stages.lsh import BatchLSH, StreamLSH


def test_cluster_feature_radius_and_nearest_match_direct_computation() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(6, 3))
    weights = rng.uniform(0.5, 3.0, size=6)
    # Cluster feature of the first five points, then add the sixth with its weight.
    cf1 = (points[:5] * weights[:5, None]).sum(axis=0)
    cf2 = float(((points[:5] ** 2).sum(axis=1) * weights[:5]).sum())
    got = radius_if_added(float(weights[:5].sum()), cf1, cf2, points[5], float(weights[5]))
    centre = (points * weights[:, None]).sum(axis=0) / weights.sum()
    expected = np.sqrt((weights * ((points - centre) ** 2).sum(axis=1)).sum() / weights.sum())
    assert abs(got - expected) < 1e-12
    # {0, 2} with unit weights: centre 1, radius 1. A lone point has radius 0.
    assert abs(radius_if_added(1.0, np.array([0.0]), 0.0, np.array([2.0]), 1.0) - 1.0) < 1e-15
    assert radius_if_added(0.0, np.zeros(1), 0.0, np.array([3.0]), 1.0) == 0.0
    check_nearest_matches_brute_force_and_breaks_ties_low()


def check_nearest_matches_brute_force_and_breaks_ties_low() -> None:
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(50, 4))
    norms = (centres**2).sum(axis=1)
    for point in rng.normal(size=(100, 4)):
        assert nearest(centres, norms, point) == int(np.argmin(((centres - point) ** 2).sum(axis=1)))
    tied = np.array([[2.0, 2.0], [0.0, 1.0], [5.0, 5.0], [0.0, 1.0]])
    assert nearest(tied, (tied**2).sum(axis=1), np.array([0.0, 1.0])) == 1
    # Equidistant, different centres: (1, 0) and (-1, 0) from the origin.
    pair = np.array([[-1.0, 0.0], [1.0, 0.0]])
    assert nearest(pair, (pair**2).sum(axis=1), np.zeros(2)) == 0
    assert nearest(pair[::-1], (pair[::-1] ** 2).sum(axis=1), np.zeros(2)) == 0


def partition_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :]))


def features(matrix: np.ndarray, metric: Metric, sets: list[np.ndarray] | None = None) -> FeatureSet:
    n = len(matrix)
    return FeatureSet(
        matrix=matrix,
        metric=metric,
        hashes=[str(i) for i in range(n)],
        mask=np.ones(n, dtype=bool),
        times=np.arange(n, dtype=float),
        islands=np.zeros(n, dtype=np.int64),
        generations=np.zeros(n, dtype=np.int64),
        sets=sets,
    )


def test_stream_lsh_matches_batch_lsh_and_minhash() -> None:
    matrix = np.random.default_rng(2).uniform(size=(300, 4))
    config = Config(lsh=True, lsh_hashes=3, seed=5)
    batch = BatchLSH(config)
    labels = batch.fit_predict(features(matrix, Metric.L2), np.zeros(300, dtype=np.int64))
    width = batch.diagnostics()["bucket_width"]
    stream = StreamLSH(config, Metric.L2, 4, width)
    keys = stream.keys(matrix, None)
    streamed = np.array([stream.assign(k) for k in keys])
    assert 1 < len(set(labels.tolist())) < 300
    assert partition_equal(labels, streamed)
    # Labels are first-seen indices.
    assert streamed[0] == 0 and streamed.max() == len(set(streamed.tolist())) - 1
    check_stream_lsh_jaccard_keys_are_minhash_signatures()


def check_stream_lsh_jaccard_keys_are_minhash_signatures() -> None:
    config = Config(lsh=True, lsh_hashes=5, seed=2)
    sets = [np.array([3, 9, 27], dtype=np.uint64), np.array([], dtype=np.uint64), np.array([1 << 40, 5], dtype=np.uint64)]
    keys = StreamLSH(config, Metric.JACCARD, 0, None).keys(np.zeros((3, 0)), sets)
    hasher = MinHasher(max(NUM_PERM, 5), 2)
    for row, ids in enumerate(sets):
        assert keys[row].tolist() == hasher.signature(ids)[:5].astype(np.int64).tolist()


def test_stream_birch_join_rule_by_hand() -> None:
    birch = StreamBirch(threshold=1.0)
    # Points 0, 1, 2 join one subcluster (radii 0.5 and sqrt(2/3)). Adding 3 to {0, 1, 2}: centre 1.5,
    # variance (0 + 1 + 4 + 9) / 4 - 2.25 = 1.25, radius 1.118 > 1, so label 1 starts. 3.5 joins it
    # (radius 0.25 for {3, 3.5}); 100 is far from both and starts label 2.
    labels = [birch.insert(np.array([x]), 0) for x in (0.0, 1.0, 2.0, 3.0, 3.5, 100.0)]
    assert labels == [0, 0, 0, 1, 1, 2]
    # Same coordinates in another upstream group never share a subcluster, and take the next global label.
    assert birch.insert(np.array([0.0]), 7) == 3
    assert birch.insert(np.array([0.0]), 7) == 3
    assert birch.insert(np.array([0.1]), 0) == 0
    # Only the nearest subcluster is tried: with subclusters at 0 and 10 (radius {0, 10} = 5 > 1), 9 joins 10.
    other = StreamBirch(1.0)
    assert [other.insert(np.array([x]), 0) for x in (0.0, 10.0, 9.0)] == [0, 1, 1]
    check_stream_birch_absorb_and_set_threshold()


def check_stream_birch_absorb_and_set_threshold() -> None:
    birch = StreamBirch(threshold=1.0)
    for x in (0.0, 1.0, 2.0):
        birch.insert(np.array([x]), 0)
    # Absorb adds without a radius test: {0, 1, 2, 50} has centre 13.25 although its radius is huge.
    birch.absorb(np.array([50.0]), 0)
    group = birch.groups[0]
    assert group.count[0] == 4 and group.centre[0, 0] == 13.25
    assert len(birch.where) == 1

    later = StreamBirch(threshold=0.5)
    # {0, 1.5} has radius 0.75 > 0.5: separate subclusters.
    assert [later.insert(np.array([x]), 0) for x in (0.0, 1.5)] == [0, 1]
    later.set_threshold(2.0)
    # Earlier points are not re-judged; the new threshold lets 3.0 join {1.5} (radius 0.75 <= 2).
    assert later.insert(np.array([3.0]), 0) == 1 and len(later.where) == 2
    later.set_threshold(0.1)
    assert later.insert(np.array([1.6]), 0) == 2
