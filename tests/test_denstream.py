from __future__ import annotations

import math

import numpy as np

from archipelago.config import Config
from archipelago.stages.denstream import StreamDenStream


def one(x: float) -> np.ndarray:
    return np.array([x])


# Closed form: a row inserted at s weighs 2^(-integral of lambda over [s, now]), lambda piecewise constant.
def decayed(s: float, now: float, segments: list[tuple[float, float]]) -> float:
    exponent = 0.0
    for i, (start, lam) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < len(segments) else math.inf
        exponent += lam * max(0.0, min(now, end) - max(s, start))
    return 2.0**-exponent


def test_weights_follow_closed_form_across_half_life_change() -> None:
    # mu * beta = 0.5 <= 1 turns pruning off, so every cluster survives.
    config = Config(denstream=True, denstream_mu=1.0, denstream_beta=0.5)
    den = StreamDenStream(config, epsilon=0.5, half_life=20.0, dimension=1)
    rows = [(0.0, 0.0), (0.0, 10.0), (100.0, 20.0), (0.0, 30.0)]
    for x, t in rows:
        den.insert(one(x), t, None, None)
    den.set_half_life(40.0)
    later = [(100.0, 50.0), (0.0, 60.0)]
    for x, t in later:
        den.insert(one(x), t, None, None)
    # lambda = 1/20 up to t=30, 1/40 after. A row at t=0 weighs 2^-(30/20 + 30/40) = 2^-2.25 at t=60, etc.
    segments = [(0.0, 1 / 20), (30.0, 1 / 40)]
    weights = den.weights()
    assert sorted(weights) == [0, 1]
    for label, x in ((0, 0.0), (1, 100.0)):
        expected = sum(decayed(t, 60.0, segments) for px, t in rows + later if px == x)
        assert math.isclose(weights[label], expected, rel_tol=1e-12)
    assert math.isclose(decayed(0.0, 60.0, segments), 2**-2.25)


def test_prune_removes_exactly_the_faded_clusters_and_labels_are_not_reused() -> None:
    # beta * mu = 5 > 1, lambda = 0.1, prune period T = ceil(log2(5 / 4) / 0.1) = 4.
    config = Config(denstream=True, denstream_mu=10.0, denstream_beta=0.5)
    den = StreamDenStream(config, epsilon=0.5, half_life=10.0, dimension=1)
    assert den.prune_period == 4 and den.potential_threshold == 5.0
    lam, period, threshold = 0.1, 4.0, 5.0
    # A: one point at 0 (label 0). C: points at 0.0 and 2.0 (label 1). B: seven points 0.5..3.5 then the
    # trigger at 4.0 (label 2). D: two points at 3.9 and 3.95 (label 3), young and below beta * mu.
    inserts = [(0.0, 0.0), (200.0, 0.0)] + [(100.0, 0.5 * k) for k in range(1, 5)] + [(200.0, 2.0)]
    inserts += [(100.0, 0.5 * k) for k in range(5, 8)] + [(300.0, 3.9), (300.0, 3.95)]
    for x, t in inserts:
        assert den.insert(one(x), t, None, None)[2] is None
    label, _, pruned = den.insert(one(100.0), 4.0, None, None)
    assert label == 2

    def cutoff(created: float) -> float:
        age = 4.0 - created
        return (2 ** (-lam * (age + period)) - 1) / (2 ** (-lam * period) - 1)

    members = {0: [0.0], 1: [0.0, 2.0], 2: [0.5 * k for k in range(1, 8)] + [4.0], 3: [3.9, 3.95]}
    expected = []
    for cluster, times in members.items():
        weight = sum(2 ** (-lam * (4.0 - t)) for t in times)
        if weight < threshold and weight < cutoff(times[0]):
            expected.append(cluster)
    # By hand: A weighs 2^-0.4 = 0.76 < cutoff 1.76; C weighs 2^-0.4 + 2^-0.2 = 1.63, also below the
    # age-4 cutoff 1.76 (a cutoff that ignored the prune period would be 1.0 and keep C); B weighs
    # about 7 >= 5; D weighs 1.99 < 5 but is above its cutoff 1.02, so it stays.
    assert expected == [0, 1]
    assert pruned == expected
    assert sorted(den.clusters) == [2, 3]
    # A point back at A's place starts a new micro-cluster; the old label never returns, even if cached.
    label, how, _ = den.insert(one(0.0), 4.5, 0, None)
    assert (label, how) == (4, "new")
    assert sorted(den.clusters) == [2, 3, 4]


def test_cache_group_search_new_placement_rules() -> None:
    config = Config(denstream=True, denstream_mu=1.0, denstream_beta=0.5)
    den = StreamDenStream(config, epsilon=0.5, half_life=100.0, dimension=1)
    assert den.insert(one(0.0), 0.0, None, 5)[:2] == (0, "new")
    # The group's last micro-cluster is reused with no radius test, even 100 away.
    assert den.insert(one(100.0), 1.0, None, 5)[:2] == (0, "group")
    # A live cached label wins over everything.
    assert den.insert(one(50.0), 2.0, 0, 9)[:2] == (0, "cache")
    # No fed cluster for group 7, nothing within epsilon of 200: a new one, remembered for the group.
    assert den.insert(one(200.0), 3.0, None, 7)[:2] == (1, "new")
    # A dead cached label falls through to the group's micro-cluster.
    assert den.insert(one(300.0), 4.0, 99, 7)[:2] == (1, "group")
    # Without a group, a dead cache falls through to the search; the cluster at ~233 is too far for 200.
    assert den.insert(one(500.0), 5.0, 99, None)[:2] == (2, "new")
    assert den.insert(one(500.0), 6.0, 99, None)[:2] == (2, "search")
    # The cache beats the group.
    assert den.insert(one(500.0), 7.0, 0, 7)[:2] == (0, "cache")
