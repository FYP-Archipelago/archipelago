from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import SyntheticRun, apply_delta, assert_view_matches, make_run, new_view

from archipelago import ClusterStream, Config, Result, Update, cluster_run

STREAM_CONFIG = {"lsh": True, "birch": True, "denstream": True}


@pytest.fixture(scope="module")
def run() -> SyntheticRun:
    return make_run(per_island=100, islands=3)


def no_non_finite(text: str) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise AssertionError(f"non-finite value in output: {token}")

    return json.loads(text, parse_constant=reject)


@pytest.mark.parametrize("stages", [{"lsh": True, "birch": True}, {"birch": True}, {"denstream": True, "birch": True}])
def test_cluster_run_invariants(run: SyntheticRun, stages: dict[str, bool]) -> None:
    result = cluster_run(run.rows, run.events, Config(**stages))
    graph = result.graph
    nodes = graph.nodes
    assert result.clustering.counts.evaluations == 300 and result.clustering.counts.rows_clustered == 300
    if "denstream" in stages:
        assert sum(nodes.visits) <= 300
        assert result.clustering.counts.faded_nodes is not None
    else:
        assert sum(nodes.visits) == 300
        assert result.clustering.counts.faded_nodes is None
    assert result.clustering.counts.nodes == len(nodes.id)
    assert sum(nodes.overall_best) == 1
    # Fitness is the sphere function, so the overall best node holds the lowest best_fitness.
    best = nodes.overall_best.index(True)
    assert nodes.best_fitness[best] == min(nodes.best_fitness)
    assert sum(nodes.island_best) == 3
    live = set(nodes.id)
    assert nodes.id == sorted(live)
    assert all(s in live and t in live and s != t for s, t in zip(graph.edges.source, graph.edges.target))
    assert all(s in live and t in live and s != t for s, t in zip(graph.migrations.source, graph.migrations.target))
    assert all(w >= 1 for w in graph.edges.weight)
    assert all(0.0 <= p <= 1.0 for p in nodes.fitness_percentile) and all(0.0 <= p <= 1.0 for p in nodes.progress)
    assert all(-1.0 <= c <= 1.0 for c in nodes.x + nodes.y + nodes.z)
    assert result.run.islands == [0, 1, 2]
    no_non_finite(result.model_dump_json())


def chunks(run: SyntheticRun, order: list[int], size: int, redeliver: int) -> Iterator[tuple[list[dict], list[dict]]]:
    # Islands take turns in blocks of `size` records, each island's records in seq order; every call
    # also repeats the last `redeliver` records of the previous block (and the start events once).
    cursor = dict.fromkeys(order, 0)
    previous: tuple[list[dict], list[dict]] = ([], [])
    first = True
    while any(cursor[i] < len(run.records[i]) for i in order):
        for island in order:
            block = run.records[island][cursor[island] : cursor[island] + size]
            cursor[island] += size
            rows = [r for kind, r in block if kind == "row"]
            events = [r for kind, r in block if kind == "event"]
            if first:
                events = [*run.start, *events]
                first = False
            sent = (previous[0][-redeliver:] + rows, previous[1][-redeliver:] + events) if redeliver else (rows, events)
            previous = (rows, events)
            yield sent
    yield [], list(run.end)


def stream_through(run: SyntheticRun, order: list[int], size: int, redeliver: int) -> tuple[list[Update], Result]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("archipelago.stream.WARMUP_ROWS", 50)
        stream = ClusterStream(STREAM_CONFIG)
        updates = [stream.cluster(rows, events) for rows, events in chunks(run, order, size, redeliver)]
        return updates, stream.finish()


@pytest.fixture(scope="module")
def streamed(run: SyntheticRun) -> dict[str, tuple[list[Update], Result]]:
    return {
        "a": stream_through(run, [0, 1, 2], size=20, redeliver=0),
        "b": stream_through(run, [2, 0, 1], size=13, redeliver=4),
    }


def scrubbed(result: Result) -> dict[str, Any]:
    dump = result.model_dump()
    dump["clustering"]["elapsed_seconds"] = 0.0
    return dump


def test_stream_result_is_independent_of_chunking_interleaving_and_redelivery(streamed: dict[str, tuple[list[Update], Result]]) -> None:
    (updates_a, a), (_updates_b, b) = streamed["a"], streamed["b"]
    assert any(u.status == "warming_up" for u in updates_a) and updates_a[-1].status == "live"
    assert a.clustering.counts.evaluations == 300
    assert scrubbed(a) == scrubbed(b)
    no_non_finite(a.model_dump_json())


def test_applying_all_updates_reproduces_the_final_graph(streamed: dict[str, tuple[list[Update], Result]]) -> None:
    for updates, result in streamed.values():
        view = new_view()
        for update in updates:
            if update.status == "live":
                apply_delta(view, update.graph)
        # Positions are not compared: finish() refits the layout on every row, which live updates
        # cannot know about. Everything else is exact (percentile and weight within tolerance).
        assert_view_matches(view, result.graph, skip=("x", "y", "z"))
        assert all(-1.0 <= view.nodes[n][axis] <= 1.0 for n in view.nodes for axis in "xyz")
