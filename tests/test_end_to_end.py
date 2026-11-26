from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import SyntheticRun, apply_delta, assert_view_matches, make_run, new_view

from archipelago import ClusterStream, Config, Result, ShapleyAttribution, Update, cluster_run

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
    for island_best in result.island_bests:
        island_rows = [r for r in run.rows if r["island_id"] == island_best.island_id]
        lowest = min(r["fitness"] for r in island_rows)
        first = min((r for r in island_rows if r["fitness"] == lowest), key=lambda r: r["eval_index"])
        assert (island_best.fitness, island_best.individual_id, island_best.generation) == (lowest, first["individual_id"], first["generation"])
    assert [b.island_id for b in result.island_bests] == [0, 1, 2]
    flagged = {n for n, flag in zip(nodes.id, nodes.island_best) if flag}
    assert flagged == {b.node for b in result.island_bests}
    assert {n for n, faded in zip(nodes.id, nodes.faded) if faded} <= flagged
    # Fitness is the sphere function, so the run best is the lowest island best; its node holds that best_fitness.
    run_best = next(b for b in result.island_bests if b.island_id == result.run_best_island)
    assert run_best.fitness == min(b.fitness for b in result.island_bests)
    assert [n for n, flag in zip(nodes.id, nodes.overall_best) if flag] == [run_best.node]
    assert all(nodes.best_fitness[nodes.id.index(b.node)] == b.fitness for b in result.island_bests)
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


def test_batch_and_stream_count_each_migrant_once(run: SyntheticRun, streamed: dict[str, tuple[list[Update], Result]]) -> None:
    # Island i receives two migrants from island i + 1, at generations 5 and 8, each sent a generation earlier.
    # The streamed variant "b" also redelivers events.
    expected = [(0, 2, 2), (1, 0, 2), (2, 1, 2)]
    arrivals = [(g, (d + 1) % 3, d, 1, g - 1) for g in (5, 8) for d in range(3)]
    batch = cluster_run(run.rows, run.events, STREAM_CONFIG)
    for result in (batch, *(result for _, result in streamed.values())):
        assert [(r.source, r.destination, r.migrants) for r in result.migration_routes] == expected
        assert [(a.generation, a.source, a.destination, a.migrants, a.source_generation) for a in result.migration_arrivals] == sorted(arrivals)


def test_batch_and_stream_give_the_same_attribution(run: SyntheticRun, streamed: dict[str, tuple[list[Update], Result]]) -> None:
    batch = cluster_run(run.rows, run.events, STREAM_CONFIG).explainability.shapley
    assert isinstance(batch, ShapleyAttribution)
    assert batch.coverage.supported == batch.coverage.evaluations == 300
    assert batch.routes.players
    for _, result in streamed.values():
        assert result.explainability.shapley == batch


def test_batch_and_stream_give_the_same_generation_history(run: SyntheticRun, streamed: dict[str, tuple[list[Update], Result]]) -> None:
    batch = cluster_run(run.rows, run.events, STREAM_CONFIG).generation_history
    assert [(s.island_id, s.generation, s.best_fitness) for s in batch] == [(0, 9, 0.0), (1, 9, 1.0), (2, 9, 2.0)]
    for _, result in streamed.values():
        assert result.generation_history == batch


def test_batch_and_stream_give_the_same_island_ends(run: SyntheticRun, streamed: dict[str, tuple[list[Update], Result]]) -> None:
    batch = cluster_run(run.rows, run.events, STREAM_CONFIG).island_ends
    assert [(end.island_id, end.termination_reason, end.generations_since_improvement) for end in batch] == [
        (0, "stagnation", 0),
        (1, "stagnation", 1),
        (2, "stagnation", 2),
    ]
    for _, result in streamed.values():
        assert result.island_ends == batch


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


def test_batch_and_stream_pick_the_same_bests_and_convergence(run: SyntheticRun, streamed: dict[str, tuple[list[Update], Result]]) -> None:
    batch = cluster_run(run.rows, run.events, STREAM_CONFIG)
    assert batch.run_best_island is not None

    # Node ids depend on how each mode clustered, so only the evaluations are compared.
    def picks(result: Result) -> list[tuple]:
        return [(b.island_id, b.fitness, b.individual_id, b.generation, b.evaluations) for b in result.island_bests]

    for _, result in streamed.values():
        assert picks(result) == picks(batch)
        assert result.run_best_island == batch.run_best_island
        assert result.convergence == batch.convergence


def test_live_updates_carry_the_direction_and_route_totals(streamed: dict[str, tuple[list[Update], Result]]) -> None:
    for updates, result in streamed.values():
        assert all(u.maximising is None for u in updates if u.status == "warming_up")
        assert all(u.maximising is False for u in updates if u.status == "live")
        # Totals only grow, and the last update has every migrant the result counts.
        totals = [sum(r.migrants for r in u.migration_routes) for u in updates]
        assert totals == sorted(totals)
        assert updates[-1].migration_routes == result.migration_routes
