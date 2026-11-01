from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from archipelago.config import DRIFT_TOLERANCE, WEIGHT_TOLERANCE
from archipelago.graph.state import GraphState, add_rows
from archipelago.models import Graph, GraphDelta

# island, label, eval_index, fitness, genome hash
SpecRow = tuple[int, int, int, float, str]


# Add rows to a state. Default coords of row g (its global index) are (g, 2g, -g).
def add_spec(state: GraphState, rows: list[SpecRow], coords: np.ndarray | None = None) -> None:
    start = state.rows
    if coords is None:
        index = np.arange(start, start + len(rows), dtype=np.float64)
        coords = np.stack([index, 2 * index, -index], axis=1)
    add_rows(
        state,
        [f"r{start + i}" for i in range(len(rows))],
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[2] for r in rows], dtype=np.int64),
        np.array([r[3] for r in rows], dtype=np.float64),
        [r[4] for r in rows],
        coords,
    )


# A minimal viewer: holds what the deltas have told it, in the order GraphDelta documents.
@dataclass
class View:
    nodes: dict[int, dict[str, Any]]
    edges: dict[int, tuple[int, int, int]]
    migrations: dict[int, tuple[int, int, int]]
    island_best: set[int]
    overall_best: int | None = None
    overlap: float = 0.0


def new_view() -> View:
    return View({}, {}, {}, set())


def _columns_of(model: Any, skip: tuple[str, ...] = ()) -> dict[str, list]:
    return {name: getattr(model, name) for name in type(model).model_fields if name not in skip}


def _overwrite(view: View, model: Any) -> None:
    columns = _columns_of(model, skip=("id",))
    for position, node in enumerate(model.id):
        for name, values in columns.items():
            view.nodes[node][name] = values[position]


def apply_delta(view: View, delta: GraphDelta) -> None:
    columns = _columns_of(delta.new_nodes, skip=("id",))
    for position, node in enumerate(delta.new_nodes.id):
        assert node not in view.nodes
        view.nodes[node] = {name: values[position] for name, values in columns.items() if values is not None}
    _overwrite(view, delta.updated_nodes)
    _overwrite(view, delta.percentiles)
    if delta.weights is not None:
        _overwrite(view, delta.weights)
    if delta.positions is not None:
        _overwrite(view, delta.positions)
    gone = set(delta.removed_nodes)
    for node in gone:
        del view.nodes[node]
    for table in (view.edges, view.migrations):
        for key in [k for k, (s, t, _) in table.items() if s in gone or t in gone]:
            del table[key]
    for table, new, update, total in (
        (view.edges, delta.new_edges, delta.edge_updates, "weight"),
        (view.migrations, delta.new_migrations, delta.migration_updates, "count"),
    ):
        for i, key in enumerate(new.id):
            table[key] = (new.source[i], new.target[i], getattr(new, total)[i])
        for i, key in enumerate(update.id):
            source, target, _ = table[key]
            table[key] = (source, target, getattr(update, total)[i])
    view.island_best = set(delta.island_best)
    view.overall_best = delta.overall_best
    view.overlap = delta.overlap


EXACT_NODE_FIELDS = (
    "island",
    "x",
    "y",
    "z",
    "visits",
    "members",
    "mean_fitness",
    "best_fitness",
    "progress",
    "first_eval",
    "last_eval",
    "in_degree",
    "out_degree",
    "islands_sharing",
    "migration_endpoint",
    "genome_hash",
)


# The viewer must equal the exact graph on every exact field; the percentile and weight may lag
# by the tolerances the module promises.
def assert_view_matches(view: View, graph: Graph, skip: tuple[str, ...] = ()) -> None:
    nodes = graph.nodes
    assert sorted(view.nodes) == nodes.id
    for position, node in enumerate(nodes.id):
        seen = view.nodes[node]
        for name in EXACT_NODE_FIELDS:
            if name not in skip:
                assert seen[name] == getattr(nodes, name)[position], (node, name)
        assert abs(seen["fitness_percentile"] - nodes.fitness_percentile[position]) <= DRIFT_TOLERANCE, node
        if nodes.weight is not None:
            assert abs(seen["weight"] - nodes.weight[position]) <= WEIGHT_TOLERANCE * abs(seen["weight"]) + 1e-15, node
    assert view.island_best == {n for n, flag in zip(nodes.id, nodes.island_best) if flag}
    best = [n for n, flag in zip(nodes.id, nodes.overall_best) if flag]
    assert view.overall_best == (best[0] if best else None)
    edges = graph.edges
    assert view.edges == {i: (s, t, w) for i, s, t, w in zip(edges.id, edges.source, edges.target, edges.weight)}
    moves = graph.migrations
    assert view.migrations == {i: (s, t, c) for i, s, t, c in zip(moves.id, moves.source, moves.target, moves.count)}
    assert view.overlap == graph.overlap


# --- a small synthetic run of real-vector genomes ---


@dataclass
class SyntheticRun:
    rows: list[dict[str, Any]]
    events: list[dict[str, Any]]
    # Per island, its rows and migration_arrive events in seq order.
    records: dict[int, list[tuple[str, dict[str, Any]]]]
    start: list[dict[str, Any]]
    end: list[dict[str, Any]]


DIMENSION = 4


# Each island random-walks from its own start, every child has the previous individual as parent
# (every fifth also the one before), every seventh child repeats its parent's genome exactly.
# Times are laid out so the clock-corrected time is 0.5 * k + 0.1 * island.
def make_run(per_island: int = 100, islands: int = 3, seed: int = 1) -> SyntheticRun:
    rng = np.random.default_rng(seed)
    ids = {island: [f"i{island}-{k}" for k in range(per_island)] for island in range(islands)}
    records: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for island in range(islands):
        genome = np.full(DIMENSION, (island - 1) * 2.0)
        records[island] = []
        previous = genome
        for k in range(per_island):
            if k % 7 == 6:
                genome = previous
            else:
                genome = np.clip(np.round(previous + 0.3 * rng.normal(size=DIMENSION), 6), -5.0, 5.0)
            text = json.dumps(genome.tolist())
            parents = [ids[island][k - 1]] if k else []
            if k % 5 == 0 and k >= 2:
                parents.append(ids[island][k - 2])
            row = {
                "island_id": island,
                "eval_index": k,
                "generation": k // 10,
                "seq": 0,
                "t_wall": 1000.0 * island + 0.5 * k + 0.1 * island,
                "individual_id": ids[island][k],
                "parent_ids": parents,
                "fitness": float((genome**2).sum()),
                "genome_encoding": "real_vector",
                "genome_hash": hashlib.sha1(text.encode()).hexdigest()[:12],
                "genome_repr": genome.tolist(),
                "genome_repr_mode": "full",
            }
            records[island].append(("row", row))
            previous = genome
            if k in (50, 80):
                origin = ids[(island + 1) % islands][k - 20]
                event = {
                    "type": "migration_arrive",
                    "island_id": island,
                    "seq": 0,
                    "t_wall": 1000.0 * island + 0.5 * k + 0.1 * island + 0.01,
                    "origin_individual_ids": [origin],
                    "arrived_individual_ids": [ids[island][k]],
                }
                records[island].append(("event", event))
        for seq, (_, record) in enumerate(records[island]):
            record["seq"] = seq

    start = [
        {
            "type": "run_start",
            "run_id": "synthetic",
            "algorithm": "ga",
            "benchmark": "sphere",
            "num_islands": islands,
            "evaluation_budget": islands * per_island,
            "evaluation_budget_per_island": per_island,
            "datasets": {"maximising": False},
            "migration": {"topology": "ring"},
        },
        *(
            {
                "type": "island_start",
                "island_id": island,
                "genome_encoding": "real_vector",
                "benchmark_params": {"dimension": DIMENSION, "lower": -5.0, "upper": 5.0},
                "clock_offset_ns": int(-1000 * island * 10**9),
            }
            for island in range(islands)
        ),
    ]
    end = [
        *({"type": "generation_end", "island_id": island, "maximising": False} for island in range(islands)),
        *({"type": "island_end", "island_id": island} for island in range(islands)),
        {"type": "run_end"},
    ]
    events = [*start, *end, *(r for recs in records.values() for kind, r in recs if kind == "event")]
    rows = [r for recs in records.values() for kind, r in recs if kind == "row"]
    # The file order of a log: by clock-corrected time.
    rows.sort(key=lambda r: r["t_wall"] - 1000.0 * r["island_id"])
    return SyntheticRun(rows, events, records, start, end)
