from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DRIFT_TOLERANCE, WEIGHT_TOLERANCE
from ..models import (
    EdgeChanges,
    Edges,
    GraphDelta,
    MigrationChanges,
    Migrations,
    NodeChanges,
    PercentileChanges,
    Positions,
    WeightChanges,
)
from .state import nodes_model

# Node fields that can change after a node is first sent, compared exactly.
CHANGING = (
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
    "x",
    "y",
    "z",
)
POSITION = ("x", "y", "z")


def _grow(array: np.ndarray, size: int, fill: float) -> np.ndarray:
    if array.shape[0] >= size:
        return array
    grown = np.full(max(size, 2 * array.shape[0], 64), fill, dtype=array.dtype)
    grown[: array.shape[0]] = array
    return grown


# What the viewer was last sent, per node / edge / migration id.
@dataclass
class Sent:
    flag: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    values: dict[str, np.ndarray] = field(default_factory=dict)
    edge_weight: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    migration_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _record(sent: Sent, cols: dict, select: np.ndarray, names: tuple[str, ...]) -> None:
    ids = cols["id"][select]
    for name in names:
        sent.values[name][ids] = cols[name][select]


def _changes(cols: dict, select: np.ndarray, model: type, names: tuple[str, ...]):
    data = {name: cols[name][select].tolist() for name in names}
    return model.model_construct(id=cols["id"][select].tolist(), **data)


# Compare the exact current columns with what was last sent, record what this update sends,
# and return the delta. refit: the layout was refitted, so every sent node gets new positions.
def diff(sent: Sent, cols: dict, nodes: int, edges: int, migrations: int, refit: bool) -> GraphDelta:
    weighted = cols["weight"] is not None
    names = (*CHANGING, "fitness_percentile", "weight") if weighted else (*CHANGING, "fitness_percentile")
    sent.flag = _grow(sent.flag, nodes, False)
    for name in names:
        sent.values[name] = _grow(sent.values.get(name, np.zeros(0)), nodes, 0.0)
    sent.edge_weight = _grow(sent.edge_weight, edges, -1)
    sent.migration_count = _grow(sent.migration_count, migrations, -1)

    ids = cols["id"]
    was = sent.flag[ids]
    live = np.zeros(nodes, dtype=bool)
    live[ids] = True
    removed = np.flatnonzero(sent.flag[:nodes] & ~live)
    sent.flag[removed] = False

    compared = tuple(n for n in CHANGING if not (refit and n in POSITION))
    changed = np.zeros(ids.size, dtype=bool)
    for name in compared:
        changed |= was & (cols[name] != sent.values[name][ids])
    drifted = was & (np.abs(cols["fitness_percentile"] - sent.values["fitness_percentile"][ids]) > DRIFT_TOLERANCE)
    shifted = np.zeros(ids.size, dtype=bool)
    if weighted:
        previous = sent.values["weight"][ids]
        shifted = was & (np.abs(cols["weight"] - previous) > WEIGHT_TOLERANCE * np.abs(previous))

    delta_new = nodes_model(cols, ~was)
    delta_updated = _changes(cols, changed, NodeChanges, CHANGING)
    delta_percentiles = _changes(cols, drifted, PercentileChanges, ("fitness_percentile",))
    delta_weights = _changes(cols, shifted, WeightChanges, ("weight",)) if weighted else None
    delta_positions = _changes(cols, was, Positions, POSITION) if refit else None
    _record(sent, cols, ~was, names)
    _record(sent, cols, changed, CHANGING)
    _record(sent, cols, drifted, ("fitness_percentile",))
    if weighted:
        _record(sent, cols, shifted, ("weight",))
    if refit:
        _record(sent, cols, was, POSITION)
    sent.flag[ids] = True

    def links(id_key: str, pair_key: str, total_key: str, record: np.ndarray, full: type, change: type, column: str):
        index = cols[id_key]
        previous = record[index]
        current = cols[total_key]
        fresh = previous < 0
        moved = ~fresh & (previous != current)
        record[index] = current
        new = full.model_construct(
            id=index[fresh].tolist(),
            source=cols[pair_key][fresh, 0].tolist(),
            target=cols[pair_key][fresh, 1].tolist(),
            **{column: current[fresh].tolist()},
        )
        return new, change.model_construct(id=index[moved].tolist(), **{column: current[moved].tolist()})

    new_edges, edge_updates = links("edge_id", "edge_pair", "edge_weight", sent.edge_weight, Edges, EdgeChanges, "weight")
    new_migrations, migration_updates = links(
        "migration_id", "migration_pair", "migration_count", sent.migration_count, Migrations, MigrationChanges, "count"
    )
    overall = ids[cols["overall_best"]]
    return GraphDelta.model_construct(
        new_nodes=delta_new,
        updated_nodes=delta_updated,
        percentiles=delta_percentiles,
        weights=delta_weights,
        removed_nodes=removed.tolist(),
        positions=delta_positions,
        new_edges=new_edges,
        edge_updates=edge_updates,
        new_migrations=new_migrations,
        migration_updates=migration_updates,
        island_best=ids[cols["island_best"]].tolist(),
        overall_best=int(overall[0]) if overall.size else None,
        overlap=cols["overlap"],
    )
