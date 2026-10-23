from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..models import Edges, Graph, Migrations, Nodes

INT_MAX = np.iinfo(np.int64).max
INT_MIN = np.iinfo(np.int64).min


def grow(array: np.ndarray, size: int, fill: float | bool = 0) -> np.ndarray:
    if array.shape[0] >= size:
        return array
    grown = np.full((max(size, 2 * array.shape[0], 64), *array.shape[1:]), fill, dtype=array.dtype)
    grown[: array.shape[0]] = array
    return grown


def _empty(dtype: type, *shape: int) -> np.ndarray:
    return np.zeros((0, *shape), dtype=dtype)


# Running per-node totals for one graph. A node is a distinct (island, label); ids are given in
# order of first appearance and never reused. Rows are the usable rows, in processing order.
# Edges and migrations are created without checking liveness, so their ids never depend on when
# a node died; only those between two live nodes are output.
@dataclass
class GraphState:
    maximising: bool
    weighted: bool
    nodes: int = 0
    node_of: dict[tuple[int, int], int] = field(default_factory=dict)
    nodes_of_label: dict[int, list[int]] = field(default_factory=dict)
    island: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    label: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    visits: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    members: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    fit_sum: np.ndarray = field(default_factory=lambda: _empty(np.float64))
    fit_min: np.ndarray = field(default_factory=lambda: _empty(np.float64))
    fit_max: np.ndarray = field(default_factory=lambda: _empty(np.float64))
    eval_sum: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    first_eval: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    last_eval: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    coord_sum: np.ndarray = field(default_factory=lambda: _empty(np.float64, 3))
    alive: np.ndarray = field(default_factory=lambda: _empty(bool))
    weight: np.ndarray = field(default_factory=lambda: _empty(np.float64))
    genome_hash: list[str] = field(default_factory=list)
    node_genomes: set[tuple[int, str]] = field(default_factory=set)
    rows: int = 0
    row_of: dict[str, int] = field(default_factory=dict)
    row_node: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    row_fitness: np.ndarray = field(default_factory=lambda: _empty(np.float64))
    row_coords: np.ndarray = field(default_factory=lambda: _empty(np.float64, 3))
    edges: int = 0
    edge_of: dict[tuple[int, int], int] = field(default_factory=dict)
    edge_pair: np.ndarray = field(default_factory=lambda: _empty(np.int64, 2))
    edge_weight: np.ndarray = field(default_factory=lambda: _empty(np.int64))
    migrations: int = 0
    migration_of: dict[tuple[int, int], int] = field(default_factory=dict)
    migration_pair: np.ndarray = field(default_factory=lambda: _empty(np.int64, 2))
    migration_count: np.ndarray = field(default_factory=lambda: _empty(np.int64))


def _new_node(state: GraphState, island: int, label: int, genome_hash: str) -> int:
    node = state.nodes
    size = node + 1
    state.island = grow(state.island, size)
    state.label = grow(state.label, size)
    state.visits = grow(state.visits, size)
    state.members = grow(state.members, size)
    state.fit_sum = grow(state.fit_sum, size)
    state.fit_min = grow(state.fit_min, size, np.inf)
    state.fit_max = grow(state.fit_max, size, -np.inf)
    state.eval_sum = grow(state.eval_sum, size)
    state.first_eval = grow(state.first_eval, size, INT_MAX)
    state.last_eval = grow(state.last_eval, size, INT_MIN)
    state.coord_sum = grow(state.coord_sum, size)
    state.alive = grow(state.alive, size)
    state.weight = grow(state.weight, size)
    state.island[node] = island
    state.label[node] = label
    state.alive[node] = True
    state.genome_hash.append(genome_hash)
    state.node_of[(island, label)] = node
    state.nodes_of_label.setdefault(label, []).append(node)
    state.nodes = size
    return node


# Add usable rows (in processing order) under their labels.
def add_rows(
    state: GraphState,
    individual_id: Sequence[str],
    island: np.ndarray,
    label: np.ndarray,
    eval_index: np.ndarray,
    fitness: np.ndarray,
    genome_hash: Sequence[str],
    coords: np.ndarray,
) -> None:
    count = len(individual_id)
    start = state.rows
    node = np.empty(count, dtype=np.int64)
    for i, key in enumerate(zip(island.tolist(), label.tolist())):
        found = state.node_of.get(key)
        node[i] = found if found is not None else _new_node(state, key[0], key[1], genome_hash[i])
        if (int(node[i]), genome_hash[i]) not in state.node_genomes:
            state.node_genomes.add((int(node[i]), genome_hash[i]))
            state.members[node[i]] += 1
        state.row_of[individual_id[i]] = start + i

    # The ufunc .at methods apply in input order, so sums accumulate row by row.
    np.add.at(state.visits, node, 1)
    np.add.at(state.fit_sum, node, fitness)
    np.minimum.at(state.fit_min, node, fitness)
    np.maximum.at(state.fit_max, node, fitness)
    np.add.at(state.eval_sum, node, eval_index)
    np.minimum.at(state.first_eval, node, eval_index)
    np.maximum.at(state.last_eval, node, eval_index)
    np.add.at(state.coord_sum, node, coords)

    size = start + count
    state.row_node = grow(state.row_node, size)
    state.row_fitness = grow(state.row_fitness, size)
    state.row_coords = grow(state.row_coords, size)
    state.row_node[start:size] = node
    state.row_fitness[start:size] = fitness
    state.row_coords[start:size] = coords
    state.rows = size


def _link(
    of: dict[tuple[int, int], int], pairs: np.ndarray, totals: np.ndarray, size: int, source: int, target: int
) -> tuple[np.ndarray, np.ndarray, int]:
    key = (source, target)
    index = of.get(key)
    if index is None:
        index = of[key] = size
        size += 1
        pairs = grow(pairs, size)
        totals = grow(totals, size)
        pairs[index] = key
    totals[index] += 1
    return pairs, totals, size


# (parent row, child row) lineage pairs within an island, in occurrence order.
def add_edges(state: GraphState, row_pairs: Sequence[tuple[int, int]]) -> None:
    for parent, child in row_pairs:
        source, target = int(state.row_node[parent]), int(state.row_node[child])
        if source != target:
            state.edge_pair, state.edge_weight, state.edges = _link(state.edge_of, state.edge_pair, state.edge_weight, state.edges, source, target)


# (origin row, arrived row) migrant pairs, in resolution order.
def add_migrations(state: GraphState, row_pairs: Sequence[tuple[int, int]]) -> None:
    for origin, arrived in row_pairs:
        source, target = int(state.row_node[origin]), int(state.row_node[arrived])
        if source != target:
            state.migration_pair, state.migration_count, state.migrations = _link(
                state.migration_of, state.migration_pair, state.migration_count, state.migrations, source, target
            )


# Mark every node carrying one of these labels as dead.
def kill_labels(state: GraphState, labels: Sequence[int]) -> None:
    died = [node for label in labels for node in state.nodes_of_label.get(int(label), ())]
    state.alive[died] = False


# Replace every row's layout coordinates (after a layout refit) and re-derive the node sums.
def set_coords(state: GraphState, coords: np.ndarray) -> None:
    state.row_coords[: state.rows] = coords
    state.coord_sum[: state.nodes] = 0.0
    np.add.at(state.coord_sum, state.row_node[: state.rows], coords)


# Exact columns of every live node, in id order, plus the live edges and migrations.
# progress divides each node's mean eval_index by span[island]: the island's per-island budget
# while live, or its evaluation count (max eval_index + 1) for a final result.
def columns(state: GraphState, span: dict[int, int]) -> dict[str, np.ndarray | list]:
    ids = np.flatnonzero(state.alive[: state.nodes])
    island = state.island[ids]
    label = state.label[ids]
    visits = state.visits[ids]
    safe = np.maximum(visits, 1)
    lowest, highest = state.fit_min[ids], state.fit_max[ids]
    # Summation rounding can land a mean just outside its members' range (34 equal values
    # summing to f + 1 ulp), which would rank the node's own rows as better than itself.
    mean = np.clip(state.fit_sum[ids] / safe, lowest, highest)
    best = highest if state.maximising else lowest
    best_oriented = -highest if state.maximising else lowest
    mean_oriented = -mean if state.maximising else mean
    fitness = state.row_fitness[: state.rows]
    ranked = np.sort(-fitness if state.maximising else fitness)
    percentile = np.searchsorted(ranked, mean_oriented, side="left") / max(state.rows, 1)
    denominator = np.array([span[i] for i in island.tolist()], dtype=np.float64)
    progress = np.clip(state.eval_sum[ids] / safe / np.maximum(denominator, 1.0), 0.0, 1.0)
    position = state.coord_sum[ids] / safe[:, None]

    alive = state.alive
    edge_pair = state.edge_pair[: state.edges]
    edge_live = np.flatnonzero(alive[edge_pair[:, 0]] & alive[edge_pair[:, 1]]) if state.edges else np.empty(0, np.int64)
    migration_pair = state.migration_pair[: state.migrations]
    migration_live = np.flatnonzero(alive[migration_pair[:, 0]] & alive[migration_pair[:, 1]]) if state.migrations else np.empty(0, np.int64)
    in_degree = np.bincount(edge_pair[edge_live, 1], minlength=state.nodes)[ids]
    out_degree = np.bincount(edge_pair[edge_live, 0], minlength=state.nodes)[ids]
    endpoint = np.zeros(state.nodes, dtype=bool)
    endpoint[migration_pair[migration_live].ravel()] = True

    # Islands per label, over live nodes only.
    pairs = np.unique(np.stack([label, island], axis=1), axis=0) if ids.size else np.empty((0, 2), np.int64)
    labels, per_label = np.unique(pairs[:, 0], return_counts=True)
    sharing = per_label[np.searchsorted(labels, label)] if ids.size else np.empty(0, np.int64)
    overlap = float(per_label.mean()) if per_label.size else 0.0

    island_best = np.zeros(ids.size, dtype=bool)
    overall_best = np.zeros(ids.size, dtype=bool)
    if ids.size:
        # Sorted by island, then best fitness, then id: the first of each island wins.
        order = np.lexsort((ids, best_oriented, island))
        heads = np.r_[True, island[order][1:] != island[order][:-1]]
        island_best[order[heads]] = True
        overall_best[int(np.argmin(best_oriented))] = True

    return {
        "id": ids,
        "island": island,
        "x": position[:, 0],
        "y": position[:, 1],
        "z": position[:, 2],
        "visits": visits,
        "members": state.members[ids],
        "mean_fitness": mean,
        "best_fitness": best,
        "fitness_percentile": percentile,
        "progress": progress,
        "first_eval": state.first_eval[ids],
        "last_eval": state.last_eval[ids],
        "in_degree": in_degree,
        "out_degree": out_degree,
        "islands_sharing": sharing,
        "island_best": island_best,
        "overall_best": overall_best,
        "migration_endpoint": endpoint[ids],
        "genome_hash": [state.genome_hash[i] for i in ids.tolist()],
        "weight": state.weight[ids] if state.weighted else None,
        "overlap": overlap,
        "edge_id": edge_live,
        "edge_pair": edge_pair[edge_live],
        "edge_weight": state.edge_weight[edge_live],
        "migration_id": migration_live,
        "migration_pair": migration_pair[migration_live],
        "migration_count": state.migration_count[migration_live],
    }


def _floats(values: np.ndarray) -> list[float | None]:
    return [v if np.isfinite(v) else None for v in values.tolist()]


def nodes_model(cols: dict, select: np.ndarray | slice = slice(None)) -> Nodes:
    weight = cols["weight"]
    return Nodes.model_construct(
        id=cols["id"][select].tolist(),
        island=cols["island"][select].tolist(),
        x=cols["x"][select].tolist(),
        y=cols["y"][select].tolist(),
        z=cols["z"][select].tolist(),
        visits=cols["visits"][select].tolist(),
        members=cols["members"][select].tolist(),
        mean_fitness=_floats(cols["mean_fitness"][select]),
        best_fitness=_floats(cols["best_fitness"][select]),
        fitness_percentile=cols["fitness_percentile"][select].tolist(),
        progress=cols["progress"][select].tolist(),
        first_eval=cols["first_eval"][select].tolist(),
        last_eval=cols["last_eval"][select].tolist(),
        in_degree=cols["in_degree"][select].tolist(),
        out_degree=cols["out_degree"][select].tolist(),
        islands_sharing=cols["islands_sharing"][select].tolist(),
        island_best=cols["island_best"][select].tolist(),
        overall_best=cols["overall_best"][select].tolist(),
        migration_endpoint=cols["migration_endpoint"][select].tolist(),
        genome_hash=np.asarray(cols["genome_hash"], dtype=object)[select].tolist(),
        weight=_floats(weight[select]) if weight is not None else None,
    )


def to_graph(cols: dict) -> Graph:
    return Graph.model_construct(
        nodes=nodes_model(cols),
        edges=Edges.model_construct(
            id=cols["edge_id"].tolist(),
            source=cols["edge_pair"][:, 0].tolist(),
            target=cols["edge_pair"][:, 1].tolist(),
            weight=cols["edge_weight"].tolist(),
        ),
        migrations=Migrations.model_construct(
            id=cols["migration_id"].tolist(),
            source=cols["migration_pair"][:, 0].tolist(),
            target=cols["migration_pair"][:, 1].tolist(),
            count=cols["migration_count"].tolist(),
        ),
        overlap=cols["overlap"],
    )
