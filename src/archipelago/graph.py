from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .grouping import Grouping
from .models import Edges, Graph, Migrations, Nodes
from .records import Evaluation


# Per-row facts about the usable rows that both graphs share, computed once.
@dataclass(frozen=True)
class RowTable:
    island: np.ndarray
    eval_index: np.ndarray
    fitness: np.ndarray
    # Fitness oriented so that lower is always better.
    oriented: np.ndarray
    sorted_oriented: np.ndarray
    # Position of the row in its island's eval_index order, divided by the island's row count.
    progress: np.ndarray
    genome_hash: list[str]
    # (parent row, child row) for every parent link within an island, in occurrence order.
    lineage: np.ndarray
    # (origin row, arrived row) for every migrant, in event order.
    migrations: np.ndarray
    maximising: bool


def _pairs(pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def build_rows(
    evaluations: Sequence[Evaluation], migrations: Sequence[tuple[str, str]], maximising: bool
) -> RowTable:
    island = np.array([ev.island_id for ev in evaluations], dtype=np.int64)
    eval_index = np.array([ev.eval_index for ev in evaluations], dtype=np.int64)
    fitness = np.array([ev.fitness for ev in evaluations], dtype=np.float64)
    oriented = -fitness if maximising else fitness

    progress = np.zeros(len(evaluations))
    if len(evaluations):
        # lexsort is stable, so equal eval_index values keep input order.
        order = np.lexsort((eval_index, island))
        sorted_islands = island[order]
        starts = np.flatnonzero(np.r_[True, sorted_islands[1:] != sorted_islands[:-1]])
        sizes = np.diff(np.r_[starts, sorted_islands.size])
        start_of = np.repeat(starts, sizes)
        size_of = np.repeat(sizes, sizes)
        progress[order] = (np.arange(order.size) - start_of) / size_of

    row_of = {ev.individual_id: row for row, ev in enumerate(evaluations)}
    lineage: list[tuple[int, int]] = []
    for child, ev in enumerate(evaluations):
        for parent_id in ev.parent_ids:
            parent = row_of.get(parent_id)
            # Parents on another island are migrated copies; migrations cover those.
            if parent is not None and island[parent] == island[child]:
                lineage.append((parent, child))

    migrant_rows = [
        (row_of[origin], row_of[arrived])
        for origin, arrived in migrations
        if origin in row_of and arrived in row_of
    ]

    return RowTable(
        island=island,
        eval_index=eval_index,
        fitness=fitness,
        oriented=oriented,
        sorted_oriented=np.sort(oriented),
        progress=progress,
        genome_hash=[ev.genome_hash for ev in evaluations],
        lineage=_pairs(lineage),
        migrations=_pairs(migrant_rows),
        maximising=maximising,
    )


# Map row pairs onto node pairs, drop self-loops, and count repeats in first-occurrence order.
def _aggregate(row_pairs: np.ndarray, row_node: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = row_node[row_pairs]
    nodes = nodes[nodes[:, 0] != nodes[:, 1]]
    if nodes.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    unique, first, counts = np.unique(nodes, axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)
    return unique[order], counts[order]


def _floats(values: np.ndarray) -> list[float | None]:
    return [v if math.isfinite(v) else None for v in values.tolist()]


def build_graph(
    rows: RowTable,
    grouping: Grouping,
    positions: np.ndarray,
    row_weights: np.ndarray | None = None,
) -> Graph:
    row_node = grouping.row_node
    k = grouping.island.size
    m = row_node.size
    visits = np.bincount(row_node, minlength=k)
    safe_visits = np.maximum(visits, 1)

    mean_fitness = np.bincount(row_node, weights=rows.fitness, minlength=k) / safe_visits
    best_oriented = np.full(k, np.inf)
    np.minimum.at(best_oriented, row_node, rows.oriented)
    best_fitness = -best_oriented if rows.maximising else best_oriented
    mean_oriented = -mean_fitness if rows.maximising else mean_fitness
    percentile = np.searchsorted(rows.sorted_oriented, mean_oriented, side="left") / max(m, 1)
    progress = np.bincount(row_node, weights=rows.progress, minlength=k) / safe_visits

    first_eval = np.full(k, np.iinfo(np.int64).max)
    last_eval = np.full(k, np.iinfo(np.int64).min)
    np.minimum.at(first_eval, row_node, rows.eval_index)
    np.maximum.at(last_eval, row_node, rows.eval_index)

    _, first_row = np.unique(row_node, return_index=True)

    edge_pairs, edge_weights = _aggregate(rows.lineage, row_node)
    migration_pairs, migration_counts = _aggregate(rows.migrations, row_node)
    in_degree = np.bincount(edge_pairs[:, 1], minlength=k)
    out_degree = np.bincount(edge_pairs[:, 0], minlength=k)
    endpoint = np.zeros(k, dtype=bool)
    endpoint[migration_pairs.ravel()] = True

    islands_of: dict[str | int, set[int]] = {}
    for identity, island in zip(grouping.identity, grouping.island.tolist()):
        islands_of.setdefault(identity, set()).add(island)
    sharing = [len(islands_of[identity]) for identity in grouping.identity]
    overlap = float(np.mean([len(s) for s in islands_of.values()])) if islands_of else 0.0

    island_best = np.zeros(k, dtype=bool)
    overall_best = np.zeros(k, dtype=bool)
    if k:
        # Sorted by island, then best fitness, then node index: the first of each island wins.
        order = np.lexsort((np.arange(k), best_oriented, grouping.island))
        heads = np.r_[True, grouping.island[order][1:] != grouping.island[order][:-1]]
        island_best[order[heads]] = True
        overall_best[int(np.argmin(best_oriented))] = True

    weight = None
    if row_weights is not None:
        node_weight = np.full(k, -np.inf)
        np.maximum.at(node_weight, row_node, row_weights)
        weight = _floats(node_weight)

    nodes = Nodes.model_construct(
        island=grouping.island.tolist(),
        x=positions[:, 0].tolist(),
        y=positions[:, 1].tolist(),
        z=positions[:, 2].tolist(),
        visits=visits.tolist(),
        members=grouping.members.tolist(),
        mean_fitness=_floats(mean_fitness),
        best_fitness=_floats(best_fitness),
        fitness_percentile=percentile.tolist(),
        progress=progress.tolist(),
        first_eval=first_eval.tolist(),
        last_eval=last_eval.tolist(),
        in_degree=in_degree.tolist(),
        out_degree=out_degree.tolist(),
        islands_sharing=sharing,
        island_best=island_best.tolist(),
        overall_best=overall_best.tolist(),
        migration_endpoint=endpoint.tolist(),
        genome_hash=[rows.genome_hash[row] for row in first_row.tolist()],
        weight=weight,
    )
    return Graph.model_construct(
        nodes=nodes,
        edges=Edges.model_construct(
            source=edge_pairs[:, 0].tolist(),
            target=edge_pairs[:, 1].tolist(),
            weight=edge_weights.tolist(),
        ),
        migrations=Migrations.model_construct(
            source=migration_pairs[:, 0].tolist(),
            target=migration_pairs[:, 1].tolist(),
            count=migration_counts.tolist(),
        ),
        overlap=overlap,
    )
