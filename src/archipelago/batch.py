from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import explainability
from .config import Config
from .convergence import convergence
from .encoding.features import build_features
from .errors import SchemaError
from .graph.grouping import majority_labels
from .graph.layout import fit, layout_info
from .graph.state import (
    GraphState,
    add_edges,
    add_migrations,
    add_rows,
    columns,
    to_graph,
)
from .models import (
    ClusteringInfo,
    Counts,
    GenerationSummary,
    IslandBest,
    IslandEnd,
    MigrationArrival,
    MigrationRoute,
    Resolved,
    Result,
    RunInfo,
)
from .records import Evaluation, parse_run
from .separation import island_separation
from .stages import run_stages

# How many malformed individual ids a warning names before it truncates.
WARNING_ID_LIMIT = 5


# Cast numpy scalars to Python types and non-finite floats to None.
def _clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clean_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _clean(value) for key, value in values.items()}


def migration_routes(route_migrants: Mapping[tuple[int, int], int]) -> list[MigrationRoute]:
    return [MigrationRoute(source=source, destination=destination, migrants=count) for (source, destination), count in sorted(route_migrants.items())]


def migration_arrivals(arrivals: Sequence[tuple[int, int, int, int, int]]) -> list[MigrationArrival]:
    return [MigrationArrival(generation=g, source=s, destination=d, migrants=m, source_generation=sent) for g, s, d, m, sent in sorted(arrivals)]


def generation_history(summaries: Mapping[tuple[int, int], GenerationSummary]) -> list[GenerationSummary]:
    return [summary for _, summary in sorted(summaries.items())]


def island_ends(ends: Mapping[int, IslandEnd]) -> list[IslandEnd]:
    return [end for _, end in sorted(ends.items())]


# The islands' best rows that columns() picked, and the island holding the run best. evaluations[r] is graph row r.
def island_bests(
    state: GraphState, cols: Mapping[str, Any], evaluations: Sequence[Evaluation], summaries: Mapping[tuple[int, int], GenerationSummary]
) -> tuple[list[IslandBest], int | None]:
    bests = []
    for island, row in sorted(cols["best_rows"].items()):
        ev = evaluations[row]
        summary = summaries.get((island, ev.generation))
        bests.append(
            IslandBest(
                island_id=island,
                fitness=float(state.row_fitness[row]),
                individual_id=ev.individual_id,
                generation=ev.generation,
                evaluations=summary.evaluations_total if summary else None,
                node=int(state.row_node[row]),
            )
        )
    run_best = cols["run_best_row"]
    return bests, evaluations[run_best].island_id if run_best is not None else None


def row_warnings(malformed: list[str], total: int) -> list[str]:
    if not malformed:
        return []
    named = ", ".join(malformed[:WARNING_ID_LIMIT])
    more = ", ..." if len(malformed) > WARNING_ID_LIMIT else ""
    return [f"{len(malformed)} of {total} evaluations have a malformed genome and are left out of the graph: {named}{more}"]


# Batch mode: cluster a finished run's logs in one go, with whole-run calibration.
def cluster_run(
    evaluations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: Config | Mapping[str, Any] | None = None,
) -> Result:
    result, _ = _cluster_run(evaluations, events, config)
    return result


# cluster_run, also returning a trace of the per-row labels, node labels and layout
# coordinates, so the verification scripts can rebuild the graph independently.
def _cluster_run(
    evaluations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: Config | Mapping[str, Any] | None = None,
) -> tuple[Result, dict[str, Any]]:
    started = time.perf_counter()
    config = config if isinstance(config, Config) else Config.model_validate(config or {})
    run = parse_run(evaluations, events)
    if run.meta.genome_encoding is None:
        raise SchemaError("no island_start events and no evaluations: the genome encoding is unknown")
    analyses = explainability.Analyses()
    explainability.add(analyses, evaluations)

    features, malformed = build_features(run, config.seed)
    total = len(run.evaluations)
    usable = features.usable
    warnings = row_warnings(malformed, total)

    assignment = run_stages(features, config)
    labels = assignment.labels
    diagnostics = {stage: clean_values(values) for stage, values in assignment.diagnostics.items()}
    hashes = [features.hashes[i] for i in usable.tolist()]
    node_labels, conflicts = majority_labels(features.islands[usable], hashes, labels[usable])
    if conflicts:
        warnings.append(f"{conflicts} location(s) received more than one cluster label; the majority label was used.")

    projection, coords = fit(features.matrix[usable], features.metric, config.seed)
    used = [run.evaluations[i] for i in usable.tolist()]
    state = GraphState(maximising=run.meta.maximising, weighted=assignment.row_weights is not None)
    add_rows(
        state,
        [ev.individual_id for ev in used],
        features.islands[usable],
        node_labels,
        np.array([ev.eval_index for ev in used], dtype=np.int64),
        np.array([ev.fitness for ev in used], dtype=np.float64),
        hashes,
        coords,
    )
    # Parents on another island are migrated copies; migrations cover those.
    lineage = [
        (parent, child)
        for child, ev in enumerate(used)
        for parent in (state.row_of.get(parent_id) for parent_id in ev.parent_ids)
        if parent is not None and used[parent].island_id == ev.island_id
    ]
    add_edges(state, lineage)
    migrant_pairs = [(origin, arrived) for origin, arrived in run.meta.migrations if origin in state.row_of and arrived in state.row_of]
    add_migrations(state, [(state.row_of[origin], state.row_of[arrived]) for origin, arrived in migrant_pairs])
    dead: list[int] = []
    if assignment.row_weights is not None:
        np.maximum.at(state.weight, state.row_node[: state.rows], assignment.row_weights[usable])
        # A node fades when DenStream pruned the micro-cluster of every one of its rows by the
        # end of the run: those rows carry weight 0.
        dead = np.flatnonzero(state.weight[: state.nodes] <= 0).tolist()
        state.alive[dead] = False

    span: dict[int, int] = {}
    for ev in run.evaluations:
        span[ev.island_id] = max(span.get(ev.island_id, 0), ev.eval_index + 1)
    cols = columns(state, span)
    graph = to_graph(cols)
    bests, run_best_island = island_bests(state, cols, used, run.meta.generation_summaries)

    run_info = RunInfo(
        run_id=run.meta.run_id,
        algorithm=run.meta.algorithm,
        benchmark=run.meta.benchmark,
        migration_topology=run.meta.migration_topology,
        islands=run.islands,
        maximising=run.meta.maximising,
        budget_per_island=run.meta.budget_per_island,
        stagnation_generations=run.meta.stagnation_generations,
    )
    result = Result(
        run=run_info,
        layout=layout_info(projection),
        graph=graph,
        separation=island_separation(
            features.matrix[usable],
            features.islands[usable],
            features.generations[usable],
            np.array([ev.seq for ev in used], dtype=np.int64),
            config.seed,
        ),
        clustering=ClusteringInfo(
            stages=config.stages,
            counts=Counts(
                evaluations=total,
                rows_clustered=int(usable.size),
                nodes=len(graph.nodes.id),
                faded_nodes=len(dead) - int(cols["faded"].sum()) if assignment.row_weights is not None else None,
            ),
            resolved=Resolved(**clean_values(assignment.resolved)),
            diagnostics=diagnostics,
            elapsed_seconds=time.perf_counter() - started,
        ),
        migration_routes=migration_routes(run.meta.route_migrants),
        migration_arrivals=migration_arrivals(run.meta.arrivals),
        generation_history=generation_history(run.meta.generation_summaries),
        island_ends=island_ends(run.meta.island_ends),
        island_bests=bests,
        run_best_island=run_best_island,
        convergence=convergence(list(run.meta.generation_summaries.values()), run.meta.maximising, list(run.meta.island_ends.values())),
        explainability=explainability.finish(analyses, run_info),
        malformed_rows=len(malformed),
        warnings=warnings,
    )
    node_label_of = dict(zip(usable.tolist(), node_labels.tolist()))
    trace = {
        "labels": labels,
        "row_weights": assignment.row_weights,
        "rows": [
            {
                "individual_id": ev.individual_id,
                "island_id": ev.island_id,
                "seq": ev.seq,
                "eval_index": ev.eval_index,
                "generation": ev.generation,
                "fitness": ev.fitness,
                "genome_hash": ev.genome_hash,
                "parent_ids": list(ev.parent_ids),
                "usable": bool(features.mask[i]),
                "stage_label": int(labels[i]) if features.mask[i] else None,
                "label": node_label_of.get(i),
            }
            for i, ev in enumerate(run.evaluations)
        ],
        "coords": coords,
        "dead_nodes": dead,
        "migration_pairs": migrant_pairs,
        "maximising": run.meta.maximising,
        "budget_per_island": run.meta.budget_per_island,
        "config": config,
    }
    return result, trace
