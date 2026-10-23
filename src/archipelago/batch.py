from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .config import Config
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
from .models import ClusteringInfo, Counts, Resolved, Result, RunInfo
from .records import parse_run
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


# Warnings for rows that can't be clustered: malformed genomes, then rows with no genome at all.
def row_warnings(malformed: list[str], total: int, missing: int) -> list[str]:
    warnings = []
    if malformed:
        named = ", ".join(malformed[:WARNING_ID_LIMIT])
        more = ", ..." if len(malformed) > WARNING_ID_LIMIT else ""
        warnings.append(f"{len(malformed)} evaluation(s) have a malformed genome: {named}{more}")
    if missing:
        warnings.append(f"{missing} of {total} evaluations have no decodable genome and are left out of the graph.")
    return warnings


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

    features, malformed = build_features(run, config.seed)
    total = len(run.evaluations)
    usable = features.usable
    rows_without_genome = total - int(usable.size)
    warnings = row_warnings(malformed, total, rows_without_genome)

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
    migrant_pairs = [
        (origin, arrived) for origin, arrived in run.meta.migrations if origin in state.row_of and arrived in state.row_of
    ]
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

    result = Result(
        run=RunInfo(
            run_id=run.meta.run_id,
            algorithm=run.meta.algorithm,
            benchmark=run.meta.benchmark,
            topology=run.meta.topology,
            islands=run.islands,
            maximising=run.meta.maximising,
        ),
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
                faded_nodes=len(dead) if assignment.row_weights is not None else None,
            ),
            resolved=Resolved(**clean_values(assignment.resolved)),
            diagnostics=diagnostics,
            elapsed_seconds=time.perf_counter() - started,
        ),
        rows_without_genome=rows_without_genome,
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
