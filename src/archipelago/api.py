from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .config import Config
from .errors import SchemaError
from .features import build_features
from .graph import build_graph, build_rows
from .grouping import group_baseline, group_clustered
from .layout import project
from .models import ClusteringInfo, Counts, LayoutInfo, Resolved, Result, RunInfo
from .records import parse_run
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


def cluster_run(
    evaluations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: Config | Mapping[str, Any] | None = None,
) -> Result:
    result, _ = _cluster_run(evaluations, events, config)
    return result


# cluster_run, also returning the per-row stage labels (input order, None when no stage ran)
# so the parity script can compare partitions.
def _cluster_run(
    evaluations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: Config | Mapping[str, Any] | None = None,
) -> tuple[Result, np.ndarray | None]:
    started = time.perf_counter()
    config = config if isinstance(config, Config) else Config.model_validate(config or {})
    run = parse_run(evaluations, events)
    if run.meta.genome_encoding is None:
        raise SchemaError("no island_start events and no evaluations: the genome encoding is unknown")

    warnings: list[str] = []
    features, malformed = build_features(run, config.seed)
    if malformed:
        named = ", ".join(malformed[:WARNING_ID_LIMIT])
        more = ", ..." if len(malformed) > WARNING_ID_LIMIT else ""
        warnings.append(f"{len(malformed)} evaluation(s) have a malformed genome: {named}{more}")
    total = len(run.evaluations)
    usable = features.usable
    rows_without_genome = total - int(usable.size)
    if rows_without_genome:
        warnings.append(
            f"{rows_without_genome} of {total} evaluations have no decodable genome and are left out of both graphs."
        )

    rows = build_rows(
        [run.evaluations[i] for i in usable.tolist()], run.meta.migrations, run.meta.maximising
    )
    layout = project(features.matrix[usable], features.metric, config.seed)

    baseline_group = group_baseline(features.islands[usable], [features.hashes[i] for i in usable.tolist()])
    baseline = build_graph(
        rows, baseline_group, layout.node_positions(baseline_group.row_node, baseline_group.island.size)
    )

    clustered = None
    labels = None
    diagnostics: dict[str, dict[str, Any]] = {}
    resolved: dict[str, float | None] = dict.fromkeys(Resolved.model_fields)
    if config.stages:
        assignment = run_stages(features, config)
        labels = assignment.labels
        diagnostics = {
            stage: {key: _clean(value) for key, value in values.items()}
            for stage, values in assignment.diagnostics.items()
        }
        resolved = {key: _clean(value) for key, value in assignment.resolved.items()}
        grouping, conflicts = group_clustered(baseline_group, labels[usable])
        if conflicts:
            warnings.append(
                f"{conflicts} location(s) received more than one cluster label; the majority label was used."
            )
        weights = assignment.row_weights[usable] if assignment.row_weights is not None else None
        clustered = build_graph(
            rows, grouping, layout.node_positions(grouping.row_node, grouping.island.size), weights
        )

    result = Result(
        run=RunInfo(
            run_id=run.meta.run_id,
            algorithm=run.meta.algorithm,
            benchmark=run.meta.benchmark,
            topology=run.meta.topology,
            islands=run.islands,
            maximising=run.meta.maximising,
        ),
        layout=LayoutInfo(
            method=layout.method,
            variance_kept_2d=layout.variance_kept_2d,
            variance_kept_3d=layout.variance_kept_3d,
        ),
        baseline=baseline,
        clustered=clustered,
        clustering=ClusteringInfo(
            stages=config.stages,
            counts=Counts(
                evaluations=total,
                rows_clustered=int(usable.size) if config.stages else 0,
                baseline_nodes=int(baseline_group.island.size),
                nodes=len(clustered.nodes.island) if clustered is not None else None,
            ),
            resolved=Resolved(**resolved),
            diagnostics=diagnostics,
            elapsed_seconds=time.perf_counter() - started,
        ),
        rows_without_genome=rows_without_genome,
        warnings=warnings,
    )
    return result, labels
