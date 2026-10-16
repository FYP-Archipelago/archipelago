from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..config import Config
from ..features import FeatureSet
from .birch import BirchStage
from .denstream import DenStreamStage
from .lsh import LSHStage


class ClusterStage(Protocol):
    name: str

    # Label every row, refining the partition in `groups`.
    def fit_predict(self, features: FeatureSet, groups: np.ndarray) -> np.ndarray: ...

    def diagnostics(self) -> dict[str, Any]: ...


@dataclass
class Assignment:
    # Per-row cluster labels in input order; only usable rows carry meaningful labels.
    labels: np.ndarray
    diagnostics: dict[str, dict[str, Any]]
    # The auto or overridden values the stages actually used; None for stages that didn't run.
    resolved: dict[str, float | None]
    row_weights: np.ndarray | None = None


# Squash arbitrary label values down to 0..k-1, keeping first-seen order.
def _relabel(labels: np.ndarray) -> np.ndarray:
    _, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first_index)
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    return remap[inverse]


# Run the enabled stages in cascade order (LSH -> BIRCH -> DenStream) over the usable rows.
def run_stages(features: FeatureSet, config: Config) -> Assignment:
    n = len(features)
    diagnostics: dict[str, dict[str, Any]] = {}
    groups = np.zeros(n, dtype=np.int64)

    stages: list[ClusterStage] = []
    if config.lsh:
        stages.append(LSHStage(config))
    if config.birch:
        stages.append(BirchStage(config))
    if config.denstream:
        stages.append(DenStreamStage(config))

    row_weights = None
    for stage in stages:
        groups = _relabel(stage.fit_predict(features, groups))
        diagnostics[stage.name] = stage.diagnostics()
        diagnostics[stage.name]["clusters_after"] = int(np.unique(groups).size)
        produced = getattr(stage, "row_weights", None)
        if produced is not None:
            row_weights = produced

    resolved = {
        "birch_threshold": diagnostics.get("birch", {}).get("threshold"),
        "denstream_epsilon": diagnostics.get("denstream", {}).get("epsilon"),
        "denstream_half_life": diagnostics.get("denstream", {}).get("half_life"),
        "lsh_bucket_width": diagnostics.get("lsh", {}).get("bucket_width"),
    }
    return Assignment(groups, diagnostics, resolved, row_weights)
