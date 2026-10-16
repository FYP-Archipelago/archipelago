
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config
from ..types import FeatureSet


class Assignment:
    def __init__(
        self,
        labels: np.ndarray,
        diagnostics: dict[str, Any],
        row_weights: np.ndarray | None = None,
    ):
        self.labels = labels
        self.diagnostics = diagnostics
        self.row_weights = row_weights

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels).size) if self.labels.size else 0


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Squash arbitrary label values down to 0..k-1, keeping first-seen order."""
    _, first_index, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first_index)
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    return remap[inverse]


def run_stages(features: FeatureSet, config: Config) -> Assignment:
    """Run the enabled stages in cascade order over the usable rows."""
    from .stages.birch import BirchStage
    from .stages.denstream import DenStreamStage
    from .stages.lsh import LSHStage

    n = len(features)
    diagnostics: dict[str, Any] = {}
    groups = np.zeros(n, dtype=np.int64)

    stages = []
    if config.lsh.enabled:
        stages.append(LSHStage(config))
    if config.birch.enabled:
        stages.append(BirchStage(config))
    if config.denstream.enabled:
        stages.append(DenStreamStage(config))

    row_weights = None
    for stage in stages:
        groups = _relabel(stage.fit_predict(features, groups))
        diagnostics[stage.name] = stage.diagnostics()
        diagnostics[stage.name]["clusters_after"] = int(np.unique(groups).size)
        produced = getattr(stage, "row_weights", None)
        if produced is not None:
            row_weights = produced

    return Assignment(groups, diagnostics, row_weights)
