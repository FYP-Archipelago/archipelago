from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


# How the usable rows (in input order) fall into graph nodes.
@dataclass(frozen=True)
class Grouping:
    # Node index of every usable row.
    row_node: np.ndarray
    # Island of every node.
    island: np.ndarray
    # What makes nodes on different islands "the same place": the genome hash on the
    # baseline, the cluster label on the clustered graph. Drives islands_sharing and overlap.
    identity: list[str] | list[int]
    # Baseline nodes merged into every node (all 1 on the baseline).
    members: np.ndarray


# One node per distinct (island, genome hash), in order of first appearance.
def group_baseline(islands: np.ndarray, hashes: list[str]) -> Grouping:
    index: dict[tuple[int, str], int] = {}
    row_node = np.empty(len(hashes), dtype=np.int64)
    for row, key in enumerate(zip(islands.tolist(), hashes)):
        row_node[row] = index.setdefault(key, len(index))
    keys = list(index)
    return Grouping(
        row_node=row_node,
        island=np.array([island for island, _ in keys], dtype=np.int64),
        identity=[genome_hash for _, genome_hash in keys],
        members=np.ones(len(keys), dtype=np.int64),
    )


# Each baseline node takes its majority stage label (ties go to the first-seen label), then
# nodes are grouped by (island, label) in order of first appearance of a member row.
# Returns the grouping and how many baseline nodes had conflicting labels.
def group_clustered(baseline: Grouping, labels: np.ndarray) -> tuple[Grouping, int]:
    per_node: list[list[int]] = [[] for _ in range(baseline.island.size)]
    for node, label in zip(baseline.row_node.tolist(), labels.tolist()):
        per_node[node].append(label)

    conflicts = 0
    node_label: list[int] = []
    for values in per_node:
        if len(set(values)) > 1:
            conflicts += 1
            node_label.append(Counter(values).most_common(1)[0][0])
        else:
            node_label.append(values[0])

    index: dict[tuple[int, int], int] = {}
    baseline_to_clustered = np.full(baseline.island.size, -1, dtype=np.int64)
    for node in baseline.row_node.tolist():
        if baseline_to_clustered[node] < 0:
            key = (int(baseline.island[node]), node_label[node])
            baseline_to_clustered[node] = index.setdefault(key, len(index))

    keys = list(index)
    grouping = Grouping(
        row_node=baseline_to_clustered[baseline.row_node],
        island=np.array([island for island, _ in keys], dtype=np.int64),
        identity=[label for _, label in keys],
        members=np.bincount(baseline_to_clustered, minlength=len(keys)),
    )
    return grouping, conflicts
