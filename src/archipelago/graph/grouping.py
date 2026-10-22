from __future__ import annotations

from collections import Counter

import numpy as np


# Batch mode: give every row the majority stage label of its (island, genome hash), so one
# genome on one island is always one node. Ties go to the first-seen label. Returns the label
# per row and how many (island, genome hash) keys had conflicting labels.
def majority_labels(islands: np.ndarray, hashes: list[str], labels: np.ndarray) -> tuple[np.ndarray, int]:
    per_key: dict[tuple[int, str], list[int]] = {}
    keys = list(zip(islands.tolist(), hashes))
    for key, label in zip(keys, labels.tolist()):
        per_key.setdefault(key, []).append(label)

    conflicts = 0
    chosen: dict[tuple[int, str], int] = {}
    for key, values in per_key.items():
        if len(set(values)) > 1:
            conflicts += 1
            chosen[key] = Counter(values).most_common(1)[0][0]
        else:
            chosen[key] = values[0]
    return np.array([chosen[key] for key in keys], dtype=np.int64), conflicts
