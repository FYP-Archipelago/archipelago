from __future__ import annotations

import math

import numpy as np

# Relative slack on the fast squared distances used to shortlist nearest-centre candidates. The
# real rounding error is around 1e-14 for these dimensions; anything within this margin of the
# fast minimum is re-measured exactly.
SHORTLIST_MARGIN = 1e-10


# Radius of a cluster feature (total weight, linear sum, square sum) after adding a weighted point.
def radius_if_added(weight: float, cf1: np.ndarray, cf2: float, point: np.ndarray, add: float) -> float:
    w = weight + add
    linear = cf1 + point * add
    square = cf2 + float(point @ point) * add
    centre = linear / w
    variance = square / w - float(centre @ centre)
    return math.sqrt(max(variance, 0.0))


# Index of the centre nearest to point by squared Euclidean distance, ties to the lowest index.
# A BLAS pass only shortlists; the choice is made on exactly computed distances, so it never
# depends on how rows were batched or which BLAS build ran.
def nearest(centres: np.ndarray, norms: np.ndarray, point: np.ndarray) -> int:
    if centres.shape[0] == 1:
        return 0
    point_norm = float(point @ point)
    fast = norms - 2.0 * (centres @ point) + point_norm
    margin = SHORTLIST_MARGIN * (point_norm + float(norms.max()) + 1e-300)
    shortlist = np.flatnonzero(fast <= fast.min() + margin)
    if shortlist.size == 1:
        return int(shortlist[0])
    exact = ((centres[shortlist] - point) ** 2).sum(axis=1)
    return int(shortlist[int(np.argmin(exact))])
