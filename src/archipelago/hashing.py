from __future__ import annotations

import hashlib

import numpy as np


def _digest(token: bytes, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(token, digest_size=8, salt=seed.to_bytes(8, "little")).digest(),
        "little",
    )


def hash_tokens(tokens: list[bytes], seed: int) -> np.ndarray:
    """Stable integer ids for set elements, so MinHash and the hashing agree."""
    return np.array([_digest(t, seed) for t in tokens], dtype=np.uint64)


def feature_hash(
    element_ids: np.ndarray, dimension: int, seed: int, normalise: bool = True
) -> np.ndarray:
    """Project a set of element ids into `dimension` dims with signed hashing."""
    vector = np.zeros(dimension, dtype=np.float64)
    if element_ids.size:
        mixed = (element_ids * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)
        index = (mixed % np.uint64(dimension)).astype(np.int64)
        sign = np.where((mixed >> np.uint64(63)) & np.uint64(1), -1.0, 1.0)
        np.add.at(vector, index, sign)
    if normalise:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
    return vector


class MinHasher:
    """Classic (a*x + b) mod p MinHash. P[collision] is exactly the Jaccard."""

    MERSENNE_PRIME = (1 << 61) - 1

    def __init__(self, num_perm: int, seed: int):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, self.MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, self.MERSENNE_PRIME, size=num_perm, dtype=np.uint64)

    def signature(self, element_ids: np.ndarray) -> np.ndarray:
        if element_ids.size == 0:
            return np.full(self.num_perm, np.uint64(self.MERSENNE_PRIME), dtype=np.uint64)
        values = element_ids.astype(np.uint64).reshape(-1, 1)
        hashed = (values * self.a + self.b) % np.uint64(self.MERSENNE_PRIME)
        return hashed.min(axis=0)
