from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from . import genome as genome_mod
from .hashing import feature_hash, hash_tokens
from .records import Run

PERMUTATION_DIMENSION = 64
TREE_DIMENSION = 128

# What a malformed genome_repr can raise while decoding or encoding.
DECODE_ERRORS = (ValueError, TypeError, KeyError, IndexError, struct.error)


# The space an encoding lives in. Decides which LSH family and layout apply.
class Metric(str, Enum):
    L2 = "l2"
    HAMMING = "hamming"
    JACCARD = "jaccard"


# Vectorised genomes for every row in input order, plus the sparse sets LSH needs for
# Jaccard spaces. Rows outside `mask` have no decodable genome and a zero vector.
@dataclass
class FeatureSet:
    matrix: np.ndarray
    metric: Metric
    hashes: list[str]
    mask: np.ndarray
    times: np.ndarray
    islands: np.ndarray
    sets: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.hashes)

    # Indices of the rows that can actually be clustered.
    @property
    def usable(self) -> np.ndarray:
        return np.flatnonzero(self.mask)


class Adapter(Protocol):
    metric: Metric
    dimension: int

    # Dense vector, plus element ids for the set-valued spaces.
    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]: ...


# Continuous positions, min-max normalised onto the unit cube.
class RealVectorAdapter:
    metric = Metric.L2

    def __init__(self, encoding: str, dimension: int, lower: float | None, upper: float | None):
        self.encoding = encoding
        self.dimension = dimension
        self.lower = lower
        self.upper = upper
        self.normalised = lower is not None and upper is not None and upper > lower

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.asarray(genome_mod.position(genome, self.encoding), dtype=np.float64)
        if self.normalised:
            x = (x - self.lower) / (self.upper - self.lower)
        return x, None


class BitstringAdapter:
    metric = Metric.HAMMING

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        return np.asarray(genome, dtype=np.float64), None


# A tour as its set of undirected edges.
class PermutationAdapter:
    metric = Metric.JACCARD

    def __init__(self, num_cities: int, seed: int):
        self.num_cities = num_cities
        self.dimension = PERMUTATION_DIMENSION
        self.seed = seed

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        tour = np.asarray(genome, dtype=np.int64)
        nxt = np.roll(tour, -1)
        low = np.minimum(tour, nxt)
        high = np.maximum(tour, nxt)
        ids = (low * self.num_cities + high).astype(np.uint64)
        ids = np.unique(ids)
        return feature_hash(ids, self.dimension, self.seed), ids


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|-?\d+\.?\d*|[(),]")


# Every subtree of a canonical prefix expression, as its own string.
def subtree_shingles(expression: str) -> list[str]:
    tokens = _TOKEN.findall(expression)
    shingles: list[str] = []
    position = 0

    def parse() -> str:
        nonlocal position
        if position >= len(tokens):
            return ""
        head = tokens[position]
        position += 1
        if position < len(tokens) and tokens[position] == "(":
            position += 1  # consume "("
            args = []
            while position < len(tokens) and tokens[position] != ")":
                args.append(parse())
                if position < len(tokens) and tokens[position] == ",":
                    position += 1
            if position < len(tokens):
                position += 1  # consume ")"
            node = f"{head}({','.join(args)})"
        else:
            node = head
        shingles.append(node)
        return node

    parse()
    return shingles


class ExpressionTreeAdapter:
    metric = Metric.JACCARD

    def __init__(self, seed: int):
        self.dimension = TREE_DIMENSION
        self.seed = seed

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        shingles = subtree_shingles(str(genome))
        ids = np.unique(hash_tokens([s.encode() for s in shingles], self.seed))
        return feature_hash(ids, self.dimension, self.seed), ids


def build_adapter(encoding: str, benchmark_params: dict[str, Any], seed: int) -> Adapter:
    if encoding in (
        genome_mod.REAL_VECTOR,
        genome_mod.REAL_VECTOR_VELOCITY,
        genome_mod.REAL_VECTOR_SIGMA,
    ):
        return RealVectorAdapter(
            encoding,
            int(benchmark_params.get("dimension", 0)),
            benchmark_params.get("lower"),
            benchmark_params.get("upper"),
        )
    if encoding == genome_mod.BITSTRING:
        return BitstringAdapter(
            int(benchmark_params.get("num_items", benchmark_params.get("dimension", 0)))
        )
    if encoding == genome_mod.PERMUTATION:
        return PermutationAdapter(
            int(benchmark_params.get("num_cities", benchmark_params.get("dimension", 0))), seed
        )
    if encoding == genome_mod.EXPRESSION_TREE:
        return ExpressionTreeAdapter(seed)
    raise ValueError(f"no feature adapter for encoding: {encoding}")


# Vectorise every evaluation. Returns the features and the individual ids of rows whose
# genome_repr was present but malformed.
def build_features(run: Run, seed: int) -> tuple[FeatureSet, list[str]]:
    encoding = run.meta.genome_encoding or ""
    adapter = build_adapter(encoding, run.meta.benchmark_params, seed)

    vectors: list[np.ndarray | None] = []
    sets: list[np.ndarray] = []
    hashes: list[str] = []
    mask: list[bool] = []
    times: list[float] = []
    islands: list[int] = []
    malformed: list[str] = []
    width = 0

    for ev in run.evaluations:
        hashes.append(ev.genome_hash)
        times.append(run.corrected_time(ev))
        islands.append(ev.island_id)
        try:
            decoded = genome_mod.decode(ev.genome_repr, ev.genome_repr_mode, ev.genome_encoding)
            encoded = adapter.encode(decoded) if decoded is not None else None
        except DECODE_ERRORS:
            malformed.append(ev.individual_id)
            encoded = None
        if encoded is None:
            vectors.append(None)
            sets.append(np.empty(0, dtype=np.uint64))
            mask.append(False)
            continue
        vector, ids = encoded
        width = max(width, vector.size)
        vectors.append(vector)
        sets.append(ids if ids is not None else np.empty(0, dtype=np.uint64))
        mask.append(True)

    matrix = np.zeros((len(vectors), width), dtype=np.float64)
    for i, vector in enumerate(vectors):
        if vector is not None:
            matrix[i, : vector.size] = vector

    features = FeatureSet(
        matrix=matrix,
        metric=adapter.metric,
        hashes=hashes,
        mask=np.asarray(mask, dtype=bool),
        times=np.asarray(times, dtype=np.float64),
        islands=np.asarray(islands, dtype=np.int64),
        sets=sets if adapter.metric is Metric.JACCARD else None,
    )
    return features, malformed
