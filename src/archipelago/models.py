from typing import Literal

from pydantic import BaseModel

# The graph models hold tens of thousands of list elements. They are built with
# model_construct() from lists that are already clean, so pydantic never re-validates them.


# Columnar node table: node i is position i in every list, in id order. Ids are stable for
# the whole run and never reused; faded nodes leave gaps.
class Nodes(BaseModel):
    id: list[int]
    island: list[int]
    x: list[float]
    y: list[float]
    z: list[float]
    visits: list[int]
    members: list[int]
    mean_fitness: list[float | None]
    best_fitness: list[float | None]
    fitness_percentile: list[float]
    progress: list[float]
    first_eval: list[int]
    last_eval: list[int]
    in_degree: list[int]
    out_degree: list[int]
    islands_sharing: list[int]
    island_best: list[bool]
    overall_best: list[bool]
    migration_endpoint: list[bool]
    genome_hash: list[str]
    weight: list[float | None] | None


# Parent -> child lineage within an island; weight = how many times the step was taken.
# source and target are node ids.
class Edges(BaseModel):
    id: list[int]
    source: list[int]
    target: list[int]
    weight: list[int]


# Migrant origin node -> arrival node; count = migrants that took this route.
# source and target are node ids.
class Migrations(BaseModel):
    id: list[int]
    source: list[int]
    target: list[int]
    count: list[int]


class Graph(BaseModel):
    nodes: Nodes
    edges: Edges
    migrations: Migrations
    # Mean number of islands per occupied cluster.
    overlap: float


class RunInfo(BaseModel):
    run_id: str
    algorithm: str
    benchmark: str
    topology: str | None
    islands: list[int]
    maximising: bool


class LayoutInfo(BaseModel):
    method: Literal["pca", "mds", "none"]
    variance_kept_2d: float
    variance_kept_3d: float


class Counts(BaseModel):
    evaluations: int
    rows_clustered: int
    nodes: int


class Resolved(BaseModel):
    birch_threshold: float | None
    denstream_epsilon: float | None
    denstream_half_life: float | None
    lsh_bucket_width: float | None


class ClusteringInfo(BaseModel):
    stages: list[str]
    counts: Counts
    resolved: Resolved
    diagnostics: dict[str, dict[str, float | int | str | bool | None]]
    elapsed_seconds: float


class Result(BaseModel):
    run: RunInfo
    layout: LayoutInfo
    graph: Graph
    clustering: ClusteringInfo
    rows_without_genome: int
    warnings: list[str]
