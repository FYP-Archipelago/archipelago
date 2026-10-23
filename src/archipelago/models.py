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
    # Nodes DenStream faded out (their micro-cluster was pruned); None without DenStream.
    faded_nodes: int | None


class Resolved(BaseModel):
    birch_threshold: float | None
    denstream_epsilon: float | None
    denstream_half_life: float | None
    lsh_bucket_width: float | None


class ClusteringInfo(BaseModel):
    stages: list[str]
    counts: Counts
    resolved: Resolved
    diagnostics: dict[str, dict[str, float | int | str | bool | list[float] | None]]
    elapsed_seconds: float


# One entry per island pair over the whole run. mmd2: how different the two populations are
# (0 = identical); p: generation-swap test, small = a real difference.
class SeparationPairs(BaseModel):
    a: list[int]
    b: list[int]
    mmd2: list[float]
    p: list[float]


# One entry per window of generations, each with its own distance scale.
class SeparationWindows(BaseModel):
    from_generation: list[int]
    typical_distance: list[float]


# One entry per (window, island pair) with both islands present; window indexes into windows.
class SeparationWindowPairs(BaseModel):
    window: list[int]
    a: list[int]
    b: list[int]
    mmd2: list[float]


# How separate the islands are in genome space (MMD via random Fourier features).
class Separation(BaseModel):
    method: Literal["mmd-random-fourier-features"]
    # Typical distance between two genomes in the run: the scale MMD measures against.
    typical_distance: float | None
    pairs: SeparationPairs
    windows: SeparationWindows
    window_pairs: SeparationWindowPairs


class Result(BaseModel):
    run: RunInfo
    layout: LayoutInfo
    graph: Graph
    clustering: ClusteringInfo
    separation: Separation
    rows_without_genome: int
    warnings: list[str]


# --- streaming updates: what changed since the previous update ---


# Live nodes already sent whose stats changed (exact comparison).
class NodeChanges(BaseModel):
    id: list[int]
    x: list[float]
    y: list[float]
    z: list[float]
    visits: list[int]
    members: list[int]
    mean_fitness: list[float | None]
    best_fitness: list[float | None]
    progress: list[float]
    first_eval: list[int]
    last_eval: list[int]
    in_degree: list[int]
    out_degree: list[int]
    islands_sharing: list[int]
    migration_endpoint: list[bool]


# Every node's rank drifts as each new row arrives, so percentiles travel on their own and only
# when one moved by more than DRIFT_TOLERANCE since it was last sent.
class PercentileChanges(BaseModel):
    id: list[int]
    fitness_percentile: list[float]


# DenStream weights that moved by more than WEIGHT_TOLERANCE (relative); they decay every tick.
class WeightChanges(BaseModel):
    id: list[int]
    weight: list[float]


# Every already-sent live node's position, after the layout was refitted.
class Positions(BaseModel):
    id: list[int]
    x: list[float]
    y: list[float]
    z: list[float]


class EdgeChanges(BaseModel):
    id: list[int]
    weight: list[int]


class MigrationChanges(BaseModel):
    id: list[int]
    count: list[int]


# Apply in this order: add new_nodes; overwrite updated_nodes, percentiles, weights and positions; drop
# removed_nodes with every edge and migration touching them; add new edges and migrations;
# overwrite their updates; set the bests and overlap.
class GraphDelta(BaseModel):
    new_nodes: Nodes
    updated_nodes: NodeChanges
    percentiles: PercentileChanges
    weights: WeightChanges | None
    removed_nodes: list[int]
    positions: Positions | None
    new_edges: Edges
    edge_updates: EdgeChanges
    new_migrations: Migrations
    migration_updates: MigrationChanges
    island_best: list[int]
    overall_best: int | None
    overlap: float


class Update(BaseModel):
    status: Literal["warming_up", "live"]
    # Set when the layout was refitted in this update (positions then carries every node).
    layout: LayoutInfo | None
    clustering: ClusteringInfo
    rows_without_genome: int
    warnings: list[str]
    graph: GraphDelta
