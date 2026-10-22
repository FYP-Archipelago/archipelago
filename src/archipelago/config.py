from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- developer constants (not settable from the UI; edit here to change) ---

# Size of the MinHash permutation pool for tour and expression-tree runs. Only the first
# lsh_hashes are used, so changing this shuffles which functions you get, but does not make
# blocks finer or coarser.
NUM_PERM = 128

# Bitstring runs sample this many bits per LSH hash instead of one projection.
# Higher -> longer block key -> more blocks.
HAMMING_BIT_MULTIPLIER = 4

# How many children a BIRCH CF-tree node holds before it splits.
# Higher -> shallower tree, slightly fewer subclusters.
BIRCH_BRANCHING_FACTOR = 50

# How many rows are sampled when resolving any "auto" distance.
# Higher -> steadier, more repeatable thresholds, but slower to compute.
CALIBRATION_SAMPLE_SIZE = 2000

# Which percentile of nearest-neighbour distance becomes the auto BIRCH threshold and
# DenStream epsilon. Higher -> bigger radius -> more merging -> fewer nodes.
NEIGHBOUR_PERCENTILE = 50.0

# Which percentile of random-pair distances becomes the auto LSH bucket width (real-vector
# runs only). Higher -> wider buckets -> fewer, bigger blocks.
LSH_WIDTH_PERCENTILE = 20.0

# Auto DenStream half-life = this x the time span of the points it streams.
# Shorter -> the past fades faster -> more pruning, fewer surviving clusters.
HALF_LIFE_FRACTION = 0.25

# How many points classical MDS embeds exactly before extending to the rest.
# Higher -> more faithful projection, but the cost grows with the square of this.
MDS_LANDMARKS = 1500

# Streaming: usable rows buffered before the first calibration and layout fit. Too few and the
# auto distances are noise; more delays the first live graph.
WARMUP_ROWS = 2000

# Streaming: calibration and the layout are redone at WARMUP_ROWS x this^k rows, because early,
# spread-out populations give distances several times larger than the converged phase.
CHECKPOINT_FACTOR = 2

# Streaming: a node's fitness_percentile drifts as every new row arrives; it is re-sent only
# when it moved by more than this (absolute). Final results are always exact.
DRIFT_TOLERANCE = 0.005

# Streaming: DenStream weights decay every update; one is re-sent only when it moved by more
# than this fraction of the last sent value, or every live node would be re-sent every update.
WEIGHT_TOLERANCE = 0.05

# Allowed range of Config.detail.
DETAIL_RANGE = (0.5, 2.0)

# Projection per feature metric: PCA is linear and suits real vectors and bitstrings,
# MDS suits tours and expression trees.
LAYOUT_BY_METRIC = {"l2": "pca", "hamming": "pca", "jaccard": "mds"}


# UI-facing settings. At least one stage must be enabled.
# Stage order is fixed (LSH -> BIRCH -> DenStream); the flags only choose which ones run.
# None on a distance means auto: measured from the run's own data, then divided by detail.
# An explicit value is used as-is and ignores detail.
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lsh: bool = False
    birch: bool = False
    denstream: bool = False

    # Divides the auto BIRCH threshold and auto DenStream epsilon.
    # >1 -> finer, more and smaller nodes. <1 -> coarser, fewer and bigger nodes.
    detail: float = Field(default=1.0, ge=DETAIL_RANGE[0], le=DETAIL_RANGE[1])

    # The furthest a point can sit from a subcluster's centre and still join it.
    # Bigger -> points merge more readily -> fewer, larger nodes.
    birch_threshold: float | None = Field(default=None, gt=0)

    # The furthest a point can sit from a micro-cluster's centre and still join it.
    # Bigger -> merges across a wider area -> far fewer nodes.
    denstream_epsilon: float | None = Field(default=None, gt=0)

    # Seconds for a point's contribution to count for half.
    # Shorter -> the past fades faster -> more pruning, fewer surviving clusters.
    denstream_half_life: float | None = Field(default=None, gt=0)

    # Sets the survival bar for a micro-cluster, together with beta.
    # Higher -> harder to survive -> MORE pruning, not less.
    denstream_mu: float = Field(default=10.0, gt=0)

    # Scales the survival bar: a cluster must hold weight beta * mu to escape pruning.
    # Higher -> more pruning. beta * mu <= 1 turns pruning off entirely.
    denstream_beta: float = Field(default=0.5, gt=0, le=1)

    # How many hash functions must ALL agree for two rows to share an LSH block.
    # More -> harder to agree -> finer partition, more and smaller blocks.
    lsh_hashes: int = Field(default=6, ge=1)

    # Seeds the LSH projections, MinHash, feature hashing, calibration sampling and MDS
    # landmarks. Keeping it fixed makes the same logs and config give the same result.
    seed: int = 0

    @model_validator(mode="after")
    def _needs_a_stage(self) -> "Config":
        if not (self.lsh or self.birch or self.denstream):
            raise ValueError("enable at least one clustering stage: lsh, birch or denstream")
        return self

    @property
    def stages(self) -> list[str]:
        flags = {"lsh": self.lsh, "birch": self.birch, "denstream": self.denstream}
        return [name for name, enabled in flags.items() if enabled]
