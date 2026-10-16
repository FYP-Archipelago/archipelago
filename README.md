# archipelago

Clusters the logs of a distributed evolutionary algorithm (dEA) run into a Search
Trajectory Network ready for the Archipelago viewer. Give it a finished run's evaluation
rows and events, and it returns two graphs: a **baseline** with one node per exact genome
on each island, and an optional **clustered** graph where nearby solutions are merged by
LSH, BIRCH and/or DenStream.

## Install

```bash
pip install git+https://github.com/FYP-Archipelago/dEA-clustering@<ref>
```

## Usage

```python
import csv
import json

from archipelago import Config, cluster_run

with open("runs/<run_id>/evaluations.csv", newline="") as handle:
    evaluations = list(csv.DictReader(handle))
with open("runs/<run_id>/run.jsonl") as handle:
    events = [json.loads(line) for line in handle if line.strip()]

result = cluster_run(evaluations, events, Config(lsh=True, birch=True))
payload = result.model_dump()  # JSON-safe: no NaN or Infinity
```

`config` may also be a plain dict (the settings payload) or `None` for defaults.
Bad logs raise `SchemaError` or `UnsupportedEncoding` (both subclass `ClusteringError`).

## Config

`Config()` means "no clustering, baseline only". Stage order is fixed
(LSH -> BIRCH -> DenStream); the flags only choose which ones run.

| field | default | meaning |
|---|---|---|
| `lsh`, `birch`, `denstream` | `False` | run that stage |
| `detail` | `1.0` (0.5-2.0) | divides the auto BIRCH threshold and auto DenStream epsilon; >1 = finer, <1 = coarser |
| `birch_threshold` | `None` (auto) | BIRCH merge radius |
| `denstream_epsilon` | `None` (auto) | DenStream micro-cluster radius |
| `denstream_half_life` | `None` (auto) | seconds for a point's weight to halve |
| `denstream_mu` | `10.0` | DenStream mu |
| `denstream_beta` | `0.5` | DenStream beta, in (0, 1] |
| `lsh_hashes` | `6` | hashes ANDed per LSH key |
| `seed` | `0` | seeds every random choice |

Auto values are measured from the run's own data and reported back in
`result.clustering.resolved`. An explicit value is used as-is and ignores `detail`.
Developer-only constants live in `src/archipelago/config.py`.

## Result

| field | contents |
|---|---|
| `run` | run_id, algorithm, benchmark, topology, islands, maximising |
| `layout` | projection method (`pca`, `mds` or `none`) and the variance it keeps in 2D and 3D |
| `baseline` | `Graph` |
| `clustered` | `Graph`, or `None` when no stage ran |
| `clustering` | stages run, counts, resolved auto values, per-stage diagnostics, elapsed seconds |
| `rows_without_genome`, `warnings` | data-quality signals |

A `Graph` is columnar: `nodes` holds one list per field (node `i` is index `i` in each),
`edges` and `migrations` refer to nodes by index, and `overlap` is the mean number of
islands per occupied genome or cluster. Node fields: `island`, `x`/`y`/`z` (in [-1, 1]),
`visits`, `members`, `mean_fitness`, `best_fitness`, `fitness_percentile` (0 = best),
`progress`, `first_eval`, `last_eval`, `in_degree`, `out_degree`, `islands_sharing`,
`island_best`, `overall_best`, `migration_endpoint`, `genome_hash` and `weight`
(DenStream only).

## Development

```bash
git clone https://github.com/FYP-Archipelago/dEA-clustering
cd dEA-clustering
uv sync
```
