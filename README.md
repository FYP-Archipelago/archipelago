# archipelago

**Archipelago** is an explainability platform for distributed (island-model) evolutionary algorithms.

This module is its clustering engine. Standard search trajectory networks (STNs) are built for a single population and don't scale to large runs. It builds them for island-model runs with per-island trajectories, migration edges, and genotypic clustering using LSH, BIRCH, DenStream or any combination of them. It also measures how separated the islands are in genome space, using MMD with random Fourier features. Runs can be clustered after they finish, or live with the network updating as the run progresses.

## Package Installation

Requires Python 3.12+.

```bash
pip install git+https://github.com/FYP-Archipelago/archipelago
```

or, in a `uv` project:

```bash
uv add git+https://github.com/FYP-Archipelago/archipelago
```

## Development

1. Clone the repository:

   ```bash
   git clone https://github.com/FYP-Archipelago/archipelago
   cd archipelago
   ```

2. Install the dependencies:

   ```bash
   uv sync
   ```

3. Install the pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Tests

To run the unit tests:

```bash
uv run pytest
```

## Generation history

Both `cluster_run()` and `ClusterStream.finish()` include `Result.generation_history`,
alongside `migration_routes`. It contains one recorded `generation_end` summary per
`(island_id, generation)`, sorted by that key; repeated delivery keeps the first summary.
Generations and cumulative `evaluations_total` are local to each island.

The summaries retain `best_fitness`, `best_so_far_fitness`, `mean_fitness`,
`std_fitness`, `diversity`, `diversity_metric`, and `diversity_sample_size` from the
runner. Fitness orientation is in `Result.run.maximising`. Missing or non-finite
metrics are `null`; the module does not reconstruct population statistics from
evaluation rows or clustered nodes. Logs without identified generation summaries
return an empty list, and older results without the field still validate.

History is part of the final result, not the live `Update` payload.

## LICENSE

The project is licensed under the [MIT License](LICENSE).
