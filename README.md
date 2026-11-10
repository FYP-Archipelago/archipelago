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

## LICENSE

The project is licensed under the [MIT License](LICENSE).
