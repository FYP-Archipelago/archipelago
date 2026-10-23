# archipelago

**Archipelago** is an explainability platform for distributed (island-model) evolutionary algorithms.

This module is its clustering engine. Standard search trajectory networks (STNs) are built for a single population and don't scale to large runs. It builds them for island-model runs with per-island trajectories, migration edges, and genotypic clustering using LSH, BIRCH, DenStream or any combination of them. It works on a finished run or streams the network live.

## Installation

Requires Python 3.12+.

```bash
pip install git+https://github.com/FYP-Archipelago/archipelago@<ref>
```

or, in a `uv` project:

```bash
uv add git+https://github.com/FYP-Archipelago/archipelago --rev <ref>
```

## Development

```bash
git clone https://github.com/FYP-Archipelago/archipelago
cd archipelago
uv sync
```

## LICENSE

The project is licensed under the [MIT License](LICENSE).
