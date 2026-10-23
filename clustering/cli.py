from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

from . import __version__
from .config import Config
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluster", description="Build a STN from dEA run logs")
    parser.add_argument("run", type=Path, help="A run directory containing run.jsonl")

    # CLI flags for stages to be included in the pipeline
    parser.add_argument("--lsh", action="store_true", help="Enable LSH blocking")
    parser.add_argument("--birch", action="store_true", help="Enable BIRCH")
    parser.add_argument("--denstream", action="store_true", help="Enable DenStream")

    parser.add_argument("--plot", action="store_true", help="Plot the base STN and clustered STNs (saved to stn.png)")
    parser.add_argument("--fplot", action="store_true", help="Plot the STN with nodes plotted by fitness (saved to fitness-stn.png)")
    parser.add_argument("--mmd", action="store_true", help="Measure how separate the islands are: MMD via random Fourier features (saved to separation.json)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output directory (default out/<run id>)")

    parser.add_argument("-c", "--config", type=Path, default=None, help="Custom YAML config path; see config/default.yaml")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Creates and set config based on passed flags
    args = build_parser().parse_args(argv)

    config = Config.load(args.config)
    config.lsh.enabled = args.lsh
    config.birch.enabled = args.birch
    config.denstream.enabled = args.denstream

    out = args.out or Path("out") / args.run.resolve().name

    # Run the actual clustering pipeline
    need_baseline = (args.plot or args.fplot) and config.any_stage_enabled
    result = run(args.run, out, config, baseline=need_baseline)

    # Plot and render STN if --plot is set
    image = None
    if args.plot:
        from .plot import render
        from .reader import RunDirectory

        stages = " -> ".join(k for k, v in config.stage_flags().items() if v)
        image = render(
            result.stn,
            RunDirectory(args.run).read_metadata(),
            result.out_dir / "stn.png",
            baseline=result.baseline_stn,
            layout=result.layout,
            baseline_layout=result.baseline_layout,
            stages=stages,
        )

    # Plot and render fitness STN if --fplot is set
    fimage = None
    if args.fplot:
        from .plot import render_fitness
        from .reader import RunDirectory

        stages = " -> ".join(k for k, v in config.stage_flags().items() if v)
        fimage = render_fitness(
            result.stn,
            RunDirectory(args.run).read_metadata(),
            result.out_dir / "fitness-stn.png",
            baseline=result.baseline_stn,
            layout=result.layout,
            baseline_layout=result.baseline_layout,
            stages=stages,
        )

    # Measure island separation if --mmd is set
    separation = None
    if args.mmd:
        from .features.registry import build_features
        from .reader import RunDirectory
        from .separation import island_separation, write_separation

        run_dir = RunDirectory(args.run)
        metadata = run_dir.read_metadata()
        features, _ = build_features(run_dir, metadata, config)
        separation = island_separation(features, config.seed)
        write_separation(result.out_dir, separation, metadata.run_id)

    # Extract run statistics from manifest
    manifest = json.loads((result.out_dir / "manifest.json").read_text())
    counts = manifest["counts"]
    baseline = manifest["baseline"]
    stages = [k for k, v in manifest["stages"].items() if v] or ["level-0"]

    # Print out statistics
    print(f"run       {manifest['source_run']['run_id']}")
    print(f"stages    {' -> '.join(stages)} (layout: {manifest['layout'].get('method')})")
    print(
        f"nodes     {counts['nodes']:,} from {counts['evaluations']:,} evaluations"
        f"  ({baseline['nodes_per_evaluation']:.4f} per evaluation)"
    )
    if baseline["level0_nodes"]:
        print(
            f"baseline  level-0 would give {baseline['level0_nodes']:,} nodes"
            f"  (reduction {baseline['reduction_vs_level0'] * 100:.1f}%)"
        )
    print(
        f"edges     {counts['edges']:,} trajectory |"
        f" {counts['node_migration_edges']} migration |"
        f" {counts['self_loops']:,} self-loops\n"
    )
    
    if separation is not None and separation["pairs"]:
        pairs = sorted(p["mmd2"] for p in separation["pairs"])
        print(f"islands   median pair MMD^2 {pairs[len(pairs) // 2]:.2g} (random Fourier features)\n")

    print(f"Pipeline ran successfully in {result.elapsed_seconds:.2f}s")
    saved = []
    if image:
        saved.append("stn.png")
    if fimage:
        saved.append("fitness-stn.png")
    if separation is not None:
        saved.append("separation.json")
    suffix = f" ({', '.join(saved)} saved)" if saved else ""
    print(f"Artifacts saved to {result.out_dir}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
