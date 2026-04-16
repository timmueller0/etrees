"""Run the reproducible reduced benchmark suite and export rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from etree.benchmark_registry import get_benchmark_registry
from etree.reduced_benchmark import build_growth_sweep_budgets, run_reduced_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/reduced_benchmark.csv"))
    parser.add_argument(
        "--skip-growth-focus",
        action="store_true",
        help="Skip the focused growth/saturation 3x3 (interval x depth) sweep.",
    )
    args = parser.parse_args()

    base = run_reduced_suite().assign(sweep="full_registry")
    frames = [base]

    if not args.skip_growth_focus:
        growth_cases = [c for c in get_benchmark_registry() if c.family == "growth/saturation"]
        growth_focus = run_reduced_suite(cases=growth_cases, budgets=build_growth_sweep_budgets()).assign(
            sweep="growth_focus_3x3"
        )
        frames.append(growth_focus)

    frame = base if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    regimes = sorted(frame["regime"].unique().tolist())
    families = sorted(frame["family"].unique().tolist())

    print(f"output_path: {args.out}")
    print(f"rows: {len(frame)}")
    print(f"regimes: {', '.join(regimes)}")
    print(f"families: {', '.join(families)}")
    print(f"sweeps: {', '.join(sorted(frame['sweep'].unique().tolist()))}")


if __name__ == "__main__":
    main()
