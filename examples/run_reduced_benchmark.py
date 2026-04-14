"""Run the reproducible reduced benchmark suite and export rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from etree.reduced_benchmark import run_reduced_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/reduced_benchmark.csv"))
    args = parser.parse_args()

    frame = run_reduced_suite()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    regimes = sorted(frame["regime"].unique().tolist())
    families = sorted(frame["family"].unique().tolist())

    print(f"output_path: {args.out}")
    print(f"rows: {len(frame)}")
    print(f"regimes: {', '.join(regimes)}")
    print(f"families: {', '.join(families)}")


if __name__ == "__main__":
    main()
