"""Run recovery benchmarks from the shared registry."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from etree.benchmark_registry import get_benchmark_registry
from etree.benchmarks import run_recovery_suite, to_csv, to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=Path("outputs/recovery_runs.csv"))
    parser.add_argument("--parquet-out", type=Path, default=None)
    args = parser.parse_args()

    cases = get_benchmark_registry()
    results = run_recovery_suite(cases)

    csv_path = to_csv(results, args.csv_out, append=True)
    parquet_path = to_parquet(results, args.parquet_out) if args.parquet_out is not None else None
    run_stamp = datetime.now(timezone.utc).isoformat()
    print(f"wrote {len(results)} rows at {run_stamp} to {csv_path}")
    if parquet_path is not None:
        print(f"wrote parquet snapshot to {parquet_path}")

    for row in results:
        print(
            f"{row.target_name:28s} regime={row.regime:21s} tier={row.tier:24s} family={row.target_family:22s} "
            f"match={row.match_category:9s} top1_mse={row.top1_mse:.3e} expr={row.top1_expr}"
        )

    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in results:
        by_group[(row.tier, row.regime)].append(row.top1_mse)

    print("\nSummary by tier/regime (mean best MSE)")
    for (tier, regime), mses in sorted(by_group.items()):
        print(f"  {tier:24s} {regime:21s} {sum(mses) / len(mses):.3e}")


if __name__ == "__main__":
    main()
