"""Run recovery benchmarks from the shared registry."""

from __future__ import annotations

from collections import defaultdict

from etree.benchmark_registry import get_benchmark_registry
from etree.benchmarks import run_recovery_suite


if __name__ == "__main__":
    cases = get_benchmark_registry()
    results = run_recovery_suite(cases)

    for row in results:
        print(
            f"{row.name:28s} tier={row.tier:24s} family={row.family:22s} "
            f"exact={str(row.exact_recovered):5s} best_mse={row.best_mse:.3e} expr={row.best_expr}"
        )

    by_tier: dict[str, list[float]] = defaultdict(list)
    for row in results:
        by_tier[row.tier].append(row.best_mse)

    print("\nSummary by tier (mean best MSE)")
    for tier, mses in by_tier.items():
        print(f"  {tier:24s} {sum(mses) / len(mses):.3e}")
