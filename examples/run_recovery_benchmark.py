"""Run recovery benchmarks from the shared registry."""

from __future__ import annotations

from collections import defaultdict

from etree.benchmark_registry import get_benchmark_registry
from etree.benchmarks import run_recovery_suite


def main() -> None:
    cases = get_benchmark_registry()
    results = run_recovery_suite(cases)

    for row in results:
        print(
            f"{row.name:28s} regime={row.regime:21s} tier={row.tier:24s} family={row.family:22s} "
            f"exact={str(row.exact_recovered):5s} best_mse={row.best_mse:.3e} expr={row.best_expr} "
            f"budget(depth={row.budget.max_depth},top_k={row.budget.top_k})"
        )

    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in results:
        by_group[(row.tier, row.regime)].append(row.best_mse)

    print("\nSummary by tier/regime (mean best MSE)")
    for (tier, regime), mses in sorted(by_group.items()):
        print(f"  {tier:24s} {regime:21s} {sum(mses) / len(mses):.3e}")


if __name__ == "__main__":
    main()
