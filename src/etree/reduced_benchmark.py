"""Compact reduced benchmark instrument for reproducible comparison sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from etree.ast import depth, pretty, size
from etree.baselines import affine_least_squares
from etree.benchmark_registry import get_benchmark_registry
from etree.benchmarks import RecoveryCase
from etree.eval import evaluate
from etree.search import shallow_search_with_report

Regime = Literal["e_only", "hybrid_affine", "baseline"]


@dataclass(frozen=True)
class IntervalBudget:
    """Fixed run budget: x-interval + depth cap + sample count."""

    interval: str
    x_min: float
    x_max: float
    max_depth: int
    num_points: int = 60


FIXED_INTERVAL_BUDGETS: tuple[IntervalBudget, ...] = (
    IntervalBudget(interval="tight", x_min=-0.6, x_max=0.6, max_depth=1),
    IntervalBudget(interval="default", x_min=-0.8, x_max=0.8, max_depth=2),
    IntervalBudget(interval="wide", x_min=-1.2, x_max=1.2, max_depth=3),
)




def _resolve_grid(case: RecoveryCase, budget: IntervalBudget) -> np.ndarray:
    lo = max(budget.x_min, case.x_min)
    hi = min(budget.x_max, case.x_max)
    if lo >= hi:
        lo, hi = case.x_min, case.x_max
    return np.linspace(lo, hi, budget.num_points)

def _target_values(case: RecoveryCase, x_grid: np.ndarray) -> np.ndarray:
    if case.target is not None:
        return evaluate(case.target, x_grid)
    if case.target_factory is not None:
        return np.asarray(case.target_factory(x_grid), dtype=float)
    raise ValueError(f"Case '{case.name}' must set target or target_factory")


def _recovery_label(case: RecoveryCase, mse: float) -> str:
    """Small explicit heuristic that can be tweaked as needed.

    - native_e targets with an AST are treated as exact only with very tiny MSE
    - non-native or factory targets can still be marked exact with tiny MSE
    - near means close but not exact
    """

    if not np.isfinite(mse):
        return "miss"

    exact_tol = 1e-12 if case.target is not None else 1e-10
    near_tol = 1e-5 if case.tier == "negative_control" else 1e-6

    if mse <= exact_tol:
        return "exact"
    if mse <= near_tol:
        return "near"
    return "miss"


def _row(
    *,
    case: RecoveryCase,
    regime: Regime,
    interval: str,
    max_depth: int,
    generated_count: int,
    valid_count: int,
    deduped_count: int,
    best_mse: float,
    runtime_sec: float,
    winner_expr: str,
    winner_depth: int,
    winner_size: int,
    recovery_label: str,
) -> dict[str, object]:
    return {
        "target": case.name,
        "family": case.family,
        "regime": regime,
        "interval": interval,
        "max_depth": max_depth,
        "generated_count": generated_count,
        "valid_count": valid_count,
        "deduped_count": deduped_count,
        "best_mse": best_mse,
        "runtime_sec": runtime_sec,
        "winner_expr": winner_expr,
        "winner_depth": winner_depth,
        "winner_size": winner_size,
        "recovery_label": recovery_label,
    }


def run_reduced_suite(
    cases: list[RecoveryCase] | None = None,
    budgets: tuple[IntervalBudget, ...] = FIXED_INTERVAL_BUDGETS,
) -> pd.DataFrame:
    """Run all registry recovery cases over fixed budgets and 3 regimes."""
    cases = get_benchmark_registry() if cases is None else cases
    rows: list[dict[str, object]] = []

    for case in cases:
        for budget in budgets:
            x_grid = _resolve_grid(case, budget)
            y_target = _target_values(case, x_grid)

            # E-only
            start = perf_counter()
            report = shallow_search_with_report(
                x_grid=x_grid,
                y_target=y_target,
                max_depth=budget.max_depth,
                top_k=1,
                dedupe_signatures=True,
                leaf_regime="e_only",
            )
            runtime = perf_counter() - start
            best = report.results[0] if report.results else None
            rows.append(
                _row(
                    case=case,
                    regime="e_only",
                    interval=budget.interval,
                    max_depth=budget.max_depth,
                    generated_count=sum(s.generated for s in report.generation_stats.per_depth),
                    valid_count=sum(s.valid for s in report.generation_stats.per_depth),
                    deduped_count=sum(s.deduplicated for s in report.generation_stats.per_depth),
                    best_mse=best.mse if best is not None else float("inf"),
                    runtime_sec=runtime,
                    winner_expr=pretty(best.expr) if best is not None else "<none>",
                    winner_depth=depth(best.expr) if best is not None else 0,
                    winner_size=size(best.expr) if best is not None else 0,
                    recovery_label=_recovery_label(case, best.mse if best is not None else float("inf")),
                )
            )

            # Hybrid affine (same telemetry path as shallow search)
            start = perf_counter()
            report_h = shallow_search_with_report(
                x_grid=x_grid,
                y_target=y_target,
                max_depth=budget.max_depth,
                top_k=1,
                dedupe_signatures=True,
                leaf_regime="e_plus_affine",
            )
            runtime_h = perf_counter() - start
            best_h = report_h.results[0] if report_h.results else None
            rows.append(
                _row(
                    case=case,
                    regime="hybrid_affine",
                    interval=budget.interval,
                    max_depth=budget.max_depth,
                    generated_count=sum(s.generated for s in report_h.generation_stats.per_depth),
                    valid_count=sum(s.valid for s in report_h.generation_stats.per_depth),
                    deduped_count=sum(s.deduplicated for s in report_h.generation_stats.per_depth),
                    best_mse=best_h.mse if best_h is not None else float("inf"),
                    runtime_sec=runtime_h,
                    winner_expr=pretty(best_h.expr) if best_h is not None else "<none>",
                    winner_depth=depth(best_h.expr) if best_h is not None else 0,
                    winner_size=size(best_h.expr) if best_h is not None else 0,
                    recovery_label=_recovery_label(case, best_h.mse if best_h is not None else float("inf")),
                )
            )

            # Baseline affine least-squares
            start = perf_counter()
            fit = affine_least_squares(x_grid=x_grid, y_target=y_target)
            runtime_b = perf_counter() - start
            rows.append(
                _row(
                    case=case,
                    regime="baseline",
                    interval=budget.interval,
                    max_depth=budget.max_depth,
                    generated_count=0,
                    valid_count=0,
                    deduped_count=0,
                    best_mse=fit.mse,
                    runtime_sec=runtime_b,
                    winner_expr=fit.expr,
                    winner_depth=1,
                    winner_size=1,
                    recovery_label=_recovery_label(case, fit.mse),
                )
            )

    return pd.DataFrame(rows)
