"""Benchmark harnesses for E-tree recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from etree.ast import Expr, pretty
from etree.eval import evaluate
from etree.search import shallow_search

Tier = Literal["native_e", "compressible_non_native", "negative_control"]
Family = Literal["growth/saturation", "geometry-like", "identity/compression", "adversarial"]


@dataclass(frozen=True)
class RecoveryCase:
    """Single symbolic regression recovery benchmark case."""

    name: str
    target: Expr | None = None
    tier: Tier = "native_e"
    family: Family = "identity/compression"
    target_factory: Callable[[np.ndarray], np.ndarray] | None = None
    x_min: float = -0.8
    x_max: float = 0.8
    num_points: int = 60
    max_depth: int = 3


@dataclass(frozen=True)
class RecoveryResult:
    """Output summary for one recovery benchmark."""

    name: str
    tier: Tier
    family: Family
    best_expr: str
    best_mse: float
    exact_recovered: bool


def _compute_target(case: RecoveryCase, x_grid: np.ndarray) -> np.ndarray:
    if case.target is not None:
        return evaluate(case.target, x_grid)
    if case.target_factory is not None:
        return np.asarray(case.target_factory(x_grid), dtype=float)
    raise ValueError(f"RecoveryCase '{case.name}' must provide target or target_factory.")


def run_recovery_case(case: RecoveryCase, top_k: int = 5) -> RecoveryResult:
    """Run one benchmark case and summarize best candidate quality."""
    x_grid = np.linspace(case.x_min, case.x_max, case.num_points)
    y_target = _compute_target(case, x_grid)
    ranked = shallow_search(
        x_grid=x_grid,
        y_target=y_target,
        max_depth=case.max_depth,
        top_k=top_k,
        dedupe_signatures=True,
    )

    if not ranked:
        return RecoveryResult(
            name=case.name,
            tier=case.tier,
            family=case.family,
            best_expr="<none>",
            best_mse=float("inf"),
            exact_recovered=False,
        )

    best = ranked[0]
    exact_recovered = False
    if case.target is not None:
        exact_recovered = pretty(best.expr) == pretty(case.target)

    return RecoveryResult(
        name=case.name,
        tier=case.tier,
        family=case.family,
        best_expr=pretty(best.expr),
        best_mse=best.mse,
        exact_recovered=exact_recovered,
    )


def run_recovery_suite(cases: list[RecoveryCase], top_k: int = 5) -> list[RecoveryResult]:
    """Run multiple recovery cases."""
    return [run_recovery_case(case, top_k=top_k) for case in cases]
