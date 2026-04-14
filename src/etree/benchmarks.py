"""Benchmark harnesses for E-tree recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etree.ast import Expr, pretty
from etree.eval import evaluate
from etree.search import shallow_search


@dataclass(frozen=True)
class RecoveryCase:
    """Single symbolic regression recovery benchmark case."""

    name: str
    target: Expr
    x_min: float = -0.8
    x_max: float = 0.8
    num_points: int = 60
    max_depth: int = 3


@dataclass(frozen=True)
class RecoveryResult:
    """Output summary for one recovery benchmark."""

    name: str
    best_expr: str
    best_mse: float
    exact_recovered: bool


def run_recovery_case(case: RecoveryCase, top_k: int = 5) -> RecoveryResult:
    """Run one benchmark case and summarize best candidate quality."""
    x_grid = np.linspace(case.x_min, case.x_max, case.num_points)
    y_target = evaluate(case.target, x_grid)
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
            best_expr="<none>",
            best_mse=float("inf"),
            exact_recovered=False,
        )

    best = ranked[0]
    return RecoveryResult(
        name=case.name,
        best_expr=pretty(best.expr),
        best_mse=best.mse,
        exact_recovered=pretty(best.expr) == pretty(case.target),
    )


def run_recovery_suite(cases: list[RecoveryCase], top_k: int = 5) -> list[RecoveryResult]:
    """Run multiple recovery cases."""
    return [run_recovery_case(case, top_k=top_k) for case in cases]
