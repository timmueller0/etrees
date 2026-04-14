"""Benchmark harnesses for E-tree recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from etree.ast import Expr, pretty
from etree.baselines import affine_least_squares, constant_least_squares, tiny_grammar_search
from etree.eval import evaluate
from etree.search import shallow_search

Tier = Literal["native_e", "compressible_non_native", "negative_control"]
Family = Literal["growth/saturation", "geometry-like", "identity/compression", "adversarial"]
Regime = Literal["e_only", "e_plus_affine_leaf", "baseline_constant", "baseline_affine", "baseline_grammar"]


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
class RecoveryBudget:
    """Shared budget used across all regimes for comparability."""

    max_depth: int
    top_k: int


@dataclass(frozen=True)
class RecoveryResult:
    """Output summary for one recovery benchmark and regime."""

    name: str
    tier: Tier
    family: Family
    regime: Regime
    best_expr: str
    best_mse: float
    exact_recovered: bool
    budget: RecoveryBudget


def _compute_target(case: RecoveryCase, x_grid: np.ndarray) -> np.ndarray:
    if case.target is not None:
        return evaluate(case.target, x_grid)
    if case.target_factory is not None:
        return np.asarray(case.target_factory(x_grid), dtype=float)
    raise ValueError(f"RecoveryCase '{case.name}' must provide target or target_factory.")


def _exact_match(case: RecoveryCase, expr_text: str) -> bool:
    if case.target is None:
        return False
    return expr_text == pretty(case.target)


def run_recovery_case(case: RecoveryCase, top_k: int = 5) -> list[RecoveryResult]:
    """Run one benchmark case across all regimes under a shared budget."""
    x_grid = np.linspace(case.x_min, case.x_max, case.num_points)
    y_target = _compute_target(case, x_grid)
    budget = RecoveryBudget(max_depth=case.max_depth, top_k=top_k)

    results: list[RecoveryResult] = []

    for regime, leaf_regime in (("e_only", "e_only"), ("e_plus_affine_leaf", "e_plus_affine")):
        ranked = shallow_search(
            x_grid=x_grid,
            y_target=y_target,
            max_depth=case.max_depth,
            top_k=top_k,
            dedupe_signatures=True,
            leaf_regime=leaf_regime,
        )

        best_expr = "<none>"
        best_mse = float("inf")
        if ranked:
            best_expr = pretty(ranked[0].expr)
            best_mse = ranked[0].mse

        results.append(
            RecoveryResult(
                name=case.name,
                tier=case.tier,
                family=case.family,
                regime=regime,
                best_expr=best_expr,
                best_mse=best_mse,
                exact_recovered=_exact_match(case, best_expr),
                budget=budget,
            )
        )

    constant_fit = constant_least_squares(x_grid=x_grid, y_target=y_target)
    results.append(
        RecoveryResult(
            name=case.name,
            tier=case.tier,
            family=case.family,
            regime="baseline_constant",
            best_expr=constant_fit.expr,
            best_mse=constant_fit.mse,
            exact_recovered=_exact_match(case, constant_fit.expr),
            budget=budget,
        )
    )

    affine_fit = affine_least_squares(x_grid=x_grid, y_target=y_target)
    results.append(
        RecoveryResult(
            name=case.name,
            tier=case.tier,
            family=case.family,
            regime="baseline_affine",
            best_expr=affine_fit.expr,
            best_mse=affine_fit.mse,
            exact_recovered=_exact_match(case, affine_fit.expr),
            budget=budget,
        )
    )

    grammar_fit = tiny_grammar_search(x_grid=x_grid, y_target=y_target, max_depth=case.max_depth)
    results.append(
        RecoveryResult(
            name=case.name,
            tier=case.tier,
            family=case.family,
            regime="baseline_grammar",
            best_expr=grammar_fit.expr,
            best_mse=grammar_fit.mse,
            exact_recovered=_exact_match(case, grammar_fit.expr),
            budget=budget,
        )
    )

    return results


def run_recovery_suite(cases: list[RecoveryCase], top_k: int = 5) -> list[RecoveryResult]:
    """Run multiple recovery cases across all regimes."""
    out: list[RecoveryResult] = []
    for case in cases:
        out.extend(run_recovery_case(case, top_k=top_k))
    return out
