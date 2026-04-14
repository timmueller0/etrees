"""Exhaustive shallow search over E-tree formulas."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from etree.ast import Expr, pretty
from etree.eval import EvaluationError, evaluate
from etree.generate import (
    GenerationStats,
    LeafRegime,
    deduplicate_by_signature,
    generate_trees,
    generate_trees_with_stats,
)


@dataclass(frozen=True)
class SearchResult:
    expr: Expr
    mse: float


@dataclass(frozen=True)
class HybridSearchResult:
    expr: Expr
    mse: float
    a: float
    b: float


@dataclass(frozen=True)
class SearchReport:
    """Search outputs including optional generation diagnostics."""

    results: tuple[SearchResult, ...]
    generation_stats: GenerationStats


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rank_candidates(
    candidates: Sequence[Expr], x_grid: np.ndarray, y_target: np.ndarray
) -> list[SearchResult]:
    """Evaluate candidates and rank by MSE ascending."""
    ranked: list[SearchResult] = []
    for expr in candidates:
        try:
            y_pred = evaluate(expr, x_grid)
        except EvaluationError:
            continue
        ranked.append(SearchResult(expr=expr, mse=mse(y_target, y_pred)))
    ranked.sort(key=lambda r: r.mse)
    return ranked


def shallow_search(
    x_grid: np.ndarray,
    y_target: np.ndarray,
    max_depth: int = 3,
    top_k: int = 5,
    dedupe_signatures: bool = True,
    leaf_regime: LeafRegime = "e_only",
) -> list[SearchResult]:
    """Generate expressions up to depth and return top-k by MSE."""
    candidates = generate_trees(max_depth=max_depth, leaf_regime=leaf_regime)
    if dedupe_signatures:
        candidates = deduplicate_by_signature(candidates, x_grid=x_grid)
    ranked = rank_candidates(candidates, x_grid=x_grid, y_target=y_target)
    return ranked[:top_k]


def shallow_search_with_report(
    x_grid: np.ndarray,
    y_target: np.ndarray,
    max_depth: int = 3,
    top_k: int = 5,
    dedupe_signatures: bool = True,
    leaf_regime: LeafRegime = "e_only",
) -> SearchReport:
    """Run shallow search and include generation telemetry."""
    candidates, stats = generate_trees_with_stats(
        max_depth=max_depth,
        x_grid=x_grid,
        dedupe_signatures=dedupe_signatures,
        leaf_regime=leaf_regime,
    )
    if dedupe_signatures:
        candidates = deduplicate_by_signature(candidates, x_grid=x_grid)
    ranked = rank_candidates(candidates, x_grid=x_grid, y_target=y_target)
    return SearchReport(results=tuple(ranked[:top_k]), generation_stats=stats)


def hybrid_search_with_affine_input(
    x_grid: np.ndarray,
    y_target: np.ndarray,
    max_depth: int,
    top_k: int,
    a_grid: Iterable[float],
    b_grid: Iterable[float],
    dedupe_signatures: bool = True,
    leaf_regime: LeafRegime = "e_only",
) -> list[HybridSearchResult]:
    """Search expressions while sweeping affine input transforms x' = a*x + b."""
    candidates = generate_trees(max_depth=max_depth, leaf_regime=leaf_regime)
    ranked: list[HybridSearchResult] = []

    for a, b in product(a_grid, b_grid):
        x_aff = a * x_grid + b
        current_candidates = candidates
        if dedupe_signatures:
            current_candidates = deduplicate_by_signature(current_candidates, x_grid=x_aff)
        for expr in current_candidates:
            try:
                y_pred = evaluate(expr, x_aff)
            except EvaluationError:
                continue
            ranked.append(HybridSearchResult(expr=expr, mse=mse(y_target, y_pred), a=float(a), b=float(b)))

    ranked.sort(key=lambda r: r.mse)
    return ranked[:top_k]


def results_to_frame(results: Sequence[SearchResult]) -> pd.DataFrame:
    """Convert ranked results to a pandas DataFrame."""
    rows = [{"expr": pretty(r.expr), "mse": r.mse} for r in results]
    return pd.DataFrame(rows)
