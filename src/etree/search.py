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
    deduplicate_by_signature,
    generate_trees,
    generate_trees_with_stats,
)


@dataclass(frozen=True)
class SearchResult:
    expr: Expr
    mse: float


@dataclass(frozen=True)
class SearchReport:
    """Search outputs including optional generation diagnostics."""

    results: tuple[SearchResult, ...]
    generation_stats: GenerationStats


@dataclass(frozen=True)
class HybridSearchResult:
    """Search result augmented with affine input transform parameters."""

    expr: Expr
    mse: float
    a: float
    b: float


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
) -> list[SearchResult]:
    """Generate expressions up to depth and return top-k by MSE."""
    candidates = generate_trees(max_depth=max_depth)
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
) -> SearchReport:
    """Run shallow search and include generation telemetry."""
    candidates, stats = generate_trees_with_stats(
        max_depth=max_depth,
        x_grid=x_grid,
        dedupe_signatures=dedupe_signatures,
    )
    if dedupe_signatures:
        candidates = deduplicate_by_signature(candidates, x_grid=x_grid)
    ranked = rank_candidates(candidates, x_grid=x_grid, y_target=y_target)
    return SearchReport(results=tuple(ranked[:top_k]), generation_stats=stats)


def hybrid_search_with_affine_input(
    x_grid: np.ndarray,
    y_target: np.ndarray,
    max_depth: int = 3,
    top_k: int = 5,
    a_grid: Iterable[float] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
    b_grid: Iterable[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    dedupe_signatures: bool = True,
) -> list[HybridSearchResult]:
    """Hybrid search: brute-force structure + coarse affine input transform sweep.

    For each structural candidate, evaluate over transformed inputs x' = a*x + b and
    keep the best affine parameters.
    """
    candidates = generate_trees(max_depth=max_depth)
    if dedupe_signatures:
        candidates = deduplicate_by_signature(candidates, x_grid=x_grid)

    ranked: list[HybridSearchResult] = []
    for expr in candidates:
        best_mse = float("inf")
        best_a = 1.0
        best_b = 0.0
        for a, b in product(a_grid, b_grid):
            x_transformed = (float(a) * x_grid) + float(b)
            try:
                y_pred = evaluate(expr, x_transformed)
            except EvaluationError:
                continue
            score = mse(y_target, y_pred)
            if score < best_mse:
                best_mse = score
                best_a = float(a)
                best_b = float(b)

        if np.isfinite(best_mse):
            ranked.append(HybridSearchResult(expr=expr, mse=best_mse, a=best_a, b=best_b))

    ranked.sort(key=lambda r: r.mse)
    return ranked[:top_k]


def results_to_frame(results: Sequence[SearchResult]) -> pd.DataFrame:
    """Convert ranked results to a pandas DataFrame."""
    rows = [{"expr": pretty(r.expr), "mse": r.mse} for r in results]
    return pd.DataFrame(rows)
