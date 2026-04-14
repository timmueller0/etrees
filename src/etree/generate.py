"""Tree generation for E-expressions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from etree.ast import Constant, ENode, Expr, Variable, depth
from etree.eval import EvaluationError, evaluate


@dataclass(frozen=True)
class DepthStats:
    """Per-depth generation and pruning statistics."""

    depth: int
    generated: int
    valid: int
    deduplicated: int
    elapsed_seconds: float


@dataclass(frozen=True)
class GenerationStats:
    """Summary statistics across all generated depths."""

    per_depth: tuple[DepthStats, ...]



def default_leaves() -> list[Expr]:
    """Leaf set used in early experiments: {x, 1}."""
    return [Variable("x"), Constant(1.0)]


def generate_trees(max_depth: int, leaves: Sequence[Expr] | None = None) -> list[Expr]:
    """Generate all E-trees with depth <= max_depth."""
    exprs, _ = generate_trees_with_stats(max_depth=max_depth, leaves=leaves)
    return exprs


def generate_trees_with_stats(
    max_depth: int,
    leaves: Sequence[Expr] | None = None,
    x_grid: np.ndarray | None = None,
    dedupe_signatures: bool = False,
    decimals: int = 8,
) -> tuple[list[Expr], GenerationStats]:
    """Generate E-trees and return per-depth growth statistics.

    If ``x_grid`` is provided, this function also computes per-depth validity
    under evaluation and optional signature deduplication counts.
    """
    if max_depth < 1:
        return [], GenerationStats(per_depth=tuple())

    leaves = list(leaves) if leaves is not None else default_leaves()
    by_depth: dict[int, list[Expr]] = {1: list(leaves)}
    depth_rows: list[DepthStats] = []

    valid, deduped = _quality_counts(
        by_depth[1],
        x_grid=x_grid,
        dedupe_signatures=dedupe_signatures,
        decimals=decimals,
    )
    depth_rows.append(
        DepthStats(depth=1, generated=len(by_depth[1]), valid=valid, deduplicated=deduped, elapsed_seconds=0.0)
    )

    for d in range(2, max_depth + 1):
        start = perf_counter()
        prev = [expr for k in range(1, d) for expr in by_depth[k]]
        current: list[Expr] = []
        for left in prev:
            for right in prev:
                node = ENode(left, right)
                if depth(node) == d:
                    current.append(node)
        by_depth[d] = current

        valid, deduped = _quality_counts(
            current,
            x_grid=x_grid,
            dedupe_signatures=dedupe_signatures,
            decimals=decimals,
        )
        depth_rows.append(
            DepthStats(
                depth=d,
                generated=len(current),
                valid=valid,
                deduplicated=deduped,
                elapsed_seconds=perf_counter() - start,
            )
        )

    all_exprs = [expr for d in range(1, max_depth + 1) for expr in by_depth[d]]
    return all_exprs, GenerationStats(per_depth=tuple(depth_rows))


def _quality_counts(
    exprs: Sequence[Expr],
    x_grid: np.ndarray | None,
    dedupe_signatures: bool,
    decimals: int,
) -> tuple[int, int]:
    if x_grid is None:
        count = len(exprs)
        return count, count

    valid_exprs: list[Expr] = []
    for expr in exprs:
        try:
            evaluate(expr, x_grid)
        except EvaluationError:
            continue
        valid_exprs.append(expr)

    dedup_count = len(valid_exprs)
    if dedupe_signatures:
        dedup_count = len(deduplicate_by_signature(valid_exprs, x_grid=x_grid, decimals=decimals))
    return len(valid_exprs), dedup_count


def deduplicate_by_signature(
    exprs: Sequence[Expr], x_grid: np.ndarray, decimals: int = 8
) -> list[Expr]:
    """Deduplicate expressions by rounded numeric output signatures."""
    kept: list[Expr] = []
    seen: set[tuple[float, ...]] = set()

    for expr in exprs:
        try:
            y = evaluate(expr, x_grid)
        except EvaluationError:
            continue
        signature = tuple(np.round(y.astype(float), decimals=decimals).tolist())
        if signature not in seen:
            seen.add(signature)
            kept.append(expr)

    return kept
