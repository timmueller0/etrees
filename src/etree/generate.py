"""Tree generation for E-expressions."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from etree.ast import Constant, ENode, Expr, Variable, depth
from etree.eval import EvaluationError, evaluate


def default_leaves() -> list[Expr]:
    """Leaf set used in early experiments: {x, 1}."""
    return [Variable("x"), Constant(1.0)]


def generate_trees(max_depth: int, leaves: Sequence[Expr] | None = None) -> list[Expr]:
    """Generate all E-trees with depth <= max_depth."""
    if max_depth < 1:
        return []

    leaves = list(leaves) if leaves is not None else default_leaves()
    by_depth: dict[int, list[Expr]] = {1: list(leaves)}

    for d in range(2, max_depth + 1):
        prev = [expr for k in range(1, d) for expr in by_depth[k]]
        current: list[Expr] = []
        for left in prev:
            for right in prev:
                node = ENode(left, right)
                if depth(node) == d:
                    current.append(node)
        by_depth[d] = current

    all_exprs = [expr for d in range(1, max_depth + 1) for expr in by_depth[d]]
    return all_exprs


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
