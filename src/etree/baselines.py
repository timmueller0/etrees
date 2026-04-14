"""Baseline models and tiny grammar search for recovery benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    """Single baseline fit output."""

    expr: str
    mse: float


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def constant_least_squares(x_grid: np.ndarray, y_target: np.ndarray) -> BaselineResult:
    """Best constant least-squares baseline y = c."""
    c = float(np.mean(y_target))
    y_pred = np.full_like(x_grid, c, dtype=float)
    return BaselineResult(expr=f"{c:.6g}", mse=_mse(y_target, y_pred))


def affine_least_squares(x_grid: np.ndarray, y_target: np.ndarray) -> BaselineResult:
    """Best affine least-squares baseline y = a*x + b."""
    design = np.column_stack([x_grid, np.ones_like(x_grid)])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_target, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    y_pred = (a * x_grid) + b
    return BaselineResult(expr=f"({a:.6g})*x + ({b:.6g})", mse=_mse(y_target, y_pred))


@dataclass(frozen=True)
class _GrammarExpr:
    text: str
    values: np.ndarray
    depth: int
    size: int


def tiny_grammar_search(
    x_grid: np.ndarray,
    y_target: np.ndarray,
    max_depth: int = 3,
    max_size: int = 9,
    slopes: Iterable[float] = (-2.0, -1.0, 0.5, 1.0, 2.0),
    intercepts: Iterable[float] = (-1.0, 0.0, 1.0),
) -> BaselineResult:
    """Tiny mixed-op grammar search under bounded depth/size.

    Grammar atoms: ``x``, ``1``, ``a*x+b``, ``exp(x)``, ``log(x)``.
    Binary ops: ``+``, ``-``, ``*``.
    """

    atoms: list[_GrammarExpr] = [
        _GrammarExpr("x", np.asarray(x_grid, dtype=float), depth=1, size=1),
        _GrammarExpr("1", np.ones_like(x_grid, dtype=float), depth=1, size=1),
        _GrammarExpr("exp(x)", np.exp(x_grid), depth=1, size=1),
    ]
    if np.all(x_grid > 0.0):
        atoms.append(_GrammarExpr("log(x)", np.log(x_grid), depth=1, size=1))

    for a in slopes:
        for b in intercepts:
            atoms.append(_GrammarExpr(f"({float(a):.6g}*x+{float(b):.6g})", (float(a) * x_grid) + float(b), depth=1, size=1))

    by_depth: dict[int, list[_GrammarExpr]] = {1: atoms}
    best: BaselineResult | None = None
    seen: set[tuple[float, ...]] = set()

    def consider(expr: _GrammarExpr) -> None:
        nonlocal best
        if not np.all(np.isfinite(expr.values)):
            return
        sig = tuple(np.round(expr.values, decimals=10).tolist())
        if sig in seen:
            return
        seen.add(sig)
        mse_val = _mse(y_target, expr.values)
        if best is None or mse_val < best.mse:
            best = BaselineResult(expr=expr.text, mse=mse_val)

    for expr in atoms:
        consider(expr)

    ops = ("+", "-", "*")
    for depth in range(2, max_depth + 1):
        current: list[_GrammarExpr] = []
        prior = [node for d in range(1, depth) for node in by_depth[d]]
        for left in prior:
            for right in prior:
                d = 1 + max(left.depth, right.depth)
                if d != depth:
                    continue
                s = 1 + left.size + right.size
                if s > max_size:
                    continue
                for op in ops:
                    if op == "+":
                        values = left.values + right.values
                    elif op == "-":
                        values = left.values - right.values
                    else:
                        values = left.values * right.values
                    current.append(_GrammarExpr(text=f"({left.text}{op}{right.text})", values=values, depth=d, size=s))

        by_depth[depth] = current
        for expr in current:
            consider(expr)

    if best is None:
        return BaselineResult(expr="<none>", mse=float("inf"))
    return best
