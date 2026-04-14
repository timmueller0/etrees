"""Canonical serialization utilities for E-trees."""

from __future__ import annotations

from etree.ast import AffineLeaf, Constant, ENode, Expr, Variable


def _format_constant(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    normalized = f"{float(value):.12g}"
    return normalized


def canonical_string(expr: Expr) -> str:
    """Return a stable canonical serialization for structural deduplication."""
    if isinstance(expr, Variable):
        return f"Var({expr.name})"
    if isinstance(expr, Constant):
        return f"Const({_format_constant(expr.value)})"
    if isinstance(expr, AffineLeaf):
        return f"Affine({expr.variable},{_format_constant(expr.a)},{_format_constant(expr.b)})"
    if isinstance(expr, ENode):
        left = canonical_string(expr.left)
        right = canonical_string(expr.right)
        return f"E({left},{right})"
    raise TypeError(f"Unsupported expression type: {type(expr)!r}")
