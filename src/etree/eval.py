"""Evaluation utilities for E-tree expressions."""

from __future__ import annotations

from typing import Union

import numpy as np

from etree.ast import AffineLeaf, Constant, ENode, Expr, Variable


class EvaluationError(Exception):
    """Base class for evaluation issues."""


class DomainError(EvaluationError):
    """Raised when the expression is outside its mathematical domain."""


class NonFiniteError(EvaluationError):
    """Raised when expression evaluation produces non-finite values."""


ArrayLike = Union[float, np.ndarray]


def _ensure_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise NonFiniteError(f"Non-finite values encountered in {name}.")


def evaluate(expr: Expr, x: ArrayLike) -> np.ndarray:
    """Evaluate an expression for scalar/array x, returning a numpy array."""
    x_arr = np.asarray(x, dtype=float)

    def _eval(node: Expr) -> np.ndarray:
        if isinstance(node, Variable):
            if node.name != "x":
                raise EvaluationError(f"Unsupported variable '{node.name}'.")
            return x_arr

        if isinstance(node, Constant):
            return np.full_like(x_arr, fill_value=float(node.value), dtype=float)

        if isinstance(node, AffineLeaf):
            if node.variable != "x":
                raise EvaluationError(f"Unsupported variable '{node.variable}' in affine leaf.")
            return (float(node.a) * x_arr) + float(node.b)

        if isinstance(node, ENode):
            left = _eval(node.left)
            right = _eval(node.right)
            if np.any(right <= 0):
                raise DomainError("log(right) requires right > 0.")
            out = np.exp(left) - np.log(right)
            _ensure_finite("E node output", out)
            return out

        raise EvaluationError(f"Unknown expression node: {type(node)!r}")

    result = _eval(expr)
    _ensure_finite("final output", result)
    return result
