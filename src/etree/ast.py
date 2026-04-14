"""AST definitions for E-tree expressions.

Core operator:
    E(left, right) = exp(left) - log(right)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple, Union


@dataclass(frozen=True)
class Variable:
    """Variable leaf node (e.g., x)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant:
    """Constant leaf node (e.g., 1.0)."""

    value: float

    def __str__(self) -> str:
        if float(self.value).is_integer():
            return str(int(self.value))
        return str(self.value)


@dataclass(frozen=True)
class ENode:
    """Binary E operator node: exp(left) - log(right)."""

    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"E({self.left}, {self.right})"


Expr = Union[Variable, Constant, ENode]


def pretty(expr: Expr) -> str:
    """Return a stable human-readable form of an expression."""
    return str(expr)


def depth(expr: Expr) -> int:
    """Return max node depth (leaf depth = 1)."""
    if isinstance(expr, (Variable, Constant)):
        return 1
    return 1 + max(depth(expr.left), depth(expr.right))


def size(expr: Expr) -> int:
    """Return total number of nodes in the tree."""
    if isinstance(expr, (Variable, Constant)):
        return 1
    return 1 + size(expr.left) + size(expr.right)


def iter_subtrees(expr: Expr) -> Iterator[Expr]:
    """Yield all subtrees in preorder, including the root."""
    yield expr
    if isinstance(expr, ENode):
        yield from iter_subtrees(expr.left)
        yield from iter_subtrees(expr.right)


def subtrees(expr: Expr) -> Tuple[Expr, ...]:
    """Collect all subtrees as an immutable tuple."""
    return tuple(iter_subtrees(expr))


def leaf_x() -> Variable:
    """Convenience constructor for variable x."""
    return Variable("x")


def leaf_one() -> Constant:
    """Convenience constructor for constant 1."""
    return Constant(1.0)
