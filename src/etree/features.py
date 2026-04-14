"""Subtree-based feature extraction and similarity for E-trees."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from etree.ast import Expr, pretty, subtrees


def subtree_multiset(expr: Expr) -> Counter[str]:
    """Represent an expression as a multiset of subtree strings."""
    return Counter(pretty(st) for st in subtrees(expr))


def multiset_jaccard(a: Counter[str], b: Counter[str]) -> float:
    """Jaccard similarity generalized for multisets."""
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    if union == 0:
        return 1.0
    return inter / union


def subtree_similarity(expr_a: Expr, expr_b: Expr) -> float:
    """Convenience wrapper around subtree multiset Jaccard."""
    return multiset_jaccard(subtree_multiset(expr_a), subtree_multiset(expr_b))


def token_set_similarity(expr_a: Expr, expr_b: Expr) -> float:
    """Naive baseline: Jaccard overlap of character-level surface tokens."""
    tok_a = set(pretty(expr_a).replace(" ", ""))
    tok_b = set(pretty(expr_b).replace(" ", ""))
    inter = len(tok_a & tok_b)
    union = len(tok_a | tok_b)
    if union == 0:
        return 1.0
    return inter / union
