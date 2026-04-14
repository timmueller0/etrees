"""etree: exploratory E-logic / E-tree research prototype."""

from etree.ast import Constant, ENode, Expr, Variable, depth, pretty, size, subtrees
from etree.benchmarks import RecoveryCase, RecoveryResult, run_recovery_case, run_recovery_suite
from etree.canonicalize import canonical_string
from etree.eval import DomainError, EvaluationError, NonFiniteError, evaluate
from etree.generate import DepthStats, GenerationStats, generate_trees_with_stats
from etree.search import SearchReport, SearchResult, shallow_search, shallow_search_with_report

__all__ = [
    "Expr",
    "Variable",
    "Constant",
    "AffineLeaf",
    "ENode",
    "pretty",
    "canonical_string",
    "depth",
    "size",
    "subtrees",
    "evaluate",
    "EvaluationError",
    "DomainError",
    "NonFiniteError",
    "DepthStats",
    "GenerationStats",
    "generate_trees_with_stats",
    "SearchResult",
    "SearchReport",
    "shallow_search",
    "shallow_search_with_report",
    "RecoveryCase",
    "RecoveryResult",
    "run_recovery_case",
    "run_recovery_suite",
]
