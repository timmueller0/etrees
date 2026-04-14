"""etree: exploratory E-logic / E-tree research prototype."""

from etree.ast import AffineLeaf, Constant, ENode, Expr, Variable, depth, pretty, size, subtrees
from etree.benchmark_registry import BENCHMARK_REGISTRY, get_benchmark_registry
from etree.baselines import BaselineResult, affine_least_squares, constant_least_squares, tiny_grammar_search
from etree.benchmarks import Family, RecoveryBudget, RecoveryCase, RecoveryResult, Regime, Tier, run_recovery_case, run_recovery_suite
from etree.canonicalize import canonical_string
from etree.eval import DomainError, EvaluationError, NonFiniteError, evaluate
from etree.generate import DepthStats, GenerationStats, generate_trees_with_stats
from etree.search import (
    HybridSearchResult,
    SearchReport,
    SearchResult,
    hybrid_search_with_affine_input,
    shallow_search,
    shallow_search_with_report,
)

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
    "HybridSearchResult",
    "SearchReport",
    "shallow_search",
    "shallow_search_with_report",
    "hybrid_search_with_affine_input",
    "BaselineResult",
    "constant_least_squares",
    "affine_least_squares",
    "tiny_grammar_search",
    "Tier",
    "Family",
    "Regime",
    "RecoveryCase",
    "RecoveryBudget",
    "RecoveryResult",
    "BENCHMARK_REGISTRY",
    "get_benchmark_registry",
    "run_recovery_case",
    "run_recovery_suite",
]
