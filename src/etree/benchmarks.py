"""Benchmark harnesses for E-tree recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from etree.ast import Expr, depth, pretty, size
from etree.baselines import affine_least_squares, constant_least_squares, tiny_grammar_search
from etree.canonicalize import canonical_string
from etree.eval import evaluate
from etree.search import SearchResult, shallow_search_with_report

Tier = Literal["native_e", "compressible_non_native", "negative_control"]
Family = Literal["growth/saturation", "geometry-like", "identity/compression", "adversarial"]
MatchCategory = Literal["exact", "algebraic", "numeric", "none"]
Regime = Literal["e_only", "e_plus_affine_leaf", "baseline_constant", "baseline_affine", "baseline_grammar"]


@dataclass(frozen=True)
class RecoveryCase:
    """Single symbolic regression recovery benchmark case."""

    name: str
    target: Expr | None = None
    tier: Tier = "native_e"
    family: Family = "identity/compression"
    target_factory: Callable[[np.ndarray], np.ndarray] | None = None
    x_min: float = -0.8
    x_max: float = 0.8
    num_points: int = 60
    max_depth: int = 3
    random_seed: int = 0


@dataclass(frozen=True)
class RecoveryBudget:
    """Shared budget used across all regimes for comparability."""

    max_depth: int
    top_k: int


@dataclass(frozen=True)
class RecoveryResult:
    """Experiment row schema for one recovery benchmark run."""

    # target metadata
    target_name: str
    target_family: Family
    tier: Tier
    regime: str
    domain_config: str
    domain_config_hash: str

    # search settings
    max_depth: int
    random_seed: int
    search_config_hash: str

    # telemetry
    candidates_generated: int
    candidates_valid: int
    domain_reject_fraction: float
    runtime_seconds: float

    # quality
    top1_expr: str
    top1_mse: float
    top5_mses: tuple[float, ...]
    match_category: MatchCategory

    # model complexity
    winner_depth: int
    winner_size: int

    @property
    def exact_recovered(self) -> bool:
        return self.match_category == "exact"

    def to_record(self) -> dict[str, Any]:
        row = dict(self.__dict__)
        row["top5_mses"] = "|".join(f"{m:.16g}" for m in self.top5_mses)
        return row


def to_csv(results: list[RecoveryResult], path: str | Path, append: bool = True) -> Path:
    """Write benchmark rows to CSV on disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r.to_record() for r in results])
    if append and out_path.exists():
        frame.to_csv(out_path, mode="a", index=False, header=False)
    else:
        frame.to_csv(out_path, index=False)
    return out_path


def to_parquet(results: list[RecoveryResult], path: str | Path) -> Path:
    """Write benchmark rows to parquet on disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r.to_record() for r in results])
    frame.to_parquet(out_path, index=False)
    return out_path


def _domain_config(case: RecoveryCase) -> dict[str, float | int]:
    return {
        "x_min": case.x_min,
        "x_max": case.x_max,
        "num_points": case.num_points,
    }


def _stable_hash(data: str) -> str:
    return sha256(data.encode("utf-8")).hexdigest()[:16]


def _match_category(best: SearchResult | None, case: RecoveryCase) -> MatchCategory:
    if best is None:
        return "none"
    if case.target is not None and pretty(best.expr) == pretty(case.target):
        return "exact"
    if case.target is not None and canonical_string(best.expr) == canonical_string(case.target):
        return "algebraic"
    if best.mse <= 1e-10:
        return "numeric"
    return "none"


def _match_category_baseline(expr_text: str, mse: float, case: RecoveryCase) -> MatchCategory:
    if case.target is not None and expr_text == pretty(case.target):
        return "exact"
    if mse <= 1e-10:
        return "numeric"
    return "none"


def _compute_target(case: RecoveryCase, x_grid: np.ndarray) -> np.ndarray:
    if case.target is not None:
        return evaluate(case.target, x_grid)
    if case.target_factory is not None:
        return np.asarray(case.target_factory(x_grid), dtype=float)
    raise ValueError(f"RecoveryCase '{case.name}' must provide target or target_factory.")


def run_recovery_case(case: RecoveryCase, top_k: int = 5) -> list[RecoveryResult]:
    """Run one benchmark case across all regimes under a shared budget."""
    x_grid = np.linspace(case.x_min, case.x_max, case.num_points)
    y_target = _compute_target(case, x_grid)

    domain_cfg = _domain_config(case)
    domain_config_text = (
        f"x_min={domain_cfg['x_min']:.12g};"
        f"x_max={domain_cfg['x_max']:.12g};"
        f"num_points={domain_cfg['num_points']}"
    )
    search_config_text = f"max_depth={case.max_depth};top_k={top_k};seed={case.random_seed};dedupe_signatures=True"

    results: list[RecoveryResult] = []

    for regime, leaf_regime in (("e_only", "e_only"), ("e_plus_affine_leaf", "e_plus_affine")):
        start = perf_counter()
        report = shallow_search_with_report(
            x_grid=x_grid,
            y_target=y_target,
            max_depth=case.max_depth,
            top_k=top_k,
            dedupe_signatures=True,
            leaf_regime=leaf_regime,
        )
        runtime_seconds = perf_counter() - start
        ranked = list(report.results)
        generated = int(sum(row.generated for row in report.generation_stats.per_depth))
        valid = int(sum(row.valid for row in report.generation_stats.per_depth))
        reject_fraction = 0.0 if generated == 0 else 1.0 - (valid / generated)
        best = ranked[0] if ranked else None
        top5 = tuple(row.mse for row in ranked[:5])

        results.append(
            RecoveryResult(
                target_name=case.name,
                target_family=case.family,
                tier=case.tier,
                regime=regime,
                domain_config=domain_config_text,
                domain_config_hash=_stable_hash(domain_config_text),
                max_depth=case.max_depth,
                random_seed=case.random_seed,
                search_config_hash=_stable_hash(search_config_text),
                candidates_generated=generated,
                candidates_valid=valid,
                domain_reject_fraction=reject_fraction,
                runtime_seconds=runtime_seconds,
                top1_expr=pretty(best.expr) if best is not None else "<none>",
                top1_mse=best.mse if best is not None else float("inf"),
                top5_mses=top5,
                match_category=_match_category(best, case),
                winner_depth=depth(best.expr) if best is not None else 0,
                winner_size=size(best.expr) if best is not None else 0,
            )
        )

    for regime_name, fit_fn in [
        ("baseline_constant", lambda: constant_least_squares(x_grid=x_grid, y_target=y_target)),
        ("baseline_affine", lambda: affine_least_squares(x_grid=x_grid, y_target=y_target)),
        ("baseline_grammar", lambda: tiny_grammar_search(x_grid=x_grid, y_target=y_target, max_depth=case.max_depth)),
    ]:
        start = perf_counter()
        fit = fit_fn()
        runtime_seconds = perf_counter() - start

        results.append(
            RecoveryResult(
                target_name=case.name,
                target_family=case.family,
                tier=case.tier,
                regime=regime_name,
                domain_config=domain_config_text,
                domain_config_hash=_stable_hash(domain_config_text),
                max_depth=case.max_depth,
                random_seed=case.random_seed,
                search_config_hash=_stable_hash(search_config_text),
                candidates_generated=0,
                candidates_valid=0,
                domain_reject_fraction=0.0,
                runtime_seconds=runtime_seconds,
                top1_expr=fit.expr,
                top1_mse=fit.mse,
                top5_mses=(fit.mse,),
                match_category=_match_category_baseline(fit.expr, fit.mse, case),
                winner_depth=0,
                winner_size=0,
            )
        )

    return results


def run_recovery_suite(cases: list[RecoveryCase], top_k: int = 5) -> list[RecoveryResult]:
    """Run multiple recovery cases across all regimes."""
    out: list[RecoveryResult] = []
    for case in cases:
        out.extend(run_recovery_case(case, top_k=top_k))
    return out
