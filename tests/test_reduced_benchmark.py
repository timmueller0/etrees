from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from etree.ast import Constant, ENode, Variable
from etree.benchmarks import RecoveryCase
from etree.reduced_benchmark import IntervalBudget, run_reduced_suite


def _small_budgets() -> tuple[IntervalBudget, ...]:
    return (
        IntervalBudget(interval="tiny_a", x_min=-0.5, x_max=0.5, max_depth=1, num_points=20),
        IntervalBudget(interval="tiny_b", x_min=-0.8, x_max=0.8, max_depth=2, num_points=20),
        IntervalBudget(interval="tiny_c", x_min=-1.0, x_max=1.0, max_depth=2, num_points=20),
    )


def test_reduced_suite_schema_and_regimes() -> None:
    case = RecoveryCase(name="simple", target=ENode(Variable("x"), Constant(1.0)))
    out = run_reduced_suite(cases=[case], budgets=_small_budgets())

    assert len(out) == 9
    assert set(out["regime"]) == {"e_only", "hybrid_affine", "baseline"}
    assert set(out.columns) == {
        "target",
        "family",
        "regime",
        "interval",
        "domain_clipped",
        "effective_interval",
        "max_depth",
        "generated_count",
        "valid_count",
        "deduped_count",
        "best_mse",
        "runtime_sec",
        "winner_expr",
        "winner_depth",
        "winner_size",
        "recovery_label",
        "useful_fit_label",
    }


def test_reduced_suite_csv_and_analysis_ingest(tmp_path: Path) -> None:
    case = RecoveryCase(name="simple", target=ENode(Variable("x"), Constant(1.0)))
    out = run_reduced_suite(cases=[case], budgets=_small_budgets())

    csv_path = tmp_path / "reduced.csv"
    out.to_csv(csv_path, index=False)

    loaded = pd.read_csv(csv_path)
    assert len(loaded) == len(out)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.analyze_reduced_benchmark",
            "--in",
            str(csv_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert "Main empirical patterns" in proc.stdout
    assert "Recovery by domain_clipped" in proc.stdout
