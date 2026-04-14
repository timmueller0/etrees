from etree.ast import Constant, ENode, Variable
from etree.benchmarks import RecoveryCase
from etree.reduced_benchmark import IntervalBudget, run_reduced_suite


def test_reduced_suite_schema_and_regimes() -> None:
    case = RecoveryCase(name="simple", target=ENode(Variable("x"), Constant(1.0)))
    budgets = (
        IntervalBudget(interval="tiny_a", x_min=-0.5, x_max=0.5, max_depth=1, num_points=20),
        IntervalBudget(interval="tiny_b", x_min=-0.8, x_max=0.8, max_depth=2, num_points=20),
        IntervalBudget(interval="tiny_c", x_min=-1.0, x_max=1.0, max_depth=2, num_points=20),
    )
    out = run_reduced_suite(cases=[case], budgets=budgets)

    assert len(out) == len(budgets) * 3
    assert set(out["regime"]) == {"e_only", "hybrid_affine", "baseline"}
    assert set(out.columns) == {
        "target",
        "family",
        "regime",
        "interval",
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
    }
