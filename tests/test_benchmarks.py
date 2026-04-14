from etree.ast import Constant, ENode, Variable
from etree.benchmarks import RecoveryCase, run_recovery_case


def test_recovery_case_runs_all_regimes() -> None:
    target = ENode(Variable("x"), Constant(1.0))
    case = RecoveryCase(name="simple", target=target, max_depth=2, num_points=40)

    out = run_recovery_case(case)

    assert len(out) == 5
    regimes = {row.regime for row in out}
    assert regimes == {
        "e_only",
        "e_plus_affine_leaf",
        "baseline_constant",
        "baseline_affine",
        "baseline_grammar",
    }
    e_only = next(row for row in out if row.regime == "e_only")
    assert e_only.exact_recovered is True
    assert e_only.top1_mse < 1e-10
    assert e_only.candidates_generated >= e_only.candidates_valid
