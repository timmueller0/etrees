from etree.ast import Constant, ENode, Variable
from etree.benchmarks import RecoveryCase, run_recovery_case


def test_recovery_case_reports_exact_match_for_simple_target() -> None:
    target = ENode(Variable("x"), Constant(1.0))
    case = RecoveryCase(name="simple", target=target, max_depth=2, num_points=40)

    out = run_recovery_case(case)

    assert out.name == "simple"
    assert out.exact_recovered is True
    assert out.best_mse < 1e-10
