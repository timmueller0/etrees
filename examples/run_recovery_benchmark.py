"""Run a small exact-recovery benchmark suite."""

from __future__ import annotations

from etree.ast import Constant, ENode, Variable
from etree.benchmarks import RecoveryCase, run_recovery_suite


if __name__ == "__main__":
    x = Variable("x")
    one = Constant(1.0)

    cases = [
        RecoveryCase(name="identity", target=x, max_depth=1),
        RecoveryCase(name="e_x", target=ENode(x, one), max_depth=2),
        RecoveryCase(name="nested", target=ENode(x, ENode(x, one)), max_depth=3),
    ]

    results = run_recovery_suite(cases)
    for row in results:
        print(
            f"{row.name}: exact={row.exact_recovered} best_mse={row.best_mse:.3e} expr={row.best_expr}"
        )
