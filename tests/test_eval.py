import numpy as np
import pytest

from etree.ast import AffineLeaf, Constant, ENode, Variable
from etree.eval import DomainError, evaluate


def test_eval_native_expression_matches_exp_minus_x() -> None:
    expr = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    x = np.array([-1.0, 0.0, 1.0])
    y = evaluate(expr, x)
    np.testing.assert_allclose(y, np.exp(x) - x)


def test_eval_affine_leaf() -> None:
    expr = AffineLeaf(variable="x", a=2.0, b=-1.0)
    x = np.array([-1.0, 0.0, 1.0])
    y = evaluate(expr, x)
    np.testing.assert_allclose(y, 2.0 * x - 1.0)


def test_domain_error_for_nonpositive_log_argument() -> None:
    expr = ENode(Constant(1.0), Variable("x"))
    x = np.array([-1.0, 0.5, 1.0])
    with pytest.raises(DomainError):
        evaluate(expr, x)
