import numpy as np

from etree.ast import Constant, ENode, Variable, pretty
from etree.eval import evaluate
from etree.search import shallow_search


def test_search_recovers_native_expression() -> None:
    target = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    x = np.linspace(-0.8, 0.8, 60)
    y = evaluate(target, x)

    results = shallow_search(x_grid=x, y_target=y, max_depth=3, top_k=5, dedupe_signatures=True)

    assert results, "Expected at least one result"
    assert pretty(results[0].expr) == pretty(target)
    assert results[0].mse < 1e-12
