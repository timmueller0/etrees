import numpy as np

from etree.ast import Constant, ENode, Variable, pretty
from etree.eval import evaluate
from etree.search import hybrid_search_with_affine_input, shallow_search, shallow_search_with_report


def test_search_recovers_native_expression() -> None:
    target = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    x = np.linspace(-0.8, 0.8, 60)
    y = evaluate(target, x)

    results = shallow_search(x_grid=x, y_target=y, max_depth=3, top_k=5, dedupe_signatures=True)

    assert results, "Expected at least one result"
    assert pretty(results[0].expr) == pretty(target)
    assert results[0].mse < 1e-12


def test_search_report_contains_generation_stats() -> None:
    target = ENode(Variable("x"), Constant(1.0))
    x = np.linspace(-0.8, 0.8, 40)
    y = evaluate(target, x)

    report = shallow_search_with_report(x_grid=x, y_target=y, max_depth=2, top_k=3)

    assert report.results
    assert len(report.generation_stats.per_depth) == 2
    assert report.generation_stats.per_depth[0].generated == 2
    assert report.generation_stats.per_depth[1].generated == 4


def test_hybrid_search_recovers_affine_input_transform() -> None:
    x = np.linspace(-0.6, 0.6, 40)
    target_expr = ENode(Variable("x"), Constant(1.0))
    y_target = evaluate(target_expr, (2.0 * x) + 1.0)

    results = hybrid_search_with_affine_input(
        x_grid=x,
        y_target=y_target,
        max_depth=2,
        top_k=3,
        a_grid=(-2.0, -1.0, 1.0, 2.0),
        b_grid=(0.0, 1.0),
    )

    assert results
    assert results[0].mse < 1e-12
    assert results[0].a == 2.0
    assert results[0].b == 1.0
