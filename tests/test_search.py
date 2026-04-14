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
    base = ENode(Variable("x"), Constant(1.0))
    x = np.linspace(-0.6, 0.6, 60)
    y = evaluate(base, 2.0 * x + 1.0)

    results = hybrid_search_with_affine_input(
        x_grid=x,
        y_target=y,
        max_depth=2,
        top_k=5,
        a_grid=(-2.0, -1.0, 0.5, 1.0, 2.0),
        b_grid=(-1.0, 0.0, 1.0),
    )

    assert results
    assert any(
        pretty(result.expr) == pretty(base)
        and np.isclose(result.a, 2.0)
        and np.isclose(result.b, 1.0)
        and result.mse < 1e-12
        for result in results
    )



def test_shallow_search_accepts_affine_leaf_regime() -> None:
    x = np.linspace(-0.8, 0.8, 50)
    y = 2.0 * x + 1.0

    results = shallow_search(
        x_grid=x,
        y_target=y,
        max_depth=1,
        top_k=3,
        leaf_regime="e_plus_affine",
    )

    assert results
    assert results[0].mse < 1e-12
