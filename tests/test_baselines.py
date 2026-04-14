import numpy as np

from etree.baselines import affine_least_squares, constant_least_squares, tiny_grammar_search


def test_constant_and_affine_least_squares() -> None:
    x = np.linspace(-1.0, 1.0, 100)
    y = 3.0 * x - 2.0

    constant_fit = constant_least_squares(x, y)
    affine_fit = affine_least_squares(x, y)

    assert constant_fit.mse > 0.1
    assert affine_fit.mse < 1e-12


def test_tiny_grammar_search_finds_affine_leaf_expression() -> None:
    x = np.linspace(-0.8, 0.8, 60)
    y = 2.0 * x + 1.0

    out = tiny_grammar_search(x, y, max_depth=2)

    assert out.mse < 1e-12
