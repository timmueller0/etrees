"""Demo: recover affine input transform with hybrid search."""

from __future__ import annotations

import numpy as np

from etree.ast import Constant, ENode, Variable
from etree.eval import evaluate
from etree.search import hybrid_search_with_affine_input


if __name__ == "__main__":
    x = np.linspace(-0.6, 0.6, 60)
    base = ENode(Variable("x"), Constant(1.0))
    y = evaluate(base, 2.0 * x + 1.0)

    results = hybrid_search_with_affine_input(
        x_grid=x,
        y_target=y,
        max_depth=2,
        top_k=5,
        a_grid=(-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
        b_grid=(-1.0, 0.0, 0.5, 1.0),
    )

    for r in results[:3]:
        print(f"mse={r.mse:.3e}\ta={r.a}\tb={r.b}\texpr={r.expr}")
