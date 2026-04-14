"""Recover a hidden shallow expression using exhaustive search."""

from __future__ import annotations

from etree.ast import Constant, ENode, Variable, pretty
from etree.eval import evaluate
from etree.search import results_to_frame, shallow_search
from etree.utils import make_grid


def main() -> None:
    hidden = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    x = make_grid(-0.8, 0.8, n=120)
    y = evaluate(hidden, x)

    results = shallow_search(x_grid=x, y_target=y, max_depth=3, top_k=10)

    print("Hidden target:", pretty(hidden))
    print(results_to_frame(results).to_string(index=False))


if __name__ == "__main__":
    main()
