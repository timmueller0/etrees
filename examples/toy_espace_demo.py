"""Tiny retrieval demo comparing token vs subtree similarity."""

from __future__ import annotations

import pandas as pd

from etree.ast import Constant, ENode, Variable, pretty
from etree.features import subtree_similarity, token_set_similarity


def main() -> None:
    query = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    library = [
        ENode(Variable("x"), ENode(Constant(1.0), Variable("x"))),
        ENode(Constant(1.0), ENode(Variable("x"), Variable("x"))),
        ENode(Variable("x"), Constant(1.0)),
        ENode(ENode(Variable("x"), Constant(1.0)), Variable("x")),
    ]

    rows = []
    for expr in library:
        rows.append(
            {
                "formula": pretty(expr),
                "token_sim": token_set_similarity(query, expr),
                "subtree_sim": subtree_similarity(query, expr),
            }
        )

    frame = pd.DataFrame(rows).sort_values("subtree_sim", ascending=False)
    print("Query:", pretty(query))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
