from etree.ast import AffineLeaf, Constant, ENode, Variable, depth, pretty, size, subtrees


def test_depth_and_size_for_nested_expression() -> None:
    expr = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    assert depth(expr) == 3
    assert size(expr) == 5


def test_pretty_and_subtrees() -> None:
    expr = ENode(Variable("x"), Constant(1.0))
    assert pretty(expr) == "E(x, 1)"
    sts = subtrees(expr)
    assert len(sts) == 3


def test_affine_leaf_is_single_node_tree() -> None:
    expr = AffineLeaf(variable="x", a=2.0, b=1.0)
    assert depth(expr) == 1
    assert size(expr) == 1
    assert pretty(expr) == "(2*x+1)"
