from etree.ast import Constant, ENode, Variable
from etree.features import subtree_multiset, subtree_similarity, token_set_similarity


def test_subtree_features_capture_structure() -> None:
    base = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    structurally_related = ENode(Variable("x"), ENode(Constant(1.0), Variable("x")))
    shallow_variant = ENode(Variable("x"), Constant(1.0))

    sim_related = subtree_similarity(base, structurally_related)
    sim_shallow = subtree_similarity(base, shallow_variant)

    assert sim_related != sim_shallow
    assert 0.0 <= sim_related <= 1.0
    assert 0.0 <= sim_shallow <= 1.0
    assert sum(subtree_multiset(base).values()) == 5


def test_naive_token_similarity_is_less_discriminative() -> None:
    a = ENode(Variable("x"), ENode(Variable("x"), Constant(1.0)))
    b = ENode(Variable("x"), ENode(Constant(1.0), Variable("x")))
    c = ENode(Constant(1.0), ENode(Variable("x"), Variable("x")))

    # same token inventory across all three forms
    assert token_set_similarity(a, b) == 1.0
    assert token_set_similarity(a, c) == 1.0
