from etree.ast import AffineLeaf, Constant
from etree.canonicalize import canonical_string


def test_canonical_string_normalizes_integerish_constants() -> None:
    assert canonical_string(Constant(1.0)) == "Const(1)"
    assert canonical_string(Constant(1.0000000000)) == "Const(1)"


def test_canonical_string_preserves_significant_precision() -> None:
    assert canonical_string(Constant(1.25)) == "Const(1.25)"


def test_canonical_string_supports_affine_leaf() -> None:
    assert canonical_string(AffineLeaf(variable="x", a=2.0, b=-1.0)) == "Affine(x,2,-1)"
