from etree.generate import generate_trees


def test_generation_counts_small_depths() -> None:
    d1 = generate_trees(max_depth=1)
    d2 = generate_trees(max_depth=2)

    assert len(d1) == 2
    assert len(d2) == 6
