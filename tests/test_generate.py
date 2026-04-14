import numpy as np

from etree.generate import affine_leaves, generate_trees, generate_trees_with_stats


def test_generation_counts_small_depths() -> None:
    d1 = generate_trees(max_depth=1)
    d2 = generate_trees(max_depth=2)

    assert len(d1) == 2
    assert len(d2) == 6


def test_generation_stats_include_expected_depth_rows() -> None:
    x = np.linspace(-0.8, 0.8, 20)
    _, stats = generate_trees_with_stats(max_depth=2, x_grid=x, dedupe_signatures=True)

    assert len(stats.per_depth) == 2
    assert stats.per_depth[0].depth == 1
    assert stats.per_depth[1].depth == 2
    assert stats.per_depth[0].generated == 2
    assert stats.per_depth[1].generated == 4
    assert stats.per_depth[0].deduplicated <= stats.per_depth[0].valid
    assert stats.per_depth[1].deduplicated <= stats.per_depth[1].valid


def test_affine_leaf_bank_size() -> None:
    leaves = affine_leaves(slopes=(-1.0, 1.0), intercepts=(0.0, 1.0))
    assert len(leaves) == 4
