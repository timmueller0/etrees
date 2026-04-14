"""Quick script to inspect E-tree growth by depth."""

from __future__ import annotations

import numpy as np

from etree.generate import generate_trees_with_stats


if __name__ == "__main__":
    x = np.linspace(-0.8, 0.8, 40)
    _, stats = generate_trees_with_stats(max_depth=4, x_grid=x, dedupe_signatures=True)

    print("depth\tgenerated\tvalid\tdeduped\telapsed_s")
    for row in stats.per_depth:
        print(
            f"{row.depth}\t{row.generated}\t{row.valid}\t{row.deduplicated}\t{row.elapsed_seconds:.6f}"
        )
