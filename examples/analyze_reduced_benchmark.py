"""Analyze reduced benchmark exports and print compact empirical patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    if frame.empty:
        print("<empty>")
        return
    print(frame.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("outputs/reduced_benchmark.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)

    summary = (
        df.groupby(["family", "regime"])
        .agg(
            runs=("target", "count"),
            median_mse=("best_mse", "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            exact_rate=("recovery_label", lambda s: (s == "exact").mean()),
        )
        .sort_index()
    )
    _print_table("Summary by family/regime", summary)

    growth = (
        df[df["regime"].isin(["e_only", "hybrid_affine"])]
        .assign(
            prune_frac=lambda x: 1.0 - (x["valid_count"] / x["generated_count"].replace(0, pd.NA)),
            dedupe_frac=lambda x: 1.0 - (x["deduped_count"] / x["valid_count"].replace(0, pd.NA)),
        )
        .groupby(["regime", "interval"])
        .agg(
            generated_mean=("generated_count", "mean"),
            valid_mean=("valid_count", "mean"),
            deduped_mean=("deduped_count", "mean"),
            prune_frac_mean=("prune_frac", "mean"),
            dedupe_frac_mean=("dedupe_frac", "mean"),
        )
        .sort_index()
    )
    _print_table("Growth/pruning summary", growth)

    recovery = (
        df.groupby(["regime", "interval", "recovery_label"])
        .size()
        .rename("count")
        .reset_index()
        .pivot_table(
            index=["regime", "interval"],
            columns="recovery_label",
            values="count",
            fill_value=0,
        )
        .sort_index()
    )
    _print_table("Recovery-rate summary (counts)", recovery)

    rates = (
        df.assign(
            exact=lambda x: (x["recovery_label"] == "exact").astype(float),
            near=lambda x: (x["recovery_label"] == "near").astype(float),
        )
        .groupby(["regime", "interval"])
        .agg(exact_rate=("exact", "mean"), near_rate=("near", "mean"), median_mse=("best_mse", "median"))
        .sort_values(["exact_rate", "near_rate", "median_mse"], ascending=[False, False, True])
    )

    print("\nMain empirical patterns from reduced suite")
    top = rates.head(1)
    worst = rates.tail(1)
    if not top.empty:
        idx = top.index[0]
        row = top.iloc[0]
        print(
            f"- Best recovery regime/interval: {idx[0]} @ {idx[1]} "
            f"(exact_rate={row['exact_rate']:.2f}, near_rate={row['near_rate']:.2f}, median_mse={row['median_mse']:.3e})."
        )
    if not worst.empty:
        idx = worst.index[0]
        row = worst.iloc[0]
        print(
            f"- Weakest recovery regime/interval: {idx[0]} @ {idx[1]} "
            f"(exact_rate={row['exact_rate']:.2f}, near_rate={row['near_rate']:.2f}, median_mse={row['median_mse']:.3e})."
        )

    hybrid = rates.xs("hybrid_affine", level="regime", drop_level=False) if ("hybrid_affine" in rates.index.get_level_values(0)) else pd.DataFrame()
    eonly = rates.xs("e_only", level="regime", drop_level=False) if ("e_only" in rates.index.get_level_values(0)) else pd.DataFrame()
    if not hybrid.empty and not eonly.empty:
        print(
            "- Hybrid affine generally improves fit over E-only when depth/interval are held fixed "
            "if its exact/near rates and median MSE dominate corresponding E-only entries."
        )


if __name__ == "__main__":
    main()
