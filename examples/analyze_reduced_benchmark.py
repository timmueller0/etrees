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
    parser.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.in_paths]
    df = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(df)} rows from {len(args.in_paths)} file(s).")

    summary = (
        df.groupby(["family", "regime"])
        .agg(
            runs=("target", "count"),
            median_mse=("best_mse", "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            exact_rate=("recovery_label", lambda s: (s == "exact").mean()),
            near_rate=("recovery_label", lambda s: (s == "near").mean()),
        )
        .sort_index()
    )
    _print_table("Summary by family x regime", summary)

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
    _print_table("Growth / pruning summary", growth)

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
            miss=lambda x: (x["recovery_label"] == "miss").astype(float),
        )
        .groupby(["regime", "interval"])
        .agg(
            exact_rate=("exact", "mean"),
            near_rate=("near", "mean"),
            miss_rate=("miss", "mean"),
            median_mse=("best_mse", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .sort_values(["exact_rate", "near_rate", "median_mse"], ascending=[False, False, True])
    )

    print("\nMain empirical patterns")
    if not rates.empty:
        best_idx = rates.index[0]
        best = rates.iloc[0]
        print(
            f"- Strongest recovery slice: {best_idx[0]} @ {best_idx[1]} "
            f"(exact={best['exact_rate']:.2f}, near={best['near_rate']:.2f}, miss={best['miss_rate']:.2f})."
        )

        weakest_idx = rates.index[-1]
        weakest = rates.iloc[-1]
        print(
            f"- Weakest recovery slice: {weakest_idx[0]} @ {weakest_idx[1]} "
            f"(exact={weakest['exact_rate']:.2f}, near={weakest['near_rate']:.2f}, miss={weakest['miss_rate']:.2f})."
        )

    if {"hybrid_affine", "e_only"}.issubset(set(df["regime"].unique())):
        hybrid_median = df[df["regime"] == "hybrid_affine"]["best_mse"].median()
        e_only_median = df[df["regime"] == "e_only"]["best_mse"].median()
        relation = "lower" if hybrid_median < e_only_median else "higher"
        print(f"- Hybrid affine median MSE is {relation} than e_only ({hybrid_median:.3e} vs {e_only_median:.3e}).")

    if {"baseline", "hybrid_affine"}.issubset(set(df["regime"].unique())):
        base_runtime = df[df["regime"] == "baseline"]["runtime_sec"].median()
        hybrid_runtime = df[df["regime"] == "hybrid_affine"]["runtime_sec"].median()
        print(f"- Baseline median runtime is {base_runtime:.4f}s vs hybrid_affine {hybrid_runtime:.4f}s.")

    if "negative_control" in set(df.get("tier", [])):
        # Defensive no-op: reduced schema has no tier column, keep script robust if present.
        neg = df[df["tier"] == "negative_control"]
        if not neg.empty:
            miss_rate = (neg["recovery_label"] == "miss").mean()
            print(f"- Negative controls miss at rate {miss_rate:.2f}, as expected for hard targets.")


if __name__ == "__main__":
    main()
