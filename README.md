# etree

`etree` is a lightweight Python research prototype for exploring **E-logic / E-trees**, built around the single operator

\[
E(x, y) = \exp(x) - \log(y)
\]

The project is designed for **fast iteration on early-stage symbolic-expression experiments**. It supports symbolic tree generation, numerical evaluation, shallow formula search, retrieval experiments, and benchmark-driven comparison of recovery regimes.

## Why this repo exists

The main goal is not human-readable notation. It is to explore whether a **uniform recursive tree language** can be useful for machine reasoning about formulas.

In particular, `etree` is a sandbox for questions like:

- Can shallow E-tree search recover hidden symbolic structure from sampled data?
- When does E-logic act as a useful **compression language** rather than an exact recovery language?
- Does affine augmentation improve symbolic recovery, or mainly numeric fit?
- Can subtree-based representations create a useful **formula geometry** for retrieval and comparison?

---

## Current research status

Current smoke-benchmark results suggest:

- **Works well:** identity/compression cases and some native E targets.
- **Mixed:** growth/saturation families, especially under `hybrid_affine` and wider intervals.
- **Fails cleanly:** adversarial targets and domain-fragile cases.

A key early finding is that **hybrid affine search improves fit quality substantially**, but in the current reduced suite it improves **exact recovery less consistently**.

---

## What this prototype includes

## 1) Expression representation

A small AST with:

- `Variable(name)`
- `Constant(value)`
- `AffineLeaf(variable, a, b)` for \(a x + b\)
- `ENode(left, right)`

Utilities include:

- pretty-printing
- `depth()` and `size()`
- subtree extraction

## 2) Evaluation and search

- Numerical evaluation on scalar or array inputs via `numpy`
- Clean custom exceptions for:
  - domain errors, especially `log(right)` with `right <= 0`
  - non-finite outputs
- Exhaustive tree generation up to configurable depth
- Optional structural and numeric-signature deduplication
- Shallow search that ranks formulas by MSE against sampled target data
- Search reports with growth and pruning telemetry
- Three recovery regimes:
  - `e_only`
  - `hybrid_affine`
  - `baseline`

### Regimes

**`e_only`**  
Searches directly over E-trees in the raw variable \(x\).

**`hybrid_affine`**  
Searches over E-trees after an affine input transform

\[
x' = a x + b
\]

This mixes discrete tree search with a small continuous parameter sweep.

**`baseline`**  
A simple non-E comparison regime used to separate symbolic recovery from cheap approximation.

## 3) Retrieval / representation experiments

- Subtree-multiset features for each formula
- Similarity via multiset-Jaccard overlap
- Naive token-based similarity as a contrast baseline

## 4) Benchmarking and diagnostics

- Per-depth generation telemetry via `generate_trees_with_stats`
- Search reports via `shallow_search_with_report`
- Recovery benchmark harness via `RecoveryCase` and `run_recovery_suite`
- Reduced benchmark runner for compact empirical comparisons
- CSV / Parquet export helpers for downstream analysis

---

## Project layout

```text
etree/
  README.md
  pyproject.toml
  .gitignore
  src/etree/
    __init__.py
    ast.py
    eval.py
    generate.py
    search.py
    benchmarks.py
    reduced_benchmark.py
    canonicalize.py
    features.py
    utils.py
  tests/
    test_ast.py
    test_eval.py
    test_generate.py
    test_search.py
    test_features.py
    test_reduced_benchmark.py
  examples/
    recover_native_expression.py
    toy_espace_demo.py
    benchmark_growth.py
    run_recovery_benchmark.py
    hybrid_affine_search_demo.py
    run_reduced_benchmark.py
    analyze_reduced_benchmark.py
```

## Quick start

Create and activate a virtual environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Run demos as modules:

```bash
python -m examples.recover_native_expression
python -m examples.toy_espace_demo
python -m examples.benchmark_growth
python -m examples.run_recovery_benchmark
python -m examples.hybrid_affine_search_demo
```

## Fresh checkout

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m examples.run_recovery_benchmark
```

All examples in `examples/` are supported via module execution:

```bash
python -m examples.<name>
```

## Reduced benchmark smoke snapshot (April 16, 2026)

Generated with:

```bash
PYTHONPATH=src python -m examples.run_reduced_benchmark --out outputs/reduced_benchmark_smoke.csv
PYTHONPATH=src python -m examples.analyze_reduced_benchmark --in outputs/reduced_benchmark_smoke.csv
```

## Technical summary

| Slice | Exact rate | Approx rate | Median MSE | Median runtime |
|---|---:|---:|---:|---:|
| hybrid_affine @ wide | 0.32 | 0.00 | 8.41e-03 | 4.38e+00 s |
| e_only @ wide | 0.32 | 0.00 | 2.49e-01 | 3.06e-03 s |
| baseline @ wide | 0.08 | 0.24 | 1.09e-02 | 1.80e-04 s |
| hybrid_affine (all rows) | 0.23 | 0.00 | 1.46e-02 | 1.49e-02 s |
| e_only (all rows) | 0.23 | 0.00 | 2.49e-01 | 4.13e-04 s |
| baseline (all rows) | 0.08 | 0.23 | 6.60e-03 | 1.36e-04 s |

In this smoke run, `hybrid_affine` substantially improved fit quality over `e_only`, especially on wide intervals, but did not improve overall exact recovery rate across all rows. The `baseline` remained much faster, but mostly delivered approximations rather than symbolic recovery.

### Additional diagnostics included in each row

- `domain_clipped` — whether the requested interval was clipped to the case domain
- `effective_interval` — actual `[lo, hi]` used
- `useful_fit_label` — one of `exact`, `near`, `approx`, `miss`

## Non-technical summary

| What we tested | What happened |
|---|---|
| Narrow/normal/wide x-ranges and small/medium/large depth limits | Wider ranges with hybrid search gave the best chance of exact matches in this smoke run. |
| Three modes (`e_only`, `hybrid_affine`, `baseline`) | `baseline` is much faster, but mostly returns rough approximations rather than exact symbolic recovery. |
| Focused growth/saturation sweep (3 intervals x 3 depths) | Going deeper improves recovery; in this smoke run there were no growth targets where wide was exact but both tight/default were non-exact. |

## Design principles

This code intentionally prioritizes:

- readability
- hackability
- low ceremony
- easy extension for small research loops

It is an exploratory research codebase, not a production symbolic algebra system.

## Current limitations

- Search cost grows quickly with depth.
- Exact recovery is still concentrated in aligned target families.
- `hybrid_affine` improves fit more consistently than exact symbolic recovery.
- Domain restrictions from `log` are a major practical constraint.
- The repo does not yet implement a full compiler from standard formulas into E-form.

## Three best next steps

### 1) Extend the leaf set and variable support

Add constants beyond 1 and move toward multi-variable support (`x`, `y`, `z`) with simple typing and validation rules.

### 2) Improve search efficiency and robustness

Add caching, memoization, grammar constraints, and train/validation splits to reduce overfitting and speed up larger sweeps.

### 3) Richer E-space representations

Add stronger canonicalization, weighted subtree kernels, and embedding-style retrieval features for better structural similarity experiments.

## Research-facing interpretation

The current evidence suggests a useful distinction:

- E-only appears strongest for native or aligned symbolic structure.
- Hybrid affine is useful when the target is structurally similar after a simple input warp.
- Baseline methods can be extremely competitive on approximation while contributing less to exact symbolic rediscovery.

That makes `etree` most promising, at least for now, as a tool for studying the boundary between:

- exact structural recovery,
- symbolic compression,
- and cheap approximation.

## Status

Exploratory, lightweight, and actively evolving.
