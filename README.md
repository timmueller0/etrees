# etree

`etree` is a lightweight Python research prototype for exploring **E-logic / E-trees**:

\[
E(x, y) = \exp(x) - \log(y)
\]

The project focuses on fast iteration for early-stage symbolic-expression experiments.

## What this prototype includes

### 1) Expression representation
- A small AST with:
  - `Variable(name)`
  - `Constant(value)`
  - `AffineLeaf(variable, a, b)` for `a*x+b`
  - `ENode(left, right)`
- Utilities for:
  - pretty-printing
  - `depth()` and `size()`
  - subtree extraction

### 2) Evaluation and search
- Numerical evaluation on scalar/array inputs (`numpy`).
- Clean custom exceptions for:
  - domain errors (`log(right)` with `right <= 0`)
  - non-finite outputs.
- Exhaustive tree generation up to configurable depth.
- Optional signature deduplication using sampled outputs.
- Shallow search that ranks formulas by MSE against target samples.
- Hybrid search mode: structural brute force + affine input transform sweep (`x' = a*x + b`).

### 3) Retrieval / representation experiments
- Subtree-multiset features for each formula.
- Similarity via multiset-Jaccard overlap.
- Baseline naive token-set similarity for contrast.

### 4) Benchmarking and diagnostics
- Per-depth generation telemetry via `generate_trees_with_stats`.
- Search reports that include growth diagnostics (`shallow_search_with_report`).
- Recovery benchmark harness (`RecoveryCase`, `run_recovery_suite`).

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
    canonicalize.py
    features.py
    utils.py
  tests/
    test_ast.py
    test_eval.py
    test_generate.py
    test_search.py
    test_features.py
  examples/
    recover_native_expression.py
    toy_espace_demo.py
    benchmark_growth.py
    run_recovery_benchmark.py
    hybrid_affine_search_demo.py
```

## Quick start

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

## How to run examples from fresh checkout

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m examples.run_recovery_benchmark
```

All examples in `examples/` are supported via module execution (`python -m examples.<name>`).

## Notes on design

This code intentionally prioritizes:
- readability,
- hackability,
- low ceremony,
- easy extension for small research loops.

## Three best next steps

1. **Extend the leaf set and variable support**
   - Add constants beyond `1` and multi-variable support (`x, y, z`) with simple typing/validation rules.
2. **Improve search efficiency and robustness**
   - Add caching/memoization, grammar constraints, and train/validation splits to reduce overfitting and speed up larger sweeps.
3. **Richer E-space representations**
   - Add canonicalization, weighted subtree kernels, and embedding-style retrieval features for better structural similarity experiments.

## Reduced benchmark smoke snapshot (April 16, 2026)

Generated with:

```bash
PYTHONPATH=src python -m examples.run_reduced_benchmark --out outputs/reduced_benchmark_smoke.csv
PYTHONPATH=src python -m examples.analyze_reduced_benchmark --in outputs/reduced_benchmark_smoke.csv
```

### Technical summary (research-facing)

| Slice | Exact rate | Approx rate | Median MSE | Median runtime |
|---|---:|---:|---:|---:|
| hybrid_affine @ wide | 0.32 | 0.00 | 8.41e-03 | 4.38e+00 s |
| e_only @ wide | 0.32 | 0.00 | 2.49e-01 | 3.06e-03 s |
| baseline @ wide | 0.08 | 0.24 | 1.09e-02 | 1.80e-04 s |
| hybrid_affine (all rows) | 0.23 | 0.00 | 1.46e-02 | 1.49e-02 s |
| e_only (all rows) | 0.23 | 0.00 | 2.49e-01 | 4.13e-04 s |
| baseline (all rows) | 0.08 | 0.23 | 6.60e-03 | 1.36e-04 s |

Additional diagnostics now included in each row:
- `domain_clipped` (whether the requested interval was clipped to the case domain),
- `effective_interval` (actual `[lo, hi]` used),
- `useful_fit_label` (`exact`, `near`, `approx`, `miss`).

### Non-technical summary

| What we tested | What happened |
|---|---|
| Narrow/normal/wide x-ranges and small/medium/large depth limits | Wider ranges with hybrid search gave the best chance of exact matches in this smoke run. |
| Three modes (`e_only`, `hybrid_affine`, `baseline`) | `baseline` is much faster, but mostly returns rough approximations rather than exact symbolic recovery. |
| Focused growth/saturation sweep (3 intervals x 3 depths) | Going deeper improves recovery; in this specific smoke run there were **no** growth targets where wide was exact but both tight/default were non-exact. |
