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

### 3) Retrieval / representation experiments
- Subtree-multiset features for each formula.
- Similarity via multiset-Jaccard overlap.
- Baseline naive token-set similarity for contrast.

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
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Run demos:

```bash
python examples/recover_native_expression.py
python examples/toy_espace_demo.py
```

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
