# ILUp — Incomplete LU Factorization (ILU(p)) in Python

> A lightweight Python implementation that builds **ILU(p) preconditioners** for sparse matrices at a chosen fill level *p*.

## Features
- **`ilu_p(A, p)`**  
  Returns a unit-lower-triangular matrix **L** and an upper-triangular matrix **U** ready to serve as a preconditioner for iterative solvers.
- **`compute_levels(A, p)`**  
  Computes level-of-fill information to illustrate how extra fill evolves.
- **`demo.ipynb` notebook**  
  Visualizes the impact of different *p* values on sparsity patterns and convergence (`experiments.png`, `level.png`).

## Requirements
Use `uv`:

```bash
# Install uv first, if you haven't
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Repository Layout

```
ilup.py        # core implementation
demo.ipynb     # interactive example
```