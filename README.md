# Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations

Code for the ICML 2026 paper [Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations](https://arxiv.org/pdf/2606.05272).

B-NRDEs extend neural rough differential equations by matching the signature algebra to the calculus and geometry of the problem. They use rooted-tree Hopf algebras to expose Itô quadratic variation and manifold covariant-derivative terms that shuffle signatures hide. This repo contains experiments for rough volatility, SO(3) forecasting, SPD covariance dynamics, baselines, checkpoint evaluation, and table generation.

## Setup

Requires Python 3.13 and `uv`.

```bash
uv sync --frozen
```

The default dependencies target CUDA-enabled JAX. For CPU-only work, adjust the JAX-related dependencies in `pyproject.toml` as needed.

## Train

Configs live in `configs/`. Example runs:

```bash
uv run train --config configs/simple_bergomi/bnrde.toml
uv run train --config configs/sg_so3_sim/nrde.toml
uv run train --config configs/spd_covariance/bnrde_geometric.toml
```

Training writes runs to `saved_models/<run>/` with `config.toml`, `best.eqx`, `last.eqx`, and `metrics.json`.

## Evaluate

Run the test epoch for a saved checkpoint:

```bash
uv run test --run_dir saved_models/<run>
```

Replicate testing:

```bash
uv run test --run_dir saved_models/<run> --seeds 1 2 3
uv run test --run_dir saved_models/<run> --seed-start 1 --seed-count 5
```

Test metrics are written to `test_metrics.json` in the run directory.

## Test Suite

```bash
uv run pytest
uv run pytest tests/test_training_metrics_json.py
```

## Optimize

```bash
uv run optimize --config configs/simple_bergomi/bnrde.toml --n-trials 20 --storage sqlite:///optuna.db
```

## Tables and Data

Generate LaTeX tables from run metrics:

```bash
uv run tables --table rough_vol --run_dir saved_models/<run>
uv run tables --table so3 --run_dir saved_models/<run> --out z_paper_content/tables/so3.tex
```

Inspect `.npz` datasets:

```bash
uv run shapes data
uv run shapes --contains rough
```
