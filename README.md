# Detecting Structural Breaks via the Adaptive Fused Lasso

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Solver-CVXPY%20%7C%20CLARABEL-orange" alt="Solver"/>
  <img src="https://img.shields.io/badge/Status-Research%20Code-lightgrey" alt="Status"/>
</p>

> **Python implementation** of the adaptive fused lasso estimator for detecting multiple structural breaks in panel data models with interactive fixed effects, based on:
>
> **Kaddoura, Y. and Westerlund, J. (2023)**. *Estimation of panel data models with random interactive effects and multiple structural breaks when T is fixed.* Journal of Business & Economic Statistics, 41, 778–790.

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Simulations](#running-the-simulations)
- [Data Generating Processes](#data-generating-processes)
- [Module Reference](#module-reference)
- [Citation](#citation)
- [License](#license)

---

## Overview

Standard panel data estimators assume slope coefficients are constant over time. In practice, economic relationships frequently undergo **structural breaks** — abrupt shifts in regression coefficients caused by policy changes, financial crises, or regime transitions.

This repository implements a **two-step adaptive fused lasso** procedure that:

1. Estimates an initial OLS coefficient path $\hat{\beta}^{\text{OLS}}$ across periods.
2. Solves a penalised least-squares problem that shrinks consecutive differences $\beta_t - \beta_{t-1}$ toward zero, automatically detecting and dating breaks.
3. Selects the regularisation parameter $\lambda$ via a data-driven **Information Criterion (IC)**.

The method is designed for panels with **fixed T** (small time dimension) and **large N**, where common factor structure induces cross-sectional dependence.

---

## Methodology

### Model

The observed panel follows:

```math
\tilde{y}_{it} = \tilde{x}_{it}' \beta_t + \tilde{u}_{it},
\qquad i = 1, \ldots, N, \quad t = 1, \ldots, T
```

where tildes denote cross-sectional demeaning to remove interactive fixed effects $\lambda_i' F_t$.

---

### Adaptive Fused Lasso Objective

```math
\hat{\mathcal{B}}_T(\lambda)
= \underset{\mathcal{B}_T}{\arg\min}
\; \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T}
\bigl(\tilde{y}_{it} - \tilde{x}_{it}' \beta_t \bigr)^2
+ \lambda \sum_{t=2}^{T} w_t \,\|\beta_t - \beta_{t-1}\|_2
```

The **adaptive weights** are

```math
w_t = \|\hat{\beta}_t^{\text{OLS}} - \hat{\beta}_{t-1}^{\text{OLS}}\|_2^{-2}
```

down-weighting differences already large in the first stage, so the lasso preferentially breaks there.

---

### Information Criterion

The tuning parameter $\lambda$ is selected by minimising:

```math
\text{IC}(\lambda)
= \frac{1}{NT} \sum_{t=1}^{T}
\|\tilde{y}_t - \tilde{X}_t \hat{\beta}_t(\lambda)\|_2^2
+ \frac{\log N}{\sqrt{N}} \cdot p \cdot \bigl(\hat{m}(\lambda) + 1\bigr)
```

---

### DGP Structure (DATA3 — Main Specification)

The richest DGP combines:

- **AR(1) common factors:** $F_t = (1-\phi) + \phi F_{t-1} + \eta_t$
- **Cross-sectional + temporal dependence** in idiosyncratic errors $\varepsilon_{it}$
- **Factor-loaded regressors:** $X_{itk} = F_t' \lambda_{ik} + \nu_{itk}$

---

## Repository Structure

## Repository structure

```text
Detecting-Breaks-Via-the-Fused-Lasso/
│
├── src/                        # Core library
│   ├── __init__.py
│   ├── dgp.py                  # Data generating processes (DATA1, DATA2, DATA3)
│   ├── estimator.py            # OLS, FGLS (fused lasso), NBOLS
│   ├── ic.py                   # Information criterion for λ selection
│   └── utils.py                # Reporting, printing, plotting
│
├── simulations/                # Monte Carlo scripts
│   └── N25_T5_m1.py            # N=25, T=5, m=1 replication
│
├── tests/                      # Pytest unit tests
│   └── test_core.py
│
├── docs/                       # LaTeX technical documentation
│   └── documentation.tex
│
├── notebooks/                  # Jupyter notebook demos
│   └── quickstart.ipynb        # Minimal end-to-end example
│
├── figures/                    # Output figures (auto-generated)
├── results/                    # Output tables (auto-generated)
│
├── environment.yml             # Conda environment
├── pyproject.toml              # Package metadata
├── LICENSE
└── README.md

---

## Installation

### Option 1 — Conda (recommended)

```bash
git clone https://github.com/YousefKad/Detecting-Breaks-Via-the-Fused-Lasso.git
cd Detecting-Breaks-Via-the-Fused-Lasso

conda env create -f environment.yml
conda activate fused-lasso-breaks
```

### Option 2 — pip

```bash
git clone https://github.com/YousefKad/Detecting-Breaks-Via-the-Fused-Lasso.git
cd Detecting-Breaks-Via-the-Fused-Lasso

pip install cvxpy numpy scipy matplotlib tabulate
```

---

## Quick Start

```python
import numpy as np
from src.dgp import DATA3
from src.estimator import Optimize
from src.ic import information_criterion
from src.utils import plot_ic_curve, plot_beta_path

# Parameters
n, T, p, m, r = 25, 5, 4, 1, 5

# Generate data
data = DATA3(r=r, m=m, T=T, n=n, p=p, phi=0.8, phi_1=0.4, pi=0.4)
X, y, beta_true, u, eps, F, y_tilde, u_tilde, X_mean, X_tilde = data.DGP1()

# Select λ via IC
lam_grid = np.logspace(-3, 3, 50)
IC_vec, m_breaks, IC_min, lam_idx, lam_star, m_star = information_criterion(
    lam_grid, y_tilde, X_tilde, p, T, n
)

# Estimate
opt = Optimize(p, T, n)
b_ols, _, _        = opt.OLS(X_tilde, y_tilde)
b_hat, m_hat, _, _ = opt.FGLS(X_tilde, y_tilde, b_ols, lam_star)

print(f"True breaks: {m}  |  Estimated: {m_hat}")
print(f"Optimal λ*:  {lam_star:.4f}")

# Visualise
plot_ic_curve(lam_grid, IC_vec, m_breaks, lam_star, save_path="figures/ic.pdf")
plot_beta_path(beta_true, b_hat, save_path="figures/beta_path.pdf")
```

---

## Running the Simulations

```bash
python -m simulations.N25_T5_m1
```

**Configurable parameters:**

| Variable    | Default           | Description                               |
|-------------|-------------------|-------------------------------------------|
| `SIM`       | 1000              | Monte Carlo replications                  |
| `SEED`      | `None`            | RNG seed (set integer for reproducibility)|
| `PHI`       | 0.8               | AR(1) persistence of common factors       |
| `PHI_1`     | 0.4               | Temporal + spatial weight on ε            |
| `PI`        | 0.4               | Temporal + spatial weight on ν (X noise)  |
| `LAM_GRID`  | `logspace(-3,3,50)` | λ search grid                           |

---

## Data Generating Processes

| Class   | Errors                     | Regressors      | Notes                         |
|---------|----------------------------|-----------------|-------------------------------|
| `DATA1` | i.i.d. N(0,1)              | i.i.d. N(0,1)   | Baseline, no factor structure |
| `DATA2` | Factor + i.i.d.            | i.i.d. N(0,1)   | Weak cross-sectional dep.     |
| `DATA3` | AR(1) factors + spatial ε  | Factor-loaded X | Main DGP (richest structure)  |

Each class exposes four scenario methods:

| Method | Break configuration                             |
|--------|-------------------------------------------------|
| `DGP1` | 1 break at $t^* = \lfloor T/2 \rfloor$         |
| `DGP2` | 2 breaks at $\lfloor T/3 \rfloor$ and $\lfloor 2T/3 \rfloor$ |
| `DGPA` | Break every period ($\beta_t = t \cdot \mathbf{1}_p$) |
| `DGPO` | No breaks ($\beta_t$ constant)                  |

---

## Module Reference

### `src.dgp`
| Class | Arguments |
|-------|-----------|
| `DATA1` | `m, T, n, p` |
| `DATA2` | `r, m, T, n, p` |
| `DATA3` | `r, m, T, n, p, phi, phi_1, pi` |

### `src.estimator.Optimize(p, T, n)`
| Method | Returns |
|--------|---------|
| `.OLS(X, y)` | `(b_hat, status, value)` |
| `.FGLS(X, y, b_init, lam, break_tol=1e-3)` | `(b_hat, m_hat, status, value)` |
| `.NBOLS(X, y)` | `(b_cvx, b_analytic, status, value)` |

### `src.ic.information_criterion(lam_grid, y, X, p, T, n)`
Returns `(IC_vector, m_breaks, IC_min, lam_idx, lam_star, m_star)`

### `src.utils`
| Function | Description |
|----------|-------------|
| `print_mc_summary(...)` | Formatted terminal output |
| `plot_ic_curve(lam_grid, IC_vec, m_breaks, lam_star)` | IC curve figure |
| `plot_beta_path(beta_true, beta_hat)` | Coefficient path figure |

---

## Citation

```bibtex
@article{kaddoura2023estimation,
  title   = {Estimation of panel data models with random interactive effects
             and multiple structural breaks when {T} is fixed},
  author  = {Kaddoura, Yousef and Westerlund, Joakim},
  journal = {Journal of Business \& Economic Statistics},
  volume  = {41},
  pages   = {778--790},
  year    = {2023}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
