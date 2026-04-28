# Detecting Structural Breaks via the Adaptive Fused Lasso

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
</p>

> **Python implementation** of the adaptive fused lasso estimator for detecting multiple structural breaks in panel data models with interactive fixed effects, based on:
>
> **Kaddoura, Y. and Westerlund, J. (2023)**. *Estimation of panel data models with random interactive effects and multiple structural breaks when T is fixed.* Journal of Business & Economic Statistics, 41, 778–790.

---

## Overview

Standard panel data estimators assume slope coefficients are constant over time. In practice, economic relationships frequently undergo **structural breaks**  or abrupt shifts in regression coefficients caused by policy changes, financial crises, or regime transitions.

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



## Installation

### Option 1 — Conda

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

## Here is an example

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

## Extra information

The code is self contained so most information on the package can be found through the code.
