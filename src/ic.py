"""
ic.py — Information Criterion for Tuning Parameter Selection
=============================================================
Implements the penalised IC used to select the regularisation
parameter λ in the adaptive fused lasso estimator.

The IC balances in-sample fit against model complexity (measured by
the estimated number of breaks m̂):

    IC(λ) = (1/(nT)) Σ_t ‖y_t − X_t β̂_t(λ)‖²
             + (log n / √n) · p · (m̂(λ) + 1)

The optimal λ* minimises IC(λ) over a log-spaced grid.

Reference
---------
Kaddoura, Y. and Westerlund, J. (2023), Section 3.2.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .estimator import Optimize


def information_criterion(
    lam_grid: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    p: int,
    T: int,
    n: int,
    break_tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, float, int, float, float]:
    """
    Compute the IC over a grid of regularisation parameters and return
    the optimal λ*.

    Parameters
    ----------
    lam_grid  : 1-D array   — candidate values of λ (e.g. np.logspace(-3, 3, 50))
    y         : array (T, n)
    X         : array (T, n, p)
    p         : int — number of regressors
    T         : int — time dimension
    n         : int — cross-sectional dimension
    break_tol : float — norm threshold for counting a break

    Returns
    -------
    IC_vector    : array (L,)  — IC value for each λ in the grid
    m_breaks     : array (L,)  — estimated break count for each λ
    IC_min       : float       — minimum IC value
    lam_idx      : int         — index of optimal λ in lam_grid
    lam_star     : float       — optimal regularisation parameter
    m_star       : float       — estimated number of breaks at λ*
    """
    L         = len(lam_grid)
    IC_vector = np.full(L, np.inf)
    m_breaks  = np.zeros(L)
    opt       = Optimize(p, T, n)

    # First-stage OLS (used as adaptive weights)
    b_ols, _, _ = opt.OLS(X, y)

    for l, lam in enumerate(lam_grid):
        try:
            b_hat, m_hat, status, _ = opt.FGLS(X, y, b_ols, lam, break_tol)
            if b_hat is None or status not in ("optimal", "optimal_inaccurate"):
                raise ValueError(f"Solver returned status: {status}")

            # Fit term
            fit = sum(
                np.linalg.norm(y[t] - X[t] @ b_hat[:, t], 2) ** 2
                for t in range(T)
            ) / (n * T)

            # Complexity penalty
            penalty = (np.log(n) / np.sqrt(n)) * p * (m_hat + 1)

            IC_vector[l] = fit + penalty
            m_breaks[l]  = m_hat

        except Exception:
            # Carry forward the previous value on solver failure
            if l > 0:
                IC_vector[l] = IC_vector[l - 1]
                m_breaks[l]  = m_breaks[l - 1]

    lam_idx  = int(np.argmin(IC_vector))
    IC_min   = float(IC_vector[lam_idx])
    lam_star = float(lam_grid[lam_idx])
    m_star   = float(m_breaks[lam_idx])

    return IC_vector, m_breaks, IC_min, lam_idx, lam_star, m_star
