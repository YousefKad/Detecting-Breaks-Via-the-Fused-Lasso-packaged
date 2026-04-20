"""
ic.py — Information Criterion for Tuning Parameter Selection
=============================================================
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
    the optimallambda!!
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
