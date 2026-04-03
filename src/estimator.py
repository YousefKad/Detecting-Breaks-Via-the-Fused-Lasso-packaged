"""
estimator.py — Fused Lasso Estimator and Benchmarks
=====================================================
Implements the adaptive fused lasso (FGLS) estimator and auxiliary
least-squares methods described in Kaddoura & Westerlund (2023).

Classes
-------
Optimize
    Container for all estimation routines.

Methods
-------
OLS   : Time-varying OLS (period-by-period least squares).
FGLS  : Adaptive fused lasso (main estimator). Minimises
            L(B) = Σ_t (1/n) ‖y_t − X_t β_t‖²
                   + λ Σ_{t≥2} w_t ‖β_t − β_{t−1}‖₂
        where w_t = ‖β^OLS_t − β^OLS_{t-1}‖₂⁻² (adaptive weights).
NBOLS : No-break OLS (pooled estimator, benchmark under H₀).
"""

import numpy as np
import cvxpy as cp


class Optimize:
    """
    Estimation engine for the fused lasso break detection method.

    Parameters
    ----------
    p : int — number of regressors
    T : int — time dimension
    n : int — cross-sectional dimension
    """

    def __init__(self, p: int, T: int, n: int) -> None:
        self.p = p
        self.T = T
        self.n = n

    # ------------------------------------------------------------------
    # OLS — time-varying, period-by-period
    # ------------------------------------------------------------------

    def OLS(self, X: np.ndarray, y: np.ndarray):
        """
        Period-by-period OLS estimator. Minimises

            Σ_t (1/n) ‖y_t − X_t β_t‖²

        separately for each t (no penalisation, no pooling).

        Parameters
        ----------
        X : array (T, n, p)
        y : array (T, n)

        Returns
        -------
        b_ols  : array (p, T)  — estimated coefficients
        status : str           — solver status
        value  : float         — optimal objective value
        """
        b     = cp.Variable((self.p, self.T))
        obj   = sum(
            (1 / self.n) * cp.sum_squares(y[t] - X[t] @ b[:, t])
            for t in range(self.T)
        )
        prob  = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.CLARABEL, verbose=False)
        return b.value, prob.status, prob.value

    # ------------------------------------------------------------------
    # FGLS — Adaptive Fused Lasso (main estimator)
    # ------------------------------------------------------------------

    def FGLS(
        self,
        X: np.ndarray,
        y: np.ndarray,
        b_init: np.ndarray,
        lam: float,
        break_tol: float = 1e-3,
    ):
        """
        Adaptive fused lasso estimator. Solves

            min_{B}  Σ_t (1/n)‖y_t − X_t β_t‖²
                     + λ Σ_{t≥2} w_t ‖β_t − β_{t−1}‖₂

        with adaptive weights  w_t = ‖b_init_t − b_init_{t-1}‖₂⁻².

        Parameters
        ----------
        X       : array (T, n, p)
        y       : array (T, n)
        b_init  : array (p, T)  — first-stage OLS (for adaptive weights)
        lam     : float         — regularisation parameter λ
        break_tol : float       — threshold below which Δβ counts as no-break

        Returns
        -------
        b_hat   : array (p, T)  — estimated time-varying coefficients
        m_hat   : int           — estimated number of breaks
        status  : str           — solver status
        value   : float         — optimal objective value
        """
        b   = cp.Variable((self.p, self.T))
        obj = (1 / self.n) * cp.sum_squares(y[0] - X[0] @ b[:, 0])

        for t in range(1, self.T):
            diff_init = np.linalg.norm(b_init[:, t] - b_init[:, t - 1], 2)
            # Avoid division by zero
            w_t = diff_init ** (-2) if diff_init > 1e-12 else 1e12
            obj += (1 / self.n) * cp.sum_squares(y[t] - X[t] @ b[:, t])
            obj += lam * w_t * cp.norm(b[:, t] - b[:, t - 1], 2)

        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.CLARABEL, verbose=False)

        # Count detected breaks
        m_hat = 0
        if b.value is not None:
            for t in range(1, self.T):
                if np.linalg.norm(b.value[:, t] - b.value[:, t - 1], 2) > break_tol:
                    m_hat += 1

        return b.value, m_hat, prob.status, prob.value

    # ------------------------------------------------------------------
    # NBOLS — No-break pooled OLS (benchmark)
    # ------------------------------------------------------------------

    def NBOLS(self, X: np.ndarray, y: np.ndarray):
        """
        Pooled (no-break) OLS estimator. Assumes β is constant across t.
        Returns both the CVXPY solution and the closed-form analytic answer.

        Parameters
        ----------
        X : array (T, n, p)
        y : array (T, n)

        Returns
        -------
        b_cvx  : array (p,)    — CVXPY solution
        b_ana  : array (p, 1)  — analytic solution via normal equations
        status : str           — solver status
        value  : float         — optimal objective value
        """
        X_pool = X.reshape(self.T * self.n, self.p)
        y_pool = y.reshape(self.T * self.n, 1)

        b      = cp.Variable(self.p)
        obj    = cp.sum_squares(y_pool - X_pool @ b)
        prob   = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.CLARABEL, verbose=False)

        b_ana  = np.linalg.inv(X_pool.T @ X_pool) @ X_pool.T @ y_pool
        return b.value, b_ana, prob.status, prob.value
