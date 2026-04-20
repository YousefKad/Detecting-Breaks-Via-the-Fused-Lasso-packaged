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
FGLS  : Adaptive fused lasso (main estimator).
NBOLS : No-break OLS (pooled estimator, benchmark under H₀).
"""
import numpy as np
import cvxpy as cp


class Optimize:
    def __init__(self, p: int, T: int, n: int) -> None:
        self.p = p
        self.T = T
        self.n = n

    # ------------------------------------------------------------------
    # OLS — time-varying, period-by-period
    # ------------------------------------------------------------------

    def OLS(self, X: np.ndarray, y: np.ndarray):
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
        X_pool = X.reshape(self.T * self.n, self.p)
        y_pool = y.reshape(self.T * self.n, 1)

        b      = cp.Variable(self.p)
        obj    = cp.sum_squares(y_pool - X_pool @ b)
        prob   = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.CLARABEL, verbose=False)

        b_ana  = np.linalg.inv(X_pool.T @ X_pool) @ X_pool.T @ y_pool
        return b.value, b_ana, prob.status, prob.value
