"""
tests/test_core.py
==================
Unit tests for DGP, Estimator, and IC modules.
Run with:  pytest tests/
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import DATA1, DATA2, DATA3
from src.estimator import Optimize
from src.ic import information_criterion


# ---------------------------------------------------------------------------
# DGP tests
# ---------------------------------------------------------------------------

class TestDATA1:
    p, n, T, m = 3, 10, 4, 1

    def test_dgp1_shapes(self):
        d = DATA1(self.m, self.T, self.n, self.p)
        X, y, beta, eps = d.DGP1()
        assert X.shape    == (self.T, self.n, self.p)
        assert y.shape    == (self.T, self.n)
        assert beta.shape == (self.p, self.T)
        assert eps.shape  == (self.T, self.n)

    def test_dgp1_one_break(self):
        d = DATA1(self.m, self.T, self.n, self.p)
        _, _, beta, _ = d.DGP1()
        split = math.floor(self.T / 2)
        # All pre-break columns identical
        for t in range(1, split):
            np.testing.assert_array_equal(beta[:, t], beta[:, 0])
        # Pre- and post-break differ (with overwhelming probability)
        assert not np.allclose(beta[:, 0], beta[:, split])

    def test_dgpo_constant(self):
        d = DATA1(self.m, self.T, self.n, self.p)
        _, _, beta, _ = d.DGPO()
        for t in range(1, self.T):
            np.testing.assert_array_equal(beta[:, t], beta[:, 0])


class TestDATA3:
    p, n, T, m, r = 2, 8, 4, 1, 2

    def _make(self):
        return DATA3(self.r, self.m, self.T, self.n, self.p, 0.8, 0.4, 0.4)

    def test_dgp1_shapes(self):
        d = self._make()
        X, y, beta, u, eps, F, y_t, u_t, X_m, X_tl = d.DGP1()
        assert X.shape    == (self.T, self.n, self.p)
        assert y.shape    == (self.T, self.n)
        assert beta.shape == (self.p, self.T)
        assert F.shape    == (self.T, self.r)

    def test_demeaned_mean_zero(self):
        """Cross-sectional mean of X_tilde should be ≈ 0."""
        d = self._make()
        _, _, _, _, _, _, _, _, _, X_tilde = d.DGP1()
        means = X_tilde.mean(axis=1)  # (T, p)
        np.testing.assert_allclose(means, 0, atol=1e-12)


# ---------------------------------------------------------------------------
# Estimator tests
# ---------------------------------------------------------------------------

class TestOptimize:
    p, n, T = 2, 15, 4

    def _data(self):
        np.random.seed(99)
        X = np.random.normal(0, 1, (self.T, self.n, self.p))
        b = np.ones((self.p, self.T))
        y = np.einsum("tnp,pt->tn", X, b) + np.random.normal(0, 0.1, (self.T, self.n))
        return X, y, b

    def test_ols_shape(self):
        X, y, _ = self._data()
        opt = Optimize(self.p, self.T, self.n)
        b_hat, status, _ = opt.OLS(X, y)
        assert b_hat.shape == (self.p, self.T)
        assert status in ("optimal", "optimal_inaccurate")

    def test_ols_close_to_true(self):
        X, y, b_true = self._data()
        opt = Optimize(self.p, self.T, self.n)
        b_hat, _, _ = opt.OLS(X, y)
        np.testing.assert_allclose(b_hat, b_true, atol=0.15)

    def test_fgls_zero_lambda_close_to_ols(self):
        """At λ ≈ 0, FGLS should recover (approximately) the OLS solution."""
        X, y, _ = self._data()
        opt = Optimize(self.p, self.T, self.n)
        b_ols, _, _ = opt.OLS(X, y)
        b_fgls, _, _, _ = opt.FGLS(X, y, b_ols, lam=1e-6)
        np.testing.assert_allclose(b_fgls, b_ols, atol=0.05)


# ---------------------------------------------------------------------------
# IC tests
# ---------------------------------------------------------------------------

class TestIC:
    p, n, T = 2, 12, 4

    def test_ic_returns_valid_lambda(self):
        np.random.seed(7)
        X = np.random.normal(0, 1, (self.T, self.n, self.p))
        b = np.array([[0.] * 2 + [1.] * 2, [0.] * 2 + [1.] * 2])
        y = np.einsum("tnp,pt->tn", X, b) + np.random.normal(0, 0.1, (self.T, self.n))
        grid = np.logspace(-2, 2, 10)
        IC_vec, m_brk, IC_min, idx, lam_star, m_star = information_criterion(
            grid, y, X, self.p, self.T, self.n
        )
        assert lam_star in grid
        assert IC_vec.shape == (len(grid),)
        assert IC_min == IC_vec[idx]
