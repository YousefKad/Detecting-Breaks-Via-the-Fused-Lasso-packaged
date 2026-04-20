

import math
import numpy as np


# ---------------------------------------------------------------------------
# DATA1 — i.i.d. errors, no interactive effects
# ---------------------------------------------------------------------------

class DATA1:
    """
    Simplest DGP: independent normal regressors and errors,
    no latent factor structure.
    """

    def __init__(self, m: int, T: int, n: int, p: int) -> None:
        self.m = m
        self.T = T
        self.p = p
        self.n = n

    def _generate_xe(self):
        X   = np.random.normal(0, 1, (self.T, self.n, self.p))
        eps = np.random.normal(0, 1, (self.T, self.n))
        return X, eps

    def _build_y(self, X, beta, eps):
        y = np.zeros((self.T, self.n))
        for t in range(self.T):
            y[t] = X[t] @ beta[:, t] + eps[t]
        return y

    def DGP1(self):
        """One structural break at floor(T/2)."""
        beta    = np.zeros((self.p, self.T))
        beta_b0 = np.random.normal(0, 1, self.p)
        beta_b1 = np.random.normal(0, 1, self.p)
        split   = math.floor(self.T / 2)
        beta[:, :split] = beta_b0[:, None]
        beta[:, split:] = beta_b1[:, None]
        X, eps = self._generate_xe()
        return X, self._build_y(X, beta, eps), beta, eps

    def DGP2(self):
        """Two structural breaks at floor(T/3) and floor(2T/3)."""
        beta    = np.zeros((self.p, self.T))
        beta_b  = [np.random.normal(0, 1, self.p) for _ in range(3)]
        s1, s2  = math.floor(self.T / 3), math.floor(2 * self.T / 3)
        beta[:, :s1]    = beta_b[0][:, None]
        beta[:, s1:s2]  = beta_b[1][:, None]
        beta[:, s2:]    = beta_b[2][:, None]
        X, eps = self._generate_xe()
        return X, self._build_y(X, beta, eps), beta, eps

    def DGPA(self):
        """A break at every period (fully time-varying coefficients)."""
        beta = np.random.normal(0, 1, (self.p, self.T))
        X, eps = self._generate_xe()
        return X, self._build_y(X, beta, eps), beta, eps

    def DGPO(self):
        """No breaks — constant coefficients across all periods."""
        beta    = np.zeros((self.p, self.T))
        beta_b0 = np.random.normal(0, 1, self.p)
        beta[:] = beta_b0[:, None]
        X, eps = self._generate_xe()
        return X, self._build_y(X, beta, eps), beta, eps


# ---------------------------------------------------------------------------
# DATA2 — factor-structured errors, i.i.d. regressors
# ---------------------------------------------------------------------------

class DATA2:
    """
    Panel DGP with a latent factor structure in the errors.

    """

    def __init__(self, r: int, m: int, T: int, n: int, p: int) -> None:
        self.r = r
        self.m = m
        self.T = T
        self.n = n
        self.p = p

    def _factor_structure(self):
        lambd = np.random.normal(1, 1, (self.n, self.r))
        F     = np.random.normal(0, 1, (self.T, self.r))
        eps   = np.random.normal(0, 1, (self.T, self.n))
        u     = F @ lambd.T + eps
        return lambd, F, eps, u

    def _demean(self, X, u):
        """Cross-sectional demeaning (removes common factor mean)."""
        u_cmean = u.mean(axis=1, keepdims=True)
        u_tilde = u - u_cmean

        X_mean  = X.mean(axis=1)                         # (T, p)
        X_tilde = X - X_mean[:, None, :]                 # (T, n, p)

        return u_tilde, X_mean, X_tilde

    def _outputs(self, X, beta, u, u_tilde, X_tilde, eps, lambd, F):
        y       = np.einsum('tnp,pt->tn', X, beta) + u
        y_tilde = np.einsum('tnp,pt->tn', X_tilde, beta) + u_tilde
        X_mean  = X.mean(axis=1)
        return X, y, beta, u, eps, lambd, F, y_tilde, u_tilde, X_mean, X_tilde

    def _make_beta(self, regime_vals):
        """Assign piecewise-constant beta given list of regime values per period."""
        beta = np.zeros((self.p, self.T))
        for t, v in enumerate(regime_vals):
            beta[:, t] = v
        return beta

    def DGP1(self):
        """One structural break: β₀ = 0·1, β₁ = 1·1 (separated regimes)."""
        split   = math.floor(self.T / 2)
        vals    = [np.zeros(self.p) if t < split else np.ones(self.p)
                   for t in range(self.T)]
        beta    = self._make_beta(vals)
        lambd, F, eps, u = self._factor_structure()
        X       = np.random.normal(0, 1, (self.T, self.n, self.p))
        u_tilde, X_mean, X_tilde = self._demean(X, u)
        return self._outputs(X, beta, u, u_tilde, X_tilde, eps, lambd, F)

    def DGP2(self):
        """Two structural breaks: beta segments are 0, 1, 2 (scaled identity)."""
        s1, s2  = math.floor(self.T / 3), math.floor(2 * self.T / 3)
        def _v(t):
            if t < s1:    return np.zeros(self.p)
            elif t < s2:  return np.ones(self.p)
            else:         return 2 * np.ones(self.p)
        beta    = self._make_beta([_v(t) for t in range(self.T)])
        lambd, F, eps, u = self._factor_structure()
        X       = np.random.normal(0, 1, (self.T, self.n, self.p))
        u_tilde, X_mean, X_tilde = self._demean(X, u)
        return self._outputs(X, beta, u, u_tilde, X_tilde, eps, lambd, F)

    def DGPA(self):
        """Break at every period: β_t = t·1_p."""
        vals    = [t * np.ones(self.p) for t in range(self.T)]
        beta    = self._make_beta(vals)
        lambd, F, eps, u = self._factor_structure()
        X       = np.random.normal(0, 1, (self.T, self.n, self.p))
        u_tilde, X_mean, X_tilde = self._demean(X, u)
        return self._outputs(X, beta, u, u_tilde, X_tilde, eps, lambd, F)

    def DGPO(self):
        """No breaks: β_t = 0 for all t."""
        beta    = np.zeros((self.p, self.T))
        lambd, F, eps, u = self._factor_structure()
        X       = np.random.normal(0, 1, (self.T, self.n, self.p))
        u_tilde, X_mean, X_tilde = self._demean(X, u)
        return self._outputs(X, beta, u, u_tilde, X_tilde, eps, lambd, F)


# ---------------------------------------------------------------------------
# DATA3 — AR(1) factors + spatially-correlated idiosyncratic errors
# ---------------------------------------------------------------------------

class DATA3:

    def __init__(
        self,
        r: int,
        m: int,
        T: int,
        n: int,
        p: int,
        phi: float,
        phi_1: float,
        pi: float,
    ) -> None:
        self.r     = r
        self.m     = m
        self.T     = T
        self.n     = n
        self.p     = p
        self.phi   = phi
        self.phi_1 = phi_1
        self.pi    = pi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ar1_factors(self) -> np.ndarray:
        """Generate AR(1) common factor matrix F of shape (T, r)."""
        F   = np.zeros((self.T, self.r))
        eta = np.random.normal(0, 1, (self.T, self.r))
        F[0] = (1 - self.phi) + eta[0]          # f_0 initialised at 0
        for t in range(1, self.T):
            F[t] = (1 - self.phi) + self.phi * F[t - 1] + eta[t]
        return F

    def _spatial_ar_eps(self) -> np.ndarray:
        """
        Spatially and temporally correlated idiosyncratic error ε.

        ε_{t,i} = φ₁ ε_{t-1,i}  +  ξ_{t,i}
                  + φ₁ Σ_{j≠i, |j-i|≤10} ξ_{t,j}
        """
        sigma_i = np.random.uniform(0.5, 1, self.n)         # shape (n,)
        xi      = np.array([[np.random.normal(0, sigma_i[i])
                              for i in range(self.n)]
                             for _ in range(self.T)])      # (T, n)
        xi_rev  = xi[:, ::-1]
        eps     = np.zeros((self.T, self.n))
        for t in range(self.T):
            for i in range(self.n):
                nbr_fwd = np.sum(self.phi_1 * xi[t, i + 1:i + 11])
                nbr_bwd = np.sum(self.phi_1 * xi_rev[t, self.n - i:self.n + 10 - i])
                if t == 0:
                    eps[t, i] = xi[t, i] + nbr_fwd + nbr_bwd
                else:
                    eps[t, i] = (self.phi_1 * eps[t - 1, i]
                                 + xi[t, i] + nbr_fwd + nbr_bwd)
        return eps

    def _spatial_ar_nu(self) -> np.ndarray:
        """
        Spatially and temporally correlated regressor noise ν of shape (T, n, p).
        """
        e     = np.random.normal(0, 1, (self.T, self.n, self.p))
        e_rev = e[:, ::-1, :]
        nu    = np.zeros((self.T, self.n, self.p))
        for i in range(self.n):
            for t in range(self.T):
                for k in range(self.p):
                    nbr_fwd = np.sum(self.pi * e[t, i + 1:i + 11, k])
                    nbr_bwd = np.sum(self.pi * e_rev[t, self.n - i:self.n + 10 - i, k])
                    if t == 0:
                        nu[t, i, k] = e[t, i, k] + nbr_fwd + nbr_bwd
                    else:
                        nu[t, i, k] = (self.pi * nu[t - 1, i, k]
                                       + e[t, i, k] + nbr_fwd + nbr_bwd)
        return nu

    def _build_X(self, F: np.ndarray, nu: np.ndarray) -> np.ndarray:
        """X_{t,i,k} = F_t' λ_{ik} + ν_{t,i,k}."""
        lambd_x = np.random.normal(2, 1, (self.n, self.p, self.r))
        X       = np.zeros((self.T, self.n, self.p))
        for t in range(self.T):
            for i in range(self.n):
                for k in range(self.p):
                    X[t, i, k] = F[t] @ lambd_x[i, k] + nu[t, i, k]
        return X

    def _demean_and_outputs(self, X, beta, u, eps, F):
        u_cmean = u.mean(axis=1, keepdims=True)
        u_tilde = u - u_cmean
        X_mean  = X.mean(axis=1)
        X_tilde = X - X_mean[:, None, :]
        y       = np.einsum('tnp,pt->tn', X, beta) + u
        y_tilde = np.einsum('tnp,pt->tn', X_tilde, beta) + u_tilde
        return X, y, beta, u, eps, F, y_tilde, u_tilde, X_mean, X_tilde

    def _make_beta(self, segment_fn) -> np.ndarray:
        beta = np.zeros((self.p, self.T))
        for t in range(self.T):
            beta[:, t] = segment_fn(t)
        return beta


    def DGP1(self):
        """
        One structural break at t* = floor(T/2).
        """
        split = math.floor(self.T / 2)
        beta  = self._make_beta(
            lambda t: np.zeros(self.p) if t < split else np.ones(self.p)
        )
        F        = self._ar1_factors()
        lambd_y  = np.random.normal(2, 1, (self.n, self.r))
        eps      = self._spatial_ar_eps()
        u        = F @ lambd_y.T + eps
        nu       = self._spatial_ar_nu()
        X        = self._build_X(F, nu)
        return self._demean_and_outputs(X, beta, u, eps, F)

    def DGP2(self):
        """
        Two structural breaks at t* = floor(T/3) and floor(2T/3).

        Regime 0: beta = 0·1_p,  Regime 1: beta = 1·1_p,  Regime 2: beta = 2·1_p
        """
        s1, s2 = math.floor(self.T / 3), math.floor(2 * self.T / 3)

        def _seg(t):
            if t < s1:    return np.zeros(self.p)
            elif t < s2:  return np.ones(self.p)
            else:         return 2 * np.ones(self.p)

        beta     = self._make_beta(_seg)
        F        = self._ar1_factors()
        lambd_y  = np.random.normal(2, 1, (self.n, self.r))
        eps      = self._spatial_ar_eps()
        u        = F @ lambd_y.T + eps
        nu       = self._spatial_ar_nu()
        X        = self._build_X(F, nu)
        return self._demean_and_outputs(X, beta, u, eps, F)

    def DGPA(self):
        """
        Break at every period: beta_t = t · 1_p (maximal instability).
        """
        beta     = self._make_beta(lambda t: t * np.ones(self.p))
        F        = self._ar1_factors()
        lambd_y  = np.random.normal(2, 1, (self.n, self.r))
        eps      = self._spatial_ar_eps()
        u        = F @ lambd_y.T + eps
        nu       = self._spatial_ar_nu()
        X        = self._build_X(F, nu)
        return self._demean_and_outputs(X, beta, u, eps, F)

    def DGPO(self):
        """
        No breaks: beta_t = 0 for all t (fully stable coefficients).
        """
        beta     = np.zeros((self.p, self.T))
        F        = self._ar1_factors()
        lambd_y  = np.random.normal(2, 1, (self.n, self.r))
        eps      = self._spatial_ar_eps()
        u        = F @ lambd_y.T + eps
        nu       = self._spatial_ar_nu()
        X        = self._build_X(F, nu)
        return self._demean_and_outputs(X, beta, u, eps, F)
