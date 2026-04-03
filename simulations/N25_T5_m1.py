"""
simulations/N25_T5_m1.py
=========================
Monte Carlo replication study:  N = 25,  T = 5,  m = 1 (one structural break)

This script reproduces the simulation results reported in Table [X] of
Kaddoura & Westerlund (2023) for the case of a single structural break
under the DATA3 DGP (AR(1) common factors + spatially correlated errors).

DGP Specification
-----------------
- n  = 25   (cross-sectional units)
- T  = 5    (time periods, fixed)
- p  = 4    (regressors)
- r  = 5    (latent factors)
- m  = 1    (one structural break at t* = floor(T/2) = 2)
- φ  = 0.8  (AR(1) persistence of common factors)
- φ₁ = 0.4  (temporal + spatial weight, idiosyncratic errors)
- π  = 0.4  (temporal + spatial weight, regressor noise)

The break occurs at t* = floor(5/2) = 2:
    β_t = 0·1_p  for t ∈ {1, 2}
    β_t = 1·1_p  for t ∈ {3, 4, 5}

Results Reported
----------------
1. Average estimated break count  (m̂_avg)
2. Frequency of wrong break count  (m_freq / sim)
3. Frequency of wrong break date   (t_freq / sim)
4. Mean Frobenius Error (MFE) of β̂ relative to β_true

Usage
-----
    python -m simulations.N25_T5_m1          (from repo root)
    python simulations/N25_T5_m1.py
"""

import sys
import os
import math
import numpy as np

# Allow running from both the repo root and the simulations/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import DATA3
from src.estimator import Optimize
from src.ic import information_criterion
from src.utils import print_mc_summary, plot_ic_curve, plot_beta_path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dimensions
p = 4
n = 25
T = 5
m = 1
r = 5

# DGP correlation parameters (DATA3)
PHI   = 0.8   # AR(1) persistence of common factors
PHI_1 = 0.4   # temporal + spatial weight for idiosyncratic errors
PI    = 0.4   # temporal + spatial weight for regressor noise

# Optimisation
LAM_GRID     = np.logspace(-3, 3, 50)   # λ search grid
BREAK_TOL    = 1e-2                      # ‖Δβ̂‖ threshold for declaring a break

# Monte Carlo
SIM = 1000
SEED = None   # set an integer for full reproducibility, e.g. SEED = 42


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------

def run_simulation(
    sim: int = SIM,
    seed: int | None = SEED,
    verbose: bool = True,
) -> dict:
    """
    Run the Monte Carlo study and return a dictionary of summary statistics.

    Parameters
    ----------
    sim     : int         — number of Monte Carlo replications
    seed    : int or None — RNG seed (None = non-reproducible)
    verbose : bool        — print progress every 100 iterations

    Returns
    -------
    dict with keys: beta_true, beta_avg, m_sim, m_freq, t_freq, mfe,
                    failed, IC_vector_last, m_breaks_last,
                    lam_star_last, b_lstar_last
    """
    if seed is not None:
        np.random.seed(seed)

    opt         = Optimize(p, T, n)
    b_lsim      = np.zeros((p, T))
    m_sim       = 0.0
    m_freq      = 0     # wrong break count
    t_freq      = 0     # right count but wrong location
    failed      = 0
    beta_last   = None

    # Store last-rep diagnostics for optional plotting
    IC_last = m_brk_last = lam_last = b_last = None

    for k in range(sim):
        try:
            data = DATA3(r, m, T, n, p, PHI, PHI_1, PI)
            X, y, beta, u, eps, F, y_tilde, u_tilde, X_mean, X_tilde = data.DGP1()
            beta_last = beta

            IC_vec, m_brk, _, lam_idx, lam_star, m_star = information_criterion(
                LAM_GRID, y_tilde, X_tilde, p, T, n, BREAK_TOL
            )
            b_w, _, _          = opt.OLS(X_tilde, y_tilde)
            b_lstar, m_lstar, _, _ = opt.FGLS(X_tilde, y_tilde, b_w, lam_star, BREAK_TOL)

            m_lstar = int(round(m_lstar))

            # ---- Accumulate ----
            b_lsim += b_lstar
            m_sim  += m_lstar

            # Wrong break count?
            if m_lstar != m:
                m_freq += 1
            else:
                # Correct count — check date
                t_star = math.floor(T / (m + 1))
                if np.linalg.norm(b_lstar[:, t_star] - b_lstar[:, t_star - 1]) < BREAK_TOL:
                    t_freq += 1

            # Save last rep for plots
            IC_last   = IC_vec
            m_brk_last = m_brk
            lam_last  = lam_star
            b_last    = b_lstar

            if verbose and (k + 1) % 100 == 0:
                print(f"  Completed {k + 1:5d}/{sim} replications "
                      f"| m̂ this rep = {m_lstar} | failed so far = {failed}")

        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  [WARNING] Rep {k+1} failed: {exc}")

    # Normalise
    n_valid = sim - failed
    b_avg   = b_lsim / max(n_valid, 1)
    m_avg   = m_sim  / max(n_valid, 1)
    mfe     = (1 / (p * T)) * np.linalg.norm(beta_last - b_avg, "fro")

    return {
        "beta_true":       beta_last,
        "beta_avg":        b_avg,
        "m_sim":           m_avg,
        "m_freq":          m_freq,
        "t_freq":          t_freq,
        "mfe":             mfe,
        "failed":          failed,
        "IC_vector_last":  IC_last,
        "m_breaks_last":   m_brk_last,
        "lam_star_last":   lam_last,
        "b_lstar_last":    b_last,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Fused Lasso Break Detection — Monte Carlo")
    print(f"  N={n}, T={T}, p={p}, r={r}, m={m}, sim={SIM}")
    print("=" * 60 + "\n")

    results = run_simulation(sim=SIM, seed=SEED, verbose=True)

    print_mc_summary(
        beta_true = results["beta_true"],
        beta_avg  = results["beta_avg"],
        m_true    = m,
        m_avg     = results["m_sim"],
        m_freq    = results["m_freq"],
        t_freq    = results["t_freq"],
        mfe       = results["mfe"],
        n=n, T=T, p=p, r=r,
        sim=SIM,
        failed=results["failed"],
    )

    # Optional: save figures
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if results["IC_vector_last"] is not None:
        plot_ic_curve(
            LAM_GRID,
            results["IC_vector_last"],
            results["m_breaks_last"],
            results["lam_star_last"],
            save_path=os.path.join(fig_dir, "IC_N25_T5_m1.pdf"),
        )

    if results["b_lstar_last"] is not None and results["beta_true"] is not None:
        plot_beta_path(
            results["beta_true"],
            results["b_lstar_last"],
            save_path=os.path.join(fig_dir, "beta_path_N25_T5_m1.pdf"),
        )
