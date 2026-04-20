"""
utils.py — Reporting and Plotting Utilities
============================================
Helper functions for summarising Monte Carlo results,
printing formatted tables, and generating publication-quality figures.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Break location check
# ---------------------------------------------------------------------------

def break_correctly_located(
    b_hat: np.ndarray,
    m_hat: int,
    m_true: int,
    T: int,
    p: int,
    break_tol: float = 1e-2,
) -> bool:

    if m_true == 0:
        return True      # no breaks to locate
    if m_hat != m_true:
        return False

    located = 0
    if m_true >= 1:
        t_star = math.floor(T / (m_true + 1))
        if np.linalg.norm(b_hat[:p, t_star] - b_hat[:p, t_star - 1]) >= break_tol:
            located += 1
    if m_true >= 2:
        t_star2 = math.floor(2 * T / (m_true + 1))
        if np.linalg.norm(b_hat[:p, t_star2] - b_hat[:p, t_star2 - 1]) >= break_tol:
            located += 1

    return located == m_true


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_mc_summary(
    beta_true: np.ndarray,
    beta_avg: np.ndarray,
    m_true: int,
    m_avg: float,
    m_freq: int,
    t_freq: int,
    mfe: float,
    n: int,
    T: int,
    p: int,
    r: int,
    sim: int,
    failed: int,
) -> None:
    """Pretty-print a Monte Carlo summary table."""
    divider = "=" * 60

    print(divider)
    print("  MONTE CARLO SUMMARY — Fused Lasso Break Detection")
    print(divider)
    print(f"  Dimensions : n={n}, T={T}, p={p}, r={r}")
    print(f"  Replications: {sim}  (failed: {failed})")
    print(divider)

    print("\n  True β matrix (p × T):")
    print(tabulate(beta_true, tablefmt="simple", floatfmt=".4f"))

    print("\n  Average estimated β̂ matrix (p × T):")
    print(tabulate(beta_avg, tablefmt="simple", floatfmt=".4f"))

    print(f"\n  True breaks     m   = {m_true}")
    print(f"  Average m̂          = {m_avg:.4f}")
    print(f"  # wrong break count = {m_freq}  "
          f"({100 * m_freq / max(sim - failed, 1):.1f}%)")
    print(f"  # wrong break date  = {t_freq}  "
          f"({100 * t_freq / max(sim - failed, 1):.1f}%)")
    print(f"  Mean Frobenius Error (MFE) = {mfe:.6f}")
    print(divider)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ic_curve(
    lam_grid: np.ndarray,
    IC_vector: np.ndarray,
    m_breaks: np.ndarray,
    lam_star: float,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color_ic = "#1f77b4"
    color_mb = "#2ca02c"

    ax1.plot(lam_grid, IC_vector, color=color_ic, marker=".", linewidth=1.5,
             markersize=4, label="IC(λ)")
    ax1.axvline(lam_star, color=color_ic, linestyle="--", alpha=0.6,
                label=f"λ* = {lam_star:.4f}")
    ax1.set_xlabel("Tuning parameter λ", fontsize=12)
    ax1.set_ylabel("IC value", color=color_ic, fontsize=12)
    ax1.tick_params(axis="y", colors=color_ic)
    ax1.set_xscale("log")
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_color(color_ic)

    ax2 = ax1.twinx()
    ax2.plot(lam_grid, m_breaks, color=color_mb, marker=".", linewidth=1.5,
             markersize=4, label="m̂(λ)")
    ax2.set_ylabel("Estimated breaks m̂", color=color_mb, fontsize=12)
    ax2.tick_params(axis="y", colors=color_mb)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(color_mb)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9, framealpha=0.7)

    fig.suptitle("IC and Estimated Breaks vs. Regularisation Parameter",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figure saved → {save_path}")

    return fig


def plot_beta_path(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot true vs. estimated coefficient paths over time.

    Parameters
    ----------
    beta_true : array (p, T) — true coefficients
    beta_hat  : array (p, T) — estimated coefficients
    save_path : str or None
    """
    p, T = beta_true.shape
    fig, axes = plt.subplots(p, 1, figsize=(8, 2 * p), sharex=True)
    if p == 1:
        axes = [axes]

    t_idx = np.arange(1, T + 1)
    for k, ax in enumerate(axes):
        ax.step(t_idx, beta_true[k], where="post", color="black",
                linewidth=2, label="True β")
        ax.step(t_idx, beta_hat[k], where="post", color="#d62728",
                linewidth=1.5, linestyle="--", label="Estimated β̂")
        ax.set_ylabel(f"β_{k+1}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Period t", fontsize=11)
    fig.suptitle("True vs. Estimated Coefficient Paths", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Figure saved → {save_path}")

    return fig
