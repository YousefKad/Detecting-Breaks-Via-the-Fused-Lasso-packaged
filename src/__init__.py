"""
Fused Lasso Structural Break Detection
=======================================
A Python implementation of the adaptive fused lasso estimator for detecting
multiple structural breaks in panel data models with interactive fixed effects.

Reference:
    Kaddoura, Y. and Westerlund, J. (2023). Estimation of panel data models
    with random interactive effects and multiple structural breaks when T is
    fixed. J. Bus. Econom. Statist., 41, pp. 778–790.
"""

from .dgp import DATA1, DATA2, DATA3
from .estimator import Optimize
from .ic import information_criterion

__all__ = ["DATA1", "DATA2", "DATA3", "Optimize", "information_criterion"]
__version__ = "1.0.0"
__author__ = "Yousef Kaddoura"
