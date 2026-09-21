"""Supervised learning utilities."""

from .preprocessing import add_bias, standardize
from .regression import LinearRegression, fit_linear_regression, polynomial_features
from .classification import LogisticRegression, fit_binary_logistic
from .neural_networks import ManualXORNetwork, MLP

__all__ = [
    "add_bias",
    "standardize",
    "LinearRegression",
    "fit_linear_regression",
    "polynomial_features",
    "LogisticRegression",
    "fit_binary_logistic",
    "ManualXORNetwork",
    "MLP",
]
