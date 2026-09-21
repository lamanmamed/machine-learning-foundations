"""Linear regression and polynomial ridge regression with manual gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LinearRegression(nn.Module):
    """Linear model matching the hand-written layer used in the project."""

    def __init__(self, num_features: int, initial_value: float = 1.0):
        super().__init__()
        weights = torch.full((1, num_features), float(initial_value))
        self.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


def mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def gradient_descent_step(
    model: LinearRegression,
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    learning_rate: float,
) -> None:
    """Apply the MSE gradient used in the original linear-regression exercise."""
    gradient = (2.0 / x.shape[0]) * (x.t() @ (y_pred - y))
    model.weight = nn.Parameter(
        model.weight - learning_rate * gradient.t(),
        requires_grad=False,
    )


@dataclass(frozen=True)
class RegressionFit:
    model: LinearRegression
    losses: list[float]


def fit_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    learning_rate: float = 0.1,
    iterations: int = 100,
) -> RegressionFit:
    model = LinearRegression(x.shape[1])
    losses: list[float] = []

    for _ in range(iterations):
        prediction = model(x)
        loss = mean_squared_error(prediction, y)
        if not torch.isfinite(loss):
            losses.append(float("inf"))
            break
        losses.append(float(loss.item()))
        gradient_descent_step(model, x, y, prediction, learning_rate)

    return RegressionFit(model=model, losses=losses)


def polynomial_features(x: torch.Tensor, degree: int = 5) -> torch.Tensor:
    """Return [1, x, x^2, ..., x^degree] for a one-dimensional input."""
    x = x.reshape(-1, 1)
    return torch.cat([x ** power for power in range(degree + 1)], dim=1)


def regularized_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    """Regularized objective used for the polynomial experiment.

    The bias is the first weight and is not regularized.
    """
    n = y_true.shape[0]
    squared_error = torch.sum((y_pred - y_true) ** 2)
    penalty = regularization * torch.sum(weights[0, 1:] ** 2)
    return (squared_error + penalty) / (2 * n)


def regularized_gradient_step(
    model: LinearRegression,
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    learning_rate: float,
    regularization: float,
) -> None:
    n = x.shape[0]
    base_gradient = ((x.t() @ (y_pred - y)) / n).t()

    bias = model.weight[0, 0] - learning_rate * base_gradient[0, 0]
    decay = 1.0 - learning_rate * regularization / n
    other = model.weight[0, 1:] * decay - learning_rate * base_gradient[0, 1:]
    model.weight = nn.Parameter(
        torch.cat([bias.reshape(1), other]).reshape(1, -1),
        requires_grad=False,
    )


def fit_polynomial_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    degree: int = 5,
    learning_rate: float = 0.1,
    regularization: float = 1.0,
    iterations: int = 1000,
) -> RegressionFit:
    features = polynomial_features(x, degree)
    model = LinearRegression(features.shape[1])
    losses: list[float] = []

    for _ in range(iterations):
        prediction = model(features)
        loss = regularized_mse(prediction, y, model.weight, regularization)
        if not torch.isfinite(loss):
            losses.append(float("inf"))
            break
        losses.append(float(loss.item()))
        regularized_gradient_step(
            model,
            features,
            y,
            prediction,
            learning_rate=learning_rate,
            regularization=regularization,
        )

    return RegressionFit(model=model, losses=losses)
