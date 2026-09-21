"""Binary and one-vs-rest logistic regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))


class LogisticRegression(nn.Module):
    """Binary logistic regression with manually updated weights."""

    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, num_features)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid(x @ self.weight.t())


def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    prediction = torch.clamp(y_pred, 1e-12, 1 - 1e-12)
    return torch.mean(
        -y_true * torch.log(prediction)
        - (1 - y_true) * torch.log(1 - prediction)
    )


def logistic_gradient_step(
    model: LogisticRegression,
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    learning_rate: float,
) -> None:
    gradient = ((y_pred - y).t() @ x) / x.shape[0]
    model.weight = nn.Parameter(
        model.weight - learning_rate * gradient,
        requires_grad=False,
    )


@dataclass(frozen=True)
class LogisticFit:
    model: LogisticRegression
    losses: list[float]


def fit_binary_logistic(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    learning_rate: float = 0.1,
    iterations: int = 1000,
) -> LogisticFit:
    model = LogisticRegression(x.shape[1])
    losses: list[float] = []

    for _ in range(iterations):
        prediction = model(x)
        loss = binary_cross_entropy(y, prediction)
        losses.append(float(loss.item()))
        logistic_gradient_step(model, x, y, prediction, learning_rate)

    return LogisticFit(model=model, losses=losses)


class OneVsRestLogistic(nn.Module):
    """Three independent sigmoid classifiers combined as in the Iris experiment."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.models = nn.ModuleList(
            [nn.Sequential(nn.Linear(num_features, 1, bias=False), nn.Sigmoid()) for _ in range(num_classes)]
        )

    def fit(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        learning_rate: float = 0.1,
        iterations: int = 1000,
    ) -> "OneVsRestLogistic":
        one_hot = F.one_hot(labels.long(), num_classes=len(self.models)).float()
        for class_index, model in enumerate(self.models):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            target = one_hot[:, class_index].reshape(-1, 1)
            for _ in range(iterations):
                optimizer.zero_grad()
                prediction = model(x)
                loss = F.binary_cross_entropy(prediction, target)
                loss.backward()
                optimizer.step()
        return self

    def independent_scores(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([model(x) for model in self.models], dim=1)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.independent_scores(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.probabilities(x), dim=1)
