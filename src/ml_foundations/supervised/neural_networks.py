"""Manual XOR network and a small PyTorch MLP for Iris."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class _SigmoidNeuron(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((1, num_features)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return torch.sigmoid(x @ self.weight.t()).reshape(-1)


class ManualXORNetwork(nn.Module):
    """Two-hidden-neuron feed-forward network with manual backpropagation."""

    def __init__(self, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = nn.ModuleList([_SigmoidNeuron(3) for _ in range(2)])
        self.output = _SigmoidNeuron(3)
        self._hidden_state: torch.Tensor | None = None
        self._output_state: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1 or x.shape[0] != 3:
            raise ValueError("XOR input must contain x1, x2, and a bias feature.")
        hidden = torch.ones(3)
        hidden[0] = self.hidden[0](x)[0]
        hidden[1] = self.hidden[1](x)[0]
        output = self.output(hidden)[0]
        self._hidden_state = hidden
        self._output_state = output.reshape(1)
        return self._output_state

    def backward_pass(self, x: torch.Tensor, target: torch.Tensor, learning_rate: float) -> None:
        if self._hidden_state is None or self._output_state is None:
            raise RuntimeError("Call forward before backward_pass.")

        hidden = self._hidden_state
        output = self._output_state[0]
        target_value = target.reshape(-1)[0]
        output_delta = (output - target_value) * output * (1.0 - output)

        hidden_deltas = []
        for index in range(2):
            activation = hidden[index]
            hidden_delta = (
                activation
                * (1.0 - activation)
                * self.output.weight[0, index]
                * output_delta
            )
            hidden_deltas.append(hidden_delta)

        with torch.no_grad():
            self.output.weight[0] -= learning_rate * output_delta * hidden
            for hidden_index, delta in enumerate(hidden_deltas):
                self.hidden[hidden_index].weight[0] -= learning_rate * delta * x

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        learning_rate: float = 5.0,
        iterations: int = 10_000,
    ) -> list[float]:
        losses: list[float] = []
        for _ in range(iterations):
            total = 0.0
            for sample, target in zip(x, y):
                prediction = self(sample)
                clipped = torch.clamp(prediction, 1e-12, 1 - 1e-12)
                loss = -target * torch.log(clipped) - (1 - target) * torch.log(1 - clipped)
                total += float(loss.item())
                self.backward_pass(sample, target.reshape(1), learning_rate)
            losses.append(total / len(x))
        return losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        values = [self(sample).detach()[0] for sample in x]
        return torch.stack(values)


class MLP(nn.Module):
    """One-hidden-layer network used in the Iris capacity experiment."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


@dataclass(frozen=True)
class MLPLosses:
    train: list[float]
    test: list[float]


def train_mlp(
    model: MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    learning_rate: float = 0.1,
    iterations: int = 1000,
) -> MLPLosses:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_losses: list[float] = []
    test_losses: list[float] = []

    for _ in range(iterations):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(x_train), y_train.long())
        train_loss.backward()
        optimizer.step()
        train_losses.append(float(train_loss.item()))

        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(x_test), y_test.long())
        test_losses.append(float(test_loss.item()))

    return MLPLosses(train=train_losses, test=test_losses)
