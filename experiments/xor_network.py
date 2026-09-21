"""Train the hand-written two-hidden-neuron network on XOR."""

import torch

from ml_foundations.supervised.neural_networks import ManualXORNetwork


x = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
y = torch.tensor([0.0, 1.0, 1.0, 0.0])

model = ManualXORNetwork(seed=42)
model.fit(x, y, learning_rate=5.0, iterations=10_000)
predictions = model.predict(x)

print("Targets:    ", y.tolist())
print("Predictions:", predictions.tolist())
print("Rounded:    ", torch.round(predictions).int().tolist())
