"""Feature preprocessing shared by the supervised experiments."""

from __future__ import annotations

import torch


def standardize(
    train: torch.Tensor,
    test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Standardize with statistics calculated from the training set only."""
    mean = train.mean(dim=0)
    std = train.std(dim=0)
    if torch.any(std == 0):
        raise ValueError("Cannot standardize a feature with zero training variance.")

    train_scaled = (train - mean) / std
    test_scaled = None if test is None else (test - mean) / std
    return train_scaled, test_scaled, mean, std


def add_bias(x: torch.Tensor) -> torch.Tensor:
    """Append a column of ones to a 2D feature matrix."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor.")
    return torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
