"""Reproduce the learning-rate comparison from the diabetes regression experiment."""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split

from ml_foundations.datasets import load_raw_diabetes
from ml_foundations.supervised.preprocessing import add_bias, standardize
from ml_foundations.supervised.regression import fit_linear_regression, mean_squared_error


x, y = load_raw_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

x_train, x_test, _, _ = standardize(x_train, x_test)
x_train = add_bias(x_train)
x_test = add_bias(x_test)

for alpha in [1.0, 0.1, 0.01, 0.001]:
    fit = fit_linear_regression(x_train, y_train, learning_rate=alpha, iterations=100)
    with torch.no_grad():
        test_mse = float(mean_squared_error(fit.model(x_test), y_test).item())
    print(
        f"alpha={alpha:<5} "
        f"train_mse={min(fit.losses):.2f} "
        f"test_mse={test_mse:.2f}"
    )
