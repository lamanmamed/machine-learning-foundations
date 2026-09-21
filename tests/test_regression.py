import unittest

import torch

from ml_foundations.supervised.regression import (
    fit_linear_regression,
    fit_polynomial_ridge,
    polynomial_features,
)


class RegressionTests(unittest.TestCase):
    def test_linear_regression_learns_simple_line(self):
        x = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        y = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
        fit = fit_linear_regression(x, y, learning_rate=0.05, iterations=1000)
        predictions = fit.model(x)
        self.assertLess(torch.mean((predictions - y) ** 2).item(), 1e-3)

    def test_polynomial_features_include_bias_through_degree_five(self):
        features = polynomial_features(torch.tensor([2.0]), degree=5)
        expected = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]])
        self.assertTrue(torch.equal(features, expected))

    def test_regularization_keeps_polynomial_fit_finite(self):
        x = torch.linspace(-1, 1, 7)
        y = (x ** 2).reshape(-1, 1)
        fit = fit_polynomial_ridge(
            x,
            y,
            degree=5,
            learning_rate=0.05,
            regularization=1.0,
            iterations=200,
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(fit.losses))))


if __name__ == "__main__":
    unittest.main()
