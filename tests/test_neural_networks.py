import unittest

import torch

from ml_foundations.supervised.neural_networks import ManualXORNetwork


class NeuralNetworkTests(unittest.TestCase):
    def test_manual_network_learns_xor(self):
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
        predictions = torch.round(model.predict(x))
        self.assertTrue(torch.equal(predictions, y))


if __name__ == "__main__":
    unittest.main()
