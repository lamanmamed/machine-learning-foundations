import unittest

import torch

from ml_foundations.supervised.preprocessing import add_bias, standardize


class PreprocessingTests(unittest.TestCase):
    def test_standardization_uses_training_statistics(self):
        train = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        test = torch.tensor([[7.0, 22.0]])
        train_scaled, test_scaled, mean, std = standardize(train, test)
        self.assertTrue(torch.allclose(train_scaled.mean(dim=0), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(test_scaled, (test - mean) / std))

    def test_add_bias_appends_ones(self):
        x = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        result = add_bias(x)
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(torch.equal(result[:, -1], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
