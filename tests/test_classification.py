import unittest

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_foundations.supervised.classification import OneVsRestLogistic, sigmoid
from ml_foundations.supervised.preprocessing import standardize


class ClassificationTests(unittest.TestCase):
    def test_sigmoid_maps_zero_to_half(self):
        self.assertAlmostEqual(float(sigmoid(torch.tensor(0.0))), 0.5)

    def test_iris_one_vs_rest_reproduces_27_of_30(self):
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        x_train, x_test, _, _ = standardize(x_train, x_test)

        torch.manual_seed(0)
        model = OneVsRestLogistic(4, 3).fit(
            x_train,
            torch.tensor(y_train),
            learning_rate=0.1,
            iterations=1000,
        )
        correct = int((model.predict(x_test).numpy() == y_test).sum())
        self.assertEqual(correct, 27)


if __name__ == "__main__":
    unittest.main()
