import unittest

import numpy as np

from ml_foundations.probabilistic.gaussian_mixture import GaussianMixtureEM, covariance_rank
from ml_foundations.probabilistic.phoneme_classifier import GMMPhonemeClassifier


class GaussianMixtureTests(unittest.TestCase):
    def test_em_separates_two_synthetic_classes(self):
        rng = np.random.RandomState(0)
        class_one = rng.normal(loc=[0.0, 0.0], scale=[0.4, 0.5], size=(80, 2))
        class_two = rng.normal(loc=[4.0, 4.0], scale=[0.5, 0.4], size=(80, 2))
        x = np.vstack([class_one, class_two])
        y = np.array([1] * len(class_one) + [2] * len(class_two))

        classifier = GMMPhonemeClassifier(2, random_state=0, reg_covar=1e-6).fit(x, y)
        self.assertGreater(classifier.evaluate(x, y).accuracy, 0.98)

    def test_dependent_third_feature_makes_covariance_rank_deficient(self):
        x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]])
        x3 = np.column_stack([x[:, 0], x[:, 1], x[:, 0] + x[:, 1]])
        self.assertLess(covariance_rank(x3), x3.shape[1])

    def test_diagonal_regularization_allows_full_covariance_fit(self):
        rng = np.random.RandomState(1)
        x = rng.normal(size=(100, 2))
        x3 = np.column_stack([x[:, 0], x[:, 1], x[:, 0] + x[:, 1]])
        model = GaussianMixtureEM(
            2,
            covariance_type="full",
            max_iter=20,
            reg_covar=0.001,
            random_state=0,
        ).fit(x3)
        self.assertIsNotNone(model.parameters_)


if __name__ == "__main__":
    unittest.main()
