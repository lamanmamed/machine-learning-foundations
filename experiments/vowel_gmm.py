"""Fit the two-phoneme GMM classifiers and demonstrate covariance singularity."""

from __future__ import annotations

import argparse

import numpy as np

from ml_foundations.datasets import load_peterson_barney
from ml_foundations.probabilistic.gaussian_mixture import GaussianMixtureEM, covariance_rank
from ml_foundations.probabilistic.phoneme_classifier import GMMPhonemeClassifier


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Path to PB_data.npy")
args = parser.parse_args()

x, y = load_peterson_barney(args.dataset)
mask = np.isin(y, [1, 2])
x_two = x[mask]
y_two = y[mask]

for components, seed in [(3, 1), (6, 3)]:
    classifier = GMMPhonemeClassifier(
        components,
        max_iter=100,
        random_state=seed,
    ).fit(x_two, y_two)
    result = classifier.evaluate(x_two, y_two)
    print(
        f"K={components}: accuracy={result.accuracy:.2%}, "
        f"error={result.error_rate:.2%}"
    )

phoneme_one = x[y == 1]
linearly_dependent = np.column_stack(
    [phoneme_one[:, 0], phoneme_one[:, 1], phoneme_one[:, 0] + phoneme_one[:, 1]]
)
print(
    "3D covariance rank:",
    covariance_rank(linearly_dependent),
    "for 3 features",
)

regularized = GaussianMixtureEM(
    3,
    covariance_type="full",
    max_iter=100,
    reg_covar=0.001,
    random_state=0,
).fit(linearly_dependent)
print("Regularized full-covariance EM completed:", regularized.parameters_ is not None)
