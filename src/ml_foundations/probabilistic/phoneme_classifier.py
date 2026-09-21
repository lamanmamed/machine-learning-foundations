"""Maximum-likelihood classification with two Gaussian mixture models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian_mixture import GaussianMixtureEM


@dataclass(frozen=True)
class ClassificationResult:
    predictions: np.ndarray
    accuracy: float

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy


class GMMPhonemeClassifier:
    """Fit one density model per phoneme and compare their likelihoods."""

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = 100,
        reg_covar: float = 0.0,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.class_labels_: tuple[int, int] | None = None
        self.models_: dict[int, GaussianMixtureEM] = {}

    def fit(self, x: np.ndarray, y: np.ndarray, class_labels: tuple[int, int] = (1, 2)) -> "GMMPhonemeClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self.class_labels_ = class_labels
        self.models_ = {}
        for offset, label in enumerate(class_labels):
            model = GaussianMixtureEM(
                self.n_components,
                covariance_type="diag",
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
                random_state=None if self.random_state is None else self.random_state + offset,
            )
            model.fit(x[y == label])
            self.models_[label] = model
        return self

    def likelihoods(self, x: np.ndarray) -> np.ndarray:
        if self.class_labels_ is None:
            raise RuntimeError("Call fit before using the classifier.")
        return np.column_stack([self.models_[label].likelihood(x) for label in self.class_labels_])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.class_labels_ is None:
            raise RuntimeError("Call fit before using the classifier.")
        likelihoods = self.likelihoods(np.asarray(x, dtype=float))
        labels = np.asarray(self.class_labels_)
        return labels[np.argmax(likelihoods, axis=1)]

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> ClassificationResult:
        predictions = self.predict(x)
        accuracy = float(np.mean(predictions == np.asarray(y)))
        return ClassificationResult(predictions=predictions, accuracy=accuracy)

    def classification_grid(
        self,
        x: np.ndarray,
        *,
        points_f1: int = 200,
        points_f2: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify a regular F1/F2 grid for plotting a decision map."""
        x = np.asarray(x, dtype=float)
        f1 = np.linspace(x[:, 0].min(), x[:, 0].max(), points_f1)
        f2 = np.linspace(x[:, 1].min(), x[:, 1].max(), points_f2)
        xx, yy = np.meshgrid(f1, f2)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        labels = self.predict(grid).reshape(xx.shape)
        return xx, yy, labels
