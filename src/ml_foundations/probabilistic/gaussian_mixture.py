"""Gaussian mixture models trained with Expectation-Maximization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianMixtureParameters:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray


class GaussianMixtureEM:
    """Mixture of Gaussians with diagonal or full covariance matrices.

    The diagonal-covariance path follows the EM updates used in the original
    vowel experiments. Full covariance is included for the singularity example.
    """

    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "diag",
        max_iter: int = 100,
        reg_covar: float = 0.0,
        random_state: int | None = None,
    ):
        if covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_type must be 'diag' or 'full'.")
        if n_components < 1:
            raise ValueError("n_components must be positive.")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.reg_covar = float(reg_covar)
        self.random_state = random_state
        self.parameters_: GaussianMixtureParameters | None = None
        self.log_likelihood_: list[float] = []

    @staticmethod
    def _component_densities(
        x: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        n_samples, n_features = x.shape
        n_components = len(weights)
        densities = np.zeros((n_samples, n_components), dtype=float)

        for index in range(n_components):
            covariance = covariances[index]
            determinant = float(np.linalg.det(covariance))
            if determinant <= 0 or not np.isfinite(determinant):
                raise np.linalg.LinAlgError(
                    "Covariance matrix is singular. Add diagonal regularization "
                    "or remove linearly dependent features."
                )

            inverse = np.linalg.pinv(covariance)
            centered = x - means[index]
            exponent = np.einsum("ni,ij,nj->n", centered, inverse, centered)
            normalizer = np.sqrt(((2 * np.pi) ** n_features) * determinant)
            densities[:, index] = weights[index] * np.exp(-0.5 * exponent) / normalizer

        return densities

    def fit(self, x: np.ndarray) -> "GaussianMixtureEM":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must have shape (n_samples, n_features).")
        n_samples, n_features = x.shape
        if n_samples < self.n_components:
            raise ValueError("Need at least one sample per mixture component.")

        rng = np.random.RandomState(self.random_state)
        means = x[rng.choice(n_samples, self.n_components, replace=False)].copy()
        weights = np.full(self.n_components, 1.0 / self.n_components)

        base_covariance = np.cov(x.T)
        if n_features == 1:
            base_covariance = np.asarray([[float(base_covariance)]])
        covariances = np.repeat(
            (base_covariance / self.n_components)[None, :, :],
            self.n_components,
            axis=0,
        )
        if self.reg_covar:
            covariances += self.reg_covar * np.eye(n_features)[None, :, :]

        self.log_likelihood_ = []

        for _ in range(self.max_iter):
            weighted = self._component_densities(x, means, covariances, weights)
            total = weighted.sum(axis=1)
            if np.any(total <= 0) or np.any(~np.isfinite(total)):
                raise FloatingPointError("Mixture likelihood became non-finite.")
            responsibilities = weighted / total[:, None]
            self.log_likelihood_.append(float(np.log(total).sum()))

            component_mass = responsibilities.sum(axis=0)
            for index in range(self.n_components):
                mass = component_mass[index]
                if mass <= 1e-12:
                    means[index] = x[rng.randint(n_samples)]
                    covariances[index] = base_covariance / self.n_components
                    if self.reg_covar:
                        covariances[index] += self.reg_covar * np.eye(n_features)
                    weights[index] = 1.0 / n_samples
                    continue

                means[index] = (x.T @ responsibilities[:, index]) / mass
                centered = x - means[index]

                if self.covariance_type == "diag":
                    variance = (
                        (centered ** 2).T @ responsibilities[:, index]
                    ) / mass
                    covariances[index] = np.diag(variance)
                else:
                    weighted_centered = centered * responsibilities[:, index, None]
                    covariances[index] = centered.T @ weighted_centered / mass

                if self.reg_covar:
                    covariances[index] += self.reg_covar * np.eye(n_features)
                weights[index] = mass / n_samples

        weights /= weights.sum()
        self.parameters_ = GaussianMixtureParameters(means, covariances, weights)
        return self

    def _require_fit(self) -> GaussianMixtureParameters:
        if self.parameters_ is None:
            raise RuntimeError("Call fit before using the model.")
        return self.parameters_

    def component_densities(self, x: np.ndarray) -> np.ndarray:
        params = self._require_fit()
        return self._component_densities(
            np.asarray(x, dtype=float),
            params.means,
            params.covariances,
            params.weights,
        )

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        return self.component_densities(x).sum(axis=1)

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        weighted = self.component_densities(x)
        return weighted / weighted.sum(axis=1, keepdims=True)


def covariance_rank(x: np.ndarray) -> int:
    """Return the rank of the sample covariance matrix."""
    covariance = np.cov(np.asarray(x, dtype=float).T)
    return int(np.linalg.matrix_rank(covariance))
