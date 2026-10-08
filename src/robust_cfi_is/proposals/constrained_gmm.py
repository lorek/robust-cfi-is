"""Trainable constrained Gaussian-mixture proposal used by CFI."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from robust_cfi_is.proposals.gmm import GaussianMixtureProposal


class ConstrainedGaussianMixture(GaussianMixtureProposal):
    """GMM with component covariance ``L L^T + beta * Sigma``.

    The fitting and smoothing operations preserve the behavior of the current
    scientific CFI implementation. The learned factors need not be triangular;
    inference materializes each operational covariance and uses its Cholesky.
    """

    def __init__(
        self,
        *,
        logits: torch.Tensor,
        means: torch.Tensor,
        factors: torch.Tensor,
        nominal_covariance: torch.Tensor,
        beta: float = 0.5,
        covariance_scale: float = 1.0,
    ) -> None:
        super().__init__(logits=logits, means=means)
        factors32 = torch.as_tensor(
            factors, dtype=torch.float32, device="cpu"
        ).clone()
        nominal32 = torch.as_tensor(
            nominal_covariance, dtype=torch.float32, device="cpu"
        ).clone()
        if factors32.shape != (
            self.n_components,
            self.dimension,
            self.dimension,
        ):
            raise ValueError("Constrained GMM factors have the wrong shape")
        if nominal32.shape != (self.dimension, self.dimension):
            raise ValueError("Nominal covariance has the wrong shape")
        if not bool(torch.isfinite(factors32).all()) or not bool(
            torch.isfinite(nominal32).all()
        ):
            raise ValueError("Constrained GMM tensors must be finite")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must lie between zero and one")
        if float(covariance_scale) <= 0.0:
            raise ValueError("covariance_scale must be positive")
        try:
            torch.linalg.cholesky(
                0.5
                * (
                    nominal32.to(torch.float64)
                    + nominal32.to(torch.float64).T
                )
            )
        except RuntimeError as exc:
            raise ValueError("Nominal covariance must be positive definite") from exc
        self.factors = factors32
        self.nominal_covariance = nominal32
        self.beta = float(beta)
        self.covariance_scale = float(covariance_scale)

    @classmethod
    def from_samples(
        cls,
        samples: torch.Tensor,
        *,
        components: int,
        nominal_covariance: torch.Tensor,
        beta: float,
        random_state: np.random.RandomState,
    ) -> "ConstrainedGaussianMixture":
        values = cls._training_samples(samples, components)
        fitted = GaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=random_state,
        ).fit(values.numpy())
        covariance = torch.as_tensor(fitted.covariances_, dtype=torch.float32)
        # This is the authoritative initialization rule: the learned remainder
        # starts as beta times the fitted covariance.
        factors = torch.linalg.cholesky(float(beta) * covariance)
        return cls(
            logits=torch.log(
                torch.as_tensor(fitted.weights_, dtype=torch.float32)
            ),
            means=torch.as_tensor(fitted.means_, dtype=torch.float32),
            factors=factors,
            nominal_covariance=nominal_covariance,
            beta=beta,
        )

    @staticmethod
    def _training_samples(samples: torch.Tensor, components: int) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        if values.ndim != 2 or values.shape[0] < components:
            raise ValueError("A GMM fit needs at least one sample per component")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("GMM fit samples must be finite")
        return values.contiguous()

    def clone(self) -> "ConstrainedGaussianMixture":
        return copy.deepcopy(self)

    def trainable_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*super().trainable_tensors(), self.factors)

    def covariances(self) -> torch.Tensor:
        remainder = self.factors @ self.factors.transpose(-1, -2)
        return self.covariance_scale * (
            remainder + self.beta * self.nominal_covariance.unsqueeze(0)
        )

    def fit_to_samples(
        self,
        samples: torch.Tensor,
        *,
        random_state: np.random.RandomState,
    ) -> None:
        values = self._training_samples(samples, self.n_components)
        fitted = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=random_state,
        ).fit(values.numpy())
        covariance = torch.as_tensor(fitted.covariances_, dtype=torch.float32)
        target_remainder = covariance - self.beta * self.nominal_covariance.unsqueeze(0)
        target_remainder = 0.5 * (
            target_remainder + target_remainder.transpose(-1, -2)
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(target_remainder)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        factors = eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues))
        with torch.no_grad():
            self.means.copy_(torch.as_tensor(fitted.means_, dtype=torch.float32))
            self.logits.copy_(
                torch.log(torch.as_tensor(fitted.weights_, dtype=torch.float32))
            )
            self.factors.copy_(factors.to(torch.float32))

    def smooth_from_previous(
        self,
        previous: "ConstrainedGaussianMixture",
        *,
        new_fraction: float,
    ) -> None:
        """Blend a newly fitted model with the preceding CFI proposal.

        ``new_fraction`` is the public smoothing value. A value of 0.75 keeps
        75 percent of the new fit and 25 percent of the previous proposal,
        matching the current CFI implementation.
        """

        if not 0.0 <= new_fraction <= 1.0:
            raise ValueError("new_fraction must lie between zero and one")
        if (
            self.n_components != previous.n_components
            or self.dimension != previous.dimension
        ):
            raise ValueError("Only matching constrained GMMs can be smoothed")
        old_fraction = 1.0 - new_fraction
        with torch.no_grad():
            self.means.mul_(new_fraction).add_(
                previous.means, alpha=old_fraction
            )
            new_weights = (
                new_fraction * self.weights
                + old_fraction * previous.weights
            )
            self.logits.copy_(torch.log(new_weights + 1e-10))

            new_remainder = self.factors @ self.factors.transpose(-1, -2)
            old_remainder = previous.factors @ previous.factors.transpose(-1, -2)
            smoothed = (
                new_fraction * new_remainder + old_fraction * old_remainder
            )
            minimum_nominal = torch.linalg.eigvalsh(
                self.nominal_covariance
            ).min()
            floor = 1e-6 * minimum_nominal
            eigenvalues, eigenvectors = torch.linalg.eigh(
                0.5 * (smoothed + smoothed.transpose(-1, -2))
            )
            eigenvalues = torch.clamp(eigenvalues, min=floor)
            self.factors.copy_(
                eigenvectors @ torch.diag_embed(torch.sqrt(eigenvalues))
            )
