"""Shared Gaussian-mixture density, sampling, and trainable parameters."""

from __future__ import annotations

import math
import copy

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


class GaussianMixtureProposal:
    """Base implementation for CPU float32 Gaussian mixtures."""

    def __init__(self, logits: torch.Tensor, means: torch.Tensor) -> None:
        logits32 = torch.as_tensor(logits, dtype=torch.float32, device="cpu").clone()
        means32 = torch.as_tensor(means, dtype=torch.float32, device="cpu").clone()
        if logits32.ndim != 1 or means32.ndim != 2:
            raise ValueError("GMM logits and means have invalid ranks")
        if logits32.shape[0] != means32.shape[0] or means32.shape[1] < 1:
            raise ValueError("GMM logits and means have inconsistent shapes")
        if not bool(torch.isfinite(logits32).all()) or not bool(
            torch.isfinite(means32).all()
        ):
            raise ValueError("GMM parameters must be finite")
        self.logits = logits32
        self.means = means32

    @property
    def n_components(self) -> int:
        return int(self.logits.numel())

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def covariances(self) -> torch.Tensor:
        raise NotImplementedError

    def trainable_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the tensors optimized by the public KL fitting routines."""

        return (self.means, self.logits)

    def enable_gradients(self) -> None:
        for tensor in self.trainable_tensors():
            tensor.requires_grad_(True)

    def clone(self) -> "GaussianMixtureProposal":
        return copy.deepcopy(self)

    def scale_tril(self) -> torch.Tensor:
        covariances = self.covariances()
        return torch.linalg.cholesky(covariances)

    def sample(
        self,
        sample_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        component_indices = torch.multinomial(
            self.weights,
            sample_size,
            replacement=True,
            generator=generator,
        )
        standard = torch.randn(
            sample_size,
            self.dimension,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        selected_factors = self.scale_tril()[component_indices]
        return self.means[component_indices] + torch.bmm(
            selected_factors,
            standard.unsqueeze(-1),
        ).squeeze(-1)

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("GMM samples have the wrong shape")
        factors = self.scale_tril()
        normalization = self.dimension * math.log(2.0 * math.pi)
        components = []
        for index in range(self.n_components):
            difference = values - self.means[index]
            solved = torch.linalg.solve_triangular(
                factors[index],
                difference.T,
                upper=False,
            ).T
            mahalanobis = torch.sum(solved * solved, dim=1)
            log_determinant = 2.0 * torch.log(
                torch.diagonal(factors[index])
            ).sum()
            components.append(
                -0.5 * (normalization + log_determinant + mahalanobis)
            )
        component_logs = torch.stack(components, dim=1)
        return torch.logsumexp(
            component_logs + torch.log_softmax(self.logits, dim=0).unsqueeze(0),
            dim=1,
        )


class UnconstrainedGaussianMixture(GaussianMixtureProposal):
    """Full-covariance GMM with an unconstrained trainable factor per component.

    The raw factors define covariances as ``B B^T``. Sampling uses the raw
    factor, matching the authoritative implementation; density evaluation uses
    the operational covariance and its Cholesky factor.
    """

    def __init__(
        self,
        *,
        logits: torch.Tensor,
        means: torch.Tensor,
        factors: torch.Tensor,
        covariance_scale: float = 1.0,
    ) -> None:
        super().__init__(logits=logits, means=means)
        factors32 = torch.as_tensor(
            factors, dtype=torch.float32, device="cpu"
        ).clone()
        if factors32.shape != (
            self.n_components,
            self.dimension,
            self.dimension,
        ):
            raise ValueError("Unconstrained GMM factors have the wrong shape")
        if not bool(torch.isfinite(factors32).all()):
            raise ValueError("Unconstrained GMM factors must be finite")
        if float(covariance_scale) <= 0.0:
            raise ValueError("covariance_scale must be positive")
        self.factors = factors32
        self.covariance_scale = float(covariance_scale)

    @staticmethod
    def _training_samples(samples: torch.Tensor, components: int) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        if values.ndim != 2 or values.shape[0] < components:
            raise ValueError("A GMM fit needs at least one sample per component")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("GMM fit samples must be finite")
        return values.contiguous()

    @classmethod
    def from_samples(
        cls,
        samples: torch.Tensor,
        *,
        components: int,
        random_state: np.random.RandomState,
        covariance_scale: float = 1.0,
    ) -> "UnconstrainedGaussianMixture":
        values = cls._training_samples(samples, components)
        fitted = GaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=random_state,
        ).fit(values.numpy())
        covariance = torch.as_tensor(fitted.covariances_, dtype=torch.float32)
        covariance = covariance * float(covariance_scale)
        return cls(
            logits=torch.log(torch.as_tensor(fitted.weights_, dtype=torch.float32)),
            means=torch.as_tensor(fitted.means_, dtype=torch.float32),
            factors=torch.linalg.cholesky(covariance),
            covariance_scale=covariance_scale,
        )

    def covariances(self) -> torch.Tensor:
        return self.factors @ self.factors.transpose(-1, -2)

    def scale_tril(self) -> torch.Tensor:
        """Match the authoritative unconstrained-density jitter policy."""

        covariances = self.covariances()
        identity = torch.eye(
            self.dimension,
            dtype=covariances.dtype,
            device=covariances.device,
        )
        factors = []
        for covariance in covariances:
            epsilon = 1e-10
            factor = None
            for _ in range(8):
                candidate, info = torch.linalg.cholesky_ex(
                    covariance + epsilon * identity,
                    upper=False,
                )
                if int(info) == 0:
                    factor = candidate
                    break
                epsilon *= 10.0
            if factor is None:
                raise RuntimeError(
                    "Unconstrained GMM covariance remained non-positive-definite"
                )
            factors.append(factor)
        return torch.stack(factors)

    def trainable_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*super().trainable_tensors(), self.factors)

    def sample(
        self,
        sample_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        component_indices = torch.multinomial(
            self.weights,
            sample_size,
            replacement=True,
            generator=generator,
        )
        standard = torch.randn(
            sample_size,
            self.dimension,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        return self.means[component_indices] + torch.bmm(
            self.factors[component_indices],
            standard.unsqueeze(-1),
        ).squeeze(-1)

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
        covariance = (
            torch.as_tensor(fitted.covariances_, dtype=torch.float32)
            * self.covariance_scale
        )
        with torch.no_grad():
            self.means.copy_(torch.as_tensor(fitted.means_, dtype=torch.float32))
            self.logits.copy_(
                torch.log(torch.as_tensor(fitted.weights_, dtype=torch.float32))
            )
            self.factors.copy_(torch.linalg.cholesky(covariance))

    def smooth_from_previous(
        self,
        previous: "UnconstrainedGaussianMixture",
        *,
        new_fraction: float,
    ) -> None:
        if not 0.0 <= new_fraction <= 1.0:
            raise ValueError("new_fraction must lie between zero and one")
        if (
            type(previous) is not type(self)
            or self.n_components != previous.n_components
            or self.dimension != previous.dimension
        ):
            raise ValueError("Only matching unconstrained GMMs can be smoothed")
        old_fraction = 1.0 - new_fraction
        with torch.no_grad():
            self.means.mul_(new_fraction).add_(
                previous.means, alpha=old_fraction
            )
            self.logits.mul_(new_fraction).add_(
                previous.logits, alpha=old_fraction
            )
            self.factors.mul_(new_fraction).add_(
                previous.factors, alpha=old_fraction
            )
