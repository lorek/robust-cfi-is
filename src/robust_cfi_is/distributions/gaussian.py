"""Explicit CPU Gaussian distribution."""

from __future__ import annotations

import math

import numpy as np
import torch


class GaussianDistribution:
    """A full-covariance Gaussian with float32 samples and float64 log density."""

    def __init__(self, mean: torch.Tensor, covariance: torch.Tensor) -> None:
        mean32 = torch.as_tensor(mean, dtype=torch.float32, device="cpu").clone()
        covariance32 = torch.as_tensor(
            covariance, dtype=torch.float32, device="cpu"
        ).clone()
        if mean32.ndim != 1 or covariance32.shape != (mean32.numel(), mean32.numel()):
            raise ValueError("Gaussian mean/covariance dimensions do not match")
        if not bool(torch.isfinite(mean32).all()) or not bool(
            torch.isfinite(covariance32).all()
        ):
            raise ValueError("Gaussian parameters must be finite")
        covariance64 = 0.5 * (
            covariance32.to(torch.float64) + covariance32.to(torch.float64).T
        )
        try:
            factor64 = torch.linalg.cholesky(covariance64)
        except RuntimeError as exc:
            raise ValueError("Gaussian covariance must be positive definite") from exc
        self._mean = mean32
        self._covariance = covariance32
        self._factor32 = factor64.to(torch.float32)
        self._factor64 = factor64

    @property
    def dimension(self) -> int:
        return int(self._mean.numel())

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.clone()

    @property
    def covariance(self) -> torch.Tensor:
        return self._covariance.clone()

    def sample(
        self,
        sample_size: int,
        *,
        generator: torch.Generator | None = None,
        numpy_random: np.random.RandomState | None = None,
    ) -> torch.Tensor:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if numpy_random is not None:
            if generator is not None:
                raise ValueError("Provide either a Torch or NumPy generator, not both")
            values = numpy_random.multivariate_normal(
                self._mean.numpy(),
                self._covariance.numpy(),
                size=sample_size,
            )
            return torch.from_numpy(values).to(torch.float32)
        standard = torch.randn(
            sample_size,
            self.dimension,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        return self._mean + standard @ self._factor32.T

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(samples, device="cpu")
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("Gaussian samples have the wrong shape")
        values64 = values.to(torch.float64)
        difference = values64 - self._mean.to(torch.float64)
        solved = torch.linalg.solve_triangular(
            self._factor64,
            difference.T,
            upper=False,
        ).T
        mahalanobis = torch.sum(solved * solved, dim=1)
        log_determinant = 2.0 * torch.log(torch.diagonal(self._factor64)).sum()
        return -0.5 * (
            self.dimension * math.log(2.0 * math.pi)
            + log_determinant
            + mahalanobis
        )
