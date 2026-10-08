"""Fixed-weight nominal/GMM defensive proposal."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from robust_cfi_is.distributions import GaussianDistribution
from robust_cfi_is.proposals.gmm import UnconstrainedGaussianMixture


class DefensiveMixtureProposal:
    """Full ``lambda*f + (1-lambda)*q_theta`` defensive density."""

    def __init__(self, nominal: GaussianDistribution,
                 adaptive: UnconstrainedGaussianMixture,
                 defensive_weight: float,
                 *,
                 numpy_random: np.random.RandomState) -> None:
        if nominal.dimension != adaptive.dimension:
            raise ValueError("Nominal and adaptive dimensions differ")
        if not 0.0 < defensive_weight < 1.0:
            raise ValueError("defensive_weight must lie strictly between 0 and 1")
        self.nominal = nominal
        self.adaptive = adaptive
        self.defensive_weight = float(defensive_weight)
        self.numpy_random = numpy_random

    @property
    def dimension(self) -> int:
        return self.adaptive.dimension

    @property
    def n_components(self) -> int:
        return self.adaptive.n_components

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        return torch.logaddexp(
            self.nominal.log_prob(values).to(values.dtype)
            + math.log(self.defensive_weight),
            self.adaptive.log_prob(values) + math.log1p(-self.defensive_weight),
        )

    def sample_stratified(self, sample_size: int, *,
                          generator: torch.Generator | None = None,
                          numpy_random: np.random.RandomState | None = None,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_size < 2:
            raise ValueError("stratified sampling needs at least two samples")
        nominal_count = min(max(int(round(self.defensive_weight * sample_size)), 1),
                            sample_size - 1)
        return (
            self.nominal.sample(
                nominal_count,
                numpy_random=self.numpy_random if numpy_random is None else numpy_random,
            ),
            self.adaptive.sample(sample_size - nominal_count, generator=generator),
        )

    def sample(self, sample_size: int, *,
               generator: torch.Generator | None = None) -> torch.Tensor:
        nominal, adaptive = self.sample_stratified(sample_size, generator=generator)
        joined = torch.cat((nominal, adaptive), dim=0)
        return joined[torch.randperm(sample_size, generator=generator)]

    def trainable_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.adaptive.trainable_tensors()

    def enable_gradients(self) -> None:
        self.adaptive.enable_gradients()

    def clone(self) -> "DefensiveMixtureProposal":
        return copy.deepcopy(self)

    def smooth_from_previous(self, previous: "DefensiveMixtureProposal", *,
                             new_fraction: float) -> None:
        if self.defensive_weight != previous.defensive_weight:
            raise ValueError("Cannot smooth defensive proposals with different weights")
        self.adaptive.smooth_from_previous(previous.adaptive, new_fraction=new_fraction)
