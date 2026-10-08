"""Analytic rare-event benchmarks supported by the public package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from robust_cfi_is.distributions import GaussianDistribution


def g1_2d(samples: torch.Tensor) -> torch.Tensor:
    """Gao et al. leaf benchmark used as ``g1`` in the paper.

    The rare event is ``g1_2d(x) > 0`` for ``X ~ N(0, I_2)``.
    """

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("g1_2d expects a tensor with shape (n, 2)")
    displacement = 3.8
    first = (samples[:, 0] + displacement) ** 2 + (
        samples[:, 1] + displacement
    ) ** 2
    second = (samples[:, 0] - displacement) ** 2 + (
        samples[:, 1] - displacement
    ) ** 2
    return 1.0 - torch.minimum(first, second)


def _require_shape(samples: torch.Tensor, dimension: int, name: str) -> None:
    if samples.ndim != 2 or samples.shape[1] != dimension:
        raise ValueError(f"{name} expects a tensor with shape (n, {dimension})")


def g2_2d(samples: torch.Tensor) -> torch.Tensor:
    """Four-branch nonlinear benchmark used as ``g2`` in the paper."""

    _require_shape(samples, 2, "g2_2d")
    root_two = 2.0**0.5
    first = 3.0 + 0.1 * (samples[:, 0] - samples[:, 1]) ** 2 - (
        samples[:, 0] + samples[:, 1]
    ) / root_two
    second = 3.0 + 0.1 * (samples[:, 0] - samples[:, 1]) ** 2 + (
        samples[:, 0] + samples[:, 1]
    ) / root_two
    third = samples[:, 0] - samples[:, 1] + 7.0 / root_two
    fourth = samples[:, 1] - samples[:, 0] + 7.0 / root_two
    return -torch.minimum(torch.minimum(first, second), torch.minimum(third, fourth))


def g3_2d(samples: torch.Tensor) -> torch.Tensor:
    _require_shape(samples, 2, "g3_2d")
    return -(
        0.1 * (samples[:, 0] - samples[:, 1]) ** 2
        - (samples[:, 0] + samples[:, 1]) / (2.0**0.5)
        + 2.5
    )


def g5_powell(samples: torch.Tensor) -> torch.Tensor:
    _require_shape(samples, 40, "g5_powell")
    blocks = samples.reshape(-1, 10, 4)
    return -(
        (blocks[:, :, 0] + 10.0 * blocks[:, :, 1]) ** 2
        + 5.0 * (blocks[:, :, 2] - blocks[:, :, 3]) ** 2
        + (blocks[:, :, 1] - 2.0 * blocks[:, :, 2]) ** 4
        + 10.0 * (blocks[:, :, 0] - blocks[:, :, 3]) ** 4
    ).sum(dim=1)


def g6_halfspace(samples: torch.Tensor) -> torch.Tensor:
    _require_shape(samples, 100, "g6_halfspace")
    return samples.sum(dim=1) / 10.0


def g10_sum_exp(samples: torch.Tensor) -> torch.Tensor:
    _require_shape(samples, 10, "g10_sum_exp")
    return torch.exp(samples).sum(dim=1)


@dataclass(frozen=True)
class RareEventProblem:
    name: str
    dimension: int
    gamma: float
    nominal: GaussianDistribution
    event_score: Callable[[torch.Tensor], torch.Tensor]
    reference_probability: float | None = None

    def indicator(self, samples: torch.Tensor) -> torch.Tensor:
        values = self.event_score(samples)
        return values > self.gamma


def get_benchmark(name: str) -> RareEventProblem:
    specifications = {
        "g1_2d_Sigma1": (2, 0.0, g1_2d, 4.793415539e-6, None, None),
        "g2b_papaioannou_2d_Sigma1": (2, 0.0, g2_2d, 2.22668e-3, None, None),
        "g3_2d_Sigma1": (2, 0.0, g3_2d, 4.20730551e-3, None, None),
        "g5_powell_40d_Sigma1": (40, -403.66366875, g5_powell, 3.15e-5, None, None),
        "g6_papaioannou_100d_Sigma1": (
            100, 3.5, g6_halfspace, 2.326290790355401e-4, None, None,
        ),
        "s10_sum_exp_gamma_20k_Sigma2": (
            10, 20_000.0, g10_sum_exp, 1.03526e-3,
            torch.arange(-9.0, 1.0), torch.arange(1.0, 11.0),
        ),
        "s10_sum_exp_gamma_40k_Sigma2": (
            10, 40_000.0, g10_sum_exp, 4.683e-4,
            torch.arange(-9.0, 1.0), torch.arange(1.0, 11.0),
        ),
        "s10_sum_exp_gamma_500k_Sigma2": (
            10, 500_000.0, g10_sum_exp, 1.77400e-5,
            torch.arange(-9.0, 1.0), torch.arange(1.0, 11.0),
        ),
    }
    try:
        dimension, gamma, score, probability, mean, diagonal = specifications[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported analytic benchmark {name!r}") from exc
    return RareEventProblem(
        name=name,
        dimension=dimension,
        gamma=gamma,
        nominal=GaussianDistribution(
            mean=torch.zeros(dimension) if mean is None else mean,
            covariance=torch.eye(dimension) if diagonal is None else torch.diag(diagonal),
        ),
        event_score=score,
        reference_probability=probability,
    )
