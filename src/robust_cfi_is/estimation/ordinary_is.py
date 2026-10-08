"""Stable ordinary, unnormalized importance sampling."""

from __future__ import annotations

import math

import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.estimation.metrics import ISResult
from robust_cfi_is.proposals import (
    CertifiedGaussianMixture,
    GaussianMixtureProposal,
    UnconstrainedGaussianMixture,
    RealNVPProposal,
)


def _finite(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _contribution_moments(
    log_weights: torch.Tensor,
    indicators: torch.Tensor,
) -> tuple[float, float, int]:
    logs = _finite(log_weights, "log weights").to(torch.float64).cpu()
    mask = torch.as_tensor(indicators, dtype=torch.bool, device="cpu")
    if logs.ndim != 1 or mask.shape != logs.shape or logs.numel() == 0:
        raise ValueError("Importance-weight inputs have inconsistent shapes")
    event_count = int(mask.sum())
    if event_count == 0:
        return 0.0, 0.0, 0

    event_logs = logs[mask]
    shift = float(event_logs.max())
    scaled = torch.zeros_like(logs)
    scaled[mask] = torch.exp(event_logs - shift)
    scaled_mean = float(scaled.mean())
    scaled_std = float(torch.std(scaled, unbiased=False))
    estimate = math.exp(shift + math.log(scaled_mean))
    standard_error = (
        0.0
        if scaled_std == 0.0
        else math.exp(shift + math.log(scaled_std)) / math.sqrt(logs.numel())
    )
    if not math.isfinite(estimate) or not math.isfinite(standard_error):
        raise ValueError("Ordinary-IS moments are not finite")
    return estimate, standard_error, event_count


def ordinary_is(
    problem: RareEventProblem,
    proposal: GaussianMixtureProposal | RealNVPProposal,
    *,
    samples: int,
    seed: int,
) -> ISResult:
    """Estimate the event probability without self-normalizing the weights."""

    if not isinstance(
        proposal,
        (CertifiedGaussianMixture, UnconstrainedGaussianMixture, RealNVPProposal),
    ):
        raise TypeError(
            "ordinary_is requires either a certified constrained final GMM "
            "or an unconstrained public GMM, or a public RealNVP proposal"
        )
    if samples <= 0:
        raise ValueError("samples must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        draws = _finite(
            proposal.sample(samples, generator=generator),
            "proposal samples",
        )
        log_target = _finite(problem.nominal.log_prob(draws), "log target").to(
            torch.float64
        )
        log_proposal = _finite(proposal.log_prob(draws), "log proposal").to(
            dtype=torch.float64, device="cpu"
        )
    log_weights = _finite(log_target - log_proposal, "log weights")
    indicators = problem.indicator(draws)
    estimate, standard_error, event_count = _contribution_moments(
        log_weights,
        indicators,
    )
    cov = standard_error / estimate if estimate > 0.0 else None
    return ISResult(
        estimator="ordinary_unnormalized_importance_sampling",
        self_normalized=False,
        sample_size=samples,
        evaluation_seed=seed,
        estimate=estimate,
        standard_error=standard_error,
        cov=cov,
        event_count=event_count,
        log_weight_minimum=float(log_weights.min()),
        log_weight_maximum=float(log_weights.max()),
    )
