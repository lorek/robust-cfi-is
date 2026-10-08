"""Exact-count ordinary IS for defensive proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.proposals import DefensiveMixtureProposal


@dataclass(frozen=True)
class DefensiveISResult:
    sample_size: int
    evaluation_seed: int
    estimate: float
    standard_error: float
    cov: float | None
    event_count: int
    nominal_count: int
    adaptive_count: int


def defensive_ordinary_is(
    problem: RareEventProblem,
    proposal: DefensiveMixtureProposal,
    *,
    samples: int,
    seed: int,
) -> DefensiveISResult:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    numpy_random = np.random.RandomState(seed)
    nominal, adaptive = proposal.sample_stratified(
        samples,
        generator=generator,
        numpy_random=numpy_random,
    )
    contributions = []
    event_count = 0
    with torch.no_grad():
        for values in (nominal, adaptive):
            indicators = problem.indicator(values)
            weights = torch.exp(
                problem.nominal.log_prob(values)
                - proposal.log_prob(values).to(torch.float64)
            )
            contributions.append(weights * indicators.to(torch.float64))
            event_count += int(indicators.sum())
    first, second = contributions
    weight = proposal.defensive_weight
    estimate = weight * first.mean() + (1.0 - weight) * second.mean()
    variance = (
        weight**2 * first.var(unbiased=False) / first.numel()
        + (1.0 - weight) ** 2 * second.var(unbiased=False) / second.numel()
    )
    standard_error = torch.sqrt(variance)
    estimate_value = float(estimate.detach())
    return DefensiveISResult(
        samples,
        seed,
        estimate_value,
        float(standard_error.detach()),
        float((standard_error / estimate).detach()) if estimate_value > 0.0 else None,
        event_count,
        first.numel(),
        second.numel(),
    )
