"""Coverage-first initialization (CFI)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.methods._proposal import fit_initial_proposal
from robust_cfi_is.proposals import (
    ConstrainedGaussianMixture,
    GaussianMixtureProposal,
)


@dataclass(frozen=True)
class CFILevel:
    index: int
    gamma: float
    elite_count: int
    beta: float | None
    target_reached: bool


@dataclass(frozen=True)
class CFIFitResult:
    proposal: GaussianMixtureProposal
    levels: tuple[CFILevel, ...]
    target_reached: bool
    final_fit_elite_count: int


def _adapt_beta(
    proposal: ConstrainedGaussianMixture,
    gamma_history: list[float],
    *,
    beta_max: float = 0.5,
    beta_min: float = 0.2,
    relative_threshold: float = 5e-3,
    plateau_levels: int = 4,
    decrease_multiplier: float = 0.85,
    increase: float = 0.01,
) -> None:
    """Apply the current constrained-CFI beta schedule.

    The production implementation reconstructs the proposal after every
    level, so its transient best-gamma attribute does not survive. The
    effective rule compares the current level with the best of the preceding
    four levels; this function preserves that behavior explicitly.
    """

    if len(gamma_history) < plateau_levels + 1:
        return
    current = gamma_history[-1]
    previous_best = max(gamma_history[-(plateau_levels + 1) : -1])
    relative_improvement = (current - previous_best) / max(abs(previous_best), 1.0)
    if relative_improvement < relative_threshold:
        proposal.beta = max(beta_min, proposal.beta * decrease_multiplier)
    else:
        proposal.beta = min(beta_max, proposal.beta + increase)


def fit_cfi(
    problem: RareEventProblem,
    *,
    components: int,
    seed: int,
    initial_samples: int,
    samples_per_level: int,
    rho: float,
    max_levels: int,
    smoothing: float,
    beta: float = 0.5,
    constrained: bool = True,
) -> CFIFitResult:
    """Fit a GMM with the authoritative unweighted elite-sample CFI updates."""

    if components < 1:
        raise ValueError("components must be positive")
    if initial_samples < components or samples_per_level < components:
        raise ValueError("CFI needs at least one sample per component")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must lie between zero and one")

    initialized = fit_initial_proposal(
        problem,
        components=components,
        constrained=constrained,
        seed=seed,
        initial_samples=initial_samples,
        beta=beta,
    )
    proposal = initialized.proposal
    torch_generator = initialized.torch_generator
    numpy_random = initialized.numpy_random

    gamma_history: list[float] = []
    level_records: list[CFILevel] = []
    target_reached = False

    for level in range(max_levels):
        samples = proposal.sample(samples_per_level, generator=torch_generator)
        values = problem.event_score(samples)
        gamma_value = float(torch.quantile(values, 1.0 - rho))
        target_reached = gamma_value > problem.gamma
        if target_reached:
            gamma_value = float(problem.gamma)
        gamma_history.append(gamma_value)
        if type(proposal) is ConstrainedGaussianMixture:
            _adapt_beta(proposal, gamma_history)

        elite = samples[values >= gamma_value]
        if elite.shape[0] < components:
            raise RuntimeError(
                f"CFI level {level} produced only {elite.shape[0]} elite samples"
            )
        updated = proposal.clone()
        if type(updated) is ConstrainedGaussianMixture:
            updated.beta = proposal.beta
        updated.fit_to_samples(elite, random_state=numpy_random)
        if smoothing > 0.0:
            updated.smooth_from_previous(
                proposal,
                new_fraction=smoothing,
            )
        proposal = updated
        if target_reached and type(proposal) is ConstrainedGaussianMixture:
            proposal.beta = 0.5
        level_records.append(
            CFILevel(
                index=level,
                gamma=gamma_value,
                elite_count=int(elite.shape[0]),
                beta=(
                    float(proposal.beta)
                    if type(proposal) is ConstrainedGaussianMixture
                    else None
                ),
                target_reached=target_reached,
            )
        )
        if target_reached:
            break

    # Preserve the production CFI final fit: after the level loop, draw one
    # fresh batch and fit without smoothing to the exact target event.
    final_samples = proposal.sample(samples_per_level, generator=torch_generator)
    final_values = problem.event_score(final_samples)
    final_elite = final_samples[final_values >= problem.gamma]
    final_elite_count = int(final_elite.shape[0])
    if final_elite_count > 0:
        proposal.fit_to_samples(final_elite, random_state=numpy_random)

    return CFIFitResult(
        proposal=proposal,
        levels=tuple(level_records),
        target_reached=target_reached,
        final_fit_elite_count=final_elite_count,
    )
