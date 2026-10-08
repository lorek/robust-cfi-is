"""Deterministic proposal construction shared by public fitting methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.proposals import (
    ConstrainedGaussianMixture,
    GaussianMixtureProposal,
    UnconstrainedGaussianMixture,
)


@dataclass(frozen=True)
class ProposalInitialization:
    proposal: GaussianMixtureProposal
    torch_generator: torch.Generator
    numpy_random: np.random.RandomState


def consume_authoritative_constructor_draws(
    generator: torch.Generator,
    *,
    components: int,
    dimension: int,
) -> None:
    """Advance Torch exactly as the authoritative GMM constructor does.

    The current source draws temporary means and logits and then overwrites
    them with the sklearn fit. Preserving these draws retains its deterministic
    training trajectory without keeping discarded public parameters.
    """

    torch.randn(
        components,
        dimension,
        dtype=torch.float32,
        generator=generator,
    )
    torch.randn(components, dtype=torch.float32, generator=generator)


def fit_initial_proposal(
    problem: RareEventProblem,
    *,
    components: int,
    constrained: bool,
    seed: int,
    initial_samples: int,
    beta: float = 0.5,
) -> ProposalInitialization:
    if components < 1 or initial_samples < components:
        raise ValueError("Proposal initialization needs one sample per component")
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed)
    numpy_random = np.random.RandomState(seed)
    samples = problem.nominal.sample(
        initial_samples,
        numpy_random=numpy_random,
    )
    if constrained:
        proposal: GaussianMixtureProposal = ConstrainedGaussianMixture.from_samples(
            samples,
            components=components,
            nominal_covariance=problem.nominal.covariance,
            beta=beta,
            random_state=numpy_random,
        )
    else:
        proposal = UnconstrainedGaussianMixture.from_samples(
            samples,
            components=components,
            random_state=numpy_random,
        )
    consume_authoritative_constructor_draws(
        torch_generator,
        components=components,
        dimension=problem.dimension,
    )
    return ProposalInitialization(proposal, torch_generator, numpy_random)


def warm_start_from_proposal(
    proposal: GaussianMixtureProposal,
    *,
    seed: int,
    initial_samples: int,
) -> ProposalInitialization:
    """Reproduce target construction followed by authoritative warm starting."""

    if initial_samples < proposal.n_components:
        raise ValueError("Warm start needs one sample per component")
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed)
    numpy_random = np.random.RandomState(seed)
    initialization = proposal.sample(initial_samples, generator=torch_generator)
    if type(proposal) is ConstrainedGaussianMixture:
        ConstrainedGaussianMixture.from_samples(
            initialization,
            components=proposal.n_components,
            nominal_covariance=proposal.nominal_covariance,
            beta=proposal.beta,
            random_state=numpy_random,
        )
    elif type(proposal) is UnconstrainedGaussianMixture:
        UnconstrainedGaussianMixture.from_samples(
            initialization,
            components=proposal.n_components,
            random_state=numpy_random,
            covariance_scale=proposal.covariance_scale,
        )
    else:
        raise TypeError("KL refinement requires a trainable public GMM proposal")
    consume_authoritative_constructor_draws(
        torch_generator,
        components=proposal.n_components,
        dimension=proposal.dimension,
    )
    return ProposalInitialization(proposal.clone(), torch_generator, numpy_random)
