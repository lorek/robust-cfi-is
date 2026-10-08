"""Fixed-proposal weighted-CE refinement for the public RealNVP methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.methods.forward_kl import (
    mean_one_fixed_proposal_weights,
    weighted_forward_kl_loss,
)
from robust_cfi_is.proposals import GaussianMixtureProposal
from robust_cfi_is.proposals.realnvp import RealNVPProposal


@dataclass(frozen=True)
class FlowFitResult:
    proposal: RealNVPProposal
    elite_count: int
    validation_elite_count: int
    best_epoch: int
    stopped_early: bool


def fit_fixed_proposal_flow(
    problem: RareEventProblem,
    generating_proposal: GaussianMixtureProposal,
    *,
    seed: int,
    initial_samples: int,
    training_samples: int,
    epochs: int,
    learning_rate: float,
    hidden_dimension: int,
    coupling_layers: int,
    device: str = "cpu",
) -> FlowFitResult:
    """Match the source fixed-proposal sampling, weighting, and early stopping."""
    if initial_samples < 1 or training_samples < 1 or epochs < 1:
        raise ValueError("flow fitting budgets must be positive")
    if learning_rate <= 0:
        raise ValueError("flow learning_rate must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # The source constructs the RealNVP after drawing its no-op init_samples.
    generating_proposal.sample(initial_samples, generator=generator)
    proposal = RealNVPProposal(
        generating_proposal,
        hidden_dimension=hidden_dimension,
        coupling_layers=coupling_layers,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        samples = generating_proposal.sample(training_samples, generator=generator)
        validation = generating_proposal.sample(
            max(1, training_samples // 5), generator=generator
        )
        elite = samples[problem.event_score(samples) >= problem.gamma]
        elite_validation = validation[
            problem.event_score(validation) >= problem.gamma
        ]
        if elite.shape[0] == 0 or elite_validation.shape[0] == 0:
            raise RuntimeError("Generating proposal produced no target-event samples")
        weights = mean_one_fixed_proposal_weights(
            problem.nominal.log_prob(elite),
            generating_proposal.log_prob(elite),
            output_dtype=torch.float64,
        ).to(device)
        validation_weights = mean_one_fixed_proposal_weights(
            problem.nominal.log_prob(elite_validation),
            generating_proposal.log_prob(elite_validation),
            output_dtype=torch.float64,
        ).to(device)
    elite = elite.to(device)
    elite_validation = elite_validation.to(device)
    optimizer = torch.optim.Adam(proposal.parameters(), lr=learning_rate)
    best = proposal.clone()
    best_value = float("inf")
    best_epoch = 0
    stopped_early = False
    for epoch in range(epochs):
        loss = weighted_forward_kl_loss(proposal.log_prob(elite), weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            value = float(weighted_forward_kl_loss(
                proposal.log_prob(elite_validation), validation_weights
            ))
        if value < best_value:
            best_value = value
            best_epoch = epoch
            best = proposal.clone()
        elif epoch - best_epoch > 100:
            stopped_early = True
            proposal = best
            break
    return FlowFitResult(
        proposal,
        int(elite.shape[0]),
        int(elite_validation.shape[0]),
        best_epoch,
        stopped_early,
    )
