"""Reverse-KL fitting for the complete defensive proposal."""

from __future__ import annotations

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.methods._proposal import fit_initial_proposal
from robust_cfi_is.methods.reverse_kl import ReverseKLFitResult, _optimize_reverse_kl
from robust_cfi_is.proposals import DefensiveMixtureProposal


def fit_defensive_reverse_kl(
    problem: RareEventProblem,
    *,
    components: int,
    defensive_weight: float,
    seed: int,
    initial_samples: int,
    training_samples: int,
    epochs: int,
    learning_rate: float = 0.005,
    tau: float = 0.25,
    baseline_rate: float = 0.5,
    smoothing: float = 0.5,
) -> ReverseKLFitResult:
    """Fit fRKL-d using ``log(lambda*f + (1-lambda)*q_theta)``."""

    initialized = fit_initial_proposal(
        problem,
        components=components,
        constrained=False,
        seed=seed,
        initial_samples=initial_samples,
    )
    proposal = DefensiveMixtureProposal(
        problem.nominal,
        initialized.proposal,
        defensive_weight,
        numpy_random=initialized.numpy_random,
    )
    return _optimize_reverse_kl(
        problem,
        proposal,
        generator=initialized.torch_generator,
        training_samples=training_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        tau=tau,
        baseline_rate=baseline_rate,
        smoothing=smoothing,
    )
