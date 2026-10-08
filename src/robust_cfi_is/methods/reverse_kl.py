"""Reverse-KL fitting with the authoritative score-function surrogate.

The optimized direction is ``KL(q_theta || h_opt)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.methods._proposal import fit_initial_proposal, warm_start_from_proposal
from robust_cfi_is.proposals import DefensiveMixtureProposal, GaussianMixtureProposal

TrainableReverseKLProposal = GaussianMixtureProposal | DefensiveMixtureProposal


@dataclass(frozen=True)
class ReverseKLStep:
    index: int
    loss: float
    mean_log_ratio: float
    temperature: float
    baseline: float


@dataclass(frozen=True)
class ReverseKLFitResult:
    proposal: TrainableReverseKLProposal
    steps: tuple[ReverseKLStep, ...]


def log_sigmoid_stable(value: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.softplus(-value)


def temperature_from_batch(
    event_margin: torch.Tensor,
    *,
    fallback: float | None,
    scale: float = 0.3,
) -> torch.Tensor:
    if event_margin.numel() < 10:
        value = fallback if fallback is not None else 1e-6
        return torch.tensor(value, dtype=event_margin.dtype, device=event_margin.device)
    return (scale * torch.std(event_margin)).clamp(min=0.02, max=0.3)


def reverse_kl_surrogate(
    log_proposal: torch.Tensor,
    log_nominal: torch.Tensor,
    event_scores: torch.Tensor,
    *,
    gamma: float,
    temperature: torch.Tensor | float,
    baseline: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reverse-KL loss and detached log-ratio diagnostic."""

    if not (
        log_proposal.shape == log_nominal.shape == event_scores.shape
        and log_proposal.numel() > 0
    ):
        raise ValueError("Reverse-KL inputs need the same non-empty shape")
    tau = torch.as_tensor(
        temperature,
        dtype=event_scores.dtype,
        device=event_scores.device,
    )
    log_gate = log_sigmoid_stable((event_scores - gamma) / tau)
    eta = (log_proposal - log_nominal - log_gate).detach()
    gradient_weights = 1.0 + eta - baseline
    return torch.mean(gradient_weights * log_proposal), eta


def _optimize_reverse_kl(
    problem: RareEventProblem,
    proposal: TrainableReverseKLProposal,
    *,
    generator: torch.Generator,
    training_samples: int,
    epochs: int,
    learning_rate: float,
    tau: float | None,
    baseline_rate: float,
    smoothing: float,
) -> ReverseKLFitResult:
    if training_samples < proposal.n_components or epochs < 1:
        raise ValueError("Reverse-KL budgets are too small")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= baseline_rate <= 1.0:
        raise ValueError("baseline_rate must lie between zero and one")
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must lie between zero and one")

    proposal.enable_gradients()
    optimizer = torch.optim.Adam(proposal.trainable_tensors(), lr=learning_rate)
    baseline = 0.0
    records: list[ReverseKLStep] = []

    # Retain the two pre-loop draws whose RNG consumption is part of the
    # authoritative deterministic trajectory.
    with torch.no_grad():
        initial_check = proposal.sample(training_samples, generator=generator)
        proposal.log_prob(initial_check)
        width_batch = proposal.sample(training_samples, generator=generator)
        torch.std(problem.event_score(width_batch) - problem.gamma)

    for index in range(epochs):
        with torch.no_grad():
            samples = proposal.sample(training_samples, generator=generator)
        log_proposal = proposal.log_prob(samples)
        # This is the same Gaussian log density as the authoritative pdf-then-
        # log expression, without materializing the intermediate probability.
        log_nominal = problem.nominal.log_prob(samples).to(samples.dtype)
        scores = problem.event_score(samples)
        temperature = temperature_from_batch(
            scores - problem.gamma,
            fallback=tau,
        )
        loss, eta = reverse_kl_surrogate(
            log_proposal,
            log_nominal,
            scores,
            gamma=problem.gamma,
            temperature=temperature,
            baseline=baseline,
        )
        next_baseline = (
            (1.0 - baseline_rate) * baseline
            + baseline_rate * float(eta.mean())
        )
        previous = proposal.clone()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if smoothing > 0.0:
            proposal.smooth_from_previous(
                previous,
                new_fraction=1.0 - smoothing,
            )
        baseline = next_baseline
        records.append(
            ReverseKLStep(
                index=index,
                loss=float(loss.detach()),
                mean_log_ratio=float(eta.mean()),
                temperature=float(temperature),
                baseline=baseline,
            )
        )
    return ReverseKLFitResult(proposal, tuple(records))


def fit_reverse_kl(
    problem: RareEventProblem,
    *,
    components: int,
    constrained: bool,
    seed: int,
    initial_samples: int,
    training_samples: int,
    epochs: int,
    learning_rate: float = 0.01,
    tau: float | None = 0.25,
    baseline_rate: float = 0.5,
    smoothing: float = 0.5,
    beta: float = 0.5,
) -> ReverseKLFitResult:
    initialized = fit_initial_proposal(
        problem,
        components=components,
        constrained=constrained,
        seed=seed,
        initial_samples=initial_samples,
        beta=beta,
    )
    return _optimize_reverse_kl(
        problem,
        initialized.proposal,
        generator=initialized.torch_generator,
        training_samples=training_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        tau=tau,
        baseline_rate=baseline_rate,
        smoothing=smoothing,
    )


def refine_reverse_kl(
    problem: RareEventProblem,
    proposal: GaussianMixtureProposal,
    *,
    seed: int,
    initial_samples: int,
    training_samples: int,
    epochs: int,
    learning_rate: float = 0.01,
    tau: float | None = 0.25,
    baseline_rate: float = 0.5,
    smoothing: float = 0.5,
) -> ReverseKLFitResult:
    initialized = warm_start_from_proposal(
        proposal,
        seed=seed,
        initial_samples=initial_samples,
    )
    return _optimize_reverse_kl(
        problem,
        initialized.proposal,
        generator=initialized.torch_generator,
        training_samples=training_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        tau=tau,
        baseline_rate=baseline_rate,
        smoothing=smoothing,
    )
