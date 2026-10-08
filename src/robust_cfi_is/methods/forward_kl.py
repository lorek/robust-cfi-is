"""Forward-KL fitting through weighted cross entropy.

The optimized direction is ``KL(h_opt || q_theta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.methods._proposal import fit_initial_proposal, warm_start_from_proposal
from robust_cfi_is.proposals import GaussianMixtureProposal


@dataclass(frozen=True)
class ForwardKLLevel:
    index: int
    gamma: float
    elite_count: int
    weight_minimum: float
    weight_maximum: float
    target_reached: bool


@dataclass(frozen=True)
class ForwardKLFitResult:
    proposal: GaussianMixtureProposal
    levels: tuple[ForwardKLLevel, ...]
    target_reached: bool
    stopped_early: bool = False


def mean_one_adaptive_ce_weights(
    log_target: torch.Tensor,
    log_generating_proposal: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Corrected stable ``f/q`` weights for standalone adaptive forward KL."""

    proposal64 = torch.as_tensor(log_generating_proposal).to(torch.float64).reshape(-1)
    target64 = torch.as_tensor(log_target, device=proposal64.device).to(
        torch.float64
    ).reshape(-1)
    if target64.shape != proposal64.shape or target64.numel() == 0:
        raise ValueError("Forward-KL log densities need the same non-empty shape")
    log_weights = target64 - proposal64
    if not bool(torch.isfinite(log_weights).all()):
        raise ValueError("Forward-KL log importance weights must be finite")
    normalized = (
        log_weights
        - torch.logsumexp(log_weights, dim=0)
        + math.log(log_weights.numel())
    )
    weights = torch.exp(normalized).to(output_dtype)
    if not bool(torch.isfinite(weights).all()) or not bool(weights.sum() > 0):
        raise ValueError("Forward-KL normalized weights must be finite and positive")
    return weights


def mean_one_fixed_proposal_weights(
    log_target: torch.Tensor,
    log_generating_proposal: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Project convention used by the fixed-proposal weighted-CE refinement."""

    proposal64 = torch.as_tensor(log_generating_proposal).to(torch.float64).reshape(-1)
    target64 = torch.as_tensor(log_target, device=proposal64.device).to(
        torch.float64
    ).reshape(-1)
    if target64.shape != proposal64.shape or target64.numel() == 0:
        raise ValueError("Forward-KL log densities need the same non-empty shape")
    log_weights = target64 - proposal64
    if not bool(torch.isfinite(log_weights).all()):
        raise ValueError("Forward-KL log importance weights must be finite")
    shifted = log_weights - log_weights.max()
    weights64 = torch.exp(shifted)
    weights64 = weights64 / (weights64.mean() + 1e-12)
    weights = weights64.to(output_dtype)
    if not bool(torch.isfinite(weights).all()) or not bool(weights.sum() > 0):
        raise ValueError("Forward-KL normalized weights must be finite and positive")
    return weights


def weighted_forward_kl_loss(
    model_log_prob: torch.Tensor,
    fixed_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted-CE surrogate for ``KL(h_opt || q_theta)``."""

    if model_log_prob.shape != fixed_weights.shape or model_log_prob.numel() == 0:
        raise ValueError("Forward-KL loss inputs need the same non-empty shape")
    return -torch.mean(model_log_prob * fixed_weights)


def _optimizer(proposal: GaussianMixtureProposal, learning_rate: float):
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    proposal.enable_gradients()
    return torch.optim.Adam(proposal.trainable_tensors(), lr=learning_rate)


def _fit_adaptive_forward_kl(
    problem: RareEventProblem,
    proposal: GaussianMixtureProposal,
    *,
    generator: torch.Generator,
    samples_per_level: int,
    rho: float,
    max_levels: int,
    epochs_per_level: int,
    learning_rate: float,
) -> ForwardKLFitResult:
    if samples_per_level < proposal.n_components:
        raise ValueError("Forward KL needs one sample per component")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    if max_levels < 1 or epochs_per_level < 1:
        raise ValueError("Forward-KL levels and epochs must be positive")
    optimizer = _optimizer(proposal, learning_rate)
    levels: list[ForwardKLLevel] = []
    target_reached = False

    for level in range(max_levels):
        with torch.no_grad():
            samples = proposal.sample(samples_per_level, generator=generator)
            values = problem.event_score(samples)
            gamma_value = float(torch.quantile(values, 1.0 - rho))
            target_reached = gamma_value > problem.gamma
            if target_reached:
                gamma_value = float(problem.gamma)
            elite = samples[values >= gamma_value]
            log_generating = proposal.log_prob(elite).detach()
            log_target = problem.nominal.log_prob(elite).detach()
            fixed_weights = mean_one_adaptive_ce_weights(
                log_target,
                log_generating,
                output_dtype=elite.dtype,
            )

        for _ in range(epochs_per_level):
            loss = weighted_forward_kl_loss(
                proposal.log_prob(elite),
                fixed_weights,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        levels.append(
            ForwardKLLevel(
                index=level,
                gamma=gamma_value,
                elite_count=int(elite.shape[0]),
                weight_minimum=float(fixed_weights.min()),
                weight_maximum=float(fixed_weights.max()),
                target_reached=target_reached,
            )
        )
        if target_reached:
            break
    return ForwardKLFitResult(proposal, tuple(levels), target_reached)


def fit_forward_kl(
    problem: RareEventProblem,
    *,
    components: int,
    constrained: bool,
    seed: int,
    initial_samples: int,
    samples_per_level: int,
    rho: float,
    max_levels: int,
    epochs_per_level: int,
    learning_rate: float = 0.01,
    beta: float = 0.5,
) -> ForwardKLFitResult:
    """Fit standalone forward KL with corrected adaptive log weights."""

    initialized = fit_initial_proposal(
        problem,
        components=components,
        constrained=constrained,
        seed=seed,
        initial_samples=initial_samples,
        beta=beta,
    )
    return _fit_adaptive_forward_kl(
        problem,
        initialized.proposal,
        generator=initialized.torch_generator,
        samples_per_level=samples_per_level,
        rho=rho,
        max_levels=max_levels,
        epochs_per_level=epochs_per_level,
        learning_rate=learning_rate,
    )


def refine_forward_kl(
    problem: RareEventProblem,
    generating_proposal: GaussianMixtureProposal,
    *,
    seed: int,
    initial_samples: int,
    training_samples: int,
    epochs: int,
    learning_rate: float = 0.01,
) -> ForwardKLFitResult:
    """Refine from a fixed proposal using source-project weighted CE."""

    if training_samples < generating_proposal.n_components or epochs < 1:
        raise ValueError("Forward-KL refinement budgets are too small")
    initialized = warm_start_from_proposal(
        generating_proposal,
        seed=seed,
        initial_samples=initial_samples,
    )
    proposal = initialized.proposal
    generator = initialized.torch_generator
    with torch.no_grad():
        samples = generating_proposal.sample(training_samples, generator=generator)
        validation = generating_proposal.sample(
            max(1, training_samples // 5),
            generator=generator,
        )
        elite = samples[problem.event_score(samples) >= problem.gamma]
        elite_validation = validation[
            problem.event_score(validation) >= problem.gamma
        ]
        if elite.shape[0] == 0 or elite_validation.shape[0] == 0:
            raise RuntimeError("Generating proposal produced no target-event samples")
        fixed_weights = mean_one_fixed_proposal_weights(
            problem.nominal.log_prob(elite),
            generating_proposal.log_prob(elite).detach(),
            output_dtype=torch.float64,
        )
        fixed_validation_weights = mean_one_fixed_proposal_weights(
            problem.nominal.log_prob(elite_validation),
            generating_proposal.log_prob(elite_validation).detach(),
            output_dtype=torch.float64,
        )

    optimizer = _optimizer(proposal, learning_rate)
    best_model = proposal.clone()
    best_validation = float("inf")
    best_epoch = 0
    mercy = 100
    stopped_early = False
    for epoch in range(epochs):
        loss = weighted_forward_kl_loss(proposal.log_prob(elite), fixed_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss = weighted_forward_kl_loss(
                proposal.log_prob(elite_validation),
                fixed_validation_weights,
            )
            value = float(validation_loss)
        if value < best_validation:
            best_validation = value
            best_epoch = epoch
            best_model = proposal.clone()
        elif epoch - best_epoch > mercy:
            proposal = best_model
            stopped_early = True
            break

    level = ForwardKLLevel(
        index=0,
        gamma=float(problem.gamma),
        elite_count=int(elite.shape[0]),
        weight_minimum=float(fixed_weights.min()),
        weight_maximum=float(fixed_weights.max()),
        target_reached=True,
    )
    return ForwardKLFitResult(
        proposal=proposal,
        levels=(level,),
        target_reached=True,
        stopped_early=stopped_early,
    )
