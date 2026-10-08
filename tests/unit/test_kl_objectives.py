import math

import torch

from robust_cfi_is.methods.forward_kl import (
    mean_one_adaptive_ce_weights,
    mean_one_fixed_proposal_weights,
    weighted_forward_kl_loss,
)
from robust_cfi_is.methods.reverse_kl import (
    log_sigmoid_stable,
    reverse_kl_surrogate,
    temperature_from_batch,
)


def test_adaptive_forward_kl_matches_authoritative_logsumexp_equation():
    log_f = torch.tensor([-1000.0, -7.0, 2.0], dtype=torch.float64)
    log_q = torch.tensor([3.0, -9.0, -4.0], dtype=torch.float64)
    actual = mean_one_adaptive_ce_weights(
        log_f,
        log_q,
        output_dtype=torch.float32,
    )
    log_ratio = log_f - log_q
    expected = torch.exp(
        log_ratio - torch.logsumexp(log_ratio, dim=0) + math.log(3)
    ).to(torch.float32)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.mean(), torch.tensor(1.0))


def test_fixed_proposal_forward_kl_matches_authoritative_shift_equation():
    log_f = torch.tensor([-13.0, -2.0, 4.0], dtype=torch.float64)
    log_q = torch.tensor([-10.0, -8.0, 1.0], dtype=torch.float64)
    actual = mean_one_fixed_proposal_weights(
        log_f,
        log_q,
        output_dtype=torch.float32,
    )
    log_ratio = log_f - log_q
    shifted = torch.exp(log_ratio - log_ratio.max())
    expected = (shifted / (shifted.mean() + 1e-12)).to(torch.float32)
    assert torch.equal(actual, expected)
    authoritative_dtype = mean_one_fixed_proposal_weights(
        log_f,
        log_q,
        output_dtype=torch.float64,
    )
    assert authoritative_dtype.dtype == torch.float64


def test_forward_kl_is_weighted_cross_entropy_not_reverse_surrogate():
    log_q = torch.tensor([-3.0, -1.0, -2.0], requires_grad=True)
    weights = torch.tensor([0.25, 2.0, 0.75])
    loss = weighted_forward_kl_loss(log_q, weights)
    loss.backward()
    assert torch.equal(log_q.grad, -weights / 3.0)
    assert torch.equal(loss, -torch.mean(log_q.detach() * weights))


def test_reverse_kl_matches_authoritative_score_surrogate_equation():
    log_q = torch.tensor([-2.0, -1.0, -4.0], requires_grad=True)
    log_f = torch.tensor([-1.5, -2.5, -3.0])
    scores = torch.tensor([-0.5, 0.25, 1.0])
    gamma = 0.1
    tau = torch.tensor(0.2)
    baseline = -0.4
    loss, eta = reverse_kl_surrogate(
        log_q,
        log_f,
        scores,
        gamma=gamma,
        temperature=tau,
        baseline=baseline,
    )
    reference_log_gate = -torch.nn.functional.softplus(
        -(scores - gamma) / tau
    )
    reference_eta = (log_q - log_f - reference_log_gate).detach()
    reference_weights = 1.0 + reference_eta - baseline
    reference_loss = torch.mean(reference_weights * log_q)
    assert torch.equal(eta, reference_eta)
    assert torch.equal(loss, reference_loss)
    loss.backward()
    assert torch.allclose(log_q.grad, reference_weights / 3.0)


def test_reverse_and_forward_objectives_cannot_be_swapped():
    log_q = torch.tensor([-2.0, -1.0, -4.0])
    log_f = torch.tensor([-1.5, -2.5, -3.0])
    scores = torch.tensor([-0.5, 0.25, 1.0])
    forward_weights = mean_one_adaptive_ce_weights(
        log_f,
        log_q,
        output_dtype=torch.float32,
    )
    _, reverse_eta = reverse_kl_surrogate(
        log_q,
        log_f,
        scores,
        gamma=0.1,
        temperature=0.2,
        baseline=0.0,
    )
    assert not torch.allclose(forward_weights, 1.0 + reverse_eta)


def test_reverse_helpers_match_authoritative_stable_gate_and_tau_policy():
    values = torch.linspace(-2.0, 2.0, 20)
    expected_tau = (0.3 * torch.std(values)).clamp(min=0.02, max=0.3)
    assert torch.equal(temperature_from_batch(values, fallback=0.25), expected_tau)
    small = torch.tensor([-1.0, 1.0])
    assert float(temperature_from_batch(small, fallback=0.25)) == 0.25
    assert torch.allclose(
        log_sigmoid_stable(values),
        torch.log(torch.sigmoid(values)),
    )
