import math

import torch

from robust_cfi_is.distributions import GaussianDistribution
from robust_cfi_is.methods.forward_kl import (
    mean_one_adaptive_ce_weights,
    mean_one_fixed_proposal_weights,
    weighted_forward_kl_loss,
)
from robust_cfi_is.methods.reverse_kl import reverse_kl_surrogate
from robust_cfi_is.proposals import (
    ConstrainedGaussianMixture,
    UnconstrainedGaussianMixture,
)


def _proposals():
    logits = torch.tensor([0.2, -0.3])
    means = torch.tensor([[-0.7, 0.1], [0.8, -0.2]])
    factors = torch.tensor(
        [
            [[0.8, 0.0], [0.1, 0.7]],
            [[0.6, 0.0], [-0.2, 0.9]],
        ]
    )
    return (
        UnconstrainedGaussianMixture(
            logits=logits,
            means=means,
            factors=factors,
        ),
        ConstrainedGaussianMixture(
            logits=logits,
            means=means,
            factors=factors,
            nominal_covariance=torch.eye(2),
            beta=0.5,
        ),
    )


def _adam_step(proposal, loss):
    proposal.enable_gradients()
    optimizer = torch.optim.Adam(proposal.trainable_tensors(), lr=0.003)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_forward_kl_one_step_matches_extracted_authoritative_equation():
    samples = torch.tensor(
        [[-1.1, 0.2], [-0.3, -0.5], [0.4, 0.6], [1.2, -0.1]]
    )
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    for public_initial in _proposals():
        reference = public_initial.clone()
        public = public_initial.clone()
        with torch.no_grad():
            log_f = nominal.log_prob(samples)
            log_generating = public_initial.log_prob(samples)
            public_weights = mean_one_adaptive_ce_weights(
                log_f,
                log_generating,
                output_dtype=samples.dtype,
            )
            ratio64 = log_f.to(torch.float64) - log_generating.to(torch.float64)
            reference_weights = torch.exp(
                ratio64
                - torch.logsumexp(ratio64, dim=0)
                + math.log(ratio64.numel())
            ).to(samples.dtype)
        public.enable_gradients()
        _adam_step(
            public,
            weighted_forward_kl_loss(public.log_prob(samples), public_weights),
        )
        reference.enable_gradients()
        _adam_step(
            reference,
            -torch.mean(reference.log_prob(samples) * reference_weights),
        )
        for actual, expected in zip(
            public.trainable_tensors(), reference.trainable_tensors()
        ):
            assert torch.equal(actual, expected)


def test_fixed_proposal_forward_kl_one_step_preserves_float64_weights():
    samples = torch.tensor(
        [[-1.1, 0.2], [-0.3, -0.5], [0.4, 0.6], [1.2, -0.1]]
    )
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    for public_initial in _proposals():
        reference = public_initial.clone()
        public = public_initial.clone()
        with torch.no_grad():
            log_f = nominal.log_prob(samples)
            log_generating = public_initial.log_prob(samples).to(torch.float64)
            public_weights = mean_one_fixed_proposal_weights(
                log_f,
                log_generating,
                output_dtype=torch.float64,
            )
            reference_ratio = log_f - log_generating
            reference_shifted = torch.exp(
                reference_ratio - reference_ratio.max()
            )
            reference_weights = reference_shifted / (
                reference_shifted.mean() + 1e-12
            )
        assert public_weights.dtype == torch.float64
        public.enable_gradients()
        _adam_step(
            public,
            weighted_forward_kl_loss(public.log_prob(samples), public_weights),
        )
        reference.enable_gradients()
        _adam_step(
            reference,
            -torch.mean(reference.log_prob(samples) * reference_weights),
        )
        for actual, expected in zip(
            public.trainable_tensors(), reference.trainable_tensors()
        ):
            assert torch.equal(actual, expected)


def test_reverse_kl_one_step_matches_extracted_authoritative_equation():
    samples = torch.tensor(
        [[-1.1, 0.2], [-0.3, -0.5], [0.4, 0.6], [1.2, -0.1]]
    )
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    scores = samples[:, 0] + 0.25 * samples[:, 1]
    tau = torch.tensor(0.2)
    baseline = 0.3
    for public_initial in _proposals():
        reference = public_initial.clone()
        public = public_initial.clone()
        public.enable_gradients()
        public_log_q = public.log_prob(samples)
        public_loss, _ = reverse_kl_surrogate(
            public_log_q,
            nominal.log_prob(samples).to(samples.dtype),
            scores,
            gamma=0.1,
            temperature=tau,
            baseline=baseline,
        )
        _adam_step(public, public_loss)

        reference.enable_gradients()
        reference_log_q = reference.log_prob(samples)
        # Exact equation extracted from train_vi.py lines 168--178.
        reference_log_s = -torch.nn.functional.softplus(
            -(scores - 0.1) / tau
        )
        reference_eta = (
            reference_log_q
            - nominal.log_prob(samples).to(samples.dtype)
            - reference_log_s
        ).detach()
        reference_loss = torch.mean(
            (1.0 + reference_eta - baseline) * reference_log_q
        )
        _adam_step(reference, reference_loss)
        for actual, expected in zip(
            public.trainable_tensors(), reference.trainable_tensors()
        ):
            assert torch.equal(actual, expected)


def test_reverse_nominal_log_prob_matches_authoritative_pdf_then_log_fixture():
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    samples = torch.tensor(
        [[-1.1, 0.2], [-0.3, -0.5], [0.4, 0.6], [1.2, -0.1]]
    )
    direct = nominal.log_prob(samples).to(samples.dtype)
    authoritative_shape = torch.log(torch.exp(nominal.log_prob(samples))).to(
        samples.dtype
    )
    assert torch.equal(direct, authoritative_shape)
