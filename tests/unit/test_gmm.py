import numpy as np
import torch

from robust_cfi_is.proposals import (
    ConstrainedGaussianMixture,
    UnconstrainedGaussianMixture,
)


def _proposal() -> ConstrainedGaussianMixture:
    return ConstrainedGaussianMixture(
        logits=torch.tensor([0.0, 0.0]),
        means=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        factors=torch.eye(2).repeat(2, 1, 1),
        nominal_covariance=torch.eye(2),
        beta=0.5,
    )


def test_constrained_covariance_is_remainder_plus_nominal_floor():
    proposal = _proposal()
    expected = 1.5 * torch.eye(2).repeat(2, 1, 1)
    assert torch.allclose(proposal.covariances(), expected)
    assert torch.allclose(proposal.weights, torch.tensor([0.5, 0.5]))


def test_gmm_sampling_and_log_density_are_finite():
    proposal = _proposal()
    samples = proposal.sample(200, generator=torch.Generator().manual_seed(11))
    log_prob = proposal.log_prob(samples)
    assert samples.shape == (200, 2)
    assert log_prob.shape == (200,)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(log_prob).all()


def test_unconstrained_gmm_covariance_has_no_nominal_floor():
    factor = torch.diag(torch.tensor([0.2, 0.3])).repeat(2, 1, 1)
    proposal = UnconstrainedGaussianMixture(
        logits=torch.tensor([0.0, 0.0]),
        means=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        factors=factor,
    )
    expected = torch.diag(torch.tensor([0.04, 0.09])).repeat(2, 1, 1)
    assert torch.allclose(proposal.covariances(), expected)
    assert torch.linalg.eigvalsh(proposal.covariances()).min() < 0.5
    samples = proposal.sample(100, generator=torch.Generator().manual_seed(7))
    assert torch.isfinite(proposal.log_prob(samples)).all()


def test_unconstrained_fit_is_deterministic_for_fixed_numpy_state():
    generator = torch.Generator().manual_seed(8)
    samples = torch.cat(
        [
            -2.0 + 0.1 * torch.randn(50, 2, generator=generator),
            2.0 + 0.1 * torch.randn(50, 2, generator=generator),
        ]
    )
    first = UnconstrainedGaussianMixture.from_samples(
        samples,
        components=2,
        random_state=np.random.RandomState(19),
    )
    second = UnconstrainedGaussianMixture.from_samples(
        samples,
        components=2,
        random_state=np.random.RandomState(19),
    )
    assert torch.equal(first.means, second.means)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.factors, second.factors)


def test_unconstrained_density_preserves_authoritative_jitter_for_singular_factor():
    proposal = UnconstrainedGaussianMixture(
        logits=torch.tensor([0.0]),
        means=torch.tensor([[0.0, 0.0]]),
        factors=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    )
    log_prob = proposal.log_prob(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert torch.isfinite(log_prob).all()
