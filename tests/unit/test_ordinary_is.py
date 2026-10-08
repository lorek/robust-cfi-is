import numpy as np
import pytest
import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.distributions import GaussianDistribution
from robust_cfi_is.estimation import ordinary_is
from robust_cfi_is.proposals import (
    ConstrainedGaussianMixture,
    UnconstrainedGaussianMixture,
    certify_final_gmm,
)


def test_ordinary_is_is_unnormalized_and_estimates_halfspace_probability():
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    problem = RareEventProblem(
        name="unit_halfspace",
        dimension=2,
        gamma=0.0,
        nominal=nominal,
        event_score=lambda samples: samples[:, 0],
    )
    learned = ConstrainedGaussianMixture(
        logits=torch.zeros(1),
        means=torch.zeros(1, 2),
        factors=(0.5**0.5) * torch.eye(2).unsqueeze(0),
        nominal_covariance=torch.eye(2),
        beta=0.5,
    )
    proposal = certify_final_gmm(learned)
    result = ordinary_is(problem, proposal, samples=10000, seed=5)
    assert result.estimator == "ordinary_unnormalized_importance_sampling"
    assert not result.self_normalized
    assert abs(result.estimate - 0.5) < 0.025
    assert result.standard_error > 0.0
    assert result.cov is not None
    assert 4500 < result.event_count < 5500


def test_ordinary_is_rejects_uncertified_constrained_proposal():
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    problem = RareEventProblem(
        name="unit_halfspace",
        dimension=2,
        gamma=0.0,
        nominal=nominal,
        event_score=lambda samples: samples[:, 0],
    )
    learned = ConstrainedGaussianMixture(
        logits=torch.zeros(1),
        means=torch.zeros(1, 2),
        factors=(0.5**0.5) * torch.eye(2).unsqueeze(0),
        nominal_covariance=torch.eye(2),
        beta=0.5,
    )
    with pytest.raises(TypeError, match="certified constrained"):
        ordinary_is(problem, learned, samples=10, seed=5)


def test_ordinary_is_accepts_unconstrained_public_proposal():
    nominal = GaussianDistribution(torch.zeros(2), torch.eye(2))
    problem = RareEventProblem(
        name="unit_halfspace",
        dimension=2,
        gamma=0.0,
        nominal=nominal,
        event_score=lambda samples: samples[:, 0],
    )
    proposal = UnconstrainedGaussianMixture.from_samples(
        nominal.sample(1000, numpy_random=np.random.RandomState(3)),
        components=1,
        random_state=np.random.RandomState(3),
    )
    result = ordinary_is(problem, proposal, samples=1000, seed=5)
    assert 0.4 < result.estimate < 0.6
