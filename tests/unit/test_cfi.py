import torch

from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.distributions import GaussianDistribution
from robust_cfi_is.methods import fit_cfi
from robust_cfi_is.proposals import UnconstrainedGaussianMixture


def test_cfi_performs_unweighted_elite_fit_and_reaches_easy_target():
    problem = RareEventProblem(
        name="unit_halfspace",
        dimension=2,
        gamma=0.5,
        nominal=GaussianDistribution(torch.zeros(2), torch.eye(2)),
        event_score=lambda samples: samples[:, 0],
    )
    result = fit_cfi(
        problem,
        components=2,
        seed=19,
        initial_samples=300,
        samples_per_level=500,
        rho=0.2,
        max_levels=3,
        smoothing=0.75,
    )
    assert result.target_reached
    assert len(result.levels) == 1
    assert result.levels[0].elite_count >= 2
    assert result.proposal.beta == 0.5
    assert result.final_fit_elite_count >= 2
    assert torch.isfinite(result.proposal.covariances()).all()


def test_cfi_u_reuses_elite_algorithm_without_covariance_floor():
    problem = RareEventProblem(
        name="unit_halfspace",
        dimension=2,
        gamma=0.5,
        nominal=GaussianDistribution(torch.zeros(2), torch.eye(2)),
        event_score=lambda samples: samples[:, 0],
    )
    result = fit_cfi(
        problem,
        components=2,
        constrained=False,
        seed=19,
        initial_samples=300,
        samples_per_level=500,
        rho=0.2,
        max_levels=3,
        smoothing=0.75,
    )
    assert result.target_reached
    assert isinstance(result.proposal, UnconstrainedGaussianMixture)
    assert all(level.beta is None for level in result.levels)
    assert torch.isfinite(result.proposal.covariances()).all()
