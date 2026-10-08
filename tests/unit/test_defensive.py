import math

import numpy as np
import torch

from robust_cfi_is.benchmarks import get_benchmark
from robust_cfi_is.estimation import defensive_ordinary_is
from robust_cfi_is.methods import fit_defensive_reverse_kl, reverse_kl_surrogate
from robust_cfi_is.proposals import DefensiveMixtureProposal, UnconstrainedGaussianMixture


def _proposal(weight=0.25):
    problem = get_benchmark("g1_2d_Sigma1")
    adaptive = UnconstrainedGaussianMixture(
        logits=torch.tensor([0.0]),
        means=torch.tensor([[1.0, -1.0]]),
        factors=torch.eye(2).reshape(1, 2, 2),
    )
    return problem, DefensiveMixtureProposal(
        problem.nominal,
        adaptive,
        weight,
        numpy_random=np.random.RandomState(11),
    )


def test_full_defensive_density_and_exact_strata():
    problem, proposal = _proposal(0.25)
    samples = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    expected = torch.logaddexp(
        problem.nominal.log_prob(samples).to(torch.float32) + math.log(0.25),
        proposal.adaptive.log_prob(samples) + math.log(0.75),
    )
    torch.testing.assert_close(proposal.log_prob(samples), expected)
    nominal, adaptive = proposal.sample_stratified(
        20, generator=torch.Generator().manual_seed(7)
    )
    assert nominal.shape == (5, 2)
    assert adaptive.shape == (15, 2)


def test_defensive_strata_preserve_separate_numpy_and_torch_streams():
    problem, proposal = _proposal(0.25)
    torch_stream = torch.Generator().manual_seed(29)
    nominal, adaptive = proposal.sample_stratified(20, generator=torch_stream)
    expected_nominal = problem.nominal.sample(
        5, numpy_random=np.random.RandomState(11)
    )
    expected_adaptive = proposal.adaptive.sample(
        15, generator=torch.Generator().manual_seed(29)
    )
    assert torch.equal(nominal, expected_nominal)
    assert torch.equal(adaptive, expected_adaptive)


def test_reverse_kl_uses_full_defensive_density():
    problem, proposal = _proposal(0.4)
    samples = proposal.sample(12, generator=torch.Generator().manual_seed(4))
    full = proposal.log_prob(samples)
    loss, eta = reverse_kl_surrogate(
        full,
        problem.nominal.log_prob(samples).to(torch.float32),
        problem.event_score(samples),
        gamma=problem.gamma,
        temperature=0.25,
        baseline=0.0,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(eta).all()
    assert not torch.allclose(full, proposal.adaptive.log_prob(samples))


def test_defensive_fit_and_estimator_smoke():
    problem = get_benchmark("g1_2d_Sigma1")
    fitted = fit_defensive_reverse_kl(
        problem,
        components=2,
        defensive_weight=0.2,
        seed=13,
        initial_samples=80,
        training_samples=120,
        epochs=1,
    )
    assert isinstance(fitted.proposal, DefensiveMixtureProposal)
    assert len(fitted.steps) == 1
    result = defensive_ordinary_is(
        problem, fitted.proposal, samples=200, seed=19
    )
    assert result.nominal_count == 40
    assert result.adaptive_count == 160
    assert math.isfinite(result.estimate)
