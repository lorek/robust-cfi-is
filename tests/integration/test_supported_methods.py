import pytest
import torch

from robust_cfi_is.api import fit_public_method
from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.distributions import GaussianDistribution
from robust_cfi_is.proposals import (
    ConstrainedGaussianMixture,
    UnconstrainedGaussianMixture,
    RealNVPProposal,
)


METHODS = (
    "frkl_u",
    "ffkl_u",
    "frkl_c",
    "ffkl_c",
    "cfi_u",
    "cfi_c",
    "cfi_c_frkl_c",
    "cfi_c_ffkl_c",
)


def _problem() -> RareEventProblem:
    return RareEventProblem(
        name="bounded_halfspace",
        dimension=2,
        gamma=-0.25,
        nominal=GaussianDistribution(torch.zeros(2), torch.eye(2)),
        event_score=lambda samples: samples[:, 0],
    )


def _fit(method: str):
    return fit_public_method(
        _problem(),
        method,
        components=2,
        seed=17,
        initial_samples=160,
        samples_per_level=240,
        rho=0.25,
        max_levels=2,
        smoothing=0.5,
        epochs=2,
        learning_rate=0.002,
        tau=0.25,
        baseline_rate=0.5,
    )


@pytest.mark.parametrize("method", METHODS)
def test_all_public_methods_have_bounded_finite_smokes(method):
    fitted = _fit(method)
    expected_type = (
        ConstrainedGaussianMixture
        if fitted.method.constrained
        else UnconstrainedGaussianMixture
    )
    assert isinstance(fitted.proposal, expected_type)
    assert torch.isfinite(fitted.proposal.means).all()
    assert torch.isfinite(fitted.proposal.logits).all()
    assert torch.isfinite(fitted.proposal.covariances()).all()
    samples = fitted.proposal.sample(
        20,
        generator=torch.Generator().manual_seed(99),
    )
    assert torch.isfinite(fitted.proposal.log_prob(samples)).all()


@pytest.mark.parametrize("method", METHODS)
def test_all_public_methods_are_deterministic(method):
    first = _fit(method).proposal
    second = _fit(method).proposal
    assert torch.equal(first.means, second.means)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.covariances(), second.covariances())


@pytest.mark.parametrize("method", ("cfi_c_frkl_c_flow", "cfi_c_ffkl_c_flow"))
def test_public_flow_variants_have_bounded_executable_smokes(method):
    fitted = fit_public_method(
        _problem(), method, components=2, seed=17, initial_samples=80,
        samples_per_level=120, rho=0.25, max_levels=2, smoothing=0.5,
        epochs=1, learning_rate=0.002, tau=0.25, baseline_rate=0.5,
        flow_hidden_dimension=4, flow_coupling_layers=2,
        flow_learning_rate=1e-4, flow_initial_samples=80,
        flow_training_samples=120, flow_epochs=1, flow_device="cpu",
    )
    assert isinstance(fitted.proposal, RealNVPProposal)
    assert fitted.proposal.theorem_certified_final_proposal is False
    samples = fitted.proposal.sample(12, generator=torch.Generator().manual_seed(31))
    assert torch.isfinite(fitted.proposal.log_prob(samples)).all()
