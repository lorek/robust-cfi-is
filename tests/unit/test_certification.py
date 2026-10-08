import torch

from robust_cfi_is.proposals import (
    TAU_ACCEPT,
    ConstrainedGaussianMixture,
    certify_final_gmm,
)


def test_final_proposal_is_built_and_certified_from_canonical_factor():
    learned = ConstrainedGaussianMixture(
        logits=torch.zeros(2),
        means=torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
        factors=torch.zeros(2, 2, 2),
        nominal_covariance=torch.eye(2),
        beta=0.5,
    )
    final = certify_final_gmm(learned)
    assert final.certification["passed"]
    assert final.certification["minimum_lambda"] >= TAU_ACCEPT
    assert final.certification["minimum_theorem_margin"] > 0.0
    assert all(row["changed"] for row in final.certification["transforms"])
    factors = final.scale_tril()
    assert torch.count_nonzero(torch.triu(factors, diagonal=1)) == 0
    assert torch.all(torch.diagonal(factors, dim1=-2, dim2=-1) > 0)


def test_certified_sampling_and_density_use_canonical_factor():
    learned = ConstrainedGaussianMixture(
        logits=torch.tensor([0.0]),
        means=torch.zeros(1, 2),
        factors=(0.5**0.5) * torch.eye(2).unsqueeze(0),
        nominal_covariance=torch.eye(2),
        beta=0.5,
    )
    final = certify_final_gmm(learned)
    samples = final.sample(100, generator=torch.Generator().manual_seed(3))
    assert samples.dtype == torch.float32
    assert torch.isfinite(final.log_prob(samples)).all()
