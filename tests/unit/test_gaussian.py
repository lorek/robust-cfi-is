import math

import numpy as np
import torch

from robust_cfi_is.distributions import GaussianDistribution


def test_standard_gaussian_log_prob_and_sampling():
    gaussian = GaussianDistribution(torch.zeros(2), torch.eye(2))
    log_probability = gaussian.log_prob(torch.zeros(1, 2))
    assert log_probability.dtype == torch.float64
    assert math.isclose(float(log_probability[0]), -math.log(2.0 * math.pi))

    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    samples = gaussian.sample(8, generator=first)
    assert samples.dtype == torch.float32
    assert torch.equal(samples, gaussian.sample(8, generator=second))

    numpy_first = np.random.RandomState(9)
    numpy_second = np.random.RandomState(9)
    assert torch.equal(
        gaussian.sample(8, numpy_random=numpy_first),
        gaussian.sample(8, numpy_random=numpy_second),
    )


def test_gaussian_rejects_non_spd_covariance():
    try:
        GaussianDistribution(torch.zeros(2), torch.tensor([[1.0, 2.0], [2.0, 1.0]]))
    except ValueError as error:
        assert "positive definite" in str(error)
    else:
        raise AssertionError("non-SPD covariance was accepted")
