import torch

from robust_cfi_is.benchmarks import get_benchmark


def test_public_analytic_benchmark_registry():
    expected = {
        "g1_2d_Sigma1": (2, 0.0),
        "g2b_papaioannou_2d_Sigma1": (2, 0.0),
        "g3_2d_Sigma1": (2, 0.0),
        "g5_powell_40d_Sigma1": (40, -403.66366875),
        "g6_papaioannou_100d_Sigma1": (100, 3.5),
        "s10_sum_exp_gamma_20k_Sigma2": (10, 20000.0),
        "s10_sum_exp_gamma_40k_Sigma2": (10, 40000.0),
        "s10_sum_exp_gamma_500k_Sigma2": (10, 500000.0),
    }
    for name, (dimension, gamma) in expected.items():
        problem = get_benchmark(name)
        assert problem.dimension == dimension
        assert problem.gamma == gamma
        values = problem.event_score(torch.zeros(2, dimension))
        assert values.shape == (2,)


def test_g10_nominal_matches_public_problem_definition():
    problem = get_benchmark("s10_sum_exp_gamma_40k_Sigma2")
    torch.testing.assert_close(problem.nominal.mean, torch.arange(-9.0, 1.0))
    torch.testing.assert_close(
        problem.nominal.covariance,
        torch.diag(torch.arange(1.0, 11.0)),
    )
