import torch

from robust_cfi_is.benchmarks import get_benchmark, g1_2d


def test_g1_definition_and_registry():
    samples = torch.tensor([[0.0, 0.0], [3.8, 3.8]], dtype=torch.float32)
    values = g1_2d(samples)
    assert torch.allclose(values, torch.tensor([-27.88, 1.0]), atol=1e-5)
    problem = get_benchmark("g1_2d_Sigma1")
    assert problem.dimension == 2
    assert problem.gamma == 0.0
    assert torch.equal(problem.nominal.covariance, torch.eye(2))
    assert problem.indicator(samples).tolist() == [False, True]


def test_g1_rejects_wrong_dimension():
    try:
        g1_2d(torch.zeros(3, 3))
    except ValueError as error:
        assert "shape (n, 2)" in str(error)
    else:
        raise AssertionError("wrong-dimensional g1 input was accepted")
