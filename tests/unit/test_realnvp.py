import pytest
import torch
from torch import nn

from robust_cfi_is.benchmarks import get_benchmark
from robust_cfi_is.estimation import ordinary_is
from robust_cfi_is.methods import mean_one_fixed_proposal_weights, weighted_forward_kl_loss
from robust_cfi_is.naming import get_method
from robust_cfi_is.proposals import RealNVPProposal, UnconstrainedGaussianMixture


def _base(dimension=2):
    return UnconstrainedGaussianMixture(
        logits=torch.zeros(1), means=torch.zeros(1, dimension),
        factors=torch.eye(dimension).unsqueeze(0),
    )


def _flow(seed=7, dimension=2):
    generator = torch.Generator().manual_seed(seed)
    return RealNVPProposal(
        _base(dimension), hidden_dimension=5, coupling_layers=3,
        generator=generator, device="cpu",
    )


class _SourceAffine(nn.Module):
    """Literal bounded reference for the authoritative coupling equations."""
    def __init__(self, dimension, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, dimension), nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, dimension)
        )

    def forward(self, x):
        masked = x * self.mask
        scale = self.scale_net(masked) * (1 - self.mask)
        translation = self.translate_net(masked) * (1 - self.mask)
        return masked + (1 - self.mask) * (x * torch.exp(scale) + translation), scale.sum(1)


class _SourceFlow(nn.Module):
    def __init__(self, dimension, hidden, count):
        super().__init__()
        layers = []
        for index in range(count):
            mask = torch.ones(dimension); mask[::2] = 0
            if index % 2: mask = 1 - mask
            layers.append(_SourceAffine(dimension, hidden, mask))
        self.coupling_layers = nn.ModuleList(layers)
        for layer in self.coupling_layers:
            nn.init.zeros_(layer.scale_net[-2].bias)
            nn.init.normal_(layer.scale_net[-2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.translate_net[-1].bias)
            nn.init.normal_(layer.translate_net[-1].weight, mean=0.0, std=1e-3)

    def forward(self, x):
        total = torch.zeros(x.shape[0])
        for layer in self.coupling_layers:
            x, increment = layer(x); total = total + increment
        return x, total


def test_round_trip_logdet_density_and_jacobian():
    flow = _flow()
    x = torch.tensor([[0.2, -0.7], [1.0, 0.3]])
    y, forward_logdet = flow.forward(x)
    recovered, inverse_logdet = flow.inverse(y)
    torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(forward_logdet + inverse_logdet, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        flow.log_prob(x), flow.base_proposal.log_prob(y) + forward_logdet
    )
    one = x[:1].clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(lambda z: flow.forward(z)[0], one)
    matrix = jacobian[0, :, 0, :]
    expected = torch.linalg.slogdet(matrix)[1]
    torch.testing.assert_close(flow.forward(one)[1][0], expected, rtol=1e-4, atol=1e-5)


def test_deterministic_sampling_and_near_identity_initialization():
    first, second = _flow(11), _flow(11)
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    torch.testing.assert_close(first.sample(8, generator=g1), first.sample(8, generator=g2))
    for layer in first.coupling_layers:
        assert torch.count_nonzero(layer.scale_net[-2].bias) == 0
        assert torch.count_nonzero(layer.translate_net[-1].bias) == 0
        assert float(layer.scale_net[-2].weight.detach().std()) < 0.003


def test_fixed_weight_objective_and_one_step_source_parity():
    flow = _flow(19)
    saved = torch.random.get_rng_state()
    torch.manual_seed(19)
    reference = _SourceFlow(2, 5, 3)
    torch.random.set_rng_state(saved)
    for actual, expected_parameter in zip(flow.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected_parameter, rtol=0, atol=0)
    x = torch.tensor([[0.1, 0.2], [1.0, -0.5], [-0.2, 0.7]])
    weights = mean_one_fixed_proposal_weights(
        _base().log_prob(x), _base().log_prob(x) - torch.tensor([0.2, -0.1, 0.3]),
        output_dtype=torch.float64,
    )
    reference_values, reference_logdet = reference(x)
    expected_log_prob = flow.base_proposal.log_prob(reference_values) + reference_logdet
    expected = -torch.mean(expected_log_prob * weights)
    torch.testing.assert_close(weighted_forward_kl_loss(flow.log_prob(x), weights), expected)
    opt_a = torch.optim.Adam(flow.parameters(), lr=1e-4)
    opt_b = torch.optim.Adam(reference.parameters(), lr=1e-4)
    for model, optimizer in ((flow, opt_a), (reference, opt_b)):
        # This is the exact source fixed-proposal weighted-CE expression.
        if model is flow:
            log_probability = model.log_prob(x)
        else:
            transformed, logdet = model(x)
            log_probability = flow.base_proposal.log_prob(transformed) + logdet
        loss = -torch.mean(log_probability * weights)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    for a, b in zip(flow.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("slug", ["cfi_c_frkl_c_flow", "cfi_c_ffkl_c_flow"])
def test_both_variants_are_empirical_flow_final_proposals(slug):
    method = get_method(slug)
    assert method.constrained_gmm_backbone
    assert method.final_proposal_family == "flow"
    assert not method.theorem_certified_final_proposal


def test_flow_is_compatible_with_ordinary_unnormalized_is():
    result = ordinary_is(get_benchmark("g1_2d_Sigma1"), _flow(), samples=64, seed=9)
    assert result.self_normalized is False
    assert result.sample_size == 64
