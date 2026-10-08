import torch
from torch import nn

import robust_cfi_is.methods.flow as flow_module
from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.distributions import GaussianDistribution


class _GeneratingProposal:
    def sample(self, count, *, generator):
        del generator
        return torch.zeros(count, 1)

    def log_prob(self, samples):
        return torch.zeros(samples.shape[0])


class _StatefulFlow(nn.Module):
    """Minimal deterministic proposal whose state advances once per epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.state = nn.Parameter(torch.tensor(0.0))

    def log_prob(self, samples):
        return self.state.expand(samples.shape[0])

    def clone(self):
        clone = _StatefulFlow()
        clone.load_state_dict(self.state_dict())
        return clone


class _IncrementingOptimizer:
    def __init__(self, parameters, *, lr):
        del lr
        self.parameters = list(parameters)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(1.0)


def _fit_with_validation_values(monkeypatch, validation_values):
    values = iter(validation_values)

    def controlled_loss(log_probability, weights):
        del weights
        if torch.is_grad_enabled():
            return log_probability.mean() * 0.0
        return log_probability.new_tensor(next(values))

    monkeypatch.setattr(flow_module, "RealNVPProposal", _StatefulFlow)
    monkeypatch.setattr(flow_module.torch.optim, "Adam", _IncrementingOptimizer)
    monkeypatch.setattr(
        flow_module,
        "mean_one_fixed_proposal_weights",
        lambda nominal, proposal, *, output_dtype: torch.ones(
            nominal.shape[0], dtype=output_dtype
        ),
    )
    monkeypatch.setattr(flow_module, "weighted_forward_kl_loss", controlled_loss)
    problem = RareEventProblem(
        name="deterministic_model_selection",
        dimension=1,
        gamma=0.0,
        nominal=GaussianDistribution(torch.zeros(1), torch.eye(1)),
        event_score=lambda samples: torch.ones(samples.shape[0]),
    )
    return flow_module.fit_fixed_proposal_flow(
        problem,
        _GeneratingProposal(),
        seed=1,
        initial_samples=1,
        training_samples=5,
        epochs=len(validation_values),
        learning_rate=1e-5,
        hidden_dimension=2,
        coupling_layers=2,
        device="cpu",
    )


def test_normal_budget_exhaustion_returns_final_proposal(monkeypatch):
    result = _fit_with_validation_values(monkeypatch, [1.0, 2.0, 3.0])

    assert not result.stopped_early
    assert result.best_epoch == 0
    assert result.proposal.state.item() == 3.0


def test_early_stopping_returns_best_validation_checkpoint(monkeypatch):
    result = _fit_with_validation_values(monkeypatch, [1.0] + [2.0] * 101)

    assert result.stopped_early
    assert result.best_epoch == 0
    assert result.proposal.state.item() == 1.0
