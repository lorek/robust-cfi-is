"""RealNVP proposal used by the two public ``+F`` variants."""

from __future__ import annotations

import copy

import torch
from torch import nn

from robust_cfi_is.proposals.gmm import GaussianMixtureProposal


class AffineCoupling(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension), nn.ReLU(),
            nn.Linear(hidden_dimension, dimension), nn.Tanh(),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension), nn.ReLU(),
            nn.Linear(hidden_dimension, dimension),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = values * self.mask
        scale = self.scale_net(masked) * (1 - self.mask)
        translation = self.translate_net(masked) * (1 - self.mask)
        result = masked + (1 - self.mask) * (values * torch.exp(scale) + translation)
        return result, torch.sum(scale, dim=1)

    def inverse(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = values * self.mask
        scale = self.scale_net(masked) * (1 - self.mask)
        translation = self.translate_net(masked) * (1 - self.mask)
        result = masked + (1 - self.mask) * ((values - translation) * torch.exp(-scale))
        return result, -torch.sum(scale, dim=1)


class RealNVPProposal(nn.Module):
    """A flow over a fixed GMM backbone; it is not theorem-certified."""

    theorem_certified_final_proposal = False
    final_proposal_family = "flow"
    constrained_gmm_backbone = True

    def __init__(
        self,
        base_proposal: GaussianMixtureProposal,
        *,
        hidden_dimension: int,
        coupling_layers: int,
        generator: torch.Generator,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if hidden_dimension < 1 or coupling_layers < 1:
            raise ValueError("RealNVP dimensions and coupling count must be positive")
        if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("flow device must be exactly 'cpu' or 'cuda:N'")
        if device.startswith("cuda:"):
            index = int(device[5:])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(f"requested flow device is unavailable: {device}")
        self.base_proposal = base_proposal.clone()
        self.dimension = base_proposal.dimension
        self.device_name = device

        saved = torch.random.get_rng_state()
        torch.random.set_rng_state(generator.get_state())
        try:
            layers = []
            for index in range(coupling_layers):
                mask = torch.ones(self.dimension)
                mask[::2] = 0
                if index % 2 == 1:
                    mask = 1 - mask
                layers.append(AffineCoupling(self.dimension, hidden_dimension, mask))
            self.coupling_layers = nn.ModuleList(layers)
            self.initialize_near_identity()
            generator.set_state(torch.random.get_rng_state())
        finally:
            torch.random.set_rng_state(saved)
        self.to(torch.device(device))

    def initialize_near_identity(self) -> None:
        """Match the authoritative last-layer initialization exactly."""
        for layer in self.coupling_layers:
            nn.init.zeros_(layer.scale_net[-2].bias)
            nn.init.normal_(layer.scale_net[-2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.translate_net[-1].bias)
            nn.init.normal_(layer.translate_net[-1].weight, mean=0.0, std=1e-3)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = values
        log_determinant = torch.zeros(values.shape[0], device=values.device)
        for layer in self.coupling_layers:
            transformed, increment = layer(transformed)
            log_determinant = log_determinant + increment
        return transformed, log_determinant

    def inverse(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = values
        log_determinant = torch.zeros(values.shape[0], device=values.device)
        for layer in reversed(self.coupling_layers):
            transformed, increment = layer.inverse(transformed)
            log_determinant = log_determinant + increment
        return transformed, log_determinant

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32).to(self.device_name)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("RealNVP samples have the wrong shape")
        base_values, log_determinant = self.forward(values)
        base_log_probability = self.base_proposal.log_prob(base_values.to("cpu"))
        return base_log_probability.to(values.device) + log_determinant

    @torch.no_grad()
    def sample(
        self, sample_size: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        base = self.base_proposal.sample(sample_size, generator=generator)
        values, _ = self.inverse(base.to(self.device_name))
        return values.to("cpu")

    def clone(self) -> "RealNVPProposal":
        return copy.deepcopy(self)
