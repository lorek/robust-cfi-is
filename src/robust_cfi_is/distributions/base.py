"""Small public protocol for proposal and nominal distributions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Distribution(Protocol):
    @property
    def dimension(self) -> int: ...

    def sample(
        self,
        sample_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor: ...
