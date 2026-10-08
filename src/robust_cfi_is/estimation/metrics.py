"""Small result schema for ordinary importance sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ISResult:
    estimator: str
    self_normalized: bool
    sample_size: int
    evaluation_seed: int
    estimate: float
    standard_error: float
    cov: float | None
    event_count: int
    log_weight_minimum: float
    log_weight_maximum: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
