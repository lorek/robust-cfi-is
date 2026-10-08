"""Strict configuration loading for supported public workflows."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from robust_cfi_is.naming import get_method
from robust_cfi_is.benchmarks import get_benchmark
from robust_cfi_is.benchmarks.resnet import (
    G7_DATA_NAME,
    G7_DATA_SHA256,
    G7_DATA_SIZE,
    G7_DIMENSION,
    G7_NOISE_SCALE,
    G7_THRESHOLD,
    RESNET18_WEIGHTS_NAME,
    RESNET18_WEIGHTS_SHA256,
    RESNET18_WEIGHTS_SIZE,
)


@dataclass(frozen=True)
class QuickstartConfig:
    schema_version: int
    benchmark: str
    method: str
    components: int
    training_seed: int
    evaluation_seed: int
    initial_samples: int
    samples_per_level: int
    rho: float
    max_levels: int
    smoothing: float
    is_samples: int
    device: str
    tracking: bool
    network: bool
    output: str
    epochs: int = 5
    learning_rate: float = 0.005
    tau: float = 0.25
    baseline_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only configuration schema_version 1 is supported")
        if self.benchmark != "g1_2d_Sigma1":
            raise ValueError("This release supports only benchmark 'g1_2d_Sigma1'")
        method = get_method(self.method)
        if method.defensive:
            raise ValueError("Defensive methods require an explicit paper workflow config")
        if self.components < 1:
            raise ValueError("components must be positive")
        if self.training_seed < 0 or self.evaluation_seed < 0:
            raise ValueError("seeds must be nonnegative")
        if self.initial_samples < self.components:
            raise ValueError("initial_samples must be at least components")
        if self.samples_per_level < self.components:
            raise ValueError("samples_per_level must be at least components")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between zero and one")
        if self.max_levels < 1:
            raise ValueError("max_levels must be positive")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must lie between zero and one")
        if self.is_samples < 1:
            raise ValueError("is_samples must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 <= self.baseline_rate <= 1.0:
            raise ValueError("baseline_rate must lie between zero and one")
        if self.device != "cpu":
            raise ValueError("Public core workflows support only explicit CPU execution")
        if self.tracking:
            raise ValueError("Tracking must be disabled in public core workflows")
        if self.network:
            raise ValueError("Network access must be disabled in public core workflows")
        output = Path(self.output)
        if output.is_absolute() or ".." in output.parts:
            raise ValueError("output must be a relative path without parent traversal")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperConfig:
    schema_version: int
    config_kind: str
    benchmark: str
    tier: str
    methods: tuple[str, ...]
    components: tuple[int, ...]
    training_seed: int
    evaluation_seeds: tuple[int, ...]
    fitting: dict[str, Any]
    final_proposal: dict[str, Any]
    evaluation: dict[str, Any]
    aggregation: dict[str, Any]
    execution: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.config_kind != "paper_workflow":
            raise ValueError("Unsupported paper configuration schema")
        is_g7 = self.benchmark == "g7_resnet"
        if not is_g7:
            get_benchmark(self.benchmark)
        if not self.methods:
            raise ValueError("Paper config needs at least one public method")
        for method in self.methods:
            get_method(method)
        if set(self.fitting) != set(self.methods):
            raise ValueError("fitting must contain exactly the configured public methods")
        if not self.components or any(value not in {6, 16} for value in self.components):
            raise ValueError("Paper components must be an explicit subset of 6 and 16")
        if self.training_seed < 0 or not self.evaluation_seeds:
            raise ValueError("Paper seeds must be explicit")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("Evaluation seeds must be unique")
        if self.tier not in {"analytic_paper", "expensive_analytic", "external_g7"}:
            raise ValueError("Unknown reproduction tier")
        if is_g7 and self.tier != "external_g7":
            raise ValueError("g7 must use the external_g7 reproduction tier")
        if not is_g7 and self.tier == "external_g7":
            raise ValueError("external_g7 tier is reserved for g7")
        for section in (
            self.fitting,
            self.final_proposal,
            self.evaluation,
            self.aggregation,
            self.execution,
        ):
            if not isinstance(section, dict):
                raise ValueError("Paper workflow phase sections must be mappings")
        if self.execution.get("device") != "cpu":
            raise ValueError("Proposal execution must use explicit CPU execution")
        if self.execution.get("tracking") is not False or self.execution.get(
            "network"
        ) is not False:
            raise ValueError("Public paper workflows require tracking/network disabled")
        output_root = Path(str(self.execution.get("output_root", "")))
        if not str(output_root) or output_root.is_absolute() or ".." in output_root.parts:
            raise ValueError("output_root must be a portable relative path")
        if self.evaluation.get("sample_size") != 100000:
            raise ValueError("Paper final-IS sample_size must be explicit and authoritative")
        if self.aggregation.get("non_cmc_runs") != "all_20":
            raise ValueError("Non-CMC aggregation must use all 20 runs")
        if is_g7:
            expected = {
                "dimension": G7_DIMENSION,
                "noise_scale": G7_NOISE_SCALE,
                "threshold": G7_THRESHOLD,
                "data": {
                    "filename": G7_DATA_NAME,
                    "size": G7_DATA_SIZE,
                    "sha256": G7_DATA_SHA256,
                },
                "weights": {
                    "filename": RESNET18_WEIGHTS_NAME,
                    "size": RESNET18_WEIGHTS_SIZE,
                    "sha256": RESNET18_WEIGHTS_SHA256,
                    "architecture": "torchvision_resnet18",
                    "identity": "IMAGENET1K_V1",
                },
            }
            if self.execution.get("event_device") != "explicit_cli_required":
                raise ValueError("g7 event_device must be supplied explicitly at evaluation")
            if self.execution.get("g7_assets") != expected:
                raise ValueError("g7 asset/scientific identity does not match the release contract")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PublicConfig = QuickstartConfig | PaperConfig


def load_config(path: str | Path) -> PublicConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load configuration: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    config_type = PaperConfig if payload.get("config_kind") == "paper_workflow" else QuickstartConfig
    for tuple_field in ("methods", "components", "evaluation_seeds"):
        if config_type is PaperConfig and isinstance(payload.get(tuple_field), list):
            payload[tuple_field] = tuple(payload[tuple_field])
    schema_fields = fields(config_type)
    expected = {field.name for field in schema_fields}
    required = {
        field.name
        for field in schema_fields
        if field.default is MISSING and field.default_factory is MISSING
    }
    actual = set(payload)
    if not required <= actual or not actual <= expected:
        missing = sorted(required - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Configuration fields do not match schema; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return config_type(**payload)
