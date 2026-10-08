"""Small public execution API."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_cfi_is.benchmarks import get_benchmark
from robust_cfi_is.config import PaperConfig, QuickstartConfig, load_config
from robust_cfi_is.estimation import ordinary_is
from robust_cfi_is.methods import (
    fit_cfi,
    fit_defensive_reverse_kl,
    fit_forward_kl,
    fit_fixed_proposal_flow,
    fit_reverse_kl,
    refine_forward_kl,
    refine_reverse_kl,
)
from robust_cfi_is.naming import MethodSpec, get_method
from robust_cfi_is.proposals import (
    DefensiveMixtureProposal,
    GaussianMixtureProposal,
    RealNVPProposal,
    certify_final_gmm,
)


@dataclass(frozen=True)
class PublicMethodFit:
    method: MethodSpec
    proposal: GaussianMixtureProposal | DefensiveMixtureProposal | RealNVPProposal
    details: dict[str, Any]


def _finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result contains a non-finite scalar")
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _execute(config: QuickstartConfig) -> dict[str, Any]:
    method = get_method(config.method)
    problem = get_benchmark(config.benchmark)
    if method.slug != "cfi_c":
        return _execute_core_method(config)
    started = time.perf_counter()
    fitted = fit_cfi(
        problem,
        components=config.components,
        seed=config.training_seed,
        initial_samples=config.initial_samples,
        samples_per_level=config.samples_per_level,
        rho=config.rho,
        max_levels=config.max_levels,
        smoothing=config.smoothing,
    )
    final_proposal = certify_final_gmm(
        fitted.proposal,
        problem.nominal.covariance,
    )
    estimate = ordinary_is(
        problem,
        final_proposal,
        samples=config.is_samples,
        seed=config.evaluation_seed,
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "benchmark": {
            "name": problem.name,
            "dimension": problem.dimension,
            "gamma": problem.gamma,
        },
        "method": {
            "slug": method.slug,
            "display_name": method.display_name,
            "objective": method.objective,
            "constrained": method.constrained,
        },
        "configuration": config.to_dict(),
        "cfi": {
            "target_reached": fitted.target_reached,
            "levels_completed": len(fitted.levels),
            "levels": [
                {
                    "index": row.index,
                    "gamma": row.gamma,
                    "elite_count": row.elite_count,
                    "beta": row.beta,
                    "target_reached": row.target_reached,
                }
                for row in fitted.levels
            ],
            "final_fit_elite_count": fitted.final_fit_elite_count,
        },
        "certification": final_proposal.certification,
        "importance_sampling": estimate.to_dict(),
        "runtime_seconds": elapsed,
    }
    return _finite_json(result)


def _cfi_details(fitted) -> dict[str, Any]:
    return {
        "target_reached": fitted.target_reached,
        "levels_completed": len(fitted.levels),
        "levels": [
            {
                "index": row.index,
                "gamma": row.gamma,
                "elite_count": row.elite_count,
                "beta": row.beta,
                "target_reached": row.target_reached,
            }
            for row in fitted.levels
        ],
        "final_fit_elite_count": fitted.final_fit_elite_count,
    }


def fit_public_method(
    problem,
    method_slug: str,
    *,
    components: int,
    seed: int,
    initial_samples: int,
    samples_per_level: int,
    rho: float,
    max_levels: int,
    smoothing: float,
    epochs: int,
    learning_rate: float,
    tau: float,
    baseline_rate: float,
    defensive_weight: float | None = None,
    flow_hidden_dimension: int | None = None,
    flow_coupling_layers: int | None = None,
    flow_learning_rate: float = 1e-5,
    flow_initial_samples: int = 25_000,
    flow_training_samples: int = 25_000,
    flow_epochs: int = 1_000,
    flow_device: str = "cpu",
) -> PublicMethodFit:
    """Fit one public method through its documented dispatch identifier."""

    method = get_method(method_slug)
    if method.slug == "frkl_d":
        if defensive_weight is None:
            raise ValueError("frkl_d requires defensive_weight")
        fitted = fit_defensive_reverse_kl(
            problem,
            components=components,
            defensive_weight=defensive_weight,
            seed=seed,
            initial_samples=initial_samples,
            training_samples=samples_per_level,
            epochs=epochs,
            learning_rate=learning_rate,
            tau=tau,
            baseline_rate=baseline_rate,
            smoothing=smoothing,
        )
        return PublicMethodFit(
            method,
            fitted.proposal,
            {
                "defensive_reverse_kl": {
                    "defensive_weight": defensive_weight,
                    "stratified_sampling": True,
                    "steps": [row.__dict__ for row in fitted.steps],
                }
            },
        )
    if method.slug in {"cfi_u", "cfi_c"}:
        fitted = fit_cfi(
            problem,
            components=components,
            constrained=method.constrained,
            seed=seed,
            initial_samples=initial_samples,
            samples_per_level=samples_per_level,
            rho=rho,
            max_levels=max_levels,
            smoothing=smoothing,
        )
        return PublicMethodFit(method, fitted.proposal, {"cfi": _cfi_details(fitted)})
    if method.slug in {"ffkl_u", "ffkl_c"}:
        fitted = fit_forward_kl(
            problem,
            components=components,
            constrained=method.constrained,
            seed=seed,
            initial_samples=initial_samples,
            samples_per_level=samples_per_level,
            rho=rho,
            max_levels=max_levels,
            epochs_per_level=epochs,
            learning_rate=learning_rate,
        )
        details = {
            "forward_kl": {
                "target_reached": fitted.target_reached,
                "levels_completed": len(fitted.levels),
                "levels": [row.__dict__ for row in fitted.levels],
            }
        }
        return PublicMethodFit(method, fitted.proposal, details)
    if method.slug in {"frkl_u", "frkl_c"}:
        fitted = fit_reverse_kl(
            problem,
            components=components,
            constrained=method.constrained,
            seed=seed,
            initial_samples=initial_samples,
            training_samples=samples_per_level,
            epochs=epochs,
            learning_rate=learning_rate,
            tau=tau,
            baseline_rate=baseline_rate,
            smoothing=smoothing,
        )
        return PublicMethodFit(
            method,
            fitted.proposal,
            {"reverse_kl": {"steps": [row.__dict__ for row in fitted.steps]}},
        )

    cfi = fit_cfi(
        problem,
        components=components,
        constrained=True,
        seed=seed,
        initial_samples=initial_samples,
        samples_per_level=samples_per_level,
        rho=rho,
        max_levels=max_levels,
        smoothing=smoothing,
    )
    reverse_methods = {"cfi_c_frkl_c", "cfi_c_frkl_c_flow"}
    forward_methods = {"cfi_c_ffkl_c", "cfi_c_ffkl_c_flow"}
    if method.slug in reverse_methods:
        refined = refine_reverse_kl(
            problem,
            cfi.proposal,
            seed=seed,
            initial_samples=initial_samples,
            training_samples=samples_per_level,
            epochs=epochs,
            learning_rate=learning_rate,
            tau=tau,
            baseline_rate=baseline_rate,
            smoothing=smoothing,
        )
        refinement = {"reverse_kl": {"steps": [row.__dict__ for row in refined.steps]}}
    elif method.slug in forward_methods:
        refined = refine_forward_kl(
            problem,
            cfi.proposal,
            seed=seed,
            initial_samples=initial_samples,
            training_samples=samples_per_level,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        refinement = {
            "forward_kl": {
                "target_reached": refined.target_reached,
                "levels": [row.__dict__ for row in refined.levels],
            }
        }
    else:
        raise AssertionError(f"Unhandled public method {method.slug}")
    details = {"cfi": _cfi_details(cfi), **refinement}
    if method.final_proposal_family == "flow":
        hidden = flow_hidden_dimension or (1024 if problem.dimension == 2 else 32)
        couplings = flow_coupling_layers or (48 if problem.dimension == 2 else 32)
        flowed = fit_fixed_proposal_flow(
            problem,
            refined.proposal,
            seed=seed,
            initial_samples=flow_initial_samples,
            training_samples=flow_training_samples,
            epochs=flow_epochs,
            learning_rate=flow_learning_rate,
            hidden_dimension=hidden,
            coupling_layers=couplings,
            device=flow_device,
        )
        details["flow"] = {
            "elite_count": flowed.elite_count,
            "validation_elite_count": flowed.validation_elite_count,
            "best_epoch": flowed.best_epoch,
            "stopped_early": flowed.stopped_early,
            "theorem_certified_final_proposal": False,
        }
        return PublicMethodFit(method, flowed.proposal, details)
    return PublicMethodFit(method, refined.proposal, details)


def _execute_core_method(config: QuickstartConfig) -> dict[str, Any]:
    problem = get_benchmark(config.benchmark)
    started = time.perf_counter()
    fitted = fit_public_method(
        problem,
        config.method,
        components=config.components,
        seed=config.training_seed,
        initial_samples=config.initial_samples,
        samples_per_level=config.samples_per_level,
        rho=config.rho,
        max_levels=config.max_levels,
        smoothing=config.smoothing,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        tau=config.tau,
        baseline_rate=config.baseline_rate,
    )
    if fitted.method.theorem_certified_final_proposal:
        inference_proposal = certify_final_gmm(
            fitted.proposal,
            problem.nominal.covariance,
        )
        certification = inference_proposal.certification
    else:
        inference_proposal = fitted.proposal
        certification = None
    estimate = ordinary_is(
        problem,
        inference_proposal,
        samples=config.is_samples,
        seed=config.evaluation_seed,
    )
    result = {
        "schema_version": 1,
        "benchmark": {
            "name": problem.name,
            "dimension": problem.dimension,
            "gamma": problem.gamma,
        },
        "method": {
            "slug": fitted.method.slug,
            "display_name": fitted.method.display_name,
            "objective": fitted.method.objective,
            "constrained": fitted.method.constrained,
            "constrained_gmm_backbone": fitted.method.constrained_gmm_backbone,
            "final_proposal_family": fitted.method.final_proposal_family,
            "theorem_certified_final_proposal": (
                fitted.method.theorem_certified_final_proposal
            ),
        },
        "configuration": config.to_dict(),
        "training": fitted.details,
        "certification": certification,
        "importance_sampling": estimate.to_dict(),
        "runtime_seconds": time.perf_counter() - started,
    }
    return _finite_json(result)


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def run_quickstart(
    config: QuickstartConfig,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    result = _execute(config)
    destination = Path(config.output if output is None else output)
    _write_result(destination, result)
    return result


def run_config(
    path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(path)
    if isinstance(config, PaperConfig):
        raise ValueError(
            "Paper configs require 'robust-cfi-is reproduce plan/evaluate'; "
            "'robust-cfi-is run' is reserved for bounded executable configs"
        )
    return run_quickstart(config, output=output)
