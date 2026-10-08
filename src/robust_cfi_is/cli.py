"""Command-line interface for supported public workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from robust_cfi_is.api import run_config
from robust_cfi_is.artifacts import (
    ArtifactManifest,
    load_q_final_json,
    resolve_artifact_path,
    verify_artifact,
)
from robust_cfi_is.benchmarks import get_benchmark, get_g7_benchmark
from robust_cfi_is.config import PaperConfig, load_config
from robust_cfi_is.estimation import ordinary_is


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robust-cfi-is")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate a public config")
    validate.add_argument("config", type=Path)

    run = subcommands.add_parser("run", help="run a supported public config")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path)

    reproduce = subcommands.add_parser("reproduce", help="plan or evaluate paper workflows")
    reproduction = reproduce.add_subparsers(dest="reproduce_command", required=True)
    plan = reproduction.add_parser("plan", help="print a non-executing workflow plan")
    plan.add_argument("config", type=Path)
    plan.add_argument("--method", required=True)
    plan.add_argument("--k", required=True, type=int)
    evaluate = reproduction.add_parser("evaluate", help="evaluate one local q_final")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--artifact-name", required=True)
    evaluate.add_argument("--artifact", type=Path)
    evaluate.add_argument("--evaluation-seed", type=int, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--g7-data", type=Path)
    evaluate.add_argument("--resnet-weights", type=Path)
    evaluate.add_argument("--event-device")

    artifacts = subcommands.add_parser("artifacts", help="inspect offline artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)
    listing = artifact_commands.add_parser("list")
    listing.add_argument("--manifest", type=Path, required=True)
    verify = artifact_commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--name", required=True)
    verify.add_argument("--path", type=Path)
    return parser


def _paper_config(path: Path) -> PaperConfig:
    config = load_config(path)
    if not isinstance(config, PaperConfig):
        raise ValueError("This command requires a paper_workflow config")
    return config


def _plan(config: PaperConfig, method: str, components: int) -> dict:
    if method not in config.methods:
        raise ValueError(f"Method {method!r} is not enabled by this config")
    if components not in config.components:
        raise ValueError(f"K={components} is not enabled by this config")
    return {
        "action": "plan_only",
        "experiment_started": False,
        "benchmark": config.benchmark,
        "method": method,
        "components": components,
        "training_seed": config.training_seed,
        "evaluation_seeds": list(config.evaluation_seeds),
        "fitting": config.fitting[method],
        "final_proposal": config.final_proposal,
        "evaluation": config.evaluation,
        "aggregation": config.aggregation,
        "execution": config.execution,
        "cost_class": config.tier,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "artifacts":
        manifest = ArtifactManifest.load(args.manifest)
        if args.artifact_command == "list":
            print(json.dumps([record.name for record in manifest.records], indent=2))
            return 0
        record = manifest.get(args.name)
        path = resolve_artifact_path(args.manifest, record, args.path)
        verified = verify_artifact(record, path)
        print(json.dumps({"name": record.name, "path": str(verified), "status": "PASS"}))
        return 0
    if args.command == "reproduce":
        config = _paper_config(args.config)
        if args.reproduce_command == "plan":
            print(json.dumps(_plan(config, args.method, args.k), indent=2, sort_keys=True))
            return 0
        manifest = ArtifactManifest.load(args.manifest)
        record = manifest.get(args.artifact_name)
        artifact_path = resolve_artifact_path(args.manifest, record, args.artifact)
        if record.benchmark != config.benchmark:
            raise ValueError("Artifact benchmark does not match config")
        if args.evaluation_seed not in config.evaluation_seeds:
            raise ValueError("Evaluation seed is not enabled by this config")
        verify_artifact(record, artifact_path)
        proposal = load_q_final_json(artifact_path)
        g7_options = (args.g7_data, args.resnet_weights, args.event_device)
        if config.benchmark == "g7_resnet":
            if any(value is None for value in g7_options):
                raise ValueError(
                    "g7 evaluation requires --g7-data, --resnet-weights, "
                    "and --event-device"
                )
            problem, assets = get_g7_benchmark(
                args.g7_data, args.resnet_weights, device=args.event_device
            )
            asset_record = assets.to_dict()
        else:
            if any(value is not None for value in g7_options):
                raise ValueError("g7 asset/device options are only valid for g7_resnet")
            problem = get_benchmark(config.benchmark)
            asset_record = None
        result = ordinary_is(
            problem,
            proposal,
            samples=int(config.evaluation["sample_size"]),
            seed=args.evaluation_seed,
        ).to_dict()
        payload = {
            "schema_version": 1,
            "record_type": "run_result",
            "benchmark": config.benchmark,
            "method": record.method,
            "components": record.components,
            "artifact_name": record.name,
            "artifact_sha256": record.sha256,
            "evaluation_seed": args.evaluation_seed,
            "proposal_artifact": {
                "name": record.name,
                "sha256": record.sha256,
                "size": record.size_bytes,
            },
            "g7_assets": asset_record,
            "result": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output), "status": "PASS"}))
        return 0
    result = run_config(args.config, output=args.output)
    output = args.output or Path(result["configuration"]["output"])
    certification = result["certification"]
    summary = {
        "output": str(output),
        "estimate": result["importance_sampling"]["estimate"],
        "standard_error": result["importance_sampling"]["standard_error"],
        "CoV": result["importance_sampling"]["cov"],
        "event_count": result["importance_sampling"]["event_count"],
        "minimum_covariance_margin": (
            certification["minimum_theorem_margin"]
            if certification is not None
            else None
        ),
        "runtime_seconds": result["runtime_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
