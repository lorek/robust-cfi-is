"""Bounded public-package clean-room installation and runtime smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_IGNORED_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "runs",
}


def escaping_symlinks(root: Path) -> list[Path]:
    root = root.resolve()
    escapes = []
    for directory, directory_names, file_names in os.walk(root, topdown=True):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in _IGNORED_NAMES and not name.endswith(".egg-info")
        ]
        for name in [*directory_names, *file_names]:
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            try:
                path.resolve(strict=True).relative_to(root)
            except (FileNotFoundError, ValueError):
                escapes.append(path)
    return escapes


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _IGNORED_NAMES or name.endswith(".egg-info")
    }


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--python", default="3.12")
    args = parser.parse_args()

    source = Path(__file__).resolve().parents[1]
    escapes = escaping_symlinks(source)
    if escapes:
        raise RuntimeError(f"Escaping or broken symlinks: {escapes}")

    with tempfile.TemporaryDirectory(prefix="robust_cfi_is-clean-room-") as directory:
        temporary = Path(directory)
        project = temporary / "project"
        shutil.copytree(source, project, symlinks=True, ignore=_ignore)
        if escaping_symlinks(project):
            raise RuntimeError("Copied project contains an escaping symlink")
        if args.scan_only:
            print(json.dumps({"copied_tree": "PASS", "symlinks": "PASS"}))
            return 0

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("VIRTUAL_ENV", None)
        environment.update(
            {
                "UV_CACHE_DIR": environment.get(
                    "UV_CACHE_DIR", str(temporary / "uv-cache")
                ),
                "UV_NO_PROGRESS": "1",
                "UV_PROJECT_ENVIRONMENT": str(temporary / "environment"),
            }
        )
        _run(
            [
                "uv",
                "sync",
                "--frozen",
                "--project",
                str(project),
                "--extra",
                "dev",
                "--extra",
                "g7",
                "--python",
                args.python,
            ],
            cwd=temporary,
            environment=environment,
        )

        python = temporary / "environment" / "bin" / "python"
        _run(
            [str(python), "-m", "pytest", "-q"],
            cwd=project,
            environment=environment,
        )
        run_directory = temporary / "run"
        run_directory.mkdir()
        config = project / "configs" / "examples" / "g1_cfi_c.yaml"
        output = run_directory / "result.json"
        driver = r'''
import json
import os
from pathlib import Path
import sys

forbidden = Path(os.environ["ROBUST_CFI_IS_FORBIDDEN_ROOT"]).resolve()
allowed = Path(os.environ["ROBUST_CFI_IS_CLEAN_ROOT"]).resolve()

def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network access is forbidden during the clean-room run")
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    try:
        candidate = Path(os.fsdecode(args[0])).resolve()
    except (OSError, RuntimeError):
        return
    try:
        candidate.relative_to(forbidden)
    except ValueError:
        return
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(f"Parent workspace access rejected: {candidate}") from exc

sys.addaudithook(audit)
import robust_cfi_is
import torch
from robust_cfi_is.api import fit_public_method
from robust_cfi_is.artifacts import ArtifactManifest, load_q_final_json, verify_artifact
from robust_cfi_is.benchmarks import RareEventProblem
from robust_cfi_is.cli import main
from robust_cfi_is.distributions import GaussianDistribution
from robust_cfi_is.naming import get_method
from robust_cfi_is.proposals import RealNVPProposal, UnconstrainedGaussianMixture

module_path = Path(robust_cfi_is.__file__).resolve()
module_path.relative_to(allowed)
status = main([
    "run",
    os.environ["ROBUST_CFI_IS_CONFIG"],
    "--output",
    os.environ["ROBUST_CFI_IS_OUTPUT"],
])
if status != 0:
    raise SystemExit(status)
payload = json.loads(Path(os.environ["ROBUST_CFI_IS_OUTPUT"]).read_text())
if not payload["certification"]["passed"]:
    raise RuntimeError("Clean-room certification failed")

problem = RareEventProblem(
    name="clean_room_halfspace",
    dimension=2,
    gamma=-0.25,
    nominal=GaussianDistribution(torch.zeros(2), torch.eye(2)),
    event_score=lambda samples: samples[:, 0],
)
methods = (
    "frkl_u", "ffkl_u", "frkl_c", "ffkl_c",
    "cfi_u", "cfi_c", "cfi_c_frkl_c", "cfi_c_ffkl_c",
)
for method in methods:
    fitted = fit_public_method(
        problem,
        method,
        components=2,
        seed=17,
        initial_samples=160,
        samples_per_level=240,
        rho=0.25,
        max_levels=2,
        smoothing=0.5,
        epochs=2,
        learning_rate=0.002,
        tau=0.25,
        baseline_rate=0.5,
    )
    if not torch.isfinite(fitted.proposal.covariances()).all():
        raise RuntimeError(f"Non-finite clean-room method smoke: {method}")

defensive = fit_public_method(
    problem,
    "frkl_d",
    components=2,
    seed=17,
    initial_samples=80,
    samples_per_level=120,
    rho=0.25,
    max_levels=2,
    smoothing=0.5,
    epochs=1,
    learning_rate=0.002,
    tau=0.25,
    baseline_rate=0.5,
    defensive_weight=0.2,
)
if not torch.isfinite(defensive.proposal.adaptive.covariances()).all():
    raise RuntimeError("Non-finite clean-room defensive smoke")

manifest_path = allowed / "project" / "reproducibility" / "artifacts.json"
manifest = ArtifactManifest.load(manifest_path)
if len(manifest.records) != 90:
    raise RuntimeError("Clean-room artifact manifest count mismatch")
fixture_manifest = ArtifactManifest.load(
    allowed / "project" / "tests" / "fixtures" / "artifacts.json"
)
fixture_record = fixture_manifest.get("test-problem-k1-cfi-c-seed100-q-final")
fixture_path = allowed / "project" / "tests" / "fixtures" / "q_final_tiny.json"
verify_artifact(fixture_record, fixture_path)
load_q_final_json(fixture_path)

g7_config = allowed / "project" / "configs" / "paper" / "g7.yaml"
if main(["validate", str(g7_config)]) != 0:
    raise RuntimeError("Clean-room g7 config validation failed")

base = UnconstrainedGaussianMixture(
    logits=torch.zeros(1), means=torch.zeros(1, 2),
    factors=torch.eye(2).unsqueeze(0),
)
flow = RealNVPProposal(
    base, hidden_dimension=4, coupling_layers=2,
    generator=torch.Generator().manual_seed(23), device="cpu",
)
draws = flow.sample(8, generator=torch.Generator().manual_seed(29))
if not torch.isfinite(flow.log_prob(draws)).all():
    raise RuntimeError("Non-finite clean-room flow smoke")
for slug in ("cfi_c_frkl_c_flow", "cfi_c_ffkl_c_flow"):
    method = get_method(slug)
    if method.final_proposal_family != "flow" or method.theorem_certified_final_proposal:
        raise RuntimeError("Flow theorem-boundary metadata changed")
'''
        run_environment = environment.copy()
        run_environment.update(
            {
                "ROBUST_CFI_IS_FORBIDDEN_ROOT": str(source.parent.resolve()),
                "ROBUST_CFI_IS_CLEAN_ROOT": str(temporary.resolve()),
                "ROBUST_CFI_IS_CONFIG": str(config),
                "ROBUST_CFI_IS_OUTPUT": str(output),
                "HOME": str(temporary / "home"),
                "TMPDIR": str(temporary / "tmp"),
            }
        )
        Path(run_environment["HOME"]).mkdir()
        Path(run_environment["TMPDIR"]).mkdir()
        _run(
            [str(python), "-I", "-c", driver],
            cwd=run_directory,
            environment=run_environment,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "copy_only": True,
                    "fresh_install": True,
                    "installed_import": "PASS",
                    "quickstart": "PASS",
                    "supported_method_smokes": "PASS",
                    "defensive_method_smoke": "PASS",
                    "flow_method_smoke": "PASS",
                    "g7_asset_free_config": "PASS",
                    "paper_configs": "PASS",
                    "artifact_manifest": "PASS",
                    "safe_q_final_fixture": "PASS",
                    "parent_workspace_access": "NONE",
                    "network_during_run": "DISABLED",
                    "estimate": payload["importance_sampling"]["estimate"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
