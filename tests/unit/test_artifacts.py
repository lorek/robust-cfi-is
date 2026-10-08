import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from robust_cfi_is.artifacts import (
    ArtifactManifest, artifact_identity, load_q_final_json,
    resolve_artifact_path, validate_mirror_updates, verify_artifact,
)
from robust_cfi_is.cli import main


FIXTURES = Path(__file__).parents[1] / "fixtures"
ROOT = Path(__file__).parents[2]


def test_manifest_has_complete_unique_certified_artifacts():
    manifest = ArtifactManifest.load(ROOT / "reproducibility/artifacts.json")
    assert len(manifest.records) == 90
    assert len({record.name for record in manifest.records}) == 90
    assert sum(record.workflow_status == "deferred_g7" for record in manifest.records) == 10
    assert manifest.cohort["availability"] == "repository_embedded"
    assert all(record.availability == "repository_embedded" for record in manifest.records)
    assert all(not record.mirrors for record in manifest.records)


def test_repository_embedded_artifacts_are_exact_and_verified():
    manifest_path = ROOT / "reproducibility/artifacts.json"
    manifest = ArtifactManifest.load(manifest_path)
    artifact_root = ROOT / "reproducibility/q_final"
    expected = {f"{record.name}.json": record for record in manifest.records}
    entries = tuple(artifact_root.iterdir())
    assert all(path.is_file() and not path.is_symlink() for path in entries)
    actual = {path.name: path for path in entries}

    assert len(expected) == len(actual) == 90
    assert set(actual) == set(expected)
    assert sum(path.stat().st_size for path in actual.values()) == sum(
        record.size_bytes for record in manifest.records
    )
    assert all(path.suffix == ".json" for path in actual.values())
    for name, record in expected.items():
        path = verify_artifact(record, actual[name])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        load_q_final_json(path)


def test_repository_local_resolution_and_explicit_override(capsys, tmp_path):
    manifest_path = ROOT / "reproducibility/artifacts.json"
    manifest = ArtifactManifest.load(manifest_path)
    record = manifest.records[0]
    expected = ROOT / "reproducibility/q_final" / f"{record.name}.json"

    assert resolve_artifact_path(manifest_path, record) == expected.resolve()
    assert main([
        "artifacts", "verify", "--manifest", str(manifest_path),
        "--name", record.name,
    ]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out
    override = tmp_path / "local-copy.json"
    override.write_bytes(expected.read_bytes())
    assert main([
        "artifacts", "verify", "--manifest", str(manifest_path),
        "--name", record.name, "--path", str(override),
    ]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out


def test_repository_local_resolution_rejects_path_traversal():
    manifest_path = ROOT / "reproducibility/artifacts.json"
    record = ArtifactManifest.load(manifest_path).records[0]
    malicious = replace(record, name="../../local-checkpoint")
    with pytest.raises(ValueError, match="canonical public slug"):
        resolve_artifact_path(manifest_path, malicious)


def test_reproduce_evaluate_uses_repository_local_artifact(monkeypatch, tmp_path):
    manifest_path = ROOT / "reproducibility/artifacts.json"
    manifest = ArtifactManifest.load(manifest_path)
    record = next(
        record
        for record in manifest.records
        if record.benchmark == "g1_2d_Sigma1"
        and record.method == "cfi_c"
        and record.components == 6
    )

    class Result:
        def to_dict(self):
            return {"estimate": 0.0, "test_double": True}

    monkeypatch.setattr("robust_cfi_is.cli.ordinary_is", lambda *args, **kwargs: Result())
    output = tmp_path / "evaluation.json"
    assert main([
        "reproduce", "evaluate", str(ROOT / "configs/paper/g1.yaml"),
        "--manifest", str(manifest_path), "--artifact-name", record.name,
        "--evaluation-seed", "100", "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_name"] == record.name
    assert payload["artifact_sha256"] == record.sha256
    assert payload["result"]["test_double"] is True


def test_local_fixture_verification_and_safe_load(tmp_path):
    manifest = ArtifactManifest.load(FIXTURES / "artifacts.json")
    record = manifest.get("test-problem-k1-cfi-c-seed100-q-final")
    path = verify_artifact(record, FIXTURES / "q_final_tiny.json")
    proposal = load_q_final_json(path)
    assert proposal.n_components == 1
    torch.testing.assert_close(proposal.means, torch.zeros(1, 2))

    damaged = tmp_path / "damaged.json"
    damaged.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_artifact(record, damaged)


def test_safe_loader_rejects_content_drift(tmp_path):
    payload = json.loads((FIXTURES / "q_final_tiny.json").read_text())
    payload["content"]["proposal"]["dimension"] = 3
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_q_final_json(drifted)


def test_future_mirror_validation_cannot_mutate_canonical_identities():
    manifest = ArtifactManifest.load(ROOT / "reproducibility/artifacts.json")
    before = tuple(artifact_identity(record) for record in manifest.records)
    name = manifest.records[0].name
    updates = validate_mirror_updates(
        manifest, {name: ["https://public.example/artifacts/q-final.json?download=1"]}
    )
    assert updates[name][0].startswith("https://")
    assert tuple(artifact_identity(record) for record in manifest.records) == before
    assert len(manifest.records) == 90
    with pytest.raises(ValueError, match="HTTPS"):
        validate_mirror_updates(manifest, {name: ["file:///local/q-final.json"]})
