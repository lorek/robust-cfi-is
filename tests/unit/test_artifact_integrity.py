import base64
import hashlib
import json
from pathlib import Path

from robust_cfi_is.artifacts import ArtifactManifest, load_q_final_json


ROOT = Path(__file__).parents[2]
MANIFEST_PATH = ROOT / "reproducibility" / "artifacts.json"
ARTIFACT_ROOT = ROOT / "reproducibility" / "q_final"


def _canonical_json_bytes(payload):
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _tensor_semantic_sha256(record):
    header = _canonical_json_bytes(
        {"dtype": "torch.float32", "shape": record["shape"]}
    )
    return _sha256(header + base64.b64decode(record["data_base64"], validate=True))


def test_embedded_artifacts_are_internally_consistent():
    manifest = ArtifactManifest.load(MANIFEST_PATH)
    assert manifest.schema_version == 1
    assert len(manifest.records) == 90
    assert len({record.name for record in manifest.records}) == 90

    for record in manifest.records:
        path = ARTIFACT_ROOT / f"{record.name}.json"
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        content = payload["content"]
        provenance = content["provenance"]

        assert len(encoded) == record.size_bytes
        assert _sha256(encoded) == record.sha256
        assert payload["schema_version"] == 1
        assert _sha256(_canonical_json_bytes(content)) == payload[
            "artifact_content_sha256"
        ]
        assert (
            record.certification["artifact_content_sha256"]
            == payload["artifact_content_sha256"]
        )
        assert record.certification["passed"] is True
        assert content["certification"]["passed"] is True
        assert content["artifact_kind"] == record.artifact_kind
        assert provenance["final_logical_id"] == (
            f"q_final::{record.benchmark}::k{record.components}::"
            f"{record.method}::seed{record.training_seed}"
        )
        assert provenance["problem_id"] == record.benchmark

        for name, tensor in content["tensors"].items():
            semantic_hash = _tensor_semantic_sha256(tensor)
            assert tensor["semantic_sha256"] == semantic_hash
            assert content["tensor_semantic_sha256"][name] == semantic_hash
        assert (
            provenance["nominal_covariance_semantic_sha256"]
            == content["tensor_semantic_sha256"]["nominal_covariance"]
        )
        load_q_final_json(path)
