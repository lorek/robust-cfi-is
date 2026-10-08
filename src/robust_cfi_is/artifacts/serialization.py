"""Safe JSON loader for canonical certified proposals stored as q_final artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robust_cfi_is.artifacts.identities import ARTIFACT_KIND
from robust_cfi_is.naming import get_method
from robust_cfi_is.proposals import CertifiedGaussianMixture
from robust_cfi_is.proposals import ETA, TAU_ACCEPT, TAU_BUILD


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _tensor_hash(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)}
    )
    raw = contiguous.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def _decode_tensor(record: dict[str, Any]) -> torch.Tensor:
    if record.get("dtype") != "float32" or record.get("byte_order") != "little":
        raise ValueError("q_final tensors must be little-endian float32")
    shape = record.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(size, int) and size >= 0 for size in shape
    ):
        raise ValueError("Invalid q_final tensor shape")
    try:
        raw = base64.b64decode(record["data_base64"], validate=True)
        array = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid q_final tensor data") from exc
    value = torch.from_numpy(array).to(torch.float32)
    if _tensor_hash(value) != record.get("semantic_sha256"):
        raise ValueError("q_final tensor semantic hash mismatch")
    return value


def load_q_final_json(path: str | Path) -> CertifiedGaussianMixture:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load q_final JSON {path}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
        raise ValueError("Unsupported q_final envelope")
    content = envelope.get("content")
    if not isinstance(content, dict):
        raise ValueError("q_final content is missing")
    expected = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    if envelope.get("artifact_content_sha256") != expected:
        raise ValueError("q_final content hash mismatch")
    if content.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Unsupported q_final artifact kind")
    provenance = content.get("provenance")
    expected_provenance_keys = {
        "beta",
        "covariance_scale_final",
        "final_logical_id",
        "learned_factor_semantic_sha256",
        "nominal_covariance_semantic_sha256",
        "problem_id",
        "reference_materialization",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance_keys:
        raise ValueError("q_final provenance fields are invalid")
    identity = re.fullmatch(
        r"q_final::([A-Za-z0-9_]+)::k([1-9][0-9]*)::([a-z0-9_]+)::seed([0-9]+)",
        str(provenance.get("final_logical_id", "")),
    )
    if identity is None:
        raise ValueError("q_final logical identity is malformed")
    problem_id, component_count, method_slug, seed = identity.groups()
    get_method(method_slug)
    if problem_id != provenance.get("problem_id") or int(seed) != 100:
        raise ValueError("q_final logical identity is inconsistent")
    for key in (
        "learned_factor_semantic_sha256",
        "nominal_covariance_semantic_sha256",
    ):
        if not _SHA256.fullmatch(str(provenance.get(key, ""))):
            raise ValueError("q_final scientific identity is malformed")
    expected_transform = {
        "version": "generalized_nominal_shift_v1",
        "float32_machine_epsilon": 2.0**-23,
        "eta": ETA,
        "tau_build": TAU_BUILD,
        "tau_accept": TAU_ACCEPT,
        "construction_dtype": "float64",
        "proposal_dtype": "float32",
    }
    if content.get("transform") != expected_transform:
        raise ValueError("q_final transform policy mismatch")
    tensors = content.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("q_final tensor records are missing")
    decoded = {
        name: _decode_tensor(tensors[name])
        for name in ("component_logits", "means", "scale_tril", "nominal_covariance")
    }
    hashes = {name: _tensor_hash(value) for name, value in decoded.items()}
    if content.get("tensor_semantic_sha256") != hashes:
        raise ValueError("q_final tensor hash index mismatch")
    proposal = content.get("proposal", {})
    if proposal.get("role") != "q_final" or proposal.get(
        "source_of_truth"
    ) != "serialized_float32_scale_tril":
        raise ValueError("Artifact is not canonical q_final")
    if proposal.get("component_count") != decoded["means"].shape[0] or proposal.get(
        "dimension"
    ) != decoded["means"].shape[1]:
        raise ValueError("q_final artifact dimensions do not match its tensors")
    if int(component_count) != proposal.get("component_count"):
        raise ValueError("q_final logical identity component count is inconsistent")
    certification = content.get("certification")
    if not isinstance(certification, dict) or certification.get("passed") is not True:
        raise ValueError("q_final certification record is not PASS")
    component_records = certification.get("components", [])
    if len(component_records) != decoded["means"].shape[0] or not all(
        row.get("passed") is True for row in component_records
    ):
        raise ValueError("q_final component certification is incomplete")
    return CertifiedGaussianMixture(
        logits=decoded["component_logits"],
        means=decoded["means"],
        scale_tril=decoded["scale_tril"],
        nominal_covariance=decoded["nominal_covariance"],
        certification=certification,
    )
