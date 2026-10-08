"""Strict offline manifest access."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_cfi_is.artifacts.identities import ARTIFACT_KIND
from robust_cfi_is.naming import get_method

_SHA256 = re.compile(r"[0-9a-f]{64}")
_AVAILABILITY = frozenset(
    {"pending_publication", "repository_embedded", "published"}
)


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    benchmark: str
    method: str
    display_name: str
    components: int
    training_seed: int
    artifact_kind: str
    sha256: str
    size_bytes: int
    certification: dict[str, Any]
    workflow_status: str
    availability: str
    mirrors: tuple[str, ...]


class ArtifactManifest:
    def __init__(self, payload: dict[str, Any]) -> None:
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError("Unsupported artifact manifest schema")
        rows = payload.get("artifacts")
        if not isinstance(rows, list):
            raise ValueError("Artifact manifest needs an artifacts list")
        records: list[ArtifactRecord] = []
        names: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Artifact records must be mappings")
            method = get_method(row.get("method"))
            if row.get("display_name") != method.display_name:
                raise ValueError("Artifact display/public method mismatch")
            if row.get("name") in names:
                raise ValueError(f"Duplicate artifact name {row.get('name')!r}")
            if row.get("artifact_kind") != ARTIFACT_KIND:
                raise ValueError("Artifact kind is unsupported")
            if not isinstance(row.get("sha256"), str) or not _SHA256.fullmatch(
                row["sha256"]
            ):
                raise ValueError("Artifact SHA-256 is malformed")
            if not isinstance(row.get("size_bytes"), int) or row["size_bytes"] <= 0:
                raise ValueError("Artifact size must be positive")
            if (
                not isinstance(row.get("components"), int)
                or row["components"] <= 0
                or row.get("training_seed") != 100
            ):
                raise ValueError("Artifact K/seed is unsupported")
            mirrors = row.get("mirrors")
            if not isinstance(mirrors, list) or not all(
                isinstance(item, str) for item in mirrors
            ):
                raise ValueError("Artifact mirrors must be a string list")
            availability = row.get("availability")
            if availability not in _AVAILABILITY:
                raise ValueError("Artifact availability is unsupported")
            if availability != "published" and mirrors:
                raise ValueError("Non-published artifacts must not advertise mirrors")
            certification = row.get("certification")
            if not isinstance(certification, dict) or certification.get("passed") is not True:
                raise ValueError("Artifact certification must record PASS")
            if not _SHA256.fullmatch(certification.get("artifact_content_sha256", "")):
                raise ValueError("Artifact certification content hash is malformed")
            names.add(row["name"])
            records.append(ArtifactRecord(**{**row, "mirrors": tuple(mirrors)}))
        self.schema_version = schema_version
        self.cohort = payload.get("cohort", {})
        if self.cohort != {
            "artifact_count": len(records),
            "availability": "repository_embedded",
            "certification_passed_count": len(records),
            "generation": "certified_final",
            "name": "certified-q-final",
        }:
            raise ValueError("Artifact cohort metadata is invalid")
        if any(record.availability != self.cohort["availability"] for record in records):
            raise ValueError("Artifact availability does not match its cohort")
        self.records = tuple(records)
        self._by_name = {record.name: record for record in records}

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load artifact manifest {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Artifact manifest root must be a mapping")
        return cls(payload)

    def get(self, name: str) -> ArtifactRecord:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(f"Unknown public artifact {name!r}") from exc
