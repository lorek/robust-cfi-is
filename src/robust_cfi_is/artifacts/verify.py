"""Content-identity verification for local artifact files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from robust_cfi_is.artifacts.manifest import ArtifactRecord


_PUBLIC_ARTIFACT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(record: ArtifactRecord, path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"Artifact does not exist: {candidate}")
    if candidate.stat().st_size != record.size_bytes:
        raise ValueError("Artifact size mismatch")
    if sha256_file(candidate) != record.sha256:
        raise ValueError("Artifact SHA-256 mismatch")
    return candidate


def resolve_artifact_path(
    manifest_path: str | Path,
    record: ArtifactRecord,
    explicit_path: str | Path | None = None,
) -> Path:
    """Resolve an explicit path or the repository-local q_final convention."""

    if explicit_path is not None:
        return Path(explicit_path)
    if not _PUBLIC_ARTIFACT_NAME.fullmatch(record.name):
        raise ValueError("Artifact name is not a canonical public slug")
    artifact_root = (Path(manifest_path).parent / "q_final").resolve()
    candidate = (artifact_root / f"{record.name}.json").resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("Repository artifact path escapes q_final") from exc
    return candidate
