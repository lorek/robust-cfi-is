"""Provider-neutral validation for future public artifact mirrors."""

from __future__ import annotations

from urllib.parse import urlsplit

from robust_cfi_is.artifacts.manifest import ArtifactManifest, ArtifactRecord


def artifact_identity(record: ArtifactRecord) -> tuple[object, ...]:
    """Return hosting-independent fields that must never change with a mirror."""
    return (
        record.name, record.sha256, record.size_bytes, record.benchmark,
        record.method, record.components, record.training_seed,
        record.artifact_kind, record.certification,
    )


def validate_mirror_updates(
    manifest: ArtifactManifest, updates: dict[str, list[str] | tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Validate proposed public URLs without modifying the manifest or artifacts."""
    result: dict[str, tuple[str, ...]] = {}
    for name, urls in updates.items():
        manifest.get(name)
        if not urls:
            raise ValueError("mirror update must contain at least one URL")
        validated = []
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("public mirrors must use absolute HTTPS URLs")
            if parsed.username or parsed.password or parsed.fragment:
                raise ValueError("public mirrors must not contain credentials or fragments")
            validated.append(url)
        result[name] = tuple(validated)
    return result
