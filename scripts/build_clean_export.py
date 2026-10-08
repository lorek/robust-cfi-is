#!/usr/bin/env python3
"""Build a deterministic, sensitive-data-scanned public source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


EXPORT_VERSION = "clean_export_v1"
MAX_FILE_SIZE = 2 * 1024 * 1024
ALLOWED_ROOT_FILES = frozenset(
    {
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    }
)
ALLOWED_ROOT_DIRECTORIES = frozenset(
    {
        ".github",
        "configs",
        "docs",
        "examples",
        "reproducibility",
        "scripts",
        "src",
        "tests",
    }
)
IGNORED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "runs",
    }
)
ALLOWED_TEXT_SUFFIXES = frozenset(
    {
        ".cff",
        ".gitignore",
        ".in",
        ".json",
        ".lock",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
        "LICENSE",
    }
)
ALLOWED_BINARY_SUFFIXES = frozenset({".gif", ".png"})
ALLOWED_SUFFIXES = ALLOWED_TEXT_SUFFIXES | ALLOWED_BINARY_SUFFIXES
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_LOCAL_HOME = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")
_GPU_UUID = re.compile(r"GPU-[0-9a-f-]{16,}", re.IGNORECASE)
_SECRET_KEY_LABEL = bytes.fromhex("50524956415445204b4559").decode("ascii")
_TOKEN = re.compile(
    r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?"
    + re.escape(_SECRET_KEY_LABEL)
    + r"-----)"
)
_SENSITIVE_URL = re.compile(
    r"https?://[^\s]*(?:access_token=|authuser=|token=|/edit(?:[/?#]|$))",
    re.IGNORECASE,
)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ignored(relative: Path) -> bool:
    return any(
        part in IGNORED_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    )


def _sensitive_findings(relative: Path, text: str) -> tuple[str, ...]:
    findings = []
    if _LOCAL_HOME.search(text):
        findings.append("absolute_local_home_path")
    if _GPU_UUID.search(text):
        findings.append("gpu_uuid")
    if _TOKEN.search(text):
        findings.append("credential_or_secret_key")
    if _SENSITIVE_URL.search(text):
        findings.append("credentialed_or_edit_url")
    return tuple(findings)


def inventory_source(source: Path) -> tuple[dict[str, Any], ...]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError("Clean-export source must be a directory")
    rows = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if _ignored(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted: {relative.as_posix()}")
        if path.is_dir():
            if (
                len(relative.parts) == 1
                and relative.name not in ALLOWED_ROOT_DIRECTORIES
            ):
                raise ValueError(f"Unexpected top-level directory: {relative.as_posix()}")
            continue
        if not path.is_file():
            raise ValueError(f"Unsupported filesystem entry: {relative.as_posix()}")
        if len(relative.parts) == 1:
            if relative.name not in ALLOWED_ROOT_FILES:
                raise ValueError(f"Unexpected top-level file: {relative.as_posix()}")
        elif relative.parts[0] not in ALLOWED_ROOT_DIRECTORIES:
            raise ValueError(f"File is outside the export allowlist: {relative.as_posix()}")
        suffix = path.suffix if path.suffix else path.name
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported public file type: {relative.as_posix()}")
        payload = path.read_bytes()
        if len(payload) > MAX_FILE_SIZE:
            raise ValueError(f"Public source file exceeds the size limit: {relative.as_posix()}")
        if suffix in ALLOWED_BINARY_SUFFIXES:
            if relative.parts[:2] != ("docs", "assets"):
                raise ValueError(
                    "Binary public assets must be under docs/assets: "
                    f"{relative.as_posix()}"
                )
            if suffix == ".png" and not payload.startswith(PNG_SIGNATURE):
                raise ValueError(f"Invalid PNG signature: {relative.as_posix()}")
            if suffix == ".gif" and not payload.startswith(GIF_SIGNATURES):
                raise ValueError(f"Invalid GIF signature: {relative.as_posix()}")
            # Scan uncompressed metadata and other ASCII-compatible material
            # without treating binary image data as public UTF-8 source.
            text = payload.decode("latin-1")
        else:
            if b"\0" in payload:
                raise ValueError(
                    f"Binary content is not permitted: {relative.as_posix()}"
                )
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Public source must be UTF-8: {relative.as_posix()}"
                ) from exc
        findings = _sensitive_findings(relative, text)
        if findings:
            raise ValueError(
                f"Sensitive-data scan failed for {relative.as_posix()}: {list(findings)}"
            )
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(rows)


def tree_sha256(rows: tuple[dict[str, Any], ...]) -> str:
    material = b"".join(
        f"{row['sha256']} {row['size_bytes']} {row['path']}\n".encode("utf-8")
        for row in rows
    )
    return _sha256(material)


def build_clean_export(
    *, source: Path, destination: Path, receipt: Path
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    receipt = receipt.resolve()
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Source and destination must be separate non-overlapping trees")
    if receipt == source or source in receipt.parents or destination in receipt.parents:
        raise ValueError("Receipt must be outside source and destination trees")
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    if receipt.exists():
        raise FileExistsError(f"Export receipt already exists: {receipt}")

    before = inventory_source(source)
    destination.mkdir(parents=True)
    for row in before:
        relative = Path(row["path"])
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, output)
        os.chmod(output, 0o644)
    after = inventory_source(source)
    exported = inventory_source(destination)
    if before != after:
        raise RuntimeError("Source tree changed while building the clean export")
    if before != exported:
        raise RuntimeError("Clean export inventory does not match its source")

    result = {
        "export_version": EXPORT_VERSION,
        "file_count": len(exported),
        "files": list(exported),
        "sensitive_data_scan": "PASS",
        "source_unchanged": True,
        "tree_sha256": tree_sha256(exported),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("xb") as stream:
        stream.write(_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_clean_export(
        source=args.source,
        destination=args.destination,
        receipt=args.receipt,
    )
    summary = {
        key: result[key]
        for key in ("export_version", "file_count", "sensitive_data_scan", "tree_sha256")
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
