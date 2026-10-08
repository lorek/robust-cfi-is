import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "build_clean_export.py"


def _source(root: Path) -> Path:
    source = root / "source"
    (source / "src" / "example").mkdir(parents=True)
    (source / "docs").mkdir()
    (source / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (source / "README.md").write_text("# Public project\n", encoding="utf-8")
    (source / "docs" / "results.md").write_text("# Results\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        '[project]\nname = "example"\n', encoding="utf-8"
    )
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / "src" / "example" / "__init__.py").write_text(
        'VALUE = "public"\n', encoding="utf-8"
    )
    (source / "docs" / "release.md").write_text(
        "Offline release.\n", encoding="utf-8"
    )
    return source


def _run(source: Path, destination: Path, receipt: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--receipt",
            str(receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_clean_export_is_deterministic_relative_and_source_preserving(tmp_path):
    source = _source(tmp_path)
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    first = _run(source, tmp_path / "first", tmp_path / "first-receipt.json")
    second = _run(source, tmp_path / "second", tmp_path / "second-receipt.json")

    assert first.returncode == second.returncode == 0
    first_receipt = (tmp_path / "first-receipt.json").read_bytes()
    assert first_receipt == (tmp_path / "second-receipt.json").read_bytes()
    receipt = json.loads(first_receipt)
    assert receipt["sensitive_data_scan"] == "PASS"
    assert receipt["source_unchanged"] is True
    assert receipt["file_count"] == len(before)
    assert all(not Path(row["path"]).is_absolute() for row in receipt["files"])
    assert str(source).encode() not in first_receipt
    assert str(tmp_path / "first").encode() not in first_receipt
    assert before == {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    for relative, payload in before.items():
        assert (tmp_path / "first" / relative).read_bytes() == payload


def test_clean_export_rejects_existing_destination(tmp_path):
    source = _source(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    result = _run(source, destination, tmp_path / "receipt.json")
    assert result.returncode != 0
    assert "already exists" in result.stderr


def test_clean_export_rejects_unexpected_file_type(tmp_path):
    source = _source(tmp_path)
    (source / "docs" / "payload.bin").write_bytes(b"not public source")
    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")
    assert result.returncode != 0
    assert "Unsupported public file type" in result.stderr


def test_clean_export_copies_png_bytes_exactly(tmp_path):
    source = _source(tmp_path)
    image = source / "docs" / "assets" / "figure.png"
    image.parent.mkdir()
    payload = b"\x89PNG\r\n\x1a\npublic-paper-figure"
    image.write_bytes(payload)

    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")

    assert result.returncode == 0
    exported = tmp_path / "export" / "docs" / "assets" / "figure.png"
    assert exported.read_bytes() == payload


def test_clean_export_copies_gif_bytes_exactly(tmp_path):
    source = _source(tmp_path)
    image = source / "docs" / "assets" / "animation.gif"
    image.parent.mkdir()
    payload = b"GIF89a" + b"public-paper-animation"
    image.write_bytes(payload)

    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")

    assert result.returncode == 0
    exported = tmp_path / "export" / "docs" / "assets" / "animation.gif"
    assert exported.read_bytes() == payload


def test_clean_export_rejects_invalid_png_signature(tmp_path):
    source = _source(tmp_path)
    image = source / "docs" / "assets" / "figure.png"
    image.parent.mkdir()
    image.write_bytes(b"not a PNG")

    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")

    assert result.returncode != 0
    assert "Invalid PNG signature" in result.stderr


def test_clean_export_rejects_invalid_gif_signature(tmp_path):
    source = _source(tmp_path)
    image = source / "docs" / "assets" / "animation.gif"
    image.parent.mkdir()
    image.write_bytes(b"not a GIF")

    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")

    assert result.returncode != 0
    assert "Invalid GIF signature" in result.stderr


def test_clean_export_rejects_png_outside_docs_assets(tmp_path):
    source = _source(tmp_path)
    image = source / "docs" / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\npublic-paper-figure")

    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")

    assert result.returncode != 0
    assert "Binary public assets must be under docs/assets" in result.stderr


def test_clean_export_rejects_symlink(tmp_path):
    source = _source(tmp_path)
    link = source / "docs" / "readme-link.md"
    try:
        link.symlink_to(source / "README.md")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")
    assert result.returncode != 0
    assert "Symlinks are not permitted" in result.stderr


@pytest.mark.parametrize("kind", ["home_path", "gpu_uuid", "edit_url"])
def test_clean_export_rejects_sensitive_receipt_material(tmp_path, kind):
    source = _source(tmp_path)
    if kind == "home_path":
        unsafe = "/" + "home" + "/" + "local-user" + "/runs/result.json"
    elif kind == "gpu_uuid":
        unsafe = "GPU-" + "a" * 32
    else:
        unsafe = (
            "https" + "://example.invalid/document/" + "edit?token=" + "x" * 24
        )
    (source / "docs" / "unsafe.json").write_text(
        json.dumps({"environment_value": unsafe}), encoding="utf-8"
    )
    result = _run(source, tmp_path / "export", tmp_path / "receipt.json")
    assert result.returncode != 0
    assert "Sensitive-data scan failed" in result.stderr
