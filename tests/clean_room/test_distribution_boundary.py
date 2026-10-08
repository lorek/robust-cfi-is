from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_q_final_distribution_exclusion_is_explicit():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "include-package-data = false" in pyproject
    assert "include CITATION.cff" in manifest
    assert "prune reproducibility/q_final" in manifest
