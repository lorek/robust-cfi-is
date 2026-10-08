from pathlib import Path
import tomllib

import yaml

import robust_cfi_is


ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"
REPOSITORY = "https://github.com/lorek/robust-cfi-is"
TITLE = "Robust Importance Sampling for Rare Events via Constrained Gaussian Mixtures"
CONFERENCE = "40th Conference on Neural Information Processing Systems (NeurIPS 2026)"
AUTHORS = [
    ("Paweł", "Lorek"),
    ("Rafał", "Nowak"),
    ("Rafał", "Topolnicki"),
    ("Tomasz", "Trzciński"),
    ("Maciej", "Zięba"),
]


def _cff_authors(entries):
    return [(entry["given-names"], entry["family-names"]) for entry in entries]


def test_release_metadata_is_consistent():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == TITLE
    assert citation["version"] == VERSION
    assert citation["date-released"] == "2026-10-04"
    assert citation["license"] == "MIT"
    assert citation["repository-code"] == REPOSITORY
    assert _cff_authors(citation["authors"]) == AUTHORS

    preferred = citation["preferred-citation"]
    assert preferred["type"] == "conference-paper"
    assert preferred["title"] == TITLE
    assert preferred["year"] == 2026
    assert preferred["conference"] == {"name": CONFERENCE}
    assert _cff_authors(preferred["authors"]) == AUTHORS

    assert project["version"] == VERSION
    assert [entry["name"] for entry in project["authors"]] == [
        f"{given} {family}" for given, family in AUTHORS
    ]
    assert project["urls"] == {"Repository": REPOSITORY}
    assert robust_cfi_is.__version__ == VERSION
