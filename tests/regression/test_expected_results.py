import json
from pathlib import Path


ROOT = Path(__file__).parents[2] / "reproducibility/expected_results"


def _load(name):
    return json.loads((ROOT / name).read_text())


def test_defensive_expected_sweep_is_exact():
    data = _load("defensive_g1.json")
    assert data["method"] == "frkl_d"
    assert [row["defensive_weight"] for row in data["rows"]] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert data["rows"][1]["mean_cov"] == 0.0754495826565223


def test_corrected_ffkl_expected_results_use_20_runs():
    data = _load("ffkl_c_final_is.json")
    assert data["method"] == "ffkl_c"
    assert len(data["cells"]) == 4
    assert all(cell["evaluation_seed_count"] == 20 for cell in data["cells"])


def test_headline_selections_match_reference_full_precision_rows():
    data = _load("headline_selections.json")
    assert data["exact_tie_policy"] == "error"
    assert len(data["selections"]) == 7
    selected = {row["setting"]: row["method"] for row in data["selections"]}
    assert selected == {
        "g1": "cfi_c_frkl_c", "g2": "cfi_c_ffkl_c",
        "g3": "cfi_c_ffkl_c", "g5": "cfi_c_frkl_c",
        "g6": "cfi_c_frkl_c", "g7": "cfi_c_frkl_c",
        "g10": "cfi_c_ffkl_c",
    }
