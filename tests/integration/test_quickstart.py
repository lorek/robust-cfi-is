import json
from pathlib import Path

import pytest

from robust_cfi_is import run_config


ROOT = Path(__file__).resolve().parents[2]


def test_approved_g1_quickstart(tmp_path):
    expected = json.loads(
        (ROOT / "reproducibility" / "expected_results" / "quickstart.json").read_text()
    )
    result = run_config(
        ROOT / "configs" / "examples" / "g1_cfi_c.yaml",
        output=tmp_path / "result.json",
    )
    assert result["method"]["slug"] == "cfi_c"
    assert result["cfi"]["target_reached"]
    assert result["certification"]["passed"]
    assert result["certification"]["minimum_theorem_margin"] > 0.0
    estimate = result["importance_sampling"]
    assert estimate["estimator"] == "ordinary_unnormalized_importance_sampling"
    assert estimate["self_normalized"] is False
    assert estimate["sample_size"] == 20000
    assert estimate["estimate"] > 0.0
    assert estimate["standard_error"] > 0.0
    assert estimate["cov"] > 0.0
    assert estimate["event_count"] > 0
    reference = expected["reference_result"]
    envelope = expected["validation_envelope"]
    assert [row["elite_count"] for row in result["cfi"]["levels"]] == reference[
        "cfi_elite_count_sequence"
    ]
    assert [row["beta"] for row in result["cfi"]["levels"]] == reference[
        "cfi_beta_sequence"
    ]
    gamma_sequence = [row["gamma"] for row in result["cfi"]["levels"]]
    expected_gamma_sequence = reference["cfi_gamma_sequence"]
    assert len(gamma_sequence) == len(expected_gamma_sequence)
    # Float32 gamma updates vary slightly by platform; counts and betas stay exact.
    assert gamma_sequence == pytest.approx(
        expected_gamma_sequence, rel=1e-4, abs=1e-4
    )
    assert result["cfi"]["final_fit_elite_count"] == reference[
        "final_fit_elite_count"
    ]
    relative_error = abs(estimate["estimate"] - reference["estimate"]) / reference[
        "estimate"
    ]
    assert relative_error <= envelope["estimate_relative_tolerance"]
    assert (
        abs(estimate["event_count"] - reference["event_count"])
        <= envelope["event_count_absolute_tolerance"]
    )
    assert (
        result["certification"]["minimum_theorem_margin"]
        >= envelope["minimum_theorem_margin_lower_bound"]
    )
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved == result
