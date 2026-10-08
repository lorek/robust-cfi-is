import pytest

from robust_cfi_is.estimation import aggregate_evaluation_rows


def _rows():
    return [
        {
            "evaluation_seed": seed,
            "estimate": float(seed),
            "log_error": float(seed) / 10,
            "cov": float(seed) / 100,
            "cmc_event_count": 0 if seed < 5 else 1,
            "variance_reduction": None if seed < 5 else float(seed) * 2,
        }
        for seed in range(20)
    ]


def test_metric_specific_cmc_accounting():
    result = aggregate_evaluation_rows(_rows())
    assert result["evaluation_run_count"] == 20
    assert result["estimate_mean_all_20"] == 9.5
    assert result["cmc_positive_count"] == 15
    assert result["mean_variance_reduction_positive_cmc"] == 24.0


def test_aggregation_rejects_missing_or_duplicate_runs():
    with pytest.raises(ValueError, match="exactly 20"):
        aggregate_evaluation_rows(_rows()[:-1])
    rows = _rows()
    rows[-1]["evaluation_seed"] = 0
    with pytest.raises(ValueError, match="unique"):
        aggregate_evaluation_rows(rows)
