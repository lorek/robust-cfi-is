"""Metric-specific aggregation for public reproducibility records."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Iterable


def aggregate_evaluation_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate 20-run records using the documented CMC accounting contract."""

    values = list(rows)
    if len(values) != 20:
        raise ValueError("Expected exactly 20 evaluation rows")
    seeds = [row.get("evaluation_seed") for row in values]
    if len(set(seeds)) != 20:
        raise ValueError("Evaluation seeds must be unique")
    estimates = [float(row["estimate"]) for row in values]
    log_errors = [float(row["log_error"]) for row in values]
    covs = [float(row["cov"]) for row in values]
    if not all(math.isfinite(value) for value in estimates + log_errors + covs):
        raise ValueError("Non-CMC metrics must be defined for all 20 runs")
    positive = [row for row in values if int(row["cmc_event_count"]) > 0]
    variance_reductions = [float(row["variance_reduction"]) for row in positive]
    if not all(math.isfinite(value) for value in variance_reductions):
        raise ValueError("Positive-CMC variance reduction must be defined")
    return {
        "evaluation_run_count": 20,
        "estimate_mean_all_20": fmean(estimates),
        "mean_log_error_all_20": fmean(log_errors),
        "mean_cov_all_20": fmean(covs),
        "cmc_positive_count": len(positive),
        "cmc_positive_denominator": 20,
        "mean_variance_reduction_positive_cmc": (
            fmean(variance_reductions) if variance_reductions else None
        ),
    }
