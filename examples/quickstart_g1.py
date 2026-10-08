"""Run the CPU-only constrained-CFI g1 quickstart."""

from __future__ import annotations

import json
from pathlib import Path

from robust_cfi_is import run_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = run_config(
        root / "configs" / "examples" / "g1_cfi_c.yaml",
        output=Path.cwd() / "runs" / "quickstart_g1.json",
    )
    summary = {
        "estimate": result["importance_sampling"]["estimate"],
        "standard_error": result["importance_sampling"]["standard_error"],
        "CoV": result["importance_sampling"]["cov"],
        "event_count": result["importance_sampling"]["event_count"],
        "minimum_covariance_margin": result["certification"][
            "minimum_theorem_margin"
        ],
        "runtime_seconds": result["runtime_seconds"],
        "result_path": str(Path.cwd() / "runs" / "quickstart_g1.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
