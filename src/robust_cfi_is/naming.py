"""Mathematically correct public method names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    """Stable public identity for a supported method."""

    slug: str
    display_name: str
    objective: str
    constrained: bool
    available: bool = True
    defensive: bool = False
    constrained_gmm_backbone: bool = False
    final_proposal_family: str = "gmm"
    theorem_certified_final_proposal: bool = False


_METHODS = {
    "frkl_u": MethodSpec(
        "frkl_u", "fRKL-u", "reverse_kl", False,
    ),
    "ffkl_u": MethodSpec(
        "ffkl_u", "fFKL-u", "forward_kl", False,
    ),
    "frkl_c": MethodSpec(
        "frkl_c", "fRKL-c", "reverse_kl", True,
        constrained_gmm_backbone=True, theorem_certified_final_proposal=True,
    ),
    "ffkl_c": MethodSpec(
        "ffkl_c", "fFKL-c", "forward_kl", True,
        constrained_gmm_backbone=True, theorem_certified_final_proposal=True,
    ),
    "cfi_u": MethodSpec(
        "cfi_u", "CFI-u", "coverage_first", False,
    ),
    "cfi_c": MethodSpec(
        "cfi_c", "CFI-c", "coverage_first_constrained", True,
        constrained_gmm_backbone=True, theorem_certified_final_proposal=True,
    ),
    "cfi_c_frkl_c": MethodSpec(
        "cfi_c_frkl_c", "CFI-c+RKL-c", "cfi_then_reverse_kl", True,
        constrained_gmm_backbone=True, theorem_certified_final_proposal=True,
    ),
    "cfi_c_ffkl_c": MethodSpec(
        "cfi_c_ffkl_c", "CFI-c+FKL-c", "cfi_then_forward_kl", True,
        constrained_gmm_backbone=True, theorem_certified_final_proposal=True,
    ),
    "frkl_d": MethodSpec(
        "frkl_d", "fRKL-d", "reverse_kl", False, defensive=True,
    ),
    "cfi_c_frkl_c_flow": MethodSpec(
        "cfi_c_frkl_c_flow", "CFI-c+RKL-c+F", "cfi_then_reverse_kl_then_flow",
        False,
        constrained_gmm_backbone=True, final_proposal_family="flow",
        theorem_certified_final_proposal=False,
    ),
    "cfi_c_ffkl_c_flow": MethodSpec(
        "cfi_c_ffkl_c_flow", "CFI-c+FKL-c+F", "cfi_then_forward_kl_then_flow",
        False,
        constrained_gmm_backbone=True, final_proposal_family="flow",
        theorem_certified_final_proposal=False,
    ),
}

def get_method(slug: str) -> MethodSpec:
    """Return a public method specification."""

    try:
        method = _METHODS[slug]
    except KeyError as exc:
        supported = ", ".join(sorted(_METHODS))
        raise ValueError(
            f"Unknown public method {slug!r}; supported methods: {supported}"
        ) from exc
    if not method.available:
        raise ValueError(f"Public method {slug!r} is not available in this release")
    return method


def supported_methods() -> tuple[MethodSpec, ...]:
    return tuple(_METHODS[key] for key in sorted(_METHODS))
