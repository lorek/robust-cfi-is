"""Public fitting methods with explicit mathematical directions."""

from robust_cfi_is.methods.cfi import CFILevel, CFIFitResult, fit_cfi
from robust_cfi_is.methods.defensive_reverse_kl import fit_defensive_reverse_kl
from robust_cfi_is.methods.forward_kl import (
    ForwardKLFitResult,
    ForwardKLLevel,
    fit_forward_kl,
    mean_one_adaptive_ce_weights,
    mean_one_fixed_proposal_weights,
    refine_forward_kl,
    weighted_forward_kl_loss,
)
from robust_cfi_is.methods.reverse_kl import (
    ReverseKLFitResult,
    ReverseKLStep,
    fit_reverse_kl,
    refine_reverse_kl,
    reverse_kl_surrogate,
)
from robust_cfi_is.methods.flow import FlowFitResult, fit_fixed_proposal_flow

__all__ = [
    "CFIFitResult",
    "CFILevel",
    "ForwardKLFitResult",
    "ForwardKLLevel",
    "FlowFitResult",
    "ReverseKLFitResult",
    "ReverseKLStep",
    "fit_cfi",
    "fit_defensive_reverse_kl",
    "fit_forward_kl",
    "fit_fixed_proposal_flow",
    "fit_reverse_kl",
    "mean_one_adaptive_ce_weights",
    "mean_one_fixed_proposal_weights",
    "refine_forward_kl",
    "refine_reverse_kl",
    "reverse_kl_surrogate",
    "weighted_forward_kl_loss",
]
