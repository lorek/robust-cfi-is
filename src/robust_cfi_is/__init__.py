"""Public API for robust coverage-first importance sampling."""

from robust_cfi_is.api import PublicMethodFit, fit_public_method, run_config, run_quickstart
from robust_cfi_is.benchmarks import get_benchmark, get_g7_benchmark
from robust_cfi_is.estimation import ISResult, ordinary_is
from robust_cfi_is.methods import (
    CFIFitResult,
    ForwardKLFitResult,
    FlowFitResult,
    ReverseKLFitResult,
    fit_cfi,
    fit_forward_kl,
    fit_fixed_proposal_flow,
    fit_reverse_kl,
    refine_forward_kl,
    refine_reverse_kl,
)
from robust_cfi_is.naming import MethodSpec, get_method
from robust_cfi_is.proposals import (
    CertifiedGaussianMixture,
    ConstrainedGaussianMixture,
    UnconstrainedGaussianMixture,
    RealNVPProposal,
    certify_final_gmm,
)

__all__ = [
    "CFIFitResult",
    "CertifiedGaussianMixture",
    "ConstrainedGaussianMixture",
    "ForwardKLFitResult",
    "FlowFitResult",
    "ISResult",
    "MethodSpec",
    "PublicMethodFit",
    "ReverseKLFitResult",
    "UnconstrainedGaussianMixture",
    "RealNVPProposal",
    "certify_final_gmm",
    "fit_cfi",
    "fit_forward_kl",
    "fit_fixed_proposal_flow",
    "fit_public_method",
    "fit_reverse_kl",
    "get_benchmark",
    "get_g7_benchmark",
    "get_method",
    "ordinary_is",
    "refine_forward_kl",
    "refine_reverse_kl",
    "run_config",
    "run_quickstart",
]

__version__ = "0.1.0"
