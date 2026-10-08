"""Canonical final-GMM construction and operational certification."""

from __future__ import annotations

import math

import torch

from robust_cfi_is.proposals.gmm import GaussianMixtureProposal


FLOAT32_MACHINE_EPSILON = 2.0**-23
ETA = math.sqrt(FLOAT32_MACHINE_EPSILON)
TAU_BUILD = 0.5 + 2.0 * ETA
TAU_ACCEPT = 0.5 + ETA


class CertificationError(ValueError):
    """Raised when a certified proposal fails operational certification."""


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _validated_nominal(nominal: torch.Tensor, dimension: int) -> torch.Tensor:
    value = torch.as_tensor(nominal, dtype=torch.float32, device="cpu").clone()
    if value.shape != (dimension, dimension) or not bool(torch.isfinite(value).all()):
        raise CertificationError("Nominal covariance is malformed")
    try:
        torch.linalg.cholesky(_symmetrize(value.to(torch.float64)))
    except RuntimeError as exc:
        raise CertificationError("Nominal covariance must be positive definite") from exc
    return value


def _generalized_eigen_report(
    covariance: torch.Tensor,
    nominal: torch.Tensor,
) -> dict[str, float]:
    covariance64 = _symmetrize(covariance.to(torch.float64))
    nominal64 = _symmetrize(nominal.to(torch.float64))
    nominal_factor = torch.linalg.cholesky(nominal64)
    left = torch.linalg.solve_triangular(
        nominal_factor,
        covariance64,
        upper=False,
    )
    whitened = torch.linalg.solve_triangular(
        nominal_factor,
        left.T,
        upper=False,
    ).T
    eigenvalues = torch.linalg.eigvalsh(_symmetrize(whitened))
    return {
        "minimum": float(eigenvalues.min()),
        "maximum": float(eigenvalues.max()),
    }


def _strict_spd_report(matrix: torch.Tensor) -> dict[str, float | bool | int | None]:
    value64 = _symmetrize(matrix.to(torch.float64))
    factor, info = torch.linalg.cholesky_ex(
        value64,
        upper=False,
        check_errors=False,
    )
    passed = bool(torch.all(info == 0))
    return {
        "passed": passed,
        "info": int(info.item()),
        "minimum_factor_diagonal": (
            float(torch.diagonal(factor).min()) if passed else None
        ),
    }


class CertifiedGaussianMixture(GaussianMixtureProposal):
    """Immutable inference proposal defined by canonical float32 factors."""

    def __init__(
        self,
        *,
        logits: torch.Tensor,
        means: torch.Tensor,
        scale_tril: torch.Tensor,
        nominal_covariance: torch.Tensor,
        certification: dict,
    ) -> None:
        super().__init__(logits=logits, means=means)
        factors = torch.as_tensor(
            scale_tril, dtype=torch.float32, device="cpu"
        ).clone()
        nominal = _validated_nominal(nominal_covariance, self.dimension)
        if factors.shape != (
            self.n_components,
            self.dimension,
            self.dimension,
        ):
            raise CertificationError("Canonical factors have the wrong shape")
        if torch.count_nonzero(torch.triu(factors, diagonal=1)).item() != 0:
            raise CertificationError("Canonical factors must be exactly lower triangular")
        if not bool(torch.all(torch.diagonal(factors, dim1=-2, dim2=-1) > 0)):
            raise CertificationError("Canonical factor diagonals must be positive")
        self._scale_tril = factors
        self.nominal_covariance = nominal
        self.certification = certification

    def covariances(self) -> torch.Tensor:
        return self._scale_tril @ self._scale_tril.transpose(-1, -2)

    def scale_tril(self) -> torch.Tensor:
        return self._scale_tril.clone()


def _certify_factors(
    factors: torch.Tensor,
    nominal: torch.Tensor,
) -> dict:
    components = []
    for index, factor32 in enumerate(factors):
        covariance64 = factor32.to(torch.float64) @ factor32.to(torch.float64).T
        eigen = _generalized_eigen_report(covariance64, nominal.to(torch.float64))
        strict_spd = _strict_spd_report(
            covariance64 - 0.5 * nominal.to(torch.float64)
        )
        accepted = bool(eigen["minimum"] >= TAU_ACCEPT)
        passed = accepted and bool(strict_spd["passed"])
        components.append(
            {
                "component_index": index,
                "post_serialization_lambda_min": eigen["minimum"],
                "post_serialization_lambda_max": eigen["maximum"],
                "acceptance_margin": eigen["minimum"] - TAU_ACCEPT,
                "theorem_margin": eigen["minimum"] - 0.5,
                "tau_accept_gate": accepted,
                "strict_spd": strict_spd,
                "passed": passed,
            }
        )
    failures = [row for row in components if not row["passed"]]
    if failures:
        details = "; ".join(
            f"component {row['component_index']} lambda_min="
            f"{row['post_serialization_lambda_min']:.17g}"
            for row in failures
        )
        raise CertificationError(f"Final GMM failed certification: {details}")
    return {
        "policy": "production_operational_certification_v1",
        "formal_verification": False,
        "tau_build": TAU_BUILD,
        "tau_accept": TAU_ACCEPT,
        "strict_theorem_boundary": 0.5,
        "comparison": "greater_than_or_equal",
        "no_isclose_or_theorem_tolerance": True,
        "components": components,
        "minimum_lambda": min(
            row["post_serialization_lambda_min"] for row in components
        ),
        "minimum_theorem_margin": min(row["theorem_margin"] for row in components),
        "minimum_acceptance_margin": min(
            row["acceptance_margin"] for row in components
        ),
        "passed": True,
    }


def certify_final_gmm(
    learned: "ConstrainedGaussianMixture",
    nominal_covariance: torch.Tensor | None = None,
) -> CertifiedGaussianMixture:
    """Construct canonical ``q_final`` and certify its serialized semantics.

    The learned operational float32 covariance is shifted in float64 only when
    needed to reach ``TAU_BUILD``. Each float64 Cholesky is then cast to a
    canonical lower-triangular float32 factor. Certification reconstructs the
    covariance from that factor in float64 and applies the independent
    generalized-eigenvalue and strict-SPD checks.
    """

    # Local import avoids making the training proposal part of this module's
    # public type hierarchy.
    from robust_cfi_is.proposals.constrained_gmm import ConstrainedGaussianMixture

    if type(learned) is not ConstrainedGaussianMixture:
        raise TypeError("Only a constrained Gaussian mixture can be certified")
    nominal = _validated_nominal(
        learned.nominal_covariance
        if nominal_covariance is None
        else nominal_covariance,
        learned.dimension,
    )
    learned_covariances = learned.covariances().detach().to(torch.float32)
    if not bool(torch.isfinite(learned_covariances).all()):
        raise CertificationError("Learned operational covariances must be finite")

    nominal64 = nominal.to(torch.float64)
    factors = []
    transforms = []
    for index, covariance32 in enumerate(learned_covariances):
        learned64 = _symmetrize(covariance32.to(torch.float64))
        before = _generalized_eigen_report(learned64, nominal64)
        alpha = max(0.0, TAU_BUILD - before["minimum"])
        target64 = _symmetrize(learned64 + alpha * nominal64)
        try:
            factor64 = torch.linalg.cholesky(target64)
        except RuntimeError as exc:
            raise CertificationError(
                f"Float64 final-proposal Cholesky failed for component {index}"
            ) from exc
        factor32 = torch.tril(factor64.to(torch.float32)).contiguous()
        factors.append(factor32)
        transforms.append(
            {
                "component_index": index,
                "pre_transform_lambda_min": before["minimum"],
                "alpha": alpha,
                "changed": bool(alpha > 0.0),
            }
        )
    canonical = torch.stack(factors).contiguous()
    certification = _certify_factors(canonical, nominal)
    certification["transforms"] = transforms
    return CertifiedGaussianMixture(
        logits=learned.logits,
        means=learned.means,
        scale_tril=canonical,
        nominal_covariance=nominal,
        certification=certification,
    )
