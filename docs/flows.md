# RealNVP `+F` methods

Two public flow variants are available through `fit_public_method`:

- `cfi_c_frkl_c_flow` (`CFI-c+RKL-c+F`), whose constrained-GMM backbone is
  the public `cfi_c_frkl_c` variant;
- `cfi_c_ffkl_c_flow` (`CFI-c+FKL-c+F`), whose constrained-GMM backbone is
  the public `cfi_c_ffkl_c` variant.

The implementation preserves the authoritative alternating masks, affine
coupling equations, inverse and log determinant, near-identity final-layer
initialization, and fixed-generating-proposal weighted-CE refinement. Random
sampling and initialization use explicit generators, and the device is
explicit. Pickle loading and orchestration beyond bounded fitting are not
public interfaces.

The backbone may satisfy the constrained-GMM finite-variance safeguard, but
the transformed proposal is a flow. Accordingly, method metadata records
`constrained_gmm_backbone=true`, `final_proposal_family=flow`, and
`theorem_certified_final_proposal=false`. Ordinary importance sampling with the
flow is supported; theorem certification is not implied.
