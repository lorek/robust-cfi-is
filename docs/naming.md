# Method naming

Public method names state the mathematical objective and covariance family:

- **fFKL** means forward KL, fitted with weighted cross entropy;
- **fRKL** means reverse KL, fitted with a score-function surrogate;
- **CFI** means coverage-first initialization;
- **-u** means an unconstrained full-covariance Gaussian mixture; and
- **-c** means a constrained mixture with covariance
  `L L^T + beta Sigma`.

`CFI-c+RKL-c` is constrained CFI followed by reverse-KL refinement.
`CFI-c+FKL-c` is constrained CFI followed by forward-KL refinement.

Normal Python, configuration, and command-line inputs use these public slugs:

| Public slug | Display name | Objective | Covariance |
|---|---|---|---|
| `frkl_u` | fRKL-u | reverse KL | unconstrained |
| `ffkl_u` | fFKL-u | forward KL | unconstrained |
| `frkl_c` | fRKL-c | reverse KL | constrained |
| `ffkl_c` | fFKL-c | forward KL | constrained |
| `cfi_u` | CFI-u | coverage-first | unconstrained |
| `cfi_c` | CFI-c | coverage-first | constrained |
| `cfi_c_frkl_c` | CFI-c+RKL-c | CFI then reverse KL | constrained |
| `cfi_c_ffkl_c` | CFI-c+FKL-c | CFI then forward KL | constrained |
| `frkl_d` | fRKL-d | reverse KL on the full defensive proposal | defensive |
| `cfi_c_frkl_c_flow` | CFI-c+RKL-c+F | fixed-proposal weighted CE flow refinement | flow over constrained GMM |
| `cfi_c_ffkl_c_flow` | CFI-c+FKL-c+F | fixed-proposal weighted CE flow refinement | flow over constrained GMM |

For each `+F` method, `constrained_gmm_backbone` is true while
`final_proposal_family` is `flow` and `theorem_certified_final_proposal` is
false. The constrained-GMM theorem does not automatically survive the flow.
