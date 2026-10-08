# Results

[Back to README](../README.md) · [Paper](https://arxiv.org/abs/2610.07485) ·
[Reproduction instructions](reproducibility.md)

This page collects the paper's motivating example and main numerical summaries.
These are reported paper results. The paper
contains the full appendix tables, sensitivity studies, and timing comparisons.
The repository implements our methods; it does not bundle every external baseline
used in the paper. See [What can I run?](#what-can-i-run) below.

## Benchmarks at a glance

We estimate $P(g(\mathbf{X})>\gamma)$ under Gaussian input distributions.
The examples test separated modes, curved boundaries, anisotropy, and
high-dimensional events.

| Benchmark | Dimension | Problem | Reference probability |
|---|---:|---|---:|
| Toy (`g0`) | 1 | Two Gaussian tails: $\lvert X\rvert > 4$ | 6.33425e-05 |
| `g1` | 2 | Two separated circular event regions ("Leaf") | 4.79e-06 |
| `g2` | 2 | Piecewise-boundary structural-reliability problem | 2.23e-03 |
| `g3` | 2 | Anisotropic structural-reliability problem with a linear tilt | 4.21e-03 |
| `g5` | 40 | Powell-function benchmark | 3.15e-05 |
| `g6` | 100 | Linear event; isolates high-dimensional scaling | 2.33e-04 |
| `g7` | 62 | ResNet loss under parameter perturbations, with a linearized evaluator | 6.00e-05 |
| `g10` | 10 | Sum of correlated lognormal variables, threshold 40,000 | 4.68e-04 |

The reference values above use the rounding in the headline table. A reference
may be analytic or numerical; it is not necessarily an exact ground truth.
For `g10`, the paper also studies thresholds 20,000 and 500,000.
See the paper's **Benchmark function definitions** appendix for the formulas
and input distributions, and [benchmark documentation](benchmarks.md)
for the available workflows.

## Reading the tables

- **Estimate:** estimated event probability.
- **log-err ↓:** $|\log I_{\mathrm{true}}-\log\hat I_R|$, averaged over evaluations.
- **CoV ↓:** the reported coefficient of variation. Read it alongside log-error
  and failure counts: a small CoV alone does not establish accuracy.
- **Variance reduction ↑:** variance reduction relative to crude Monte Carlo
  (CMC). A value of 1 means no reduction.
- **FG:** the final constrained GMM is covered by the paper's finite-variance
  condition. This is not a guarantee of small finite-budget error. A dash in
  this column means no coverage by this theorem, not a proof of infinite variance.

`CFI` denotes coverage-first initialization; `FKL` and `RKL` denote forward
and reverse KL. The prefix `f` denotes a standalone baseline, `-c` covariance
constraints, `-u` an unconstrained model, and `+F` flow refinement.
See [method naming](naming.md) for the corresponding code identifiers.

For the main evaluation protocol, the paper uses $R=10^5$ samples per
evaluation and 20 independent evaluation runs for CMC and the evaluated importance-sampling methods.
Non-CMC importance-sampling estimates, mean log-error, and CoV retain all 20 runs.
Only the CMC-relative variance-reduction metric excludes runs in which CMC
observes no event. External baseline protocols are described in the paper.

## Motivating example: two Gaussian tails

Table 1 of the paper: $X\sim\mathcal{N}(0,1)$, event $|X|>4$, and $K=20$.
The reference probability is **6.33425e-05**. Bold marks the best reported
variance reduction, log-error, and CoV. A dash in a metric column denotes an
unavailable value.

| Method | FG | Estimate | Variance reduction ↑ | log-err ↓ | CoV ↓ |
|---|:---:|---:|---:|---:|---:|
| CMC |  | 6.55e-05 | 1 | 2.65e-01 | 4.06e-01 |
| fRKL-u |  | 6.30e-05 | 2.96e+02 | 1.88e-02 | 2.37e-02 |
| fFKL-u |  | 0 | — | — | — |
| fRKL-c | ✓ | 6.32e-05 | 5.89e+02 | 1.38e-02 | 1.67e-02 |
| fFKL-c | ✓ | 6.36e-05 | 3.67e+03 | 6.41e-03 | 6.65e-03 |
| AMS |  | 6.33e-05 | 4.19e-01 | 1.04e-01 | 5.98e-01 |
| Subset Simulation |  | 6.33e-05 | 6.81e-01 | 5.61e-02 | 5.23e-01 |
| Safe-ICE |  | 6.17e-05 | 5.25e-02 | 2.93e-02 | 5.45e-01 |
| CFI-u |  | 0 | — | — | — |
| CFI-c | ✓ | 6.34e-05 | 2.94e+03 | 5.17e-03 | 7.44e-03 |
| CFI-c+RKL-c | ✓ | 6.33e-05 | 2.90e+03 | 5.72e-03 | 7.51e-03 |
| CFI-c+FKL-c | ✓ | 6.33e-05 | 5.56e+03 | 4.49e-03 | 5.42e-03 |
| CFI-c+RKL-c+F |  | 6.32e-05 | 3.82e+03 | 4.62e-03 | 6.54e-03 |
| CFI-c+FKL-c+F |  | 6.34e-05 | **5.81e+03** | **4.39e-03** | **5.29e-03** |

CFI-u misses the event in this example. Constrained CFI reaches both tails;
forward-KL refinement improves efficiency further, with a smaller additional
gain from the flow. The `+F` results are empirical and do not carry the GMM
finite-variance guarantee.

## Headline results across benchmarks

The following three tables reproduce the three blocks of the paper's headline
comparison, split for readability. Each row selects the lowest-CoV method
**within that block**, at $K=16$ and $R=10^5$. All three blocks must be read
together; the proposed methods do not win on every benchmark or every metric.
The `g10` row uses the 40,000 threshold.

### Best baseline by CoV

Selection among CMC, fFKL-c, fRKL-c, CFI-u, AMS, and Subset Simulation.

| Benchmark | Method | log-err ↓ | CoV ↓ |
|---|---|---:|---:|
| g1 | CMC | 1.43e+00 | 3.92e-03 |
| g2 | CMC | 4.14e-02 | 2.46e-03 |
| g3 | CMC | 4.25e-02 | 3.31e-03 |
| g5 | fRKL-c | 7.31e-02 | 3.19e-02 |
| g6 | fRKL-c | 5.90e-03 | 6.08e-03 |
| g7 | fRKL-c | 5.26e-02 | 8.88e-03 |
| g10 | fFKL-c | 1.24e-02 | 6.01e-03 |

### Best proposed method with a theorem-covered final GMM

| Benchmark | Method | log-err ↓ | CoV ↓ |
|---|---|---:|---:|
| g1 | CFI-c+RKL-c | 6.26e-03 | 7.24e-03 |
| g2 | CFI-c+FKL-c | 3.39e-03 | 4.85e-03 |
| g3 | CFI-c+FKL-c | 3.25e-03 | 4.35e-03 |
| g5 | CFI-c+RKL-c | 7.28e-02 | 4.30e-02 |
| g6 | CFI-c+RKL-c | 7.02e-03 | 6.12e-03 |
| g7 | CFI-c+RKL-c | 5.44e-02 | 6.80e-03 |
| g10 | CFI-c+FKL-c | 1.14e-02 | 6.80e-03 |

### Best proposed method overall

| Benchmark | Method | log-err ↓ | CoV ↓ |
|---|---|---:|---:|
| g1 | CFI-c+FKL-c+F | 3.16e-03 | 3.92e-03 |
| g2 | CFI-c+RKL-c+F | 2.87e-03 | 1.86e-03 |
| g3 | CFI-c+FKL-c+F | 2.80e-03 | 3.31e-03 |
| g5 | CFI-c+RKL-c+F | 6.45e-02 | 2.99e-02 |
| g6 | CFI-c+RKL-c+F | 5.94e-03 | 5.43e-03 |
| g7 | CFI-c+RKL-c+F | 5.49e-02 | 6.24e-03 |
| g10 | CFI-c+FKL-c+F | 1.61e-02 | 6.22e-03 |

The flow variants improve the reported CoV over the selected constrained-GMM
variants here. This does not imply an improvement in every metric: on `g10`,
for example, log-error is larger with the selected flow variant, and the
standalone fFKL-c baseline has the lowest CoV of the three blocks.

## What can I run?

- Start with the [CPU quickstart](../README.md#cpu-quickstart). It is a small
  educational example, not a reproduction of a paper table.
- Consult [reproducibility instructions](reproducibility.md) for paper
  configs, planning, and evaluation of the 90 stored constrained-GMM proposals.
- See [flow support](flows.md) for the executable `+F` methods and their
  distinction from the stored certified-GMM artifacts.
- For `g7`, read [external-input requirements](g7.md). The prepared tensor
  and pretrained weights are not bundled; the earlier preparation script does
  not exactly reproduce the accepted tensor.

The full paper also reports $K=6$ comparisons, unconstrained KL variants,
additional `g10` thresholds, defensive-IS sensitivity, and training costs.
Those remain in the paper rather than duplicating the entire appendix here.
