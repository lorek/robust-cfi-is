# Benchmarks

The package contains the analytic event functions and nominal distributions
for g1, g2, g3, g5, g6, and g10. The g10 configs are deliberately split into
20k, 40k, and 500k threshold variants so a threshold cannot change implicitly.

Paper configs record fitting, final-proposal policy, final IS, aggregation,
seeds, sample budgets, component counts, and execution constraints separately.
`robust-cfi-is reproduce plan` prints these values without starting an experiment.

The g6 workflow is classified as expensive analytic. g7 is an external-asset,
offline workflow with a fixed 62-coordinate linearized ResNet18 loss event.
Its proposal remains on CPU while the event device must be stated explicitly;
CPU support is a correctness path, not a claim about the paper-performance
environment. See [`g7.md`](g7.md).
