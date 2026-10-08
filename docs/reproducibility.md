# Reproducibility

The public workflows have four tiers.

1. **Quickstart** runs the small CPU g1/CFI-c example. It is educational, not
   a paper-table reproduction.
2. **Analytic paper workflows** provide explicit configs for g1, g2, g3, g5,
   and all three published g10 thresholds.
3. **Expensive analytic workflows** currently cover g6; certified proposal
   artifacts are strongly preferred for evaluation.
4. **External-asset g7** provides asset-free validation/planning and explicit
   offline evaluation with a local certified proposal, prepared tensor, local
   ResNet18 weights, and explicit event device.

The two supported `+F` variants are executable for bounded fitting and ordinary
IS. Other external baselines remain outside this package.

```console
robust-cfi-is validate configs/paper/g1.yaml
robust-cfi-is reproduce plan configs/paper/g6.yaml --method ffkl_c --k 16
robust-cfi-is reproduce plan configs/paper/g7.yaml --method cfi_c --k 6
```

To evaluate the repository-embedded certified proposal:

```console
robust-cfi-is reproduce evaluate configs/paper/g1.yaml \
  --manifest reproducibility/artifacts.json \
  --artifact-name g1-k16-cfi-c-seed100-q-final \
  --evaluation-seed 100 \
  --output runs/g1-cfi-c-k16-seed100.json
```

With no `--artifact` argument, the command resolves
`reproducibility/q_final/g1-k16-cfi-c-seed100-q-final.json` relative to the
manifest, verifies its recorded size and SHA-256, and loads it locally. An
explicit alternative local copy remains supported with
`--artifact /path/to/q_final.json`; it is subject to the same identity checks.

The estimator is ordinary, unnormalized importance sampling. Non-CMC
estimates, mean log error, and CoV use all 20 evaluation runs. `CMC x/20`
means that x of the 20 CMC runs observed the event. Only the CMC-dependent
variance-reduction ratio uses the positive-CMC subset.

Core commands are CPU-first, offline, and tracking-free. Configs use relative
output roots and explicit seeds. For g7, add the three mandatory options:

```console
robust-cfi-is reproduce evaluate configs/paper/g7.yaml \
  --manifest reproducibility/artifacts.json \
  --artifact-name g7-k16-cfi-c-ffkl-c-seed100-q-final \
  --evaluation-seed 100 \
  --g7-data /local/imagenetv2-matched-frequency-format-val_first24.pt \
  --resnet-weights /local/resnet18-f37072fd.pth --event-device cpu \
  --output runs/g7-seed100.json
```

The repository stores all 90 certified proposals as q_final JSON files under
`reproducibility/q_final/`; no external mirror, public artifact URL, or
download step is required. No normal execution path performs a download or
cache lookup. The q_final files are intentionally excluded from Python wheels
and sdists, so artifact-assisted paper evaluation requires a Git checkout or
VCS source archive. The manifest records each proposal's local identity.
