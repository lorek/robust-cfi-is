# Proposal artifacts

`reproducibility/artifacts.json` is the schema-version-1 content-identity index
for the 90 certified Gaussian-mixture proposals used by the supported
paper workflows. Names describe the benchmark, component count, public method,
seed, and `q_final` role.

The 90 artifact files are stored directly in Git under
`reproducibility/q_final/`. The manifest records
`availability: repository_embedded`; empty mirror lists are intentional because
the primary workflow requires no external artifact host, URL, download, or
cache fallback. The manifest is the content-identity authority: every file's
recorded size and SHA-256 are checked before the safe loader reads it.

By default, both artifact verification and paper evaluation resolve a record
to the following manifest-relative path:

```text
<manifest-parent>/q_final/<record-name>.json
```

An explicit alternative local path remains supported through `--path` for
`robust-cfi-is artifacts verify` and `--artifact` for
`robust-cfi-is reproduce evaluate`. Neither form performs network access.
Optional future mirror metadata is provider-neutral, validates metadata only,
and does not download, upload, or mutate the manifest.

```console
robust-cfi-is artifacts list --manifest reproducibility/artifacts.json
robust-cfi-is artifacts verify --manifest reproducibility/artifacts.json \
  --name g1-k6-cfi-c-seed100-q-final
```

To verify an alternative local copy explicitly:

```console
robust-cfi-is artifacts verify --manifest reproducibility/artifacts.json \
  --name g1-k6-cfi-c-seed100-q-final \
  --path /path/to/q_final.json
```

The offline loader accepts the canonical schema-version-1 JSON envelope. It validates
content and tensor semantic hashes and reconstructs the float32 factors that
define `q_final`. Pickle is not part of the loading contract.

The repository and normal VCS source archives contain all 90 proposals.
Python wheels and Python sdists intentionally contain zero q_final files, so a
Git checkout or VCS source tree is required for artifact-assisted paper
evaluation.

The manifest includes the ten g7 proposal identities. They can be
evaluated by the explicit offline g7 workflow after the proposal, prepared
tensor, and weights have been obtained and independently verified.
