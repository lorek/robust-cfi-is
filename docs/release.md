# Package boundary

The package supports the CPU quickstart, public GMM methods, defensive fRKL-d,
analytic paper configs, explicit offline g7 evaluation, the two supported
RealNVP `+F` variants, canonical artifact verification, and ordinary
unnormalized IS. AMS, Subset Simulation, Safe-ICE, external baseline
implementations, and orchestration beyond the documented workflows are not
presented as supported interfaces.

The package does not include external g7 assets and does not regenerate the
required g7 tensor. It does not claim theorem certification for a flow
proposal. The 90 certified proposals are stored as `q_final` JSON files
under `reproducibility/q_final/` and indexed by
`reproducibility/artifacts.json`. The primary workflow requires no artifact
mirror or artifact URL.

Project source, project documentation (including the paper-derived README
Figure 1), and the 90 certified proposals are covered by the same MIT
grant. The five joint rightsholders are Paweł Lorek, Rafał Nowak,
Rafał Topolnicki, Tomasz Trzciński, and Maciej Zięba. The prepared g7 tensor,
ResNet weights, ImageNetV2 image bytes, and third-party dependency code or
assets are outside that project grant and are not redistributed, bundled, or
hosted. Their exact required sizes and SHA-256 identities, offline checks, and
the earlier preparation script's `DOES_NOT_EXACTLY_REPRODUCE` status are
documented in [`third-party-assets.md`](third-party-assets.md).

Version 0.1.0 citation metadata is recorded in
[`CITATION.cff`](../CITATION.cff); the preferred paper citation uses the
`conference-paper` form, and the README BibTeX key is `lorek2026robust`. The
paper-derived Figure 1 is project documentation under the same MIT grant.

The q_final files are Git/VCS source-tree data and are intentionally excluded
from Python wheels and sdists. Artifact-assisted paper evaluation therefore
uses a Git checkout or VCS source archive. Manifest-relative local resolution
is the default, explicit alternative local paths remain supported, and neither
route downloads artifacts or falls back to a network or cache.

`scripts/build_clean_export.py` creates a new allowlisted source tree and a
deterministic relative-path, size, and SHA-256 receipt. It rejects symlinks,
unexpected top-level content, unsupported or oversized files, binary data,
local home paths, GPU UUIDs, credential-like text, existing destinations, and
overlapping source/output trees. The source inventory is checked before and
after copying.

The workflow at `.github/workflows/test.yml` is owned by this standalone
package. CI executes on CPU, but the cross-platform lock may resolve
platform-specific PyTorch runtime packages on Linux and is not described as a
CPU-only dependency lock.
