# Release Cleanup Audit — 2026-06-10

This audit records the public-release checks performed before pushing the
repository for committee review.

## Hygiene Actions

- Root-level draft protocols and private operational notes were copied into
  public `docs/` locations where appropriate and archived locally under the
  ignored `archive/internal_notes/` directory.
- WSL `*:Zone.Identifier` metadata streams were removed.
- Python `__pycache__` directories under `src/`, `scripts/`, and `tests/` were
  removed.
- `.gitignore` was added to exclude virtual environments, MLflow stores, raw
  logs, large checkpoints, third-party checkouts, local mirrors, private notes,
  and copyrighted source PDFs.
- Public E1/E2 compute provenance was sanitized to refer to a generic remote
  workstation rather than private host details.

## Verification

- Unit tests: `65 passed`, with NumPy deprecation warnings only.
- Figure regeneration:
  - `scripts/render_paper_quality_figures.py`
  - `scripts/render_experiment_b_threshold_hpa_vector.py`
- Vector check:
  - `deeprm_p1_first_fit_sensitivity.pdf`: 0 embedded `/Subtype /Image` objects.
  - `decima_paired_delta_distributions.pdf`: 0 embedded `/Subtype /Image` objects.
  - `threshold_vs_hpa_v2_under_lag.pdf`: 0 embedded `/Subtype /Image` objects.
- Sensitive publish-path scan: no private VM alias, IP address, password phrase,
  jump-host string, or fixed rootless Docker socket path was found in public
  paths selected for git.

## Public Entry Points

- [README](../../README.md)
- [Protocol index](../PROTOCOL_INDEX.md)
- [Results index](../RESULTS_INDEX.md)
- [Reproducibility guide](../REPRODUCIBILITY.md)
- [Security and provenance boundary](../SECURITY_AND_PROVENANCE.md)
