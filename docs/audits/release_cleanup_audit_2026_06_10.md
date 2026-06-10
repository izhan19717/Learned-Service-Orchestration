# Repository Cleanup Audit — 2026-06-10

This audit records repository-structure and verification checks performed
before the initial push.

## Hygiene Actions

- Root-level draft protocols were copied into stable `docs/` locations where
  appropriate.
- WSL `*:Zone.Identifier` metadata streams were removed.
- Python `__pycache__` directories under `src/`, `scripts/`, and `tests/` were
  removed.
- `.gitignore` was added for generated runtime state, upstream simulator
  checkouts, local mirrors, and local reading material.
- E1/E2 compute provenance was rewritten as a generic remote-workstation
  policy.

## Verification

- Unit tests: `65 passed`, with NumPy deprecation warnings only.
- Figure regeneration:
  - `scripts/render_paper_quality_figures.py`
  - `scripts/render_experiment_b_threshold_hpa_vector.py`
- Vector check:
  - `deeprm_p1_first_fit_sensitivity.pdf`: 0 embedded `/Subtype /Image` objects.
  - `decima_paired_delta_distributions.pdf`: 0 embedded `/Subtype /Image` objects.
  - `threshold_vs_hpa_v2_under_lag.pdf`: 0 embedded `/Subtype /Image` objects.
- Tracked-file scan: no machine-specific connection strings or authentication
  material were found in tracked files.

## Public Entry Points

- [README](../../README.md)
- [Protocol index](../PROTOCOL_INDEX.md)
- [Results index](../RESULTS_INDEX.md)
- [Reproducibility guide](../REPRODUCIBILITY.md)
- [Security and provenance boundary](../SECURITY_AND_PROVENANCE.md)
