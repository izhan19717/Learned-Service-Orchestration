# Security and Provenance Boundary

This repository is prepared for public scientific review. It deliberately
separates reproducibility artifacts from private operational material.

## Excluded From Git

The following are excluded by `.gitignore`:

- Local Python environments and caches.
- MLflow local databases and artifact directories.
- Raw training logs and terminal transcripts.
- Large checkpoints and intermediate simulator state.
- Third-party simulator checkouts.
- Private operational notes containing hostnames, IP addresses, SSH aliases, or
  connection details.
- Copyrighted PDFs obtained for local reading.

## Included In Git

The following are included:

- Source code under `src/`.
- Experiment and artifact-generation scripts under `scripts/`.
- Tests under `tests/`.
- Frozen and amended public protocol documents under `docs/protocols/`.
- Paper-ready tables, figures, manifests, and public reports under
  `results/paper/`.
- External-source checkout instructions under `external/README.md`.

## Sensitive-String Audit

Before publication, run:

Before publication, scan tracked files for credentials, host aliases, IP
addresses, and SSH jump-host configuration. Expected result for tracked files:
no hits. Hits in ignored private notes should not be staged.

## Recommended Rerun Provenance

For any future canonical rerun, record:

- Git commit hash of this repository.
- Python version and package lockfile.
- External simulator repository URL and commit hash.
- Docker image digest when Decima/TF1 workflows are used.
- MLflow run ID or equivalent immutable run identifier.
- SHA256 manifest for generated result files.
