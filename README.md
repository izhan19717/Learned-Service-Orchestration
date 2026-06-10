# Learned Service Orchestration

Reproducibility package for the empirical section of the CISOSE position-paper
study on learned service orchestration. The package evaluates whether three
published reinforcement-learning orchestration methods retain their reported
advantage under literature-calibrated observation lag, workload-tail shift, and
adversarial observation perturbations.

The repository is organized for committee review: protocols, source code,
tests, scripts, paper-ready tables, figures, and public provenance notes are
included. Private runtime state, raw logs, hostnames, IP addresses, passwords,
virtual environments, local MLflow stores, and large intermediate checkpoints
are intentionally excluded.

## Methods

The empirical package covers three method families and task classes:

- **DeepRM**: multi-resource job scheduling; primary metric is mean slowdown.
- **Rossi/RLAD**: horizontal/vertical autoscaling; primary metric is total
  autoscaling cost, with Rossi 2019 Table I reproduction diagnostics.
- **Decima**: DAG scheduling in the official simulator-compatible workflow;
  primary metric is mean job completion time.

The pre-registered comparison statistic is
`Delta = metric(classical comparator) - metric(RL)`. Positive values mean the
RL method is better; negative values mean the classical comparator is better.

## Repository Layout

```text
docs/
  protocols/              Frozen calibration, preregistration, amendments.
  protocols/extensions/   Additional pre-registered Experiments A, B, C, E1, E2.
  public_notes/           Public implementation and provenance notes.
results/paper/
  deeprm/                 DeepRM tables, figures, and generated manifests.
  rossi/                  Rossi reproduction, perturbation, and diagnostics.
  decima/                 Decima reproduction, perturbation, and diagnostics.
  experiments/            Additional paper-strengthening experiments.
scripts/                  Reproduction, evaluation, and figure-rendering scripts.
src/                      Python packages for DeepRM, Rossi, Decima, and common tools.
tests/                    Unit and scientific-alignment regression tests.
external/                 Instructions for cloning third-party simulators.
docker/                   Docker support for Decima/TF1 workflows.
```

Important entry points:

- [Protocol index](docs/PROTOCOL_INDEX.md)
- [Results index](docs/RESULTS_INDEX.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Public implementation notes](docs/public_notes/implementation_notes_public.md)
- [External simulator setup](external/README.md)

## Environment

Python 3.11 or newer is expected. The lightweight Python workflow can be
installed with:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The Decima official simulator workflow also requires Docker and the external
Decima simulator checkout described in [external/README.md](external/README.md).
GPU acceleration is optional for these released scripts; the published
evaluation pipelines are CPU-compatible and were run on a remote workstation
for canonical artifacts.

## Quick Verification

Run the unit and alignment checks:

```bash
python -m pytest
```

Regenerate the paper-quality vector figures that were recently cleaned for
submission:

```bash
python scripts/render_paper_quality_figures.py
python scripts/render_experiment_b_threshold_hpa_vector.py
```

The generated files are written under `results/paper/**/figures/` and mirrored
to root-level `figures/` during local work. The public canonical copies are the
ones under `results/paper`.

## Reproducing Paper Artifacts

The committed paper artifacts are under [results/paper](results/paper). To
regenerate summary tables and plots from available result tables:

```bash
python scripts/generate_deeprm_paper_artifacts.py
python scripts/generate_rossi_reproduction_artifacts.py
python scripts/generate_decima_paper_artifacts.py
python scripts/generate_e1_e2_paper_artifacts.py
```

Full training and simulator reproduction runs are substantially more expensive
than artifact regeneration. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
for the command-level workflow and compute notes.

## Scientific Protocol

The primary nine-prediction design is frozen in
[docs/protocols/preregistration_v2_2.md](docs/protocols/preregistration_v2_2.md).
Magnitude calibration is recorded in
[docs/protocols/calibration_v2_2.md](docs/protocols/calibration_v2_2.md).
Protocol amendments and later paper-strengthening experiments are indexed in
[docs/PROTOCOL_INDEX.md](docs/PROTOCOL_INDEX.md).

Results are reported even when they falsify the position paper's original
prediction. Post-run diagnostics are kept separate from preregistered protocol
text so that reviewers can distinguish frozen decisions from later
interpretation.

## What Is Not Included

The public package excludes local-only runtime material:

- `.venv/`, caches, pyc files, and WSL metadata streams.
- Local MLflow stores (`mlruns/`, `mlartifacts/`, `mlflow.db`).
- Raw terminal logs and long training transcripts.
- Large checkpoints and intermediate rollout state.
- Private VM/SSH operational notes and credentials.
- Copyrighted PDFs used only for local source reading.
- Third-party simulator repositories; clone them using `external/README.md`.

These exclusions are deliberate. The committed repository is the reviewable
scientific package: source, protocols, tests, scripts, and paper-ready outputs.
