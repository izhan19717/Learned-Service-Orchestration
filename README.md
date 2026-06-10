# Learned Service Orchestration

This repository contains the empirical artifact for the learned service
orchestration study. It implements and evaluates three published reinforcement
learning methods under literature-calibrated perturbations:

- **DeepRM** for multi-resource job scheduling.
- **Rossi/RLAD** for container autoscaling.
- **Decima** for DAG scheduling.

The central statistic is
`Delta = metric(classical comparator) - metric(RL)`. Positive `Delta` means
the learned policy has the lower metric. Negative `Delta` means the comparator
has the lower metric. The method-specific metrics are mean slowdown for DeepRM,
total autoscaling cost for Rossi/RLAD, and mean job completion time for Decima.

## Contents

```text
docs/
  protocols/              Calibration, preregistration, and amendments.
  protocols/extensions/   Experiments A, B, C, E1, and E2.
  audits/                 Release and paper-traceability audits.
results/paper/
  deeprm/                 Reported DeepRM tables, figures, and manifests.
  rossi/                  Rossi reproduction, perturbation, and diagnostics.
  decima/                 Decima reproduction, perturbation, and diagnostics.
  experiments/            Additional analyses used by the paper.
scripts/                  Evaluation, reproduction, and figure scripts.
src/                      Python implementations and shared utilities.
tests/                    Regression and scientific-alignment tests.
external/                 Required upstream simulator checkout instructions.
docker/                   Decima/TF1 Docker support.
```

Main entry points:

- [Protocol index](docs/PROTOCOL_INDEX.md)
- [Results index](docs/RESULTS_INDEX.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Implementation notes](docs/implementation_notes.md)
- [Paper v4 traceability audit](docs/audits/paper_v4_traceability_audit_2026_06_10.md)
- [External simulator setup](external/README.md)

## Environment

Use Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The Decima official-simulator workflow also requires Docker and the upstream
Decima checkout described in [external/README.md](external/README.md).

## Verification

Run the regression tests:

```bash
python -m pytest
```

Regenerate the vector figures used in the paper:

```bash
python scripts/render_paper_quality_figures.py
python scripts/render_experiment_b_threshold_hpa_vector.py
```

Canonical figure outputs are under `results/paper/**/figures/`.

## Regenerating Reported Artifacts

The reported tables, figures, and manifests are under
[results/paper](results/paper). To regenerate summaries from the committed
tables and JSON files:

```bash
python scripts/generate_deeprm_paper_artifacts.py
python scripts/generate_rossi_reproduction_artifacts.py
python scripts/generate_decima_paper_artifacts.py
python scripts/generate_e1_e2_paper_artifacts.py
```

Full training and official-simulator reruns require substantially more compute
than table and figure regeneration. Command-level workflows are listed in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Protocol

The primary nine-prediction design is recorded in
[docs/protocols/preregistration_v2_2.md](docs/protocols/preregistration_v2_2.md).
Perturbation calibration is recorded in
[docs/protocols/calibration_v2_2.md](docs/protocols/calibration_v2_2.md).
Protocol amendments and extension experiments are indexed in
[docs/PROTOCOL_INDEX.md](docs/PROTOCOL_INDEX.md).

Frozen protocol documents are kept separate from post-run audits and
diagnostics. This separation is necessary because several results falsify the
original predictions.
