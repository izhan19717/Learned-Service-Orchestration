# Reproducibility Guide

This guide describes how to verify the public package and how to regenerate the
paper-ready artifacts. Full training runs can take many hours and may require
the external simulators described in [external/README.md](../external/README.md).

## 1. Install the Python Package

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The package exposes three command-line entry points:

```bash
cisose-deeprm protocol
cisose-rossi protocol
cisose-decima protocol
```

## 2. Run Fast Verification

```bash
python -m pytest
```

These tests cover simulator invariants, statistics helpers, perturbation
machinery, Rossi source-derived parameters, and DeepRM author-alignment
regressions. Passing tests do not replace full experiment execution, but they
are the expected first check for reviewers.

## 3. Clone External Simulators

For Rossi and Decima full workflows:

```bash
mkdir -p external

git clone https://github.com/effereds/rlad-core-simulator external/rlad-core-simulator
git -C external/rlad-core-simulator checkout d6a4ff136907eb1bd9e8b4151a9162231ce0ee6a

git clone https://github.com/hongzimao/decima-sim external/decima-sim
git -C external/decima-sim checkout c010dd74ff4b7566bd0ac989c90a32cfbc630d84
```

The public repository intentionally ignores these checkouts.

## 4. Regenerate Paper Artifacts From Existing Tables

These commands regenerate the paper-ready summaries and plots from committed
tables and JSON outputs:

```bash
python scripts/generate_deeprm_paper_artifacts.py
python scripts/generate_rossi_reproduction_artifacts.py
python scripts/generate_decima_paper_artifacts.py
python scripts/generate_e1_e2_paper_artifacts.py
python scripts/render_paper_quality_figures.py
python scripts/render_experiment_b_threshold_hpa_vector.py
```

Canonical outputs are written under `results/paper`.

## 5. DeepRM Full Workflow

The DeepRM training entry point is:

```bash
cisose-deeprm train --author-source --load 0.7 --iterations 1000 \
  --num-jobsets 100 --rollouts-per-jobset 20 --checkpoint-interval 50 \
  --eval-interval 10 --train-end all-done --max-episode-steps 2000 \
  --rollout-workers 16 --run-label author_source_rescue
```

After training, run clean and perturbation evaluations with the frozen
checkpoint:

```bash
cisose-deeprm evaluate-clean --checkpoint results/checkpoints/author_source_rescue/policy_final.pt
cisose-deeprm evaluate-perturbations --checkpoint results/checkpoints/author_source_rescue/policy_final.pt
python scripts/run_deeprm_p1_first_fit_sensitivity.py
```

The committed repository does not include large checkpoint files. Re-running
training is required if a reviewer wants to reproduce from raw training rather
than inspect the committed paper artifacts.

## 6. Rossi/RLAD Full Workflow

After cloning `external/rlad-core-simulator`, inspect the source-derived
protocol:

```bash
cisose-rossi protocol
cisose-rossi reproduce-table-i
python scripts/run_rossi_online_p1_lag_sweep.py
python scripts/run_rossi_online_p2_p3.py
python scripts/generate_rossi_diagnostics_qa.py
```

Additional Rossi experiments:

```bash
python scripts/run_experiment_a_block_bootstrap_rossi.py
python scripts/run_experiment_b_hpa_v2_rossi.py
python scripts/run_hpa_v2_config_sensitivity_rossi.py
python scripts/run_e1_rossi_magnitude_sweep.py
python scripts/run_e2_companion_a_rossi_rescore.py
```

## 7. Decima Full Workflow

After cloning `external/decima-sim`, run preflight/readiness checks:

```bash
cisose-decima preflight
cisose-decima reference-commands
python scripts/run_decima_preflight.py
```

The official Decima training and evaluation workflow is Docker-based:

```bash
python scripts/run_decima_tf1_reproduction.py
python scripts/run_decima_tf1_test_gate.py
python scripts/run_decima_tf1_perturb_eval.py
```

Comparator sensitivities:

```bash
python scripts/run_decima_srtf_sensitivity.py
python scripts/run_e1_decima_magnitude_sweep.py
```

The public Decima package labels reconstructed SRTF/Graphene-style sensitivity
artifacts separately from official Graphene evidence because the public
simulator did not ship a directly executable single-resource Graphene
comparator.

## 8. Provenance and Privacy Boundary

The paper-ready result artifacts include MLflow run IDs where generated during
the study, but the local MLflow database and artifact store are not committed.
Private remote-workstation connection details are deliberately omitted. Any
future rerun should record the public commit hash, package version, external
simulator commit hashes, and generated artifact manifests.
