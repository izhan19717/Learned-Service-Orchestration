# Reproducibility

This file lists the commands used to verify the repository and regenerate the
reported artifacts. Full official-simulator runs can take many hours.

## 1. Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Command-line entry points:

```bash
cisose-deeprm protocol
cisose-rossi protocol
cisose-decima protocol
```

## 2. Tests

```bash
python -m pytest
```

The tests cover simulator invariants, statistics helpers, perturbation
machinery, Rossi source-derived constants, Decima adapter behavior, and DeepRM
author-alignment checks.

## 3. External Simulators

Rossi and Decima full workflows require upstream simulator checkouts:

```bash
mkdir -p external

git clone https://github.com/effereds/rlad-core-simulator external/rlad-core-simulator
git -C external/rlad-core-simulator checkout d6a4ff136907eb1bd9e8b4151a9162231ce0ee6a

git clone https://github.com/hongzimao/decima-sim external/decima-sim
git -C external/decima-sim checkout c010dd74ff4b7566bd0ac989c90a32cfbc630d84
```

## 4. Tables and Figures

Regenerate summaries and figures from committed result tables and JSON files:

```bash
python scripts/generate_deeprm_paper_artifacts.py
python scripts/generate_rossi_reproduction_artifacts.py
python scripts/generate_decima_paper_artifacts.py
python scripts/generate_e1_e2_paper_artifacts.py
python scripts/render_paper_quality_figures.py
python scripts/render_experiment_b_threshold_hpa_vector.py
python scripts/render_decima_e1_paper_panel.py
```

Outputs are written under `results/paper`.

## 5. DeepRM Workflow

Training command:

```bash
cisose-deeprm train --author-source --load 0.7 --iterations 1000 \
  --num-jobsets 100 --rollouts-per-jobset 20 --checkpoint-interval 50 \
  --eval-interval 10 --train-end all-done --max-episode-steps 2000 \
  --rollout-workers 16 --run-label author_source_rescue
```

Clean and perturbation evaluation:

```bash
cisose-deeprm evaluate-clean --checkpoint results/checkpoints/author_source_rescue/policy_final.pt
cisose-deeprm evaluate-perturbations --checkpoint results/checkpoints/author_source_rescue/policy_final.pt
python scripts/run_deeprm_p1_first_fit_sensitivity.py
```

The committed `results/paper/deeprm` directory contains the reported tables and
figures. A from-training reproduction requires regenerating the checkpoint.

## 6. Rossi/RLAD Workflow

After cloning `external/rlad-core-simulator`:

```bash
cisose-rossi protocol
cisose-rossi reproduce-table-i
python scripts/run_rossi_online_p1_lag_sweep.py
python scripts/run_rossi_online_p2_p3.py
python scripts/generate_rossi_diagnostics_qa.py
```

Additional Rossi analyses:

```bash
python scripts/run_experiment_a_block_bootstrap_rossi.py
python scripts/run_experiment_b_hpa_v2_rossi.py
python scripts/run_hpa_v2_config_sensitivity_rossi.py
python scripts/run_e1_rossi_magnitude_sweep.py
python scripts/run_e2_companion_a_rossi_rescore.py
```

## 7. Decima Workflow

After cloning `external/decima-sim`:

```bash
cisose-decima preflight
cisose-decima reference-commands
python scripts/run_decima_preflight.py
```

Official simulator training and evaluation:

```bash
python scripts/run_decima_tf1_reproduction.py
python scripts/run_decima_tf1_test_gate.py
python scripts/run_decima_tf1_perturb_eval.py
```

Comparator sensitivity analyses:

```bash
python scripts/run_decima_srtf_sensitivity.py
python scripts/run_e1_decima_magnitude_sweep.py
```

The Decima SRTF/Graphene-style comparator is a reconstructed sensitivity
analysis. It is not treated as executable official Graphene evidence.

## 8. Provenance For New Runs

Record the following for any new canonical run:

- Repository commit hash.
- Python version and lockfile.
- Upstream simulator URL and commit hash.
- Docker image digest for Decima/TF1 runs.
- MLflow run ID or equivalent immutable run identifier.
- SHA256 manifest for generated result files.
