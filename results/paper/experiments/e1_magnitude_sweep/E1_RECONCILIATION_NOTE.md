# E1 Harness Alignment Note

Date: 2026-06-04

Delta convention: `metric(comparator) - metric(RL)`. Positive delta means the learned/RL method has the lower metric.

## 1. DeepRM E1/Table IV Alignment

The canonical DeepRM E1 sweep is aligned to the locked DeepRM Table IV
pipeline. This note records the harness invariants used for the reported E1
artifacts.

Required alignment points:

- Checkpoint: `results/checkpoints/author_source_aligned/load_0.7/policy_final.pt` with SHA256 `1d439eb4f5b47a7b1242a9824cd3db60f1f685ca0e53d57907bbf35d860e5f7e`.
- Primary comparator: `Tetris*` (`TetrisScheduler(source_dot=True)`, alpha=0.5 plus source dot-product packing).
- Secondary comparator: SJF.
- Policy-generator offsets: `1000+i`, `2000+i`, `3000+i`, matching `cisose-deeprm evaluate-perturbations`.
- Statistic seeds: `seed+10000/20000/30000`, matching the locked Table IV evaluation.

Implementation:

- `scripts/run_e1_deeprm_magnitude_sweep.py` implements the aligned checkpoint,
  comparator, policy-generator, MLflow naming, and statistic-seed settings.
- The canonical DeepRM E1 artifact set was finalized through MLflow run
  `2d4588396f474862bbf61bfebbf16ba4`.

Acceptance check against locked Table IV:

| Cell | DeepRM mean | Tetris* mean | Delta | 95% CI | Status |
|---|---:|---:|---:|---:|---|
| P1 k=10 | 73.914617 | 269.630410 | 195.715793 | [185.105374, 205.917432] | exact match |
| P2 alpha=1.5 | 49.750894 | 141.537924 | 91.787030 | [79.846609, 104.568941] | exact match |
| P3 epsilon=0.05 | 36.236038 | 59.804964 | 23.568926 | [21.715522, 25.363305] | exact match |

Conclusion:

The DeepRM E1 artifact set is Table-IV-compatible. The E1 table can be
tabulated without conflicting with the locked main-study DeepRM anchor rows.

## 2. Decima Lambda 0.25 Check

The Decima E1 sweep contains a non-monotonic P1 lag point: lambda=0.25 is
significantly negative while lambda=0, 0.5, and larger values are positive.

Code check:

- `scripts/decima_tf1_perturb_eval.py::_install_lag` computes `lag_mean = lag_lambda * expected_stage_duration`.
- Each scheduled task receives `delay = _deterministic_exponential_lag(...)`.
- The delay is a Python float and is added directly to `Task.finish_time` and `Task.duration`.
- `external/decima-sim/spark_env/timeline.py` stores event keys directly in a heap; no integer time-step cast is applied.
- No `int`, `round`, `floor`, or `ceil` call appears in the lag injection path.

Data check:

- All P1 lambda cells use the same 30 experiment seeds in the same order.
- Lambda=0.25 is not a one-window outlier: 24 of 30 paired deltas are negative, with median delta -16,271.12.
- Lambda=0.5 flips back positive: 23 of 30 paired deltas are positive, with median delta 66,158.18.

Important nuance:

The deterministic exponential draw includes `mean` in its hash key. Therefore each lambda cell has a valid deterministic exponential delay distribution for that lambda, and both schedulers are paired within the cell, but adjacent lambda values are not common-random-number scaled versions of the same underlying per-task uniforms. This can contribute to non-monotonic magnitude curves. It is not a fractional-step rounding artifact.

Conclusion:

The Decima lambda=0.25 result is retained as a real per-cell paired reversal
under the implemented deterministic lag sampler. It is not evidence for a
smooth monotone physical response. The Decima interpretation therefore rests on
the official-simulator reproduction result, anchor-level robustness, and P2/P3
characterisation rather than on a monotone P1 lag curve.

## Updated Artifacts

- Corrected DeepRM E1 table: `results/paper/experiments/e1_magnitude_sweep/deeprm/tables/e1_deeprm_magnitude_sweep.csv`
- Corrected DeepRM E1 figure: `results/paper/experiments/e1_magnitude_sweep/deeprm/figures/e1_deeprm_magnitude_sweep.pdf`
- Decima P1 per-seed reconciliation table: `results/paper/experiments/e1_magnitude_sweep/decima/tables/decima_e1_p1_lag_per_seed_reconciliation.csv`
- Extension report: `results/paper/experiments/e1_magnitude_sweep/E1_E2_EXTENSION_REPORT.md`
