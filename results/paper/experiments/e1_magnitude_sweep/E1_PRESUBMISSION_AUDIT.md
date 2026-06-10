# E1 Pre-Submission Audit

Date: 2026-06-04

Delta convention: `metric(comparator) - metric(RL)`. Positive delta means the learned/RL method has the lower metric.

## 1. DeepRM E1/Table IV Reconciliation

Claude's concern was correct: the first DeepRM E1 sweep was not using the same operational harness as the locked DeepRM Table IV pipeline.

Root causes:

- Checkpoint mismatch: first E1 used `results/checkpoints/author_source_rescue/load_0.7/policy_iter_1000.pt`; Table IV used `results/checkpoints/author_source_rescue/load_0.7/policy_final.pt` with SHA256 `1d439eb4f5b47a7b1242a9824cd3db60f1f685ca0e53d57907bbf35d860e5f7e`.
- Comparator mismatch: first E1 used `SourceTetris`, a pure source-style dot-product packer; Table IV used `Tetris*` (`TetrisScheduler(source_dot=True)`, alpha=0.5 plus source dot-product packing).
- Policy/statistic seed mismatch: first E1 used hash-derived policy-generator seeds and `seed+1000/2000/3000` statistic seeds; Table IV used CLI offsets `1000+i`, `2000+i`, `3000+i` for policy sampling and `seed+10000/20000/30000` for statistics.

Corrective action:

- Patched `scripts/run_e1_deeprm_magnitude_sweep.py` to use `policy_final.pt`, `Tetris*` as the primary comparator, SJF as the secondary comparator, CLI policy-generator offsets, sanitized MLflow metric names, and locked Table IV statistic seeds.
- Moved the superseded local result directory to `results/paper/experiments/e1_magnitude_sweep/deeprm_20260604_superseded_iter1000_sourcepacker`.
- Reran DeepRM E1 on the VM and finalized artifacts through MLflow run `2d4588396f474862bbf61bfebbf16ba4`.

Acceptance check against locked Table IV:

| Cell | DeepRM mean | Tetris* mean | Delta | 95% CI | Status |
|---|---:|---:|---:|---:|---|
| P1 k=10 | 73.914617 | 269.630410 | 195.715793 | [185.105374, 205.917432] | exact match |
| P2 alpha=1.5 | 49.750894 | 141.537924 | 91.787030 | [79.846609, 104.568941] | exact match |
| P3 epsilon=0.05 | 36.236038 | 59.804964 | 23.568926 | [21.715522, 25.363305] | exact match |

Conclusion:

The DeepRM E1 mismatch is resolved. The corrected E1 table can be tabulated without conflicting with Table IV. The earlier SourceTetris/policy-iteration run must not be used as a paper-facing DeepRM E1 result.

## 2. Decima Lambda 0.25 Sanity Check

Claude's concern was the non-monotonic P1 lag point: lambda=0.25 is significantly negative while lambda=0, 0.5, and larger values are positive.

Code audit:

- `scripts/decima_tf1_perturb_eval.py::_install_lag` computes `lag_mean = lag_lambda * expected_stage_duration`.
- Each scheduled task receives `delay = _deterministic_exponential_lag(...)`.
- The delay is a Python float and is added directly to `Task.finish_time` and `Task.duration`.
- `external/decima-sim/spark_env/timeline.py` stores event keys directly in a heap; no integer time-step cast is applied.
- No `int`, `round`, `floor`, or `ceil` call appears in the lag injection path.

Data audit:

- All P1 lambda cells use the same 30 experiment seeds in the same order.
- Lambda=0.25 is not a one-window outlier: 24 of 30 paired deltas are negative, with median delta -16,271.12.
- Lambda=0.5 flips back positive: 23 of 30 paired deltas are positive, with median delta 66,158.18.

Important nuance:

The deterministic exponential draw includes `mean` in its hash key. Therefore each lambda cell has a valid deterministic exponential delay distribution for that lambda, and both schedulers are paired within the cell, but adjacent lambda values are not common-random-number scaled versions of the same underlying per-task uniforms. This can contribute to non-monotonic magnitude curves. It is not a fractional-step rounding artifact.

Conclusion:

Keep the Decima lambda=0.25 result, but describe it carefully: it is a real per-cell paired reversal under the implemented deterministic lag sampler, not evidence for a smooth monotone physical response. The paper should not overclaim Decima P1 monotonicity; the Decima case remains supported by non-reproduction, anchor-level robustness, and P3/P2 characterization rather than a simple monotone lag curve.

## Updated Artifacts

- Corrected DeepRM E1 table: `results/paper/experiments/e1_magnitude_sweep/deeprm/tables/e1_deeprm_magnitude_sweep.csv`
- Corrected DeepRM E1 figure: `results/paper/experiments/e1_magnitude_sweep/deeprm/figures/e1_deeprm_magnitude_sweep.pdf`
- Decima P1 per-seed audit table: `results/paper/experiments/e1_magnitude_sweep/decima/tables/decima_e1_p1_lag_per_seed_audit.csv`
- Updated handoff report: `results/paper/experiments/e1_magnitude_sweep/E1_E2_CLAUDE_HANDOFF_REPORT.md`
