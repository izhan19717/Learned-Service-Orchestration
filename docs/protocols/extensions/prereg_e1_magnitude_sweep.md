# Pre-registration E1 — Perturbation-Magnitude Sweep

**Status:** Frozen before any sweep run. Confirmatory robustness extension of the nine pre-registered predictions.
**Registered:** 2026-06-03, before first E1 sweep execution.
**Commit/provenance:** release repository commit plus SHA256 manifests and
MLflow run IDs recorded in the generated artifacts. Private hostnames and IP
addresses are intentionally excluded from this public package.
**Compute:** remote workstation for canonical evaluation; local workstation for
smoke tests only. Evaluation only; no retraining.

## 1. Purpose
Test whether the seven-of-nine falsification verdicts and the HPA-v2 lag attenuation are properties of the methods or artifacts of the single anchors used in the main study. Equivalently, demonstrate that the paper meets its own D3 requirement (magnitudes calibrated and swept, not chosen to produce a result).

## 2. What is fixed (no degrees of freedom after this point)
- Policies: the exact trained checkpoints used in the main study (MLflow run IDs: DeepRM [ID], Decima [ID], Rossi [ID]). No retraining at any magnitude. The fragility claim concerns a fixed policy degrading under perturbation.
- Comparators: each method's bundled comparator as shipped (SourceTetris and SJF for DeepRM; `dynamic_partition` for Decima; the RLAD threshold for Rossi), plus the HPA-v2 controller at the Table V configuration (50% target, 300 s scale-down) for the Rossi lag sweep.
- Workloads, simulator versions, evaluation horizon, and the 30-seed / 30-window sets: identical to the main study.
- Metric per method (unchanged): DeepRM mean slowdown; Decima mean JCT; Rossi total cost.
- Sign convention (unchanged): Δ = metric(comparator) − metric(RL). Positive Δ means RL is better; the fragility prediction expects Δ < 0.

## 3. Perturbation grids (pre-committed)
- **P1 observation lag.** DeepRM, Rossi: k ∈ {0, 1, 2, 5, 10, 20, 50} (anchor k=10, from Borg telemetry P95). Decima: λ ∈ {0.0, 0.25, 0.5, 1.0, 1.5, 2.0} (anchor λ=1.0 = upper bound of the simulator's calibrated regime; λ>1.0 is out-of-regime, see §6).
- **P2 workload tail.** DeepRM, Rossi: Pareto shape α ∈ {2.5, 2.0, 1.75, 1.5, 1.3, 1.1} (anchor α=1.5; smaller α = heavier tail; we stop at 1.1 to keep a finite mean). Decima: TPC-H tail weight w ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (anchor w=0.5).
- **P3 adversarial FGSM.** All three: ε ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2} (anchor ε=0.05), in the normalised observation space.
- **HPA-v2 vs Rossi lag.** Same k grid as P1-Rossi.

## 4. Sample size, pairing, and statistics (identical to main study)
- DeepRM, Decima: 30 paired seeds per (method, perturbation, magnitude). Same seed → same workload realisation for RL and comparator and across clean/perturbed.
- Rossi: 30 non-overlapping windows per cell.
- 95% CI: paired percentile bootstrap, 5000 resamples.
- p-value: paired sign-flip, 1e5 flips, one-sided in the pre-registered direction.
- Rossi only: moving-block bootstrap at L ∈ {5, 10}; block CIs are primary for Rossi.
- Multiple comparisons: the confirmatory family of nine remains as Holm-corrected in the main study and is NOT re-pooled with the sweep. Within the sweep, Holm-Bonferroni is applied across the magnitude points of each (method, perturbation) curve, reported as adjusted p-values. This split is stated in the paper.

## 5. Hypotheses and decision rules (pre-committed, including disconfirming outcomes)
- **H1 — within-regime sign stability (P1, P3).** Across the realistic sub-grid (k ≤ 10 for DeepRM/Rossi; λ ≤ 1.0 for Decima; ε ≤ 0.05), the anchor verdict sign is preserved at every grid point.
  - Confirmation (strengthens): the falsification sign holds across the realistic range → "verdicts are magnitude-robust within the calibrated regime."
  - **Disconfirmation (weakens; we commit to reporting it):** if at any realistic magnitude ≤ anchor the verdict reverses so that RL degrades more than the comparator (Δ < 0, significant after Holm), the corresponding fragility prediction is NOT robustly falsified. We will revise the relevant claim in §V and the abstract and report the magnitude dependence, even though it supports the fragility narrative the paper argues against.
- **H2 — HPA-v2 attenuation stability.** At every k in the grid, |Δ(HPA-v2 vs Rossi, k)| < 0.25 × Δ(bundled threshold, k=10) — the production-controller gap stays below a quarter of the bundled collapse at all lags.
  - Confirmation (strengthens): the production controller never approaches the bundled collapse at any lag → the "+965 is an artifact" claim holds beyond the single anchor.
  - **Disconfirmation (weakens):** if at any k the HPA-v2 gap reaches ≥ 0.5 × the bundled collapse, the "artifact" framing is too strong and we soften it in §V-E.
- **H3 — tail effect characterisation (P2).** Reported as characterisation, not confirmation. We state, per method, whether a monotone significant fragility trend emerges across the α/w grid. If a clean monotone fragility trend emerges for any method, we report that the tail class is more dangerous than the single anchor suggested (a finding, not a failure).

## 6. Out-of-regime handling (pre-committed)
Magnitudes beyond calibration (λ ∈ {1.5, 2.0}; α = 1.1 if the simulator exhibits infinite-variance instability) are plotted but excluded from the H1/H2 verdicts and labelled "out-of-regime." Instability is declared by a pre-set sanity bound (NaN/divergent metric, or queue length exceeding the simulator's representable maximum).

## 7. Reporting
All grid points are reported regardless of outcome; no magnitude is dropped post hoc except under the §6 instability rule. Primary artifact is a three-panel figure (Δ with CI band vs magnitude per perturbation, anchor marked, zero line marked) plus the HPA-v2-vs-Rossi lag curve against the bundled-collapse reference line. Raw per-cell tables are released with the artifact.

## 8. Deviations
Any deviation from this plan is logged here with rationale and timestamp before analysis of the affected cell.

- 2026-06-04: DeepRM E1 reached the severe P3 point `epsilon=0.2` and one
  adversarial rollout exceeded the pre-set `max_steps=100000` termination
  guard before final summary artifacts were written. This point is outside the
  confirmatory H1 regime (`epsilon <= 0.05`). Before analysing the affected
  cell, the DeepRM E1 runner was amended to persist each cell immediately and
  to report max-step failures as explicit noncompletion/instability rows rather
  than crashing the sweep. Complete lower-magnitude cells remain analysed
  normally; noncompletion cells are excluded from Holm calculations because no
  finite paired delta exists.
- 2026-06-04: Pre-submission audit found the first completed DeepRM E1 sweep was
  not Table-IV-compatible: it used `policy_iter_1000.pt`, a pure source-style
  packer labelled `SourceTetris`, and hash-derived stochastic-policy generator
  seeds. The locked DeepRM main-study Table IV used `policy_final.pt`,
  `TetrisScheduler(source_dot=True)` (`Tetris*`, alpha=0.5 plus source dot-product
  packing), and CLI generator offsets `1000+i`, `2000+i`, `3000+i`. The old
  DeepRM E1 artifacts are therefore superseded for paper use. The DeepRM E1
  sweep is rerun with the locked checkpoint, locked `Tetris*` comparator, SJF as
  the secondary comparator, and the same policy-generator offsets as
  `cisose-deeprm evaluate-perturbations`.
