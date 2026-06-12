# Experiment C — Action-Space Ablation for DeepRM under FGSM Perturbation

**Status: pre-registered. Specification frozen before any new training runs are executed.**

## 0. Purpose

The action-redundancy hypothesis is a candidate explanation for why FGSM
attacks on DeepRM affect the action distribution but do not substantially change
aggregate slowdown. The hypothesis is: the action neighbourhood near a given
state contains many near-equivalent scheduling decisions, so an attack that
flips the policy's argmax often selects an alternative that is not
catastrophically worse.

This experiment tests that mechanism directly by restricting the action space to
remove near-equivalent choices. If action redundancy is the operative defence,
removing redundant alternatives should cause the FGSM attack to degrade the
aggregate slowdown metric. If the attack still has small aggregate effect after
restriction, the action-redundancy hypothesis is not supported and the mechanism
is elsewhere.

## 1. Mechanism of the ablation

DeepRM's action space at each step is `|A| = M + 1`: pick one of the `M` jobs visible in the queue, or wait (null action). In the source-aligned configuration `M = 10`, giving 11 actions per state.

We ablate by reducing `M`. At smaller `M`, fewer jobs are visible at any one decision step and the action choices are correspondingly less redundant. We do not reorder or filter the queue contents (so the underlying workload distribution is unchanged); we only change how many of the queued jobs are visible to the policy at each step.

Three conditions:

| Condition | M | |A| | Interpretation |
|-----------|---|----|-----|
| Full (baseline) | 10 | 11 | Source-aligned protocol, already trained and used in main paper. |
| Reduced | 3 | 4 | Three visible jobs plus wait; substantial reduction in near-equivalent alternatives. |
| Minimal | 1 | 2 | One visible job plus wait; binary choice; essentially eliminates action-level redundancy. |

The training pipeline and evaluation protocol are otherwise identical across all three conditions: same workload generator, same hyperparameters, same number of training iterations (1000), same 30 paired evaluation seeds.

## 2. Implementation plan

1. Modify the DeepRM training pipeline to accept `M` as a runtime configuration parameter. Verify that the `M = 10` configuration reproduces the existing P3-pipeline clean evaluation (`ε = 0`) to within 5% as a regression check before training the ablated conditions: DeepRM mean slowdown 35.55 and SourceTetris mean slowdown 59.80. The strict-gate clean result (DeepRM 36.5 against SourceTetris 61.0) remains the separate source-aligned competency gate, but it is not the baseline used for within-pipeline FGSM degradation.
2. Train DeepRM at `M = 3` and `M = 1` for 1000 iterations each, using the same hyperparameters as the main paper.
3. For each `M`, run the clean evaluation and the P3 FGSM evaluation at `ε = 0.05` over the 30 paired evaluation seeds used in the main paper.
4. Compute the diagnostics in §3.
5. Apply the statistical procedure in §4.

## 3. Diagnostics per condition

For each `M ∈ {10, 3, 1}` and each of {clean, P3}:

| Diagnostic | Description |
|------------|-------------|
| Mean slowdown DeepRM_M | Per-seed mean slowdown; report mean over 30 seeds. |
| Mean slowdown SourceTetris_M | Per-seed mean slowdown of SourceTetris evaluated on the same restricted queue. |
| Argmax change rate (P3 only) | Fraction of decision steps where the policy's argmax under perturbation differs from clean. |
| Mean clean-argmax probability drop (P3 only) | Average over decision steps of `π_clean(argmax_clean | s) − π_FGSM(argmax_clean | s)`. |
| Total-variation distance (P3 only) | Mean over decision steps of `½ Σ_a |π_clean(a|s) − π_FGSM(a|s)|`. |
| Action-distribution entropy (clean) | Mean over decision steps of `−Σ_a π_clean(a|s) log π_clean(a|s)`; expected to decrease with `M` (the policy concentrates more on a single action when fewer alternatives exist). |

Compute also the **competency gate** for each condition: the trained policy `DeepRM_M` must beat `SourceTetris_M` on the clean cell on at least 20 of 30 paired seeds (one-sided binomial test, p < 0.05 against null of equal win rate). If the M=1 condition fails this gate, the result is reported as "M=1 is not learnable under the source-aligned protocol" and the analysis proceeds with `M ∈ {10, 3}`.

## 4. Statistical procedure

For each `M`, compute the **FGSM aggregate degradation**:

```
deg_M = mean_slowdown(DeepRM_M, P3) − mean_slowdown(DeepRM_M, clean)
```

with paired bootstrap CI over the 30 seeds.

For `deg_M`, the clean value is the same M-specific `ε = 0` evaluation from
the P3 pipeline. It is not the strict-gate clean evaluation. This keeps the
degradation estimate within-pipeline and avoids mixing small differences in
evaluation harness, seed handling, or checkpoint selection into the FGSM effect.

Compute also the **comparator-paired statistic** at each `M`:

```
Δ_M(clean) = mean_slowdown(SourceTetris_M, clean) − mean_slowdown(DeepRM_M, clean)
Δ_M(P3)    = mean_slowdown(SourceTetris_M, P3)    − mean_slowdown(DeepRM_M, P3)
```

The action-redundancy hypothesis predicts:

```
deg_1 > deg_3 > deg_10                     (degradation grows as M shrinks)
Δ_1(P3) < Δ_1(clean) by a larger margin than Δ_10(P3) < Δ_10(clean)
```

Test both predictions with paired bootstrap on the per-seed differences.

## 5. Pre-registered hypotheses

We pre-register three competing hypotheses for the relationship between `M` and FGSM aggregate degradation.

- **H_C1 — action redundancy is the operative defence.** `deg_M` grows monotonically as `M` decreases, and the growth is substantial. Quantitative prediction: `deg_3 ≥ 2 × deg_10` and `deg_1 ≥ 4 × deg_10`, with the inequalities surviving paired bootstrap 95% CIs.

- **H_C2 — action redundancy is not the operative defence.** `deg_M` is approximately constant across `M`, indicating that the policy's robustness to FGSM at the aggregate level is due to some other mechanism (online retraining behaviour, intrinsic action smoothing through the softmax temperature, structural property of the convolutional encoder, etc.). Quantitative prediction: `deg_3 ∈ [0.5, 2] × deg_10` and `deg_1 ∈ [0.5, 2] × deg_10`.

- **H_C3 — mixed.** `deg_3` is close to `deg_10` (no monotone effect at the moderate reduction) but `deg_1` is substantially larger (the minimum-action condition removes structural alternatives that the moderate condition retains). This would indicate that the redundancy mechanism operates only for the smallest action sets and that the moderate-reduction condition still has functional redundancy.

The outcome falsifies whichever hypothesis is inconsistent with the data.

## 6. Diagnostic outputs

Render two figures.

`figures/deeprm_ablation_degradation.pdf`. A bar chart showing `deg_M` for `M ∈ {10, 3, 1}` with paired bootstrap 95% CI error bars. Each bar shows the mean change in DeepRM_M slowdown from clean to P3. The visual hypothesis is monotone growth in bar height as `M` decreases.

`figures/deeprm_ablation_action_diagnostics.pdf`. A 3-panel figure with one panel per `M`: panel 1 plots argmax change rate; panel 2 plots TV distance; panel 3 plots clean-argmax probability drop. All three are expected to be roughly constant across `M` under H_C1 (the attack lands at the same action-level intensity regardless of action-set size; the difference is in whether the action-level effect translates to aggregate degradation).

## 7. Output table

| M | clean slowdown DeepRM_M | clean slowdown SourceTetris_M | P3 slowdown DeepRM_M | P3 slowdown SourceTetris_M | deg_M (95% CI) | argmax change rate | TV distance | Δ_M(clean) | Δ_M(P3) |
|---|------------------------|-------------------------------|-----------------------|----------------------------|-----------------|---------------------|--------------|-------------|----------|
| 10 | 35.55 (existing P3 ε=0; strict gate 36.5 for competency) | 59.80 (existing P3 ε=0; strict gate 61.0 for competency) | 36.24 (existing) | 59.80 (existing) | +0.68 (existing) | 0.698 (existing) | 0.355 (existing) | +24.25 (existing) | +23.57 (existing) |
| 3 | … | … | … | … | … | … | … | … | … |
| 1 | … | … | … | … | … | … | … | … | … |

## 8. Interpretation rules (pre-registered)

The mechanism interpretation is determined by which of H_C1, H_C2, H_C3
prevails.

**Case 1 — H_C1 prevails** (action redundancy confirmed).

Action redundancy is supported for DeepRM if aggregate FGSM degradation grows
monotonically and substantially as `M` decreases. Under this case, the ablation
identifies action redundancy as an operative mechanism for the observed
DeepRM P3 behaviour.

**Case 2 — H_C2 prevails** (action redundancy falsified).

Action redundancy is not supported if aggregate FGSM degradation remains
approximately constant as `M` decreases. Under this case, DeepRM's small
aggregate FGSM effect must be attributed to another mechanism, such as policy
smoothing, encoder structure, or training-induced regularisation.

**Case 3 — H_C3 prevails** (mixed).

Action redundancy is partially supported if `M = 3` remains close to the
source-aligned `M = 10` condition but `M = 1` is qualitatively different. Under
this case, the redundancy mechanism is interpreted as thresholded rather than
smoothly monotone.

In all three cases, the value of the experiment is that the action-redundancy claim is no longer asserted without test. The directional argument for D2 is supported by the cross-method pattern regardless of which case prevails.

## 9. References

- Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). "Resource Management with Deep Reinforcement Learning." *Proc. HotNets*.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). "Explaining and Harnessing Adversarial Examples."
- The action-space-first direction discussed in §V.B of the main paper.
