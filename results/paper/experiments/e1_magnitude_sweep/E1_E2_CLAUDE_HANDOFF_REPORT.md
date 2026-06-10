# E1/E2 Extension Scientific Report

Generated after VM completion on 2026-06-04. Delta convention is unchanged: `metric(comparator) - metric(RL)`. Positive delta means the learned/RL method has the lower metric; negative delta means the comparator has the lower metric.

## Run/Artifact Status

| Component | Status | MLflow run | Primary table | Figure |
|---|---|---|---|---|
| E1 DeepRM | complete after Table-IV-compatible rerun; `epsilon=0.2` reported as noncompletion | `2d4588396f474862bbf61bfebbf16ba4` | `results/paper/experiments/e1_magnitude_sweep/deeprm/tables/e1_deeprm_magnitude_sweep.csv` | `results/paper/experiments/e1_magnitude_sweep/deeprm/figures/e1_deeprm_magnitude_sweep.pdf` |
| E1 Decima | complete | `587abbac142e4d83aecd3924271d8909` | `results/paper/experiments/e1_magnitude_sweep/decima/tables/e1_decima_magnitude_sweep.csv` | `results/paper/experiments/e1_magnitude_sweep/decima/figures/e1_decima_magnitude_sweep.pdf` |
| E1 Rossi | complete | `777296676ab7416481572a92988c0899` | `results/paper/experiments/e1_magnitude_sweep/rossi/tables/e1_rossi_magnitude_sweep.csv` | `results/paper/experiments/e1_magnitude_sweep/rossi/figures/e1_rossi_magnitude_sweep.pdf` |
| E2 Companion A | complete; no new policy training | `3c972e2a15f94c84924c40f9c8afc55e` | `results/paper/experiments/e2_objective_native/tables/e2_companion_a_weight_rescore.csv` | `results/paper/experiments/e2_objective_native/figures/e2_companion_a_rescore.pdf` |

## Confirmatory E1 Takeaways

- DeepRM was rerun after a pre-submission harness audit. The superseded DeepRM E1 run used `policy_iter_1000.pt`, a pure source-style packer labelled `SourceTetris`, and hash-derived stochastic-policy generator seeds. The corrected run uses the locked Table IV checkpoint `policy_final.pt`, the locked `Tetris*` comparator (`TetrisScheduler(source_dot=True)`), SJF as the secondary comparator, and the same policy/statistic seed offsets as `cisose-deeprm evaluate-perturbations`.
- DeepRM vs Tetris*: all three E1 anchor rows now match Table IV exactly, including CIs: P1 k=10 delta=195.72 [185.11, 205.92], P2 alpha=1.5 delta=91.79 [79.85, 104.57], P3 epsilon=0.05 delta=23.57 [21.72, 25.36].
- DeepRM vs Tetris*: P1 remains positive for every lag k in the grid; P2 remains positive for every Pareto alpha tested; P3 remains positive through the confirmatory epsilon<=0.05 range. Epsilon=0.1 remains positive but weaker; epsilon=0.2 is noncompletion.
- DeepRM vs SJF: P1 and P2 remain positive for every magnitude in the grid. P3 remains positive through epsilon=0.05; epsilon=0.1 is weak/marginal and epsilon=0.2 is noncompletion.
- DeepRM severe P3 epsilon=0.2 exceeded the pre-set max-step guard and is reported as noncompletion/instability, not as a finite paired delta. This does not affect the confirmatory epsilon<=0.05 decision rule.
- Decima P1 is not sign-stable inside the realistic lambda<=1 regime: lambda=0.25 is negative (-13,305.0, Holm p(Delta<0)=0.00264) while lambda=0.5 and 1.0 are strongly positive. Audit found no fractional-step rounding in the lag injection path: delays are continuous floats added directly to task `finish_time` and `duration`. The deterministic exponential draw includes the lag mean in its hash key, so adjacent lambda cells are valid per-cell paired tests but not common-random-number scaled versions of the same delay draws. This weakens any simple monotone “Decima P1 falsified across magnitudes” claim.
- Decima P3 remains positive throughout epsilon in [0, 0.2]; anchor epsilon=0.05 delta=1,903.6. FGSM does not produce the expected comparator-over-Decima reversal in this sweep.
- Rossi HPA-v2 attenuation is confirmed: max |HPA-v2 delta|=100.97, which is 10.5% of the bundled-threshold k=10 collapse (965.09), below the 25% criterion.
- Rossi P2 is negative at every alpha tested (range -120.65 to -90.01); the threshold comparator has lower total cost throughout the tail sweep.
- Rossi P3 is negative at every epsilon tested (anchor epsilon=0.05 delta=-91.00); the bucket-flip result is robust across the epsilon grid.

## Characterisation Notes

- Decima P2 is not a fragility curve. Delta increases with tail weight; w=0.5 is positive but marginal after within-curve Holm (p(Delta>0)=0.0636), while w=0.75 and w=1.0 are positive and Holm-significant.
- Rossi bundled-threshold P1 is magnitude-dependent: clean/small lags k=0,1,2 are negative, but k>=5 becomes strongly positive and grows to 1,266.7 at k=50. The large k=10 collapse remains real for the bundled comparator, but it is not a smooth all-lag property.
- E2 Companion A: under clean windows, threshold remains lower-scoring than Rossi for every churn weight (deltas -487.00 to -99.04). Under P1 k=10, Rossi remains lower-scoring than threshold for every churn weight (deltas 978.44 to 1,047.1). This is a rescoring result only; full native-objective retraining remains held because Rossi is tabular online, not a fixed neural checkpoint policy.

## Recommended Paper Integration

- Use the Decima E1 plot to soften the Decima P1 claim: report the lambda=0.25 sign reversal explicitly.
- Use the DeepRM E1 plot to show robust falsification through the realistic P1/P3 range, while marking epsilon=0.2 as noncompletion.
- Use the Rossi E1 plot to separate two claims: bundled-threshold lag collapse is robust for k>=5, but HPA-v2 attenuation remains far below the bundled collapse across all k.
- Keep E2 Companion A as an appendix/sensitivity result, not as full E2 native-policy evidence.
