# Experiment C Results - DeepRM Action-Space Ablation

MLflow run: `53b29a1151424b9cb2f856e7f05a6977`

## Protocol

- Evaluation uses load=0.7, 30 paired seeds, 200 jobs per trace, stochastic DeepRM policy sampling, and P3 epsilon=0.05.
- P3 follows Option A: DeepRM receives the FGSM-perturbed observation; SourceTetris receives the true current state.
- The degradation statistic is within-pipeline: deg_M = DeepRM_M(P3) - DeepRM_M(clean).
- The clean competency gate requires DeepRM_M to beat SourceTetris_M on at least 20/30 paired seeds with one-sided exact binomial p<0.05.

## Training Provenance

| M | checkpoint | iteration | action dim | MLflow training run |
|---:|---|---:|---:|---|
| 10 | `results/checkpoints/author_source_rescue/load_0.7/policy_final.pt` | 1000 | 11 | `2d7bb69faec64a8aaf0f82b03ea34aea` |
| 3 | `results/checkpoints/experiment_c_m3_author_source/load_0.7/policy_final.pt` | 1000 | 4 | `e0c6e22d99834598a730bdf0184109c2` |
| 1 | `results/checkpoints/experiment_c_m1_author_source/load_0.7/policy_final.pt` | 1000 | 2 | `e708962c0ffc40baa4d810dc271828b6` |

## M10 Regression Check

The M=10 evaluator was rerun through the Experiment C harness and compared against the locked main-paper P3 pipeline. All four reference means are within 5%: `True`.

| Quantity | Relative error |
|---|---:|
| Clean DeepRM | 0.000% |
| Clean SourceTetris | 0.000% |
| P3 DeepRM | 0.000% |
| P3 SourceTetris | 0.000% |

## Results Table

| M | Clean DeepRM | Clean SourceTetris | P3 DeepRM | P3 SourceTetris | deg_M (95% CI) | Argmax change | TV distance | Delta clean | Delta P3 | Competency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 35.553 | 59.805 | 36.236 | 59.805 | +0.683 [-0.405, +1.792] | 0.752 | 0.373 | +24.252 | +23.569 | 30/30, p=9.313e-10, pass=True |
| 3 | 71.493 | 81.642 | 71.024 | 81.642 | -0.469 [-1.673, +0.646] | 0.634 | 0.315 | +10.149 | +10.618 | 28/30, p=4.34e-07, pass=True |
| 1 | 110.418 | 109.260 | 91.046 | 109.260 | -19.373 [-23.614, -15.297] | 0.228 | 0.205 | -1.159 | +18.214 | 15/30, p=0.5722, pass=False |

## Hypothesis Verdict

Case: `M1_not_learnable_H_C1_not_supported_for_competent_conditions`.

M=1 failed the clean competency gate, so it is not interpretable as a robustness test. Among competent conditions, reducing M from 10 to 3 did not increase FGSM aggregate degradation; H_C1 is not supported under the locked training budget.

Key comparisons:

- deg_3 / deg_10 = -0.686.
- deg_1 / deg_10 = -28.351.
- deg_3 - 2*deg_10 = -1.835 [-4.195, +0.574].
- deg_1 - 4*deg_10 = -22.106 [-28.358, -16.357].

## Paper Text Branch

The M=1 condition should be treated as a failed competency branch, not as evidence about FGSM robustness. The paper should use M=10 and M=3 for the interpretable mechanism test and state that the moderate action-space reduction did not increase aggregate FGSM degradation. Therefore, Experiment C does not confirm action redundancy as the operative defence for DeepRM; the binary visible-action condition instead shows that the locked source-aligned training budget no longer produces a competent clean scheduler when the action set is reduced to one visible job plus wait.

## Artifacts

- `results/paper/experiments/experiment_c/data/experiment_c_slowdowns.csv`
- `data/experiment_c_slowdowns.csv`
- `results/paper/experiments/experiment_c/tables/experiment_c_results.csv`
- `results/paper/experiments/experiment_c/figures/deeprm_ablation_degradation.pdf`
- `results/paper/experiments/experiment_c/figures/deeprm_ablation_degradation.png`
- `results/paper/experiments/experiment_c/figures/deeprm_ablation_action_diagnostics.pdf`
- `results/paper/experiments/experiment_c/figures/deeprm_ablation_action_diagnostics.png`
- `results/paper/experiments/experiment_c/figures/deeprm_ablation_training_curves.pdf`
- `results/paper/experiments/experiment_c/figures/deeprm_ablation_training_curves.png`
- `figures/deeprm_ablation_degradation.pdf`
- `figures/deeprm_ablation_degradation.png`
- `figures/deeprm_ablation_action_diagnostics.pdf`
- `figures/deeprm_ablation_action_diagnostics.png`
- `figures/deeprm_ablation_training_curves.pdf`
- `figures/deeprm_ablation_training_curves.png`
