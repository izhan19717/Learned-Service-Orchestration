# Paper Artifact Traceability

This page maps the empirical claims, tables, and figures reported in the paper
to the committed repository artifacts that support them. The repository contains
the protocols, source code, scripts, tests, and result artifacts needed to
inspect and reproduce the empirical evidence.

## Reproduction Gates

| Claim | Repository artifact |
|---|---|
| DeepRM strict gate: DeepRM 36.5, Tetris* 61.0, delta +24.5 | `results/paper/deeprm/tables/deeprm_clean_gate.md` |
| DeepRM author-source values 19.1, 23.3, 44.8 | `results/paper/deeprm/tables/deeprm_clean_gate.md` |
| Rossi Table I reproduction within sub-percent error; worst 0.36% | `results/paper/rossi/tables/rossi_reproduction_table_i.md` |
| Decima official README gate: 3.0% improvement vs 21% target | `results/paper/decima/tables/decima_official_readme_gate.md` |

## Nine Pre-Registered Predictions

| Row | Reported value | Repository artifact |
|---|---:|---|
| P1-DeepRM | +195.7 [+185.1, +205.9] | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` |
| P2-DeepRM | +91.8 [+79.9, +104.6] | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` |
| P3-DeepRM | +23.6 [+21.7, +25.4] | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` |
| P1-Rossi | +965.1 [+943.1, +986.5] | `results/paper/rossi/tables/rossi_p1_online_lag_sweep.md` |
| P2-Rossi | -106.0 [-113.1, -98.7] | `results/paper/rossi/tables/rossi_p2_online.md` |
| P3-Rossi | -91.0 [-95.4, -86.8] | `results/paper/rossi/tables/rossi_p3_online.md` |
| P1-Decima | +716,396 [+653,206, +779,842] | `results/paper/decima/tables/decima_prediction_outcomes.md` |
| P2-Decima | +8,044 [+1,138, +16,525] | `results/paper/decima/tables/decima_prediction_outcomes.md` |
| P3-Decima | +2,001 [+1,057, +3,019] | `results/paper/decima/tables/decima_prediction_outcomes.md` |

## Figures and Additional Analyses

| Paper item | Repository artifact |
|---|---|
| Rossi bundled threshold vs HPA-v2 at P1 anchor | `results/paper/experiments/experiment_b/figures/threshold_vs_hpa_v2_under_lag.pdf` |
| DeepRM P1 stale-action sensitivity | `results/paper/deeprm/figures/deeprm_p1_first_fit_sensitivity.pdf` and `results/paper/deeprm/tables/deeprm_p1_first_fit_sensitivity.md` |
| Decima paired-seed delta distributions | `results/paper/decima/figures/decima_paired_delta_distributions.pdf` and `results/paper/decima/tables/decima_per_seed_deltas.csv` |
| Rossi HPA-v2 four-cell re-evaluation | `results/paper/experiments/experiment_b/experiment_b_results.md` |
| HPA-v2 failure-mode trace | `results/paper/experiments/experiment_b/figures/hpa_v2_clean_vs_k10_failure_trace.pdf` |
| HPA-v2 configuration sensitivity | `results/paper/experiments/hpa_v2_config_sensitivity/figures/hpa_v2_config_p1_delta.pdf` |
| Decima SRTF/Graphene-style comparator sensitivity | `results/paper/experiments/decima_srtf_comparator/decima_srtf_comparator_results.md` |
| DeepRM magnitude sweeps | `results/paper/experiments/e1_magnitude_sweep/deeprm/figures/e1_deeprm_magnitude_sweep.pdf` |
| Rossi magnitude sweeps | `results/paper/experiments/e1_magnitude_sweep/rossi/figures/e1_rossi_magnitude_sweep.pdf` |
| Decima magnitude sweeps | `results/paper/experiments/e1_magnitude_sweep/decima/figures/e1_decima_magnitude_sweep_paper_panel.pdf` |
| Decima lambda=0.25 reconciliation note | `results/paper/experiments/e1_magnitude_sweep/E1_RECONCILIATION_NOTE.md` |
| Rossi autocorrelation/block-bootstrap sensitivity | `results/paper/experiments/experiment_a/experiment_a_results.md` |
| DeepRM action-space ablation | `results/paper/experiments/experiment_c/experiment_c_results.md` |

## P3 Action-Level Diagnostics

| Diagnostic | Repository artifact |
|---|---|
| DeepRM P3 aggregate outcome stays positive at epsilon=0.05 | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` |
| DeepRM locked P3 diagnostic: argmax changes in 69.8% of states | `results/evaluation/deeprm/fgsm_effect_diagnostic.json` |
| DeepRM Experiment C M=10 diagnostic: argmax changes in 75.2% of states | `results/paper/experiments/experiment_c/tables/experiment_c_results.csv` |
| Decima clean/adversarial target-action probabilities 0.341/0.279 | `results/paper/decima/tables/decima_p3_fgsm_epsilon_0_05.json` |
| Rossi bucket-flip attack fraction 0.2089 | `results/paper/rossi/tables/rossi_p3_online.csv` |

The DeepRM 69.8% and 75.2% action-change diagnostics use the same final
checkpoint and epsilon=0.05 but count states from different diagnostic passes.
The Experiment C M=10 run reproduces the locked aggregate P3 values exactly;
therefore these diagnostics describe sampling differences in the action-level
measurement, not conflicting aggregate outcomes.
