# Paper v4 Traceability Audit — 2026-06-10

Source audited: local manuscript PDF `CISOSE_Paper_RL_V4.pdf`.

Purpose: verify that the empirical claims in the finalized paper PDF are backed
by public repository artifacts before pushing the release package.

## Summary Verdict

The major empirical claims in the paper are backed by tracked artifacts under
`docs/` and `results/paper/`.

Two manuscript-side corrections remain:

1. Section IV still contains a repository-URL placeholder.
2. The paper states that DeepRM FGSM changes the argmax in 69.8% of states. The
   committed Experiment C rerun reports 75.2% for the M=10 harness. The 69.8%
   value is present in the frozen Experiment C preregistration as the earlier
   locked diagnostic baseline. The aggregate P3 result is unchanged, but the
   diagnostic percentage should be cited with its source if retained.

No missing tracked artifact was found for the paper's primary tables, figures,
or empirical conclusions.

## Reproduction Gates

| Paper claim | Public artifact | Status |
|---|---|---|
| DeepRM strict gate: DeepRM 36.5, Tetris* 61.0, delta +24.5 | `results/paper/deeprm/tables/deeprm_clean_gate.md` | Present |
| DeepRM author-source values 19.1, 23.3, 44.8 | `results/paper/deeprm/tables/deeprm_clean_gate.md` | Present |
| Rossi Table I reproduction within sub-percent error; worst 0.36% | `results/paper/rossi/tables/rossi_reproduction_table_i.md` | Present |
| Decima official README gate fails: 3.0% improvement vs 21% target | `results/paper/decima/tables/decima_official_readme_gate.md` | Present |

## Table IV: Nine Pre-Registered Predictions

| Row | Paper value | Public artifact | Status |
|---|---:|---|---|
| P1-DeepRM | +195.7 [+185.1, +205.9] | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` | Present |
| P2-DeepRM | +91.8 [+79.9, +104.6] | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` | Present |
| P3-DeepRM | +23.6 [+21.7, +25.4] | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` | Present |
| P1-Rossi | +965.1 [+943.1, +986.5] | `results/paper/rossi/tables/rossi_p1_online_lag_sweep.md` | Present |
| P2-Rossi | -106.0 [-113.1, -98.7] | `results/paper/rossi/tables/rossi_p2_online.md` | Present |
| P3-Rossi | -91.0 [-95.4, -86.8] | `results/paper/rossi/tables/rossi_p3_online.md` | Present |
| P1-Decima | +716,396 [+653,206, +779,842] | `results/paper/decima/tables/decima_prediction_outcomes.md` | Present |
| P2-Decima | +8,044 [+1,138, +16,525] | `results/paper/decima/tables/decima_prediction_outcomes.md` | Present |
| P3-Decima | +2,001 [+1,057, +3,019] | `results/paper/decima/tables/decima_prediction_outcomes.md` | Present |

## Figures and Follow-Up Analyses

| Paper item | Public artifact | Status |
|---|---|---|
| Figure 1: Rossi bundled threshold vs HPA-v2 at P1 anchor | `results/paper/experiments/experiment_b/figures/threshold_vs_hpa_v2_under_lag.pdf` | Present |
| Figure 3: DeepRM P1 stale-action sensitivity | `results/paper/deeprm/figures/deeprm_p1_first_fit_sensitivity.pdf` and `.csv` | Present |
| Figure 4: Decima paired-seed delta distributions | `results/paper/decima/figures/decima_paired_delta_distributions.pdf` and per-seed CSV | Present |
| Table V: Rossi HPA-v2 four-cell re-evaluation | `results/paper/experiments/experiment_b/experiment_b_results.md` | Present |
| Figure 5: HPA-v2 failure-mode trace | `results/paper/experiments/experiment_b/figures/hpa_v2_clean_vs_k10_failure_trace.pdf` | Present |
| Figure 6: HPA-v2 configuration sensitivity | `results/paper/experiments/hpa_v2_config_sensitivity/figures/hpa_v2_config_p1_delta.pdf` | Present |
| Decima SRW/SRTF-style comparator sensitivity | `results/paper/experiments/decima_srtf_comparator/decima_srtf_comparator_results.md` | Present |
| Figure 7: DeepRM magnitude sweeps | `results/paper/experiments/e1_magnitude_sweep/deeprm/figures/e1_deeprm_magnitude_sweep.pdf` | Present |
| Figure 8: Rossi magnitude sweeps | `results/paper/experiments/e1_magnitude_sweep/rossi/figures/e1_rossi_magnitude_sweep.pdf` | Present |
| Decima lambda=0.25 audit and non-monotonicity note | `results/paper/experiments/e1_magnitude_sweep/E1_PRESUBMISSION_AUDIT.md` | Present |
| Rossi autocorrelation/block bootstrap sensitivity | `results/paper/experiments/experiment_a/experiment_a_results.md` | Present |
| DeepRM action-space ablation | `results/paper/experiments/experiment_c/experiment_c_results.md` | Present |

## P3 Action-Level Diagnostics

| Diagnostic claim | Public artifact | Status |
|---|---|---|
| DeepRM P3 aggregate outcome stays positive at epsilon=0.05 | `results/paper/deeprm/tables/deeprm_prediction_outcomes.md` | Present |
| DeepRM argmax-change diagnostic, earlier locked baseline 0.698 | `docs/protocols/extensions/experiment_c_action_ablation_deeprm.md` | Present as preregistration baseline |
| DeepRM M=10 Experiment C rerun argmax-change diagnostic 0.752 | `results/paper/experiments/experiment_c/tables/experiment_c_results.csv` | Present; differs from paper text |
| Decima clean/adversarial target-action probabilities 0.341/0.279 | `results/paper/decima/tables/decima_p3_fgsm_epsilon_0_05.json` | Present |
| Rossi bucket-flip attack fraction 0.2089 | `results/paper/rossi/tables/rossi_p3_online.csv` | Present |

## Manuscript File Boundary

The manuscript PDF is not part of the tracked artifact set. This repository
tracks the protocols, code, scripts, tests, and result artifacts that support
the manuscript's empirical claims.
