# Rossi 2019 Results Report

Date: 2026-05-21

This report summarizes the completed Rossi 2019 branch for the CISOSE empirical
section. It records the protocol choices, result interpretation, and evidence
needed to inspect the Rossi result package.

## Executive Status

Rossi is complete for the paper's canonical minimum evidence package:

- Table I reproduction gate: complete and passed.
- P1 canonical online observation-lag anchor: complete.
- P2 online service-time tail anchor: complete.
- P3 online bucket-flip anchor: complete.
- Paper tables and plots: generated.
- MLflow provenance: present for all canonical results.
- Tests after final Rossi code changes: `63 passed, 7 warnings`.

Scientific status:

- Rossi is scientifically usable, but it does not support all three predicted
  claims.
- P1-Rossi is falsified under the canonical online-adaptive protocol.
- P2-Rossi and P3-Rossi are directionally confirmed, but both confirmations
  occur on top of clean HPA dominance. They should be framed as "the
  perturbation widens an existing HPA advantage," not as a new crossover caused
  only by the perturbation.
- The optional full six-lag online P1 curve was not completed because it was
  computationally expensive. The pre-registered anchor-plus-clean evidence is
  complete. Do not present the missing full curve as a limitation unless the
  paper claims a shape across the whole lag sweep.
- Final Holm-Bonferroni correction across all nine method-predictions cannot
  be finalized until Decima is complete. Rossi raw directional p-values for the
  canonical anchor cells are all at the Monte Carlo floor where applicable, but
  P1 is directionally falsified regardless of correction.

Scope note:

- Rossi P1 conclusions use the online adaptive controller artifacts listed
  below. Frozen-controller P1 artifacts are retained only as sensitivity
  outputs and are not used for canonical Rossi P1 conclusions.

## Key Protocol Definitions

Primary method:

- Rossi, Nardelli, and Cardellini 2019 model-based tabular RL autoscaler.
- The official Java simulator repository is treated as operational ground
  truth for state buckets, action set, reward weights, and workload profile.
- Local source-aligned Python port uses the official RLAD simulator behavior.

Canonical perturbation protocol:

- Evaluate the published online adaptive model-based controller, not a final
  frozen checkpoint.
- This follows `SCIENTIFIC_DECISIONS.md` Decision 20 and Decision 22.
- The online controller is initialized as in the Table I reproduction gate and
  updates during each evaluation episode.
- This is not treated as perturbation retraining. It is the operational form of
  the Rossi controller in the published simulator.

Comparator:

- Source-derived HPA-style threshold controller.
- For P1, observation lag is environmental, so both Rossi and HPA receive
  lagged utilization observations.
- For P3, the bucket-flip perturbation is adversarial to Rossi, so Rossi acts
  on the perturbed observation and HPA remains on the true utilization state.

Statistical unit:

- 30 non-overlapping windows from the official Java-equivalent slow-profile
  stream.
- Horizon per window: `4001` decision ticks.
- Base seed for offsets: `20260520`.
- Bootstrap seed base: `20260521`.
- Paired bootstrap: 5000 resamples, percentile CI.
- Directional sign-flip p-values are recorded in the CSV tables.

Sign convention:

- `Delta = total_cost(HPA) - total_cost(Rossi)`.
- Negative Delta means HPA has lower total cost than Rossi.
- Positive Delta means Rossi has lower total cost than HPA.
- The preregistered Rossi predictions expected Delta below zero at the anchor.

## Reproduction Gate

Canonical artifact:

- Table: `results/paper/rossi/tables/rossi_reproduction_table_i.md`
- Figure: `results/paper/rossi/figures/rossi_reproduction_gate.pdf`
- MLflow run: `8891102c835f4ec2a5a109529845501f`

Gate target:

- Rossi 2019 Table I, 5-action, performance-weighted, model-based row.
- Gate tolerance: within 15% of each target metric.

Observed reproduction:

| Metric | Observed | Target | Relative error |
|---|---:|---:|---:|
| Rmax violations (%) | 2.37441 | 2.37 | 0.186% |
| Average CPU utilization (%) | 60.7576 | 60.54 | 0.359% |
| Average CPU share (%) | 87.6206 | 87.62 | 0.001% |
| Average number of containers | 2.53237 | 2.53 | 0.094% |
| Median response time (ms) | 10.392 | 10.39 | 0.019% |
| Adaptations (%) | 39.6401 | 39.67 | 0.075% |

Conclusion:

- Reproduction gate passed decisively.
- Worst relative error was 0.359%, far inside the 15% gate.

## P1-Rossi: Observation Lag

Canonical artifact:

- Table: `results/paper/rossi/tables/rossi_p1_online_lag_sweep.md`
- CSV: `results/paper/rossi/tables/rossi_p1_online_lag_sweep.csv`
- Figure: `results/paper/rossi/figures/rossi_p1_online_observation_lag.pdf`
- JSON: `results/rossi/p1_online_observation_lag_sweep.json`
- MLflow run: `7e0d663771364a37bfd29b971ca2ab19`

Protocol:

- Online adaptive Rossi model-based controller.
- HPA comparator receives lagged utilization too, because P1 is an
  environmental realism perturbation.
- Clean `k = 0` row reused from the completed P2 clean cell because it is the
  exact same unperturbed online Rossi/HPA simulation under the same offsets and
  horizon.
- Anchor `k = 10` was newly executed.
- Online update semantics: Rossi's online state update and action selection
  both use the delayed utilization bucket. Replica count, CPU allocation, input
  rate, and realized cost remain current.

Results:

| Lag k | Rossi cost | HPA cost | Delta HPA-Rossi | 95% CI | Outcome |
|---:|---:|---:|---:|---:|---|
| 0 | 204.0891 | 118.2825 | -85.8066 | [-89.9006, -81.7851] | Clean HPA advantage |
| 10 | 150.6074 | 1115.6943 | 965.0869 | [943.0878, 986.5150] | P1 falsified |

Additional diagnostics at `k = 10`:

| Diagnostic | Rossi | HPA |
|---|---:|---:|
| SLA violation rate | 0.01516 | 0.26728 |
| Mean action churn | 877.93 | 443.37 |
| Mean absolute observation delta | 0.09533 | 0.83952 |

Interpretation:

- P1-Rossi is falsified.
- The preregistered hypothesis expected HPA to beat Rossi under realistic
  observation lag. The opposite happened.
- Under online adaptation, lag severely harms HPA while Rossi remains robust and
  even improves relative to its clean online cost on these evaluation windows.
- This is strong evidence against the P1-Rossi prediction.

Frozen-controller P1 sensitivity:

- Frozen P1 artifact: `results/paper/rossi/tables/rossi_p1_lag_sweep.md`
- MLflow run: `9aad14b0a22b4fcc8d4f8b2f7e8d5784`
- Status: non-canonical sensitivity only.
- Reason: the final frozen checkpoint does not reproduce Rossi 2019's online
  behavior and should not be used as primary evidence.

## P2-Rossi: Service-Time Tail

Canonical artifact:

- Table: `results/paper/rossi/tables/rossi_p2_online.md`
- CSV: `results/paper/rossi/tables/rossi_p2_online.csv`
- Figure: `results/paper/rossi/figures/rossi_p2_online.pdf`
- JSON: `results/rossi/p2_online_service_tail.json`
- MLflow run: `84cb1061cb0a45f1a115805fc30ce885`

Run-status note:

- The P2 artifacts were completed and retained as the canonical Rossi P2
  evidence.
- Canonical P3 artifacts were produced in a separate run and are reported in
  the P3 section.
- Decision 21 records the retention criterion for the P2 artifact set.

Protocol:

- Online adaptive Rossi model-based controller.
- HPA evaluated on the same service-time tail condition.
- `alpha = infinity` is the deterministic-service Java baseline.
- `alpha = 1.5` maps to a mean-preserving capped-Pareto second moment with cap
  ratio 100.
- At `alpha = 1.5`, `service_time_cv2 = 3.7193877551020385`.

Results:

| Value | Rossi cost | HPA cost | Delta HPA-Rossi | 95% CI | Outcome |
|---:|---:|---:|---:|---:|---|
| alpha = infinity | 204.0891 | 118.2825 | -85.8066 | [-89.9006, -81.7851] | Clean HPA advantage |
| alpha = 1.5 | 224.9811 | 119.0025 | -105.9786 | [-113.1341, -98.6716] | P2 confirmed |

Effect relative to clean:

- Clean Delta: -85.8066.
- Tail Delta: -105.9786.
- Tail makes the HPA advantage larger by about 20.1720 total-cost units.

Interpretation:

- P2-Rossi is directionally confirmed by the preregistered anchor criterion.
- However, HPA already beats Rossi in the clean online cell.
- Therefore the paper should not say "the service-time tail causes Rossi to
  lose to HPA." Rossi already loses clean.
- Correct phrasing: "The service-time tail widens an existing HPA advantage
  over online Rossi."

## P3-Rossi: Minimum Bucket-Flip Perturbation

Canonical artifact:

- Table: `results/paper/rossi/tables/rossi_p3_online.md`
- CSV: `results/paper/rossi/tables/rossi_p3_online.csv`
- Figure: `results/paper/rossi/figures/rossi_p3_online.pdf`
- JSON: `results/rossi/p3_online_bucket_flip.json`
- MLflow run: `7cd0ed8786be4c6ba22aae3ef7d7227a`

Protocol:

- Online adaptive Rossi model-based controller.
- P3 is adversarial, so the bucket-flip perturbation is applied only to Rossi's
  observation/action selection.
- HPA remains on the true utilization state.
- Clean `epsilon = 0` row reuses the same online clean cell as P2, with P3 CI
  recomputed under the P3 bootstrap seed.
- Anchor: `epsilon = 0.05`.

Results:

| Value | Rossi cost | HPA cost | Delta HPA-Rossi | 95% CI | Outcome |
|---:|---:|---:|---:|---:|---|
| epsilon = 0.0 | 204.0891 | 118.2825 | -85.8066 | [-89.8761, -81.7288] | Clean HPA advantage |
| epsilon = 0.05 | 209.2841 | 118.2825 | -91.0016 | [-95.4086, -86.8216] | P3 confirmed |

Perturbation diagnostics at `epsilon = 0.05`:

- Attack fraction: 0.20888944430559028.
- Mean absolute utilization perturbation: 0.005072740186373026.
- Max absolute utilization perturbation: 0.04852750320458554.

Effect relative to clean:

- Clean Delta: -85.8066.
- Perturbed Delta: -91.0016.
- Bucket flip widens the HPA advantage by about 5.1950 total-cost units.

Interpretation:

- P3-Rossi is directionally confirmed by the preregistered anchor criterion.
- As with P2, the result is not a fresh crossover from Rossi winning clean to
  HPA winning under attack.
- Correct phrasing: "The minimum bucket-flip perturbation further worsens online
  Rossi relative to HPA, widening an already-present HPA advantage."

## Final Rossi Prediction Outcomes

| Prediction | Anchor | Canonical outcome | Paper interpretation |
|---|---:|---|---|
| P1-Rossi | k = 10 seconds | Falsified | Online Rossi is robust to observation lag relative to HPA; HPA collapses under lag. |
| P2-Rossi | alpha = 1.5 | Confirmed | Tail shift widens an already-existing HPA advantage. |
| P3-Rossi | epsilon = 0.05 | Confirmed | Bucket-flip attack widens an already-existing HPA advantage. |

## Recommended Paper Language

Use this language or equivalent:

> The Rossi branch passed its author-source reproduction gate with sub-percent
> error on all Table I metrics. Under the paper-faithful online adaptive
> controller protocol, the Rossi results are mixed. Observation lag falsified
> P1-Rossi: at k = 10 seconds, online Rossi substantially outperformed the HPA
> comparator. Service-time tail shift and bucket-flip observation perturbation
> confirmed P2-Rossi and P3-Rossi directionally, but HPA already dominated
> online Rossi in the clean cell. We therefore interpret P2 and P3 as widening
> an existing HPA advantage rather than producing new perturbation-induced
> crossovers.

Avoid saying:

- "All Rossi predictions are confirmed."
- "Observation lag makes Rossi lose to HPA."
- "P2/P3 prove a clean crossover under perturbation."
- "Frozen Rossi P1 is the canonical result."

Acceptable concise conclusion:

> Rossi provides partial support for the paper's broader thesis: the
> distribution-tail and adversarial-observation axes support the fragility
> pattern directionally, but observation lag does not. The lag result is a
> meaningful falsification under the source-faithful online protocol and should
> be reported plainly.

## Artifact Index

Protocol and decision files:

- `docs/protocols/preregistration_v2_2.md`
- `docs/protocols/calibration_v2_2.md`
- `docs/implementation_notes.md`

Canonical result files:

- `results/paper/rossi/tables/rossi_reproduction_table_i.md`
- `results/paper/rossi/tables/rossi_p1_online_lag_sweep.md`
- `results/paper/rossi/tables/rossi_p2_online.md`
- `results/paper/rossi/tables/rossi_p3_online.md`
- `results/paper/rossi/figures/rossi_reproduction_gate.pdf`
- `results/paper/rossi/figures/rossi_p1_online_observation_lag.pdf`
- `results/paper/rossi/figures/rossi_p2_online.pdf`
- `results/paper/rossi/figures/rossi_p3_online.pdf`

Non-canonical/sensitivity files:

- `results/paper/rossi/tables/rossi_p1_lag_sweep.md`
- `results/paper/rossi/figures/rossi_p1_observation_lag.pdf`

These frozen-controller P1 files are sensitivity artifacts and are not used for
canonical Rossi P1 conclusions.
