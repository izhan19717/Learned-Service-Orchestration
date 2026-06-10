# Rossi Methodology QA

Date: 2026-05-21

This note answers the six methodology questions about the canonical Rossi
online-adaptive results.

Generated diagnostics:

- MLflow run: `f9affee6a335408ea4bbc82a7b31a36a`
- JSON: `results/rossi/methodology_qa_diagnostics.json`
- Per-window summary CSV:
  `results/paper/rossi/tables/rossi_per_window_delta_diagnostics.csv`
- Per-window delta vector CSV:
  `results/paper/rossi/tables/rossi_per_window_delta_vectors.csv`
- P1 median-window failure trace:
  `results/paper/rossi/figures/rossi_p1_k10_median_window_failure_trace.pdf`
- HPA clean-vs-lag failure trace:
  `results/paper/rossi/figures/rossi_hpa_clean_vs_k10_failure_trace.pdf`
- Q convergence diagnostic:
  `results/paper/rossi/figures/rossi_clean_q_l1_convergence_diagnostic.pdf`
  and
  `results/paper/rossi/tables/rossi_clean_q_l1_convergence_diagnostic.csv`

## Q1. Per-Window Initialization Protocol

Implemented behavior: **(a) cold start per window**.

For each evaluation window, the runner constructs a fresh
`ModelBasedController(DEFAULT_CONFIG)`. Therefore:

- Q-table starts from the model-based controller's initialized table.
- Transition counts/probabilities start from the identity prior.
- Unknown-cost table starts from zeros.
- There is no carry-over from one evaluation window to the next.
- No fixed pretrained checkpoint is loaded for canonical online P1/P2/P3.

This applies to:

- P1 online anchor at `k = 10`.
- P2 online clean/tail cells.
- P3 online attack cell.

The clean `k = 0`/`alpha = infinity`/`epsilon = 0` cell is also a set of 30
independent cold-start online trajectories. It is reused across P1/P2/P3 when
the cell is exactly the same unperturbed simulation.

Scientific implication:

- Clean-cell HPA dominance is partly a sample-efficiency / online-adaptation
  result. It does not mean a long-running Rossi controller carried continuously
  across many windows would necessarily remain worse.
- The current canonical experiment evaluates the operational Rossi controller
  as an online learner restarted at the beginning of each official-profile
  window, matching the way the Table I reproduction gate is run.

## Q2. Mean Absolute Observation Delta Definition

Exact formula:

For each controller trajectory,

`mean_abs_obs_delta = mean_t | observed_utilization_t - true_current_utilization_t |`

where:

- `true_current_utilization_t` is the utilization in that controller's own
  closed-loop simulator state at tick `t`.
- `observed_utilization_t` is the delayed utilization supplied to that same
  controller.
- The reported Rossi/HPA values are then averaged over the 30 windows.

Important correction:

- Under P1 Option B, both controllers receive the same **lag rule and lag
  magnitude**, not the same scalar observation sequence.
- Rossi and HPA induce different replica/CPU trajectories, so their true
  utilization time series differ.
- Their delayed observations therefore also differ.

This is why P1 `k = 10` reports:

| Controller | Mean abs observation delta |
|---|---:|
| Rossi | 0.095334 |
| HPA | 0.839516 |

Conclusion:

- P1 is still Option B in the closed-loop environmental sense: both controllers
  are evaluated under 10-second stale telemetry.
- But the paper must not say "both controllers receive the identical lagged
  observation sequence." The correct phrasing is: "each controller receives
  stale telemetry from the system state induced by its own actions."

## Q3. Table I Reproduction Window Identity

The Table I reproduction gate uses the **first 4001-tick window** of the
official profile, i.e. offset `0`.

The 30 canonical P1/P2/P3 evaluation windows use these offsets:

`12003, 28007, 36009, 56014, 60015, 76019, 84021, 96024, 116029, 120030, 148037, 152038, 180045, 184046, 192048, 224056, 240060, 248062, 256064, 260065, 276069, 348087, 380095, 392098, 396099, 432108, 436109, 456114, 492123, 512128`

Therefore the Table I first window is **excluded** from the 30 evaluation
windows.

First-window total-cost diagnostic:

| Window | Rossi cost | HPA cost | Delta HPA-Rossi |
|---|---:|---:|---:|
| Table I first window, offset 0 | 165.3164 | 98.4150 | -66.9014 |
| 30-window clean average | 204.0891 | 118.2825 | -85.8066 |

Conclusion:

- The Rossi Table I reproduction gate is not one of the 30 perturbation windows.
- HPA also beats Rossi on the Table I first window by total cost, though that
  HPA comparison is not the Table I reproduction target.
- The clean HPA advantage is not caused by a single included/excluded first
  window artifact.

## Q4. Per-Window Delta Distribution

Sign convention:

`Delta = total_cost(HPA) - total_cost(Rossi)`.

- Negative Delta: HPA lower cost.
- Positive Delta: Rossi lower cost.

Summary over the 30 paired windows:

| Cell | Min | Q25 | Median | Q75 | Max | Mean | Rossi beats HPA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | -106.8504 | -93.7264 | -83.4706 | -78.1576 | -58.1652 | -85.8066 | 0/30 |
| P1 `k=10` | 782.9536 | 934.8154 | 962.0975 | 1001.3152 | 1099.8661 | 965.0869 | 30/30 |
| P2 `alpha=1.5` | -146.9042 | -115.3530 | -105.9335 | -97.2865 | -63.1332 | -105.9786 | 0/30 |
| P3 `epsilon=0.05` | -125.4729 | -97.3356 | -90.1914 | -83.5713 | -68.4655 | -91.0016 | 0/30 |

The complete 30-window vectors are in:

`results/paper/rossi/tables/rossi_per_window_delta_vectors.csv`

Interpretation:

- There is no clean window where Rossi beats HPA under the cold-start online
  protocol.
- P1 `k = 10` is not concentrated in a few extreme windows. Rossi beats HPA in
  all 30 windows, and even the minimum Delta is strongly positive
  (`782.9536`).
- P2 and P3 are likewise not outlier-driven: HPA beats Rossi in all 30 windows
  for the anchor cells.

## Q5. HPA Failure-Mode Trace Under Lag

Representative window:

- Chosen as the P1 `k = 10` median-Delta window.
- Seed/window index: `9`.
- Offset: `120030`.
- Delta HPA-Rossi: `966.5838`.
- Figure:
  `results/paper/rossi/figures/rossi_p1_k10_median_window_failure_trace.pdf`

This figure has four panels:

- HPA replica count.
- HPA observed utilization versus true utilization.
- Rossi replica count.
- Rossi observed utilization versus true utilization.

Representative-window diagnostics:

| Quantity | HPA | Rossi |
|---|---:|---:|
| Total cost | 1116.6750 | 150.0912 |
| Mean abs observation delta | 0.848201 | 0.099910 |
| Replica min | 1 | 1 |
| Replica median | 3 | 4 |
| Replica max | 10 | 10 |

Additional HPA-only diagnostics on this median window:

- HPA spends 40.36% of ticks at 1 replica.
- HPA spends 14.52% of ticks at 10 replicas.
- HPA mean replica count: 4.243.
- HPA true utilization: mean 0.735, median 0.382, p95 2.135, max 2.985.
- HPA absolute observation error: mean 0.848, p95 2.044, max 2.683.
- HPA SLA violation rate: 0.2677.

Clean-versus-lag HPA trace:

- Figure:
  `results/paper/rossi/figures/rossi_hpa_clean_vs_k10_failure_trace.pdf`
- Same representative window: offset `120030`.

| HPA quantity | k = 0 | k = 10 |
|---|---:|---:|
| Total cost | 118.4040 | 1116.6750 |
| SLA violation rate | 0.000500 | 0.267683 |
| Mean abs observation delta | 0.000000 | 0.848201 |
| Replica min | 1 | 1 |
| Replica median | 4 | 3 |
| Replica mean | 3.238 | 4.243 |
| Replica max | 5 | 10 |
| Fraction at 1 replica | 0.1612 | 0.4036 |
| Fraction at 10 replicas | 0.0000 | 0.1452 |
| True utilization p95 | 0.650 | 2.135 |

Interpretation:

- HPA is not merely slightly delayed. Its own threshold feedback loop creates
  large closed-loop mismatch between stale and current utilization.
- The controller spends substantial time at both the minimum and maximum
  replica boundaries, consistent with lag-driven oscillation / over-correction.
- This explains why HPA collapses under P1 while online Rossi does not.

## Q6. Rossi Q-Table Convergence Diagnostic

Representative clean window:

- Chosen as the median-Delta clean window.
- Seed/window index: `1`.
- Offset: `28007`.
- Clean Delta HPA-Rossi: `-83.6915`.
- Rossi total cost: `193.7435`.

Formula:

For each 100-tick block,

`L1_Q_change(block) = sum_{s,a} | Q_after_block(s,a) - Q_before_block(s,a) |`

The final partial block is tick 4000 only.

Selected values:

| Block | L1 Q change |
|---|---:|
| 0-99 | 6089.1598 |
| 100-199 | 1904.4536 |
| 200-299 | 243.9071 |
| 300-399 | 28.6817 |
| 800-899 | 2129.0563 |
| 1100-1199 | 1364.3018 |
| 3300-3399 | 702.9222 |
| 3400-3499 | 815.3231 |
| 3900-3999 | 286.0460 |
| 4000-4000 | 2.6095 |

Median 100-tick block L1 change: `243.9071`.

Final action distribution:

- Over the final 100 ticks, action counts are `{no_op: 100}`.
- Shannon entropy over the final 100 selected actions: `0.0` bits.

Interpretation:

- Rossi's model-based controller is not an epsilon-exploration policy. It acts
  greedily, so "still exploring" is not the right diagnosis.
- The Q-table change rate drops sharply early, but it does not monotonically
  converge to near-zero over the whole window. Later workload changes still
  induce substantial Q-table updates.
- The final action distribution is deterministic in the representative clean
  window, but the Q-table is still adapting in earlier late-window blocks.
- Therefore clean-cell HPA dominance should be described primarily as a
  cold-start online adaptation/sample-efficiency issue, not stochastic
  exploration noise.
