# Experiment B Results — HPA-v2 Comparator for Rossi

MLflow run: `9b609ba12448413aa92263b8859e5854`

HPA-v2 target utilisation is 50%, treated as a representative production-grade user-specified configuration, not a Kubernetes universal default.
P1 applies shared telemetry lag to Rossi and HPA-v2. P3 applies bucket flipping only to Rossi's discretised Q-table representation; HPA-v2 reads true continuous utilisation.
Rossi costs are loaded from the locked canonical online Rossi runs on the same 30 offsets; only the new HPA-v2 comparator is newly evaluated here.

## Sanity Checks

| Check | Passed | Key metrics |
|---|---:|---|
| S1_clean_stability | True | hpa_v2_replica_std=0.8607, bundled_threshold_replica_std=1.218, hpa_v2_replica_min=1, hpa_v2_replica_max=6 |
| S2_sync_cadence | True | max_distinct_replicas_in_15_tick_window=2 |
| S3_stabilization | True | max_replicas_during_overload=10, replicas_250_ticks_after_underload=10, final_replicas=4 |

## Results

| Cell | Rossi cost | HPA-v2 cost | Δ HPA-v2-Rossi | iid 95% CI | block L=10 95% CI | Holm p2s L=10 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Clean | 204.089 | 172.532 | -31.558 | [-36.372, -26.800] | [-35.246, -27.897] | 0.9954 | HPA-v2 dominates Rossi |
| P1 lag k=10 | 150.607 | 173.634 | 23.027 | [18.780, 27.164] | [20.408, 25.227] | 0.9954 | H_P1a |
| P2 tail alpha=1.5 | 224.981 | 172.561 | -52.420 | [-59.986, -45.033] | [-56.864, -47.710] | 0.9954 | HPA-v2 dominates Rossi |
| P3 bucket-flip epsilon=0.05 | 209.284 | 172.532 | -36.753 | [-41.711, -32.516] | [-40.037, -35.104] | 0.9954 | HPA-v2 dominates Rossi |

## Interpretation

H_P1a prevails: HPA-v2 does not reproduce the bundled-threshold lag collapse. Clean verdict: HPA-v2 dominates Rossi. P2 verdict: HPA-v2 dominates Rossi. P3 verdict: HPA-v2 dominates Rossi. The sign convention is Δ = cost(HPA-v2) - cost(Rossi); negative values favor HPA-v2 and positive values favor Rossi. The original bundled-threshold P1 anchor was approximately +965, so the HPA-v2 P1 value should be interpreted as strong evidence that the dramatic bundled-threshold lag collapse does not generalise to a stabilization-windowed comparator. The block-sign Holm values are conservative because L=10 supplies only three sign blocks; the paired iid CIs and point estimates are the more informative magnitude summaries for this follow-up sensitivity.

## Artifact Paths

- Costs CSV: `results/paper/experiments/experiment_b/data/experiment_b_costs.csv`
- Costs CSV copy: `data/experiment_b_costs.csv`
- Figure: `results/paper/experiments/experiment_b/figures/hpa_v2_clean_vs_k10_failure_trace.pdf`
- Figure: `results/paper/experiments/experiment_b/figures/hpa_v2_clean_vs_k10_failure_trace.png`
- Figure: `figures/hpa_v2_clean_vs_k10_failure_trace.pdf`
- Figure: `figures/hpa_v2_clean_vs_k10_failure_trace.png`
- Figure: `results/paper/experiments/experiment_b/figures/threshold_vs_hpa_v2_under_lag.pdf`
- Figure: `results/paper/experiments/experiment_b/figures/threshold_vs_hpa_v2_under_lag.png`
- Figure: `figures/threshold_vs_hpa_v2_under_lag.pdf`
- Figure: `figures/threshold_vs_hpa_v2_under_lag.png`
