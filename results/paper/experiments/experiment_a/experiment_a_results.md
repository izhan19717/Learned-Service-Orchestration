# Experiment A Results — Rossi Block-Bootstrap Sensitivity

This analysis uses the existing 30 ordered Rossi evaluation windows. No simulator, training, or controller run was executed.

Comparator label: `bundled_threshold` for the RLAD simulator's bundled threshold controller. Older files retain `hpa` in column names, but this report does not interpret that controller as Kubernetes HPA.

Official Experiment A p-values are two-sided block sign-flip p-values. One-sided observed-sign p-values are reported only as a compatibility column with the main-paper convention.

## Part A1 — Autocorrelation Diagnostic

Bartlett 95% large-sample band: ±0.358.

| Cell | rho1 | max abs rho(1..10) | LB(5) p | LB(10) p | Diagnostic |
|---|---:|---:|---:|---:|---|
| Clean | 0.243 | 0.319 | 0.178 | 0.335 | ok |
| P1 lag k=10 | 0.043 | 0.369 | 0.218 | 0.527 | ok |
| P2 tail alpha=1.5 | 0.077 | 0.257 | 0.683 | 0.708 | ok |
| P3 bucket-flip epsilon=0.05 | 0.094 | 0.229 | 0.905 | 0.939 | ok |

## Part A2 — Block Bootstrap and Block Sign-Flip

| Cell | Δ | iid 95% CI | MBB 95% CI L=5 | MBB 95% CI L=10 | iid p(one-sided obs.) | block p2s L=5 | block p2s L=10 | Holm p2s L=5 | Holm p2s L=10 | block p1s compat L=5 | block p1s compat L=10 | Holm p1s compat L=5 | Holm p1s compat L=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | -85.807 | [-89.901, -81.785] | [-89.765, -83.820] | [-88.024, -84.399] | 1e-05 | 0.03113 | 0.2517 | n/a | n/a | 0.01508 | 0.1268 | n/a | n/a |
| P1 lag k=10 | 965.087 | [943.088, 986.515] | [956.097, 983.640] | [961.788, 978.870] | 1e-05 | 0.03097 | 0.2488 | 0.1239 | 0.7465 | 0.01518 | 0.1239 | 0.06072 | 0.3718 |
| P2 tail alpha=1.5 | -105.979 | [-113.134, -98.672] | [-111.783, -101.755] | [-110.015, -102.968] | 1e-05 | 0.03188 | 0.2508 | 0.1239 | 0.7465 | 0.01591 | 0.126 | 0.06072 | 0.3718 |
| P3 bucket-flip epsilon=0.05 | -91.002 | [-95.409, -86.822] | [-96.249, -88.900] | [-95.249, -89.983] | 1e-05 | 0.03103 | 0.252 | 0.1239 | 0.7465 | 0.01521 | 0.1261 | 0.06072 | 0.3718 |

## Familywise Interpretation

All four Rossi cells satisfy the pre-registered lag-1 autocorrelation threshold |rho1| < 0.3. No Rossi block-bootstrap CI changes zero-containment status relative to the iid CI. Some MBB intervals differ from iid width by more than 20%: Clean L=5 width ratio 0.73; Clean L=10 width ratio 0.45; P1 lag k=10 L=5 width ratio 0.63; P1 lag k=10 L=10 width ratio 0.39; P2 tail alpha=1.5 L=5 width ratio 0.69; P2 tail alpha=1.5 L=10 width ratio 0.49; P3 bucket-flip epsilon=0.05 L=10 width ratio 0.61. Under the official two-sided block sign-flip plus Holm convention, Rossi familywise rejections do not survive for: P1-Rossi L=5, P2-Rossi L=5, P3-Rossi L=5, P1-Rossi L=10, P2-Rossi L=10, P3-Rossi L=10. This is driven by the discreteness of block sign-flip tests with only 6 sign blocks at L=5 and 3 at L=10; it should be reported as a conservative sensitivity result, not as evidence that the point estimates are small. Even under the one-sided observed-sign compatibility convention, the Rossi block-sign familywise values do not pass Holm for: P1-Rossi L=5, P2-Rossi L=5, P3-Rossi L=5, P1-Rossi L=10, P2-Rossi L=10, P3-Rossi L=10. As expected from the p-value convention decision, Decima P2 also does not survive the official two-sided Holm convention at L=5/10; the main-paper row should be footnoted as marginal under two-sided sensitivity.

## Artifact Paths

- ACF CSV: `results/paper/experiments/experiment_a/tables/rossi_acf_diagnostic.csv`
- Block-bootstrap CSV: `results/paper/experiments/experiment_a/tables/rossi_block_bootstrap_results.csv`
- Figure: `results/paper/experiments/experiment_a/figures/rossi_acf_diagnostic.pdf`
- Figure: `results/paper/experiments/experiment_a/figures/rossi_acf_diagnostic.png`
- Figure: `figures/rossi_acf_diagnostic.pdf`
- Figure: `figures/rossi_acf_diagnostic.png`
- Replicates: `results/paper/experiments/experiment_a/data/experiment_a_replicates_L5.npz`
- Replicates: `data/experiment_a_replicates_L5.npz`
- Replicates: `results/paper/experiments/experiment_a/data/experiment_a_replicates_L10.npz`
- Replicates: `data/experiment_a_replicates_L10.npz`
