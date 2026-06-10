# Decima P2 Tail Shift

MLflow run: `08bd14eb7ec64fc4a0f5d1745589779d`

Tail weight: `0.5`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 139774 |
| Decima mean JCT | 131730 |
| Delta mean | 8043.68 |
| 95% CI low | 1138.49 |
| 95% CI high | 16524.6 |
| p_less | 0.96712 |
| p_greater | 0.0328897 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
