# Decima P2 Tail Shift

MLflow run: `24aec21c78864f0eb1ff0cd59d74cbba`

Tail weight: `0.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 59029.4 |
| Decima mean JCT | 57657.4 |
| Delta mean | 1371.97 |
| 95% CI low | 620.157 |
| 95% CI high | 2163.63 |
| p_less | 0.99914 |
| p_greater | 0.000869991 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
