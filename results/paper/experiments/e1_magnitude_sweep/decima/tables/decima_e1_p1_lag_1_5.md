# Decima P1 Observation Lag

MLflow run: `e8ec4471c4a1487aadc248b6409f6c94`

Lag lambda: `1.5`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 3.74214e+06 |
| Decima mean JCT | 2.6165e+06 |
| Delta mean | 1.12564e+06 |
| 95% CI low | 1.04597e+06 |
| 95% CI high | 1.20836e+06 |
| p_less | 1 |
| p_greater | 9.9999e-06 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
