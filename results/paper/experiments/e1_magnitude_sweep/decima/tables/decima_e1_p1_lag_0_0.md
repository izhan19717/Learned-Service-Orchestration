# Decima P1 Observation Lag

MLflow run: `bdcda628d3c44da38e54f9595eea131e`

Lag lambda: `0.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64790.9 |
| Decima mean JCT | 62676 |
| Delta mean | 2114.85 |
| 95% CI low | 1197.31 |
| 95% CI high | 3124.29 |
| p_less | 0.99998 |
| p_greater | 2.99997e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
