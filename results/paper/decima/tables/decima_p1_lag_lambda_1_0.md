# Decima P1 Observation Lag

MLflow run: `bd9a8a337b9f4c25a6ec3489beaa8cca`

Lag lambda: `1.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 2.12046e+06 |
| Decima mean JCT | 1.40407e+06 |
| Delta mean | 716396 |
| 95% CI low | 653206 |
| 95% CI high | 779842 |
| p_less | 1 |
| p_greater | 9.9999e-06 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
