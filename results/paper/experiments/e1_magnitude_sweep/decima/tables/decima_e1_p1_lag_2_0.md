# Decima P1 Observation Lag

MLflow run: `d4a70ce1b32f4d34ac01867499cffe52`

Lag lambda: `2.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 5.36971e+06 |
| Decima mean JCT | 3.9249e+06 |
| Delta mean | 1.44481e+06 |
| 95% CI low | 1.34377e+06 |
| 95% CI high | 1.54701e+06 |
| p_less | 1 |
| p_greater | 9.9999e-06 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
