# Decima P1 Observation Lag

MLflow run: `ac588551230b45febe599327292b868c`

Lag lambda: `0.5`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 577657 |
| Decima mean JCT | 470148 |
| Delta mean | 107509 |
| 95% CI low | 63851.5 |
| 95% CI high | 153411 |
| p_less | 0.99999 |
| p_greater | 1.99998e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
