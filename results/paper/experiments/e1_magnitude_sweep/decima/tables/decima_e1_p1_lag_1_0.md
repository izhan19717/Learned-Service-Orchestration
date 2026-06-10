# Decima P1 Observation Lag

MLflow run: `6dd77c4ee20441b7a411a2a878528e56`

Lag lambda: `1.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 2.12133e+06 |
| Decima mean JCT | 1.40604e+06 |
| Delta mean | 715287 |
| 95% CI low | 652674 |
| 95% CI high | 778857 |
| p_less | 1 |
| p_greater | 9.9999e-06 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
