# Decima P1 Observation Lag

MLflow run: `1cafd93f9d784d43aad69c0ba11ea613`

Lag lambda: `1.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 164101 |
| Decima mean JCT | 185351 |
| Delta mean | -21250.3 |
| 95% CI low | -21250.3 |
| 95% CI high | -21250.3 |
| p_less | 0.499755 |
| p_greater | 1 |
| Prediction confirmed | True |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
