# Decima P1 Observation Lag

MLflow run: `f0bf09a22003477db5d1267eab6485a7`

Lag lambda: `0.25`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 172408 |
| Decima mean JCT | 185713 |
| Delta mean | -13305 |
| 95% CI low | -19833 |
| 95% CI high | -6168.27 |
| p_less | 0.000439996 |
| p_greater | 0.99957 |
| Prediction confirmed | True |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
