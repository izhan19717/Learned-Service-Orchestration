# Decima P2 Tail Shift

MLflow run: `514d00f96cc2447d96ce40d1e5d132db`

Tail weight: `0.25`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 88260 |
| Decima mean JCT | 86960.5 |
| Delta mean | 1299.51 |
| 95% CI low | -326.289 |
| 95% CI high | 3271.32 |
| p_less | 0.909721 |
| p_greater | 0.0902891 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
