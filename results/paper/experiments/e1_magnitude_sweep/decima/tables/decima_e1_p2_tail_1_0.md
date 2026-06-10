# Decima P2 Tail Shift

MLflow run: `f165ea431cf44c25a8b0b9f10d877c74`

Tail weight: `1.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 281470 |
| Decima mean JCT | 240678 |
| Delta mean | 40791.8 |
| 95% CI low | 19899.6 |
| 95% CI high | 64994.6 |
| p_less | 0.99993 |
| p_greater | 7.99992e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
