# Decima P2 Tail Shift

MLflow run: `a8369b181c2943ea9f5b43665c52f892`

Tail weight: `0.5`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 140123 |
| Decima mean JCT | 131286 |
| Delta mean | 8836.86 |
| 95% CI low | 1329.86 |
| 95% CI high | 18016.6 |
| p_less | 0.96822 |
| p_greater | 0.0317897 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
