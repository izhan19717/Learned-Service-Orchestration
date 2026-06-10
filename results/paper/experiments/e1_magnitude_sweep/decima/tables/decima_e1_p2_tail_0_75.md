# Decima P2 Tail Shift

MLflow run: `bbb5c4b4fb8e440599cd9a8171eb444c`

Tail weight: `0.75`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 205044 |
| Decima mean JCT | 185147 |
| Delta mean | 19896.8 |
| 95% CI low | 7332.43 |
| 95% CI high | 35099.4 |
| p_less | 0.99918 |
| p_greater | 0.000829992 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
