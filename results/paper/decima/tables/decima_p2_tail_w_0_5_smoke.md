# Decima P2 Tail Shift

MLflow run: `cf4abd611ede42a381162ff560a6304a`

Tail weight: `0.5`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 93828.1 |
| Decima mean JCT | 98449.3 |
| Delta mean | -4621.21 |
| 95% CI low | -8780.52 |
| 95% CI high | -461.905 |
| p_less | 0.247318 |
| p_greater | 1 |
| Prediction confirmed | True |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
