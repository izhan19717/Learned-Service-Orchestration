# Decima P3 Adversarial Node Features

MLflow run: `d5ee9e57a0dc4715adcb33f4c173f23f`

FGSM epsilon: `0.05`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 31492.3 |
| Decima mean JCT | 35953.1 |
| Delta mean | -4460.82 |
| 95% CI low | -4460.82 |
| 95% CI high | -4460.82 |
| p_less | 0.499755 |
| p_greater | 1 |
| Prediction confirmed | True |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
