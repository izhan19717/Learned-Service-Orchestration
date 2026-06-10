# Decima P3 Adversarial Node Features

MLflow run: `e680949abb04454a81e5034d04cb48d7`

FGSM epsilon: `0.05`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64819 |
| Decima mean JCT | 62818.3 |
| Delta mean | 2000.65 |
| 95% CI low | 1057.36 |
| 95% CI high | 3019.18 |
| p_less | 0.99992 |
| p_greater | 8.99991e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
