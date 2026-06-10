# Decima P3 Adversarial Node Features

MLflow run: `d63922760b20447cbf706dea0e78a922`

FGSM epsilon: `0.1`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64599.1 |
| Decima mean JCT | 62891.5 |
| Delta mean | 1707.65 |
| 95% CI low | 800.179 |
| 95% CI high | 2688.63 |
| p_less | 0.99958 |
| p_greater | 0.000429996 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
