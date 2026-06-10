# Decima P3 Adversarial Node Features

MLflow run: `76b9ae2e2c8a48dc8e15915443ee17aa`

FGSM epsilon: `0.2`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64802.8 |
| Decima mean JCT | 63927.2 |
| Delta mean | 875.619 |
| 95% CI low | 135.152 |
| 95% CI high | 1689.58 |
| p_less | 0.98086 |
| p_greater | 0.0191498 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
