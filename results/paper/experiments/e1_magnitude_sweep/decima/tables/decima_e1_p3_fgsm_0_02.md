# Decima P3 Adversarial Node Features

MLflow run: `8e06cadf1897473793efe32ff5de3c72`

FGSM epsilon: `0.02`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64924.6 |
| Decima mean JCT | 62762.1 |
| Delta mean | 2162.56 |
| 95% CI low | 1251.66 |
| 95% CI high | 3176.75 |
| p_less | 0.99999 |
| p_greater | 1.99998e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
