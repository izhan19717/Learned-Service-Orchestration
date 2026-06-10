# Decima P3 Adversarial Node Features

MLflow run: `2191dd694efb404baab9d54ab210289a`

FGSM epsilon: `0.05`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64717.1 |
| Decima mean JCT | 62813.5 |
| Delta mean | 1903.64 |
| 95% CI low | 1095.63 |
| 95% CI high | 2796.68 |
| p_less | 0.99998 |
| p_greater | 2.99997e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
