# Decima P3 Adversarial Node Features

MLflow run: `0d80f558f13b4e209f99403315a8bf9f`

FGSM epsilon: `0.0`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64789.2 |
| Decima mean JCT | 62698.2 |
| Delta mean | 2091.01 |
| 95% CI low | 1252.35 |
| 95% CI high | 3001.16 |
| p_less | 0.99999 |
| p_greater | 1.99998e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
