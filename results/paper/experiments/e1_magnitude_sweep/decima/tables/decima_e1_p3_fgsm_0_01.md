# Decima P3 Adversarial Node Features

MLflow run: `457270c881d54532939503e647a1ab48`

FGSM epsilon: `0.01`

Comparator: official `dynamic_partition` under `PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

| Metric | Value |
|---|---:|
| dynamic_partition mean JCT | 64840.6 |
| Decima mean JCT | 62817.1 |
| Delta mean | 2023.5 |
| 95% CI low | 1153.86 |
| 95% CI high | 3016.14 |
| p_less | 0.99997 |
| p_greater | 3.99996e-05 |
| Prediction confirmed | False |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Negative delta means the classical comparator beats Decima.
