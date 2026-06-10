# E2 Companion Analysis A

MLflow run: `3c972e2a15f94c84924c40f9c8afc55e`

This is deterministic rescoring of fixed Rossi/threshold rollouts; no policy training is performed.

Primary paper-facing churn variant: `adaptation_nonnoop`.

| Cell | w_churn | Rossi score | Threshold score | Delta threshold-Rossi | 95% CI | Holm p(Delta<0) | Holm p(Delta>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.01 | 217.617 | 118.578 | -99.0393 | [-103.513, -94.6321] | 4.99995e-05 | 1 |
| clean | 0.05 | 267.533 | 114.982 | -152.551 | [-160.454, -144.935] | 4.99995e-05 | 1 |
| clean | 0.10 | 329.928 | 110.486 | -219.442 | [-232.277, -205.959] | 4.99995e-05 | 1 |
| clean | 0.20 | 454.717 | 101.495 | -353.222 | [-377.947, -328.44] | 4.99995e-05 | 1 |
| clean | 0.30 | 579.507 | 92.5041 | -487.003 | [-523.77, -453.256] | 4.99995e-05 | 1 |
| p1_lag_k10 | 0.01 | 156.224 | 1134.67 | 978.445 | [955.551, 1000.32] | 1 | 4.99995e-05 |
| p1_lag_k10 | 0.05 | 177.57 | 1165.49 | 987.918 | [965.398, 1009.56] | 1 | 4.99995e-05 |
| p1_lag_k10 | 0.10 | 204.252 | 1204.01 | 999.759 | [977.468, 1021.39] | 1 | 4.99995e-05 |
| p1_lag_k10 | 0.20 | 257.616 | 1281.06 | 1023.44 | [999.621, 1048.67] | 1 | 4.99995e-05 |
| p1_lag_k10 | 0.30 | 310.981 | 1358.1 | 1047.12 | [1020.77, 1074.65] | 1 | 4.99995e-05 |

Delta is `score(threshold) - score(Rossi)`. Positive values mean Rossi has the lower rescored scalar objective.

- Summary CSV: `results/paper/experiments/e2_objective_native/tables/e2_companion_a_weight_rescore.csv`
- Raw component CSV: `results/paper/experiments/e2_objective_native/data/e2_companion_a_components.csv`
