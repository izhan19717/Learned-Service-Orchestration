# HPA-v2 Configuration Sensitivity for Rossi

MLflow run: `b85623d345864887b7eeac4ab2b3c14a`

Scope: post-hoc sensitivity analysis, not a replacement for Experiment B.

Grid: HPA-v2 target utilization `{40%, 50%, 60%, 70%}` crossed with scale-down stabilization `{300s, 0s}`. Sync period, tolerance, scale-up stabilization, and min/max replica bounds are held fixed.

Delta is `cost(HPA-v2) - cost(Rossi)`. Negative values favor HPA-v2; positive values favor Rossi.

## Interpretation

- P1 delta range across configs: `-34.216` to `61.688`.
- P1 remains far below the bundled-threshold collapse anchor: `True`.
- Clean HPA-v2 dominates all configs: `False`.
- P2 HPA-v2 dominates all configs: `True`.
- P3 HPA-v2 dominates all configs: `False`.

## Summary Table

| Config | Cell | Rossi cost | HPA-v2 cost | Delta | 95% CI | HPA wins |
|---|---|---:|---:|---:|---:|---:|
| target=40%, down=300s | Clean | 204.089 | 213.829 | +9.740 | [+4.170, +15.311] | 7/30 |
| target=40%, down=300s | P1 lag k=10 | 150.607 | 212.295 | +61.688 | [+56.773, +66.702] | 0/30 |
| target=40%, down=300s | P2 tail alpha=1.5 | 224.981 | 213.829 | -11.152 | [-19.304, -3.268] | 19/30 |
| target=40%, down=300s | P3 bucket-flip epsilon=0.05 | 209.284 | 213.829 | +4.545 | [-1.008, +9.843] | 10/30 |
| target=50%, down=300s | Clean | 204.089 | 172.532 | -31.558 | [-36.401, -27.074] | 30/30 |
| target=50%, down=300s | P1 lag k=10 | 150.607 | 173.634 | +23.027 | [+18.679, +27.221] | 0/30 |
| target=50%, down=300s | P2 tail alpha=1.5 | 224.981 | 172.561 | -52.420 | [-60.155, -44.940] | 30/30 |
| target=50%, down=300s | P3 bucket-flip epsilon=0.05 | 209.284 | 172.532 | -36.753 | [-41.588, -32.570] | 30/30 |
| target=60%, down=300s | Clean | 204.089 | 146.206 | -57.883 | [-62.310, -53.557] | 30/30 |
| target=60%, down=300s | P1 lag k=10 | 150.607 | 146.028 | -4.579 | [-8.721, -0.898] | 17/30 |
| target=60%, down=300s | P2 tail alpha=1.5 | 224.981 | 147.617 | -77.365 | [-85.002, -70.043] | 30/30 |
| target=60%, down=300s | P3 bucket-flip epsilon=0.05 | 209.284 | 146.206 | -63.078 | [-67.800, -58.778] | 30/30 |
| target=70%, down=300s | Clean | 204.089 | 129.827 | -74.263 | [-78.337, -70.187] | 30/30 |
| target=70%, down=300s | P1 lag k=10 | 150.607 | 135.407 | -15.201 | [-19.953, -10.479] | 27/30 |
| target=70%, down=300s | P2 tail alpha=1.5 | 224.981 | 164.326 | -60.655 | [-69.744, -51.232] | 30/30 |
| target=70%, down=300s | P3 bucket-flip epsilon=0.05 | 209.284 | 129.827 | -79.458 | [-83.819, -75.266] | 30/30 |
| target=40%, down=0s | Clean | 204.089 | 161.962 | -42.127 | [-46.380, -37.849] | 30/30 |
| target=40%, down=0s | P1 lag k=10 | 150.607 | 162.165 | +11.558 | [+7.386, +15.868] | 5/30 |
| target=40%, down=0s | P2 tail alpha=1.5 | 224.981 | 161.962 | -63.019 | [-70.169, -56.174] | 30/30 |
| target=40%, down=0s | P3 bucket-flip epsilon=0.05 | 209.284 | 161.962 | -47.322 | [-51.554, -43.385] | 30/30 |
| target=50%, down=0s | Clean | 204.089 | 132.798 | -71.291 | [-75.595, -67.264] | 30/30 |
| target=50%, down=0s | P1 lag k=10 | 150.607 | 133.234 | -17.373 | [-21.274, -13.377] | 29/30 |
| target=50%, down=0s | P2 tail alpha=1.5 | 224.981 | 133.188 | -91.793 | [-98.979, -84.430] | 30/30 |
| target=50%, down=0s | P3 bucket-flip epsilon=0.05 | 209.284 | 132.798 | -76.486 | [-80.905, -72.383] | 30/30 |
| target=60%, down=0s | Clean | 204.089 | 114.559 | -89.530 | [-93.675, -85.545] | 30/30 |
| target=60%, down=0s | P1 lag k=10 | 150.607 | 117.639 | -32.968 | [-36.680, -29.132] | 30/30 |
| target=60%, down=0s | P2 tail alpha=1.5 | 224.981 | 121.190 | -103.792 | [-111.302, -96.310] | 30/30 |
| target=60%, down=0s | P3 bucket-flip epsilon=0.05 | 209.284 | 114.559 | -94.725 | [-99.104, -90.688] | 30/30 |
| target=70%, down=0s | Clean | 204.089 | 103.608 | -100.481 | [-104.761, -96.407] | 30/30 |
| target=70%, down=0s | P1 lag k=10 | 150.607 | 116.391 | -34.216 | [-39.414, -29.051] | 29/30 |
| target=70%, down=0s | P2 tail alpha=1.5 | 224.981 | 166.968 | -58.013 | [-68.745, -47.112] | 28/30 |
| target=70%, down=0s | P3 bucket-flip epsilon=0.05 | 209.284 | 103.608 | -105.676 | [-110.658, -100.558] | 30/30 |

## Figures

- `results/paper/experiments/hpa_v2_config_sensitivity/figures/hpa_v2_config_p1_delta.pdf`
- `results/paper/experiments/hpa_v2_config_sensitivity/figures/hpa_v2_config_p1_delta.png`
- `results/paper/experiments/hpa_v2_config_sensitivity/figures/hpa_v2_config_delta_heatmap.pdf`
- `results/paper/experiments/hpa_v2_config_sensitivity/figures/hpa_v2_config_delta_heatmap.png`

## Scientific Reading

The result is a configuration sensitivity around Experiment B. Use it to assess whether the HPA-v2 finding is robust to representative HPA target and stabilization choices; it does not alter the locked Rossi reproduction or perturbation protocol.
