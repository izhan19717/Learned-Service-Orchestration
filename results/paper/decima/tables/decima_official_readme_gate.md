# Decima Official README Test Gate

MLflow run: `36fb0fff65284fb2878a55755dc47b25`

Gate passed: `False`

| Scheme | Mean JCT | Total reward | Jobs | Decision steps | Runtime seconds |
|---|---:|---:|---:|---:|---:|
| dynamic_partition | 62746.5 | -3137.95 | 5001 | 90265 | 81.2065 |
| learn | 60856.2 | -3043.42 | 5001 | 142821 | 524.559 |

## Gate

| Metric | Value |
|---|---:|
| Observed mean-JCT improvement (%) | 3.01258 |
| Target improvement (%) | 21 |
| Relative error to target | 0.856544 |
| Within 15% relative target tolerance | False |
| Learn beats dynamic partition | True |

Note: visualization side effects from the stock README test are disabled by default here; the simulator loop, seeds, agents, saved model, and test scale match the README gate.
