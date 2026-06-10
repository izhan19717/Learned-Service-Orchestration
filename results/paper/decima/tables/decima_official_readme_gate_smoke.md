# Decima Official README Test Gate

MLflow run: `ab058fdb43954736b20ed6ba31e4178a`

Gate passed: `False`

| Scheme | Mean JCT | Total reward | Jobs | Decision steps | Runtime seconds |
|---|---:|---:|---:|---:|---:|
| dynamic_partition | 31366.1 | -6.58689 | 21 | 368 | 0.258987 |
| learn | 31499.7 | -6.61493 | 21 | 495 | 1.6882 |

## Gate

| Metric | Value |
|---|---:|
| Observed mean-JCT improvement (%) | -0.425694 |
| Target improvement (%) | 21 |
| Relative error to target | 1.02027 |
| Within 15% relative target tolerance | False |
| Learn beats dynamic partition | False |

Note: visualization side effects from the stock README test are disabled by default here; the simulator loop, seeds, agents, saved model, and test scale match the README gate.
