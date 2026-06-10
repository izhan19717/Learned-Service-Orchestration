# Decima Prediction Outcomes

All Decima P1/P2/P3 perturbation cells are complete under
`PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

These are official-simulator results against the README-exposed
`dynamic_partition` comparator. They must not be described as Graphene
or full Spark-testbed headline results.

| Prediction | Anchor | Status | Delta | 95% CI | Reason |
|---|---|---|---:|---:|---|
| P1-Decima observation lag | `lambda=1.0` | Falsified under simulator-gate amendment | 716396 | [653206, 779842] | Decima remains lower-JCT than dynamic_partition in aggregate for this perturbation cell. |
| P2-Decima workload tail | `w=0.5` | Falsified under simulator-gate amendment | 8043.68 | [1138.49, 16524.6] | Decima remains lower-JCT than dynamic_partition in aggregate for this perturbation cell. |
| P3-Decima adversarial node features | `epsilon=0.05` | Falsified under simulator-gate amendment | 2000.65 | [1057.36, 3019.18] | Decima remains lower-JCT than dynamic_partition in aggregate for this perturbation cell. |

Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. A
positive delta means Decima has lower mean JCT than the comparator.
