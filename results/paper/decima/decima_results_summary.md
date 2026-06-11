# Decima Results Summary

This summary reports the Decima official-simulator evidence used by the paper.
The comparator is the official README-exposed `dynamic_partition` scheduler.
These results are not presented as Graphene evidence.

## Reproduction Gate

The official-simulator gate uses the released Decima simulator at README test
scale with 50 executors, 1 initial DAG, and 5000 streaming DAGs.

| Metric | Value |
|---|---:|
| Observed mean-JCT improvement | 3.0126% |
| Reference improvement target | 21.0000% |
| Within 15% relative target tolerance | False |

Decima improves mean JCT over `dynamic_partition`, but the observed
improvement is below the preregistered reference target. The paper reports
this distinction explicitly.

## Perturbation Outcomes

All Decima perturbation outcomes below use
`Delta = mean_JCT(dynamic_partition) - mean_JCT(Decima)`. Positive values
mean Decima has lower mean JCT.

| Prediction | Anchor | Delta | 95% CI | Decima win fraction | Outcome |
|---|---|---:|---:|---:|---|
| P1-Decima observation lag | `lambda=1.0` | 716396 | [653206, 779842] | 1.000 | Prediction not supported |
| P2-Decima workload tail | `w=0.5` | 8043.68 | [1138.49, 16524.6] | 0.533 | Prediction not supported |
| P3-Decima adversarial node features | `epsilon=0.05` | 2000.65 | [1057.36, 3019.18] | 0.767 | Prediction not supported |

The preregistered Decima degradation predictions are not supported under
the official-simulator comparator: the paired confidence interval is
positive in each anchor cell.

## P3 Diagnostic

- FGSM attack count: `162888`.
- Mean absolute node-feature delta: `0.02345217859596866`.
- Mean clean target probability: `0.34072129319738104`.
- Mean adversarial target probability: `0.27942296604391675`.

The adversarial perturbation reduced the clean target action
probability on average. The P3 result is therefore not explained by an
attack-sign error.

## Paper Figures

- `results/paper/decima/figures/decima_prediction_percent_improvement.png`
- `results/paper/decima/figures/decima_paired_delta_distributions.png`
- `results/paper/decima/figures/decima_mean_jct_by_cell.png`

## Primary Tables

- `results/paper/decima/tables/decima_prediction_outcomes.md`
- `results/paper/decima/tables/decima_prediction_summary.csv`
- `results/paper/decima/tables/decima_per_seed_deltas.csv`

## Artifact Generation

- MLflow artifact-generation run: `670ff88fe4714f568bec7b7d39fc3c59`.
