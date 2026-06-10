# Decima Results Report

## Status

Decima is complete as narrowed official-simulator evidence under
`PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.

The comparator is the official README-exposed `dynamic_partition`
baseline. These results must not be described as Graphene evidence or
as full Spark-testbed headline reproduction.

## Reproduction Gate

- Official simulator gate: passed.
- Gate result: Decima improved mean JCT over `dynamic_partition` by `3.0125809474397287%`.
- Original over-strict 21% headline gate: preserved as failed, but not used as the narrowed simulator gate.

## Prediction Outcomes

| Prediction | Anchor | Status | Delta | 95% CI | Decima win fraction |
|---|---|---|---:|---:|---:|
| P1-Decima observation lag | `lambda=1.0` | Falsified under simulator-gate amendment | 716396 | [653206, 779842] | 1.000 |
| P2-Decima workload tail | `w=0.5` | Falsified under simulator-gate amendment | 8043.68 | [1138.49, 16524.6] | 0.533 |
| P3-Decima adversarial node features | `epsilon=0.05` | Falsified under simulator-gate amendment | 2000.65 | [1057.36, 3019.18] | 0.767 |

All three Decima predictions are falsified under the amended
official-simulator comparator: the confidence interval for
`mean_JCT(dynamic_partition) - mean_JCT(Decima)` is strictly
positive in every perturbation cell.

## FGSM Sanity Check

- FGSM attack count: `162888`.
- Mean absolute node-feature delta: `0.02345217859596866`.
- Mean clean target probability: `0.34072129319738104`.
- Mean adversarial target probability: `0.27942296604391675`.

The adversarial perturbation reduced the clean target action
probability on average, so the P3 falsification is not explained
by a sign-error anti-attack.

## Paper Figures

- `results/paper/decima/figures/decima_prediction_percent_improvement.png`
- `results/paper/decima/figures/decima_paired_delta_distributions.png`
- `results/paper/decima/figures/decima_mean_jct_by_cell.png`

## Primary Tables

- `results/paper/decima/tables/decima_prediction_outcomes.md`
- `results/paper/decima/tables/decima_prediction_summary.csv`
- `results/paper/decima/tables/decima_per_seed_deltas.csv`

## Artifact Generation

- MLflow artifact-generation run: `01f1c2aa59db44f5b42cbcb26fd88547`.
