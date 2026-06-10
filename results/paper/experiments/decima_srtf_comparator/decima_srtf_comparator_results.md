# Decima SRTF-Style Comparator Sensitivity

MLflow run: `fc912bd6b86847c888e05aed2d0b5383`

Scope: post-hoc sensitivity analysis. This is not Graphene evidence and does not replace the amended official `dynamic_partition` Decima results.

Comparator: dependency-aware SRTF-style shortest remaining work. It chooses the arrived job DAG with the smallest estimated remaining work, then the ready node with the smallest estimated remaining node work, and allocates currently available source executors work-conservatively.

Delta is `mean_JCT(SRTF) - mean_JCT(Decima)`. Negative values favor SRTF; positive values favor Decima.

## Results

| Cell | SRTF mean JCT | Decima mean JCT | Delta | 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| Clean | 207880 | 62917.4 | +144963 | [+117354, +172083] | Decima beats SRTF |
| P1 lag lambda=1.0 | 3.17771e+06 | 1.4042e+06 | +1.77351e+06 | [+1.73134e+06, +1.81583e+06] | Decima beats SRTF |
| P2 tail w=0.5 | 540361 | 131687 | +408674 | [+368358, +447917] | Decima beats SRTF |
| P3 FGSM epsilon=0.05 | 208335 | 62836.1 | +145498 | [+116860, +174462] | Decima beats SRTF |

## Figures

- `results/paper/experiments/decima_srtf_comparator/figures/decima_srtf_delta_by_cell.pdf`
- `results/paper/experiments/decima_srtf_comparator/figures/decima_srtf_delta_by_cell.png`

## Scientific Reading

This analysis answers whether the Decima conclusions are sensitive to replacing the README-exposed `dynamic_partition` comparator with a simple shortest-remaining-work classical scheduler. Because SRTF is reconstructed locally and is not an official Decima README comparator, it should be reported only as an alternative-comparator sensitivity.
