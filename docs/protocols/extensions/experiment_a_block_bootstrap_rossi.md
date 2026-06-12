# Experiment A — Block-Bootstrap Sensitivity Analysis for the Rossi Cells

**Status: pre-registered. Specification frozen before any analysis is executed.**

## 0. Purpose

The Rossi paired statistics are computed over 30 non-overlapping windows drawn
from a single continuous workload trajectory. The iid bootstrap and paired
sign-flip tests used in the main analysis assume that these window-level paired
differences are approximately independent. If the per-window paired difference
series `Δᵢ = cost_comparator(i) − cost_Rossi(i)` exhibits temporal
autocorrelation, iid procedures can understate variance and overstate
significance. This experiment supplies a dependence-robust sensitivity check.

The experiment has two parts. Part A1 is a diagnostic: report the autocorrelation function of `Δᵢ` for each Rossi cell so that the reader can judge the magnitude of any dependence. Part A2 is the corrective analysis: rerun the paired CIs and p-values using a moving-block bootstrap and a block sign-flip test at two block lengths, and compare to the iid results reported in the main paper.

No new simulator runs are required. The experiment consumes the per-window cost series already produced for the Rossi P1, P2, P3, and clean cells.

## 1. Input data

The Rossi evaluation pipeline provides the following arrays (one entry per window, indexed `i = 1, …, 30` in trajectory order):

| Cell | Array name (proposed) | Description |
|------|----------------------|-------------|
| Clean | `cost_clean_rossi_online[i]`, `cost_clean_threshold[i]` | Total cost per window, online-adaptive Rossi controller vs bundled threshold, under no perturbation. |
| P1 lag (`k=10` s) | `cost_p1_rossi_online[i]`, `cost_p1_threshold[i]` | Same, with 10-second observation lag applied to both controllers. |
| P2 tail (Pareto `α=1.5`) | `cost_p2_rossi_online[i]`, `cost_p2_threshold[i]` | Same, with service-time distribution shifted to Pareto α=1.5 (capped second moment as in the main run). |
| P3 bucket-flip (ε=0.05) | `cost_p3_rossi_online[i]`, `cost_p3_threshold[i]` | Same, with minimum bucket-flip on the discretised utilisation observation. |

For each cell construct the paired difference `Δᵢ = cost_threshold[i] − cost_rossi_online[i]`. The published Δ values at the anchor are: clean −85.8, P1 +965.1, P2 −106.0, P3 −91.0.

## 2. Part A1 — Autocorrelation diagnostic

For each cell:

1. Compute the sample autocorrelation `ρ̂(h)` of the `Δᵢ` series at lags `h ∈ {1, 2, …, 10}`. Use the standard biased estimator `ρ̂(h) = Σᵢ (Δᵢ − Δ̄)(Δᵢ₊ₕ − Δ̄) / Σᵢ (Δᵢ − Δ̄)²`.
2. Compute Bartlett 95% large-sample bands `±1.96/√N` with `N = 30`. (These will be approximately `±0.358`, which is wide because of the small sample; report them as such.)
3. Compute the Ljung-Box statistic at lag 5 and at lag 10 with the corresponding p-values from the chi-squared distribution.

**Output (Part A1).** A 4×11 table with rows for {clean, P1, P2, P3} and columns for `ρ̂(1), …, ρ̂(10), LB(5) p, LB(10) p`. Render also as a single 4-panel figure (one panel per cell) showing the ACF bar chart with the Bartlett bands as horizontal dashed lines. Save the figure as `figures/rossi_acf_diagnostic.pdf`.

**Pre-registered expectation for Part A1.** Given that windows are non-overlapping 4001-tick contiguous slices and the per-window statistic is an integrated quantity over the entire window, the autocorrelation should be small to moderate. We pre-register the expectation `|ρ̂(1)| < 0.3` for all four cells; values larger than 0.3 at lag 1 or any lag whose Ljung-Box p < 0.05 would indicate that the iid bootstrap is not defensible for that cell.

## 3. Part A2 — Moving-block bootstrap and block sign-flip test

For each cell, repeat the inference procedure at two block lengths `L ∈ {5, 10}`.

### 3.1 Moving-block bootstrap (Künsch 1989)

1. From the `Δᵢ` series of length `N = 30`, form the set of overlapping blocks `Bⱼ = (Δⱼ, Δⱼ₊₁, …, Δⱼ₊ₗ₋₁)` for `j = 1, …, N − L + 1`.
2. To form one bootstrap replicate of length `N`: sample `⌈N/L⌉` blocks independently and uniformly with replacement from `{B₁, …, B_{N−L+1}}`, concatenate, and truncate to length `N`.
3. Compute the replicate mean `Δ̄*`.
4. Repeat for `B = 5000` replicates.
5. The 95% block-bootstrap CI is the [2.5%, 97.5%] empirical quantile of `{Δ̄*₁, …, Δ̄*ᵦ}`.

### 3.2 Block sign-flip test

The iid sign-flip test in the main paper randomly negates individual `Δᵢ`. The block analogue negates entire blocks.

1. Divide `Δ` into `⌈N/L⌉` non-overlapping consecutive blocks (the last block is shorter if `N` is not divisible by `L`).
2. For each of `M = 10⁵` permutations, independently flip the sign of each block with probability 0.5 (a fair-coin flip per block); compute `Δ̄_perm`.
3. Official two-sided p-value: `p = (1 + |{m : |Δ̄_perm,m| ≥ |Δ̄_obs|}|) / (1 + M)` (the +1 in numerator and denominator is the standard Bayesian-conservative correction for finite Monte Carlo).
4. Compatibility one-sided p-value: report the one-sided p-value in the observed sign direction, `p = P(Δ̄_perm ≥ Δ̄_obs)` when `Δ̄_obs > 0` and `p = P(Δ̄_perm ≤ Δ̄_obs)` when `Δ̄_obs < 0`, using the same +1 finite-Monte-Carlo correction.

The two-sided p-value is the official Experiment A result. The one-sided
observed-direction p-value is reported only as a compatibility column with the
main paper's existing directional p-value convention.

### 3.3 Holm-Bonferroni correction

Apply Holm-Bonferroni across the family of all nine main-paper predictions, replacing the unadjusted p-values for the three Rossi perturbed cells (P1, P2, P3) with their block sign-flip p-values at each block length. The official Experiment A Holm correction uses the two-sided p-values. Report, as a compatibility sensitivity, the Holm-adjusted values that would result from the one-sided observed-direction p-values. The non-Rossi cells keep their original main-paper p-values, with the convention used for each column stated explicitly.

## 4. Recorded statistics (Part A2)

The Part A2 table records one row for each Rossi cell: clean, P1 lag `k=10`,
P2 tail `α=1.5`, and P3 bucket-flip `ε=0.05`. The recorded fields are:

- anchor paired delta `Δ`;
- iid 95% paired-bootstrap CI;
- moving-block-bootstrap 95% CI at `L=5` and `L=10`;
- iid sign-flip p-value;
- block sign-flip two-sided p-value at `L=5` and `L=10`;
- Holm-adjusted two-sided p-value at `L=5` and `L=10` for perturbed cells;
- one-sided observed-direction compatibility p-value at `L=5` and `L=10`.

Save the raw arrays of bootstrap replicates and sign-flip permutations to `data/experiment_a_replicates_L5.npz` and `data/experiment_a_replicates_L10.npz` so the analysis is independently reproducible.

## 5. Interpretation rules (pre-registered)

The block bootstrap is reported as a sensitivity analysis around the main results, not as a replacement. The interpretation rules are:

1. **If `ρ̂(1)` is below 0.3 for all four cells AND the block-bootstrap CIs are within 20% of the iid bootstrap CIs at both block lengths AND no cell's family-wise corrected verdict changes:** interpret the iid analysis as stable under the dependence-robust sensitivity check.

2. **If any cell's block-bootstrap CI changes its zero-containment status AND/OR any cell's family-wise corrected verdict changes:** report that cell as sensitive to dependence correction. The most likely affected cell is Rossi P2 because its iid CI is the closest to zero among the Rossi cells; for P1 the effect size of +965.1 is order-of-magnitude larger than the iid CI width, so the verdict almost certainly survives even with substantial autocorrelation inflation.

3. **If `LB(10)` p-value is below 0.01 in any cell:** treat temporal dependence as material for that cell and include the per-cell autocorrelation diagnostic alongside its block-bootstrap result.

## 6. References

- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." *Annals of Statistics* 17(3): 1217–1241.
- Politis, D. N., & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." *Econometric Reviews* 23(1): 53–70.
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics* 6(2): 65–70.
