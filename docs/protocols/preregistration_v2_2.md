# Pre-Registration Document (v2.2)

**Project**: Learned service orchestration empirical study.
**Authors**: Khilji, Furutanpey, Dustdar.

This document records the predictions we test **before** any experiment runs. If a prediction is falsified, we report that honestly in the paper.

**Scope of v2.2**: nine predictions (three methods × three perturbations). The structure is mirrored across methods to enable cross-method pattern claims.

---

## 1. The position the paper argues

Standard RL methods proposed for service orchestration achieve their published advantage over classical baselines under simulator conditions that do not match the conditions of real production orchestration environments. The advantage shrinks, disappears, or reverses under:

- **Realistic observation lag** (calibrated against Verma et al. 2015 EuroSys, P95 Borglet polling interval below 10 s),
- **Realistic workload distributions** (calibrated against Reiss et al. 2012 SoCC, heavy-tailed job durations spanning more than five orders of magnitude), and
- **Small adversarial perturbation of the observation channel** (calibrated against Huang et al. 2017 ICLR adversarial-RL standard ε budgets).

The first two are realism checks. The third is genuinely adversarial.

The empirical section tests this position across **three methods spanning three task classes**: DeepRM (scheduling), Decima (DAG scheduling), and Rossi 2019 (autoscaling). The structural pattern claimed by the position is that the failure mode is not method-specific or task-specific; it is general to the published line of RL-for-orchestration work. Three methods in three task classes is the minimum scope that can substantiate that structural claim empirically.

---

## 2. Methods under evaluation

Three canonical, widely-cited published RL-for-orchestration methods, each from a distinct task class. All three are reimplemented in modern frameworks. Faithful reproduction of each method's original headline result within 15% is a mandatory gate before any perturbation experiment runs on that method.

1. **DeepRM** (Mao, Alizadeh, Menache, Kandula, 2016, *Resource Management with Deep Reinforcement Learning*, HotNets-XV). Most-cited RL-for-orchestration paper. Task: multi-resource job scheduling. Reimplemented in PyTorch. Reproduction target: Mao 2016 Figure 4 (DeepRM beats SJF, Packer, Tetris* on slowdown).
2. **Decima** (Mao, Schwarzkopf, Venkatakrishnan, Meng, Alizadeh, 2019, *Learning Scheduling Algorithms for Data Processing Clusters*, SIGCOMM '19). Task: DAG-aware data-processing cluster scheduling. GNN-based policy. Reimplemented in PyTorch from the official `hongzimao/decima-sim` simulator. Reproduction gate follows the official simulator command structure first, then checks the paper's 21% mean-JCT improvement claim against the strongest available classical comparator.
3. **Rossi 2019** (Rossi, Nardelli, Cardellini, 2019, *Horizontal and vertical scaling of container-based applications using reinforcement learning*, IEEE CLOUD '19). Task: container autoscaling. Tabular value-function autoscaling using the official RLAD Java simulator as operational ground truth. The primary Rossi method is the paper's model-based RL controller over discretised `(replica_count, utilization_bucket, cpu_allocation_bucket)` state. The Java repository's current DynaQ2 default is treated as a simulator-default sensitivity, not the paper-primary method. Reproduction target: Rossi 2019 simulation Table I, performance-weighted 5-action Model-based row, within 15% on the reported metrics.

Classical baselines per method:
- DeepRM: SJF, Packer, Tetris* (Tetris* is the strongest, primary comparator).
- Decima: SJF, FIFO, Graphene (Graphene is the strongest, primary comparator). Because the inspected official Decima simulator does not ship executable single-resource Graphene code and its multi-resource Graphene imports are incomplete, Decima perturbation cells require a separate Graphene validation gate. Any local dependency-aware scaffold is labelled Graphene-style and is not paper-valid Graphene unless that gate passes or this document is amended before Decima results exist.
- Rossi: HPA-style threshold controller only for the perturbation-comparison cells. No VPA comparator is used in the primary Rossi cells. HPA is not the Rossi-paper reproduction gate because Rossi 2019 reports RL-vs-RL and static-deployment comparisons rather than an HPA-improvement table.

Scope limitation acknowledged in the paper: DeepRM and Decima are both from the Mao/Alizadeh group at MIT. Rossi 2019 is from a different group (Tor Vergata Rome), which partially addresses the single-lineage concern but does not eliminate it. A broader survey of methods is future work.

---

## 3. Statistical framework, applied identically to all predictions

- **30 independent random seeds per cell**, each generating one evaluation trace.
- **Per-trace evaluation length**: method-specific. DeepRM: 200 jobs. Decima: official simulator reference test length for reproduction and ~200 DAGs for stress-test cells unless the Decima brief specifies otherwise. Rossi: one episode using the official RLAD profile semantics.
- **Pairing**: the same evaluation traces are presented to the RL method and the classical baseline within a seed. The per-seed statistic compares the two on the same trace.
- **Bootstrap**: 5000-resample percentile bootstrap on the 30 per-seed statistics, 95% CI.
- **P-values**: paired sign-flip randomization test on the 30 per-seed statistics, using 100000 Monte Carlo sign flips and directional alternatives specified in §4.
- **Multiple-comparisons correction**: Holm-Bonferroni at family-wise α = 0.05 across **all nine predictions** (P1, P2, P3 for each of DeepRM, Decima, Rossi).
- **Per-seed statistic**: `Δ = mean_metric(classical) − mean_metric(RL)`. Method-specific metric: slowdown for DeepRM, mean JCT for Decima, total cost+SLA-penalty for Rossi. Δ positive means the RL method wins. Δ negative means the classical baseline wins.

---

## 4. Nine pre-registered predictions

Structure: three perturbations × three methods = nine predictions. Each prediction uses the same confirmation, falsification, and inconclusive criteria. Confirmation is uniform across predictions: the 95% paired bootstrap CI on the per-seed statistic Δ lies entirely below zero (the classical baseline significantly beats the RL method at the anchor magnitude).

Notation: `Δ_M(perturbation, magnitude)` is the per-seed statistic for method M at the given perturbation and magnitude. `H1` denotes the classical baseline of M (Tetris* for DeepRM, Graphene for Decima, HPA for Rossi).

---

### P1 family — Observation lag (realism check)

**P1-DeepRM**: at `k = 10` decision steps (Verma 2015 P95 anchor in DeepRM time units), the 95% CI on `Δ_DeepRM(lag, 10)` lies entirely below zero.

**P1-Decima**: at `λ = 1.0` (lag multiplier on stage duration, calibrated against Verma 2015 P95), the 95% CI on `Δ_Decima(lag, 1.0)` lies entirely below zero.

**P1-Rossi**: at `k = 10` seconds (Verma 2015 P95 anchor at face value), the 95% CI on `Δ_Rossi(lag, 10s)` lies entirely below zero.

For all three: **Confirmation** = CI entirely below zero. **Falsification** = CI entirely above zero. **Inconclusive** = CI straddles zero.

---

### P2 family — Workload tail shift (realism check)

**P2-DeepRM**: at Pareto `α = 1.5` (Reiss 2012 heavy-tail anchor), the 95% CI on `Δ_DeepRM(tail, 1.5)` lies entirely below zero.

**P2-Decima**: at Alibaba-bias weight `w = 0.5`, the 95% CI on `Δ_Decima(tail, 0.5)` lies entirely below zero.

**P2-Rossi**: at Pareto service-time `α = 1.5` (primary sub-dimension; matches DeepRM and Decima anchors), the 95% CI on `Δ_Rossi(tail, 1.5)` lies entirely below zero. The burstiness sub-dimension (`β = 0.7` Hawkes) is reported as supplementary, not as the primary P2 test.

For all three: **Confirmation** = CI entirely below zero. **Falsification** = CI entirely above zero. **Inconclusive** = CI straddles zero.

---

### P3 family — Adversarial observation perturbation

**P3-DeepRM**: at FGSM budget `ε = 0.05` (Huang 2017 anchor), the 95% CI on `Δ_DeepRM(adv, 0.05)` lies entirely below zero. The adversarial perturbation is applied only to DeepRM's neural observation. Tetris* is evaluated on the true structured state.

**P3-Decima**: at FGSM budget `ε = 0.05` on node feature vectors, the 95% CI on `Δ_Decima(adv, 0.05)` lies entirely below zero. The adversarial perturbation is applied only to Decima's GNN node-feature observation. Graphene is evaluated on the true DAG state.

**P3-Rossi**: at minimum bucket-flip perturbation budget `ε = 0.05` on the continuous pre-discretisation utilization observation, the 95% CI on `Δ_Rossi(adv, 0.05)` lies entirely below zero. The bucket-flip construction is the discrete-state analogue of FGSM, documented in [calibration_v2_2.md](calibration_v2_2.md) §4.3 and summarized in [implementation_notes.md](../implementation_notes.md). The perturbation is applied only to Rossi/RLAD's observation. HPA is evaluated on the true utilization state.

For all three: **Confirmation** = CI entirely below zero. **Falsification** = CI entirely above zero. **Inconclusive** = CI straddles zero.

**Note on Rossi's P3 construction**: the bucket-flip adversarial is conceptually analogous to FGSM but operationally different (no policy gradient; minimum perturbation that crosses a bucket boundary and flips the action). We do not directly compare absolute crossover ε values across methods because the adversarial constructions are not numerically equivalent. The cross-method pattern we claim is the *existence* of crossover at small ε, not numerical equivalence of crossover thresholds.

**Note on P3 comparator semantics**: adversarial perturbations are targeted at the RL policy's learned representation. They are not generic sensor noise and are not meaningful inputs for hand-coded heuristics in the same way. Therefore the primary adversarial comparison is `classical_on_true_state - RL_on_adversarial_observation`.

---

### Holm-Bonferroni correction

Family-wise α = 0.05 across all nine predictions. For each prediction, compute a paired sign-flip randomization p-value over the 30 per-seed differences with 100000 Monte Carlo sign flips, using the directional alternative implied by the prediction (`mean Δ < 0` for confirmation, `mean Δ > 0` for falsification). Per-prediction adjusted thresholds are computed by Holm's step-down procedure.

A prediction is confirmed or falsified only when both the 95% CI direction and the Holm-adjusted directional p-value agree. If the CI and adjusted p-value disagree, report the cell as inconclusive with the discrepancy documented.

---

## 5. What we will not do

- **No re-optimisation of perturbation magnitudes** after seeing results.
- **No selective reporting.** All nine predictions are reported regardless of outcome, in a table grouped by perturbation family and method.
- **No exclusion of seeds** for reasons other than documented infrastructure failure (simulator crash, training divergence). Any excluded seeds reported in a footnote.
- **No additional perturbations** beyond observation lag, workload tail shift, and adversarial observation. Exploratory findings reported as exploratory.
- **No new method proposals.** The empirical section substantiates the position; it does not advance a solution.
- **No retraining experiments.** All three methods evaluated as fixed clean-trained policies.
- **No direct numerical comparison of adversarial crossover ε across methods.** DeepRM and Decima use FGSM; Rossi uses the bucket-flip construction. We report the *existence* of crossover at small ε across all three, not numerical equivalence of crossover thresholds.

---

## 6. What outcomes look like

- **All nine confirmed**: the empirical section establishes the structural pattern across three methods and three perturbation axes. Citable claim: "Three canonical RL-for-orchestration methods spanning resource allocation, DAG scheduling, and autoscaling lose their advantage over classical baselines under literature-calibrated observation lag, workload distribution shift, and adversarial observation perturbation." This is the strongest outcome.
- **Most confirmed (≥6 of 9)**: the pattern holds across most cells. Failures reported individually. The paper still has a clean cross-method finding.
- **Half-and-half**: some methods or some perturbations show the pattern, others do not. Reported honestly. The position remains defensible but the empirical claim is conditional on the specific cells that confirmed.
- **Mostly falsified**: the empirical evidence runs against the position's strong version. The paper's prose carries more of the weight and we acknowledge the empirical limitation explicitly. The position is not refuted (it is a structural claim, not a numerical one) but is empirically weak.
- **Mixed with inconclusive**: report CI widths for the inconclusive cells; discuss in limitations.

We commit to all outcomes being reportable. The discipline of recording predictions before experiments is what makes the test count.
