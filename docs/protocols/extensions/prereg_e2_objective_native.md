# Pre-registration E2 — Objective-Native (Constrained) Demonstration for D1

**Status:** Frozen before any E2 training run. Confirmatory.
**Registered:** 2026-06-03, before first E2 execution.
**Commit/provenance:** release repository commit plus SHA256 manifests and
MLflow run IDs recorded in the generated artifacts. Private hostnames and IP
addresses are intentionally excluded from this public package.
**Compute:** remote workstation for canonical evaluation; local workstation for
smoke tests only.

## 1. Purpose
Test D1's central claim on a controlled comparison. Encoding an SLO as one term of a scalar reward (the published Rossi objective) is hypothesised to (a) buy apparent robustness with an operationally costly behaviour the scalar under-prices (reconfiguration churn), and (b) yield a "winner" that depends on what is measured. We compare it against an objective-native formulation of the same task, in which the SLO is a hard constraint, trained with an off-the-shelf constrained-RL method. F-constrained is a diagnostic comparator, the D1 analogue of HPA-v2; it is NOT proposed as a method or a contribution.

## 2. Task and formulations (pre-committed)
Task: Rossi horizontal/vertical autoscaling in the RLAD simulator. Same observation space, action space, network architecture, and per-step compute budget for both formulations.
- **F-scalar (published baseline).** Reward = 0.90·(SLA compliance) + 0.09·(−resource cost) + 0.01·(−reconfiguration churn). Identical to the reproduced Rossi policy used in the main study (MLflow [ID]).
- **F-constrained (objective-native).** Minimise expected resource cost subject to E[SLA-violation rate] ≤ τ, with NO churn term and NO SLA term in the objective. Primary solver: Lagrangian PPO (PPO-Lagrangian / RCPO), citing the constrained-MDP formulation (Altman 1999). Library and version recorded on commit ([e.g. omnisafe X.Y]).
- **Constraint threshold τ (pre-committed):** set to the clean-cell SLA-violation rate achieved by F-scalar, so the two policies are matched on the operational invariant. The comparison is then "at equal SLA compliance, what does each formulation do to cost and churn, especially under perturbation."
- **Fallback (pre-committed):** if Lagrangian PPO fails the training-stability gate (§4), switch to a fixed-penalty / barrier formulation (large fixed penalty on SLA violation, no weighted SLA term), also citing CMDP. If that also fails, report the null per §6.

## 3. Training protocol (pre-committed)
- Both formulations trained on the same RLAD environment, same workload corpus, 5 training seeds each.
- Same total environment steps for both (record the budget).
- **Competency gate for F-constrained (pre-committed):** on a held-out clean validation set it must (a) satisfy E[SLA violation] ≤ τ + 0.5 percentage points, and (b) avoid degenerate policies, checked by a sanity band on mean replica count (neither pinned at the min nor at the max bound). Tuning budget capped at N = 12 configurations (record the search space). Failure after the cap triggers the §2 fallback.

## 4. Evaluation (pre-committed)
Both policies evaluated on clean + the three perturbations at the main-study anchors (P1 k=10, P2 α=1.5, P3 ε=0.05), 30 paired windows each. Metrics reported SEPARATELY, never as a scalar:
- SLA-violation rate
- Resource cost
- Reconfiguration churn (scaling actions per window)
- Overload peak (max utilisation above capacity)
For each metric and cell: mean, 95% block-bootstrap CI (L ∈ {5,10}), and the paired difference (F-scalar − F-constrained) with sign-flip p-value (1e5 flips), Holm-corrected across the four-metric family within each cell.

## 5. Hypotheses and decision rules (pre-committed, including disconfirming outcomes)
- **H1 — the scalar objective hides churn.** Under lag (P1), at matched SLA, churn(F-scalar) − churn(F-constrained) > 0 with the 95% CI excluding 0.
  - Confirmation (supports D1): the scalar formulation's apparent robustness is bought with churn the scalar under-prices.
  - **Disconfirmation (weakens D1; committed):** if churn(F-scalar) ≤ churn(F-constrained) at matched SLA under lag, or the difference is not significant, the churn-hiding mechanism is not demonstrated. We soften D1 to a formulation-level argument and report the null in §VI-A.
- **H2 — the ranking depends on the metric.** Under at least one perturbation, at matched SLA, F-scalar is better on the published scalar objective but worse on at least one disaggregated operational metric (churn or overload peak), CI excluding 0.
  - Confirmation: the scalar metric is not operationally interpretable.
  - Disconfirmation: if F-scalar dominates F-constrained on every disaggregated metric at matched SLA, H2 is dropped.
- **H3 — clean-cell parity (confound guard).** On the clean cell, at matched SLA, the two policies' resource cost is within ±10%.
  - If F-constrained's clean cost is worse than this band, the comparison is confounded by F-constrained simply being a weaker policy in this simulator; we report that honestly rather than as support for D1.

## 6. Negative-result commitment
If F-constrained cannot be trained to match τ under both the primary and fallback formulations within the tuning cap, we report that the objective-native formulation could not be realised in this simulator at matched SLA, and D1 retains only its formulation-level (non-empirical) support plus Companion Analysis A.

## 7. Companion Analysis A — reward-weight sensitivity on existing logs (zero new training, runs regardless of E2 outcome)
Using the already-logged F-scalar rollouts (clean and P1 lag) that record SLA, cost, and churn separately, recompute the published scalar objective under churn weights w_churn ∈ {0.01, 0.05, 0.10, 0.20, 0.30} (renormalised), and recompute the Rossi-vs-comparator verdict at each.
- Pre-committed hypothesis: the "Rossi is robust under lag" verdict holds only near the published w_churn = 0.01 and reverses at operationally plausible churn weights.
- This is a deterministic re-scoring of fixed logs; it is reported in full and de-risks E2 (it delivers the weighting-dependence point even if §6 triggers).

## 8. Fixed factors, statistics machinery, deviations
Statistics identical to the main study (30 paired windows, 5000-resample bootstrap, 1e5 sign-flips, Holm; Rossi block-bootstrap primary). Record solver/library versions, τ, the tuning search space, seeds, and compute. Any deviation is logged here with rationale and timestamp before analysis of the affected comparison.

## 9. Identity safeguard
F-constrained is an off-the-shelf CMDP comparator used diagnostically, exactly as HPA-v2 is an off-the-shelf controller. The paper makes no claim of F-constrained as a method or contribution. This preserves the position-paper scope.
