# Experiment B — Production-Grade HPA-Equivalent Threshold Baseline for the Rossi Cells

**Status: pre-registered. Specification frozen before any new simulator runs are executed.**

## 0. Purpose

The threshold controller bundled with the RLAD simulator, which is the heuristic
comparator used in the Rossi paper, lacks the stabilisation features implemented
by production threshold autoscalers. Its bang-bang oscillation under observation
lag is partly a property of the bundled implementation rather than a property of
heuristic control in general. This experiment tests whether the Rossi P1 result
generalises from the simulator-bundled threshold controller to a
production-grade HPA-equivalent controller.

This experiment replaces the bundled threshold with a Kubernetes-HPA-equivalent controller implementing the autoscaling/v2 algorithm with default stabilisation features, and reruns Rossi P1, P2, P3, and clean cells with the new comparator.

## 1. Algorithm specification (Kubernetes HPA, autoscaling/v2)

The HPA-equivalent controller implements the algorithm documented in the Kubernetes autoscaling/v2 API.

### 1.1 Core formula

At each sync, given current replica count `currentReplicas`, current observed utilisation `currentMetric`, and target utilisation `desiredMetric`:

```
ratio = currentMetric / desiredMetric
desiredReplicas = ceil(currentReplicas * ratio)
```

### 1.2 Tolerance

If `|ratio − 1| < tolerance`, no scaling action is taken in this sync cycle (`desiredReplicas := currentReplicas`).

### 1.3 Stabilisation windows

The controller maintains a history of `desiredReplicas` recommendations over the last `max(scaleDown.stabilizationWindowSeconds, scaleUp.stabilizationWindowSeconds)` simulator ticks.

- **For scale-down** (`desiredReplicas < currentReplicas`): the actually-applied recommendation is the *maximum* of `desiredReplicas` over the last `scaleDown.stabilizationWindowSeconds` ticks (most conservative scale-down).
- **For scale-up** (`desiredReplicas > currentReplicas`): the actually-applied recommendation is the *minimum* of `desiredReplicas` over the last `scaleUp.stabilizationWindowSeconds` ticks. With the default `scaleUp.stabilizationWindowSeconds = 0`, the recommendation acts immediately.
- **For no change**: no stabilisation logic invoked.

### 1.4 Rate policies

After stabilisation, apply the scale-up and scale-down rate policies. Defaults:

- `scaleUp.policies = [{type: Percent, value: 100, periodSeconds: 15}, {type: Pods, value: 4, periodSeconds: 15}]`, `selectPolicy: Max` — the controller may scale up by at most max(100% of current, 4 pods) per 15-second period.
- `scaleDown.policies = [{type: Percent, value: 100, periodSeconds: 15}]` — the controller may scale down by at most 100% of current per 15-second period.

### 1.5 Sync period

The control loop runs every `syncPeriodSeconds` simulator ticks. Default 15.

### 1.6 Replica bounds

`minReplicas` and `maxReplicas` clip the applied recommendation. We use the same bounds as the bundled threshold: `minReplicas = 1`, `maxReplicas = 10`.

### 1.7 Defaults used in this experiment

```yaml
syncPeriodSeconds: 15
targetUtilization: 0.50         # representative production-grade CPU target; user-specified in Kubernetes
tolerance: 0.10                  # 10% deadband around target
minReplicas: 1
maxReplicas: 10
behavior:
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
      - {type: Percent, value: 100, periodSeconds: 15}
  scaleUp:
    stabilizationWindowSeconds: 0
    policies:
      - {type: Percent, value: 100, periodSeconds: 15}
      - {type: Pods, value: 4, periodSeconds: 15}
    selectPolicy: Max
```

The 50% target utilisation is a representative production-grade configuration
for conservative autoscaling. Kubernetes does not define a universal default
`target.averageUtilization`; users specify this value. Stabilisation windows,
tolerance, sync period, and scale-rate policies retain the Kubernetes defaults
documented for autoscaling/v2. We do not tune any of these parameters against
the Rossi workload.

### 1.8 Observation channel

The HPA controller reads `currentMetric` as a continuous CPU-utilisation signal.
Under clean and P2 (service-time tail), the value read is the true current
utilisation. Under P1 (observation lag `k`), HPA-v2 and Rossi both read the
lagged continuous utilisation from `k` simulator ticks ago because lag is a
shared telemetry-channel perturbation.

Under P3 (bucket-flip ε), Rossi alone sees the bucket-flipped discretised state
used to index its tabular Q-table. HPA-v2 reads the true continuous utilisation,
identical to the bundled threshold's reading under the main paper's P3 protocol.
The bucket-flip perturbation is representation-specific and has no operationally
meaningful action on a controller that does not bucket the continuous utilisation
signal. This preserves Option A consistency: learned-controller observation
attacks apply to the learned policy's representation, while heuristic
comparators read the true state.

## 2. Implementation plan

1. Create a new controller class `HPAv2Controller` in the Rossi controller module used by the experiment pipeline. The class implements the algorithm in §1.
2. Add the new controller to the simulator's controller registry alongside the existing bundled threshold controller and `RossiOnlineController`. Use the registry/reporting key `hpa_v2`. Report the existing simulator-bundled threshold as `bundled_threshold`, not as HPA, to avoid conflating the paper's comparator with a Kubernetes-equivalent controller.
3. Verify the implementation with three sanity tests (§3).
4. Run the four cells (clean, P1 k=10s, P2 α=1.5, P3 ε=0.05) for 30 paired non-overlapping windows of the official WorkloadGenerator2 profile (M/G/1, λ=0.4 req/s, exponential mean 100 ms service times). The seeds and window boundaries must match the existing Rossi evaluation exactly so that windows are paired between the new HPA runs and the existing Rossi-online runs.
5. Compute the paired statistic `Δᵢ = cost_hpa_v2(i) − cost_rossi_online(i)` for each window and each cell.
6. Run the statistical analysis in §4.

## 3. Sanity checks (must pass before §4 is executed)

These are pre-implementation gates. Failure on any of them indicates an implementation bug that must be fixed before the perturbation cells are run.

**S1 — clean stability.** In the clean cell, the HPA controller's replica count over a representative window must not oscillate bang-bang. Quantitative criterion: the standard deviation of the replica count over the window must be less than 1.0 replicas (compare to the bundled threshold's clean-cell replica-count std, which should be available from existing diagnostics).

**S2 — sync cadence.** Replica count changes occur only at sync boundaries. Quantitative criterion: in any representative window, the number of distinct replica counts observed within any 15-tick interval is at most 2 (the value before the sync and the value after).

**S3 — stabilisation correctness.** Inject a synthetic util sequence `[0.9, 0.9, 0.9, 0.2, 0.2, 0.2, …]` (a step from overload to underload). The HPA controller should scale up at the first sync after the overload and remain at maximum until at least 300 simulator ticks after the step to underload, regardless of the subsequent low recommendations. Quantitative criterion: from the step to underload at time `t₀`, the replica count at time `t₀ + 250` is at least the maximum reached during the overload phase.

If any sanity check fails, the analysis reports the failure and halts. We do not move on to perturbation cells until the controller behaves as the autoscaling/v2 algorithm specifies.

## 4. Statistical procedure

The procedure mirrors the main paper. For each cell:

1. Compute `Δᵢ = cost_hpa_v2(i) − cost_rossi_online(i)` for `i = 1, …, 30`.
2. Compute mean Δ at the anchor, the 95% iid paired percentile bootstrap CI with 5000 resamples, the 95% moving-block bootstrap CI at block lengths `L ∈ {5, 10}` (using the same procedure as Experiment A §3.1), the iid paired sign-flip p-value over 10⁵ flips, and the block sign-flip p-value at `L = 5` and `L = 10`.
3. Apply Holm-Bonferroni across the four HPA-vs-Rossi cells (clean, P1, P2, P3). (This is a separate family from the original nine-cell family; the original nine remain as reported in the main paper.)

## 5. Pre-registered hypotheses

We pre-register **two competing hypotheses** for the P1 cell, and one for each of the other three cells. These predictions are committed before any HPA runs are executed.

### 5.1 P1 (observation lag k=10s)

- **H_P1a — bundled-threshold mechanism does not generalise.** The HPA controller does not oscillate bang-bang under k=10 lag because its 300-second scale-down stabilisation prevents oscillation faster than 5 minutes. The HPA controller may over-provision (drift to high replica counts on early overload observations and stay there) or may be approximately stable. We predict that `Δ_HPA-Rossi(P1) > Δ_threshold-Rossi(P1) - 800`, i.e. the HPA's lag-induced cost is substantially smaller than the bundled threshold's. Magnitude prediction: `Δ_HPA-Rossi(P1) ∈ [-200, +400]` (HPA either slightly better or slightly worse than online Rossi at the anchor).

- **H_P1b — bundled-threshold mechanism does generalise.** The HPA controller still produces cost-bearing scaling pathology under k=10 lag because the scale-up channel has no stabilisation, allowing it to react to a single lagged-high observation by scaling to maximum, after which the scale-down stabilisation locks it there at high resource cost. We predict `Δ_HPA-Rossi(P1) ∈ [+300, +900]` (HPA still substantially worse than Rossi at the anchor, though by less than the bundled threshold's +965).

The correct outcome between H_P1a and H_P1b is the empirical question this experiment is designed to answer.

### 5.2 P2 (service-time tail α=1.5)

The bundled threshold dominates Rossi in clean and P2 cells (`Δ_threshold-Rossi(P2) = −106.0`). We predict the HPA controller will also dominate Rossi in both clean and P2 cells because the threshold logic is well-suited to the workload profile and HPA's stabilisation features do not affect the steady-state behaviour. Magnitude prediction: `Δ_HPA-Rossi(P2) ∈ [−200, −20]`.

### 5.3 P3 (bucket-flip ε=0.05)

The bucket-flip perturbation modifies Rossi's discretised state representation,
not the shared continuous telemetry channel. HPA-v2 therefore reads the true
continuous utilisation under P3, while Rossi reads the bucket-flipped state. We
predict the HPA controller dominates Rossi in the P3 cell as it does clean
because P3 is a learned-representation attack and HPA-v2 is not exposed to that
representation-specific corruption. Magnitude prediction:
`Δ_HPA-Rossi(P3) ∈ [−200, −20]`.

### 5.4 Clean

The HPA controller should dominate Rossi in the clean cell similarly to the bundled threshold (`Δ_threshold-Rossi(clean) = −85.8`). Magnitude prediction: `Δ_HPA-Rossi(clean) ∈ [−200, −20]`.

## 6. Diagnostic outputs

For both the clean cell and the P1 cell, render a representative-window failure trace in the same style as `figures/rossi_hpa_clean_vs_k10_failure_trace.pdf` from the main paper, but for the *new* HPA controller alongside the *new* HPA at lag. Save as `figures/hpa_v2_clean_vs_k10_failure_trace.pdf`. This figure visualises whether the HPA-v2 controller oscillates, over-provisions, or remains stable under lag.

Additionally, for the P1 cell, render a side-by-side comparison of the *bundled threshold* and the *HPA-v2 controller* in the same representative window under k=10 lag, saved as `figures/threshold_vs_hpa_v2_under_lag.pdf`. This directly visualises the mechanism difference.

## 7. Output table (companion to Table IV)

| Cell | Comparator | Δ | iid 95% CI | block 95% CI (L=10) | iid p<sub>u</sub> | p<sub>H</sub> (family of 4) | Outcome |
|------|-----------|---|------------|----------------------|--------------------|---------------------------|---------|
| Clean | HPA-v2 | … | … | … | … | … | … |
| P1 lag k=10 | HPA-v2 | … | … | … | … | … | … |
| P2 tail α=1.5 | HPA-v2 | … | … | … | … | … | … |
| P3 bucket-flip ε=0.05 | HPA-v2 | … | … | … | … | … | … |

## 8. Interpretation rules (pre-registered)

The P1 interpretation is determined by which of H_P1a or H_P1b prevails.

**Case 1 — H_P1a prevails** (HPA does not collapse under lag; the bundled-threshold mechanism does not generalise).

This outcome weakens the cross-method P1 narrative by showing that the
bundled-threshold mechanism does not generalise to the HPA-v2 comparator. The
Rossi P1 result is then interpreted as comparator-dependent rather than as a
general property of threshold control under lag.

**Case 2 — H_P1b prevails** (HPA still produces high cost under lag, via over-provisioning or another mechanism).

This outcome strengthens the cross-method P1 narrative by showing that the
lag-fragility result survives the HPA-v2 comparator. The Rossi P1 result is
then interpreted as robust to this comparator strengthening, with the diagnostic
trace used to identify whether the mechanism is oscillation, over-provisioning,
or another HPA-v2 failure mode.

**Either way**, the comparator-honesty point made by D3 in §V.C is reinforced: a published paper's evaluation against its bundled baseline can leave a substantial gap uncovered.

## 9. References

- Kubernetes autoscaling/v2 documentation: stabilisation windows, scale-up and scale-down rate policies.
- KEP-853 (Configurable HPA Scale Velocity) for the policy semantics.
- Rossi, F. et al. (2019). "Horizontal and Vertical Scaling of Container-Based Applications Using Reinforcement Learning."
