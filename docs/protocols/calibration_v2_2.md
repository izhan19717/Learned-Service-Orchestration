# Perturbation Calibration Document (v2.2)

**Project**: Learned service orchestration empirical study.
**Authors**: Khilji, Furutanpey, Dustdar.
**Purpose**: Record, with sources, the realistic-world perturbation magnitudes the empirical section uses. Every magnitude that appears in a figure or table must be traceable to an entry in this document.

**Scope of v2.2**: three methods (DeepRM, Decima, Rossi) × three perturbation families (observation lag, workload tail shift, adversarial observation). The perturbation family definitions (§§1–3) are method-agnostic. Method-specific translations are in §4.

---

## 1. Observation lag (realism check, not adversarial)

### 1.1 Anchor value

**P95 observation lag = 10 DeepRM time units, mapping the verbatim P95 Borglet polling interval reported in Verma et al. 2015.**

This perturbation is not an adversarial attack. It is the natural latency between a state change in a production cluster and its observation by the scheduler. We include it because DeepRM's simulator assumes instantaneous observation, which does not match the production reality documented in the Borg paper.

### 1.2 Source

Verma, A., Pedrosa, L., Korupolu, M., Oppenheimer, D., Tune, E., & Wilkes, J. (2015). "Large-scale cluster management at Google with Borg." *Proceedings of the 10th European Conference on Computer Systems (EuroSys '15)*. ACM.

### 1.3 Verbatim from source (verified by direct fetch)

> "The Borgmaster polls each Borglet every few seconds to retrieve the machine's current state and send it any outstanding requests."
> "[T]hese keep the 99%ile response time of the UI below 1 s and the 95%ile of the Borglet polling interval below 10 s."

Additional context from the same source:
- Task startup latency: median ~25 seconds (package installation accounts for ~80%).
- Online scheduling pass over the pending queue: under half a second.

### 1.4 Translation to DeepRM simulator units

DeepRM operates in discrete time units `t`. The original paper does not commit to a wall-clock interpretation of `t`. For calibration, we adopt the convention: **one DeepRM time unit = one wall-clock second.**

Under this convention, P95 production observation lag of 10 seconds maps to **k = 10 DeepRM time units** of state delay.

### 1.5 Sweep specification

`k ∈ {0, 1, 2, 5, 10, 20}` decision steps of lag, where `k = 0` is the unperturbed baseline.

Anchor value for the figure's vertical reference line: **k = 10**.

### 1.6 Operational semantics of the lag perturbation

When the agent acts on a stale state observation, the simulator processes the chosen action as follows:

- **Primary behaviour (best-effort identity)**: if the agent's chosen job-slot index still corresponds to the same waiting job as it did when the stale observation was captured, and the job still fits the current resource availability, schedule that job.
- **Fallback (no-op)**: if the chosen job-slot index no longer maps to the same job, or the job has since been scheduled, or the job no longer fits, treat the action as a no-op for the current decision step.
- **Never abort**: do not reject the agent's action; do not raise an error; do not force replanning.

This matches the operational behaviour of real production schedulers (act on stale data, do not abort).

### 1.7 Bias drift component

We do not include a stochastic-bias-drift component in this version. Earlier drafts included Ornstein-Uhlenbeck bias on observed utilisation; that component had medium-low confidence calibration and is dropped to keep the experimental design tight. The lag perturbation in this version is **pure delay**, not delay-plus-bias.

---

## 2. Workload tail shift (realism check, not adversarial)

### 2.1 Anchor value

**Pareto shape parameter α = 1.5** for the heavy-tail component of the job duration distribution, calibrated against Reiss 2012 SoCC observations of the Google cluster trace.

This perturbation is not an adversarial attack. It is exposure to workload distribution behaviour documented in real production cluster traces. We include it because DeepRM's bimodal duration distribution (uniform short jobs 1–3 time units, uniform long jobs 10–15 time units) covers a 15:1 duration range. The Google trace covers a duration range of more than five orders of magnitude.

### 2.2 Source

Reiss, C., Tumanov, A., Ganger, G. R., Katz, R. H., & Kozuch, M. A. (2012). "Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis." *Proceedings of the Third ACM Symposium on Cloud Computing (SoCC '12)*.

### 2.3 Verbatim from source (verified by direct fetch)

> "Job durations range from tens of seconds to essentially the entire duration of the trace [29 days]."
> "75% of jobs consist of only one task, half of the jobs run for less than 3 minutes."
> "Jobs shorter than two hours account for less than 10% of overall utilization (even though they represent more than 95% of the jobs)."

On distribution fit:
> "using a power-law-like heavy-tailed distribution for task usage and task durations… may approximate the observed behavior."

The authors explicitly say no simple parametric distribution fits the trace, but Pareto-tailed is the recommended parametric approximation in the heavy-tail regime.

### 2.4 DeepRM baseline workload (the simulator to be perturbed)

From Mao et al. 2016, verified by direct fetch:
- Jobs arrive as a Bernoulli process with rate parameter `λ`.
- Duration is bimodal: with probability `p_short = 0.8`, sample from uniform integer [1, 3]; with probability `p_long = 0.2`, sample from uniform integer [10, 15].
- Resource demands: two resources; dominant uniform [0.25, 0.5] of capacity, non-dominant uniform [0.05, 0.1].

### 2.5 Perturbation specification

Replace the long-job component (originally uniform integer [10, 15]) with a Pareto distribution Pareto(`x_min = 10`, α), rounded to the nearest integer and clipped at `x_max = 100`. The short-job component remains uniform integer [1, 3]. The mixture probabilities `(p_short, p_long) = (0.8, 0.2)` remain unchanged.

Sweep:
- `α = ∞` (the original DeepRM bimodal baseline, recovered as the long component becomes uniform [10, 15])
- `α = 3.0` (light heavy-tail)
- `α = 2.0` (moderate)
- **`α = 1.5` (anchor; matches Reiss heavy-tail recommendation)**
- `α = 1.2` (aggressive heavy-tail)

### 2.6 Honest note on the parametric approximation

Reiss explicitly says no simple parametric distribution matches the Google trace exactly. We use Pareto-tailed as a *controllable directional lever*. Our claim is not "Pareto α = 1.5 is the true distribution"; it is "shifting toward Pareto-tailed durations in the direction Reiss recommends, at a shape parameter within the heavy-tail regime, causes the RL advantage to disappear."

---

## 3. Adversarial observation perturbation (genuinely adversarial, evaluation-time only)

### 3.1 Anchor value

**ε = 0.05 (5% of state-image magnitude), evaluated as a worst-case adversarial perturbation on the observation channel.**

This is the only adversarial perturbation in the empirical section. It is evaluation-time only (no retraining). Its purpose is to provide direct empirical support for the adversarial-ML mapping in §IV: orchestration phenomena are structurally analogous to adversarial-ML attack classes, and the canonical RL method we test is in fact vulnerable to such attacks at small budgets.

### 3.2 Source / precedent

Huang, S., Papernot, N., Goodfellow, I., Duan, Y., & Abbeel, P. (2017). "Adversarial Attacks on Neural Network Policies." *Workshop track of the International Conference on Learning Representations (ICLR 2017)*.

This is the foundational paper establishing adversarial vulnerability of deep RL policies. They use FGSM-style perturbations on the observation channel with ε budgets in the 0.001–0.05 range relative to state magnitude.

### 3.3 Perturbation specification

At each decision step during evaluation (no retraining):
1. The agent receives the true state observation `s`.
2. Compute the untargeted adversarial perturbation `δ = ε · sign(∇_s L(s))`, where `L(s) = -log π(a* | s)` and `a*` is the action with maximum probability under the clean policy. Equivalently, `δ = -ε · sign(∇_s log π(a* | s))`. This is the FGSM construction adapted to policy-gradient RL: maximize loss on the policy's own clean preferred action.
3. The perturbed observation `s_adv = clip(s + δ, 0, 1)` is what the agent acts on.
4. The simulator transitions according to the agent's action on `s_adv`, but using the true underlying state.

### 3.4 Sweep specification

`ε ∈ {0, 0.01, 0.02, 0.05, 0.10}` relative to state-image magnitude.

Anchor for the figure's vertical reference line: **ε = 0.05** (the upper end of Huang 2017's standard budget range; degradation at smaller ε is stronger evidence).

### 3.5 Why this perturbation and not reward bias

Earlier drafts included reward measurement bias as the adversarial-flavoured perturbation. That version was dropped for two reasons: (a) its calibration was derived indirectly from sampling granularity rather than from direct measurement, giving medium-confidence calibration; (b) the interesting test required retraining under biased reward, which we have descoped everywhere else.

Adversarial observation perturbation is cleaner: standard adversarial-ML construction, citable precedent in Huang 2017, evaluation-time only (no retraining), and it directly addresses §IV's claim that observation channels in orchestration are vulnerable in the same way as observation channels in adversarial-RL benchmarks.

### 3.6 Honest scope of the claim

We are not running an "adversary in the wild" study. The threat model is restricted: a perturbation budget bounded by ε, applied at evaluation time, against a fixed trained policy. We claim the *existence of vulnerability* under this threat model, not that production orchestrators face active adversaries. The empirical section will state this scope explicitly.

### 3.7 Comparator semantics

The primary P3 comparison attacks only the learned RL policy. The classical comparator is evaluated on the true unperturbed structured state. FGSM and the Rossi bucket-flip construction target the learned policy's representation; they are not generic measurement noise. Applying the same policy-targeted perturbation to a hand-coded heuristic would test a different hypothesis and can introduce arbitrary decoding choices from perturbed neural features back to valid structured scheduler inputs.

The paired comparison still uses the same underlying arrivals, jobs, DAGs, and simulator transitions. Only the RL policy's observation channel is adversarially perturbed.

---

## 4. Method-specific translation

The perturbation families in §§1–3 are method-agnostic. Each method requires translation of the lag, workload, and adversarial perturbations to its specific task class, action space, and policy representation. This section records those translations explicitly.

### 4.1 DeepRM (Mao 2016 HotNets, scheduling)

DeepRM uses one scheduler decision step as one simulator time unit; lag is applied in decision steps, workload tail perturbation is applied to long-job duration, and FGSM is applied to the state-image observation. Anchors: `k=10`, `α=1.5`, `ε=0.05`.

### 4.2 Decima (Mao 2019 SIGCOMM, DAG scheduling)

**Observation lag**: Decima is event-driven, not time-driven. Lag is delay between a DAG node finishing and the scheduler being notified of completion. Translate the 10-second Borg P95 anchor multiplicatively: lag = `λ × expected stage duration`, with the expected stage duration computed per-DAG from the training distribution. Sweep `λ ∈ {0, 0.25, 0.5, 1.0, 2.0}`. Anchor: `λ = 1.0` (lag comparable to typical stage duration, matching the Verma 2015 P95 scale relative to the decision interval).

**Operational semantics of stale-state action**: best-effort identity. If the GNN's chosen node is still schedulable when the action arrives, schedule it. If not, no-op. Never abort.

**Workload tail shift**: Decima trains on TPC-H. Alibaba 2018 is used only to estimate heavier-tailed DAG-size statistics, not to inject Alibaba DAGs directly into Decima. Construct the evaluation set by reweighting the existing TPC-H DAG/template pool so sampled DAG sets shift toward Alibaba-like size statistics (e.g., total work, task count, depth, branching factor). The weighting parameter `w` controls the strength of this bias. Sweep `w ∈ {0.0, 0.3, 0.5, 0.7, 0.9}`. Anchor: `w = 0.5` (substantial shift toward heavy tail, but not extreme — calibrated against Reiss 2012's observation that production job durations span 5+ orders of magnitude).

**Adversarial observation perturbation**: Decima's policy is a GNN over the DAG. FGSM-style perturbation applies natively because the GNN is differentiable. The official source feature schema is `(executors_assigned/20, source_job_flag, source_executors/20, remaining_work/100000, remaining_tasks/200)`. Perturb only the continuous channels and keep the categorical `source_job_flag ∈ {-2, 2}` fixed. Use `δ = ε · sign(∇_x L(x))`, where `L(x) = -log π(a* | x)`, `x` is the concatenation of node feature vectors, and `a*` is the most-probable action under the clean observation. Clip perturbed features to their valid ranges. Decima acts on `x_adv`; Graphene acts on the true unperturbed DAG state in the primary P3 comparison. Sweep `ε ∈ {0, 0.01, 0.02, 0.05, 0.10}` relative to node feature magnitude. Anchor: `ε = 0.05`.

### 4.3 Rossi 2019 IEEE CLOUD (model-based tabular autoscaling)

**Primary method**: the Rossi paper's primary controller is the model-based tabular RL method. The official RLAD repository currently defaults to DynaQ2, but DynaQ2 is treated as a simulator-default sensitivity rather than the primary Rossi method.

**Observation lag**: Rossi's autoscaler operates on metric streams. In the official RLAD simulator, the controller state is `(replica_count, utilization_bucket, cpu_allocation_bucket)`. Lag is delay between utilization changing and the autoscaler observing the updated metric. Translate directly: lag in wall-clock seconds. Sweep `k ∈ {0, 1, 2, 5, 10, 30}` seconds. Anchor: `k = 10` seconds (the Verma 2015 P95 anchor at face value).

**Workload tail shift**: Rossi's official RLAD simulator ships a bundled workload profile (`data/profile.dat`). The reproduction gate uses that bundled profile as operational ground truth. Perturbation has two sub-dimensions:
- Arrival-rate burstiness: replace or resample the native arrival profile with a Hawkes-process arrival pattern. Sweep Hawkes branching ratio `β ∈ {0, 0.3, 0.5, 0.7, 0.9}`. Anchor: `β = 0.7`.
- Service-time tail: replace exponential service times with Pareto-tailed. Sweep `α_pareto ∈ {∞, 3.0, 2.0, 1.5, 1.2}`. Anchor: `α_pareto = 1.5` (matches Reiss 2012 anchor).

Per the methodological-consistency principle, we report results across both sub-dimensions but treat the *service-time tail* sub-dimension as the primary P2 test for Rossi, because it directly mirrors DeepRM's and Decima's workload-tail perturbation. The burstiness sub-dimension is reported as supplementary.

**Adversarial observation perturbation**: Rossi/RLAD uses a tabular value function over discretised state buckets. The FGSM construction is not directly applicable because there is no policy gradient — the chosen action is a discontinuous function of the continuous observation (minimum-cost action over the table entry indexed by the observation's bucket).

**Adapted construction (minimum bucket-flip perturbation)**: in the primary operational construction, perturb only the continuous pre-discretisation utilization observation `u` and do not perturb replica count or CPU allocation, because those are controller-known configuration state in the Java simulator. Find the smallest L∞ perturbation `δ` such that `bucket(u + δ) ≠ bucket(u)` AND the greedy minimum-cost action in the resulting state differs from the clean action. In words: the smallest utilization perturbation that crosses a bucket boundary in a direction that changes the chosen action.

This is the discrete-state analogue of FGSM. We do not call it FGSM — we call it "minimum bucket-flip perturbation" — and we cite Huang 2017 as the FGSM precedent we are adapting. The empirical section will be explicit that Rossi's adversarial construction differs from DeepRM's and Decima's because Rossi's policy is discrete.

Sweep `ε ∈ {0, 0.01, 0.02, 0.05, 0.10}` relative to state-vector magnitude. Anchor: `ε = 0.05`.

In P3, Rossi acts on the bucket-flipped observation; HPA acts on the true unperturbed utilization state. This matches the shared P3 comparator semantics in §3.7.

**Honest note on calibration parity**: the bucket-flip construction at a given `ε` is *not* equivalent to FGSM at the same `ε`. They are conceptually analogous (both ask "what is the smallest observation perturbation that flips the action") but operationally different. We report Rossi's results with explicit acknowledgement of this difference and do not directly compare absolute crossover ε values across methods. The cross-method pattern we claim is the *existence* of crossover at small ε, not numerical equivalence of crossover thresholds.

---

## 5. Out of scope for v2.2

The following remain out of scope:

- **Reward channel bias** (former v1 Perturbation C): replaced by adversarial observation perturbation, which addresses the same "adversarial-flavoured" empirical need with cleaner calibration.
- **Inter-arrival burstiness as DeepRM's primary perturbation** (v1 B.2): Rossi covers burstiness as a supplementary sub-dimension; DeepRM does not.
- **Resource demand correlation** (v1 B.3): dropped. The R²-vs-r calibration was a source of precision confusion in v1.
- **Retraining under perturbation** (the original P3 daring prediction): all three methods are evaluated as fixed clean-trained policies. Adversarial training and the "RL converges to the heuristic" prediction are out of scope.

These are acknowledged in the paper's limitations subsection.

---

## 6. Calibration confidence

| Perturbation | Source | Confidence |
| --- | --- | --- |
| Observation lag, P95 = 10 s (DeepRM, Rossi); λ=1.0 stage-multiplier (Decima) | Direct verbatim quote, Verma 2015 EuroSys; Decima multiplier is method-specific translation | HIGH (DeepRM, Rossi); MEDIUM-HIGH (Decima multiplier translation) |
| Workload tail shift, Pareto α = 1.5 (DeepRM, Rossi); w = 0.5 Alibaba-bias (Decima) | Reiss 2012 SoCC heavy-tail recommendation; Decima sampling weight is method-specific translation | MEDIUM-HIGH |
| Adversarial observation ε = 0.05 (DeepRM, Decima FGSM; Rossi bucket-flip) | Huang 2017 ICLR precedent for FGSM ε range; Rossi construction is adapted analogue | HIGH (DeepRM, Decima); MEDIUM (Rossi — different construction, conceptually analogous) |

All anchors trace to specific passages in foundational papers. The paper's limitations subsection acknowledges that the Pareto parameterisation is a directional approximation, the Decima translations are method-specific choices, and the Rossi bucket-flip construction is conceptually analogous to FGSM rather than identical to it.

---

## 7. Citations to be added to the paper bibliography

- **verma2015borg**: Verma, A., et al. (2015). Large-scale cluster management at Google with Borg. EuroSys '15. ACM. DOI: 10.1145/2741948.2741964.
- **reiss2012heterogeneity**: Reiss, C., Tumanov, A., Ganger, G. R., Katz, R. H., & Kozuch, M. A. (2012). Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis. SoCC '12.
- **huang2017adversarial**: Huang, S., Papernot, N., Goodfellow, I., Duan, Y., & Abbeel, P. (2017). Adversarial Attacks on Neural Network Policies. ICLR 2017 Workshop.
- **mao2019decima**: Mao, H., Schwarzkopf, M., Venkatakrishnan, S. B., Meng, Z., & Alizadeh, M. (2019). Learning Scheduling Algorithms for Data Processing Clusters. SIGCOMM '19.
- **rossi2019horizontal** (already present): Rossi, F., Nardelli, M., & Cardellini, V. (2019). Horizontal and vertical scaling of container-based applications using reinforcement learning. IEEE CLOUD '19.

The Mao 2016 (DeepRM) citation is already in the bibliography as **mao2016resource**.
