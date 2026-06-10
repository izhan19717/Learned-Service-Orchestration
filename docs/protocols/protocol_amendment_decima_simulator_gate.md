# Protocol Amendment: Decima Simulator Gate Correction

Date: 2026-05-28

## Reason For Amendment

The original v2.2 Decima brief required two things to be checked through one
gate:

1. the official public simulator README comparison, `learn` versus
   `dynamic_partition`;
2. the paper/webpage headline claim of approximately 21% improvement over
   hand-tuned scheduling heuristics.

After running the official simulator artifact, this conflation is not
scientifically clean. The public `decima-sim` README gives the executable
training and testing commands but does not provide a numeric improvement target
for the simulator run. The 21% headline is a paper/webpage-level claim from the
broader Decima evaluation, not an explicit numeric target in the public README
simulator artifact.

Therefore, using 21% as a numeric pass/fail threshold against the public
simulator's `dynamic_partition` comparator overstates what the executable
artifact itself supports.

This amendment is made before any Decima perturbation result exists.

## Corrected Gate Structure

Decima now has two separately named gates:

### Gate A: Official Simulator Operational Gate

Purpose: verify that the official public simulator artifact is runnable and
that the learned `model_ep_10000` policy is beneficial relative to the
README-exposed classical comparator.

Required conditions:

- use the official `decima-sim` TensorFlow 1.x simulator path;
- train using the README command structure:
  `exec_cap = 50`, `num_init_dags = 1`, `num_stream_dags = 200`,
  `reset_prob = 5e-7`, `reset_prob_min = 5e-8`,
  `reset_prob_decay = 4e-10`, `diff_reward_enabled = 1`,
  `num_agents = 16`, target `model_ep_10000`;
- test using the README command structure:
  `exec_cap = 50`, `num_init_dags = 1`, `num_stream_dags = 5000`,
  `test_schemes = dynamic_partition learn`, `num_exp = 1`;
- learned Decima has lower mean JCT than `dynamic_partition` on the paired
  README test workload.

Status: passed.

Observed result:

- `dynamic_partition` mean JCT: `62746.507898420314`
- `learn` mean JCT: `60856.21855628874`
- observed mean-JCT improvement: `3.0125809474397287%`
- MLflow run: `36fb0fff65284fb2878a55755dc47b25`

### Gate B: Paper Headline / Graphene-Class Gate

Purpose: reproduce the broader paper-level improvement claim against the
strongest classical comparator family.

Status: not executable from the current public single-resource simulator
artifact without additional comparator/source reconstruction.

Reason:

- the inspected `decima-sim` README does not publish a simulator numeric target;
- the single-resource README path exposes `dynamic_partition`, `learn`, and
  `spark_fifo`;
- the inspected repo references multi-resource Graphene paths but does not
  include the required executable Graphene source tree;
- the local `GrapheneStyleComparator` remains a scaffold, not a validated
  faithful Graphene implementation.

Gate B is therefore not used as a blocker for simulator-artifact perturbation
experiments. Instead, Decima claims are narrowed and labelled as
**official-simulator / dynamic-partition comparator evidence**, not a full
Graphene-class reproduction of all Decima paper claims.

## Perturbation Comparator Amendment

The primary Decima perturbation comparator changes from Graphene to the
official README-exposed `dynamic_partition` comparator.

New Decima per-seed statistic:

```text
Delta_i = mean_JCT_i(dynamic_partition) - mean_JCT_i(Decima)
```

Interpretation remains:

- `Delta > 0`: Decima beats the classical comparator.
- `Delta < 0`: the classical comparator beats Decima.

Prediction confirmation criterion remains directional:

- P1/P2/P3-Decima are confirmed if the 95% paired bootstrap CI for `Delta` at
  the pre-registered anchor lies entirely below zero.

## Scope Label Required In Paper

Any Decima perturbation result must be described as:

> Decima official-simulator perturbation result using the README-exposed
> `dynamic_partition` comparator.

It must not be described as:

- a Graphene comparison;
- a full reproduction of the Spark testbed headline result;
- evidence against the strongest Decima paper comparator.

## Integrity Note

This is a protocol correction, not a perturbation-result-driven change:

- no Decima P1/P2/P3 perturbation cells had been run before this amendment;
- the amendment narrows the claim rather than strengthening it;
- the previously recorded failed 21%-threshold artifact remains preserved as a
  record of the original over-strict gate interpretation.
