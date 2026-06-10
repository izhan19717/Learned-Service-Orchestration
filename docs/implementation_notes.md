# Implementation Notes

This document records implementation decisions required to interpret the
reported results.

## Scope

The repository implements fixed-policy evaluations for three published
reinforcement-learning orchestration methods:

- DeepRM, evaluated on mean slowdown.
- Rossi/RLAD, evaluated on total autoscaling cost and Rossi 2019 Table I
  reproduction metrics.
- Decima, evaluated with the official simulator-compatible workflow and
  perturbation evaluation scripts.

Perturbation magnitudes and decision rules are defined in
[calibration_v2_2.md](protocols/calibration_v2_2.md) and
[preregistration_v2_2.md](protocols/preregistration_v2_2.md).

## Observability

Training, reproduction gates, and primary evaluation runs were tracked with
MLflow during execution. The reported metrics, generated manifests, tables,
figures, and run identifiers used by the paper are under
[results/paper](../results/paper).

## Comparator Semantics

P1 observation lag and P2 workload-tail shifts are environmental perturbations.
They are applied symmetrically to the learned policy and the comparator unless
a method-specific protocol states otherwise.

P3 adversarial perturbations target the learned policy's observation
representation. The comparator reads the true structured state. This applies to
DeepRM FGSM, Decima FGSM, and Rossi's tabular bucket-flip analogue.

## DeepRM

The final DeepRM run uses the author-source-aligned simulator configuration,
official-style stochastic policy sampling, and the source-aligned Tetris*
comparator. The strict v2.2 clean gate and fixed-policy perturbation
evaluations are reported under
[results/paper/deeprm](../results/paper/deeprm).

The additional P1 stale-action sensitivity replaces the locked no-op fallback
with first-fit allocation on the current true state when the stale identity
action is invalid. It is reported as a methodology sensitivity, not as a
replacement for the locked protocol.

## Rossi/RLAD

The Rossi implementation is source-aligned to the public RLAD Java simulator.
The reproduction gate targets Rossi 2019 Table I, performance-weighted
5-action model-based row. The bundled threshold controller is the comparator
for the primary Rossi perturbation cells. HPAv2Controller is used only in the
HPA-v2 sensitivity extension.

Evaluation windows are cold-start online learning trajectories unless an
artifact states otherwise. Diagnostics report per-window deltas, observation
delta definitions, Q-table convergence, and representative lag traces.

## Decima

Decima uses the official simulator-compatible execution path for reproduction
and perturbation evaluation. The public simulator does not provide a directly
executable single-resource Graphene comparator. The main Decima outputs
therefore report the executable official-simulator comparator. The reconstructed
SRTF/Graphene-style comparator is reported as a sensitivity analysis.

## External Sources

Required upstream simulator checkouts and commits are listed in
[external/README.md](../external/README.md).
