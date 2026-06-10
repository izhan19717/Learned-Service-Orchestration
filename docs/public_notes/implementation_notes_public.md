# Public Implementation Notes

This document records the implementation decisions that are necessary for
independent review. It intentionally excludes private hostnames, IP addresses,
passwords, SSH configuration, and raw runtime logs.

## Scope

The repository implements the empirical package for three published
reinforcement-learning orchestration methods:

- DeepRM, evaluated on mean slowdown.
- Rossi/RLAD, evaluated on total autoscaling cost and Table I reproduction
  metrics.
- Decima, evaluated with the official simulator-compatible workflow and
  perturbation evaluation scripts.

All primary experiments are fixed-policy evaluations after clean-policy
training or source-aligned reproduction. Perturbation magnitudes and decision
rules are defined in [calibration_v2_2.md](../protocols/calibration_v2_2.md)
and [preregistration_v2_2.md](../protocols/preregistration_v2_2.md).

## Observability

Training, reproduction gates, and primary evaluation runs were tracked with
MLflow during execution. The public repository does not include the local
MLflow store because it contains large, machine-specific runtime state. The
paper-ready outputs under [results/paper](../../results/paper) retain the
reported metrics, manifests, tables, figures, and run identifiers where they
were used in generated artifacts.

## Comparator Semantics

P1 observation lag and P2 workload-tail shifts are environmental perturbations.
They are applied symmetrically to the RL policy and the relevant classical
comparator unless a method-specific protocol states otherwise.

P3 adversarial perturbations are policy-representation attacks. They are
applied to the RL policy's observation representation only. The classical
comparator reads the true structured state. This applies to DeepRM FGSM, Decima
FGSM, and Rossi's tabular bucket-flip analogue.

## DeepRM

The final DeepRM run uses the author-source-aligned simulator configuration,
including official-style stochastic policy sampling and the source-aligned
Tetris* comparator. The strict v2.2 clean gate and the fixed-policy
perturbation evaluations are reported under
[results/paper/deeprm](../../results/paper/deeprm).

The additional P1 stale-action sensitivity uses an alternative fallback in
which an invalid stale identity action falls back to first-fit allocation on
the current true state. This sensitivity is reported as methodology
sensitivity, not as a replacement for the locked protocol.

## Rossi/RLAD

The Rossi implementation is source-aligned to the public RLAD Java simulator.
The reproduction gate targets Rossi 2019 Table I, performance-weighted
5-action model-based row. The bundled threshold controller is reported as the
paper comparator; HPAv2Controller is used only in the explicit HPA-v2
sensitivity extension.

Evaluation windows are cold-start online learning trajectories unless an
artifact explicitly states otherwise. Diagnostic artifacts report per-window
deltas, observation-delta definitions, Q-table convergence, and representative
lag failure traces.

## Decima

Decima uses the official simulator-compatible execution path for reproduction
and perturbation evaluation. The public Decima simulator does not ship an
executable single-resource Graphene comparator in a directly reusable form.
Accordingly, the main Decima outputs report the official simulator comparator
used in the executable workflow, and the reconstructed SRTF/Graphene-style
comparator is labelled as a sensitivity analysis rather than evidence from an
official Graphene implementation.

## External Sources

Third-party simulator repositories are not vendored into this release package.
They are documented in [external/README.md](../../external/README.md) with
source URLs and expected local checkout locations.

## Raw Artifacts Not Committed

The following files are intentionally excluded from git:

- Python virtual environments.
- Local MLflow stores and SQLite databases.
- Raw training logs and terminal transcripts.
- Large checkpoints and intermediate rollout state.
- Private operational audit notes containing hostnames or connection details.
- Copyrighted PDFs obtained for local reading.

The committed package is therefore a reproducibility and review package, not a
full binary snapshot of every local runtime artifact.
