# Decima Start Alignment Report

Date: 2026-05-21

This report records the Decima branch start state after DeepRM and Rossi
closure. It is a pre-execution alignment artifact, not a reproduction result.

## Decision

Decima execution is started, but only within the reproduction/preflight scope.
Decima P1/P2/P3 perturbation cells remain locked.

Allowed now:

- MLflow-tracked Decima preflight and smoke diagnostics.
- Official README reproduction-gate implementation/execution.
- Graphene comparator validation research.

Not allowed yet:

- Decima P1/P2/P3 perturbation sweeps.
- Paper claims using the local `GrapheneStyleComparator` as faithful Graphene.
- Any Decima cross-method result table entry.

The controlling protocol decision is `SCIENTIFIC_DECISIONS.md` Decision 23.

## Authoritative Anchors

- Official simulator repository:
  `https://github.com/hongzimao/decima-sim`
- Local clone: `external/decima-sim`
- Inspected commit: `c010dd74ff4b7566bd0ac989c90a32cfbc630d84`
- Official README training command:
  `train.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 200 ...`
- Official README test command:
  `test.py --exec_cap 50 --num_init_dags 1 --num_stream_dags 5000 --test_schemes dynamic_partition learn ...`
- Graphene paper/reference:
  `https://www.usenix.org/conference/osdi16/technical-sessions/presentation/grandl_graphene`

## MLflow Runs Started

| Run | ID | Status |
|---|---|---|
| Decima start preflight | `346600a91fa44152ad3e883cc2c12dee` | finished |
| Decima PyTorch/data smoke | `6bc35607f0b84fd09695b8110e397821` | finished |
| Graphene-style scaffold smoke | `c335935819f64f178ba69b8f5b8b5845` | finished |

## Current Readiness

Official artifact readiness:

- `external/decima-sim` exists.
- Official README exists.
- Official TPC-H template pool loads successfully.
- Template count: `154`.
- Max nodes: `18`.
- Max edges: `19`.
- Max depth: `9`.
- Max work: `7560108.111111112`.

Toolchain:

- PyTorch available: `2.12.0+cu130`.
- CUDA available through PyTorch.
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`, 8188 MiB.
- TensorFlow package is not installed.
- Gym package is not installed.
- Docker is not available on PATH.

Model scaffold:

- PyTorch Decima policy parameter count: `4770`.
- Source-aligned dimensions are implemented and unit-tested:
  node input dim `5`, job input dim `3`, hidden dims `[16, 8]`, output dim `8`,
  actor hidden layers `[32, 16, 8]`.

## Closed Gates

Two gates are currently closed:

1. Official README reproduction gate.
   - Required comparison: `learn` versus `dynamic_partition`.
   - Required scale: 50 executors, 200 streaming training DAGs, 5000 streaming
     test DAGs, reference checkpoint `model_ep_10000`.
   - Status: not run.

2. Graphene validation gate.
   - Graphene is the preregistered comparator for Decima P1/P2/P3.
   - The inspected single-resource official Decima README path does not provide
     Graphene.
   - `multi_resource_test.py` references Graphene, but the required
     `multi_resource_agents` and `multi_resource_env` directories are absent.
   - The local comparator is a `GrapheneStyleComparator` scaffold only.
   - Status: not validated; not paper evidence.

## Scientific Alignment

We are aligned on perturbation definitions:

- P1-Decima: observation/event lag multiplier `lambda = 1.0` anchor.
- P2-Decima: Alibaba-style TPC-H reweighting `w = 0.5` anchor.
- P3-Decima: FGSM node-feature perturbation `epsilon = 0.05` anchor.
- P3 comparator semantics: Decima perturbed, Graphene on true DAG state.

But those perturbation cells cannot be launched yet. The next scientific task
is reproduction, not perturbation.

## Immediate Next Step

Build and run the official README reproduction gate in the current environment.
Because the official code is TensorFlow 1-era and TensorFlow is unavailable in
the active Python 3.11 environment, the likely route is the existing PyTorch
port plus a simulator-compatible rollout/training/evaluation loop. The port
must reproduce the official `learn` versus `dynamic_partition` reference before
any Decima P1/P2/P3 result is allowed.

If reproduction fails within the 15% gate, Decima must be paused and a
`REPRODUCTION_DEBUG.md` artifact written before any perturbation work.
