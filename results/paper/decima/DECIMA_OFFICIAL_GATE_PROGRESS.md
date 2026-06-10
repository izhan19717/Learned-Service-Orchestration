# Decima Official Reproduction-Gate Progress

Date: 2026-05-21

This is an implementation/readiness artifact, not a Decima reproduction result
and not a perturbation result.

## What Was Fixed

The Decima PyTorch port now separates two concepts that the official source
keeps distinct:

- Simulator duration data points: `5, 10, 20, 40, 50, 60, 80, 100`.
- Actor executor-limit actions under the README gate: `1..50`.

The official source constructs the actor with `range(1, args.exec_cap + 1)`.
With `exec_cap = 50`, the PyTorch actor must therefore have 50 executor-limit
actions. The port now does.

## What Was Implemented

- TensorFlow-free official simulator adapter for `external/decima-sim`.
- Official observation translation into PyTorch actor tensors.
- Dense PyTorch/Numpy message-passing path equivalent to the official Decima
  `Postman` path.
- Official valid-action masks for node and executor-limit decisions.
- Official `DynamicPartitionAgent` baseline execution through the authors'
  simulator.
- PyTorch actor execution through the authors' simulator.
- Tiny source-compatible training smoke with differential reward, piecewise
  linear baseline, checkpointing, and optimizer-state capture.

## MLflow Runs

| Run | ID | Scope | Status |
|---|---|---|---|
| Official simulator smoke | `b18d1818835f47a997fa52851eef07b8` | 5 streaming DAGs, dynamic partition vs untrained PyTorch actor | finished |
| Official training smoke | `1d0238ad90524803973353254cd3275e` | 1 epoch, 1 agent, 2 streaming DAGs | finished |
| Calibration probe | `1b5702ea8b144ba8a4aab232d10826ad` | 3 epochs, 2 agents, 20 streaming DAGs | finished |
| Official TF1 timing probe | `0e0f4c6839be40f4ab918b89b8209b7f` | 1 epoch, 16 agents, 200 streaming DAGs | finished |
| Official TF1 full training | `ca124f29adbc46e389654fa6b0a455ab` | interrupted by laptop restart at epoch 8705 | interrupted |
| Official TF1 recovery | `f9919381ed6e47efb6874f7bca706a58` | resume from `model_ep_8700`, run 1300 additional epochs | running |

## Official Simulator Smoke Result

This smoke is deliberately small and does not use README reproduction scale.

| Scheme | Finished jobs | Total reward | Mean JCT | Decisions |
|---|---:|---:|---:|---:|
| dynamic_partition | 6 | -2.23542 | 37257.0 | 125 |
| untrained PyTorch Decima | 6 | -5.50790 | 91798.33 | 176 |

The untrained policy being worse is expected. The important result is that both
schemes complete the same official simulator workload without action-contract
failures.

## Training Smoke Checkpoint

Checkpoint:
`results/checkpoints/decima/official_smoke/model_ep_1.pt`

Verified metadata:

- epoch: `1`
- executor action levels: `50`
- policy parameter tensors: `46`
- optimizer state entries: `46`

This confirms that future Decima resumes can preserve optimizer state rather
than repeating the DeepRM optimizer-reset weakness.

## Calibration Probe

The calibration probe is still below README reproduction scale, but it is large
enough to exercise multiple epochs, multiple rollouts per epoch, checkpointing,
and optimizer updates.

| Epochs | Agents | Stream DAGs | Mean total reward | Mean JCT | Checkpoint |
|---:|---:|---:|---:|---:|---|
| 3 | 2 | 20 | -17.740325 | 84477.74 | `results/checkpoints/decima/official_calibration_probe/model_ep_3.pt` |

Verified calibration checkpoint metadata:

- epoch: `3`
- executor action levels: `50`
- optimizer state entries: `46`
- recorded loss: `-0.09381471027110638`

## Official TF1 README Training Launch

Docker is available on the laptop, so the official README reproduction training
is now running through the authors' TensorFlow 1.x code path inside a pinned
Docker image:

`cisose-decima-tf1:1.15.5`

The active run uses:

- `train.py`
- `exec_cap = 50`
- `num_init_dags = 1`
- `num_stream_dags = 200`
- `reset_prob = 5e-7`
- `reset_prob_min = 5e-8`
- `reset_prob_decay = 4e-10`
- `diff_reward_enabled = 1`
- `num_agents = 16`
- `model_save_interval = 100`
- target checkpoint: `model_ep_10000`

Active launch metadata:

- MLflow run: `ca124f29adbc46e389654fa6b0a455ab`
- host wrapper PID: `64033`
- Docker container: `cisose_decima_tf1_model_ep_10000`
- log: `logs/training/decima_tf1_model_ep_10000_docker.log`
- checkpoint folder: `results/checkpoints/decima/official_tf1_readme`
- watchdog state: `results/decima/tf1_reproduction_launch_state.json`

The wrapper stops the Docker container once `model_ep_10000` exists. This is
intentional: the official `train.py` leaves worker processes waiting after the
requested epoch range.

The first buffered launch was intentionally stopped before any checkpoint
because progress logs were not visible in real time. The active launch uses
`python -u train.py`.

Observed early progress on the active run:

| Epoch | Worker reward seconds | Advantage seconds | Gradient seconds | Apply seconds |
|---:|---:|---:|---:|---:|
| 1 | 30.2618 | 1.5878 | 16.2987 | 0.0603 |
| 2 | 28.6088 | 1.6012 | 11.9845 | 0.0052 |

At this rate, the expected wall-clock time is several days. This is acceptable
for the reproduction gate, but Decima perturbation cells remain locked.

## Restart Recovery

On 2026-05-27 the laptop restarted and interrupted the first full TF1 Docker
training run. The final visible log lines were:

```text
training epoch 8705
time="2026-05-27T07:29:37+02:00" level=error msg="error waiting for container: unexpected EOF"
```

The latest complete checkpoint was `model_ep_8700`; epochs 8701 through 8705
were not preserved because checkpoints are saved every 100 epochs.

Recovery action:

- Resume source checkpoint:
  `results/checkpoints/decima/official_tf1_readme/model_ep_8700`
- Recovery MLflow run: `f9919381ed6e47efb6874f7bca706a58`
- Recovery container: `cisose_decima_tf1_resume_8700_to_10000`
- Host wrapper PID: `1727`
- Recovery log:
  `logs/training/decima_tf1_model_ep_10000_resume_from_8700_docker.log`
- Recovery checkpoint folder:
  `results/checkpoints/decima/official_tf1_readme_resume_from_8700`
- Local recovery target: `model_ep_1300`
- Cumulative final alias:
  `results/checkpoints/decima/official_tf1_readme/model_ep_10000`

Resume parameters:

- `--saved_model /workspace/results/checkpoints/decima/official_tf1_readme/model_ep_8700`
- `--num_ep 1301`
- `--model_save_interval 100`
- `--reset_prob 5e-08`
- `--entropy_weight_init 0.0001`

The reset probability and entropy weight are set to their epoch-8700 schedule
values. The official Python-side moving average reward buffer is not part of
the TensorFlow checkpoint and therefore restarts; this is retained as a
documented recovery caveat.

## Second Restart Recovery

The first recovery run was interrupted by another laptop restart before it
reached a checkpoint. The final visible line in
`logs/training/decima_tf1_model_ep_10000_resume_from_8700_docker.log` was:

```text
training epoch 31
```

Because the first recovery used `model_save_interval = 100`, local epoch 31 did
not produce a TensorFlow checkpoint. The durable state is still cumulative
`model_ep_8700`; local epochs 1 through 31 from the interrupted recovery are
discarded.

Second recovery action:

- Resume source checkpoint:
  `results/checkpoints/decima/official_tf1_readme/model_ep_8700`
- Recovery MLflow run: `bcadf0d8b7454cd28129596badf43fe6`
- Recovery container:
  `cisose_decima_tf1_resume_8700_to_10000_attempt2`
- Host wrapper PID: `4311`
- Recovery log:
  `logs/training/decima_tf1_model_ep_10000_resume_from_8700_attempt2_docker.log`
- Recovery checkpoint folder:
  `results/checkpoints/decima/official_tf1_readme_resume_from_8700_attempt2`
- Local recovery target: `model_ep_1300`
- Cumulative final alias:
  `results/checkpoints/decima/official_tf1_readme/model_ep_10000`

Second recovery parameters:

- `--saved_model /workspace/results/checkpoints/decima/official_tf1_readme/model_ep_8700`
- `--num_ep 1301`
- `--model_save_interval 25`
- `--reset_prob 5e-08`
- `--entropy_weight_init 0.0001`

The shorter checkpoint interval is an operational durability safeguard after
two laptop restarts. It does not change the simulator, reward, optimizer,
policy architecture, number of workers, or training update rule. The second
recovery has entered training and reached `training epoch 2`.

## Verification

Full local test suite:

`65 passed, 9 warnings`

The warnings are NumPy pickle/deprecation warnings from the official Decima
TPC-H `.npy` files, not test failures.

## Final Training Completion

The second recovery completed local epoch 1300, which maps to cumulative
`model_ep_10000` after resuming from `model_ep_8700`.

Final checkpoint files:

- `results/checkpoints/decima/official_tf1_readme_resume_from_8700_attempt2/model_ep_1300.*`
- `results/checkpoints/decima/official_tf1_readme/model_ep_10000.*`

The watchdog crashed after copying the final checkpoint files because the
destination TensorFlow `checkpoint` pointer file was root-owned from Docker.
The checkpoint data, index, and meta files were already present. The pointer
file was corrected to:

```text
model_checkpoint_path: "model_ep_10000"
all_model_checkpoint_paths: "model_ep_10000"
```

This is treated as a bookkeeping/permission issue after training completion,
not as a learned-policy failure.

## Official README Test Gate Result

The official README-scale test gate was run through the TF1 Docker path with
MLflow:

- MLflow run: `36fb0fff65284fb2878a55755dc47b25`
- Saved model: `results/checkpoints/decima/official_tf1_readme/model_ep_10000`
- `exec_cap = 50`
- `num_init_dags = 1`
- `num_stream_dags = 5000`
- `num_exp = 1`
- schemes: `dynamic_partition`, `learn`
- seed base: `num_ep = 10000000`, matching the official `test.py` default

Artifacts:

- `results/paper/decima/tables/decima_official_readme_gate.md`
- `results/paper/decima/tables/decima_official_readme_gate.json`
- `results/paper/decima/tables/decima_official_readme_gate_per_exp.csv`
- `logs/training/decima_tf1_readme_test_gate.log`
- final closure MLflow run: `1d40ec5d1d75432d8257f0d4cbe6f814`

Observed result:

| Scheme | Mean JCT | Total reward | Jobs | Decision steps |
|---|---:|---:|---:|---:|
| dynamic_partition | 62746.5079 | -3137.95286 | 5001 | 90265 |
| learn | 60856.2186 | -3043.41949 | 5001 | 142821 |

The learned policy beats `dynamic_partition`, but only by
`3.0125809474397287%` mean JCT. The v2.2 Decima reproduction target is the
brief's `21%` improvement target with ±15% relative tolerance because the
README does not publish a numeric reference. The observed relative error is
`0.8565437644076319`.

Gate passed: `False`.

## Final Status

Decima is scientifically closed as a reproduction-gate failure under v2.2.

Consequences:

1. Decima P1, P2, and P3 perturbation cells are not run.
2. Decima must not be used as evidence for the cross-method perturbation
   claims in the paper.
3. The Graphene comparator gate remains unresolved, but the reproduction gate
   failure is already sufficient to exclude Decima from the empirical
   perturbation evidence.
4. Decima may be discussed only as a failed reproduction / excluded method
   unless a future protocol amendment authorizes new Decima work before any
   new perturbation results are seen.
