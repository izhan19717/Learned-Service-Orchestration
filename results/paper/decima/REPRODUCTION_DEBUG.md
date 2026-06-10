# Decima Reproduction Debug

## Bottom Line

Decima did not pass the v2.2 official README reproduction gate.

The learned policy beats the official `dynamic_partition` comparator on the
README-scale test workload, but the improvement is much smaller than the
pre-specified target:

- observed mean-JCT improvement: `3.0125809474397287%`
- target improvement: `21%`
- relative error to target: `0.8565437644076319`
- gate passed: `False`

Per `04_codex_brief_decima.md`, Decima perturbation cells are therefore not
run.

## Training Provenance

Official source:

- repository: `https://github.com/hongzimao/decima-sim`
- local path: `external/decima-sim`
- commit: `c010dd74ff4b7566bd0ac989c90a32cfbc630d84`

Training command family followed the README:

- `exec_cap = 50`
- `num_init_dags = 1`
- `num_stream_dags = 200`
- `reset_prob = 5e-7`
- `reset_prob_min = 5e-8`
- `reset_prob_decay = 4e-10`
- `diff_reward_enabled = 1`
- `num_agents = 16`
- target checkpoint: `model_ep_10000`

Execution notes:

- Initial full training reached cumulative `model_ep_8700` before laptop
  restart.
- Recovery attempt 1 restarted from `model_ep_8700` but was interrupted before
  saving any new checkpoint.
- Recovery attempt 2 restarted from `model_ep_8700`, used a shorter checkpoint
  interval of 25 for durability only, and completed local `model_ep_1300`.
- Local `model_ep_1300` was copied as cumulative `model_ep_10000`.

The official TensorFlow saver restores the actor graph parameters. The
Python-side moving average reward buffer in the official training loop is not
checkpointed and therefore restarts after recovery; this is a documented
execution caveat.

## Completion Metadata Issue

The recovery watchdog crashed after copying the final checkpoint data files
because the destination TensorFlow `checkpoint` pointer file was root-owned
from Docker:

```text
PermissionError: [Errno 13] Permission denied:
results/checkpoints/decima/official_tf1_readme/checkpoint
```

This occurred after the final `model_ep_10000` checkpoint files existed. The
pointer file was corrected to:

```text
model_checkpoint_path: "model_ep_10000"
all_model_checkpoint_paths: "model_ep_10000"
```

This was a metadata/permission issue, not a simulator or optimizer crash.

## Test Gate Protocol

MLflow run: `36fb0fff65284fb2878a55755dc47b25`

Test scale:

- `exec_cap = 50`
- `num_init_dags = 1`
- `num_stream_dags = 5000`
- `num_exp = 1`
- schemes: `dynamic_partition`, `learn`
- saved model: `results/checkpoints/decima/official_tf1_readme/model_ep_10000`
- seed base: `num_ep = 10000000`, the official `test.py` default

The numeric evaluator mirrors the official `test.py` loop:

1. seed environment with `args.num_ep + exp`
2. reset environment
3. run each scheme until completion
4. accumulate `total_reward`
5. compute mean JCT from finished DAG completion times

Visualization side effects are disabled. This does not affect scheduling
behavior, action selection, seeds, or reward accumulation, and avoids the large
arrays allocated by the stock visualization path at 5000-DAG scale.

## Gate Metrics

| Scheme | Mean JCT | Total reward | Jobs | Decision steps | Runtime seconds |
|---|---:|---:|---:|---:|---:|
| dynamic_partition | 62746.507898420314 | -3137.9528600001604 | 5001 | 90265 | 81.20647478103638 |
| learn | 60856.21855628874 | -3043.419490000149 | 5001 | 142821 | 524.5594110488892 |

Reward-derived mean JCT exactly matches direct finished-job mean JCT up to
floating-point tolerance, which validates the metric extraction:

- dynamic partition: `62746.50789842352`
- learn: `60856.21855629172`

## Interpretation

The learned policy is not catastrophically broken: it improves mean JCT over
`dynamic_partition` by about 3%. However, this is far below the pre-specified
reproduction target. Under the Decima brief's non-negotiable gate, this is a
failed reproduction.

This result does not support running Decima P1/P2/P3. Running perturbations
after a failed clean reproduction would test a different object than the
published Decima method and would weaken the paper's scientific integrity.

## Consequence For The Paper

Decima should not be used as primary empirical perturbation evidence under v2.2.
The paper can mention it as an attempted but failed reproduction, with the
exact gate result reported transparently.

The active empirical evidence therefore comes from DeepRM and Rossi unless a
future protocol amendment authorizes new Decima work before any new Decima
perturbation results are inspected.
