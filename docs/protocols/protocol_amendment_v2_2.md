# Protocol Amendment v2.2

Date: 2026-05-15

Status: active before any DeepRM perturbation/evaluation results.

## Purpose

This amendment activates the three-method/nine-prediction empirical plan while
preserving the already-running DeepRM clean-policy training as valid. It records
corrections made before Decima or Rossi implementation begins and before any
cross-method evaluation table exists.

## Active Documents

- DeepRM clean-training brief: `03_codex_brief.md`
- Three-method calibration: `calibration New.md`
- Three-method pre-registration: `preregistration New.md`
- Decima brief: `04_codex_brief_decima.md`
- Rossi brief: `05_codex_brief_rossi.md`
- Source extraction record: `IMPLEMENTATION_NOTES.md`

## DeepRM Continuity

The DeepRM clean-policy training started under v2.1 remains canonical for v2.2.
The training configuration is unchanged:

- same simulator
- same workload family
- same hyperparameters
- same primary load cell
- same clean-policy reproduction target

Only the reporting structure changes: DeepRM is now one method in a
three-method, nine-prediction family. DeepRM perturbation evaluations have not
started at the time of this amendment, so no evaluation result is invalidated.

## DeepRM Gate Failure And Author-Source Rescue Addendum

Date: 2026-05-18

Status: active after DeepRM clean-gate evaluation, before any DeepRM
perturbation result.

The v2.2 DeepRM clean checkpoint completed 1000 iterations but failed the clean
reproduction gate. Therefore it is **not** a valid checkpoint for P1, P2, or P3.
The gate failure is recorded in `REPRODUCTION_DEBUG.md` and summarized in
`results/evaluation/deeprm/reproduction_gate_v2_2_summary.json`.

The failed run is preserved as a failed reproduction attempt, not overwritten.
Any further DeepRM rescue attempt must be explicitly labeled
`author-source reproduction` and must use the public DeepRM repository as the
operational ground truth for implementation details that conflict with the
paper prose.

Author-source rescue requirements:

- public source: `https://github.com/hongzimao/deeprm`
- inspected commit: `fa7841122d4c2993fa0a7ce1667ea1f171865d9d`
- source-discrete resource slots: dominant `5..10` of 10, other `1..2` of 10
- source extra-info state column: `time_since_last_new_job`
- source policy parameter count: `89,851`
- paper-mode parameter count retained for audit: `89,451`
- source reward timing: valid `Allocate` receives reward `0`; reward is
  computed only after `MoveOn`
- source training horizon from `run_script.py`: `simu_len = 200`
- source episode limit from `run_script.py`: `eps_max_len = 800`
- source direct policy-gradient load-sweep path is primary
- README supervised warm-start remains a separate explicitly labeled
  sensitivity only if the direct source path fails
- author-exact `Tetris` baseline means the source's operational label:
  `SourceTetris = other_agents.get_packer_action`, using dot product, not
  cosine or combined `Tetris*`
- the stricter combined `Tetris*` comparator remains the pre-registered v2.2
  perturbation comparator and must be reported separately
- author-style reproduction evaluation and the stricter v2.2 no-drop/drain
  evaluation must be reported separately

Low-cost implementation checks completed before any new long run:

- `author_source_config()` implemented.
- Source-discrete demand sampling implemented.
- Optional source extra-info state column implemented.
- Source reward timing implemented.
- Source no-external-admission/drop behavior implemented.
- Source exclusive max-start allocation boundary implemented.
- Source dot-product packing option implemented.
- Training can build an 89,851-parameter source-aligned policy.
- CLI `author-preflight` passed:
  MLflow run `a2158ef566a8471e98dd1b90aebcd116`.
- Cheap source-aligned training smoke passed:
  MLflow run `715ed885fa554a8c9974da840efa4ded`.
- Author-source evaluation smoke passed and is separated from strict v2.2
  evaluation:
  MLflow run `27234590163d470084da9f0f9d6e809d`.
- Regression tests covering the source/paper mismatch passed:
  `47 passed`.

The next long DeepRM run must be the source-aligned `lambda = 0.7` rescue only;
the remaining load-sweep policies wait until `lambda = 0.7` passes the
source-style reproduction gate.

### Author-Source Rescue Completion

Date: 2026-05-20

The source-aligned `lambda = 0.7` rescue completed and passed both required
DeepRM clean gates.

- Training/resume MLflow run: `2d7bb69faec64a8aaf0f82b03ea34aea`
- Final checkpoint:
  `results/checkpoints/author_source_rescue/load_0.7/policy_final.pt`
- Final checkpoint SHA256:
  `1d439eb4f5b47a7b1242a9824cd3db60f1f685ca0e53d57907bbf35d860e5f7e`
- Author-source reproduction gate MLflow run:
  `684ea029f34f495398b16661a5315860`
- Strict v2.2 clean gate MLflow run:
  `d5a14b32e67143078c565b927ccade5d`

This checkpoint is the valid DeepRM checkpoint for P1/P2/P3. The older
paper-mode checkpoint remains a failed reproduction and must not be used for
DeepRM perturbation results.

DeepRM perturbation evaluation was then executed before any Decima/Rossi
evaluation result:

- MLflow run: `ed667925d5ea4185a280ca34a9a2ef63`
- Artifact: `results/evaluation/deeprm/perturbation_sweeps_v2_2.json`
- P1/P2/P3 DeepRM anchor cells are falsified under DeepRM-only Holm
  correction; final v2.2 cross-method Holm correction remains pending
  Decima/Rossi p-values.

## Prediction Family

v2.2 contains nine pre-registered predictions:

- P1-DeepRM, P2-DeepRM, P3-DeepRM
- P1-Decima, P2-Decima, P3-Decima
- P1-Rossi, P2-Rossi, P3-Rossi

Holm-Bonferroni correction expands from the v2.1 DeepRM-only family of three
predictions to the v2.2 cross-method family of nine predictions. The family-wise
alpha remains 0.05. P-values use the paired sign-flip randomization test over
the 30 per-seed paired differences with 100000 Monte Carlo sign flips, with
directional alternatives matching the pre-registered prediction.

## P3 Comparator Semantics

The primary adversarial comparison now uses:

```text
classical baseline on true state - RL method on adversarial observation
```

This applies to all three P3 predictions:

- DeepRM acts on FGSM-perturbed neural state image; Tetris* acts on true
  structured state.
- Decima acts on FGSM-perturbed GNN node features; Graphene acts on true DAG
  state.
- Rossi/RLAD acts on the minimum bucket-flipped utilization observation; HPA
  acts on true utilization state.

Rationale: adversarial perturbations are targeted at the learned policy's
representation. They are not generic observation noise and are not meaningful
inputs for hand-coded heuristics in the same way. Applying the same targeted
perturbation to the classical baseline would test a different hypothesis.

## FGSM Sign Correction

FGSM for DeepRM and Decima uses the untargeted loss-increasing direction:

```text
L(s) = -log pi(a* | s)
delta = epsilon * sign(grad_s L(s))
```

Equivalently:

```text
delta = -epsilon * sign(grad_s log pi(a* | s))
```

where `a*` is the clean policy's most-probable action. This correction prevents
the implementation from accidentally increasing confidence in the clean action.

## Rossi Corrections

The official Java simulator `https://github.com/effereds/rlad-core-simulator`
is Rossi's operational ground truth. Inspected commit:
`d6a4ff136907eb1bd9e8b4151a9162231ce0ee6a`.

The complete Rossi 2019 paper PDF was obtained and inspected on 2026-05-15.
The paper-primary method is the model-based RL controller, not the Java
repository's current `AGENT_DYNAQ2` default. DynaQ2 remains useful as an
artifact-default sensitivity, but it is not the primary Rossi method for the
paper's reproduction gate.

Corrections:

- Primary Rossi method is paper-primary `AGENT_RLMB` / model-based tabular RL.
  The reproduction gate is the simulation Table I performance-focused
  5-action row, not an HPA-improvement claim.
- Primary comparator is HPA-style threshold control only. VPA is removed from
  the Rossi primary comparator.
- HPA-style parameters come from the Java source: scale out above utilization
  0.7; scale in below `0.75 * 0.7 * (replicas - 1) / replicas`.
- Official simulator default controller is `AGENT_DYNAQ2`, not plain
  Q-learning; this is recorded as the simulator-default sensitivity rather than
  the Rossi primary method.
- Official default state is `(replica_count, utilization_bucket,
  cpu_allocation_bucket)`, not `(replica_count, cpu_util_bucket,
  arrival_rate_bucket)`.
- Reward coefficients are `0.09` resource, `0.01` reconfiguration, `0.90` SLA.
- The bundled `data/profile.dat` is the reproduction workload ground truth.
- Rossi P3 perturbs utilization only in the primary bucket-flip construction;
  replica count and CPU allocation remain controller-known state.

Simulation reproduction target for the primary row:

- weights: `w_perf = 0.90`, `w_res = 0.09`, `w_adp = 0.01`
- action model: 5-action horizontal-or-vertical
- policy: Model-based
- Table I targets: `Rmax` violations `2.37%`, average CPU utilization
  `60.54%`, average CPU share `87.62%`, average containers `2.53`, median
  response time `10.39 ms`, adaptations `39.67%`

The paper does not report a clean HPA threshold-baseline improvement number.
HPA is therefore retained as our cross-method classical comparator for
P1/P2/P3, but not as the Rossi-paper reproduction gate.

## Decima Corrections

The official simulator `https://github.com/hongzimao/decima-sim` is the Decima
operational ground truth for simulator command structure. Inspected commit:
`c010dd74ff4b7566bd0ac989c90a32cfbc630d84`.

Corrections:

- Reference command uses 50 executors, 1 initial DAG, 200 streaming jobs for
  training, the README model folder
  `./models/stream_200_job_diff_reward_reset_5e-7_5e-8/`, and tests
  `model_ep_10000` on 5000 streaming jobs with `--canvs_visualization 0` and
  `--num_exp 1`.
- The reproduction gate first checks the official README reference comparison
  (`learn` versus `dynamic_partition`), then validates the paper-level
  comparator set used in our nine-prediction table.
- Alibaba 2018 is used only to derive a shifted synthetic DAG-size
  distribution over the native TPC-H pool. We do not convert Alibaba DAGs into
  Decima/Spark stage profiles.
- Decima P3 attacks only Decima's continuous GNN node-feature channels. The
  official categorical `source_job_flag` channel is not perturbed. Graphene acts
  on true DAG state.
- The inspected official Decima commit contains `multi_resource_test.py` with a
  `graphene` scheme reference but does not contain the imported
  `multi_resource_agents` or `multi_resource_env` source directories. Therefore
  the local comparator implementation is labelled `GrapheneStyleComparator`
  until a faithful Graphene reference is obtained/validated or the Decima
  comparator is explicitly protocol-amended before any Decima result is
  produced.

## MLflow Observability

All new Decima and Rossi training, evaluation, bootstrap, figure, and table
commands must run through MLflow, matching the DeepRM observability standard.
Each run must log:

- method and brief version
- protocol document checksums
- source repository URL and inspected commit for external simulators
- configuration parameters
- seeds and derived per-cell seeds
- perturbation family and magnitude
- comparator semantics
- raw per-seed results
- bootstrap outputs
- figures/tables as artifacts

No result is valid for the paper unless its MLflow run ID and protocol/source
manifest are recorded.

## Unchanged Numeric Anchors

All anchor magnitudes remain unchanged:

- observation lag: DeepRM/Rossi `k = 10`; Decima `lambda = 1.0`
- workload tail: DeepRM/Rossi Pareto `alpha = 1.5`; Decima `w = 0.5`
- adversarial observation: `epsilon = 0.05`
