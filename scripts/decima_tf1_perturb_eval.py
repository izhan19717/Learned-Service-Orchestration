#!/usr/bin/env python
"""Decima official-simulator perturbation evaluator.

Executed inside the TensorFlow 1.x Docker image with CWD
``external/decima-sim``. The evaluator currently supports the amended Decima P2
tail-shift cell by replacing the uniform TPC-H size sampler with a deterministic
weighted sampler over the official TPC-H size pool.
"""

from __future__ import print_function

import csv
import hashlib
import json
import os
import sys
import time

import matplotlib

matplotlib.use("agg")
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.getcwd())

from actor_agent import ActorAgent
from heuristic_agent import DynamicPartitionAgent
from msg_passing_path import get_unfinished_nodes_summ_mat
from param import args
from spark_env.env import Environment


_LAG_STATE = {"seed": 0}


def main():
    output_json = os.environ["DECIMA_PERTURB_OUTPUT_JSON"]
    output_csv = os.environ["DECIMA_PERTURB_OUTPUT_CSV"]
    perturbation = os.environ.get("DECIMA_PERTURBATION", "tail")
    tail_weight = float(os.environ.get("DECIMA_TAIL_WEIGHT", "0.0"))
    lag_lambda = float(os.environ.get("DECIMA_LAG_LAMBDA", "0.0"))
    fgsm_epsilon = float(os.environ.get("DECIMA_FGSM_EPSILON", "0.0"))
    bootstrap_seed = int(os.environ.get("DECIMA_BOOTSTRAP_SEED", "99017"))
    progress_interval = int(os.environ.get("DECIMA_PROGRESS_INTERVAL_STEPS", "0"))

    if perturbation not in ("tail", "lag", "fgsm"):
        raise ValueError("unknown perturbation {}".format(perturbation))

    _ensure_parent(output_json)
    _ensure_parent(output_csv)
    perturbation_metadata = {}
    if perturbation == "tail":
        perturbation_metadata.update(_install_tail_sampler(tail_weight))
    elif perturbation == "lag":
        perturbation_metadata.update(_install_lag(lag_lambda))

    tf.set_random_seed(args.seed)
    env = Environment()
    learn_session = tf.Session()
    learn_cls = AdversarialActorAgent if perturbation == "fgsm" else ActorAgent
    learn_kwargs = {"fgsm_epsilon": fgsm_epsilon} if perturbation == "fgsm" else {}
    agents = {
        "dynamic_partition": DynamicPartitionAgent(),
        "learn": learn_cls(
            learn_session,
            args.node_input_dim,
            args.job_input_dim,
            args.hid_dims,
            args.output_dim,
            args.max_depth,
            range(1, args.exec_cap + 1),
            **learn_kwargs
        ),
    }

    rows = []
    started_at = time.time()
    for exp in range(args.num_exp):
        seed = args.num_ep + exp
        print("Experiment {} of {} seed={}".format(exp + 1, args.num_exp, seed), flush=True)
        for scheme in ("dynamic_partition", "learn"):
            row = _run_scheme(
                env,
                agents[scheme],
                scheme=scheme,
                exp=exp,
                seed=seed,
                progress_interval=progress_interval,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            _write_csv(output_csv, rows)
            _write_json(
                output_json,
                _payload(
                    rows,
                    perturbation=perturbation,
                    tail_weight=tail_weight,
                    lag_lambda=lag_lambda,
                    fgsm_epsilon=fgsm_epsilon,
                    bootstrap_seed=bootstrap_seed,
                    started_at=started_at,
                    perturbation_metadata=perturbation_metadata,
                    learn_agent=agents["learn"],
                ),
            )

    _write_csv(output_csv, rows)
    _write_json(
        output_json,
        _payload(
            rows,
            perturbation=perturbation,
            tail_weight=tail_weight,
            lag_lambda=lag_lambda,
            fgsm_epsilon=fgsm_epsilon,
            bootstrap_seed=bootstrap_seed,
            started_at=started_at,
            perturbation_metadata=perturbation_metadata,
            learn_agent=agents["learn"],
        ),
    )


class AdversarialActorAgent(ActorAgent):
    """ActorAgent variant that applies FGSM to node features before acting."""

    def __init__(self, *args_, **kwargs):
        self.fgsm_epsilon = float(kwargs.pop("fgsm_epsilon"))
        super(AdversarialActorAgent, self).__init__(*args_, **kwargs)
        self.node_input_grad = tf.gradients(self.adv_loss, self.node_inputs)[0]
        self.attack_count = 0
        self.attack_abs_delta_sum = 0.0
        self.clean_target_prob_sum = 0.0
        self.adv_target_prob_sum = 0.0

    def invoke_model(self, obs):
        node_inputs, job_inputs, \
            job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, \
            exec_commit, moving_executors, \
            exec_map, action_map = self.translate_state(obs)

        gcn_mats, gcn_masks, dag_summ_backward_map, \
            running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags)

        node_valid_mask, job_valid_mask = \
            self.get_valid_masks(job_dags, frontier_nodes,
                source_job, num_source_exec, exec_map, action_map)

        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        node_act_probs, job_act_probs, _, _ = self.predict(
            node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map)

        target_node = int(np.argmax(node_act_probs[0]))
        target_job = job_dags.index(action_map[target_node].job_dag)
        target_exec = int(np.argmax(job_act_probs[0, target_job]))

        node_act_vec = np.zeros_like(node_act_probs)
        node_act_vec[0, target_node] = 1.0
        job_act_vec = np.zeros_like(job_act_probs)
        job_act_vec[0, target_job, target_exec] = 1.0

        grad = self.sess.run(self.node_input_grad, feed_dict={i: d for i, d in zip(
            [self.node_inputs] + [self.job_inputs] +
            [self.node_valid_mask] + [self.job_valid_mask] +
            self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats +
            [self.dag_summ_backward_map] + [self.node_act_vec] +
            [self.job_act_vec] + [self.adv],
            [node_inputs] + [job_inputs] +
            [node_valid_mask] + [job_valid_mask] +
            gcn_mats + gcn_masks +
            [summ_mats, running_dags_mat] +
            [dag_summ_backward_map] + [node_act_vec] +
            [job_act_vec] + [np.ones([1, 1], dtype=np.float32)])
        })

        adv_node_inputs = _fgsm_node_inputs(node_inputs, grad, self.fgsm_epsilon)
        adv_node_act_probs, adv_job_act_probs, node_acts, job_acts = self.predict(
            adv_node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map)

        self.attack_count += 1
        self.attack_abs_delta_sum += float(np.mean(np.abs(adv_node_inputs - node_inputs)))
        self.clean_target_prob_sum += float(node_act_probs[0, target_node] * job_act_probs[0, target_job, target_exec])
        self.adv_target_prob_sum += float(adv_node_act_probs[0, target_node] * adv_job_act_probs[0, target_job, target_exec])

        return node_acts, job_acts, \
               adv_node_act_probs, adv_job_act_probs, \
               adv_node_inputs, job_inputs, \
               node_valid_mask, job_valid_mask, \
               gcn_mats, gcn_masks, summ_mats, \
               running_dags_mat, dag_summ_backward_map, \
               exec_map, job_dags_changed


def _fgsm_node_inputs(node_inputs, grad, epsilon):
    adv = np.array(node_inputs, copy=True)
    if epsilon <= 0:
        return adv
    continuous_channels = [0, 2, 3, 4]
    adv[:, continuous_channels] += float(epsilon) * np.sign(grad[:, continuous_channels])
    adv[:, 1] = node_inputs[:, 1]
    exec_upper = float(args.exec_cap) / 20.0
    adv[:, 0] = np.clip(adv[:, 0], 0.0, exec_upper)
    adv[:, 2] = np.clip(adv[:, 2], 0.0, exec_upper)
    for channel in (3, 4):
        upper = max(float(np.max(node_inputs[:, channel])), float(np.max(adv[:, channel])), 0.0)
        adv[:, channel] = np.clip(adv[:, channel], 0.0, upper)
    return adv


def _install_tail_sampler(tail_weight):
    import spark_env.env as env_mod
    import spark_env.job_generator as job_gen
    from utils import OrderedSet

    sizes = list(args.tpch_size)
    scores = np.array([_size_score(size) for size in sizes], dtype=np.float64)
    if tail_weight == 0:
        probs = np.ones(len(sizes), dtype=np.float64) / float(len(sizes))
    else:
        normalized = scores / np.max(scores)
        weights = np.power(normalized, tail_weight)
        probs = weights / np.sum(weights)

    def generate_tpch_jobs_weighted(np_random, timeline, wall_time):
        job_dags = OrderedSet()
        t = 0
        for _ in range(args.num_init_dags):
            query_idx = str(np_random.randint(args.tpch_num) + 1)
            query_size = str(np_random.choice(sizes, p=probs))
            job_dag = job_gen.load_job(args.job_folder, query_size, query_idx, wall_time, np_random)
            job_dag.start_time = t
            job_dag.arrived = True
            job_dags.add(job_dag)

        for _ in range(args.num_stream_dags):
            t += int(np_random.exponential(args.stream_interval))
            query_size = str(np_random.choice(sizes, p=probs))
            query_idx = str(np_random.randint(args.tpch_num) + 1)
            job_dag = job_gen.load_job(args.job_folder, query_size, query_idx, wall_time, np_random)
            job_dag.start_time = t
            timeline.push(t, job_dag)
        return job_dags

    def generate_jobs_weighted(np_random, timeline, wall_time):
        if args.query_type != "tpch":
            raise ValueError("tail perturbation is implemented only for TPC-H jobs")
        return generate_tpch_jobs_weighted(np_random, timeline, wall_time)

    job_gen.generate_tpch_jobs = generate_tpch_jobs_weighted
    job_gen.generate_jobs = generate_jobs_weighted
    env_mod.generate_jobs = generate_jobs_weighted
    metadata = {
        "tail_weight": tail_weight,
        "tpch_sizes": sizes,
        "tpch_size_scores": scores.tolist(),
        "tpch_size_probabilities": probs.tolist(),
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def _install_lag(lag_lambda):
    from spark_env.task import Task

    expected_stage_duration = _expected_stage_duration()
    lag_mean = float(lag_lambda) * expected_stage_duration
    original_schedule = Task.schedule

    def schedule_with_lag(self, start_time, duration, executor):
        original_schedule(self, start_time, duration, executor)
        if lag_mean <= 0:
            return
        delay = _deterministic_exponential_lag(
            lag_mean,
            _LAG_STATE["seed"],
            self.node.job_dag.name,
            self.node.job_dag.start_time,
            self.node.idx,
            self.idx,
        )
        self.finish_time += delay
        self.duration += delay

    Task.schedule = schedule_with_lag
    metadata = {
        "lag_lambda": lag_lambda,
        "expected_stage_duration_ms": expected_stage_duration,
        "lag_mean_ms": lag_mean,
        "lag_semantics": "deterministic per-task exponential notification delay added to task completion events for both schedulers",
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def _expected_stage_duration():
    durations = []
    for size in args.tpch_size:
        query_path = os.path.join(args.job_folder, size)
        for query_idx in range(1, args.tpch_num + 1):
            path = os.path.join(query_path, "task_duration_{}.npy".format(query_idx))
            task_durations = np.load(path, allow_pickle=True).item()
            for task_duration in task_durations.values():
                values = []
                for bucket in ("first_wave", "rest_wave", "fresh_durations"):
                    for observed in task_duration[bucket].values():
                        values.extend(list(observed))
                if values:
                    durations.append(float(np.mean(values)))
    if not durations:
        raise RuntimeError("could not compute Decima expected stage duration")
    return float(np.mean(np.array(durations, dtype=np.float64)))


def _deterministic_exponential_lag(mean, seed, job_name, start_time, node_idx, task_idx):
    key = "{}|{}|{}|{}|{}|{}".format(seed, job_name, start_time, node_idx, task_idx, mean)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    u = (integer + 0.5) / float(2 ** 64)
    return float(-mean * np.log(max(1e-12, 1.0 - u)))


def _run_scheme(env, agent, scheme, exp, seed, progress_interval):
    _LAG_STATE["seed"] = int(seed)
    env.seed(seed)
    env.reset()
    obs = env.observe()
    total_reward = 0.0
    done = False
    decisions = 0
    t0 = time.time()
    while not done:
        node, use_exec = agent.get_action(obs)
        obs, reward, done = env.step(node, use_exec)
        total_reward += float(reward)
        decisions += 1
        if progress_interval > 0 and decisions % progress_interval == 0:
            print(
                "scheme={} exp={} decision_steps={} wall_time={}".format(
                    scheme, exp, decisions, env.wall_time.curr_time
                ),
                flush=True,
            )

    durations = np.array(
        [job.completion_time - job.start_time for job in env.finished_job_dags],
        dtype=np.float64,
    )
    completion_times = np.array([job.completion_time for job in env.finished_job_dags], dtype=np.float64)
    return {
        "scheme": scheme,
        "exp": int(exp),
        "seed": int(seed),
        "num_jobs": int(len(durations)),
        "decision_steps": int(decisions),
        "total_reward": float(total_reward),
        "mean_jct": float(np.mean(durations)),
        "median_jct": float(np.median(durations)),
        "p95_jct": float(np.percentile(durations, 95)),
        "reward_implied_mean_jct": float(-total_reward * args.reward_scale / max(len(durations), 1)),
        "makespan": float(np.max(completion_times)),
        "elapsed_seconds": float(time.time() - t0),
    }


def _payload(
    rows,
    *,
    perturbation,
    tail_weight,
    lag_lambda,
    fgsm_epsilon,
    bootstrap_seed,
    started_at,
    perturbation_metadata,
    learn_agent,
):
    paired = _paired_rows(rows)
    deltas = np.array(
        [pair["dynamic_partition_mean_jct"] - pair["learn_mean_jct"] for pair in paired],
        dtype=np.float64,
    )
    ci_low, ci_high = _bootstrap_ci(deltas, seed=bootstrap_seed)
    p_less, p_greater = _sign_flip_pvalues(deltas, seed=bootstrap_seed + 1)
    aggregate = {
        "dynamic_partition_mean_jct": _scheme_mean(rows, "dynamic_partition", "mean_jct"),
        "learn_mean_jct": _scheme_mean(rows, "learn", "mean_jct"),
        "delta_mean": float(np.mean(deltas)) if len(deltas) else float("nan"),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "p_less": p_less,
        "p_greater": p_greater,
        "num_pairs": int(len(deltas)),
        "prediction_confirmed": bool(ci_high < 0.0) if len(deltas) else False,
    }
    return {
        "method": "decima",
        "protocol_amendment": "PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md",
        "perturbation": perturbation,
        "tail_weight": tail_weight,
        "lag_lambda": lag_lambda,
        "fgsm_epsilon": fgsm_epsilon,
        "anchor_weight": 0.5,
        "anchor_lag_lambda": 1.0,
        "anchor_fgsm_epsilon": 0.05,
        "delta_definition": "mean_JCT(dynamic_partition)-mean_JCT(Decima)",
        "perturbation_metadata": _attack_metadata(perturbation_metadata, learn_agent),
        "args": {
            "exec_cap": args.exec_cap,
            "num_init_dags": args.num_init_dags,
            "num_stream_dags": args.num_stream_dags,
            "num_exp": args.num_exp,
            "num_ep_seed_base": args.num_ep,
            "saved_model": args.saved_model,
        },
        "rows": rows,
        "paired": paired,
        "aggregate": aggregate,
        "elapsed_seconds_total": float(time.time() - started_at),
    }


def _attack_metadata(metadata, learn_agent):
    out = dict(metadata)
    if hasattr(learn_agent, "attack_count"):
        count = max(int(learn_agent.attack_count), 1)
        out.update(
            {
                "fgsm_attack_count": int(learn_agent.attack_count),
                "fgsm_mean_abs_node_feature_delta": float(learn_agent.attack_abs_delta_sum / count),
                "fgsm_mean_clean_target_prob": float(learn_agent.clean_target_prob_sum / count),
                "fgsm_mean_adv_target_prob": float(learn_agent.adv_target_prob_sum / count),
            }
        )
    return out


def _paired_rows(rows):
    by_exp = {}
    for row in rows:
        by_exp.setdefault(row["exp"], {})[row["scheme"]] = row
    paired = []
    for exp in sorted(by_exp):
        if "dynamic_partition" not in by_exp[exp] or "learn" not in by_exp[exp]:
            continue
        dp = by_exp[exp]["dynamic_partition"]
        learn = by_exp[exp]["learn"]
        paired.append(
            {
                "exp": int(exp),
                "seed": int(dp["seed"]),
                "dynamic_partition_mean_jct": float(dp["mean_jct"]),
                "learn_mean_jct": float(learn["mean_jct"]),
                "delta": float(dp["mean_jct"] - learn["mean_jct"]),
            }
        )
    return paired


def _bootstrap_ci(deltas, seed, resamples=5000):
    if len(deltas) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _sign_flip_pvalues(deltas, seed, resamples=100000):
    if len(deltas) == 0:
        return float("nan"), float("nan")
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(deltas)))
    null_means = (signs * deltas).mean(axis=1)
    return (
        float((np.count_nonzero(null_means <= observed) + 1.0) / (resamples + 1.0)),
        float((np.count_nonzero(null_means >= observed) + 1.0) / (resamples + 1.0)),
    )


def _scheme_mean(rows, scheme, key):
    values = [row[key] for row in rows if row["scheme"] == scheme]
    return float(np.mean(np.array(values, dtype=np.float64))) if values else float("nan")


def _size_score(size):
    if size.endswith("g"):
        return float(size[:-1])
    return float(size)


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        "scheme",
        "exp",
        "seed",
        "num_jobs",
        "decision_steps",
        "total_reward",
        "mean_jct",
        "median_jct",
        "p95_jct",
        "reward_implied_mean_jct",
        "makespan",
        "elapsed_seconds",
    ]
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


if __name__ == "__main__":
    main()
