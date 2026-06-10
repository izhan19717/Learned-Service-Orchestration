#!/usr/bin/env python
"""Decima TF1 evaluator with an SRTF-style comparator.

This script runs inside the official Decima TensorFlow 1.x Docker image with
working directory ``external/decima-sim``. It is a post-hoc sensitivity around
the amended official-simulator Decima results, not an official Graphene claim.
"""

from __future__ import print_function

import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("agg")
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/workspace/scripts")

from actor_agent import ActorAgent
from agent import Agent
from param import args
from spark_env.env import Environment

from decima_tf1_perturb_eval import (
    AdversarialActorAgent,
    _attack_metadata,
    _bootstrap_ci,
    _ensure_parent,
    _install_lag,
    _install_tail_sampler,
    _run_scheme,
    _sign_flip_pvalues,
    _write_csv,
    _write_json,
)


class SRTFAgent(Agent):
    """Dependency-aware shortest-remaining-work comparator.

    The policy chooses the arrived job DAG with smallest estimated remaining
    work, then the ready node with smallest estimated remaining node work. It is
    work-conservative over the currently available source executors and does not
    preempt already-running tasks.
    """

    def __init__(self):
        Agent.__init__(self)

    def get_action(self, obs):
        job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, \
            exec_commit, moving_executors, action_map = obs

        if len(frontier_nodes) == 0:
            return None, num_source_exec

        candidates = []
        for node in frontier_nodes:
            assignable = _assignable_tasks(node, exec_commit, moving_executors)
            if assignable <= 0:
                continue
            job_work = _remaining_job_work(node.job_dag, exec_commit, moving_executors)
            node_work = _remaining_node_work(node, exec_commit, moving_executors)
            start_time = node.job_dag.start_time if node.job_dag.start_time is not None else 0
            candidates.append((job_work, node_work, start_time, node.idx, node, assignable))

        if not candidates:
            return None, num_source_exec

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        _, _, _, _, node, assignable = candidates[0]
        use_exec = min(assignable, num_source_exec)
        if use_exec <= 0:
            return None, num_source_exec
        return node, use_exec


def main():
    output_json = os.environ["DECIMA_SRTF_OUTPUT_JSON"]
    output_csv = os.environ["DECIMA_SRTF_OUTPUT_CSV"]
    perturbation = os.environ.get("DECIMA_PERTURBATION", "clean")
    tail_weight = float(os.environ.get("DECIMA_TAIL_WEIGHT", "0.0"))
    lag_lambda = float(os.environ.get("DECIMA_LAG_LAMBDA", "0.0"))
    fgsm_epsilon = float(os.environ.get("DECIMA_FGSM_EPSILON", "0.0"))
    bootstrap_seed = int(os.environ.get("DECIMA_BOOTSTRAP_SEED", "123017"))
    progress_interval = int(os.environ.get("DECIMA_PROGRESS_INTERVAL_STEPS", "0"))

    if perturbation not in ("clean", "tail", "lag", "fgsm"):
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
        "srtf": SRTFAgent(),
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
        for scheme in ("srtf", "learn"):
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
    deltas = np.array([pair["srtf_mean_jct"] - pair["learn_mean_jct"] for pair in paired], dtype=np.float64)
    ci_low, ci_high = _bootstrap_ci(deltas, seed=bootstrap_seed)
    p_less, p_greater = _sign_flip_pvalues(deltas, seed=bootstrap_seed + 1)
    aggregate = {
        "srtf_mean_jct": _scheme_mean(rows, "srtf", "mean_jct"),
        "learn_mean_jct": _scheme_mean(rows, "learn", "mean_jct"),
        "delta_mean": float(np.mean(deltas)) if len(deltas) else float("nan"),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "p_less": p_less,
        "p_greater": p_greater,
        "num_pairs": int(len(deltas)),
        "srtf_beats_decima": bool(ci_high < 0.0) if len(deltas) else False,
        "decima_beats_srtf": bool(ci_low > 0.0) if len(deltas) else False,
    }
    return {
        "method": "decima",
        "scope": "post_hoc_srtf_style_comparator_sensitivity",
        "protocol_amendment": "PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md",
        "comparator": "SRTF-style dependency-aware shortest remaining work",
        "perturbation": perturbation,
        "tail_weight": tail_weight,
        "lag_lambda": lag_lambda,
        "fgsm_epsilon": fgsm_epsilon,
        "delta_definition": "mean_JCT(SRTF)-mean_JCT(Decima)",
        "delta_interpretation": "negative means SRTF beats Decima; positive means Decima beats SRTF",
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


def _paired_rows(rows):
    by_exp = {}
    for row in rows:
        by_exp.setdefault(row["exp"], {})[row["scheme"]] = row
    paired = []
    for exp in sorted(by_exp):
        if "srtf" not in by_exp[exp] or "learn" not in by_exp[exp]:
            continue
        srtf = by_exp[exp]["srtf"]
        learn = by_exp[exp]["learn"]
        paired.append(
            {
                "exp": int(exp),
                "seed": int(srtf["seed"]),
                "srtf_mean_jct": float(srtf["mean_jct"]),
                "learn_mean_jct": float(learn["mean_jct"]),
                "delta": float(srtf["mean_jct"] - learn["mean_jct"]),
            }
        )
    return paired


def _scheme_mean(rows, scheme, key):
    values = [row[key] for row in rows if row["scheme"] == scheme]
    return float(np.mean(np.array(values, dtype=np.float64))) if values else float("nan")


def _assignable_tasks(node, exec_commit, moving_executors):
    return max(
        0,
        node.num_tasks
        - node.next_task_idx
        - exec_commit.node_commit[node]
        - moving_executors.count(node),
    )


def _remaining_node_work(node, exec_commit, moving_executors):
    remaining_tasks = max(
        0,
        node.num_tasks
        - node.num_finished_tasks
        - exec_commit.node_commit[node]
        - moving_executors.count(node),
    )
    return float(remaining_tasks) * _rough_task_duration(node)


def _remaining_job_work(job_dag, exec_commit, moving_executors):
    work = 0.0
    for node in job_dag.nodes:
        if node.tasks_all_done:
            continue
        work += _remaining_node_work(node, exec_commit, moving_executors)
    return float(work)


def _rough_task_duration(node):
    if node.num_tasks <= 0:
        return 0.0
    return float(node.tasks[-1].duration)


if __name__ == "__main__":
    main()
