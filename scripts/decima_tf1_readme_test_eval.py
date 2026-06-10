#!/usr/bin/env python
"""Metrics-producing Decima README test loop.

This script is executed inside the official TensorFlow 1.x Docker image with
the working directory set to ``external/decima-sim``. It intentionally mirrors
the control flow in the official ``test.py`` while writing numeric metrics for
the reproduction gate.
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

from actor_agent import ActorAgent
from heuristic_agent import DynamicPartitionAgent
from param import args
from spark_agent import SparkAgent
from spark_env.canvas import visualize_dag_time_save_pdf, visualize_executor_usage
from spark_env.env import Environment


def main():
    metrics_output = os.environ.get(
        "DECIMA_METRICS_OUTPUT",
        "/workspace/results/paper/decima/tables/decima_official_readme_gate.json",
    )
    csv_output = os.environ.get(
        "DECIMA_METRICS_CSV",
        "/workspace/results/paper/decima/tables/decima_official_readme_gate_per_exp.csv",
    )
    write_visualizations = os.environ.get("DECIMA_WRITE_VISUALIZATIONS", "0") == "1"
    progress_interval = int(os.environ.get("DECIMA_PROGRESS_INTERVAL_STEPS", "5000"))

    output_dir = os.path.dirname(metrics_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_dir = os.path.dirname(csv_output)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    tf.set_random_seed(args.seed)
    env = Environment()
    agents = {}

    for scheme in args.test_schemes:
        if scheme == "learn":
            sess = tf.Session()
            agents[scheme] = ActorAgent(
                sess,
                args.node_input_dim,
                args.job_input_dim,
                args.hid_dims,
                args.output_dim,
                args.max_depth,
                range(1, args.exec_cap + 1),
            )
        elif scheme == "dynamic_partition":
            agents[scheme] = DynamicPartitionAgent()
        elif scheme == "spark_fifo":
            agents[scheme] = SparkAgent(exec_cap=args.exec_cap)
        else:
            raise ValueError("scheme {} not recognized".format(scheme))

    rows = []
    started_at = time.time()
    for exp in range(args.num_exp):
        print("Experiment {} of {}".format(exp + 1, args.num_exp), flush=True)
        for scheme in args.test_schemes:
            seed = args.num_ep + exp
            env.seed(seed)
            env.reset()
            agent = agents[scheme]
            obs = env.observe()
            total_reward = 0.0
            done = False
            decision_steps = 0
            scheme_start = time.time()

            while not done:
                node, use_exec = agent.get_action(obs)
                obs, reward, done = env.step(node, use_exec)
                total_reward += reward
                decision_steps += 1
                if progress_interval > 0 and decision_steps % progress_interval == 0:
                    print(
                        "scheme={} exp={} decision_steps={} wall_time={}".format(
                            scheme, exp, decision_steps, env.wall_time.curr_time
                        ),
                        flush=True,
                    )

            row = _summarize_finished_jobs(
                scheme=scheme,
                exp=exp,
                seed=seed,
                total_reward=total_reward,
                decision_steps=decision_steps,
                elapsed_seconds=time.time() - scheme_start,
                finished_job_dags=list(env.finished_job_dags),
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            _write_csv(csv_output, rows)
            _write_json(metrics_output, _payload(rows, started_at))

            if write_visualizations:
                if args.canvs_visualization:
                    visualize_dag_time_save_pdf(
                        env.finished_job_dags,
                        env.executors,
                        args.result_folder
                        + "visualization_exp_{}_scheme_{}.png".format(exp, scheme),
                        plot_type="app",
                    )
                else:
                    visualize_executor_usage(
                        env.finished_job_dags,
                        args.result_folder
                        + "visualization_exp_{}_scheme_{}.png".format(exp, scheme),
                    )

    _write_csv(csv_output, rows)
    _write_json(metrics_output, _payload(rows, started_at))


def _summarize_finished_jobs(
    scheme, exp, seed, total_reward, decision_steps, elapsed_seconds, finished_job_dags
):
    flow_times = np.array(
        [job.completion_time - job.start_time for job in finished_job_dags],
        dtype=np.float64,
    )
    start_times = np.array([job.start_time for job in finished_job_dags], dtype=np.float64)
    completion_times = np.array(
        [job.completion_time for job in finished_job_dags], dtype=np.float64
    )
    num_jobs = int(len(flow_times))
    reward_implied_mean_jct = (
        float(-total_reward * args.reward_scale / num_jobs) if num_jobs > 0 else float("nan")
    )
    return {
        "scheme": scheme,
        "exp": int(exp),
        "seed": int(seed),
        "num_jobs": num_jobs,
        "decision_steps": int(decision_steps),
        "total_reward": float(total_reward),
        "mean_jct": float(np.mean(flow_times)) if num_jobs else float("nan"),
        "median_jct": float(np.median(flow_times)) if num_jobs else float("nan"),
        "p95_jct": float(np.percentile(flow_times, 95)) if num_jobs else float("nan"),
        "reward_implied_mean_jct": reward_implied_mean_jct,
        "makespan": float(np.max(completion_times)) if num_jobs else float("nan"),
        "first_start_time": float(np.min(start_times)) if num_jobs else float("nan"),
        "last_completion_time": float(np.max(completion_times)) if num_jobs else float("nan"),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _payload(rows, started_at):
    aggregate = {}
    for scheme in sorted(set(row["scheme"] for row in rows)):
        scheme_rows = [row for row in rows if row["scheme"] == scheme]
        aggregate[scheme] = {
            "num_exp": len(scheme_rows),
            "mean_total_reward": _mean([row["total_reward"] for row in scheme_rows]),
            "mean_jct": _mean([row["mean_jct"] for row in scheme_rows]),
            "median_jct": _mean([row["median_jct"] for row in scheme_rows]),
            "p95_jct": _mean([row["p95_jct"] for row in scheme_rows]),
            "mean_num_jobs": _mean([row["num_jobs"] for row in scheme_rows]),
            "mean_decision_steps": _mean([row["decision_steps"] for row in scheme_rows]),
            "mean_elapsed_seconds": _mean([row["elapsed_seconds"] for row in scheme_rows]),
        }

    gate = {}
    if "dynamic_partition" in aggregate and "learn" in aggregate:
        dp_jct = aggregate["dynamic_partition"]["mean_jct"]
        learn_jct = aggregate["learn"]["mean_jct"]
        dp_reward = aggregate["dynamic_partition"]["mean_total_reward"]
        learn_reward = aggregate["learn"]["mean_total_reward"]
        observed_improvement_pct = 100.0 * (dp_jct - learn_jct) / dp_jct
        reward_improvement_pct = 100.0 * (learn_reward - dp_reward) / abs(dp_reward)
        target_improvement_pct = 21.0
        relative_error = abs(observed_improvement_pct - target_improvement_pct) / target_improvement_pct
        gate = {
            "comparison": "learn_vs_dynamic_partition",
            "metric": "mean_jct",
            "dynamic_partition_mean_jct": dp_jct,
            "learn_mean_jct": learn_jct,
            "observed_improvement_pct": observed_improvement_pct,
            "reward_improvement_pct": reward_improvement_pct,
            "target_improvement_pct": target_improvement_pct,
            "relative_error_to_target": relative_error,
            "within_15pct_relative_of_target": bool(relative_error <= 0.15),
            "learn_beats_dynamic_partition": bool(learn_jct < dp_jct),
            "gate_passed": bool(learn_jct < dp_jct and relative_error <= 0.15),
            "gate_note": "README does not publish a numeric reference; v2.2 uses the brief's 21 pct improvement target with 15 pct relative tolerance.",
        }

    return {
        "args": {
            "exec_cap": args.exec_cap,
            "num_init_dags": args.num_init_dags,
            "num_stream_dags": args.num_stream_dags,
            "num_exp": args.num_exp,
            "num_ep_seed_base": args.num_ep,
            "test_schemes": list(args.test_schemes),
            "saved_model": args.saved_model,
            "canvs_visualization": args.canvs_visualization,
            "visualizations_written": os.environ.get("DECIMA_WRITE_VISUALIZATIONS", "0")
            == "1",
        },
        "rows": rows,
        "aggregate": aggregate,
        "gate": gate,
        "elapsed_seconds_total": float(time.time() - started_at),
    }


def _mean(values):
    return float(np.mean(np.array(values, dtype=np.float64))) if values else float("nan")


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
        "first_start_time",
        "last_completion_time",
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


if __name__ == "__main__":
    main()
