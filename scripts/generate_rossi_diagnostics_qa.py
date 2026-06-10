#!/usr/bin/env python3
"""Generate Rossi diagnostic answers for methodology QA."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.tracking import start_run, write_json_artifact
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ModelBasedController, ThresholdHPAController
from cisose_rossi.evaluation import metrics
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_rossi_v2_2"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
HORIZON = DEFAULT_CONFIG.time_limit + 1


class TrackingModelBasedController(ModelBasedController):
    def __init__(self, *, block_size: int = 100):
        super().__init__(DEFAULT_CONFIG)
        self.block_size = block_size
        self._ticks = 0
        self._block_start_q = self.q.copy()
        self.block_l1: list[dict[str, float | int]] = []

    def update(self, service, cost, input_rate) -> None:
        super().update(service, cost, input_rate)
        self._ticks += 1
        if self._ticks % self.block_size == 0:
            self._finish_block()

    def finalize(self) -> None:
        if self._ticks % self.block_size != 0:
            self._finish_block()

    def _finish_block(self) -> None:
        start = self._ticks - (self._ticks % self.block_size or self.block_size)
        self.block_l1.append(
            {
                "block_start_tick": int(start),
                "block_end_tick": int(self._ticks - 1),
                "l1_q_change": float(np.sum(np.abs(self.q - self._block_start_q))),
            }
        )
        self._block_start_q = self.q.copy()


def load_result(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def delta_summary(name: str, deltas: list[float]) -> dict[str, object]:
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "cell": name,
        "n_windows": int(len(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "rossi_beats_hpa_windows": int(np.sum(arr > 0.0)),
        "rossi_beats_hpa_fraction": float(np.mean(arr > 0.0)),
        "deltas": [float(x) for x in arr],
    }


def collect_delta_summaries() -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    p1 = load_result("results/rossi/p1_online_observation_lag_sweep.json")
    p2 = load_result("results/rossi/p2_online_service_tail.json")
    p3 = load_result("results/rossi/p3_online_bucket_flip.json")
    cells = {
        "clean_k0_alpha_inf_epsilon0": p2["p2"]["cells"][0],
        "p1_k10": next(c for c in p1["p1"]["cells"] if c["lag"] == 10),
        "p2_alpha_1_5": next(c for c in p2["p2"]["cells"] if c["value"] == "1.5"),
        "p3_epsilon_0_05": next(c for c in p3["p3"]["cells"] if float(c["value"]) == 0.05),
    }
    summaries = []
    delta_vectors = {}
    for name, cell in cells.items():
        deltas = [
            float(hpa) - float(rossi)
            for hpa, rossi in zip(cell["hpa_total_costs"], cell["rossi_total_costs"], strict=True)
        ]
        summaries.append(delta_summary(name, deltas))
        delta_vectors[name] = deltas
    return summaries, delta_vectors


def write_delta_table(summaries: list[dict[str, object]]) -> None:
    table_dir = ROOT / "results" / "paper" / "rossi" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "rossi_per_window_delta_diagnostics.csv"
    fields = [
        "cell",
        "n_windows",
        "min",
        "q25",
        "median",
        "q75",
        "max",
        "mean",
        "rossi_beats_hpa_windows",
        "rossi_beats_hpa_fraction",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")


def run_trace_for_window(offset: int, *, lag: int) -> dict[str, object]:
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    rates = tuple(sequence[offset : offset + HORIZON])
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        ModelBasedController(DEFAULT_CONFIG),
        rates,
        horizon=HORIZON,
        observation_lag_steps=lag,
        observation_applies_to_update=True,
    )
    hpa_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=HORIZON,
        observation_lag_steps=lag,
        observation_applies_to_update=True,
    )
    fig_dir = ROOT / "results" / "paper" / "rossi" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / "rossi_p1_k10_median_window_failure_trace.pdf"
    png_path = fig_dir / "rossi_p1_k10_median_window_failure_trace.png"
    t = np.arange(len(hpa_records))
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(6.7, 5.2), sharex=True)
    axes[0].plot(t, [r.replicas_before for r in hpa_records], color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("HPA replicas")
    axes[1].plot(t, [r.utilization for r in hpa_records], label="true", color="#333333", linewidth=0.7)
    axes[1].plot(
        t,
        [r.observed_utilization for r in hpa_records],
        label="observed",
        color="#d62728",
        linewidth=0.7,
        alpha=0.85,
    )
    axes[1].set_ylabel("HPA util.")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    axes[2].plot(t, [r.replicas_before for r in rossi_records], color="#1f77b4", linewidth=0.8)
    axes[2].set_ylabel("Rossi replicas")
    axes[3].plot(
        t,
        [r.utilization for r in rossi_records],
        label="true",
        color="#333333",
        linewidth=0.7,
    )
    axes[3].plot(
        t,
        [r.observed_utilization for r in rossi_records],
        label="observed",
        color="#d62728",
        linewidth=0.7,
        alpha=0.85,
    )
    axes[3].set_ylabel("Rossi util.")
    axes[3].set_xlabel("Decision tick")
    axes[3].legend(frameon=False, ncol=2, loc="upper right")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    mlflow.log_artifact(str(pdf_path), artifact_path="paper/figures")
    mlflow.log_artifact(str(png_path), artifact_path="paper/figures")
    return {
        "offset": int(offset),
        "lag": int(lag),
        "rossi_total_cost": metrics(rossi_records).total_cost,
        "hpa_total_cost": metrics(hpa_records).total_cost,
        "delta_hpa_minus_rossi": metrics(hpa_records).total_cost - metrics(rossi_records).total_cost,
        "hpa_replica_min": int(min(r.replicas_before for r in hpa_records)),
        "hpa_replica_median": float(np.median([r.replicas_before for r in hpa_records])),
        "hpa_replica_max": int(max(r.replicas_before for r in hpa_records)),
        "rossi_replica_min": int(min(r.replicas_before for r in rossi_records)),
        "rossi_replica_median": float(np.median([r.replicas_before for r in rossi_records])),
        "rossi_replica_max": int(max(r.replicas_before for r in rossi_records)),
        "hpa_mean_abs_observation_delta": float(
            np.mean(np.abs([r.observation_delta for r in hpa_records]))
        ),
        "rossi_mean_abs_observation_delta": float(
            np.mean(np.abs([r.observation_delta for r in rossi_records]))
        ),
        "figure_pdf": str(pdf_path.relative_to(ROOT)),
        "figure_png": str(png_path.relative_to(ROOT)),
    }


def run_q_convergence(offset: int) -> dict[str, object]:
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    rates = tuple(sequence[offset : offset + HORIZON])
    controller = TrackingModelBasedController(block_size=100)
    records = RladSimulator(DEFAULT_CONFIG).run(controller, rates, horizon=HORIZON)
    controller.finalize()
    final_records = records[-100:]
    counts = Counter(r.action_label for r in final_records)
    probs = np.asarray([count / len(final_records) for count in counts.values()], dtype=np.float64)
    entropy_nats = float(-np.sum(probs * np.log(probs)))
    entropy_bits = float(entropy_nats / np.log(2.0))
    fig_dir = ROOT / "results" / "paper" / "rossi" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / "rossi_clean_q_l1_convergence_diagnostic.pdf"
    png_path = fig_dir / "rossi_clean_q_l1_convergence_diagnostic.png"
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    xs = [row["block_end_tick"] for row in controller.block_l1]
    ys = [row["l1_q_change"] for row in controller.block_l1]
    ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.1, color="#1f77b4")
    ax.set_xlabel("Block end tick")
    ax.set_ylabel("L1(Q end - Q start)")
    ax.set_title("Rossi clean Q-table change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    mlflow.log_artifact(str(pdf_path), artifact_path="paper/figures")
    mlflow.log_artifact(str(png_path), artifact_path="paper/figures")
    table_dir = ROOT / "results" / "paper" / "rossi" / "tables"
    csv_path = table_dir / "rossi_clean_q_l1_convergence_diagnostic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["block_start_tick", "block_end_tick", "l1_q_change"],
        )
        writer.writeheader()
        writer.writerows(controller.block_l1)
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    return {
        "offset": int(offset),
        "rossi_total_cost": metrics(records).total_cost,
        "block_size": 100,
        "block_l1_first": controller.block_l1[0],
        "block_l1_median": float(np.median([row["l1_q_change"] for row in controller.block_l1])),
        "block_l1_last": controller.block_l1[-1],
        "block_l1_rows": controller.block_l1,
        "final_100_tick_action_counts": dict(counts),
        "final_100_tick_entropy_nats": entropy_nats,
        "final_100_tick_entropy_bits": entropy_bits,
        "figure_pdf": str(pdf_path.relative_to(ROOT)),
        "figure_png": str(png_path.relative_to(ROOT)),
        "csv": str(csv_path.relative_to(ROOT)),
    }


def first_window_costs() -> dict[str, object]:
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=HORIZON)
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        ModelBasedController(DEFAULT_CONFIG),
        sequence,
        horizon=HORIZON,
    )
    hpa_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        sequence,
        horizon=HORIZON,
    )
    return {
        "offset": 0,
        "horizon": HORIZON,
        "rossi_total_cost": metrics(rossi_records).total_cost,
        "hpa_total_cost": metrics(hpa_records).total_cost,
        "delta_hpa_minus_rossi": metrics(hpa_records).total_cost - metrics(rossi_records).total_cost,
    }


def main() -> None:
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="rossi-methodology-qa-diagnostics",
        role="diagnostic",
        params={
            "method": "rossi_rlad",
            "rlad_repo_url": RLAD_REPO_URL,
            "rlad_commit": RLAD_COMMIT,
            "profile_sha256": PROFILE_SHA256,
            "horizon": HORIZON,
        },
    ) as run:
        summaries, delta_vectors = collect_delta_summaries()
        write_delta_table(summaries)
        p1_offsets = load_result("results/rossi/p1_online_observation_lag_sweep.json")["offsets"]
        p1_deltas = np.asarray(delta_vectors["p1_k10"], dtype=np.float64)
        median_idx = int(np.argsort(np.abs(p1_deltas - np.median(p1_deltas)))[0])
        median_trace = run_trace_for_window(p1_offsets[median_idx], lag=10)
        clean_deltas = np.asarray(delta_vectors["clean_k0_alpha_inf_epsilon0"], dtype=np.float64)
        clean_median_idx = int(np.argsort(np.abs(clean_deltas - np.median(clean_deltas)))[0])
        q_convergence = run_q_convergence(p1_offsets[clean_median_idx])
        result = {
            "mlflow_run_id": run.info.run_id,
            "per_window_delta_summaries": summaries,
            "table_i_first_window": first_window_costs(),
            "evaluation_offsets": p1_offsets,
            "table_i_first_window_in_evaluation_offsets": 0 in p1_offsets,
            "p1_k10_median_delta_window": {
                "seed_index": median_idx,
                "offset": p1_offsets[median_idx],
                "delta_hpa_minus_rossi": float(p1_deltas[median_idx]),
                **median_trace,
            },
            "clean_median_delta_q_convergence_window": {
                "seed_index": clean_median_idx,
                "offset": p1_offsets[clean_median_idx],
                "delta_hpa_minus_rossi": float(clean_deltas[clean_median_idx]),
                **q_convergence,
            },
            "metric_definitions": {
                "delta_hpa_minus_rossi": "total_cost(HPA) - total_cost(Rossi)",
                "mean_abs_observation_delta": (
                    "mean over ticks of abs(observed_utilization - true_current_utilization) "
                    "within that controller's own closed-loop trajectory"
                ),
                "q_l1_block_change": (
                    "For each 100-tick block, sum(abs(Q_after_block - Q_before_block)) "
                    "after online Bellman updates."
                ),
                "final_action_entropy": (
                    "Shannon entropy over action labels selected in the final 100 ticks."
                ),
            },
        }
        mlflow.log_metric("p1_k10_median_trace_delta", median_trace["delta_hpa_minus_rossi"])
        mlflow.log_metric("q_final_100_tick_entropy_bits", q_convergence["final_100_tick_entropy_bits"])
        write_json_artifact(
            ROOT / "results" / "rossi" / "methodology_qa_diagnostics.json",
            result,
            run_id=run.info.run_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
