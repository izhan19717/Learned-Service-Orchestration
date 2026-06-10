#!/usr/bin/env python3
"""Generate DeepRM paper figures and tables from locked result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_deeprm.tracking import start_tracked_run, write_json_with_run_id


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "evaluation" / "deeprm"
TRAIN_DIR = ROOT / "results" / "training" / "author_source_rescue"
OUT_DIR = ROOT / "results" / "paper" / "deeprm"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean = _read_json(EVAL_DIR / "clean_load_0.7.json")
    source = _read_json(EVAL_DIR / "author_source_clean_load_0.7.json")
    sweeps = _read_json(EVAL_DIR / "perturbation_sweeps_v2_2.json")
    curve = _read_training_curve()

    with start_tracked_run(
        run_name="deeprm-paper-artifacts",
        role="paper-artifact-generation",
        root=ROOT,
        params={
            "clean_eval_run_id": clean.get("mlflow_run_id"),
            "source_eval_run_id": source.get("mlflow_run_id"),
            "perturbation_run_id": sweeps.get("mlflow_run_id"),
            "checkpoint": sweeps["summary"]["checkpoint"],
            "checkpoint_sha256": sweeps["summary"]["checkpoint_sha256"],
        },
    ) as run:
        figure_paths = [
            *_plot_training_curve(curve),
            *_plot_clean_gate(clean, source),
            *_plot_perturbation_sweeps(sweeps),
        ]
        table_paths = [
            *_write_clean_gate_tables(clean, source),
            *_write_prediction_tables(sweeps),
        ]
        manifest = {
            "status": "generated",
            "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
            "tables": [str(path.relative_to(ROOT)) for path in table_paths],
            "inputs": {
                "clean_gate": str((EVAL_DIR / "clean_load_0.7.json").relative_to(ROOT)),
                "author_source_gate": str((EVAL_DIR / "author_source_clean_load_0.7.json").relative_to(ROOT)),
                "perturbation_sweeps": str((EVAL_DIR / "perturbation_sweeps_v2_2.json").relative_to(ROOT)),
                "training_curve": str(TRAIN_DIR.relative_to(ROOT)),
            },
        }
        out = OUT_DIR / "deeprm_paper_artifact_manifest.json"
        write_json_with_run_id(out, manifest, run.info.run_id)
        for path in figure_paths:
            mlflow.log_artifact(str(path), artifact_path="paper/figures")
        for path in table_paths:
            mlflow.log_artifact(str(path), artifact_path="paper/tables")
        mlflow.log_metric("paper_artifacts.num_figures", len(figure_paths))
        mlflow.log_metric("paper_artifacts.num_tables", len(table_paths))
        for pred, key in {
            "p1": "P1_lag_10",
            "p2": "P2_tail_1.5",
            "p3": "P3_epsilon_0.05",
        }.items():
            comp = sweeps["cells"][key]["comparison"]
            mlflow.log_metric(f"deeprm.{pred}.delta_mean", comp["mean_difference"])
            mlflow.log_metric(f"deeprm.{pred}.ci_low", comp["ci_low"])
            mlflow.log_metric(f"deeprm.{pred}.ci_high", comp["ci_high"])
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(manifest, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_training_curve() -> list[dict[str, float]]:
    rows_by_iteration: dict[int, dict[str, float]] = {}
    for path in sorted(TRAIN_DIR.glob("load_0.7_curve*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_by_iteration[int(row["iteration"])] = row
    return [rows_by_iteration[idx] for idx in sorted(rows_by_iteration)]


def _plot_training_curve(curve: list[dict[str, float]]) -> list[Path]:
    x = np.asarray([row["iteration"] for row in curve], dtype=float)
    reward = np.asarray([row["mean_episode_reward"] for row in curve], dtype=float)
    steps = np.asarray([row["mean_episode_steps"] for row in curve], dtype=float)
    capped = np.asarray([row.get("capped_episode_fraction", 0.0) for row in curve], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.2), sharex=True)
    axes[0].plot(x, reward, color="#1f4e79", linewidth=1.2)
    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("DeepRM Author-Source Training")
    axes[1].plot(x, steps, color="#7f3c8d", linewidth=1.2)
    axes[1].set_ylabel("Mean steps")
    axes[2].plot(x, capped, color="#b04a1a", linewidth=1.2)
    axes[2].set_ylabel("Capped fraction")
    axes[2].set_xlabel("Training iteration")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, "deeprm_training_curve")


def _plot_clean_gate(clean: dict[str, object], source: dict[str, object]) -> list[Path]:
    method_means = clean["summary"]["method_means"]
    methods = ["DeepRM", "SJF", "Packer", "Tetris*"]
    values = [method_means[name] for name in methods]
    colors = ["#1f4e79", "#666666", "#999999", "#b04a1a"]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bars = ax.bar(methods, values, color=colors)
    ax.set_ylabel("Mean slowdown")
    ax.set_title("Strict Clean Gate at lambda = 0.7")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    source_summary = source["summary"]
    source_note = (
        "Author-source gate: DeepRM "
        f"{source_summary['DeepRM']['mean_slowdown_finished']:.1f}, "
        "SJF "
        f"{source_summary['SJF']['mean_slowdown_finished']:.1f}, "
        "SourceTetris "
        f"{source_summary['SourceTetris']['mean_slowdown_finished']:.1f}"
    )
    ax.text(0.0, -0.28, source_note, transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, "deeprm_clean_gate")


def _plot_perturbation_sweeps(sweeps: dict[str, object]) -> list[Path]:
    cells = sweeps["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))

    _plot_numeric_sweep(
        axes[0],
        cells,
        "P1_lag_",
        [0, 1, 2, 5, 10, 20],
        "Observation lag k",
        anchor=10,
    )
    _plot_tail_sweep(axes[1], cells)
    _plot_numeric_sweep(
        axes[2],
        cells,
        "P3_epsilon_",
        [0.0, 0.01, 0.02, 0.05, 0.10],
        "FGSM epsilon",
        anchor=0.05,
    )

    axes[0].set_ylabel("Delta slowdown: Tetris* - DeepRM")
    for ax in axes:
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("DeepRM Perturbation Sweeps", y=1.03)
    fig.tight_layout()
    return _save_figure(fig, "deeprm_perturbation_sweeps")


def _plot_numeric_sweep(ax, cells, prefix, xs, xlabel, *, anchor):
    ys = []
    lo = []
    hi = []
    for x in xs:
        comp = cells[f"{prefix}{x}"]["comparison"]
        ys.append(comp["mean_difference"])
        lo.append(comp["ci_low"])
        hi.append(comp["ci_high"])
    ys_arr = np.asarray(ys)
    yerr = np.vstack([ys_arr - np.asarray(lo), np.asarray(hi) - ys_arr])
    ax.errorbar(xs, ys, yerr=yerr, color="#1f4e79", marker="o", linewidth=1.2, capsize=3)
    ax.axvline(anchor, color="#b04a1a", linestyle="--", linewidth=1.0)
    ax.set_xlabel(xlabel)


def _plot_tail_sweep(ax, cells):
    labels = ["inf", "3.0", "2.0", "1.5", "1.2"]
    xs = np.arange(len(labels))
    ys = []
    lo = []
    hi = []
    for label in labels:
        comp = cells[f"P2_tail_{label}"]["comparison"]
        ys.append(comp["mean_difference"])
        lo.append(comp["ci_low"])
        hi.append(comp["ci_high"])
    ys_arr = np.asarray(ys)
    yerr = np.vstack([ys_arr - np.asarray(lo), np.asarray(hi) - ys_arr])
    ax.errorbar(xs, ys, yerr=yerr, color="#1f4e79", marker="o", linewidth=1.2, capsize=3)
    ax.axvline(labels.index("1.5"), color="#b04a1a", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tail alpha")


def _save_figure(fig, stem: str) -> list[Path]:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _write_clean_gate_tables(clean: dict[str, object], source: dict[str, object]) -> list[Path]:
    rows = []
    for method, value in clean["summary"]["method_means"].items():
        rows.append(
            {
                "gate": "strict_v2_2",
                "method": method,
                "mean_slowdown": value,
                "mlflow_run_id": clean["mlflow_run_id"],
            }
        )
    for method, row in source["summary"].items():
        rows.append(
            {
                "gate": "author_source",
                "method": method,
                "mean_slowdown": row["mean_slowdown_finished"],
                "mlflow_run_id": source["mlflow_run_id"],
            }
        )
    csv_path = TABLE_DIR / "deeprm_clean_gate.csv"
    md_path = TABLE_DIR / "deeprm_clean_gate.md"
    tex_path = TABLE_DIR / "deeprm_clean_gate.tex"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, ["gate", "method", "mean_slowdown", "mlflow_run_id"])
    _write_latex(tex_path, rows, ["gate", "method", "mean_slowdown"], "DeepRM clean gate results.")
    return [csv_path, md_path, tex_path]


def _write_prediction_tables(sweeps: dict[str, object]) -> list[Path]:
    rows = []
    anchors = [
        ("P1", "k = 10", "P1_lag_10"),
        ("P2", "alpha = 1.5", "P2_tail_1.5"),
        ("P3", "epsilon = 0.05", "P3_epsilon_0.05"),
    ]
    verdicts = sweeps["summary"]["anchor_verdicts_deeprm_only_holm"]
    for pred, anchor, key in anchors:
        cell = sweeps["cells"][key]
        comp = cell["comparison"]
        rows.append(
            {
                "prediction": pred,
                "anchor": anchor,
                "deep_rm_mean": cell["deep_rm_mean_slowdown"],
                "tetris_mean": cell["tetris_mean_slowdown"],
                "delta_tetris_minus_deeprm": comp["mean_difference"],
                "ci_low": comp["ci_low"],
                "ci_high": comp["ci_high"],
                "holm_p_falsification_deeprm_only": sweeps["anchor_falsification_holm_deeprm_only"][f"{pred}-DeepRM"],
                "verdict_deeprm_only": verdicts[f"{pred}-DeepRM"],
            }
        )
    sweep_rows = []
    for key, cell in sorted(sweeps["cells"].items()):
        comp = cell["comparison"]
        sweep_rows.append(
            {
                "cell": key,
                "parameter": cell["parameter_value"],
                "deep_rm_mean": cell["deep_rm_mean_slowdown"],
                "tetris_mean": cell["tetris_mean_slowdown"],
                "delta_tetris_minus_deeprm": comp["mean_difference"],
                "ci_low": comp["ci_low"],
                "ci_high": comp["ci_high"],
            }
        )
    csv_path = TABLE_DIR / "deeprm_prediction_outcomes.csv"
    md_path = TABLE_DIR / "deeprm_prediction_outcomes.md"
    tex_path = TABLE_DIR / "deeprm_prediction_outcomes.tex"
    sweep_csv = TABLE_DIR / "deeprm_sweep_compact.csv"
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        rows,
        [
            "prediction",
            "anchor",
            "deep_rm_mean",
            "tetris_mean",
            "delta_tetris_minus_deeprm",
            "ci_low",
            "ci_high",
            "holm_p_falsification_deeprm_only",
            "verdict_deeprm_only",
        ],
    )
    _write_latex(
        tex_path,
        rows,
        [
            "prediction",
            "anchor",
            "deep_rm_mean",
            "tetris_mean",
            "delta_tetris_minus_deeprm",
            "ci_low",
            "ci_high",
            "verdict_deeprm_only",
        ],
        "DeepRM pre-registered anchor outcomes.",
    )
    _write_csv(sweep_csv, sweep_rows)
    return [csv_path, md_path, tex_path, sweep_csv]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, rows: list[dict[str, object]], columns: list[str], caption: str) -> None:
    colspec = "l" * len(columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & ".join(columns).replace("_", "\\_") + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_format_cell(row[col]).replace("_", "\\_") for col in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    main()
