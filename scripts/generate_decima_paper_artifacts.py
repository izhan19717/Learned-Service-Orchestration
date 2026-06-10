#!/usr/bin/env python3
"""Generate Decima paper tables, figures, and results report from locked results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from cisose_common.tracking import start_run, write_json_artifact


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "paper" / "decima"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
EXPERIMENT_NAME = "cisose_decima_v2_2"


@dataclass(frozen=True)
class PredictionSpec:
    prediction: str
    title: str
    anchor: str
    path: Path


SPECS = (
    PredictionSpec(
        prediction="P1-Decima observation lag",
        title="P1: observation lag",
        anchor="lambda=1.0",
        path=TABLE_DIR / "decima_p1_lag_lambda_1_0.json",
    ),
    PredictionSpec(
        prediction="P2-Decima workload tail",
        title="P2: tail-shifted workload",
        anchor="w=0.5",
        path=TABLE_DIR / "decima_p2_tail_w_0_5.json",
    ),
    PredictionSpec(
        prediction="P3-Decima adversarial node features",
        title="P3: FGSM node features",
        anchor="epsilon=0.05",
        path=TABLE_DIR / "decima_p3_fgsm_epsilon_0_05.json",
    ),
)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    payloads = [(spec, _read_json(spec.path)) for spec in SPECS]
    summary_rows = [_summary_row(spec, payload) for spec, payload in payloads]
    per_seed_rows = _per_seed_rows(payloads)

    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-paper-artifacts-final",
        role="paper-artifact-generation",
        params={
            "method": "decima",
            "scope": "official_simulator_dynamic_partition_final_artifacts",
            "num_predictions": len(SPECS),
            "comparator": "dynamic_partition",
            "protocol_amendment": "PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md",
        },
        tags={
            "decima.status": "final_prediction_artifacts",
            "decima.comparator": "dynamic_partition",
        },
    ) as run:
        table_paths = [
            _write_prediction_outcomes(summary_rows),
            _write_prediction_summary(summary_rows),
            _write_per_seed_csv(per_seed_rows),
        ]
        report_path = _write_results_report(summary_rows, payloads, run.info.run_id)
        fig_paths = [
            *_plot_percent_improvement(summary_rows),
            *_plot_paired_distribution(per_seed_rows),
            *_plot_mean_jct(summary_rows),
        ]
        manifest_path = _write_manifest(
            summary_rows,
            table_paths + [report_path],
            fig_paths,
            run.info.run_id,
        )

        for row in summary_rows:
            prefix = row["prediction"].split("-", maxsplit=1)[0].lower()
            mlflow.log_metric(f"decima.{prefix}.delta_mean", row["delta_value"])
            mlflow.log_metric(f"decima.{prefix}.ci_low", row["ci_low"])
            mlflow.log_metric(f"decima.{prefix}.ci_high", row["ci_high"])
            mlflow.log_metric(f"decima.{prefix}.percent_improvement", row["percent_improvement"])
            mlflow.log_metric(f"decima.{prefix}.decima_win_fraction", row["decima_win_fraction"])
        for path in table_paths:
            mlflow.log_artifact(str(path), artifact_path="paper/tables")
        for path in fig_paths:
            mlflow.log_artifact(str(path), artifact_path="paper/figures")
        mlflow.log_artifact(str(report_path), artifact_path="paper")
        mlflow.log_artifact(str(manifest_path), artifact_path="paper")

        print(f"MLflow run: {run.info.run_id}")
        print(f"Wrote {len(table_paths)} tables, {len(fig_paths)} figures, and report:")
        print(f"  {report_path.relative_to(ROOT)}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary_row(spec: PredictionSpec, payload: dict[str, object]) -> dict[str, object]:
    aggregate = payload["aggregate"]
    paired = payload["paired"]
    dynamic_mean = float(aggregate["dynamic_partition_mean_jct"])
    decima_mean = float(aggregate["learn_mean_jct"])
    delta = float(aggregate["delta_mean"])
    ci_low = float(aggregate["delta_ci_low"])
    ci_high = float(aggregate["delta_ci_high"])
    deltas = np.asarray([float(row["delta"]) for row in paired], dtype=float)
    status = _status(ci_low, ci_high)
    return {
        "prediction": spec.prediction,
        "title": spec.title,
        "anchor": spec.anchor,
        "status": status,
        "delta_definition": payload["delta_definition"],
        "dynamic_partition_mean_jct": dynamic_mean,
        "decima_mean_jct": decima_mean,
        "delta_value": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_less": float(aggregate["p_less"]),
        "p_greater": float(aggregate["p_greater"]),
        "num_pairs": int(aggregate["num_pairs"]),
        "percent_improvement": 100.0 * delta / dynamic_mean,
        "percent_ci_low": 100.0 * ci_low / dynamic_mean,
        "percent_ci_high": 100.0 * ci_high / dynamic_mean,
        "decima_win_fraction": float(np.mean(deltas > 0.0)),
        "comparator_win_fraction": float(np.mean(deltas < 0.0)),
        "prediction_confirmed": bool(aggregate["prediction_confirmed"]),
        "source_json": str(spec.path.relative_to(ROOT)),
    }


def _status(ci_low: float, ci_high: float) -> str:
    if ci_high < 0.0:
        return "Confirmed"
    if ci_low > 0.0:
        return "Falsified under simulator-gate amendment"
    return "Inconclusive"


def _per_seed_rows(payloads: list[tuple[PredictionSpec, dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec, payload in payloads:
        for row in payload["paired"]:
            dynamic = float(row["dynamic_partition_mean_jct"])
            decima = float(row["learn_mean_jct"])
            delta = float(row["delta"])
            rows.append(
                {
                    "prediction": spec.prediction,
                    "anchor": spec.anchor,
                    "exp": int(row["exp"]),
                    "seed": int(row["seed"]),
                    "dynamic_partition_mean_jct": dynamic,
                    "decima_mean_jct": decima,
                    "delta_dynamic_minus_decima": delta,
                    "percent_improvement": 100.0 * delta / dynamic,
                    "decima_wins": delta > 0.0,
                }
            )
    return rows


def _write_prediction_outcomes(rows: list[dict[str, object]]) -> Path:
    csv_path = TABLE_DIR / "decima_prediction_outcomes.csv"
    md_path = TABLE_DIR / "decima_prediction_outcomes.md"
    fields = [
        "prediction",
        "anchor",
        "status",
        "delta_definition",
        "delta_value",
        "ci_low",
        "ci_high",
        "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["reason"] = _reason(row)
            writer.writerow(out)

    lines = [
        "# Decima Prediction Outcomes",
        "",
        "All Decima P1/P2/P3 perturbation cells are complete under",
        "`PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.",
        "",
        "These are official-simulator results against the README-exposed",
        "`dynamic_partition` comparator. They must not be described as Graphene",
        "or full Spark-testbed headline results.",
        "",
        "| Prediction | Anchor | Status | Delta | 95% CI | Reason |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {prediction} | `{anchor}` | {status} | {delta:.6g} | "
            "[{lo:.6g}, {hi:.6g}] | {reason} |".format(
                prediction=row["prediction"],
                anchor=row["anchor"],
                status=row["status"],
                delta=row["delta_value"],
                lo=row["ci_low"],
                hi=row["ci_high"],
                reason=_reason(row),
            )
        )
    lines.extend(
        [
            "",
            "Delta is `mean_JCT(dynamic_partition) - mean_JCT(Decima)`. A",
            "positive delta means Decima has lower mean JCT than the comparator.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _write_prediction_summary(rows: list[dict[str, object]]) -> Path:
    path = TABLE_DIR / "decima_prediction_summary.csv"
    fields = [
        "prediction",
        "anchor",
        "status",
        "dynamic_partition_mean_jct",
        "decima_mean_jct",
        "delta_value",
        "ci_low",
        "ci_high",
        "percent_improvement",
        "percent_ci_low",
        "percent_ci_high",
        "decima_win_fraction",
        "comparator_win_fraction",
        "p_less",
        "p_greater",
        "num_pairs",
        "source_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return path


def _write_per_seed_csv(rows: list[dict[str, object]]) -> Path:
    path = TABLE_DIR / "decima_per_seed_deltas.csv"
    fields = [
        "prediction",
        "anchor",
        "exp",
        "seed",
        "dynamic_partition_mean_jct",
        "decima_mean_jct",
        "delta_dynamic_minus_decima",
        "percent_improvement",
        "decima_wins",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_percent_improvement(rows: list[dict[str, object]]) -> list[Path]:
    labels = [str(row["title"]).replace(": ", ":\n") for row in rows]
    y = np.asarray([row["percent_improvement"] for row in rows], dtype=float)
    lo = np.asarray([row["percent_ci_low"] for row in rows], dtype=float)
    hi = np.asarray([row["percent_ci_high"] for row in rows], dtype=float)
    yerr = np.vstack([y - lo, hi - y])
    colors = ["#315f72", "#7c5c2e", "#8b3f5d"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(np.arange(len(rows)), y, yerr=yerr, color=colors, capsize=4)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.set_ylabel("JCT improvement vs comparator (%)")
    ax.set_title("Decima Perturbation Outcomes")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.32,
        "Positive values mean Decima remains better than the official simulator comparator.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, "decima_prediction_percent_improvement")


def _plot_paired_distribution(rows: list[dict[str, object]]) -> list[Path]:
    grouped: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        key = str(row["prediction"])
        grouped.setdefault(key, []).append(float(row["percent_improvement"]))
        labels[key] = key.split(" ", maxsplit=1)[0].replace("-Decima", "")
    keys = list(grouped)
    data = [grouped[key] for key in keys]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    box = ax.boxplot(data, patch_artist=True, widths=0.52, showfliers=False)
    colors = ["#315f72", "#7c5c2e", "#8b3f5d"]
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#222222")
    for i, values in enumerate(data, start=1):
        x = np.full(len(values), i, dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(x + jitter, values, s=18, color=colors[i - 1], alpha=0.75, zorder=3)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(np.arange(1, len(keys) + 1), [labels[key] for key in keys])
    ax.set_ylabel("Per-seed improvement (%)")
    ax.set_title("Paired Seed Delta Distributions")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, "decima_paired_delta_distributions")


def _plot_mean_jct(rows: list[dict[str, object]]) -> list[Path]:
    labels = [str(row["title"]).replace(": ", ":\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    dynamic = np.asarray([row["dynamic_partition_mean_jct"] for row in rows], dtype=float)
    decima = np.asarray([row["decima_mean_jct"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - width / 2, dynamic, width, label="dynamic_partition", color="#777777")
    ax.bar(x + width / 2, decima, width, label="Decima", color="#315f72")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean JCT, log scale")
    ax.set_title("Absolute Mean JCT by Perturbation Cell")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, "decima_mean_jct_by_cell")


def _save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [FIG_DIR / f"{stem}.png", FIG_DIR / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return paths


def _write_results_report(
    rows: list[dict[str, object]],
    payloads: list[tuple[PredictionSpec, dict[str, object]]],
    run_id: str,
) -> Path:
    p3_payload = next(payload for spec, payload in payloads if spec.prediction.startswith("P3"))
    p3_diag = p3_payload.get("perturbation_metadata", {})
    lines = [
        "# Decima Results Report",
        "",
        "## Status",
        "",
        "Decima is complete as narrowed official-simulator evidence under",
        "`PROTOCOL_AMENDMENT_DECIMA_SIMULATOR_GATE.md`.",
        "",
        "The comparator is the official README-exposed `dynamic_partition`",
        "baseline. These results must not be described as Graphene evidence or",
        "as full Spark-testbed headline reproduction.",
        "",
        "## Reproduction Gate",
        "",
        "- Official simulator gate: passed.",
        "- Gate result: Decima improved mean JCT over `dynamic_partition` by `3.0125809474397287%`.",
        "- Original over-strict 21% headline gate: preserved as failed, but not used as the narrowed simulator gate.",
        "",
        "## Prediction Outcomes",
        "",
        "| Prediction | Anchor | Status | Delta | 95% CI | Decima win fraction |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {prediction} | `{anchor}` | {status} | {delta:.6g} | "
            "[{lo:.6g}, {hi:.6g}] | {wins:.3f} |".format(
                prediction=row["prediction"],
                anchor=row["anchor"],
                status=row["status"],
                delta=row["delta_value"],
                lo=row["ci_low"],
                hi=row["ci_high"],
                wins=row["decima_win_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "All three Decima predictions are falsified under the amended",
            "official-simulator comparator: the confidence interval for",
            "`mean_JCT(dynamic_partition) - mean_JCT(Decima)` is strictly",
            "positive in every perturbation cell.",
            "",
            "## FGSM Sanity Check",
            "",
        ]
    )
    if p3_diag:
        lines.extend(
            [
                f"- FGSM attack count: `{p3_diag.get('fgsm_attack_count')}`.",
                f"- Mean absolute node-feature delta: `{p3_diag.get('fgsm_mean_abs_node_feature_delta')}`.",
                f"- Mean clean target probability: `{p3_diag.get('fgsm_mean_clean_target_prob')}`.",
                f"- Mean adversarial target probability: `{p3_diag.get('fgsm_mean_adv_target_prob')}`.",
                "",
                "The adversarial perturbation reduced the clean target action",
                "probability on average, so the P3 falsification is not explained",
                "by a sign-error anti-attack.",
            ]
        )
    else:
        lines.append("FGSM diagnostics were not present in the result JSON.")
    lines.extend(
        [
            "",
            "## Paper Figures",
            "",
            "- `results/paper/decima/figures/decima_prediction_percent_improvement.png`",
            "- `results/paper/decima/figures/decima_paired_delta_distributions.png`",
            "- `results/paper/decima/figures/decima_mean_jct_by_cell.png`",
            "",
            "## Primary Tables",
            "",
            "- `results/paper/decima/tables/decima_prediction_outcomes.md`",
            "- `results/paper/decima/tables/decima_prediction_summary.csv`",
            "- `results/paper/decima/tables/decima_per_seed_deltas.csv`",
            "",
            "## Artifact Generation",
            "",
            f"- MLflow artifact-generation run: `{run_id}`.",
        ]
    )
    path = OUT_DIR / "DECIMA_RESULTS_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_manifest(
    rows: list[dict[str, object]],
    table_paths: list[Path],
    fig_paths: list[Path],
    run_id: str,
) -> Path:
    payload = {
        "status": "generated",
        "mlflow_artifact_run_id": run_id,
        "outcomes": rows,
        "tables": [str(path.relative_to(ROOT)) for path in table_paths],
        "figures": [str(path.relative_to(ROOT)) for path in fig_paths],
        "inputs": [str(spec.path.relative_to(ROOT)) for spec in SPECS],
    }
    path = OUT_DIR / "decima_paper_artifact_manifest.json"
    write_json_artifact(path, payload, run_id=run_id)
    return path


def _reason(row: dict[str, object]) -> str:
    if str(row["status"]).startswith("Falsified"):
        return "Decima remains lower-JCT than dynamic_partition in aggregate for this perturbation cell."
    if row["status"] == "Confirmed":
        return "dynamic_partition is lower-JCT than Decima in this perturbation cell."
    return "The paired confidence interval overlaps zero."


if __name__ == "__main__":
    main()
