#!/usr/bin/env python3
"""Generate paper-facing figures and summaries for E1/E2 extensions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = ROOT / "results" / "paper" / "experiments" / "e1_magnitude_sweep"
E2_ROOT = ROOT / "results" / "paper" / "experiments" / "e2_objective_native"


ANCHORS = {
    "p1_lag": 10.0,
    "p2_tail": 1.5,
    "p3_fgsm": 0.05,
    "p1_threshold": 10.0,
    "p1_hpa_v2": 10.0,
    "p2_threshold": 1.5,
    "p3_threshold": 0.05,
}

DECIMA_ANCHORS = {
    "p1_lag": 1.0,
    "p2_tail": 0.5,
    "p3_fgsm": 0.05,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None or value == "" or value == "NA":
        return None
    return float(value)


def plot_decima() -> Path | None:
    rows = read_csv(E1_ROOT / "decima" / "tables" / "e1_decima_magnitude_sweep.csv")
    if not rows:
        return None
    fig_dir = E1_ROOT / "decima" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("p1_lag", "P1 observation lag", "Lag multiplier lambda"),
        ("p2_tail", "P2 DAG-size tail", "Tail weight w"),
        ("p3_fgsm", "P3 FGSM", "epsilon"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    for ax, (curve, title, xlabel) in zip(axes, specs, strict=True):
        subset = [row for row in rows if row["curve"] == curve]
        xs = np.asarray([float(row["magnitude"]) for row in subset], dtype=float)
        ys = np.asarray([float(row["delta_dynamic_partition_minus_decima"]) for row in subset], dtype=float)
        lows = np.asarray([float(row["ci_low"]) for row in subset], dtype=float)
        highs = np.asarray([float(row["ci_high"]) for row in subset], dtype=float)
        ax.axhline(0.0, color="#2b2b2b", linewidth=0.8)
        ax.axvline(DECIMA_ANCHORS[curve], color="#7f7f7f", linestyle=":", linewidth=0.9)
        ax.fill_between(xs, lows, highs, color="#b9d7f0", alpha=0.55, linewidth=0)
        ax.plot(xs, ys, color="#2268a8", marker="o", markersize=3.8, linewidth=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Delta comparator - Decima\n(mean JCT)", fontsize=8)
    out_png = fig_dir / "e1_decima_magnitude_sweep.png"
    out_pdf = fig_dir / "e1_decima_magnitude_sweep.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def plot_deeprm() -> Path | None:
    rows = read_csv(E1_ROOT / "deeprm" / "tables" / "e1_deeprm_magnitude_sweep.csv")
    if not rows:
        return None
    fig_dir = E1_ROOT / "deeprm" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("p1_lag", "P1 observation lag", "Lag k"),
        ("p2_tail", "P2 job-size tail", "Pareto alpha"),
        ("p3_fgsm", "P3 FGSM", "epsilon"),
    ]
    comparators = sorted({row["comparator"] for row in rows})
    colors = {"Tetris*": "#2268a8", "SourceTetris": "#7f7f7f", "SJF": "#b24c3f"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.25, 2.62))
    for ax, (curve, title, xlabel) in zip(axes, specs, strict=True):
        for comparator in comparators:
            subset = [
                row for row in rows
                if row["curve"] == curve
                and row["comparator"] == comparator
                and row.get("status", "complete") == "complete"
                and as_float(row.get("delta_comparator_minus_deeprm")) is not None
            ]
            if not subset:
                continue
            xs = np.asarray([float(row["magnitude"]) for row in subset], dtype=float)
            order = np.argsort(xs)
            xs = xs[order]
            ys = np.asarray([float(row["delta_comparator_minus_deeprm"]) for row in subset], dtype=float)[order]
            lows = np.asarray([float(row["ci_low"]) for row in subset], dtype=float)[order]
            highs = np.asarray([float(row["ci_high"]) for row in subset], dtype=float)[order]
            color = colors.get(comparator, "#555555")
            ax.fill_between(xs, lows, highs, color=color, alpha=0.17, linewidth=0)
            ax.plot(xs, ys, color=color, marker="o", markersize=3.8, linewidth=1.45, label=comparator)
        incomplete = [
            row for row in rows
            if row["curve"] == curve and row.get("status", "complete") != "complete"
        ]
        for row in incomplete:
            ax.axvline(float(row["magnitude"]), color="#9a9a9a", linestyle="--", linewidth=0.85)
        ax.axhline(0.0, color="#2b2b2b", linewidth=0.85)
        ax.axvline(ANCHORS[curve], color="#7f7f7f", linestyle=":", linewidth=1.0)
        ax.set_title(title, pad=4)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Delta comparator - DeepRM\n(mean slowdown)")
    axes[0].set_xticks([0, 10, 20, 50])
    axes[1].set_xticks([1.5, 2.0, 2.5])
    axes[2].set_xticks([0.00, 0.05, 0.10, 0.20])
    axes[-1].legend(frameon=False, loc="center right", handlelength=1.7, borderaxespad=0.2)
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.20, top=0.85, wspace=0.30)
    out_png = fig_dir / "e1_deeprm_magnitude_sweep.png"
    out_pdf = fig_dir / "e1_deeprm_magnitude_sweep.pdf"
    fig.savefig(out_png, dpi=450, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return out_png


def write_artifact_index(generated: list[Path]) -> Path:
    note = E1_ROOT / "E1_E2_ARTIFACT_INDEX.md"
    known_figures = [
        E1_ROOT / "decima" / "figures" / "e1_decima_magnitude_sweep_inset.pdf",
        E1_ROOT / "decima" / "figures" / "e1_decima_magnitude_sweep.png",
        E1_ROOT / "deeprm" / "figures" / "e1_deeprm_magnitude_sweep.png",
        E1_ROOT / "rossi" / "figures" / "e1_rossi_magnitude_sweep.png",
        E2_ROOT / "figures" / "e2_companion_a_rescore.png",
    ]
    pieces = [
        "# E1/E2 Artifact Index",
        "",
        "This file indexes the paper-facing E1/E2 extension artifacts generated from completed result CSVs.",
        "",
        "## Available Figures",
        "",
    ]
    available = [path for path in known_figures if path.exists()]
    if available:
        pieces.extend(f"- `{path.relative_to(ROOT)}`" for path in available)
    else:
        pieces.append("- none")
    pieces.extend(["", "## Result Files", ""])
    for path in [
        E1_ROOT / "decima" / "tables" / "e1_decima_magnitude_sweep.csv",
        E1_ROOT / "deeprm" / "tables" / "e1_deeprm_magnitude_sweep.csv",
        E1_ROOT / "rossi" / "tables" / "e1_rossi_magnitude_sweep.csv",
        E2_ROOT / "tables" / "e2_companion_a_weight_rescore.csv",
    ]:
        if path.exists():
            pieces.append(f"- `{path.relative_to(ROOT)}`")
    note.write_text("\n".join(pieces) + "\n", encoding="utf-8")
    return note


def main() -> None:
    generated = []
    for func in (plot_decima, plot_deeprm):
        path = func()
        if path is not None:
            generated.append(path)
    note = write_artifact_index(generated)
    print(note.relative_to(ROOT))
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
