#!/usr/bin/env python3
"""Render the README evidence-summary figure from committed result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "repository_evidence_summary.svg"

DEEPRM_P1 = ROOT / "results" / "paper" / "deeprm" / "tables" / "deeprm_p1_first_fit_sensitivity.csv"
ROSSI_P1 = ROOT / "results" / "paper" / "rossi" / "tables" / "rossi_p1_online_lag_sweep.csv"
HPA_V2 = (
    ROOT
    / "results"
    / "paper"
    / "experiments"
    / "hpa_v2_config_sensitivity"
    / "tables"
    / "hpa_v2_config_sensitivity_summary.csv"
)
DECIMA_GATE = ROOT / "results" / "paper" / "decima" / "tables" / "decima_official_readme_gate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--preview-png", type=Path, default=None)
    args = parser.parse_args()

    _set_style()
    fig = render_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg", bbox_inches="tight", metadata={"Date": None})
    if args.preview_png is not None:
        fig.savefig(args.preview_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.relative_to(ROOT))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 14.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def render_summary() -> plt.Figure:
    deep = _deeprm_values()
    rossi = _rossi_values()
    decima = _decima_gate_values()

    fig = plt.figure(figsize=(11.8, 7.0))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.98,
        bottom=0.115,
        top=0.765,
        hspace=0.62,
        wspace=0.27,
    )
    fig.text(
        0.5,
        0.955,
        "Evidence checks behind the reported conclusions",
        ha="center",
        va="top",
        fontsize=22,
        weight="bold",
        color="#111827",
    )
    fig.text(
        0.5,
        0.905,
        "The artifact tests the claims under pre-registration, stronger comparators, injection sensitivity, and official-simulator reproduction.",
        ha="center",
        va="top",
        fontsize=12.2,
        color="#374151",
    )

    ax_outcomes = fig.add_subplot(grid[0, 0])
    ax_rossi = fig.add_subplot(grid[0, 1])
    ax_deeprm = fig.add_subplot(grid[1, 0])
    ax_decima = fig.add_subplot(grid[1, 1])

    _draw_outcome_panel(ax_outcomes)
    _draw_rossi_comparator_panel(ax_rossi, rossi)
    _draw_deeprm_sensitivity_panel(ax_deeprm, deep)
    _draw_decima_gate_panel(ax_decima, decima)

    fig.text(
        0.5,
        0.035,
        "Positive Delta = metric(comparator) - metric(RL). Detailed confidence intervals, corrected tests, and diagnostics are under results/paper/.",
        ha="center",
        va="center",
        fontsize=10.6,
        color="#374151",
    )
    return fig


def _deeprm_values() -> dict[str, float]:
    rows = _read_csv(DEEPRM_P1)
    row = next(row for row in rows if float(row["lag"]) == 10.0)
    locked = float(row["locked_delta"])
    first_fit = float(row["delta_tetris_minus_deeprm"])
    reduction_pct = 100.0 * (locked - first_fit) / locked
    return {"locked": locked, "first_fit": first_fit, "reduction_pct": reduction_pct}


def _rossi_values() -> dict[str, float]:
    rossi_rows = _read_csv(ROSSI_P1)
    bundled = next(row for row in rossi_rows if float(row["lag"]) == 10.0)
    hpa_rows = [
        row
        for row in _read_csv(HPA_V2)
        if row["cell"] == "p1"
    ]
    hpa_deltas = np.asarray([float(row["delta_mean"]) for row in hpa_rows], dtype=float)
    return {
        "bundled": float(bundled["delta_hpa_minus_rossi"]),
        "hpa_min": float(np.min(hpa_deltas)),
        "hpa_max": float(np.max(hpa_deltas)),
    }


def _decima_gate_values() -> dict[str, float]:
    gate = _read_json(DECIMA_GATE)["gate"]
    return {
        "observed": float(gate["observed_improvement_pct"]),
        "target": float(gate["target_improvement_pct"]),
    }


def _draw_outcome_panel(ax: plt.Axes) -> None:
    methods = ["DeepRM", "Rossi/RLAD", "Decima"]
    predictions = ["P1 lag", "P2 tail", "P3 attack"]
    statuses = np.asarray(
        [
            ["F", "F", "F"],
            ["F", "C", "C"],
            ["F", "F", "F"],
        ]
    )
    colors = {"F": "#b45309", "C": "#047857"}

    ax.set_title("A. Pre-registered predictions", loc="left", y=1.13, pad=0, weight="bold")
    ax.text(
        0.0,
        1.035,
        "Seven of nine predicted degradation cells fail.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.7,
        color="#374151",
    )
    for y, method in enumerate(methods):
        for x, _prediction in enumerate(predictions):
            status = statuses[y, x]
            ax.scatter(x, y, s=980, marker="s", color=colors[status], edgecolor="white", linewidth=1.4)
            ax.text(x, y, status, ha="center", va="center", color="white", fontsize=15, weight="bold")

    ax.set_xticks(range(len(predictions)), predictions)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(2.55, -0.55)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.5,
        -0.20,
        "F = falsified; C = confirmed",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        color="#374151",
    )


def _draw_rossi_comparator_panel(ax: plt.Axes, values: dict[str, float]) -> None:
    bundled = values["bundled"]
    hpa_min = values["hpa_min"]
    hpa_max = values["hpa_max"]
    ratio = bundled / hpa_max

    ax.set_title("B. Rossi comparator standard", loc="left", y=1.13, pad=0, weight="bold")
    ax.text(
        0.0,
        1.035,
        "The bundled lag collapse is not representative of HPA-v2.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.7,
        color="#374151",
    )
    ax.axvline(0.0, color="#111827", linewidth=0.9)
    ax.barh([1.0], [bundled], height=0.34, color="#b45309", alpha=0.88)
    ax.hlines(0.0, hpa_min, hpa_max, color="#2563eb", linewidth=8, alpha=0.78)
    ax.scatter([hpa_min, hpa_max], [0.0, 0.0], s=64, color="#1d4ed8", zorder=3)
    ax.text(bundled + 20, 1.0, f"+{bundled:.0f}", va="center", ha="left", fontsize=11, weight="bold")
    ax.text(hpa_max + 22, 0.0, f"HPA-v2 range: {hpa_min:.0f} to +{hpa_max:.0f}", va="center", ha="left", fontsize=10.2)
    ax.text(
        0.70,
        0.84,
        f"bundled effect is {ratio:.1f}x the HPA-v2 maximum",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.4,
        color="#7c2d12",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#fff7ed", "edgecolor": "#fed7aa"},
    )
    ax.set_yticks([0.0, 1.0], ["HPA-v2 configs", "bundled threshold"])
    ax.set_xlabel("Delta total cost at lag k = 10")
    ax.set_xlim(-120, 1060)
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _draw_deeprm_sensitivity_panel(ax: plt.Axes, values: dict[str, float]) -> None:
    labels = ["locked no-op", "first-fit fallback"]
    vals = [values["locked"], values["first_fit"]]
    colors = ["#6b7280", "#1f5b85"]
    x = np.arange(len(vals))

    ax.set_title("C. DeepRM stale-action rule", loc="left", y=1.13, pad=0, weight="bold")
    ax.text(
        0.0,
        1.035,
        "Changing only the invalid-action fallback changes the P1 magnitude.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.7,
        color="#374151",
    )
    bars = ax.bar(x, vals, color=colors, width=0.52)
    for bar, value in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            f"+{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.text(
        0.5,
        0.79,
        f"{values['reduction_pct']:.0f}% smaller",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#1e3a5f",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#eff6ff", "edgecolor": "#bfdbfe"},
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Delta slowdown at k = 10")
    ax.set_ylim(0, 235)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_decima_gate_panel(ax: plt.Axes, values: dict[str, float]) -> None:
    observed = values["observed"]
    target = values["target"]
    tolerance_low = target * 0.85
    tolerance_high = target * 1.15

    ax.set_title("D. Decima official-simulator gate", loc="left", y=1.13, pad=0, weight="bold")
    ax.text(
        0.0,
        1.035,
        "The released simulator checkpoint improves JCT, but not at the target anchor.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.7,
        color="#374151",
    )
    ax.axvspan(tolerance_low, tolerance_high, color="#dcfce7", alpha=0.9, label="15% target band")
    ax.barh([1], [target], height=0.34, color="#047857", alpha=0.70, label="target")
    ax.barh([0], [observed], height=0.34, color="#b45309", alpha=0.90, label="observed")
    ax.text(observed + 0.7, 0, f"{observed:.1f}%", va="center", ha="left", fontsize=11, weight="bold")
    ax.text(target + 0.7, 1, f"{target:.0f}%", va="center", ha="left", fontsize=11, weight="bold")
    ax.text(
        0.60,
        0.18,
        "gate not met",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#7c2d12",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#fff7ed", "edgecolor": "#fed7aa"},
    )
    ax.set_yticks([0, 1], ["observed", "target"])
    ax.set_xlabel("Mean-JCT improvement over dynamic_partition (%)")
    ax.set_xlim(0, 26)
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


if __name__ == "__main__":
    main()
