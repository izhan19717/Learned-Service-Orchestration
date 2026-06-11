#!/usr/bin/env python3
"""Render the README evidence-summary figure from committed result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#d8dee9",
    "panel": "#ffffff",
    "paper": "#f8fafc",
    "amber": "#b45309",
    "amber_light": "#fff7ed",
    "green": "#047857",
    "green_light": "#ecfdf5",
    "blue": "#1f5b85",
    "blue_light": "#eff6ff",
    "slate": "#6b7280",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--preview-png", type=Path, default=None)
    args = parser.parse_args()

    _set_style()
    fig = render_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg", bbox_inches="tight", metadata={"Date": None})
    _strip_trailing_whitespace(args.output)
    if args.preview_png is not None:
        fig.savefig(args.preview_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output.relative_to(ROOT))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # GitHub/browser SVG rendering can fall back to serif fonts. Converting
            # text to paths keeps the README figure visually stable.
            "svg.fonttype": "path",
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def render_summary() -> plt.Figure:
    deep = _deeprm_values()
    rossi = _rossi_values()
    decima = _decima_gate_values()

    fig = plt.figure(figsize=(12.0, 7.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_header(ax)
    _draw_prediction_card(ax, 0.055, 0.515, 0.405, 0.345)
    _draw_rossi_card(ax, 0.535, 0.515, 0.410, 0.345, rossi)
    _draw_deeprm_card(ax, 0.055, 0.135, 0.405, 0.330, deep)
    _draw_decima_card(ax, 0.535, 0.135, 0.410, 0.330, decima)

    ax.text(
        0.5,
        0.055,
        "Delta = metric(comparator) - metric(RL). Full CIs, corrected tests, seeds, and diagnostics are in results/paper/.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    return fig


def _deeprm_values() -> dict[str, float]:
    row = next(row for row in _read_csv(DEEPRM_P1) if float(row["lag"]) == 10.0)
    locked = float(row["locked_delta"])
    first_fit = float(row["delta_tetris_minus_deeprm"])
    reduction_pct = 100.0 * (locked - first_fit) / locked
    return {"locked": locked, "first_fit": first_fit, "reduction_pct": reduction_pct}


def _rossi_values() -> dict[str, float]:
    bundled = next(row for row in _read_csv(ROSSI_P1) if float(row["lag"]) == 10.0)
    hpa_rows = [row for row in _read_csv(HPA_V2) if row["cell"] == "p1"]
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


def _draw_header(ax: plt.Axes) -> None:
    ax.text(
        0.055,
        0.935,
        "Empirical evidence summary",
        ha="left",
        va="top",
        fontsize=24,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.055,
        0.887,
        "Pre-registered perturbations, stronger comparator checks, injection sensitivity, and official-simulator reproduction.",
        ha="left",
        va="top",
        fontsize=12.5,
        color=COLORS["muted"],
    )


def _card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=COLORS["line"],
            facecolor=COLORS["panel"],
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.026, y + h - 0.043, title, ha="left", va="top", fontsize=14.8, weight="bold", color=COLORS["ink"])
    ax.text(x + 0.026, y + h - 0.076, subtitle, ha="left", va="top", fontsize=9.8, color=COLORS["muted"])


def _pill(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    text_color: str,
    fontsize: float,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.022",
            linewidth=0,
            facecolor=face,
            transform=ax.transAxes,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold", color=text_color)


def _draw_prediction_card(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    _card(ax, x, y, w, h, "A. Pre-registered predictions", "Seven predicted degradation cells fail under the locked anchors.")

    ax.text(x + 0.040, y + 0.150, "7/9", ha="left", va="center", fontsize=50, weight="bold", color=COLORS["amber"])
    ax.text(x + 0.045, y + 0.087, "predictions falsified", ha="left", va="center", fontsize=12.0, weight="bold", color=COLORS["ink"])

    methods = ["DeepRM", "Rossi/RLAD", "Decima"]
    preds = ["P1", "P2", "P3"]
    statuses = [["F", "F", "F"], ["F", "C", "C"], ["F", "F", "F"]]
    x0 = x + 0.270
    y0 = y + 0.165
    dx = 0.055
    dy = 0.055
    ax.text(x0 - 0.003, y0 + 0.054, "P1", ha="center", va="center", fontsize=9.4, weight="bold", color=COLORS["muted"])
    ax.text(x0 + dx, y0 + 0.054, "P2", ha="center", va="center", fontsize=9.4, weight="bold", color=COLORS["muted"])
    ax.text(x0 + 2 * dx, y0 + 0.054, "P3", ha="center", va="center", fontsize=9.4, weight="bold", color=COLORS["muted"])
    for r, method in enumerate(methods):
        yy = y0 - r * dy
        ax.text(x0 - 0.060, yy, method, ha="right", va="center", fontsize=9.4, color=COLORS["ink"])
        for c, pred in enumerate(preds):
            xx = x0 + c * dx
            status = statuses[r][c]
            color = COLORS["green"] if status == "C" else COLORS["amber"]
            ax.add_patch(Rectangle((xx - 0.015, yy - 0.020), 0.030, 0.040, facecolor=color, edgecolor="none", transform=ax.transAxes))
            ax.text(xx, yy, status, ha="center", va="center", fontsize=9.2, weight="bold", color="white")


def _draw_rossi_card(ax: plt.Axes, x: float, y: float, w: float, h: float, values: dict[str, float]) -> None:
    _card(ax, x, y, w, h, "B. Rossi comparator standard", "The bundled lag failure is not representative of HPA-v2.")
    bundled = values["bundled"]
    hpa_min = values["hpa_min"]
    hpa_max = values["hpa_max"]
    ratio = bundled / hpa_max

    ax.text(x + 0.036, y + 0.174, f"{ratio:.1f}x", ha="left", va="center", fontsize=34, weight="bold", color=COLORS["amber"])
    ax.text(x + 0.214, y + 0.184, "bundled effect relative", ha="left", va="center", fontsize=10.6, color=COLORS["muted"])
    ax.text(x + 0.214, y + 0.154, "to HPA-v2 maximum", ha="left", va="center", fontsize=11.8, weight="bold", color=COLORS["ink"])

    left = x + 0.052
    right = x + w - 0.048
    base_y = y + 0.040
    scale_min, scale_max = -80.0, 1000.0

    def sx(v: float) -> float:
        return left + (v - scale_min) / (scale_max - scale_min) * (right - left)

    zero = sx(0.0)
    ax.plot([left, right], [base_y, base_y], color=COLORS["line"], linewidth=1.4, transform=ax.transAxes, clip_on=False)
    ax.plot([zero, zero], [base_y - 0.025, base_y + 0.09], color=COLORS["ink"], linewidth=1.0, transform=ax.transAxes, clip_on=False)

    y_b = base_y + 0.062
    ax.plot([zero, sx(bundled)], [y_b, y_b], color=COLORS["amber"], linewidth=11, solid_capstyle="round", transform=ax.transAxes)
    ax.text(left, y_b - 0.028, "bundled threshold", ha="left", va="center", fontsize=9.3, color=COLORS["muted"])
    ax.text(sx(bundled), y_b + 0.029, f"+{bundled:.0f}", ha="right", va="center", fontsize=11.4, weight="bold", color=COLORS["amber"])

    y_h = base_y + 0.002
    ax.plot([sx(hpa_min), sx(hpa_max)], [y_h, y_h], color="#2563eb", linewidth=7, solid_capstyle="round", transform=ax.transAxes)
    ax.scatter([sx(hpa_min), sx(hpa_max)], [y_h, y_h], s=38, color="#1d4ed8", transform=ax.transAxes, zorder=3)
    ax.text(left, y_h - 0.035, "HPA-v2 grid", ha="left", va="center", fontsize=9.3, color=COLORS["muted"])
    ax.text(sx(hpa_max) + 0.008, y_h, f"{hpa_min:.0f} to +{hpa_max:.0f}", ha="left", va="center", fontsize=9.3, color=COLORS["ink"])


def _draw_deeprm_card(ax: plt.Axes, x: float, y: float, w: float, h: float, values: dict[str, float]) -> None:
    _card(ax, x, y, w, h, "C. DeepRM stale-action rule", "Changing only the invalid-action fallback changes the P1 magnitude.")
    locked = values["locked"]
    first_fit = values["first_fit"]
    ymax = 220.0
    bar_w = 0.072
    x1 = x + 0.090
    x2 = x + 0.225
    base = y + 0.055
    max_h = 0.135

    for xx, value, color, label in (
        (x1, locked, COLORS["slate"], "locked no-op"),
        (x2, first_fit, COLORS["blue"], "first-fit"),
    ):
        height = value / ymax * max_h
        ax.add_patch(Rectangle((xx, base), bar_w, height, facecolor=color, edgecolor="none", transform=ax.transAxes))
        ax.text(xx + bar_w / 2, base + height + 0.018, f"+{value:.0f}", ha="center", va="center", fontsize=13.0, weight="bold", color=COLORS["ink"])
        ax.text(xx + bar_w / 2, base - 0.030, label, ha="center", va="center", fontsize=9.2, color=COLORS["muted"])

    ax.text(x + 0.318, y + 0.176, f"{values['reduction_pct']:.0f}%", ha="center", va="center", fontsize=34, weight="bold", color=COLORS["blue"])
    ax.text(x + 0.318, y + 0.126, "smaller at", ha="center", va="center", fontsize=10.8, color=COLORS["muted"])
    ax.text(x + 0.318, y + 0.096, "lag k = 10", ha="center", va="center", fontsize=10.8, weight="bold", color=COLORS["ink"])


def _draw_decima_card(ax: plt.Axes, x: float, y: float, w: float, h: float, values: dict[str, float]) -> None:
    _card(ax, x, y, w, h, "D. Decima official-simulator gate", "Released checkpoint improves JCT, but misses the target.")
    observed = values["observed"]
    target = values["target"]
    max_pct = 26.0

    left = x + 0.055
    right = x + w - 0.055
    bar_w = right - left
    y_target = y + 0.182
    y_observed = y + 0.100

    target_low = target * 0.85 / max_pct
    target_high = target * 1.15 / max_pct
    ax.add_patch(
        Rectangle(
            (left + target_low * bar_w, y + 0.067),
            (target_high - target_low) * bar_w,
            0.160,
            facecolor="#dcfce7",
            edgecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.add_patch(Rectangle((left, y_target), target / max_pct * bar_w, 0.036, facecolor="#4fa38b", edgecolor="none", transform=ax.transAxes))
    ax.add_patch(Rectangle((left, y_observed), observed / max_pct * bar_w, 0.036, facecolor=COLORS["amber"], edgecolor="none", transform=ax.transAxes))
    ax.text(left - 0.010, y_target + 0.018, "target", ha="right", va="center", fontsize=9.8, color=COLORS["muted"])
    ax.text(left - 0.010, y_observed + 0.018, "observed", ha="right", va="center", fontsize=9.8, color=COLORS["muted"])
    ax.text(left + target / max_pct * bar_w + 0.010, y_target + 0.018, f"{target:.0f}%", ha="left", va="center", fontsize=12.5, weight="bold", color=COLORS["ink"])
    ax.text(left + observed / max_pct * bar_w + 0.010, y_observed + 0.018, f"{observed:.1f}%", ha="left", va="center", fontsize=12.5, weight="bold", color=COLORS["ink"])
    _pill(ax, x + 0.284, y + 0.067, 0.104, 0.043, "gate not met", COLORS["amber_light"], COLORS["amber"], 10.4)


if __name__ == "__main__":
    main()
