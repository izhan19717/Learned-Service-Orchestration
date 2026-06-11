#!/usr/bin/env python3
"""Render the manuscript-facing Decima E1 magnitude-sweep panel."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
TABLE = (
    ROOT
    / "results"
    / "paper"
    / "experiments"
    / "e1_magnitude_sweep"
    / "decima"
    / "tables"
    / "e1_decima_magnitude_sweep.csv"
)
FIG_DIR = (
    ROOT
    / "results"
    / "paper"
    / "experiments"
    / "e1_magnitude_sweep"
    / "decima"
    / "figures"
)
STEM = "e1_decima_magnitude_sweep_paper_panel"


def read_rows() -> list[dict[str, str]]:
    with TABLE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def curve_arrays(rows: list[dict[str, str]], curve: str, scale: float) -> tuple[np.ndarray, ...]:
    subset = [row for row in rows if row["curve"] == curve]
    subset.sort(key=lambda row: float(row["magnitude"]))
    x = np.asarray([float(row["magnitude"]) for row in subset], dtype=float)
    y = np.asarray([float(row["delta_dynamic_partition_minus_decima"]) / scale for row in subset], dtype=float)
    lo = np.asarray([float(row["ci_low"]) / scale for row in subset], dtype=float)
    hi = np.asarray([float(row["ci_high"]) / scale for row in subset], dtype=float)
    return x, y, lo, hi


def style_axis(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="#222222", linewidth=0.75, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#333333", labelsize=8, width=0.8)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.55, zorder=0)


def draw_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
    ax.fill_between(x, lo, hi, color="#8ec3ec", alpha=0.45, linewidth=0, zorder=2)
    ax.plot(
        x,
        y,
        color="#2878b8",
        marker="o",
        markersize=4.2,
        linewidth=1.75,
        markeredgewidth=0,
        zorder=3,
    )


def main() -> None:
    rows = read_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.4,
            "axes.titlesize": 10.1,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.titleweight": "regular",
        }
    )

    fig = plt.figure(figsize=(5.95, 3.85))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        left=0.115,
        right=0.985,
        top=0.925,
        bottom=0.135,
        hspace=0.64,
        wspace=0.50,
    )
    ax_p1 = fig.add_subplot(gs[0, :])
    ax_p2 = fig.add_subplot(gs[1, 0])
    ax_p3 = fig.add_subplot(gs[1, 1])

    # P1 uses a million-JCT-unit scale in the main axis. The inset is necessary
    # because the lambda=0.25 reversal is real but visually compressed by the
    # large-lag cells.
    x, y, lo, hi = curve_arrays(rows, "p1_lag", 1_000_000.0)
    style_axis(ax_p1)
    draw_curve(ax_p1, x, y, lo, hi)
    ax_p1.axvline(1.0, color="#555555", linestyle=":", linewidth=1.0, zorder=4)
    ax_p1.set_title("P1 observation lag", pad=4)
    ax_p1.set_xlabel(r"Lag multiplier $\lambda$")
    ax_p1.set_ylabel(r"$\Delta$ (mean JCT, $\times 10^6$)")
    ax_p1.set_xlim(-0.04, 2.08)
    ax_p1.set_ylim(-0.09, 1.62)

    inset = inset_axes(ax_p1, width="35%", height="56%", loc="upper left", borderpad=1.05)
    mask = x <= 0.5
    style_axis(inset)
    draw_curve(inset, x[mask], y[mask] * 1000.0, lo[mask] * 1000.0, hi[mask] * 1000.0)
    inset.set_xlim(-0.03, 0.53)
    inset.set_ylim(-36.0, 158.0)
    inset.set_xticks([0.0, 0.25, 0.5])
    inset.set_xticklabels(["0", "0.25", "0.5"])
    inset.set_yticks([0.0, 50.0, 100.0, 150.0])
    inset.set_title(r"$\Delta$ ($\times 10^3$)", fontsize=7.4, pad=2)
    inset.tick_params(axis="both", labelsize=6.5, pad=1.8)
    inset.grid(axis="y", color="#edf2f7", linewidth=0.45)
    row_025 = next(row for row in rows if row["curve"] == "p1_lag" and row["magnitude"] == "0.25")
    delta_025 = float(row_025["delta_dynamic_partition_minus_decima"])
    p_h_025 = float(row_025["holm_less_curve"])
    inset.plot([0.25], [delta_025 / 1000.0], marker="o", color="#1f5f94", markersize=4.0, zorder=5)
    inset.text(
        0.035,
        0.88,
        rf"$\lambda=0.25$" + "\n" + rf"$\Delta={delta_025 / 1000:.1f}$k, $p_H={p_h_025:.3f}$",
        transform=inset.transAxes,
        fontsize=6.5,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "linewidth": 0.55},
        zorder=8,
    )

    x, y, lo, hi = curve_arrays(rows, "p2_tail", 1000.0)
    style_axis(ax_p2)
    draw_curve(ax_p2, x, y, lo, hi)
    ax_p2.axvline(0.5, color="#555555", linestyle=":", linewidth=1.0, zorder=4)
    ax_p2.set_title("P2 DAG-size tail", pad=4)
    ax_p2.set_xlabel("Tail weight w")
    ax_p2.set_ylabel(r"$\Delta$ ($\times 10^3$)")
    ax_p2.set_xlim(-0.04, 1.04)
    ax_p2.set_ylim(-4.0, 70.0)

    x, y, lo, hi = curve_arrays(rows, "p3_fgsm", 1000.0)
    style_axis(ax_p3)
    draw_curve(ax_p3, x, y, lo, hi)
    ax_p3.axvline(0.05, color="#555555", linestyle=":", linewidth=1.0, zorder=4)
    ax_p3.set_title("P3 FGSM node features", pad=4)
    ax_p3.set_xlabel(r"$\epsilon$")
    ax_p3.set_ylabel(r"$\Delta$ ($\times 10^3$)")
    ax_p3.set_xlim(-0.01, 0.21)
    ax_p3.set_ylim(-0.15, 3.35)

    for suffix in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{STEM}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=420, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
            if suffix == "svg":
                text = path.read_text(encoding="utf-8")
                path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
