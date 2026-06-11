#!/usr/bin/env python3
"""Render selected paper figures as clean vector graphics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "figures"

DEEPRM_TABLE = ROOT / "results" / "paper" / "deeprm" / "tables" / "deeprm_p1_first_fit_sensitivity.csv"
DEEPRM_LOCKED_TABLE = ROOT / "results" / "paper" / "deeprm" / "tables" / "deeprm_sweep_compact.csv"
DEEPRM_FIG_DIR = ROOT / "results" / "paper" / "deeprm" / "figures"

DECIMA_TABLE = ROOT / "results" / "paper" / "decima" / "tables" / "decima_per_seed_deltas.csv"
DECIMA_FIG_DIR = ROOT / "results" / "paper" / "decima" / "figures"

HPA_TABLE = (
    ROOT
    / "results"
    / "paper"
    / "experiments"
    / "hpa_v2_config_sensitivity"
    / "tables"
    / "hpa_v2_config_sensitivity_summary.csv"
)
HPA_FIG_DIR = ROOT / "results" / "paper" / "experiments" / "hpa_v2_config_sensitivity" / "figures"


def main() -> None:
    _set_style()
    paths = []
    paths.extend(render_deeprm_stale_action())
    paths.extend(render_decima_paired_distribution())
    paths.extend(render_hpa_v2_config_sensitivity())
    for path in paths:
        print(path.relative_to(ROOT))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.4,
            "axes.titlesize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "path.simplify": True,
            "path.simplify_threshold": 0.0,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_deeprm_stale_action() -> list[Path]:
    rows = sorted(_read_csv(DEEPRM_TABLE), key=lambda row: float(row["lag"]))
    locked_rows = {
        float(row["parameter"]): row
        for row in _read_csv(DEEPRM_LOCKED_TABLE)
        if row["cell"].startswith("P1_lag_")
    }
    lags = np.asarray([float(row["lag"]) for row in rows], dtype=float)
    alt_mean = np.asarray([float(row["delta_tetris_minus_deeprm"]) for row in rows], dtype=float)
    alt_low = np.asarray([float(row["ci_low"]) for row in rows], dtype=float)
    alt_high = np.asarray([float(row["ci_high"]) for row in rows], dtype=float)
    locked_mean = np.asarray([float(locked_rows[lag]["delta_tetris_minus_deeprm"]) for lag in lags], dtype=float)
    locked_low = np.asarray([float(locked_rows[lag]["ci_low"]) for lag in lags], dtype=float)
    locked_high = np.asarray([float(locked_rows[lag]["ci_high"]) for lag in lags], dtype=float)

    colors = {"locked": "#747474", "first_fit": "#1f5b85", "anchor": "#b84a17", "grid": "#e2e2e2"}
    fig, ax = plt.subplots(figsize=(3.52, 2.32))
    _ci_line(ax, lags, locked_mean, locked_low, locked_high, colors["locked"], "Locked no-op fallback")
    _ci_line(ax, lags, alt_mean, alt_low, alt_high, colors["first_fit"], "First-fit fallback sensitivity")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axvline(10.0, color=colors["anchor"], linestyle=(0, (3.2, 2.2)), linewidth=1.0)
    ax.set_title("DeepRM P1 stale-action sensitivity", pad=4.0)
    ax.set_xlabel("Observation lag k")
    ax.set_ylabel("Delta slowdown\n(Tetris* - DeepRM)")
    ax.set_xlim(-0.35, 20.65)
    ax.set_ylim(-8, 382)
    ax.set_xticks([0, 1, 2, 5, 10, 20])
    ax.set_yticks([0, 80, 160, 240, 320])
    ax.grid(True, color=colors["grid"], linewidth=0.55)
    ax.legend(frameon=False, loc="upper left", handlelength=2.1, borderaxespad=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.205, right=0.99, bottom=0.215, top=0.88)
    return _save(fig, DEEPRM_FIG_DIR, "deeprm_p1_first_fit_sensitivity")


def _ci_line(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
    label: str,
) -> None:
    err = np.vstack([mean - low, high - mean])
    ax.errorbar(
        x,
        mean,
        yerr=err,
        color=color,
        marker="o",
        markersize=4.6,
        markeredgewidth=0.0,
        linewidth=1.55,
        elinewidth=1.05,
        capsize=3.1,
        label=label,
        zorder=3,
    )


def render_decima_paired_distribution() -> list[Path]:
    rows = _read_csv(DECIMA_TABLE)
    order = [
        ("P1-Decima observation lag", "P1"),
        ("P2-Decima workload tail", "P2"),
        ("P3-Decima adversarial node features", "P3"),
    ]
    grouped = []
    labels = []
    for prefix, label in order:
        values = [float(row["percent_improvement"]) for row in rows if row["prediction"].startswith(prefix)]
        if len(values) != 30:
            raise RuntimeError(f"expected 30 paired seeds for {label}, found {len(values)}")
        grouped.append(np.asarray(values, dtype=float))
        labels.append(label)

    colors = ["#2f6f83", "#8a6a3d", "#9a4f70"]
    fig, ax = plt.subplots(figsize=(3.52, 2.36))
    positions = np.arange(1, len(grouped) + 1)
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#e67622", "linewidth": 1.25},
        whiskerprops={"color": "#2b2b2b", "linewidth": 0.9},
        capprops={"color": "#2b2b2b", "linewidth": 0.9},
        boxprops={"linewidth": 0.9, "edgecolor": "#2b2b2b"},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.34)

    rng = np.random.default_rng(20260607)
    for idx, values in enumerate(grouped, start=1):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        jitter = np.clip(jitter, -0.11, 0.11)
        ax.scatter(
            np.full(len(values), idx, dtype=float) + jitter,
            values,
            s=18,
            color=colors[idx - 1],
            alpha=0.74,
            linewidths=0.3,
            edgecolors="#263238",
            zorder=3,
        )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Per-seed improvement (%)")
    ax.set_title("Decima paired-seed delta distributions", pad=4.0)
    ax.set_ylim(-12.5, 41.5)
    ax.set_yticks([-10, 0, 10, 20, 30, 40])
    ax.grid(True, axis="y", color="#e2e2e2", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.86)
    return _save(fig, DECIMA_FIG_DIR, "decima_paired_delta_distributions")


def render_hpa_v2_config_sensitivity() -> list[Path]:
    rows = [row for row in _read_csv(HPA_TABLE) if row["cell"] == "p1"]
    colors = {"down300": "#1f77b4", "down0": "#e67e22", "grid": "#e2e2e2", "anchor": "#7a7a7a"}
    fig = plt.figure(figsize=(3.58, 2.68))
    gs = fig.add_gridspec(2, 1, height_ratios=(0.34, 1.0), hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_top)

    for down, marker, color, label in (
        (300, "o", colors["down300"], "scale-down 300 s"),
        (0, "s", colors["down0"], "scale-down 0 s"),
    ):
        series = sorted(
            [row for row in rows if int(row["scale_down_stabilization_seconds"]) == down],
            key=lambda row: float(row["target_utilization"]),
        )
        xs = np.asarray([100.0 * float(row["target_utilization"]) for row in series], dtype=float)
        ys = np.asarray([float(row["delta_mean"]) for row in series], dtype=float)
        low = np.asarray([float(row["ci_low"]) for row in series], dtype=float)
        high = np.asarray([float(row["ci_high"]) for row in series], dtype=float)
        yerr = np.vstack([ys - low, high - ys])
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=marker,
            markersize=4.8,
            linewidth=1.6,
            elinewidth=1.05,
            capsize=3.0,
            color=color,
            label=label,
            zorder=3,
        )

    ax_top.axhline(965.0, color=colors["anchor"], linestyle=(0, (4.0, 2.2)), linewidth=1.0)
    ax_top.text(70.6, 965.0, "bundled-threshold\nP1 anchor (+965)", ha="left", va="center", fontsize=7.7, color="#555555")
    ax.axhline(0.0, color="#222222", linewidth=0.8)

    ax.set_xlim(38.5, 72.0)
    ax.set_ylim(-45, 75)
    ax_top.set_ylim(930, 1000)
    ax.set_xticks([40, 50, 60, 70])
    ax.set_yticks([-40, 0, 40])
    ax_top.set_yticks([965])
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for axis in (ax_top, ax):
        axis.spines["right"].set_visible(False)
        axis.grid(True, color=colors["grid"], linewidth=0.55)

    break_kwargs = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=6, linestyle="none", color="#333333", mec="#333333", mew=0.85, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kwargs)
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, **break_kwargs)

    fig.suptitle("Rossi P1 under HPA-v2 configuration sensitivity", y=0.98, fontsize=9.8)
    fig.text(0.02, 0.43, "Delta cost\n(HPA-v2 - Rossi)", rotation=90, va="center", ha="center", fontsize=9.4)
    ax.set_xlabel("HPA target utilization (%)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.7, borderaxespad=0.25)
    fig.subplots_adjust(left=0.19, right=0.86, bottom=0.16, top=0.88)
    return _save(fig, HPA_FIG_DIR, "hpa_v2_config_p1_delta")


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for directory in (out_dir, FIG_ROOT):
        pdf = directory / f"{stem}.pdf"
        svg = directory / f"{stem}.svg"
        png = directory / f"{stem}.png"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.025)
        _strip_trailing_whitespace(svg)
        fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.025)
        paths.extend([pdf, svg, png])
    plt.close(fig)
    return paths


def _strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
