#!/usr/bin/env python3
"""Render selected paper figures as clean vector graphics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "figures"

DEEPRM_TABLE = ROOT / "results" / "paper" / "deeprm" / "tables" / "deeprm_p1_first_fit_sensitivity.csv"
DEEPRM_LOCKED_TABLE = ROOT / "results" / "paper" / "deeprm" / "tables" / "deeprm_sweep_compact.csv"
DEEPRM_FIG_DIR = ROOT / "results" / "paper" / "deeprm" / "figures"

DECIMA_TABLE = ROOT / "results" / "paper" / "decima" / "tables" / "decima_per_seed_deltas.csv"
DECIMA_FIG_DIR = ROOT / "results" / "paper" / "decima" / "figures"


def main() -> None:
    _set_style()
    paths = []
    paths.extend(render_deeprm_stale_action())
    paths.extend(render_decima_paired_distribution())
    for path in paths:
        print(path.relative_to(ROOT))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
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
    fig, ax = plt.subplots(figsize=(4.95, 2.65))
    _ci_line(ax, lags, locked_mean, locked_low, locked_high, colors["locked"], "Locked no-op fallback")
    _ci_line(ax, lags, alt_mean, alt_low, alt_high, colors["first_fit"], "First-fit fallback sensitivity")
    ax.axhline(0.0, color="#222222", linewidth=0.65)
    ax.axvline(10.0, color=colors["anchor"], linestyle=(0, (3.2, 2.2)), linewidth=0.8)
    ax.text(10.15, 367, "anchor", color=colors["anchor"], fontsize=6.4, va="top", ha="left")
    ax.set_title("DeepRM P1 stale-action sensitivity", pad=3.5)
    ax.set_xlabel("Observation lag k")
    ax.set_ylabel("Delta slowdown (Tetris* - DeepRM)")
    ax.set_xlim(-0.35, 20.65)
    ax.set_ylim(-8, 382)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(True, color=colors["grid"], linewidth=0.45)
    ax.legend(frameon=False, loc="upper left", handlelength=2.4, borderaxespad=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.19, top=0.90)
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
        markersize=3.8,
        markeredgewidth=0.0,
        linewidth=1.15,
        elinewidth=0.85,
        capsize=2.8,
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
    fig, ax = plt.subplots(figsize=(4.65, 2.45))
    positions = np.arange(1, len(grouped) + 1)
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#e67622", "linewidth": 1.05},
        whiskerprops={"color": "#2b2b2b", "linewidth": 0.75},
        capprops={"color": "#2b2b2b", "linewidth": 0.75},
        boxprops={"linewidth": 0.75, "edgecolor": "#2b2b2b"},
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
            s=13,
            color=colors[idx - 1],
            alpha=0.72,
            linewidths=0.25,
            edgecolors="#263238",
            zorder=3,
        )
    ax.axhline(0.0, color="#222222", linewidth=0.65)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Per-seed improvement (%)")
    ax.set_title("Decima paired-seed delta distributions", pad=3.5)
    ax.set_ylim(-12.5, 41.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="y", color="#e2e2e2", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.17, top=0.88)
    return _save(fig, DECIMA_FIG_DIR, "decima_paired_delta_distributions")


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
        fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.025)
        paths.extend([pdf, svg, png])
    plt.close(fig)
    return paths


if __name__ == "__main__":
    main()
