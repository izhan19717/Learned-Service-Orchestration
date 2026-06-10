#!/usr/bin/env python3
"""Render a publication-grade vector version of Experiment B's lag trace."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cisose_rossi.config import DEFAULT_CONFIG
from cisose_rossi.controllers import HPAv2Controller, ThresholdHPAController
from cisose_rossi.simulator import RladSimulator, StepRecord
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "experiment_b" / "figures"
ROOT_FIG_DIR = ROOT / "figures"
COSTS_PATH = ROOT / "results" / "paper" / "experiments" / "experiment_b" / "data" / "experiment_b_costs.csv"
FILENAME = "threshold_vs_hpa_v2_under_lag"
HORIZON = DEFAULT_CONFIG.time_limit + 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    rates = _median_p1_rates()
    threshold = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=10,
        observation_applies_to_update=True,
    )
    hpa = RladSimulator(DEFAULT_CONFIG).run(
        HPAv2Controller(DEFAULT_CONFIG),
        rates,
        horizon=len(rates),
        observation_lag_steps=10,
    )
    _render(threshold, hpa)
    for path in [
        OUT_DIR / f"{FILENAME}.pdf",
        OUT_DIR / f"{FILENAME}.png",
        ROOT_FIG_DIR / f"{FILENAME}.pdf",
        ROOT_FIG_DIR / f"{FILENAME}.png",
    ]:
        print(path.relative_to(ROOT))


def _median_p1_rates() -> tuple[float, ...]:
    with COSTS_PATH.open(newline="", encoding="utf-8") as f:
        p1_rows = [row for row in csv.DictReader(f) if row["cell"] == "p1"]
    if not p1_rows:
        raise RuntimeError(f"no P1 rows found in {COSTS_PATH}")
    median = sorted(p1_rows, key=lambda row: float(row["delta_hpa_v2_minus_rossi"]))[len(p1_rows) // 2]
    offset = int(median["offset"])
    profile = load_profile(PROFILE_PATH)
    sequence = java_slow_profile_sequence(profile, steps=len(profile) - 1)
    return tuple(sequence[offset : offset + HORIZON])


def _series(records: tuple[StepRecord, ...]) -> dict[str, np.ndarray]:
    return {
        "time": np.asarray([record.time for record in records], dtype=float),
        "replicas": np.asarray([record.replicas_before for record in records], dtype=float),
        "true_util": np.asarray([record.utilization for record in records], dtype=float),
        "observed_util": np.asarray([record.observed_utilization for record in records], dtype=float),
    }


def _render(threshold: tuple[StepRecord, ...], hpa: tuple[StepRecord, ...]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": True,
            "path.simplify_threshold": 0.0,
        }
    )
    colors = {
        "replicas": "#1f5d73",
        "true": "#2f2f2f",
        "observed": "#b21b35",
        "grid": "#e2e2e2",
        "zero": "#8a8a8a",
    }
    panels = [
        (_series(threshold), "Bundled threshold\nlag k=10"),
        (_series(hpa), "HPA-v2\nlag k=10"),
    ]
    util_max = max(float(np.max(panel[0]["true_util"])) for panel in panels)
    util_max = max(util_max, max(float(np.max(panel[0]["observed_util"])) for panel in panels))
    util_ylim = min(3.35, max(1.1, util_max * 1.03))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.05, 3.25),
        sharex=True,
        gridspec_kw={"height_ratios": [0.88, 1.0], "hspace": 0.13, "wspace": 0.18},
    )
    fig.subplots_adjust(left=0.065, right=0.992, bottom=0.13, top=0.90)

    for col, (data, title) in enumerate(panels):
        ax_rep = axes[0, col]
        ax_util = axes[1, col]
        ax_rep.step(
            data["time"],
            data["replicas"],
            where="post",
            color=colors["replicas"],
            linewidth=0.65 if col == 0 else 0.95,
            solid_capstyle="butt",
            rasterized=False,
        )
        ax_rep.set_title(title, pad=3.0)
        ax_rep.set_ylim(0.5, DEFAULT_CONFIG.max_replication + 0.5)
        ax_rep.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        ax_util.plot(
            data["time"],
            data["true_util"],
            color=colors["true"],
            linewidth=0.72,
            label="true",
            rasterized=False,
        )
        ax_util.plot(
            data["time"],
            data["observed_util"],
            color=colors["observed"],
            linewidth=0.62,
            alpha=0.9,
            label="observed",
            rasterized=False,
        )
        ax_util.axhline(0.5, color=colors["zero"], linewidth=0.55, linestyle=(0, (2.2, 2.2)), zorder=0)
        ax_util.set_ylim(0.0, util_ylim)
        ax_util.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_util.set_xlabel("Simulation tick")
        if col == 1:
            ax_util.legend(frameon=False, loc="upper right", handlelength=2.5, borderaxespad=0.2)

    axes[0, 0].set_ylabel("Replicas")
    axes[1, 0].set_ylabel("CPU utilization")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    for ax in axes.ravel():
        ax.grid(True, color=colors["grid"], linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, HORIZON - 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    for directory in (OUT_DIR, ROOT_FIG_DIR):
        fig.savefig(directory / f"{FILENAME}.pdf", bbox_inches="tight", pad_inches=0.015)
        fig.savefig(directory / f"{FILENAME}.png", dpi=450, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()
