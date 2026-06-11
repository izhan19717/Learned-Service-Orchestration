#!/usr/bin/env python3
"""Render the README evidence-summary SVG from committed result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    args = parser.parse_args()

    values = {
        "deeprm": _deeprm_values(),
        "rossi": _rossi_values(),
        "decima": _decima_gate_values(),
    }
    svg = render_svg(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(args.output.relative_to(ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _deeprm_values() -> dict[str, float]:
    row = next(row for row in _read_csv(DEEPRM_P1) if float(row["lag"]) == 10.0)
    locked = float(row["locked_delta"])
    first_fit = float(row["delta_tetris_minus_deeprm"])
    return {
        "locked": locked,
        "first_fit": first_fit,
        "reduction_pct": 100.0 * (locked - first_fit) / locked,
    }


def _rossi_values() -> dict[str, float]:
    bundled = next(row for row in _read_csv(ROSSI_P1) if float(row["lag"]) == 10.0)
    hpa_deltas = [
        float(row["delta_mean"])
        for row in _read_csv(HPA_V2)
        if row["cell"] == "p1"
    ]
    hpa_max = max(hpa_deltas)
    return {
        "bundled": float(bundled["delta_hpa_minus_rossi"]),
        "hpa_min": min(hpa_deltas),
        "hpa_max": hpa_max,
        "ratio": float(bundled["delta_hpa_minus_rossi"]) / hpa_max,
    }


def _decima_gate_values() -> dict[str, float]:
    gate = _read_json(DECIMA_GATE)["gate"]
    return {
        "observed": float(gate["observed_improvement_pct"]),
        "target": float(gate["target_improvement_pct"]),
    }


def render_svg(values: dict[str, dict[str, float]]) -> str:
    deep = values["deeprm"]
    rossi = values["rossi"]
    decima = values["decima"]

    rossi_bundled_width = 390
    rossi_hpa_width = 86
    decima_target_width = round(decima["target"] / 26.0 * 380)
    decima_observed_width = round(decima["observed"] / 26.0 * 380)
    deeprm_base_y = 632
    deeprm_locked_height = round(deep["locked"] / 220.0 * 88)
    deeprm_first_fit_height = round(deep["first_fit"] / 220.0 * 88)
    deeprm_locked_top = deeprm_base_y - deeprm_locked_height
    deeprm_first_fit_top = deeprm_base_y - deeprm_first_fit_height

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 1200 760">
  <title id="title">Empirical evidence summary</title>
  <desc id="desc">Evidence summary for the learned service orchestration artifact: preregistered prediction outcomes, Rossi comparator sensitivity, DeepRM stale-action sensitivity, and Decima official-simulator reproduction gate.</desc>
  <style>
    .title {{ font: 800 42px Arial, Helvetica, sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 400 19px Arial, Helvetica, sans-serif; fill: #475569; }}
    .card-title {{ font: 800 24px Arial, Helvetica, sans-serif; fill: #0f172a; }}
    .card-subtitle {{ font: 400 15px Arial, Helvetica, sans-serif; fill: #475569; }}
    .label {{ font: 400 16px Arial, Helvetica, sans-serif; fill: #475569; }}
    .label-dark {{ font: 700 18px Arial, Helvetica, sans-serif; fill: #0f172a; }}
    .small {{ font: 400 14px Arial, Helvetica, sans-serif; fill: #475569; }}
    .big-amber {{ font: 800 88px Arial, Helvetica, sans-serif; fill: #b45309; }}
    .big-blue {{ font: 800 72px Arial, Helvetica, sans-serif; fill: #1f5b85; }}
    .metric {{ font: 800 28px Arial, Helvetica, sans-serif; fill: #0f172a; }}
    .metric-amber {{ font: 800 25px Arial, Helvetica, sans-serif; fill: #b45309; }}
    .status {{ font: 800 21px Arial, Helvetica, sans-serif; fill: #ffffff; }}
    .card {{ fill: #ffffff; stroke: #d8dee9; stroke-width: 2; }}
    .amber {{ fill: #b45309; }}
    .green {{ fill: #047857; }}
    .blue {{ fill: #2563eb; }}
    .slate {{ fill: #6b7280; }}
    .gate {{ fill: #dcfce7; }}
    .gate-pill {{ fill: #fff7ed; }}
    .line {{ stroke: #d8dee9; stroke-width: 3; }}
    .axis {{ stroke: #0f172a; stroke-width: 2; }}
  </style>

  <rect x="0" y="0" width="1200" height="760" fill="#ffffff" />
  <text class="title" x="74" y="70">Empirical evidence summary</text>
  <text class="subtitle" x="74" y="101">Pre-registered perturbations, stronger comparator checks, injection sensitivity, and official-simulator reproduction.</text>

  <rect class="card" x="64" y="132" width="508" height="270" rx="18" />
  <text class="card-title" x="100" y="185">A. Pre-registered predictions</text>
  <text class="card-subtitle" x="100" y="211">Seven predicted degradation cells fail under the locked anchors.</text>
  <text class="big-amber" x="105" y="308">7/9</text>
  <text class="label-dark" x="115" y="342">predictions falsified</text>
  <text class="label-dark" x="288" y="262">DeepRM</text>
  <text class="label-dark" x="264" y="310">Rossi/RLAD</text>
  <text class="label-dark" x="293" y="358">Decima</text>
  <text class="label-dark" x="404" y="245">P1</text>
  <text class="label-dark" x="473" y="245">P2</text>
  <text class="label-dark" x="542" y="245">P3</text>
  {status_grid()}

  <rect class="card" x="628" y="132" width="508" height="270" rx="18" />
  <text class="card-title" x="664" y="185">B. Rossi comparator standard</text>
  <text class="card-subtitle" x="664" y="211">The bundled lag failure is not representative of HPA-v2.</text>
  <text class="big-amber" x="680" y="294">{rossi['ratio']:.1f}x</text>
  <text class="label" x="855" y="273">bundled effect relative</text>
  <text class="label-dark" x="855" y="303">to HPA-v2 maximum</text>
  <line class="line" x1="712" y1="338" x2="1060" y2="338" />
  <line class="axis" x1="740" y1="328" x2="740" y2="358" />
  <line x1="740" y1="338" x2="{740 + rossi_bundled_width}" y2="338" stroke="#b45309" stroke-width="18" stroke-linecap="round" />
  <text class="metric-amber" x="1045" y="317">+{rossi['bundled']:.0f}</text>
  <text class="small" x="712" y="376">bundled threshold</text>
  <line class="line" x1="712" y1="410" x2="1060" y2="410" />
  <line class="axis" x1="740" y1="397" x2="740" y2="434" />
  <line x1="725" y1="410" x2="{725 + rossi_hpa_width}" y2="410" stroke="#2563eb" stroke-width="12" stroke-linecap="round" />
  <circle cx="725" cy="410" r="8" fill="#2563eb" />
  <circle cx="{725 + rossi_hpa_width}" cy="410" r="8" fill="#2563eb" />
  <text class="label-dark" x="828" y="416">{rossi['hpa_min']:.0f} to +{rossi['hpa_max']:.0f}</text>
  <text class="small" x="712" y="449">HPA-v2 grid</text>

  <rect class="card" x="64" y="420" width="508" height="250" rx="18" />
  <text class="card-title" x="100" y="473">C. DeepRM stale-action rule</text>
  <text class="card-subtitle" x="100" y="499">Changing only the invalid-action fallback changes the P1 magnitude.</text>
  <rect class="slate" x="175" y="{deeprm_locked_top}" width="88" height="{deeprm_locked_height}" />
  <rect fill="#1f5b85" x="338" y="{deeprm_first_fit_top}" width="88" height="{deeprm_first_fit_height}" />
  <text class="metric" x="193" y="{deeprm_locked_top - 12}">+{deep['locked']:.0f}</text>
  <text class="metric" x="356" y="{deeprm_first_fit_top - 12}">+{deep['first_fit']:.0f}</text>
  <text class="small" x="174" y="642">locked no-op</text>
  <text class="small" x="353" y="642">first-fit</text>
  <text class="big-blue" x="425" y="555" text-anchor="middle">{deep['reduction_pct']:.0f}%</text>
  <text class="label" x="425" y="592" text-anchor="middle">smaller at</text>
  <text class="label-dark" x="425" y="622" text-anchor="middle">lag k = 10</text>

  <rect class="card" x="628" y="420" width="508" height="250" rx="18" />
  <text class="card-title" x="664" y="473">D. Decima official-simulator gate</text>
  <text class="card-subtitle" x="664" y="499">Released checkpoint improves JCT, but misses the target.</text>
  <rect class="gate" x="917" y="520" width="112" height="130" />
  <text class="label" x="664" y="552">target</text>
  <rect fill="#4fa38b" x="735" y="528" width="{decima_target_width}" height="32" />
  <text class="metric" x="1037" y="554">{decima['target']:.0f}%</text>
  <text class="label" x="664" y="625">observed</text>
  <rect class="amber" x="735" y="601" width="{decima_observed_width}" height="32" />
  <text class="metric" x="807" y="627">{decima['observed']:.1f}%</text>
  <rect class="gate-pill" x="936" y="617" width="154" height="44" rx="22" />
  <text class="metric-amber" x="1013" y="646" text-anchor="middle">gate not met</text>

  <text class="small" x="600" y="718" text-anchor="middle">Delta = metric(comparator) - metric(RL). Full CIs, corrected tests, seeds, and diagnostics are in results/paper/.</text>
</svg>
"""
    return svg


def status_grid() -> str:
    xs = [388, 457, 526]
    ys = [266, 314, 362]
    statuses = [
        ["F", "F", "F"],
        ["F", "C", "C"],
        ["F", "F", "F"],
    ]
    parts = []
    for row, y in zip(statuses, ys, strict=True):
        for status, x in zip(row, xs, strict=True):
            klass = "green" if status == "C" else "amber"
            parts.append(f'<rect class="{klass}" x="{x}" y="{y}" width="40" height="34" />')
            parts.append(f'<text class="status" x="{x + 20}" y="{y + 24}" text-anchor="middle">{status}</text>')
    return "\n  ".join(parts)


if __name__ == "__main__":
    main()
