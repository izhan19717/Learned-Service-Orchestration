#!/usr/bin/env python3
"""Render README evidence-summary SVGs from committed result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

KEY_FINDINGS = ASSET_DIR / "readme_key_findings.svg"
SENSITIVITY = ASSET_DIR / "readme_sensitivity_checks.svg"

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
    values = {
        "deeprm": _deeprm_values(),
        "rossi": _rossi_values(),
        "decima": _decima_gate_values(),
    }
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FINDINGS.write_text(_key_findings_svg(values), encoding="utf-8")
    SENSITIVITY.write_text(_sensitivity_svg(values), encoding="utf-8")
    print(KEY_FINDINGS.relative_to(ROOT))
    print(SENSITIVITY.relative_to(ROOT))


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
    bundled_value = float(bundled["delta_hpa_minus_rossi"])
    return {
        "bundled": bundled_value,
        "hpa_min": min(hpa_deltas),
        "hpa_max": hpa_max,
        "ratio": bundled_value / hpa_max,
    }


def _decima_gate_values() -> dict[str, float]:
    gate = _read_json(DECIMA_GATE)["gate"]
    return {
        "observed": float(gate["observed_improvement_pct"]),
        "target": float(gate["target_improvement_pct"]),
    }


def _common_style() -> str:
    return """
  <style>
    .title { font: 800 34px Arial, Helvetica, sans-serif; fill: #0f172a; }
    .subtitle { font: 400 17px Arial, Helvetica, sans-serif; fill: #475569; }
    .card { fill: #ffffff; stroke: #d8dee9; stroke-width: 2; }
    .eyebrow { font: 800 13px Arial, Helvetica, sans-serif; letter-spacing: .08em; fill: #475569; }
    .card-title { font: 800 22px Arial, Helvetica, sans-serif; fill: #0f172a; }
    .label { font: 400 15px Arial, Helvetica, sans-serif; fill: #475569; }
    .label-dark { font: 700 17px Arial, Helvetica, sans-serif; fill: #0f172a; }
    .small { font: 400 13px Arial, Helvetica, sans-serif; fill: #475569; }
    .big { font: 800 72px Arial, Helvetica, sans-serif; fill: #0f172a; }
    .big-amber { font: 800 76px Arial, Helvetica, sans-serif; fill: #b45309; }
    .big-blue { font: 800 68px Arial, Helvetica, sans-serif; fill: #1f5b85; }
    .metric { font: 800 24px Arial, Helvetica, sans-serif; fill: #0f172a; }
    .metric-amber { font: 800 22px Arial, Helvetica, sans-serif; fill: #b45309; }
    .status { font: 800 18px Arial, Helvetica, sans-serif; fill: #ffffff; }
    .amber { fill: #b45309; }
    .green { fill: #047857; }
    .blue { fill: #2563eb; }
    .slate { fill: #6b7280; }
    .blue-dark { fill: #1f5b85; }
    .line { stroke: #d8dee9; stroke-width: 3; }
    .axis { stroke: #0f172a; stroke-width: 2; }
    .target-band { fill: #dcfce7; }
    .pill { fill: #fff7ed; }
  </style>"""


def _key_findings_svg(values: dict[str, dict[str, float]]) -> str:
    rossi = values["rossi"]
    decima = values["decima"]
    mini = _status_squares(94, 288, 24, 10)
    decima_target_width = round(decima["target"] / 26.0 * 210)
    decima_observed_width = round(decima["observed"] / 26.0 * 210)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 1200 390">
  <title id="title">Key empirical findings</title>
  <desc id="desc">Three headline findings from the learned service orchestration artifact.</desc>
{_common_style()}
  <rect x="0" y="0" width="1200" height="390" fill="#ffffff" />
  <text class="title" x="60" y="58">Key empirical findings</text>
  <text class="subtitle" x="60" y="88">The repository ties the paper claims to reproducible protocols, scripts, tables, and figures.</text>

  <rect class="card" x="60" y="120" width="335" height="220" rx="18" />
  <text class="eyebrow" x="94" y="158">PRE-REGISTERED TESTS</text>
  <text class="big-amber" x="94" y="235">7/9</text>
  <text class="card-title" x="94" y="266">predictions falsified</text>
  {mini}

  <rect class="card" x="432" y="120" width="335" height="220" rx="18" />
  <text class="eyebrow" x="466" y="158">COMPARATOR STANDARD</text>
  <text class="big-amber" x="466" y="235">{rossi['ratio']:.1f}x</text>
  <text class="card-title" x="466" y="266">bundled effect inflation</text>
  <text class="label" x="466" y="296">relative to the HPA-v2 maximum</text>
  <text class="small" x="466" y="322">Rossi P1 lag cell, k = 10</text>

  <rect class="card" x="805" y="120" width="335" height="220" rx="18" />
  <text class="eyebrow" x="839" y="158">OFFICIAL-SIMULATOR GATE</text>
  <text class="big-amber" x="839" y="235">{decima['observed']:.1f}%</text>
  <text class="card-title" x="839" y="266">observed Decima gain</text>
  <rect class="target-band" x="839" y="295" width="210" height="26" />
  <rect fill="#4fa38b" x="839" y="295" width="{decima_target_width}" height="26" />
  <rect class="amber" x="839" y="326" width="{decima_observed_width}" height="12" />
  <text class="small" x="1058" y="315">target {decima['target']:.0f}%</text>
</svg>
"""


def _sensitivity_svg(values: dict[str, dict[str, float]]) -> str:
    deep = values["deeprm"]
    rossi = values["rossi"]

    base_y = 340
    locked_h = round(deep["locked"] / 220.0 * 110)
    first_fit_h = round(deep["first_fit"] / 220.0 * 110)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 1200 430">
  <title id="title">Sensitivity checks</title>
  <desc id="desc">Protocol and comparator sensitivity checks for the learned service orchestration artifact.</desc>
{_common_style()}
  <rect x="0" y="0" width="1200" height="430" fill="#ffffff" />
  <text class="title" x="60" y="58">Sensitivity checks</text>
  <text class="subtitle" x="60" y="88">Two diagnostics show why the empirical interpretation depends on operational details.</text>

  <rect class="card" x="60" y="120" width="520" height="250" rx="18" />
  <text class="card-title" x="94" y="165">DeepRM stale-action rule</text>
  <text class="label" x="94" y="193">Changing only the invalid-action fallback changes the P1 magnitude.</text>
  <rect class="slate" x="145" y="{base_y - locked_h}" width="92" height="{locked_h}" />
  <rect class="blue-dark" x="305" y="{base_y - first_fit_h}" width="92" height="{first_fit_h}" />
  <text class="metric" x="166" y="{base_y - locked_h - 14}">+{deep['locked']:.0f}</text>
  <text class="metric" x="326" y="{base_y - first_fit_h - 14}">+{deep['first_fit']:.0f}</text>
  <text class="small" x="142" y="366">locked no-op</text>
  <text class="small" x="319" y="366">first-fit</text>
  <text class="big-blue" x="475" y="265" text-anchor="middle">{deep['reduction_pct']:.0f}%</text>
  <text class="label" x="475" y="300" text-anchor="middle">smaller at</text>
  <text class="label-dark" x="475" y="326" text-anchor="middle">lag k = 10</text>

  <rect class="card" x="620" y="120" width="520" height="250" rx="18" />
  <text class="card-title" x="654" y="165">Rossi comparator sensitivity</text>
  <text class="label" x="654" y="193">The bundled threshold collapse is far outside the HPA-v2 grid.</text>
  <text class="big-amber" x="654" y="276">{rossi['ratio']:.1f}x</text>
  <text class="label-dark" x="660" y="306">bundled / HPA-v2 max</text>
  <text class="metric-amber" x="1008" y="232">+{rossi['bundled']:.0f}</text>
  <line x1="900" y1="252" x2="1080" y2="252" stroke="#b45309" stroke-width="16" stroke-linecap="round" />
  <text class="small" x="900" y="286">bundled threshold</text>
  <line x1="900" y1="322" x2="960" y2="322" stroke="#2563eb" stroke-width="10" stroke-linecap="round" />
  <text class="label-dark" x="975" y="328">{rossi['hpa_min']:.0f} to +{rossi['hpa_max']:.0f}</text>
  <text class="small" x="900" y="356">HPA-v2 grid</text>
</svg>
"""


def _status_squares(x0: int, y0: int, size: int, gap: int) -> str:
    statuses = ["F", "F", "F", "F", "C", "C", "F", "F", "F"]
    parts = []
    for i, status in enumerate(statuses):
        x = x0 + i * (size + gap)
        cls = "green" if status == "C" else "amber"
        parts.append(f'<rect class="{cls}" x="{x}" y="{y0}" width="{size}" height="{size}" rx="4" />')
        parts.append(
            f'<text class="status" x="{x + size / 2:.1f}" y="{y0 + 18}" text-anchor="middle" font-size="13">{status}</text>'
        )
    return "\n  ".join(parts)


if __name__ == "__main__":
    main()
