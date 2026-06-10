#!/usr/bin/env python3
"""Experiment A: block-bootstrap sensitivity analysis for Rossi cells."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import chi2

from cisose_common.stats import holm_bonferroni
from cisose_common.tracking import start_run


EXPERIMENT_NAME = "cisose_rossi_v2_2"
SEED = 20260529
BOOTSTRAP_REPLICATES = 5000
SIGN_FLIP_REPLICATES = 100_000
BLOCK_LENGTHS = (5, 10)

VECTOR_PATH = ROOT / "results" / "paper" / "rossi" / "tables" / "rossi_per_window_delta_vectors.csv"
ROSSI_P1_PATH = ROOT / "results" / "paper" / "rossi" / "tables" / "rossi_p1_online_lag_sweep.csv"
ROSSI_P2_PATH = ROOT / "results" / "paper" / "rossi" / "tables" / "rossi_p2_online.csv"
ROSSI_P3_PATH = ROOT / "results" / "paper" / "rossi" / "tables" / "rossi_p3_online.csv"
DEEPRM_SWEEP_PATH = ROOT / "results" / "evaluation" / "deeprm" / "perturbation_sweeps_v2_2.json"
DECIMA_SUMMARY_PATH = ROOT / "results" / "paper" / "decima" / "tables" / "decima_prediction_summary.csv"

OUT_DIR = ROOT / "results" / "paper" / "experiments" / "experiment_a"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
ROOT_FIG_DIR = ROOT / "figures"
ROOT_DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class RossiCell:
    key: str
    label: str
    vector_column: str
    published_path: Path
    published_selector: tuple[str, str]
    prediction_name: str | None = None


ROSSI_CELLS = (
    RossiCell(
        key="clean",
        label="Clean",
        vector_column="clean_delta_hpa_minus_rossi",
        published_path=ROSSI_P1_PATH,
        published_selector=("lag", "0"),
    ),
    RossiCell(
        key="p1",
        label="P1 lag k=10",
        vector_column="p1_k10_delta_hpa_minus_rossi",
        published_path=ROSSI_P1_PATH,
        published_selector=("lag", "10"),
        prediction_name="P1-Rossi",
    ),
    RossiCell(
        key="p2",
        label="P2 tail alpha=1.5",
        vector_column="p2_alpha_1_5_delta_hpa_minus_rossi",
        published_path=ROSSI_P2_PATH,
        published_selector=("value", "1.5"),
        prediction_name="P2-Rossi",
    ),
    RossiCell(
        key="p3",
        label="P3 bucket-flip epsilon=0.05",
        vector_column="p3_epsilon_0_05_delta_hpa_minus_rossi",
        published_path=ROSSI_P3_PATH,
        published_selector=("value", "0.05"),
        prediction_name="P3-Rossi",
    ),
)


def main() -> None:
    _ensure_dirs()
    vectors = pd.read_csv(VECTOR_PATH)
    rossi_results = _analyze_rossi_cells(vectors)
    family_tables = _familywise_tables(rossi_results)

    acf_csv = _write_acf_table(rossi_results)
    block_csv = _write_block_table(rossi_results, family_tables)
    fig_paths = _plot_acf(rossi_results)
    npz_paths = _write_replicates(rossi_results)
    manifest_path = _write_manifest(rossi_results, family_tables, [acf_csv, block_csv], fig_paths, npz_paths)
    report_path = _write_report(rossi_results, family_tables, acf_csv, block_csv, fig_paths, npz_paths)

    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="experiment-a-rossi-block-bootstrap",
        role="experiment_a_block_bootstrap",
        params={
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "sign_flip_replicates": SIGN_FLIP_REPLICATES,
            "block_lengths": list(BLOCK_LENGTHS),
            "n_windows": int(len(vectors)),
            "official_p_value_convention": "two_sided_block_sign_flip",
            "compatibility_p_value_convention": "one_sided_observed_sign",
            "comparator_label": "bundled_threshold",
        },
        tags={
            "experiment": "A",
            "method": "rossi",
            "analysis": "moving_block_bootstrap_and_block_sign_flip",
        },
    ) as run:
        for key, result in rossi_results.items():
            mlflow.log_metric(f"{key}.delta_mean", result["mean"])
            mlflow.log_metric(f"{key}.acf_lag1", result["acf"][0])
            mlflow.log_metric(f"{key}.ljung_box_lag5_p", result["ljung_box"][5]["p_value"])
            mlflow.log_metric(f"{key}.ljung_box_lag10_p", result["ljung_box"][10]["p_value"])
            for block_len in BLOCK_LENGTHS:
                block = result["block"][block_len]
                mlflow.log_metric(f"{key}.L{block_len}.ci_low", block["ci_low"])
                mlflow.log_metric(f"{key}.L{block_len}.ci_high", block["ci_high"])
                mlflow.log_metric(f"{key}.L{block_len}.p_two_sided", block["p_two_sided"])
                mlflow.log_metric(f"{key}.L{block_len}.p_one_sided_observed", block["p_one_sided_observed"])
        for block_len, table in family_tables.items():
            for name, value in table["official_two_sided_holm"].items():
                mlflow.log_metric(f"L{block_len}.official_holm.{_metric_key(name)}", value)
            for name, value in table["compat_one_sided_holm"].items():
                mlflow.log_metric(f"L{block_len}.compat_holm.{_metric_key(name)}", value)
        for path in [acf_csv, block_csv, manifest_path, report_path, *fig_paths, *npz_paths]:
            artifact_path = _artifact_group(path)
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        for protocol in (
            ROOT / "00_MASTER_coordination.md",
            ROOT / "EXPERIMENT_A_block_bootstrap_rossi.md",
            ROOT / "EXPERIMENT_B_hpa_baseline_rossi.md",
            ROOT / "EXPERIMENT_C_action_ablation_deeprm.md",
        ):
            if protocol.exists():
                mlflow.log_artifact(str(protocol), artifact_path="protocol/new_experiments")
        print(f"MLflow run: {run.info.run_id}")
        print(str(report_path.relative_to(ROOT)))
        print(str(block_csv.relative_to(ROOT)))
        for path in fig_paths:
            print(str(path.relative_to(ROOT)))


def _ensure_dirs() -> None:
    for directory in (OUT_DIR, TABLE_DIR, FIG_DIR, DATA_DIR, ROOT_FIG_DIR, ROOT_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _analyze_rossi_cells(vectors: pd.DataFrame) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for idx, cell in enumerate(ROSSI_CELLS):
        diffs = vectors[cell.vector_column].to_numpy(dtype=np.float64)
        published = _published_row(cell)
        acf = _acf(diffs, max_lag=10)
        ljung = {lag: _ljung_box(acf, len(diffs), lag) for lag in (5, 10)}
        block = {
            block_len: _block_analysis(
                diffs,
                block_len=block_len,
                seed=SEED + 1000 * idx + block_len,
            )
            for block_len in BLOCK_LENGTHS
        }
        results[cell.key] = {
            "cell": cell,
            "differences": diffs,
            "mean": float(np.mean(diffs)),
            "iid_ci_low": float(published["ci_low"]),
            "iid_ci_high": float(published["ci_high"]),
            "iid_p_less": float(published["p_less_than_zero"]),
            "iid_p_greater": float(published["p_greater_than_zero"]),
            "iid_p_two_sided": _two_sided(
                float(published["p_less_than_zero"]),
                float(published["p_greater_than_zero"]),
            ),
            "iid_p_one_sided_observed": _observed_one_sided(
                float(np.mean(diffs)),
                float(published["p_less_than_zero"]),
                float(published["p_greater_than_zero"]),
            ),
            "acf": acf,
            "ljung_box": ljung,
            "block": block,
        }
    return results


def _published_row(cell: RossiCell) -> dict[str, str]:
    selector_column, selector_value = cell.published_selector
    with cell.published_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row[selector_column] == selector_value:
                return row
    raise ValueError(f"No row {selector_column}={selector_value} in {cell.published_path}")


def _acf(values: np.ndarray, *, max_lag: int) -> np.ndarray:
    centered = values - float(np.mean(values))
    denom = float(np.sum(centered * centered))
    if denom == 0.0:
        return np.full(max_lag, np.nan, dtype=np.float64)
    out = []
    for lag in range(1, max_lag + 1):
        out.append(float(np.sum(centered[:-lag] * centered[lag:]) / denom))
    return np.asarray(out, dtype=np.float64)


def _ljung_box(acf: np.ndarray, n: int, lag: int) -> dict[str, float]:
    rho = acf[:lag]
    q = float(n * (n + 2) * np.sum((rho * rho) / np.arange(n - 1, n - lag - 1, -1)))
    return {"statistic": q, "p_value": float(chi2.sf(q, df=lag))}


def _block_analysis(diffs: np.ndarray, *, block_len: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    blocks = np.asarray([diffs[start : start + block_len] for start in range(0, n - block_len + 1)])
    n_blocks_sampled = math.ceil(n / block_len)
    sample_idx = rng.integers(0, len(blocks), size=(BOOTSTRAP_REPLICATES, n_blocks_sampled))
    boot_means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        sample = blocks[sample_idx[i]].reshape(-1)[:n]
        boot_means[i] = float(np.mean(sample))

    nonoverlap_blocks = [diffs[start : min(start + block_len, n)] for start in range(0, n, block_len)]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(SIGN_FLIP_REPLICATES, len(nonoverlap_blocks)))
    null_means = np.empty(SIGN_FLIP_REPLICATES, dtype=np.float64)
    for i in range(SIGN_FLIP_REPLICATES):
        total = 0.0
        count = 0
        for sign, block in zip(signs[i], nonoverlap_blocks, strict=True):
            total += float(sign) * float(np.sum(block))
            count += len(block)
        null_means[i] = total / count

    observed = float(np.mean(diffs))
    tolerance = 1e-12 * max(1.0, abs(observed))
    p_two_sided = (np.count_nonzero(np.abs(null_means) + tolerance >= abs(observed)) + 1.0) / (
        SIGN_FLIP_REPLICATES + 1.0
    )
    if observed >= 0.0:
        p_one = (np.count_nonzero(null_means + tolerance >= observed) + 1.0) / (
            SIGN_FLIP_REPLICATES + 1.0
        )
    else:
        p_one = (np.count_nonzero(null_means - tolerance <= observed) + 1.0) / (
            SIGN_FLIP_REPLICATES + 1.0
        )

    return {
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "p_two_sided": float(p_two_sided),
        "p_one_sided_observed": float(p_one),
        "bootstrap_means": boot_means,
        "sign_flip_null_means": null_means,
        "num_sign_blocks": len(nonoverlap_blocks),
    }


def _familywise_tables(rossi_results: dict[str, dict[str, object]]) -> dict[int, dict[str, dict[str, float]]]:
    base_two_sided, base_one_sided = _non_rossi_family_pvalues()
    tables: dict[int, dict[str, dict[str, float]]] = {}
    for block_len in BLOCK_LENGTHS:
        official = dict(base_two_sided)
        compat = dict(base_one_sided)
        for result in rossi_results.values():
            cell: RossiCell = result["cell"]  # type: ignore[assignment]
            if cell.prediction_name is None:
                continue
            block = result["block"][block_len]  # type: ignore[index]
            official[cell.prediction_name] = float(block["p_two_sided"])
            compat[cell.prediction_name] = float(block["p_one_sided_observed"])
        tables[block_len] = {
            "official_two_sided_unadjusted": official,
            "official_two_sided_holm": holm_bonferroni(official),
            "compat_one_sided_unadjusted": compat,
            "compat_one_sided_holm": holm_bonferroni(compat),
        }
    return tables


def _non_rossi_family_pvalues() -> tuple[dict[str, float], dict[str, float]]:
    two_sided: dict[str, float] = {}
    one_sided: dict[str, float] = {}

    deeprm = json.loads(DEEPRM_SWEEP_PATH.read_text(encoding="utf-8"))["cells"]
    for name, key in {
        "P1-DeepRM": "P1_lag_10",
        "P2-DeepRM": "P2_tail_1.5",
        "P3-DeepRM": "P3_epsilon_0.05",
    }.items():
        comp = deeprm[key]["comparison"]
        p_less = float(comp["p_less_than_zero"])
        p_greater = float(comp["p_greater_than_zero"])
        observed = float(comp["mean_difference"])
        two_sided[name] = _two_sided(p_less, p_greater)
        one_sided[name] = _observed_one_sided(observed, p_less, p_greater)

    decima = pd.read_csv(DECIMA_SUMMARY_PATH)
    for name, prefix in {
        "P1-Decima": "P1-Decima",
        "P2-Decima": "P2-Decima",
        "P3-Decima": "P3-Decima",
    }.items():
        row = decima[decima["prediction"].str.startswith(prefix)].iloc[0]
        p_less = float(row["p_less"])
        p_greater = float(row["p_greater"])
        observed = float(row["delta_value"])
        two_sided[name] = _two_sided(p_less, p_greater)
        one_sided[name] = _observed_one_sided(observed, p_less, p_greater)

    return two_sided, one_sided


def _two_sided(p_less: float, p_greater: float) -> float:
    return min(1.0, 2.0 * min(p_less, p_greater))


def _observed_one_sided(observed: float, p_less: float, p_greater: float) -> float:
    return p_greater if observed >= 0.0 else p_less


def _write_acf_table(results: dict[str, dict[str, object]]) -> Path:
    path = TABLE_DIR / "rossi_acf_diagnostic.csv"
    fields = ["cell", *[f"acf_lag_{lag}" for lag in range(1, 11)], "lb5_statistic", "lb5_p", "lb10_statistic", "lb10_p"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results.values():
            cell: RossiCell = result["cell"]  # type: ignore[assignment]
            acf: np.ndarray = result["acf"]  # type: ignore[assignment]
            row = {"cell": cell.label}
            for lag, value in enumerate(acf, start=1):
                row[f"acf_lag_{lag}"] = value
            row["lb5_statistic"] = result["ljung_box"][5]["statistic"]  # type: ignore[index]
            row["lb5_p"] = result["ljung_box"][5]["p_value"]  # type: ignore[index]
            row["lb10_statistic"] = result["ljung_box"][10]["statistic"]  # type: ignore[index]
            row["lb10_p"] = result["ljung_box"][10]["p_value"]  # type: ignore[index]
            writer.writerow(row)
    return path


def _write_block_table(
    results: dict[str, dict[str, object]],
    family_tables: dict[int, dict[str, dict[str, float]]],
) -> Path:
    path = TABLE_DIR / "rossi_block_bootstrap_results.csv"
    fields = [
        "cell",
        "delta_anchor",
        "iid_ci_low",
        "iid_ci_high",
        "iid_p_one_sided_observed",
        "iid_p_two_sided",
        "block_L5_ci_low",
        "block_L5_ci_high",
        "block_L5_ci_width_ratio_vs_iid",
        "block_L5_p_two_sided",
        "block_L5_p_one_sided_observed",
        "block_L5_holm_two_sided",
        "block_L5_holm_one_sided_compat",
        "block_L10_ci_low",
        "block_L10_ci_high",
        "block_L10_ci_width_ratio_vs_iid",
        "block_L10_p_two_sided",
        "block_L10_p_one_sided_observed",
        "block_L10_holm_two_sided",
        "block_L10_holm_one_sided_compat",
        "zero_containment_changed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results.values():
            cell: RossiCell = result["cell"]  # type: ignore[assignment]
            row: dict[str, object] = {
                "cell": cell.label,
                "delta_anchor": result["mean"],
                "iid_ci_low": result["iid_ci_low"],
                "iid_ci_high": result["iid_ci_high"],
                "iid_p_one_sided_observed": result["iid_p_one_sided_observed"],
                "iid_p_two_sided": result["iid_p_two_sided"],
            }
            iid_contains_zero = _contains_zero(float(result["iid_ci_low"]), float(result["iid_ci_high"]))
            changed = False
            for block_len in BLOCK_LENGTHS:
                block = result["block"][block_len]  # type: ignore[index]
                width_ratio = (float(block["ci_high"]) - float(block["ci_low"])) / (
                    float(result["iid_ci_high"]) - float(result["iid_ci_low"])
                )
                row[f"block_L{block_len}_ci_low"] = block["ci_low"]
                row[f"block_L{block_len}_ci_high"] = block["ci_high"]
                row[f"block_L{block_len}_ci_width_ratio_vs_iid"] = width_ratio
                row[f"block_L{block_len}_p_two_sided"] = block["p_two_sided"]
                row[f"block_L{block_len}_p_one_sided_observed"] = block["p_one_sided_observed"]
                if cell.prediction_name:
                    row[f"block_L{block_len}_holm_two_sided"] = family_tables[block_len][
                        "official_two_sided_holm"
                    ][cell.prediction_name]
                    row[f"block_L{block_len}_holm_one_sided_compat"] = family_tables[block_len][
                        "compat_one_sided_holm"
                    ][cell.prediction_name]
                else:
                    row[f"block_L{block_len}_holm_two_sided"] = "n/a"
                    row[f"block_L{block_len}_holm_one_sided_compat"] = "n/a"
                changed = changed or (
                    iid_contains_zero != _contains_zero(float(block["ci_low"]), float(block["ci_high"]))
                )
            row["zero_containment_changed"] = changed
            writer.writerow(row)
    return path


def _plot_acf(results: dict[str, dict[str, object]]) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    band = 1.96 / math.sqrt(30)
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.3), sharex=True, sharey=True)
    axes_flat = axes.reshape(-1)
    for ax, result in zip(axes_flat, results.values(), strict=True):
        cell: RossiCell = result["cell"]  # type: ignore[assignment]
        acf: np.ndarray = result["acf"]  # type: ignore[assignment]
        lags = np.arange(1, 11)
        ax.axhline(0.0, color="#333333", linewidth=0.7)
        ax.axhline(band, color="#8f2d2d", linestyle="--", linewidth=0.8)
        ax.axhline(-band, color="#8f2d2d", linestyle="--", linewidth=0.8)
        ax.bar(lags, acf, color="#4c78a8", width=0.68)
        ax.set_title(cell.label)
        ax.set_xticks(lags)
        ax.set_ylim(-0.55, 0.55)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_xlabel("Lag")
    axes[1, 1].set_xlabel("Lag")
    axes[0, 0].set_ylabel("Autocorrelation")
    axes[1, 0].set_ylabel("Autocorrelation")
    fig.suptitle("Rossi Per-Window Delta Autocorrelation", y=0.99, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf = FIG_DIR / "rossi_acf_diagnostic.pdf"
    png = FIG_DIR / "rossi_acf_diagnostic.png"
    root_pdf = ROOT_FIG_DIR / "rossi_acf_diagnostic.pdf"
    root_png = ROOT_FIG_DIR / "rossi_acf_diagnostic.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    fig.savefig(root_pdf)
    fig.savefig(root_png, dpi=300)
    plt.close(fig)
    return [pdf, png, root_pdf, root_png]


def _write_replicates(results: dict[str, dict[str, object]]) -> list[Path]:
    paths: list[Path] = []
    for block_len in BLOCK_LENGTHS:
        payload = {}
        for key, result in results.items():
            block = result["block"][block_len]  # type: ignore[index]
            payload[f"{key}_bootstrap_means"] = block["bootstrap_means"]
            payload[f"{key}_sign_flip_null_means"] = block["sign_flip_null_means"]
            payload[f"{key}_observed_mean"] = np.asarray([result["mean"]], dtype=np.float64)
        for directory in (DATA_DIR, ROOT_DATA_DIR):
            path = directory / f"experiment_a_replicates_L{block_len}.npz"
            np.savez_compressed(path, **payload)
            paths.append(path)
    return paths


def _write_manifest(
    results: dict[str, dict[str, object]],
    family_tables: dict[int, dict[str, dict[str, float]]],
    table_paths: Iterable[Path],
    fig_paths: Iterable[Path],
    npz_paths: Iterable[Path],
) -> Path:
    path = OUT_DIR / "experiment_a_manifest.json"
    payload = {
        "experiment": "A",
        "method": "Rossi",
        "input_vectors": str(VECTOR_PATH.relative_to(ROOT)),
        "n_windows": int(len(next(iter(results.values()))["differences"])),
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sign_flip_replicates": SIGN_FLIP_REPLICATES,
        "block_lengths": list(BLOCK_LENGTHS),
        "official_p_value_convention": "two-sided block sign-flip",
        "compatibility_p_value_convention": "one-sided observed-sign block sign-flip",
        "family_tables": family_tables,
        "tables": [str(path.relative_to(ROOT)) for path in table_paths],
        "figures": [str(path.relative_to(ROOT)) for path in fig_paths],
        "replicate_files": [str(path.relative_to(ROOT)) for path in npz_paths],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _write_report(
    results: dict[str, dict[str, object]],
    family_tables: dict[int, dict[str, dict[str, float]]],
    acf_csv: Path,
    block_csv: Path,
    fig_paths: list[Path],
    npz_paths: list[Path],
) -> Path:
    path = OUT_DIR / "experiment_a_results.md"
    root_copy = ROOT / "experiment_a_results.md"
    lines = [
        "# Experiment A Results — Rossi Block-Bootstrap Sensitivity",
        "",
        "This analysis uses the existing 30 ordered Rossi evaluation windows. No simulator, training, or controller run was executed.",
        "",
        "Comparator label: `bundled_threshold` for the RLAD simulator's bundled threshold controller. Older files retain `hpa` in column names, but this report does not interpret that controller as Kubernetes HPA.",
        "",
        "Official Experiment A p-values are two-sided block sign-flip p-values. One-sided observed-sign p-values are reported only as a compatibility column with the main-paper convention.",
        "",
        "## Part A1 — Autocorrelation Diagnostic",
        "",
        f"Bartlett 95% large-sample band: ±{1.96 / math.sqrt(30):.3f}.",
        "",
            "| Cell | rho1 | max abs rho(1..10) | LB(5) p | LB(10) p | Diagnostic |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results.values():
        cell: RossiCell = result["cell"]  # type: ignore[assignment]
        acf: np.ndarray = result["acf"]  # type: ignore[assignment]
        max_abs = float(np.max(np.abs(acf)))
        lb5 = float(result["ljung_box"][5]["p_value"])  # type: ignore[index]
        lb10 = float(result["ljung_box"][10]["p_value"])  # type: ignore[index]
        diagnostic = _acf_diagnostic(float(acf[0]), lb5, lb10)
        lines.append(
            f"| {cell.label} | {acf[0]:.3f} | {max_abs:.3f} | {lb5:.3g} | {lb10:.3g} | {diagnostic} |"
        )

    lines.extend(
        [
            "",
            "## Part A2 — Block Bootstrap and Block Sign-Flip",
            "",
            "| Cell | Δ | iid 95% CI | MBB 95% CI L=5 | MBB 95% CI L=10 | iid p(one-sided obs.) | block p2s L=5 | block p2s L=10 | Holm p2s L=5 | Holm p2s L=10 | block p1s compat L=5 | block p1s compat L=10 | Holm p1s compat L=5 | Holm p1s compat L=10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results.values():
        cell: RossiCell = result["cell"]  # type: ignore[assignment]
        b5 = result["block"][5]  # type: ignore[index]
        b10 = result["block"][10]  # type: ignore[index]
        holm5 = "n/a"
        holm10 = "n/a"
        holm5_compat = "n/a"
        holm10_compat = "n/a"
        if cell.prediction_name:
            holm5 = f"{family_tables[5]['official_two_sided_holm'][cell.prediction_name]:.4g}"
            holm10 = f"{family_tables[10]['official_two_sided_holm'][cell.prediction_name]:.4g}"
            holm5_compat = f"{family_tables[5]['compat_one_sided_holm'][cell.prediction_name]:.4g}"
            holm10_compat = f"{family_tables[10]['compat_one_sided_holm'][cell.prediction_name]:.4g}"
        lines.append(
            "| {label} | {mean:.3f} | [{iid_lo:.3f}, {iid_hi:.3f}] | "
            "[{b5_lo:.3f}, {b5_hi:.3f}] | [{b10_lo:.3f}, {b10_hi:.3f}] | "
            "{iid_p:.4g} | {b5_p2:.4g} | {b10_p2:.4g} | {holm5} | {holm10} | "
            "{b5_p1:.4g} | {b10_p1:.4g} | {holm5_compat} | {holm10_compat} |".format(
                label=cell.label,
                mean=float(result["mean"]),
                iid_lo=float(result["iid_ci_low"]),
                iid_hi=float(result["iid_ci_high"]),
                b5_lo=float(b5["ci_low"]),
                b5_hi=float(b5["ci_high"]),
                b10_lo=float(b10["ci_low"]),
                b10_hi=float(b10["ci_high"]),
                iid_p=float(result["iid_p_one_sided_observed"]),
                b5_p2=float(b5["p_two_sided"]),
                b10_p2=float(b10["p_two_sided"]),
                holm5=holm5,
                holm10=holm10,
                b5_p1=float(b5["p_one_sided_observed"]),
                b10_p1=float(b10["p_one_sided_observed"]),
                holm5_compat=holm5_compat,
                holm10_compat=holm10_compat,
            )
        )

    lines.extend(
        [
            "",
            "## Familywise Interpretation",
            "",
            _family_interpretation(results, family_tables),
            "",
            "## Artifact Paths",
            "",
            f"- ACF CSV: `{acf_csv.relative_to(ROOT)}`",
            f"- Block-bootstrap CSV: `{block_csv.relative_to(ROOT)}`",
        ]
    )
    for fig_path in fig_paths:
        lines.append(f"- Figure: `{fig_path.relative_to(ROOT)}`")
    for npz_path in npz_paths:
        lines.append(f"- Replicates: `{npz_path.relative_to(ROOT)}`")

    report = "\n".join(lines) + "\n"
    path.write_text(report, encoding="utf-8")
    root_copy.write_text(report, encoding="utf-8")
    return path


def _family_interpretation(
    results: dict[str, dict[str, object]],
    family_tables: dict[int, dict[str, dict[str, float]]],
) -> str:
    rho_flags = []
    zero_flags = []
    width_flags = []
    for result in results.values():
        cell: RossiCell = result["cell"]  # type: ignore[assignment]
        acf: np.ndarray = result["acf"]  # type: ignore[assignment]
        if abs(float(acf[0])) >= 0.3:
            rho_flags.append(cell.label)
        iid_contains = _contains_zero(float(result["iid_ci_low"]), float(result["iid_ci_high"]))
        iid_width = float(result["iid_ci_high"]) - float(result["iid_ci_low"])
        for block_len in BLOCK_LENGTHS:
            block = result["block"][block_len]  # type: ignore[index]
            if iid_contains != _contains_zero(float(block["ci_low"]), float(block["ci_high"])):
                zero_flags.append(f"{cell.label} L={block_len}")
            ratio = (float(block["ci_high"]) - float(block["ci_low"])) / iid_width
            if not (0.8 <= ratio <= 1.2):
                width_flags.append(f"{cell.label} L={block_len} width ratio {ratio:.2f}")

    verdict_changes = []
    compat_verdict_changes = []
    for block_len in BLOCK_LENGTHS:
        for name in ("P1-Rossi", "P2-Rossi", "P3-Rossi"):
            if family_tables[block_len]["official_two_sided_holm"][name] >= 0.05:
                verdict_changes.append(f"{name} L={block_len}")
            if family_tables[block_len]["compat_one_sided_holm"][name] >= 0.05:
                compat_verdict_changes.append(f"{name} L={block_len}")
    decima_p2_flags = [
        block_len
        for block_len in BLOCK_LENGTHS
        if family_tables[block_len]["official_two_sided_holm"]["P2-Decima"] >= 0.05
    ]

    parts = []
    if rho_flags:
        parts.append("Lag-1 autocorrelation exceeds the pre-registered 0.3 threshold for: " + ", ".join(rho_flags) + ".")
    else:
        parts.append("All four Rossi cells satisfy the pre-registered lag-1 autocorrelation threshold |rho1| < 0.3.")
    if zero_flags:
        parts.append("At least one block-bootstrap CI changes zero-containment status: " + ", ".join(zero_flags) + ".")
    else:
        parts.append("No Rossi block-bootstrap CI changes zero-containment status relative to the iid CI.")
    if width_flags:
        parts.append("Some MBB intervals differ from iid width by more than 20%: " + "; ".join(width_flags) + ".")
    else:
        parts.append("All Rossi MBB CI widths are within 20% of the iid CI widths.")
    if verdict_changes:
        parts.append(
            "Under the official two-sided block sign-flip plus Holm convention, Rossi familywise rejections do not survive for: "
            + ", ".join(verdict_changes)
            + ". This is driven by the discreteness of block sign-flip tests with only 6 sign blocks at L=5 and 3 at L=10; it should be reported as a conservative sensitivity result, not as evidence that the point estimates are small."
        )
    else:
        parts.append("The Rossi familywise verdicts survive the official two-sided block sign-flip Holm sensitivity.")
    if compat_verdict_changes:
        parts.append(
            "Even under the one-sided observed-sign compatibility convention, the Rossi block-sign familywise values do not pass Holm for: "
            + ", ".join(compat_verdict_changes)
            + "."
        )
    if decima_p2_flags:
        parts.append(
            "As expected from the p-value convention decision, Decima P2 also does not survive the official two-sided Holm convention at L="
            + "/".join(str(x) for x in decima_p2_flags)
            + "; the main-paper row should be footnoted as marginal under two-sided sensitivity."
        )
    return " ".join(parts)


def _acf_diagnostic(rho1: float, lb5: float, lb10: float) -> str:
    flags = []
    if abs(rho1) >= 0.3:
        flags.append("|rho1| >= 0.3")
    if lb5 < 0.05:
        flags.append("LB5 p < 0.05")
    if lb10 < 0.05:
        flags.append("LB10 p < 0.05")
    return "ok" if not flags else "; ".join(flags)


def _contains_zero(low: float, high: float) -> bool:
    return low <= 0.0 <= high


def _artifact_group(path: Path) -> str:
    if path.suffix == ".npz":
        return "experiment_a/data"
    if path.suffix in {".pdf", ".png"}:
        return "experiment_a/figures"
    if path.suffix == ".csv":
        return "experiment_a/tables"
    return "experiment_a"


def _metric_key(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return str(value)


if __name__ == "__main__":
    main()
