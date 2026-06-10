#!/usr/bin/env python3
"""Diagnose Rossi frozen-checkpoint clean performance on the Table I window."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from cisose_common.tracking import sha256_file, start_run, write_json_artifact
from cisose_rossi.checkpointing import load_model_based_checkpoint
from cisose_rossi.config import DEFAULT_CONFIG, PROFILE_SHA256, RLAD_COMMIT, RLAD_REPO_URL
from cisose_rossi.controllers import ThresholdHPAController
from cisose_rossi.evaluation import metrics, table_i_metrics
from cisose_rossi.simulator import RladSimulator
from cisose_rossi.workload import java_slow_profile_sequence, load_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_rossi_v2_2"
PROFILE_PATH = ROOT / "external" / "rlad-core-simulator" / "data" / "profile.dat"
CHECKPOINT_PATH = ROOT / "results" / "rossi" / "checkpoints" / "model_based_table_i_clean.npz"


def main() -> None:
    profile = load_profile(PROFILE_PATH)
    rates = java_slow_profile_sequence(profile, steps=DEFAULT_CONFIG.time_limit + 1)
    rossi, metadata = load_model_based_checkpoint(CHECKPOINT_PATH, freeze=True)
    rossi_records = RladSimulator(DEFAULT_CONFIG).run(
        rossi,
        rates,
        horizon=DEFAULT_CONFIG.time_limit + 1,
    )
    hpa_records = RladSimulator(DEFAULT_CONFIG).run(
        ThresholdHPAController(DEFAULT_CONFIG),
        rates,
        horizon=DEFAULT_CONFIG.time_limit + 1,
    )
    rossi_metrics = metrics(rossi_records)
    hpa_metrics = metrics(hpa_records)
    result = {
        "diagnostic": "rossi_frozen_checkpoint_clean_first_window",
        "interpretation": (
            "The Table I gate reproduces the online model-based trajectory. "
            "This diagnostic evaluates the final checkpoint frozen on the same first window."
        ),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "checkpoint_metadata": metadata,
        "profile_sha256": PROFILE_SHA256,
        "rlad_commit": RLAD_COMMIT,
        "horizon": DEFAULT_CONFIG.time_limit + 1,
        "rossi_total_cost": rossi_metrics.total_cost,
        "hpa_total_cost": hpa_metrics.total_cost,
        "delta_hpa_minus_rossi": hpa_metrics.total_cost - rossi_metrics.total_cost,
        "rossi_table_i_metrics": table_i_metrics(rossi_records).__dict__,
        "hpa_table_i_metrics": table_i_metrics(hpa_records).__dict__,
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="rossi-frozen-clean-diagnostic",
        role="diagnostic",
        params={
            "method": "rossi_rlad",
            "diagnostic": result["diagnostic"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "rlad_repo_url": RLAD_REPO_URL,
            "rlad_commit": RLAD_COMMIT,
            "profile_sha256": PROFILE_SHA256,
        },
    ) as run:
        mlflow.log_metric("rossi_total_cost", rossi_metrics.total_cost)
        mlflow.log_metric("hpa_total_cost", hpa_metrics.total_cost)
        mlflow.log_metric("delta_hpa_minus_rossi", result["delta_hpa_minus_rossi"])
        mlflow.log_metric(
            "rossi_rmax_violations_pct",
            result["rossi_table_i_metrics"]["rmax_violations_pct"],
        )
        out = ROOT / "results" / "rossi" / "frozen_checkpoint_clean_diagnostic.json"
        write_json_artifact(out, result, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
