#!/usr/bin/env python3
"""Run and log Decima preflight/readiness checks to MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from cisose_common.tracking import start_run, write_json_artifact
from cisose_decima.cli import EXPERIMENT_NAME
from cisose_decima.config import DECIMA_COMMIT, DECIMA_REPO_URL, DEFAULT_CONFIG
from cisose_decima.gates import current_readiness
from cisose_decima.model import DecimaPolicy, parameter_count
from cisose_decima.preflight import decima_preflight_report
from cisose_decima.reproduction import reference_command_payload


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = decima_preflight_report(ROOT)
    readiness = current_readiness()
    policy = DecimaPolicy()
    payload = {
        "scope": "decima_start_preflight",
        "preflight": report,
        "readiness": {
            "ready_for_perturbations": readiness.ready_for_perturbations,
            "official_readme_reproduction": readiness.official_readme_reproduction.__dict__,
            "graphene_validation": readiness.graphene_validation.__dict__,
        },
        "reference_commands": reference_command_payload(),
        "policy_parameter_count": parameter_count(policy),
        "start_decision": {
            "start_allowed": True,
            "allowed_scope": [
                "official README reproduction implementation/execution",
                "Graphene comparator validation research",
                "MLflow-tracked smoke/preflight diagnostics",
            ],
            "not_allowed_scope": [
                "Decima P1/P2/P3 perturbation cells",
                "paper claims using GrapheneStyleComparator as faithful Graphene",
            ],
        },
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="decima-start-preflight",
        role="preflight",
        params={
            "method": "decima",
            "decima_repo_url": DECIMA_REPO_URL,
            "decima_commit": DECIMA_COMMIT,
            "exec_cap": DEFAULT_CONFIG.exec_cap,
            "policy_parameter_count": payload["policy_parameter_count"],
            "ready_for_perturbations": readiness.ready_for_perturbations,
            "official_readme_reproduction_passed": readiness.official_readme_reproduction.passed,
            "graphene_validation_passed": readiness.graphene_validation.passed,
        },
    ) as run:
        mlflow.log_metric("policy_parameter_count", payload["policy_parameter_count"])
        mlflow.log_metric(
            "ready_for_perturbations",
            1.0 if readiness.ready_for_perturbations else 0.0,
        )
        mlflow.log_metric(
            "official_readme_reproduction_passed",
            1.0 if readiness.official_readme_reproduction.passed else 0.0,
        )
        mlflow.log_metric(
            "graphene_validation_passed",
            1.0 if readiness.graphene_validation.passed else 0.0,
        )
        out = ROOT / "results" / "decima" / "start_preflight.json"
        write_json_artifact(out, payload, run_id=run.info.run_id)
        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
