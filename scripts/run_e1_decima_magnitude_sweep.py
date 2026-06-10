#!/usr/bin/env python3
"""E1 Decima perturbation-magnitude sweep orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow

from cisose_common.stats import holm_bonferroni
from cisose_common.tracking import start_run, write_json_artifact


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "cisose_e1_magnitude_sweep"
IMAGE = "cisose-decima-tf1:1.15.5"
OUT_DIR = ROOT / "results" / "paper" / "experiments" / "e1_magnitude_sweep" / "decima"
TABLE_DIR = OUT_DIR / "tables"
DATA_DIR = OUT_DIR / "data"

LAG_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
TAIL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FGSM_EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-exp", type=int, default=30)
    parser.add_argument("--num-stream-dags", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=99017)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument(
        "--saved-model",
        type=Path,
        default=Path("results/checkpoints/decima/official_tf1_readme/model_ep_10000"),
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--build-image", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def docker_env() -> dict[str, str]:
    env = os.environ.copy()
    if "DOCKER_HOST" not in env and Path(f"/run/user/{os.getuid()}/docker.sock").exists():
        env["DOCKER_HOST"] = f"unix:///run/user/{os.getuid()}/docker.sock"
    return env


def ensure_image(image: str, *, build: bool) -> None:
    env = docker_env()
    check = subprocess.run(["docker", "image", "inspect", image], cwd=ROOT, env=env, capture_output=True, text=True)
    if check.returncode == 0:
        return
    if not build:
        raise RuntimeError(f"Docker image {image!r} not found and --no-build-image set")
    # Rootless Docker in the remote execution environment can lose DNS inside
    # the build sandbox while the host resolver works. Host networking affects
    # only package resolution during image construction, not the
    # simulator/evaluation semantics.
    subprocess.run(
        ["docker", "build", "--network=host", "-t", image, "docker/decima-tf1"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def cell_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for value in LAG_LAMBDAS:
        specs.append({"curve": "p1_lag", "perturbation": "lag", "magnitude": value, "arg": "--lag-lambda"})
    for value in TAIL_WEIGHTS:
        specs.append({"curve": "p2_tail", "perturbation": "tail", "magnitude": value, "arg": "--tail-weight"})
    for value in FGSM_EPSILONS:
        specs.append({"curve": "p3_fgsm", "perturbation": "fgsm", "magnitude": value, "arg": "--fgsm-epsilon"})
    return specs


def safe_value(value: object) -> str:
    return str(value).replace(".", "_").replace("-", "m")


def run_cell(spec: dict[str, object], args: argparse.Namespace) -> Path:
    perturbation = str(spec["perturbation"])
    magnitude = spec["magnitude"]
    curve = str(spec["curve"])
    stem = f"decima_e1_{curve}_{safe_value(magnitude)}"
    output_json = TABLE_DIR / f"{stem}.json"
    output_csv = DATA_DIR / f"{stem}_raw.csv"
    output_md = TABLE_DIR / f"{stem}.md"
    log_file = ROOT / "logs" / "training" / f"{stem}.log"
    if args.resume_existing and output_json.exists():
        try:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            if "aggregate" in payload:
                print(f"Skipping existing complete cell {stem}", flush=True)
                return output_json
        except Exception:
            pass
    command = [
        sys.executable,
        "scripts/run_decima_tf1_perturb_eval.py",
        "--image",
        args.image,
        "--container-name",
        f"cisose_{stem}",
        "--perturbation",
        perturbation,
        str(spec["arg"]),
        str(magnitude),
        "--num-exp",
        str(args.num_exp),
        "--num-stream-dags",
        str(args.num_stream_dags),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--saved-model",
        str(args.saved_model),
        "--output-json",
        str(output_json.relative_to(ROOT)),
        "--output-csv",
        str(output_csv.relative_to(ROOT)),
        "--output-md",
        str(output_md.relative_to(ROOT)),
        "--log-file",
        str(log_file.relative_to(ROOT)),
        "--force-remove-container",
    ]
    print("Running " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=docker_env(), check=True)
    return output_json


def summarize(json_paths: list[Path], run_id: str, params: dict[str, object]) -> None:
    rows = []
    p_less_by_curve: dict[str, dict[str, float]] = {}
    p_greater_by_curve: dict[str, dict[str, float]] = {}
    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        agg = payload["aggregate"]
        perturbation = payload["perturbation"]
        if perturbation == "lag":
            curve = "p1_lag"
            magnitude = payload["lag_lambda"]
        elif perturbation == "tail":
            curve = "p2_tail"
            magnitude = payload["tail_weight"]
        else:
            curve = "p3_fgsm"
            magnitude = payload["fgsm_epsilon"]
        key = f"{curve}:{magnitude}"
        p_less_by_curve.setdefault(curve, {})[key] = float(agg["p_less"])
        p_greater_by_curve.setdefault(curve, {})[key] = float(agg["p_greater"])
        rows.append(
            {
                "curve": curve,
                "magnitude": magnitude,
                "dynamic_partition_mean_jct": agg["dynamic_partition_mean_jct"],
                "decima_mean_jct": agg["learn_mean_jct"],
                "delta_dynamic_partition_minus_decima": agg["delta_mean"],
                "ci_low": agg["delta_ci_low"],
                "ci_high": agg["delta_ci_high"],
                "p_less_than_zero": agg["p_less"],
                "p_greater_than_zero": agg["p_greater"],
                "source_json": str(path.relative_to(ROOT)),
            }
        )
    holm_less = {curve: holm_bonferroni(vals) for curve, vals in p_less_by_curve.items()}
    holm_greater = {curve: holm_bonferroni(vals) for curve, vals in p_greater_by_curve.items()}
    for row in rows:
        key = f"{row['curve']}:{row['magnitude']}"
        row["holm_less_curve"] = holm_less[row["curve"]][key]
        row["holm_greater_curve"] = holm_greater[row["curve"]][key]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "e1_decima_magnitude_sweep.csv"
    md_path = OUT_DIR / "e1_decima_magnitude_sweep.md"
    json_path = OUT_DIR / "e1_decima_magnitude_sweep.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json_artifact(
        json_path,
        {"experiment": "E1", "method": "decima", "params": params, "summary": rows},
        run_id=run_id,
    )
    lines = [
        "# E1 Decima Magnitude Sweep",
        "",
        f"MLflow parent run: `{run_id}`",
        "",
        "| Curve | Magnitude | Delta dynamic_partition-Decima | 95% CI | Holm p(Delta<0) | Holm p(Delta>0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {curve} | {magnitude} | {delta_dynamic_partition_minus_decima:.6g} | "
            "[{ci_low:.6g}, {ci_high:.6g}] | {holm_less_curve:.6g} | {holm_greater_curve:.6g} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(f"- CSV: `{csv_path.relative_to(ROOT)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(csv_path), artifact_path="paper/tables")
    mlflow.log_artifact(str(md_path), artifact_path="paper")


def main() -> None:
    args = parse_args()
    for directory in (OUT_DIR, TABLE_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    params = {
        "protocol": "PREREG_E1_magnitude_sweep",
        "method": "decima",
        "num_exp": args.num_exp,
        "num_stream_dags": args.num_stream_dags,
        "bootstrap_seed": args.bootstrap_seed,
        "image": args.image,
        "saved_model": str(args.saved_model),
        "lag_lambdas": list(LAG_LAMBDAS),
        "tail_weights": list(TAIL_WEIGHTS),
        "fgsm_epsilons": list(FGSM_EPSILONS),
        "compute_policy": "canonical on remote workstation; local workstation only for smoke",
    }
    with start_run(
        root=ROOT,
        experiment_name=EXPERIMENT_NAME,
        run_name="e1-decima-magnitude-sweep-parent",
        role="e1_magnitude_sweep_parent",
        params=params,
        tags={"experiment": "E1", "method": "decima"},
    ) as run:
        ensure_image(args.image, build=args.build_image)
        paths = [run_cell(spec, args) for spec in cell_specs()]
        summarize(paths, run.info.run_id, params)
        print(f"MLflow run: {run.info.run_id}")
        print(str((OUT_DIR / "e1_decima_magnitude_sweep.md").relative_to(ROOT)))


if __name__ == "__main__":
    main()
