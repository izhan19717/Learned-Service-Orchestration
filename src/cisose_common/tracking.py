"""Small MLflow helpers shared by Rossi and Decima implementations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow


ACTIVE_DOC_FILES = (
    "docs/protocols/calibration_v2_2.md",
    "docs/protocols/preregistration_v2_2.md",
    "docs/protocols/protocol_amendment_v2_2.md",
    "docs/protocols/protocol_amendment_decima_simulator_gate.md",
    "docs/implementation_notes.md",
    "docs/PROTOCOL_INDEX.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if path.exists() and path.is_file()}


def source_protocol_manifest(root: Path) -> dict[str, str]:
    paths = [root / name for name in ACTIVE_DOC_FILES]
    for folder in ("src", "configs"):
        base = root / folder
        if base.exists():
            paths.extend(sorted(base.rglob("*.py")))
            paths.extend(sorted(base.rglob("*.yaml")))
    return manifest(paths)


def environment_summary() -> dict[str, str]:
    summary = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for cmd, key in [
        (["/usr/bin/lscpu"], "lscpu"),
        (["/usr/bin/free", "-h"], "memory"),
        (["/usr/lib/wsl/lib/nvidia-smi"], "nvidia_smi"),
    ]:
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
            summary[key] = result.stdout.strip()[:4000]
        except Exception as exc:  # pragma: no cover - provenance only
            summary[key] = f"unavailable: {exc}"
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["torch_cuda_built"] = str(torch.version.cuda)
        summary["torch_cuda_available"] = str(torch.cuda.is_available())
        summary["torch_cuda_device_count"] = str(torch.cuda.device_count())
        if torch.cuda.is_available():
            summary["torch_cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - optional dependency
        summary["torch"] = f"unavailable: {exc}"
    return summary


def configure_mlflow(root: Path, experiment_name: str) -> None:
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str(root / tracking_uri.removeprefix("sqlite:///"))
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            experiment_name,
            artifact_location=str(root / "mlartifacts" / experiment_name),
        )
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(
    *,
    root: Path,
    experiment_name: str,
    run_name: str,
    role: str,
    params: dict[str, object] | None = None,
    tags: dict[str, object] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    configure_mlflow(root, experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        all_tags = {
            "protocol.version": "v2.2",
            "run.role": role,
        }
        if tags:
            all_tags.update({k: str(v) for k, v in tags.items()})
        mlflow.set_tags(all_tags)
        if params:
            mlflow.log_params({k: _stringify(v) for k, v in params.items()})
        _log_provenance(root)
        yield run


def write_json_artifact(path: Path, payload: dict[str, object], *, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched["mlflow_run_id"] = run_id
    path.write_text(json.dumps(enriched, indent=2, sort_keys=True, default=str) + "\n")
    mlflow.log_artifact(str(path), artifact_path="results")


def _stringify(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _log_provenance(root: Path) -> None:
    out_dir = root / "logs" / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "v2_2_manifest.json"
    env_path = out_dir / "v2_2_environment.json"
    manifest_path.write_text(
        json.dumps(source_protocol_manifest(root), indent=2, sort_keys=True) + "\n"
    )
    env_path.write_text(json.dumps(environment_summary(), indent=2, sort_keys=True) + "\n")
    mlflow.log_artifact(str(manifest_path), artifact_path="provenance")
    mlflow.log_artifact(str(env_path), artifact_path="provenance")
    for name in ACTIVE_DOC_FILES:
        path = root / name
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="protocol")
