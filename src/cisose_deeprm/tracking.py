"""MLflow tracking and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import mlflow

from cisose_deeprm import __version__
from cisose_deeprm.protocol import DOC_FILES, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, PROTOCOL_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(paths: Iterable[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            out[str(path)] = sha256_file(path)
    return out


def protocol_manifest(root: Path) -> dict[str, str]:
    files = [root / name for name in DOC_FILES]
    files.extend(sorted((root / "src").rglob("*.py")) if (root / "src").exists() else [])
    files.extend(sorted((root / "configs").rglob("*.yaml")) if (root / "configs").exists() else [])
    return manifest(files)


def environment_summary() -> dict[str, str]:
    summary = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "package_version": __version__,
    }
    for cmd, key in [
        (["/usr/bin/lscpu"], "lscpu"),
        (["/usr/bin/free", "-h"], "memory"),
        (["/usr/lib/wsl/lib/nvidia-smi"], "nvidia_smi"),
    ]:
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
            summary[key] = result.stdout.strip()[:4000]
        except Exception as exc:  # pragma: no cover - defensive provenance only
            summary[key] = f"unavailable: {exc}"
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["torch_cuda_built"] = str(torch.version.cuda)
        summary["torch_cuda_available"] = str(torch.cuda.is_available())
        summary["torch_cuda_device_count"] = str(torch.cuda.device_count())
        if torch.cuda.is_available():
            summary["torch_cuda_device_0"] = torch.cuda.get_device_name(0)
            summary["torch_cuda_capability_0"] = str(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - optional dependency diagnostics
        summary["torch_cuda"] = f"unavailable: {exc}"
    return summary


def configure_mlflow(root: Path) -> None:
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    if tracking_uri.startswith("file:./"):
        tracking_uri = "file:" + str(root / tracking_uri.removeprefix("file:./"))
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = root / tracking_uri.removeprefix("sqlite:///")
        tracking_uri = "sqlite:///" + str(db_path)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=str(root / "mlartifacts" / MLFLOW_EXPERIMENT),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


@contextmanager
def start_tracked_run(
    *,
    run_name: str,
    role: str,
    root: Path | None = None,
    nested: bool = False,
    params: dict[str, object] | None = None,
    tags: dict[str, object] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    root = root or Path.cwd()
    configure_mlflow(root)
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        all_tags = {
            "protocol.version": PROTOCOL_VERSION,
            "run.role": role,
            "package.version": __version__,
        }
        if tags:
            all_tags.update({k: str(v) for k, v in tags.items()})
        mlflow.set_tags(all_tags)
        if params:
            mlflow.log_params({k: _stringify_param(v) for k, v in params.items()})
        _log_manifest_and_environment(root)
        yield run


def _stringify_param(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _log_manifest_and_environment(root: Path) -> None:
    artifact_dir = root / "logs" / "provenance"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.json"
    env_path = artifact_dir / "environment.json"
    manifest_path.write_text(json.dumps(protocol_manifest(root), indent=2, sort_keys=True) + "\n")
    env_path.write_text(json.dumps(environment_summary(), indent=2, sort_keys=True) + "\n")
    mlflow.log_artifact(str(manifest_path), artifact_path="provenance")
    mlflow.log_artifact(str(env_path), artifact_path="provenance")
    decisions = root / "SCIENTIFIC_DECISIONS.md"
    if decisions.exists():
        mlflow.log_artifact(str(decisions), artifact_path="protocol")


def write_json_with_run_id(path: Path, payload: dict[str, object], run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["mlflow_run_id"] = run_id
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    mlflow.log_artifact(str(path), artifact_path="results")
