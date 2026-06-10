"""Pre-execution checks for Decima readiness."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

from cisose_decima.reproduction import reference_command_payload
from cisose_decima.tpch import load_tpch_templates, template_summary


def decima_preflight_report(root: Path) -> dict[str, object]:
    repo_root = root / "external" / "decima-sim"
    tpch_root = repo_root / "spark_env" / "tpch"
    templates = load_tpch_templates(tpch_root) if tpch_root.exists() else ()
    graphene_reference = _graphene_reference_status(repo_root)
    official = _official_reference_status(repo_root, templates)
    return {
        "scope": "pre_execution_preflight",
        "official_readme_reference": official,
        "graphene_reference": graphene_reference,
        "toolchain": _toolchain_status(),
        "execution_status": {
            "official_tf1_backend": "unavailable_in_current_python_env",
            "pytorch_port": "official_simulator_adapter_and_training_smoke_present_not_yet_reproduction_validated",
            "safe_next_step": "run official-train-scale timing probe, then launch official README reproduction gate if stable",
            "perturbation_execution_allowed": False,
        },
        "reference_commands": reference_command_payload(),
        "allowed_now": {
            "official_reproduction_gate_after_deeprm": official["passed"],
            "decima_perturbation_cells": False,
            "reason": "Decima perturbation cells require official reproduction plus Graphene validation.",
        },
        "execution_only_gates_remaining": [
            "official-train-scale timing probe for epoch time and stability",
            "train/test official README reference checkpoint at full scale",
            "validate Graphene-style comparator performance before Decima P1/P2/P3 or exclude/amend Decima before reporting",
        ],
        "pre_execution_blockers_remaining": [],
        "pre_execution_guardrails": [
            "No executable Graphene reference exists in the inspected official Decima commit; this is resolved pre-execution by labelling the local comparator Graphene-style and keeping the paper-evidence gate closed until validation or exclusion/amendment.",
        ],
    }


def _toolchain_status() -> dict[str, object]:
    torch_status: dict[str, object]
    try:
        import torch

        torch_status = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - provenance only
        torch_status = {"available": False, "error": str(exc)}
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    gpu_query = None
    if Path(nvidia_smi).exists() or shutil.which(nvidia_smi):
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            gpu_query = result.stdout.strip() or result.stderr.strip()
        except Exception as exc:  # pragma: no cover - provenance only
            gpu_query = f"unavailable: {exc}"
    return {
        "torch": torch_status,
        "tensorflow_available": importlib.util.find_spec("tensorflow") is not None,
        "gym_available": importlib.util.find_spec("gym") is not None,
        "docker_available": shutil.which("docker") is not None,
        "nvidia_smi": gpu_query,
    }


def _official_reference_status(repo_root: Path, templates: tuple[object, ...]) -> dict[str, object]:
    return {
        "passed": repo_root.exists() and len(templates) == 154,
        "repo_exists": repo_root.exists(),
        "readme_exists": (repo_root / "README.md").exists(),
        "tpch_template_count": len(templates),
        "tpch_summary": template_summary(templates) if templates else {},
    }


def _graphene_reference_status(repo_root: Path) -> dict[str, object]:
    multi_test = repo_root / "multi_resource_test.py"
    agent_dir = repo_root / "multi_resource_agents"
    env_dir = repo_root / "multi_resource_env"
    imports_graphene = False
    if multi_test.exists():
        imports_graphene = "MultiResGrapheneAgent" in multi_test.read_text(errors="replace")
    return {
        "passed": imports_graphene and agent_dir.exists() and env_dir.exists(),
        "multi_resource_test_exists": multi_test.exists(),
        "multi_resource_test_references_graphene": imports_graphene,
        "multi_resource_agents_dir_exists": agent_dir.exists(),
        "multi_resource_env_dir_exists": env_dir.exists(),
        "local_comparator_status": "graphene_style_scaffold_only",
        "protocol_guardrail": "do_not_call_local_scaffold_faithful_graphene_before_validation",
    }
