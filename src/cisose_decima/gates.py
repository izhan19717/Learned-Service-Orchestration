"""Explicit Decima reproduction and comparator validation gates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    details: dict[str, object]


@dataclass(frozen=True)
class DecimaReadiness:
    official_readme_reproduction: GateResult
    graphene_validation: GateResult

    @property
    def ready_for_perturbations(self) -> bool:
        return self.official_readme_reproduction.passed and self.graphene_validation.passed


def official_readme_gate_pending() -> GateResult:
    return GateResult(
        name="official_readme_reproduction",
        passed=False,
        details={
            "required_comparison": "learn_vs_dynamic_partition",
            "exec_cap": 50,
            "train_num_stream_dags": 200,
            "test_num_stream_dags": 5000,
            "reference_model_epoch": 10000,
            "status": "not_run",
        },
    )


def graphene_gate_pending() -> GateResult:
    return GateResult(
        name="graphene_validation",
        passed=False,
        details={
            "required_comparator": "Graphene",
            "source_status": "official_decima_repo_exposes_multi_resource_test_imports_but_not_multi_resource_agent_env_sources",
            "current_scaffold": "GrapheneStyleComparator",
            "reason": "Graphene is not part of the official single-resource README command. The inspected official Decima commit does not include executable Graphene agent sources, so any local Graphene-style scaffold must be validated or the Decima comparator must be protocol-amended before Decima P1/P2/P3.",
            "paper_evidence_allowed": False,
            "status": "not_run",
        },
    )


def current_readiness() -> DecimaReadiness:
    return DecimaReadiness(
        official_readme_reproduction=official_readme_gate_pending(),
        graphene_validation=graphene_gate_pending(),
    )
