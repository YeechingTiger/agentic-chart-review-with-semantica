"""Normalized deterministic runtime-control decisions.

The deepagents hooks remain the production enforcement points.  These adapters
give their decisions a common module contract without moving clinical rules out
of the existing answer gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .modules import ModuleAsset, ModuleContractError, ModuleRegistry

CONTROL_DECISIONS = frozenset({"ALLOW", "DENY", "REQUIRE"})


@dataclass(frozen=True)
class ProposedAction:
    action_type: str
    patient_id: str = ""
    tool_name: str = ""
    provider: str = ""
    contains_phi: bool = False
    projected_cost_usd: float = 0.0
    answer: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunControlState:
    patient_scope: str
    allowed_tools: tuple[str, ...] = ()
    allowed_providers: tuple[str, ...] = ()
    spent_usd: float = 0.0
    max_usd: float = 0.0
    proof_valid: bool = False
    open_conflicts: int = 0
    open_threads: int = 0


@dataclass(frozen=True)
class ControlDecision:
    decision: str
    control_ref: str
    reason: str
    requirement: str = ""

    def __post_init__(self) -> None:
        if self.decision not in CONTROL_DECISIONS:
            raise ModuleContractError(
                f"unknown runtime-control decision {self.decision!r}"
            )


class PatientScopeControl:
    module_ref = "patient-scope-control@1.0.0"

    def check(
        self, proposed_action: ProposedAction, run_state: RunControlState
    ) -> ControlDecision:
        if (
            proposed_action.patient_id
            and proposed_action.patient_id != run_state.patient_scope
        ):
            return ControlDecision(
                "DENY", self.module_ref, "cross-patient action denied"
            )
        return ControlDecision("ALLOW", self.module_ref, "patient scope matches")


class ProviderBoundaryControl:
    module_ref = "provider-boundary-control@1.0.0"

    def check(
        self, proposed_action: ProposedAction, run_state: RunControlState
    ) -> ControlDecision:
        if (
            proposed_action.contains_phi
            and proposed_action.provider
            and proposed_action.provider not in set(run_state.allowed_providers)
        ):
            return ControlDecision(
                "DENY",
                self.module_ref,
                "PHI-bearing action targets an unapproved provider",
            )
        return ControlDecision("ALLOW", self.module_ref, "provider boundary satisfied")


class ToolAllowlistControl:
    module_ref = "tool-allowlist-control@1.0.0"

    def check(
        self, proposed_action: ProposedAction, run_state: RunControlState
    ) -> ControlDecision:
        if (
            proposed_action.tool_name
            and proposed_action.tool_name not in set(run_state.allowed_tools)
        ):
            return ControlDecision(
                "DENY", self.module_ref, "tool absent from declared allowlist"
            )
        return ControlDecision("ALLOW", self.module_ref, "tool is declared")


class SpendLimitControl:
    module_ref = "spend-limit-control@1.0.0"

    def check(
        self, proposed_action: ProposedAction, run_state: RunControlState
    ) -> ControlDecision:
        projected = run_state.spent_usd + proposed_action.projected_cost_usd
        if run_state.max_usd >= 0 and projected > run_state.max_usd:
            return ControlDecision(
                "DENY", self.module_ref, "hard spend ceiling would be exceeded"
            )
        return ControlDecision("ALLOW", self.module_ref, "spend ceiling satisfied")


class EvidenceAnswerControl:
    """Adapter for the structural obligations owned by ``answer_gate``."""

    module_ref = "evidence-answer-control@1.0.0"

    def check(
        self, proposed_action: ProposedAction, run_state: RunControlState
    ) -> ControlDecision:
        status = str(proposed_action.answer.get("status") or "")
        if status != "FOUND":
            return ControlDecision(
                "ALLOW", self.module_ref, "non-FOUND answer follows abstention contract"
            )
        if not run_state.proof_valid:
            return ControlDecision(
                "REQUIRE",
                self.module_ref,
                "FOUND requires a valid evidence proof",
                "valid_evidence_proof",
            )
        if run_state.open_conflicts or run_state.open_threads:
            return ControlDecision(
                "REQUIRE",
                self.module_ref,
                "FOUND requires closure of discovered conflicts and threads",
                "close_open_obligations",
            )
        return ControlDecision("ALLOW", self.module_ref, "answer obligations satisfied")


class RuntimeControlRegistry:
    def __init__(self) -> None:
        self.modules = ModuleRegistry()

    def register(self, asset: ModuleAsset, implementation: Any) -> None:
        if asset.module_kind != "RUNTIME_CONTROL":
            raise ModuleContractError(
                f"{asset.ref}: expected a RUNTIME_CONTROL module"
            )
        self.modules.register_asset(asset)
        self.modules.register_implementation(
            asset.implementation_id,
            lambda implementation=implementation: implementation,
        )

    def resolve(self, ref: str):
        asset = self.modules.resolve(ref)
        return asset, self.modules.implementation(asset)()


def builtin_runtime_control_registry() -> RuntimeControlRegistry:
    registry = RuntimeControlRegistry()
    controls = (
        ("patient-scope-control", "control.patient_scope.v1", PatientScopeControl()),
        (
            "provider-boundary-control",
            "control.provider_boundary.v1",
            ProviderBoundaryControl(),
        ),
        ("tool-allowlist-control", "control.tool_allowlist.v1", ToolAllowlistControl()),
        ("spend-limit-control", "control.spend_limit.v1", SpendLimitControl()),
        (
            "evidence-answer-control",
            "control.evidence_answer.v1",
            EvidenceAnswerControl(),
        ),
    )
    for module_id, implementation_id, implementation in controls:
        registry.register(
            ModuleAsset(
                module_id=module_id,
                version="1.0.0",
                module_kind="RUNTIME_CONTROL",
                runner_type="CODE",
                input_channels=("proposed_action", "run_state"),
                output_schema="acr.control_decision/1",
                implementation_id=implementation_id,
                supported_truth_modes=("BLIND",),
                maximum_authority="BLOCK_CURRENT_ANSWER",
                description="Deterministic in-request control adapter.",
                owner="runtime-owner",
            ),
            implementation,
        )
    return registry
