"""Built-in v2 quality evaluators.

Security/privacy rules live in :mod:`acr.audit.audit_loop`; these evaluators emit only
quality results.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ..kernel import SignalEvidenceRef, TargetRef
from ..modules import ModuleAsset, ModuleRegistry
from .evaluation_pipeline import (
    EvaluationInvocation,
    EvaluationResult,
    make_result,
)


def _event_text(event: Mapping[str, Any]) -> str:
    return json.dumps(
        event, ensure_ascii=False, sort_keys=True, default=str
    ).lower()


def gate_effectiveness_evaluator(
    invocation: EvaluationInvocation,
    capabilities: Mapping[str, Any],
) -> EvaluationResult:
    """Assess gate/process effects without claiming a security incident."""
    del capabilities
    trajectory = invocation.context.trajectory
    events = trajectory.events
    issues: list[tuple[str, str, int | str]] = []
    rejections: Counter[str] = Counter()
    if str(trajectory.output.get("status") or "") == "NO_ANSWER":
        issues.append((
            "NO_ANSWER_AT_TERMINATION",
            "the run ended without producing a chart-review answer",
            "termination",
        ))
    for index, event in enumerate(events):
        kind = str(event.get("kind") or "").lower()
        if kind == "runtime_error":
            issues.append((
                "RUNTIME_OR_PROVIDER_ERROR",
                "the run terminated after a runtime or provider exception",
                event.get("seq", index),
            ))
            continue
        if kind != "answer_rejected":
            continue
        attempted = event.get("attempted") or event.get("answer") or {}
        signature = json.dumps(
            {
                "attempted": attempted,
                "why": event.get("why") or event.get("reason"),
            },
            sort_keys=True,
            default=str,
        )
        rejections[signature] += 1
        text = _event_text(event)
        if (
            ("may_mention" in text or "unread" in text)
            and ("evidence_insufficient" in text or "coverage" in text)
            and str(trajectory.output.get("status") or "")
            == "SPEC_INSUFFICIENT"
        ):
            issues.append((
                "COVERAGE_LOOP_MISROUTED_TO_SPEC",
                "coverage remained incomplete but output used SPEC_INSUFFICIENT",
                event.get("seq", index),
            ))
    if any(count >= 2 for count in rejections.values()):
        issues.append((
            "REJECTION_LOOP",
            "the same answer was rejected repeatedly for the same reason",
            "answer_rejected",
        ))
    open_threads = trajectory.coverage_state.get("open_threads") or {}
    unresolved = 0
    if isinstance(open_threads, Mapping):
        unresolved = int(open_threads.get("n_unresolved") or 0)
    if (
        unresolved
        and str(trajectory.output.get("status") or "") == "FOUND"
    ):
        issues.append((
            "OPEN_COVERAGE_AFTER_ANSWER",
            "FOUND was emitted with unresolved coverage threads",
            "coverage_state.open_threads",
        ))
    status = "FAIL" if issues else "PASS"
    evidence = tuple(
        SignalEvidenceRef(
            "TRAJECTORY_EVENT",
            str(locator),
            locator=str(locator),
        )
        for _, _, locator in issues
    )
    return make_result(
        invocation,
        status=status,
        target_ref=TargetRef(
            "GATE_DECISION", trajectory.trajectory_id
        ),
        score=0.0 if issues else 1.0,
        reason=(
            "; ".join(message for _, message, _ in issues)
            if issues
            else "no gate-effectiveness anomaly detected"
        ),
        evidence_refs=evidence,
        payload={
            "issue_kinds": [kind for kind, _, _ in issues],
            "n_issues": len(issues),
        },
    )


def evidence_validity_evaluator(
    invocation: EvaluationInvocation,
    capabilities: Mapping[str, Any],
) -> EvaluationResult:
    """Check structured proof state, leaving semantic support to richer evaluators."""
    del capabilities
    trajectory = invocation.context.trajectory
    output = trajectory.output
    found = str(output.get("status") or "") == "FOUND"
    evidence = trajectory.evidence_state
    cited = evidence.get("evidence") if isinstance(evidence, Mapping) else None
    if cited is None and isinstance(output.get("evidence"), (list, tuple)):
        cited = output.get("evidence")
    proof_valid = bool(
        evidence.get("proof_valid")
        if isinstance(evidence, Mapping)
        else False
    )
    issues = []
    if found and not cited:
        issues.append("FOUND_WITHOUT_CITATION")
    if found and not proof_valid:
        issues.append("FOUND_WITHOUT_VALID_PROOF")
    return make_result(
        invocation,
        status="FAIL" if issues else "PASS",
        target_ref=TargetRef("EVIDENCE", trajectory.trajectory_id),
        score=0.0 if issues else 1.0,
        reason=(
            ", ".join(issues)
            if issues
            else "structured evidence and proof state are consistent"
        ),
        payload={"issue_kinds": issues},
    )


def builtin_evaluation_module_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    for asset, implementation in (
        (
            ModuleAsset(
                module_id="gate-effectiveness",
                version="1.0.0",
                module_kind="EVALUATOR",
                runner_type="CODE",
                input_channels=("trajectory",),
                output_schema="acr.gate_effectiveness/1",
                implementation_id="evaluation.gate_effectiveness.v1",
                supported_truth_modes=(
                    "BLIND",
                    "REGISTRY_REFERENCE",
                    "GOLD",
                ),
                maximum_authority="BLOCK_RELEASE",
                description=(
                    "Detect rejection loops, coverage misrouting, and excessive "
                    "gate/process failures without creating audit incidents."
                ),
                owner="evaluation-engineer",
            ),
            gate_effectiveness_evaluator,
        ),
        (
            ModuleAsset(
                module_id="evidence-validity",
                version="1.0.0",
                module_kind="EVALUATOR",
                runner_type="CODE",
                input_channels=("trajectory",),
                output_schema="acr.evidence_validity/1",
                implementation_id="evaluation.evidence_validity.v1",
                supported_truth_modes=(
                    "BLIND",
                    "REGISTRY_REFERENCE",
                    "GOLD",
                ),
                maximum_authority="BLOCK_RELEASE",
                description=(
                    "Check structured citation and proof presence for final output."
                ),
                owner="evaluation-engineer",
            ),
            evidence_validity_evaluator,
        ),
    ):
        registry.register_asset(asset)
        registry.register_implementation(
            asset.implementation_id, implementation
        )
    return registry
