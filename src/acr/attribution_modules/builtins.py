"""Built-in causal attribution modules and their deterministic submission gates."""
from __future__ import annotations

from typing import Any

from .registry import (
    AttributionStage,
    AttributionStageProfile,
    AttributionStageRegistry,
)


def _target_gate(report: Any, ctx: Any) -> list[str]:
    errors = []
    if getattr(report, "target_event", None) is None:
        errors.append("select one target event before attribution")
    primary = report.primary_cause
    if primary.cause != "UNRESOLVED" and primary.relation_to_target != "EXPLAINS":
        errors.append(
            "resolved primary cause must relation_to_target=EXPLAINS; "
            "unrelated defects are contributing findings")
    return errors


def _counterfactual_gate(report: Any, ctx: Any) -> list[str]:
    if report.primary_cause.cause == "UNRESOLVED":
        return []
    target = report.target_event
    tests = [
        test for test in report.counterfactual_tests
        if target is not None and test.target_event_id == target.event_id
    ]
    if not tests:
        return ["resolved attribution needs a counterfactual test for the target event"]
    if (report.primary_cause.status in ("LIKELY", "CONFIRMED")
            and not any(test.outcome == "SUPPORTED" for test in tests)):
        return [
            (
                "LIKELY/CONFIRMED attribution requires a SUPPORTED target "
                "counterfactual; otherwise downgrade to POSSIBLE or UNRESOLVED"
            )
        ]
    return []


def _skeptic_gate(report: Any, ctx: Any) -> list[str]:
    review = report.skeptic_review
    if review is None:
        return ["causal attribution requires a structured skeptic review"]
    if review.verdict in ("REVISE", "UNRESOLVED") and report.primary_cause.cause != "UNRESOLVED":
        return ["skeptic did not accept the attribution; submit UNRESOLVED"]
    return []


def builtin_attribution_registry() -> AttributionStageRegistry:
    registry = AttributionStageRegistry()
    for module in (
        AttributionStage(
            "trace-reconstruction", "analysis",
            "Reconstruct searches, reads, evidence, decisions, gate events, and termination.",
            tool_names=("inspect_trace", "inspect_spec"),
            instructions=(
                "Reconstruct the recorded run before opening the chart. Separate what the "
                "trace proves from what is inferred."
            ),
        ),
        AttributionStage(
            "target-framing", "analysis",
            "Select the exact mismatch or process anomaly the report explains.",
            requires=("trace-reconstruction",),
            tool_names=("list_target_events", "select_target_event"),
            instructions=(
                "Select one target event. Every proposed cause must state whether it explains, "
                "contributes to, is unrelated to, or is unknown for that exact event."
            ),
            validate=_target_gate,
        ),
        AttributionStage(
            "targeted-probe", "tool",
            "Open a discriminating question before any same-patient chart expansion.",
            requires=("target-framing",),
            tool_names=(
                "open_attribution_probe", "list_documents",
                "search_documents", "read_document",
            ),
            instructions=(
                "Read chart material only to discriminate named rival causes, never to redo "
                "the extraction from scratch."
            ),
        ),
        AttributionStage(
            "cause-hypothesis", "analysis",
            "Record primary, contributing, unrelated, and ruled-out causal hypotheses.",
            requires=("target-framing",),
            tool_names=("record_cause", "rule_out_cause"),
            instructions=(
                "A real defect that would not change the selected target is UNRELATED_DEFECT, "
                "not the primary cause."
            ),
        ),
        AttributionStage(
            "counterfactual-replay", "eval",
            "Test the proposed mechanism with a bounded replay or explicit unrun obligation.",
            requires=("cause-hypothesis", "targeted-probe"),
            tool_names=("inspect_causal_stage", "record_counterfactual_test"),
            instructions=(
                "State what should change if the cause were removed, run the smallest safe "
                "test available, and record SUPPORTED, REFUTED, INCONCLUSIVE, or NOT_RUN."
            ),
            validate=_counterfactual_gate,
        ),
        AttributionStage(
            "skeptic-review", "skill",
            "Self-challenge the draft before a separate tool-free skeptic model reviews it.",
            requires=("counterfactual-replay",),
            tool_names=("submit_skeptic_review",),
            instructions=(
                "Record the investigator's strongest objection and untested alternative. "
                "After submission, a separate model call independently reviews target "
                "alignment and may force the final report to UNRESOLVED."
            ),
            validate=_skeptic_gate,
        ),
        AttributionStage(
            "citation-authority-gate", "gate",
            "Validate citations, truth boundary, certainty, and semantic-patch authority.",
        ),
        AttributionStage(
            "structured-submission", "output",
            "Submit the validated attribution report.",
            tool_names=("submit_attribution",),
        ),
    ):
        registry.register(module)
    registry.register_profile(AttributionStageProfile(
        "causal-attribution-v1",
        (
            "trace-reconstruction", "target-framing", "targeted-probe",
            "cause-hypothesis", "counterfactual-replay", "skeptic-review",
            "citation-authority-gate", "structured-submission",
        ),
        "Target-framed causal attribution with counterfactual and skeptic gates.",
    ))
    return registry
