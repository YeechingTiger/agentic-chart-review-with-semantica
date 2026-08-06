from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.core import site
from acr.core.local_artifacts import LocalArtifactStore
from acr.diagnosis import attribution as A

ROOT = Path(__file__).resolve().parents[1]


def packet(mode=A.BLIND, **overrides):
    values = {
        "case_id": "CASE001", "spec_id": "SPEC.x", "spec_hash": "abc",
        "mode": mode,
        "manifest_ref": A.ArtifactRef("/outside/run.manifest.json", "a" * 64),
        "trace_ref": A.ArtifactRef("/outside/run.jsonl", "b" * 64),
        "manifest": {"spec_id": "SPEC.x", "spec_hash": "abc",
                     "answer": {"status": "SPEC_INSUFFICIENT", "value": {}}},
        "trace": ({"seq": 1, "kind": "answer_rejected", "why": "same check"},),
        "rule_catalogue": ({"rule_id": "decision_rule.1", "text": "Use pathology."},),
        "detector_findings": (),
        "behavior_signature": {"answer": {"status": "SPEC_INSUFFICIENT"}},
    }
    values.update(overrides)
    return A.AttributionPacket(**values)


def test_local_store_private_atomic_json_and_idempotent_jsonl(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = store.write_json("x/value.json", {"a": 1})
    assert path == tmp_path / "x/value.json"
    assert os.stat(tmp_path).st_mode & 0o777 == 0o700
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert store.append_jsonl("x/events.jsonl", {"v": 1}, idempotency_key="same")
    assert not store.append_jsonl("x/events.jsonl", {"v": 2}, idempotency_key="same")
    assert len((tmp_path / "x/events.jsonl").read_text().splitlines()) == 1


def test_blind_packet_recursively_rejects_key_material():
    with pytest.raises(A.AttributionError, match="key-bearing"):
        packet(manifest={"nested": {"expected_output": {"histology": "8140"}}})


def test_unresolved_registry_reference_cannot_become_confirmed_clinical_cause():
    p = packet(
        mode=A.REGISTRY_REFERENCE,
        registry_reference={
            "case_id": "CASE001", "adjudication": "UNRESOLVED",
            "registry_value": {"histology": "8140"},
        },
    )
    assert not p.semantic_patch_allowed
    with pytest.raises(A.AttributionError, match="cannot be CONFIRMED"):
        A.CauseFinding(
            cause="EVIDENCE_INTERPRETATION", status="CONFIRMED",
            evidence_class="JUDGED", rationale="model thinks so",
            evidence=(A.EvidenceRef("trace", "seq:1"),),
        )


def test_evidence_reference_accepts_prompt_friendly_aliases_but_normalizes_them():
    assert A.EvidenceRef.from_dict(
        {"kind": "trace_event", "ref": "seq:1"}).kind == "trace"
    assert A.EvidenceRef.from_dict(
        {"kind": "rule_id", "ref": "decision_rule.1"}).kind == "spec_rule"
    assert A.EvidenceRef.from_dict(
        {"source_type": "trace_event", "source_id": "seq:1"}).to_dict() == {
            "kind": "trace", "ref": "seq:1", "detail": ""}


def test_primary_plus_contributing_causes_cluster_by_structure(tmp_path):
    primary = A.CauseFinding(
        cause="ANSWER_CHECK_OR_GATE", status="CONFIRMED",
        evidence_class="DETERMINISTIC", rationale="same refusal repeated",
        evidence=(A.EvidenceRef("detector", "rejection_loop"),),
        parameter_id="answer_check_rejection_messages",
    )
    contributing = A.CauseFinding(
        cause="SPEC_FORM", status="POSSIBLE", evidence_class="JUDGED",
        rationale="the wording admits two readings",
        evidence=(A.EvidenceRef("spec_rule", "decision_rule.1"),),
        parameter_id="precedence_conflict_rule",
    )
    report = A.AttributionReport(
        case_id="CASE001", spec_id="SPEC.x", mode=A.REGISTRY_REFERENCE,
        primary_cause=primary, contributing_causes=(contributing,),
        alternatives_considered=("gate loop", "clinical disagreement"),
        probes=(A.AttributionProbe(
            "p1", "challenge", ("loop", "not loop"), "repeat evidence",
            confirmation=True),),
        termination_reason="closed", confirmation_performed=True,
        confirmation_new_conflict=False,
    )
    library = A.ErrorCaseLibrary(LocalArtifactStore(tmp_path), "pilot")
    library.add_case(A.ErrorCaseEvent(
        case_id="CASE001", event="SELECTED", lifecycle="OPEN",
        run_ref={"sha256": "m1"}, reasons=("registry_disagreement:histology",)))
    assert library.add_attribution(report, manifest_sha256="m1")
    assert not library.add_attribution(report, manifest_sha256="m1")
    assert library.rows("attributions.jsonl")[0]["manifest_sha256"] == "m1"
    clusters = A.cluster_reports([report])
    assert clusters[0].primary_cause == "ANSWER_CHECK_OR_GATE"
    assert clusters[0].contributing_tags == ("SPEC_FORM",)
    summary = A.summarize_library(library)
    assert summary["n_attributions"] == 1
    assert summary["signal_clusters"] == [{
        "signal": "registry_disagreement:histology",
        "case_ids": ["CASE001"], "n_cases": 1,
    }]


def test_unadjudicated_mode_cannot_claim_human_adjudicated_evidence():
    primary = A.CauseFinding(
        cause="UNRESOLVED", status="UNRESOLVED",
        evidence_class="HUMAN_ADJUDICATED", rationale="not actually adjudicated",
        evidence=())
    with pytest.raises(A.AttributionError, match="no human-adjudicated truth"):
        A.AttributionReport(
            case_id="CASE001", spec_id="SPEC.x", mode=A.REGISTRY_REFERENCE,
            primary_cause=primary, contributing_causes=(),
            alternatives_considered=("a", "b"), probes=(),
            termination_reason="unresolved", confirmation_performed=False,
            confirmation_new_conflict=False)


def test_chart_reads_require_trace_then_probe_and_have_no_patient_argument():
    p = packet()
    chart = Corpus(ROOT / str(site.corpus_root())).chart("SYN0001")
    ctx = A.AttributionRuntimeContext(p, chart, max_chart_reads=2, max_usd=1.0)
    tools = {tool.name: tool for tool in A.attribution_tools(ctx)}
    assert "patient" not in tools["read_document"].args

    blocked = json.loads(tools["read_document"].invoke({"note_id": next(iter(chart._docs))}))
    assert blocked["error"] == "TRACE_FIRST"
    json.loads(tools["inspect_trace"].invoke({}))
    blocked = json.loads(tools["read_document"].invoke({"note_id": next(iter(chart._docs))}))
    assert blocked["error"] == "PROBE_REQUIRED"
    opened = json.loads(tools["open_attribution_probe"].invoke({
        "question": "was the witness missed?",
        "alternatives": ["retrieval miss", "interpretation error"],
        "expected_discriminator": "whether the establishing text was surfaced",
        "confirmation": False,
    }))
    assert opened["opened"]["probe_id"] == "probe-1"
    note_id = next(iter(chart._docs))
    result = json.loads(tools["read_document"].invoke({"note_id": note_id}))
    assert result["patient_scope"] == "CASE001"
    assert ctx.chart_reads == 1
    draft = json.loads(tools["record_cause"].invoke({
        "cause": "EVIDENCE_INTERPRETATION", "status": "POSSIBLE",
        "evidence_class": "JUDGED", "rationale": "provisional",
        "evidence": [{"kind": "note", "ref": note_id}],
        "route_owner": "registrar",
    }))
    assert draft["accepted"] is True

    # A provider can omit required tool arguments despite the advertised JSON schema. The
    # runtime must return a typed gate rejection, not crash the whole attribution.
    rejected = json.loads(tools["submit_attribution"].func())
    assert rejected["accepted"] is False
    assert ctx.submission_rejections


def test_packet_and_probe_citations_are_resolved_not_merely_nonempty():
    p = packet(
        mode=A.REGISTRY_REFERENCE,
        registry_reference={
            "case_id": "CASE001", "adjudication": "UNRESOLVED",
            "registry_value": {"histology": "8140"},
        })
    chart = Corpus(ROOT / str(site.corpus_root())).chart("SYN0001")
    ctx = A.AttributionRuntimeContext(p, chart, max_chart_reads=0, max_usd=1.0)
    ctx.probes.append(A.AttributionProbe(
        "probe-1", "q", ("a", "b"), "d", confirmation=True))
    valid = A.CauseFinding(
        cause="REFERENCE_OR_GOLD", status="LIKELY", evidence_class="JUDGED",
        rationale="reference differs",
        evidence=(
            A.EvidenceRef("packet", "registry_reference.registry_value.histology"),
            A.EvidenceRef("probe", "probe-1"),
        ))
    assert A._citation_errors(valid, ctx) == []
    invalid = A.CauseFinding(
        cause="REFERENCE_OR_GOLD", status="LIKELY", evidence_class="JUDGED",
        rationale="invented",
        evidence=(A.EvidenceRef("packet", "registry_reference.not_a_field"),))
    assert "not an exact" in A._citation_errors(invalid, ctx)[0]


def test_selection_is_a_signal_not_a_truth_verdict():
    p = packet(
        mode=A.REGISTRY_REFERENCE,
        registry_reference={
            "case_id": "CASE001", "adjudication": "UNRESOLVED",
            "registry_value": {"histology": "8140"},
        },
        manifest={
            "spec_id": "SPEC.x", "spec_hash": "abc", "gate_validated": True,
            "answer": {"status": "FOUND", "value": {"histology": "8230"}},
        },
    )
    assert A.selection_reasons(p) == ("registry_disagreement:histology",)


def test_deepagents_runner_accepts_a_structured_runtime_attribution():
    pytest.importorskip("deepagents")
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    target_id = A.derive_target_events(packet())[0].event_id

    class Script(BaseChatModel):
        turn: int = 0

        @property
        def _llm_type(self):
            return "attribution-script"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages: list[BaseMessage], stop=None,
                      run_manager: CallbackManagerForLLMRun | None = None,
                      **kwargs) -> ChatResult:
            i, self.turn = self.turn, self.turn + 1
            calls = [
                ("list_target_events", {}),
                ("select_target_event", {"target_event_id": target_id}),
                ("inspect_trace", {}),
                ("open_attribution_probe", {
                    "question": "is the recorded rejection real?",
                    "alternatives": ["runtime failure", "false detector"],
                    "expected_discriminator": "the rejection event exists",
                    "confirmation": True,
                }),
                ("record_counterfactual_test", {
                    "kind": "TRACE_REPLAY",
                    "intervention": "remove the recorded runtime failure",
                    "prediction": "the target process anomaly would not occur",
                    "outcome": "SUPPORTED",
                    "observation": "the trace event is the target runtime failure",
                    "evidence": [{"kind": "trace", "ref": "seq:1"}],
                }),
                ("submit_skeptic_review", {
                    "verdict": "PASS",
                    "rationale": "the deterministic trace event is aligned to the target",
                    "objections": [],
                    "untested_alternatives": [],
                    "evidence": [{"kind": "trace", "ref": "seq:1"}],
                }),
                ("submit_attribution", {
                    "primary_cause": {
                        "cause": "RUNTIME_OR_PROVIDER", "status": "CONFIRMED",
                        "evidence_class": "DETERMINISTIC",
                        "rationale": "the trace contains the recorded runtime failure",
                        "evidence": [{"kind": "trace", "ref": "seq:1"}],
                        "parameter_id": "agent_system_prompt",
                        "relation_to_target": "EXPLAINS",
                        "causal_strength": "COUNTERFACTUAL_SUPPORTED",
                        "mechanism": "the runtime failure directly produced the target event",
                        "counterfactual_prediction": (
                            "without the failure the target event would not occur"),
                    },
                    "contributing_causes": [],
                    "alternatives_considered": ["runtime failure", "false detector"],
                    "termination_reason": "obligations closed",
                    "confirmation_performed": True,
                    "confirmation_new_conflict": False,
                }),
            ]
            if i < len(calls):
                name, args = calls[i]
                message = AIMessage(
                    content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])
            else:
                message = AIMessage(content="done")
            return ChatResult(generations=[ChatGeneration(message=message)])

    class SkepticScript(BaseChatModel):
        @property
        def _llm_type(self):
            return "independent-skeptic-script"

        def _generate(self, messages: list[BaseMessage], stop=None,
                      run_manager: CallbackManagerForLLMRun | None = None,
                      **kwargs) -> ChatResult:
            message = AIMessage(content=json.dumps({
                    "verdict": "PASS",
                    "rationale": (
                        "the proposed runtime cause directly matches the selected event"),
                    "objections": [],
                    "untested_alternatives": [],
                    "evidence": [{"kind": "trace", "ref": "seq:1"}],
                }))
            return ChatResult(generations=[ChatGeneration(message=message)])

    report = A.run_attribution_agent(
        packet=packet(), chart=Corpus(ROOT / str(site.corpus_root())).chart("SYN0001"),
        model=Script(), max_model_calls=10, max_usd=1.0, max_chart_reads=0,
        skeptic_model=SkepticScript(),
    )
    assert report.primary_cause.cause == "RUNTIME_OR_PROVIDER"
    assert report.primary_cause.status == "CONFIRMED"
    assert report.confirmation_performed
    assert report.target_event.event_id == target_id
    assert report.counterfactual_tests[0].outcome == "SUPPORTED"
    assert report.skeptic_review.verdict == "PASS"
    assert report.skeptic_review.reviewer == "INDEPENDENT_MODEL"


def test_independent_skeptic_conflict_downgrades_primary_to_unresolved():
    target = A.TargetEvent(
        event_id="TE-1", kind="PROCESS_ANOMALY", field="",
        observed="gate loop", reference_signal=None, source="TRACE",
        truth_status="DETERMINISTIC", question="what caused the loop?",
    )
    cause = A.CauseFinding(
        cause="ANSWER_CHECK_OR_GATE", status="LIKELY",
        evidence_class="DETERMINISTIC", rationale="the same rejection recurred",
        evidence=(A.EvidenceRef("trace", "seq:1"),),
        relation_to_target="EXPLAINS",
        causal_strength="COUNTERFACTUAL_SUPPORTED",
        mechanism="the gate returned the same rejection without changing state",
        counterfactual_prediction="removing the repeated rejection closes the loop",
    )
    report = A.AttributionReport(
        case_id="CASE001", spec_id="SPEC.x", mode=A.BLIND,
        primary_cause=cause, contributing_causes=(),
        alternatives_considered=("gate defect", "provider retry"),
        probes=(A.AttributionProbe(
            "probe-1", "is the loop real?", ("gate defect", "provider retry"),
            "replay the gate", confirmation=True,
        ),),
        termination_reason="candidate complete",
        confirmation_performed=True,
        confirmation_new_conflict=False,
        target_event=target,
        counterfactual_tests=(A.CounterfactualTest(
            "cf-1", "TE-1", "GATE_REPLAY", "remove repeated rejection",
            "loop closes", "SUPPORTED", "deterministic replay closed the loop",
            (A.EvidenceRef("trace", "seq:1"),),
        ),),
        skeptic_review=A.SkepticReview(
            "PASS", "self-challenge found no objection", (), (),
            (A.EvidenceRef("trace", "seq:1"),),
        ),
    )
    downgraded = A._apply_independent_skeptic(
        report,
        A.SkepticReview(
            "REVISE",
            "the replay does not isolate the provider-retry alternative",
            ("counterfactual is not discriminative",),
            ("provider retry",),
            reviewer="INDEPENDENT_MODEL",
        ),
    )
    assert downgraded.primary_cause.cause == "UNRESOLVED"
    assert downgraded.contributing_causes[0].cause == "ANSWER_CHECK_OR_GATE"
    assert downgraded.skeptic_review.reviewer == "INDEPENDENT_MODEL"


def test_attribution_meta_evaluation_is_owned_by_attribution_module():
    predictions = [
        {
            "case_id": "CASE-1",
            "primary_cause": {
                "cause": "RUNTIME_OR_PROVIDER",
                "status": "CONFIRMED",
            },
            "citation_valid": True,
            "scope_violations": 0,
        },
        {
            "case_id": "CASE-2",
            "primary_cause": {
                "cause": "ANSWER_CHECK_OR_GATE",
                "status": "CONFIRMED",
            },
            "citation_valid": True,
            "scope_violations": 0,
        },
    ]
    adjudications = [
        {"case_id": "CASE-1", "primary_cause": "RUNTIME_OR_PROVIDER"},
        {"case_id": "CASE-2", "primary_cause": "ANSWER_CHECK_OR_GATE"},
    ]
    report = A.meta_evaluate_attributions(
        predictions,
        adjudications,
        min_cases=2,
        min_macro_f1=1.0,
    )
    assert report["status"] == "CERTIFIED_SCREEN"
    assert report["macro_f1"] == 1.0


def test_eval_skills_are_method_and_are_absent_unless_asked_for():
    """The diagnostic method reaches the attribution prompt, and only when supplied.

    `acr attribute case` passes nothing, and its prompt must be the one it rendered before eval
    skills existed — byte for byte, including the blank lines. A default that quietly grew a
    newline is a default that quietly grew, and the next reader cannot tell which parts of the
    prompt were measured and which drifted in.
    """
    import inspect

    assert "\nACTIVE MODULES\n\n\nCERTAINTY\n" in A._attribution_system_prompt(packet())
    assert inspect.signature(A.run_attribution_agent).parameters[
        "eval_skills_prompt"].default == ""

    with_skills = A._attribution_system_prompt(
        packet(), eval_skills_prompt="  DIAGNOSTIC METHOD. YOU DO NOT SCORE.  ")
    assert "DIAGNOSTIC METHOD. YOU DO NOT SCORE." in with_skills
    # Beside the stage instructions, ahead of the certainty fence: method first, then the rule
    # about what may be called CONFIRMED, which no skill is allowed to soften.
    assert (with_skills.index("ACTIVE MODULES")
            < with_skills.index("DIAGNOSTIC METHOD")
            < with_skills.index("CERTAINTY"))
