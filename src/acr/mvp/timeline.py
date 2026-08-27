"""Validate a Langtrace export and replay it into deterministic ReAct cycles.

Terra is allowed to interpret this skeleton; it is never allowed to create, delete, reorder or
move an action/observation.  Observed state and declared state are deliberately disjoint so a
runtime claim about Standing cannot turn into a server fact merely by sharing a tool result.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any

from acr.mvp.decision_receipts import receipt_from_action
from acr.mvp.langtrace_io import LangtraceReviewTrace
from acr.mvp.task_presentation import content_hash


class IncompleteTraceError(RuntimeError):
    """The trace cannot carry reconstruction because its export is not provably complete."""


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _span_id(span: dict[str, Any]) -> str:
    return str(span.get("spanId") or span.get("span_id") or span.get("id") or "")


def _parent_span_id(span: dict[str, Any]) -> str:
    return str(span.get("parentSpanId") or span.get("parent_span_id")
               or span.get("parentId") or span.get("parent_id") or "")


@dataclass(frozen=True, slots=True)
class TraceCompletenessManifest:
    schema: str
    trace_id: str
    ordered_event_ids: tuple[str, ...]
    ordered_span_ids: tuple[str, ...]
    parent_links: tuple[tuple[str, str], ...]
    event_count: int
    span_count: int
    content_hash: str
    export_status: str
    issues: tuple[str, ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_trace_completeness(review: LangtraceReviewTrace) -> TraceCompletenessManifest:
    """Prove event count/hash/order and the span parent structure advertised by the root."""
    issues: list[str] = []
    roots = [span for span in review.spans
             if _attrs(span).get("acr.trace.schema") in {
                 "acr.langtrace.v1", "acr.langtrace.v2"}]
    root_attrs = _attrs(roots[0]) if len(roots) == 1 else {}
    if len(roots) != 1:
        issues.append(f"expected one ACR root span, found {len(roots)}")
    if root_attrs.get("acr.export.status") != "COMPLETE":
        issues.append("root span does not declare a COMPLETE export")

    events = sorted(review.layer1_events, key=lambda event: int(event.get("seq", 0)))
    seqs = [event.get("seq") for event in events]
    expected_seqs = list(range(1, len(events) + 1))
    if seqs != expected_seqs:
        issues.append(f"Layer-1 seqs are not contiguous: {seqs!r}")
    event_hash = content_hash(events)
    expected_count = root_attrs.get("acr.layer1.event_count")
    try:
        expected_count = int(expected_count)
    except (TypeError, ValueError):
        expected_count = -1
    if expected_count != len(events):
        issues.append(f"event count mismatch: root={expected_count}, fetched={len(events)}")
    if root_attrs.get("acr.layer1.content_hash") != event_hash:
        issues.append("Layer-1 content hash does not match the root manifest")

    event_span_seqs: list[int] = []
    for span in review.spans:
        attrs = _attrs(span)
        if "acr.layer1.event_json" in attrs:
            try:
                event_span_seqs.append(int(attrs.get("acr.layer1.seq")))
            except (TypeError, ValueError):
                issues.append("a Layer-1 span has no integer seq")
    if sorted(event_span_seqs) != expected_seqs:
        issues.append("Layer-1 event spans do not own each fetched event exactly once")

    span_ids = [_span_id(span) for span in review.spans]
    if any(not span_id for span_id in span_ids) or len(set(span_ids)) != len(span_ids):
        issues.append("span identifiers are missing or duplicated")
    known_ids = {span_id for span_id in span_ids if span_id}
    parent_links: list[tuple[str, str]] = []
    root_ids = {_span_id(span) for span in roots}
    for span in review.spans:
        span_id, parent_id = _span_id(span), _parent_span_id(span)
        if not span_id or span_id in root_ids:
            continue
        if not parent_id or parent_id not in known_ids:
            issues.append(f"span {span_id} has an unavailable parent {parent_id!r}")
        else:
            parent_links.append((span_id, parent_id))

    event_span_by_seq = {
        int(_attrs(span)["acr.layer1.seq"]): _span_id(span)
        for span in review.spans
        if _attrs(span).get("acr.layer1.seq") is not None
        and str(_attrs(span).get("acr.layer1.seq")).isdigit()
    }
    root_order = [_span_id(span) for span in roots if _span_id(span)]
    event_order = [event_span_by_seq[seq] for seq in expected_seqs
                   if seq in event_span_by_seq]
    remaining = sorted(known_ids - set(root_order) - set(event_order))
    payload: dict[str, Any] = {
        "schema": "acr.trace_completeness.v1",
        "trace_id": review.trace_id,
        "ordered_event_ids": tuple(f"layer1:{seq}" for seq in expected_seqs),
        "ordered_span_ids": tuple([*root_order, *event_order, *remaining]),
        "parent_links": tuple(sorted(parent_links)),
        "event_count": len(events),
        "span_count": len(review.spans),
        "content_hash": event_hash,
        "export_status": "COMPLETE" if not issues else "INCOMPLETE",
        "issues": tuple(issues),
    }
    payload["manifest_hash"] = content_hash(payload)
    return TraceCompletenessManifest(**payload)


def require_complete(review: LangtraceReviewTrace,
                     manifest: TraceCompletenessManifest) -> None:
    fresh = build_trace_completeness(review)
    if fresh.manifest_hash != manifest.manifest_hash:
        raise IncompleteTraceError("trace completeness manifest does not match fetched content")
    if manifest.export_status != "COMPLETE":
        raise IncompleteTraceError("; ".join(manifest.issues) or "trace export is incomplete")


def _initial_state() -> dict[str, Any]:
    return {
        "observed_state": {
            "surfaced_notes": [], "read_notes": [], "search_refs": [],
            "finding_call_refs": [], "evidence_refs": [], "citation_resolutions": [],
            "submissions": [], "gates": [], "result_status": None,
        },
        "declared_state": {
            "testimony_refs": [], "decision_receipt_refs": [],
            "open_question": None, "candidate_set": [],
            "uncertainties": [], "findings": [],
        },
    }


def _append_unique(values: list[Any], value: Any) -> None:
    if value is not None and value not in values:
        values.append(value)


def _event_receipt(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("kind") != "tool_call":
        return None
    action = {"event_ref": f"layer1:{int(event['seq'])}", "tool": event.get("tool"),
              "args": event.get("args") or {}}
    return receipt_from_action(action, event.get("result") or {})


def _apply_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    observed, declared = state["observed_state"], state["declared_state"]
    seq = int(event["seq"])
    if event.get("kind") == "answer_accepted":
        observed["result_status"] = event.get("status")
        return
    if event.get("kind") != "tool_call":
        return
    tool, args, result = event.get("tool"), event.get("args") or {}, event.get("result") or {}
    if tool == "list_documents":
        for doc in result.get("documents") or []:
            _append_unique(observed["surfaced_notes"], doc.get("note_id"))
    elif tool == "search":
        observed["search_refs"].append(f"search:{args.get('query', '')}")
        for hit in result.get("hits") or []:
            _append_unique(observed["surfaced_notes"], hit.get("note_id"))
    elif tool == "read" and event.get("ok"):
        _append_unique(observed["read_notes"], args.get("note_id"))
    elif tool == "record_evidence" and result.get("recorded"):
        observed["evidence_refs"].append(f"evidence:{len(observed['evidence_refs']) + 1}")
    elif tool == "record_finding" and result.get("recorded"):
        ref = str(result.get("finding_ref") or f"finding:{len(observed['finding_call_refs']) + 1}")
        observed["finding_call_refs"].append(ref)
        declared["findings"].append({
            "finding_ref": ref, "note_id": args.get("note_id"), "field": args.get("field"),
            "standing": args.get("standing"), "assertion_class": args.get("assertion_class"),
            "event_time": args.get("event_time"), "record_time": args.get("record_time"),
            "carried_forward": args.get("carried_forward"),
            "span": (result.get("server_fact") or {}).get("span"),
            "quote": result.get("quote"),
            "decision_testimony_ref": (
                result.get("testimony_ref") or args.get("decision_testimony_ref")),
        })
        testimony_ref = result.get("testimony_ref")
        if testimony_ref:
            declared["testimony_refs"].append(str(testimony_ref))
            receipt = _event_receipt(event) or {}
            if receipt.get("receipt_id"):
                declared["decision_receipt_refs"].append(str(receipt["receipt_id"]))
            declared["open_question"] = args.get("facing")
            declared["candidate_set"] = list(args.get("alternatives") or [])
            if args.get("uncertainty"):
                declared["uncertainties"].append(args["uncertainty"])
            observed["citation_resolutions"].append({
                "testimony_ref": str(testimony_ref),
                "cited": copy.deepcopy(result.get("citation_resolutions") or []),
                "checked_facts": copy.deepcopy(result.get("checked_fact_resolutions") or []),
            })
    elif tool == "note_decision":
        ref = str(result.get("testimony_ref") or f"decision:{seq}")
        declared["testimony_refs"].append(ref)
        receipt = _event_receipt(event) or {}
        if receipt.get("receipt_id"):
            declared["decision_receipt_refs"].append(str(receipt["receipt_id"]))
        declared["open_question"] = args.get("facing")
        declared["candidate_set"] = list(args.get("alternatives") or [])
        if args.get("uncertainty"):
            declared["uncertainties"].append(args["uncertainty"])
        observed["citation_resolutions"].append({
            "testimony_ref": ref,
            "cited": copy.deepcopy(result.get("citation_resolutions") or []),
            "checked_facts": copy.deepcopy(result.get("checked_fact_resolutions") or []),
        })
    elif tool == "submit_answer":
        observed["submissions"].append({"event_ref": f"layer1:{seq}",
                                        "status": args.get("status")})
        observed["gates"].append({"event_ref": f"layer1:{seq}",
                                  "accepted": bool(result.get("accepted")),
                                  "why": result.get("why") or result.get("note")})


def build_react_cycles(review: LangtraceReviewTrace,
                       manifest: TraceCompletenessManifest) -> list[dict[str, Any]]:
    """Replay one fixed cycle per Layer-1 tool interaction and terminal event."""
    require_complete(review, manifest)
    state = _initial_state()
    events = sorted(review.layer1_events, key=lambda event: int(event.get("seq", 0)))
    prior_ref: str | None = None
    cycles: list[dict[str, Any]] = []
    for event in events:
        seq = int(event["seq"])
        event_ref = f"layer1:{seq}"
        if event.get("kind") == "run_meta":
            prior_ref = event_ref
            continue
        if event.get("kind") not in {"tool_call", "answer_accepted"}:
            prior_ref = event_ref
            continue
        before = copy.deepcopy(state)
        tool = event.get("tool")
        result = event.get("result") or {}
        receipt = _event_receipt(event)
        is_testimony = (
            event.get("kind") == "tool_call"
            and tool in {"note_decision", "record_finding"}
            and (tool == "note_decision" or bool(result.get("testimony_ref")))
        )
        testimony_refs = []
        receipt_refs = []
        declared_question = None
        if is_testimony:
            args = event.get("args") or {}
            testimony_refs = [str(result.get("testimony_ref") or f"decision:{seq}")]
            if receipt is not None:
                receipt_refs = [str(receipt["receipt_id"])]
            declared_question = args.get("facing")
        if event.get("kind") == "answer_accepted":
            structural = "TERMINATION"
            actions: list[dict[str, Any]] = []
            observations = [{"event_ref": event_ref, "kind": "answer_accepted",
                             "status": event.get("status"), "value": event.get("value")}]
        else:
            structural = "SUBMISSION" if tool == "submit_answer" else "TOOL_INTERACTION"
            actions = [{"event_ref": event_ref, "tool": tool,
                        "args": copy.deepcopy(event.get("args") or {}),
                        "ok": bool(event.get("ok"))}]
            observations = [{"event_ref": event_ref,
                             "result": copy.deepcopy(event.get("result") or {})}]
        _apply_event(state, event)
        cycles.append({
            "cycle_id": f"{review.run_id}:cycle:{seq}",
            "run_id": review.run_id,
            "source_event_ids": [event_ref],
            "source_seq_range": [seq, seq],
            "source_event_time": event.get("ts"),
            "state_before": before,
            "trigger_event_refs": [prior_ref] if prior_ref else [],
            "declared_open_question": declared_question,
            "decision_testimony_refs": testimony_refs,
            "decision_receipt_refs": receipt_refs,
            "actions": actions,
            "observations": observations,
            "state_after": copy.deepcopy(state),
            "structural_kind": structural,
            "has_decision_testimony": is_testimony,
            "has_decision_receipt": bool(receipt_refs),
        })
        prior_ref = event_ref
    return cycles
