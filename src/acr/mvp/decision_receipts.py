"""Taxonomy-neutral, server-sealed records of runtime decision testimony.

The acting agent owns the testimony: the question it faced, the option it selected, and the
sources it claimed to use.  The chart tool server owns the receipt: when that testimony was
recorded, which references existed, and what observable state was available at that moment.

No post-run decision taxonomy belongs in this module.  A receipt is durable input to one or more
versioned Decision Episode projections; it is never itself a Semantica ``Decision``.
"""
from __future__ import annotations

from typing import Any, Mapping

from acr.mvp.task_presentation import content_hash

RUNTIME_DECISION_RECEIPT_SCHEMA = "acr.runtime_decision_receipt.v1"
RUNTIME_RECEIPT_PROVENANCE = "SERVER_SEALED_RUNTIME_RECEIPT"
LEGACY_RECEIPT_PROVENANCE = "LEGACY_TRACE_DERIVED"

_FORBIDDEN_TAXONOMY_FIELDS = frozenset({
    "category", "decision_function", "decision_subject", "episode",
    "error_type", "taxonomy", "taxonomy_version",
})


class DecisionReceiptError(ValueError):
    """A purported runtime receipt is malformed or no longer matches its seal."""


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _verified_refs(*resolution_groups: Any) -> list[str]:
    refs: list[str] = []
    for group in resolution_groups:
        for row in group if isinstance(group, list) else []:
            if not isinstance(row, Mapping) or not row.get("ref"):
                continue
            if row.get("verified") is True or row.get("status") == "CLAIMED_AND_VERIFIED":
                ref = str(row["ref"])
                if ref not in refs:
                    refs.append(ref)
    return refs


def _seal_runtime_decision_receipt(
        *, receipt_id: str, testimony_ref: str, source_event_ref: str, source_tool: str,
        question: Any, selected: Any,
        rationale: Any, alternatives: Any, basis_sources: Any, cited_refs: Any,
        checked_discriminating_fact_refs: Any, rule_coverage_claim: Any,
        provisional_inference: Any, uncertainty: Any, server_state: Mapping[str, Any],
        citation_resolutions: Any, checked_fact_resolutions: Any,
        structured_commitment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Seal one contemporaneous decision record without assigning an audit taxonomy.

    ``structured_commitment`` preserves a domain-native commitment already made by a tool call
    (currently one note's Standing).  It is source material, not a claim that the later projector
    must use any particular category or episode shape.
    """
    testimony = {
        "question": str(question or ""),
        "selected": str(selected or ""),
        "rationale": str(rationale or ""),
        "alternatives": _strings(alternatives),
        "basis_sources": _strings(basis_sources),
        "cited_refs": _strings(cited_refs),
        "checked_discriminating_fact_refs": _strings(
            checked_discriminating_fact_refs),
        "rule_coverage_claim": (str(rule_coverage_claim)
                                if rule_coverage_claim is not None else None),
        "provisional_inference": (str(provisional_inference)
                                  if provisional_inference is not None else None),
        "uncertainty": str(uncertainty) if uncertainty is not None else None,
    }
    payload: dict[str, Any] = {
        "schema": RUNTIME_DECISION_RECEIPT_SCHEMA,
        "receipt_id": str(receipt_id),
        "testimony_ref": str(testimony_ref),
        "source_event_ref": str(source_event_ref),
        "source_tool": str(source_tool),
        "testimony": testimony,
        "server_facts": {
            "state_at_recording": dict(server_state),
            "citation_resolutions": list(citation_resolutions or []),
            "checked_fact_resolutions": list(checked_fact_resolutions or []),
            "verified_input_refs": _verified_refs(
                citation_resolutions, checked_fact_resolutions),
        },
        "provenance": RUNTIME_RECEIPT_PROVENANCE,
    }
    if structured_commitment is not None:
        payload["structured_commitment"] = dict(structured_commitment)
    payload["receipt_hash"] = content_hash(payload)
    validate_runtime_decision_receipt(payload)
    return payload


def make_runtime_decision_receipt(
        tool: str, args: Mapping[str, Any], result: Mapping[str, Any],
        state_before: Mapping[str, Any], *, source_event_ref: str) -> dict[str, Any] | None:
    """Seal one successful decision-bearing tool call at the Layer-1 observation seam.

    The caller returns its lean tool result to the agent and writes this richer envelope only to
    the immutable trace.  This keeps instrumentation out of the next model observation while
    ensuring the receipt sees both pre-call state and the server-verified result.
    """
    testimony_ref = result.get("testimony_ref")
    if tool not in {"note_decision", "record_finding"} or not testimony_ref:
        return None
    selected = args.get("decision") if tool == "note_decision" else args.get("standing")
    commitment = None
    if tool == "record_finding":
        commitment = {
            "schema": "acr.runtime_note_finding_commitment.v1",
            "finding_ref": result.get("finding_ref"),
            "note_ref": f"note:{args.get('note_id')}",
            "field": str(args.get("field") or ""),
            "standing": str(args.get("standing") or ""),
            "assertion_class": str(args.get("assertion_class") or ""),
            "event_time": args.get("event_time"),
            "record_time": args.get("record_time"),
            "carried_forward": args.get("carried_forward"),
        }
    seq = str(source_event_ref).split(":", 1)[-1]
    return _seal_runtime_decision_receipt(
        receipt_id=f"decision-receipt:{seq}", testimony_ref=str(testimony_ref),
        source_event_ref=source_event_ref, source_tool=tool,
        question=args.get("facing"), selected=selected, rationale=args.get("because"),
        alternatives=args.get("alternatives"), basis_sources=args.get("basis_sources"),
        cited_refs=args.get("cited_refs"),
        checked_discriminating_fact_refs=args.get("checked_discriminating_fact_refs"),
        rule_coverage_claim=args.get("rule_coverage_claim"),
        provisional_inference=args.get("provisional_inference"),
        uncertainty=args.get("uncertainty"), server_state=state_before,
        citation_resolutions=result.get("citation_resolutions"),
        checked_fact_resolutions=result.get("checked_fact_resolutions"),
        structured_commitment=commitment,
    )


def validate_runtime_decision_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain validated receipt and reject taxonomy leakage or a broken seal."""
    row = dict(receipt)
    if row.get("schema") != RUNTIME_DECISION_RECEIPT_SCHEMA:
        raise DecisionReceiptError("unknown runtime Decision Receipt schema")
    leaked = _FORBIDDEN_TAXONOMY_FIELDS.intersection(row)
    if leaked:
        raise DecisionReceiptError(
            f"runtime Decision Receipt contains projection taxonomy: {sorted(leaked)!r}")
    if not row.get("receipt_id") or not row.get("testimony_ref") \
            or not row.get("source_event_ref") or not row.get("source_tool"):
        raise DecisionReceiptError("runtime Decision Receipt lacks stable source identity")
    testimony = row.get("testimony")
    if not isinstance(testimony, Mapping):
        raise DecisionReceiptError("runtime Decision Receipt lacks testimony")
    for field in ("question", "selected", "rationale", "alternatives", "basis_sources",
                  "cited_refs", "checked_discriminating_fact_refs", "rule_coverage_claim",
                  "provisional_inference", "uncertainty"):
        if field not in testimony:
            raise DecisionReceiptError(f"runtime Decision Receipt lacks testimony.{field}")
    facts = row.get("server_facts")
    if not isinstance(facts, Mapping) or "state_at_recording" not in facts:
        raise DecisionReceiptError("runtime Decision Receipt lacks server facts")
    supplied_hash = row.pop("receipt_hash", None)
    if not supplied_hash or content_hash(row) != supplied_hash:
        raise DecisionReceiptError("runtime Decision Receipt seal does not match its content")
    row["receipt_hash"] = supplied_hash
    return row


def receipt_from_action(action: Mapping[str, Any], result: Mapping[str, Any]) \
        -> dict[str, Any] | None:
    """Read a current receipt from one observed tool interaction.

    Older traces legitimately return ``None``.  Their testimony can still be projected by the
    explicit lower-confidence legacy path in ``reconstruct``; this module never invents a seal.
    """
    if action.get("tool") not in {"note_decision", "record_finding"}:
        return None
    receipt = result.get("decision_receipt")
    if not isinstance(receipt, Mapping):
        return None
    row = validate_runtime_decision_receipt(receipt)
    if str(action.get("event_ref") or "") != row["source_event_ref"]:
        raise DecisionReceiptError("runtime Decision Receipt points at a different event")
    if str(action.get("tool") or "") != row["source_tool"]:
        raise DecisionReceiptError("runtime Decision Receipt names a different source tool")
    if str(result.get("testimony_ref") or "") != row["testimony_ref"]:
        raise DecisionReceiptError("runtime Decision Receipt testimony differs from tool result")
    testimony = row["testimony"]
    expected_selected = (action.get("args") or {}).get(
        "decision" if action.get("tool") == "note_decision" else "standing")
    mirrored = {
        "question": (action.get("args") or {}).get("facing"),
        "selected": expected_selected,
        "rationale": (action.get("args") or {}).get("because"),
        "alternatives": _strings((action.get("args") or {}).get("alternatives")),
        "basis_sources": _strings((action.get("args") or {}).get("basis_sources")),
        "cited_refs": _strings((action.get("args") or {}).get("cited_refs")),
        "checked_discriminating_fact_refs": _strings(
            (action.get("args") or {}).get("checked_discriminating_fact_refs")),
        "rule_coverage_claim": (action.get("args") or {}).get("rule_coverage_claim"),
        "provisional_inference": (action.get("args") or {}).get("provisional_inference"),
        "uncertainty": (action.get("args") or {}).get("uncertainty"),
    }
    normalized = {
        key: (str(value) if value is not None and key not in {
            "alternatives", "basis_sources", "cited_refs",
            "checked_discriminating_fact_refs"} else value)
        for key, value in mirrored.items()
    }
    normalized["question"] = str(mirrored["question"] or "")
    normalized["selected"] = str(mirrored["selected"] or "")
    normalized["rationale"] = str(mirrored["rationale"] or "")
    if dict(testimony) != normalized:
        raise DecisionReceiptError("runtime Decision Receipt testimony differs from action")
    facts = row["server_facts"]
    if list(facts.get("citation_resolutions") or []) \
            != list(result.get("citation_resolutions") or []):
        raise DecisionReceiptError("runtime Decision Receipt citations differ from tool result")
    if list(facts.get("checked_fact_resolutions") or []) \
            != list(result.get("checked_fact_resolutions") or []):
        raise DecisionReceiptError("runtime Decision Receipt checked facts differ from tool result")
    if action.get("tool") == "record_finding":
        args = action.get("args") or {}
        expected_commitment = {
            "schema": "acr.runtime_note_finding_commitment.v1",
            "finding_ref": result.get("finding_ref"),
            "note_ref": f"note:{args.get('note_id')}",
            "field": str(args.get("field") or ""),
            "standing": str(args.get("standing") or ""),
            "assertion_class": str(args.get("assertion_class") or ""),
            "event_time": args.get("event_time"),
            "record_time": args.get("record_time"),
            "carried_forward": args.get("carried_forward"),
        }
        if row.get("structured_commitment") != expected_commitment:
            raise DecisionReceiptError(
                "runtime Decision Receipt commitment differs from finding result")
    elif "structured_commitment" in row:
        raise DecisionReceiptError(
            "generic runtime Decision Receipt contains a structured finding commitment")
    return row
