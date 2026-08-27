"""Run-scoped human review view over one explicitly selected Decision Episode analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from acr.mvp.decision_receipts import receipt_from_action
from acr.mvp.task_presentation import ContractSnapshot

_PRIORITY_REVIEW_CODES = frozenset({
    "MISSING_RUNTIME_TESTIMONY",
    "MODEL_KNOWLEDGE_USED",
    "RULE_NOT_OFFERED",
    "PROVISIONAL_INFERENCE_USED",
    "NO_APPLICABLE_RULE_CLAIMED",
    "RULE_APPLICATION_REVIEW",
    "INVENTORY_PAGE_PARTIAL",
})

_PHASES_BY_FUNCTION = {
    "where_to_look": ("EVIDENCE_DISCOVERY", "Find the evidence"),
    "is_this_it": ("EVIDENCE_JUDGMENT", "Judge the evidence"),
    "what_it_asserts": ("EVIDENCE_JUDGMENT", "Judge the evidence"),
    "when_it_happened": ("EVIDENCE_JUDGMENT", "Judge the evidence"),
    "standing": ("EVIDENCE_JUDGMENT", "Judge the evidence"),
    "same_or_ordered": ("EVIDENCE_RESOLUTION", "Compare the evidence"),
    "corroborate": ("EVIDENCE_RESOLUTION", "Compare the evidence"),
    "which_wins": ("EVIDENCE_RESOLUTION", "Resolve the evidence"),
    "scope": ("CASE_ASSESSMENT", "Assess the case"),
    "infer": ("CASE_ASSESSMENT", "Assess the case"),
    "is_it_absent": ("CASE_ASSESSMENT", "Assess the case"),
    "enough": ("CASE_ASSESSMENT", "Decide whether evidence is enough"),
    "what_to_answer": ("ANSWER_FORMATION", "Form the answer"),
    "other": ("OTHER_DECISION", "Other material decision"),
}


def _phase_for(decision_function: Any, *,
               actions: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    if str(decision_function) == "where_to_look" \
            and any(row.get("tool") == "record_evidence" for row in actions or []):
        return "EVIDENCE_ASSEMBLY", "Assemble the evidence record"
    return _PHASES_BY_FUNCTION.get(
        str(decision_function), ("OTHER_DECISION", "Other material decision"))


def _cycles_for(artifact: dict[str, Any], episode: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = set(episode.get("source_cycle_ids") or [])
    return [cycle for cycle in artifact.get("cycles") or []
            if cycle.get("cycle_id") in wanted]


def _choice_cycle(artifact: dict[str, Any], cycles: list[dict[str, Any]]) \
        -> dict[str, Any] | None:
    """Locate the last cycle that actually bears the episode's material choice."""
    annotations = artifact.get("cycle_annotations") or {}
    bearing = [cycle for cycle in cycles
               if (annotations.get(str(cycle.get("cycle_id"))) or {}).get("role")
               == "DECISION_BEARING"]
    if bearing:
        if len(bearing) != 1:
            raise ValueError("an atomic Decision Episode must have exactly one bearing cycle")
        return bearing[0]
    testified = [cycle for cycle in cycles if cycle.get("decision_testimony_refs")]
    return testified[-1] if testified else (cycles[-1] if cycles else None)


def _testimonies(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        result_by_event = {row.get("event_ref"): row.get("result") or {}
                           for row in cycle.get("observations") or []}
        for action in cycle.get("actions") or []:
            tool = action.get("tool")
            if tool not in {"note_decision", "record_finding"}:
                continue
            args = action.get("args") or {}
            result = result_by_event.get(action.get("event_ref"), {})
            receipt = receipt_from_action(action, result)
            if tool == "record_finding" and not result.get("testimony_ref"):
                # Pre-upgrade findings may point at an earlier testimony. That link is added
                # explicitly by human_review_view; it is not local testimony for this choice.
                continue
            reported = result.get("self_reported") or {}
            sealed = (receipt or {}).get("testimony") or {}
            facts = (receipt or {}).get("server_facts") or {}
            decision = (sealed.get("selected") if receipt else
                        (args.get("decision") if tool == "note_decision" else
                         reported.get("decision") or
                         f"{args.get('note_id')} is {args.get('standing')} for "
                         f"{args.get('field')} ({args.get('assertion_class')})"))
            rows.append({
                "testimony_ref": (receipt or {}).get("testimony_ref")
                or result.get("testimony_ref")
                or next(iter(cycle.get("decision_testimony_refs") or []), None),
                "runtime_receipt_ref": (receipt or {}).get("receipt_id"),
                "runtime_receipt_schema": (receipt or {}).get("schema"),
                "runtime_receipt_hash": (receipt or {}).get("receipt_hash"),
                "runtime_receipt_provenance": (receipt or {}).get("provenance")
                or "LEGACY_TRACE_DERIVED",
                "source_cycle_id": cycle.get("cycle_id"),
                "link_scope": "EPISODE_LOCAL",
                "provenance": "SELF_REPORTED",
                "testimony_tool": tool,
                "facing": (sealed.get("question") if receipt else args.get("facing")),
                "decision": decision,
                "because": (sealed.get("rationale") if receipt else args.get("because")),
                "basis_sources": (sealed.get("basis_sources") if receipt
                                  else args.get("basis_sources") or []),
                "rule_coverage_claim": (sealed.get("rule_coverage_claim") if receipt
                                        else args.get("rule_coverage_claim")),
                "provisional_inference": (sealed.get("provisional_inference") if receipt
                                          else args.get("provisional_inference")),
                "alternatives": (sealed.get("alternatives") if receipt
                                 else args.get("alternatives") or []),
                "uncertainty": (sealed.get("uncertainty") if receipt
                                else args.get("uncertainty")),
                "structured_commitment": (receipt or {}).get("structured_commitment"),
                "citation_resolutions": (facts.get("citation_resolutions") if receipt
                                         else result.get("citation_resolutions") or []),
                "checked_fact_resolutions": (facts.get("checked_fact_resolutions") if receipt
                                             else result.get("checked_fact_resolutions") or []),
                "reference_resolution_provenance": "DETERMINISTIC_DERIVED",
            })
    return rows


def _findings(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for cycle in cycles:
        declared = (cycle.get("state_after") or {}).get("declared_state") or {}
        for finding in declared.get("findings") or []:
            ref = str(finding.get("finding_ref") or f"finding:{len(rows) + 1}")
            rows[ref] = {**finding, "standing_provenance": "SELF_REPORTED",
                         "call_and_span_provenance": "SERVER_FACT"}
    return list(rows.values())


def _finding_testimony_consumers(cycles: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map testimony refs to the finding events they were asked to justify.

    One old testimony reused by several finding calls is deterministic evidence that the
    instrumentation bundled independently reviewable standings. It does not prove the semantic
    rationale was wrong, but it does prove the audit unit was not atomic.
    """
    consumers: dict[str, list[str]] = {}
    for cycle in cycles:
        results = {str(row.get("event_ref")): row.get("result") or {}
                   for row in cycle.get("observations") or []}
        for action in cycle.get("actions") or []:
            if action.get("tool") != "record_finding" or not action.get("ok"):
                continue
            args = action.get("args") or {}
            result = results.get(str(action.get("event_ref")), {})
            ref = result.get("testimony_ref") or args.get("decision_testimony_ref")
            if ref:
                consumers.setdefault(str(ref), []).append(str(action.get("event_ref")))
    return consumers


def _evidence_packet(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The smallest inspectable source packet beside a decision: quote + standing + time."""
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        observations = {str(row.get("event_ref")): row.get("result") or {}
                        for row in episode.get("observations") or []}
        for action in episode.get("actions") or []:
            tool, args = action.get("tool"), action.get("args") or {}
            result = observations.get(str(action.get("event_ref")), {})
            if tool == "record_finding" and result.get("recorded"):
                rows.append({
                    "kind": "NOTE_FINDING", "event_ref": action.get("event_ref"),
                    "finding_ref": result.get("finding_ref"),
                    "note_id": args.get("note_id"), "field": args.get("field"),
                    "standing": args.get("standing"),
                    "assertion_class": args.get("assertion_class"),
                    "event_time": args.get("event_time"),
                    "record_time": args.get("record_time"),
                    "carried_forward": args.get("carried_forward"),
                    "span": (result.get("server_fact") or {}).get("span"),
                    "quote": result.get("quote"),
                    "claim_provenance": "SELF_REPORTED",
                    "quote_provenance": "SERVER_RESOLVED",
                })
            elif tool == "record_evidence" and result.get("recorded"):
                rows.append({
                    "kind": "ANSWER_EVIDENCE", "event_ref": action.get("event_ref"),
                    "note_id": args.get("note_id"), "field": args.get("field"),
                    "supports": args.get("supports"),
                    "span": [args.get("start"), args.get("end")],
                    "quote": result.get("quote"),
                    "claim_provenance": "SELF_REPORTED",
                    "quote_provenance": "SERVER_RESOLVED",
                })
    return _unique_rows(rows)


def _disposition(testimonies: list[dict[str, Any]]) -> str:
    resolutions = [row for testimony in testimonies
                   for row in [*(testimony.get("citation_resolutions") or []),
                               *(testimony.get("checked_fact_resolutions") or [])]]
    verified_refs = {str(row.get("ref")) for row in resolutions
                     if row.get("ref") is not None
                     and (row.get("status") == "CLAIMED_AND_VERIFIED"
                          or row.get("verified") is True)}
    statuses = {str(row.get("status")) for row in resolutions
                if str(row.get("ref")) not in verified_refs}
    claims = {str(row.get("rule_coverage_claim")) for row in testimonies}
    if statuses & {"CLAIMED_UNKNOWN"}:
        return "INSTRUMENTATION_QUESTION"
    if claims & {"AMBIGUOUS_RULE", "CONFLICTING_RULES", "COVERED_WITH_INTERPRETATION"}:
        return "RULE_APPLICATION_QUESTION"
    if claims == {"OPERATIONAL_DISCRETION"}:
        return "PERMITTED_DISCRETION"
    if claims & {"NO_APPLICABLE_RULE"}:
        return "NEEDS_ATTRIBUTION"
    return "NEEDS_HUMAN_ADJUDICATION"


def _grounding_assessment(testimonies: list[dict[str, Any]], *,
                          decision_function: Any, decision_subject: Any) -> dict[str, Any]:
    """Separate reference availability, semantic support, and model judgment."""
    local = [row for row in testimonies if row.get("link_scope") == "EPISODE_LOCAL"]
    resolutions = [
        resolution
        for testimony in local
        for resolution in [*(testimony.get("citation_resolutions") or []),
                           *(testimony.get("checked_fact_resolutions") or [])]
        if resolution.get("ref") is not None
    ]
    verified = [row for row in resolutions
                if row.get("verified") is True
                or row.get("status") == "CLAIMED_AND_VERIFIED"]
    if not resolutions:
        reference_status = "NO_REFERENCES_CLAIMED"
    elif len(verified) == len(resolutions):
        reference_status = "ALL_REFERENCES_RESOLVED"
    elif verified:
        reference_status = "SOME_REFERENCES_UNRESOLVED"
    else:
        reference_status = "NO_REFERENCES_RESOLVED"

    claims = list(dict.fromkeys(
        str(row.get("rule_coverage_claim")) for row in local
        if row.get("rule_coverage_claim")
    ))
    bases = {str(source) for row in local for source in row.get("basis_sources") or []}
    subject = str(decision_subject or "")
    if "own_knowledge" in bases:
        judgment_mode = "OUTSIDE_MATERIAL_MODEL_JUDGMENT"
    elif subject in {"retrieval_query_batch", "retrieval_source", "retrieval_document_set"}:
        judgment_mode = "POLICY_GUIDED_OPERATIONAL_JUDGMENT"
    elif subject == "retrieval_inventory":
        judgment_mode = "OPERATIONAL_INSTRUCTION_APPLICATION"
    elif set(claims) & {"NO_APPLICABLE_RULE", "OPERATIONAL_DISCRETION"}:
        judgment_mode = "UNGUIDED_OR_OPERATIONAL_MODEL_JUDGMENT"
    elif claims:
        judgment_mode = "POLICY_APPLICATION_JUDGMENT"
    else:
        judgment_mode = "MODEL_JUDGMENT_WITHOUT_RUNTIME_GROUNDING_CLAIM"

    return {
        "reference_resolution_status": reference_status,
        "resolved_refs": [str(row["ref"]) for row in verified],
        "unresolved_refs": [str(row["ref"]) for row in resolutions if row not in verified],
        "reference_resolution_provenance": "SERVER_FACT_OR_DETERMINISTIC_DERIVED",
        "agent_rule_coverage_claims": claims,
        "rule_coverage_claim_provenance": "SELF_REPORTED",
        "semantic_entailment_status": "NOT_ESTABLISHED_BY_REFERENCE_CHECK",
        "semantic_entailment_provenance": "NOT_ASSESSED",
        "judgment_mode": judgment_mode,
        "uses_model_judgment": bool(decision_function),
        "uses_outside_material_knowledge": "own_knowledge" in bases,
    }


def _review_attention(testimonies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Route reviewable claims without silently upgrading them into error findings."""
    rows: list[dict[str, Any]] = []
    if not testimonies:
        rows.append({
            "code": "MISSING_RUNTIME_TESTIMONY",
            "severity": "REVIEW",
            "title": "No runtime Decision Testimony",
            "detail": "The trace contains this reconstructed decision but no contemporaneous "
                      "self-report of its basis.",
            "provenance": "SERVER_FACT",
            "is_error": False,
            "route": "INSTRUMENTATION_QUESTION",
        })
    if any("own_knowledge" in (row.get("basis_sources") or []) for row in testimonies):
        rows.append({
            "code": "MODEL_KNOWLEDGE_USED",
            "severity": "REVIEW",
            "title": "Model used knowledge outside the supplied material",
            "detail": "The agent self-reported own_knowledge as a basis for this decision.",
            "provenance": "SELF_REPORTED",
            "is_error": False,
            "route": "NEEDS_HUMAN_ADJUDICATION",
        })
    unavailable_refs = sorted({
        str(resolution.get("ref"))
        for testimony in testimonies
        for resolution in testimony.get("citation_resolutions") or []
        if resolution.get("status") == "CLAIMED_NOT_OFFERED"
    })
    if unavailable_refs:
        rows.append({
            "code": "RULE_NOT_OFFERED",
            "severity": "REVIEW",
            "title": "Claimed rule was not supplied to this run",
            "detail": "The repository knows this reference, but it was absent from the "
                      "run's Task Presentation.",
            "refs": unavailable_refs,
            "provenance": "DETERMINISTIC_DERIVED",
            "is_error": False,
            "route": "NEEDS_ATTRIBUTION",
        })
    all_resolutions = [
        resolution
        for testimony in testimonies
        for resolution in [*(testimony.get("citation_resolutions") or []),
                           *(testimony.get("checked_fact_resolutions") or [])]
    ]
    verified_refs = {
        str(resolution.get("ref")) for resolution in all_resolutions
        if resolution.get("ref") is not None
        and (resolution.get("status") == "CLAIMED_AND_VERIFIED"
             or resolution.get("verified") is True)
    }
    unverified_refs = sorted({
        str(resolution.get("ref"))
        for resolution in all_resolutions
        if resolution.get("ref") is not None
        and str(resolution.get("ref")) not in verified_refs
        and (resolution.get("status") == "CLAIMED_UNKNOWN"
             or resolution.get("verified") is False)
    })
    if unverified_refs:
        rows.append({
            "code": "REFERENCE_UNVERIFIED",
            "severity": "REVIEW",
            "title": "Claimed basis could not be verified",
            "detail": "The deterministic resolver could not tie this reference to material "
                      "available or observed in the run.",
            "refs": unverified_refs,
            "provenance": "DETERMINISTIC_DERIVED",
            "is_error": False,
            "route": "NEEDS_ATTRIBUTION",
        })
    inferences = list(dict.fromkeys(
        str(row.get("provisional_inference")).strip()
        for row in testimonies if str(row.get("provisional_inference") or "").strip()
    ))
    if inferences:
        rows.append({
            "code": "PROVISIONAL_INFERENCE_USED",
            "severity": "REVIEW",
            "title": "Agent added a provisional inference",
            "detail": "This assumption was stated at runtime and needs semantic review.",
            "inferences": inferences,
            "provenance": "SELF_REPORTED",
            "is_error": False,
            "route": "NEEDS_HUMAN_ADJUDICATION",
        })
    if any(row.get("rule_coverage_claim") == "NO_APPLICABLE_RULE" for row in testimonies):
        rows.append({
            "code": "NO_APPLICABLE_RULE_CLAIMED",
            "severity": "REVIEW",
            "title": "Agent found no applicable supplied rule",
            "detail": "This is a runtime claim; it does not prove a guideline gap until "
                      "the situation and offered clauses are adjudicated.",
            "provenance": "SELF_REPORTED",
            "is_error": False,
            "route": "NEEDS_ATTRIBUTION",
        })
    interpretive_claims = list(dict.fromkeys(
        str(row.get("rule_coverage_claim")) for row in testimonies
        if row.get("rule_coverage_claim") in {
            "COVERED_WITH_INTERPRETATION", "AMBIGUOUS_RULE", "CONFLICTING_RULES"}
    ))
    if interpretive_claims:
        rows.append({
            "code": "RULE_APPLICATION_REVIEW",
            "severity": "REVIEW",
            "title": "Rule application required interpretation",
            "detail": "Review whether the offered clause actually supports this application; "
                      "the runtime claim alone is not semantic proof.",
            "claims": interpretive_claims,
            "provenance": "SELF_REPORTED",
            "is_error": False,
            "route": "RULE_APPLICATION_QUESTION",
        })
    return rows


def _presentation(run_dir: Path | None, expected_hash: str) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = Path(run_dir) / "task_presentation.json"
    if not path.exists():
        return None
    snapshot = ContractSnapshot.from_path(path)
    if snapshot.presentation_hash != expected_hash:
        raise ValueError("Task Presentation does not match the selected analysis")
    return snapshot.to_dict()


def _execution_summary(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    submissions, gates, result, final_submission = [], [], None, None
    for cycle in cycles:
        result_by_event = {row.get("event_ref"): row.get("result") or {}
                           for row in cycle.get("observations") or []}
        for action in cycle.get("actions") or []:
            if action.get("tool") == "submit_answer":
                args = action.get("args") or {}
                observed = result_by_event.get(action.get("event_ref"), {})
                submission = {
                    "event_ref": action.get("event_ref"),
                    "status": args.get("status"),
                    "value": args.get("value"),
                    "reasoning": args.get("reasoning"),
                    "accepted": observed.get("accepted") is True,
                }
                submissions.append({"event_ref": action.get("event_ref"),
                                    "status": args.get("status")})
                final_submission = submission
        observed = (cycle.get("state_after") or {}).get("observed_state") or {}
        gates = observed.get("gates") or gates
        result = observed.get("result_status") or result
    if final_submission is not None and not final_submission["accepted"]:
        final_submission["accepted"] = any(
            row.get("accepted") is True or row.get("passed") is True for row in gates)
    return {"submissions": submissions, "gates": gates, "result": result,
            "final_submission": final_submission}


def _process_ladder(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    if not cycles:
        return {"surfaced": 0, "read": 0, "judged": 0, "cited": 0}
    state = (cycles[-1].get("state_after") or {})
    observed, declared = state.get("observed_state") or {}, state.get("declared_state") or {}
    cited = {str(row.get("ref")) for packet in observed.get("citation_resolutions") or []
             for row in packet.get("cited") or []}
    return {"surfaced": len(observed.get("surfaced_notes") or []),
            "read": len(observed.get("read_notes") or []),
            "judged": len(declared.get("findings") or []), "cited": len(cited),
            "provenance": "DETERMINISTIC_DERIVED",
            "interpretation": "process counts, not Complete Answer or clinical coverage"}


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _hydrate_causal_edges(graph_edges: list[dict[str, Any]],
                          assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore readable assertion evidence onto graph-confirmed causal triples.

    Semantica core projection hashes potentially identifying runtime locators. The sealed analysis
    retains the complete CausalAssertion. Human review may show that evidence only after confirming
    that its typed triple is actually present in ContextGraph.
    """
    def triple(row: dict[str, Any]) -> tuple[str, str, str]:
        return (str(row.get("source_episode_id") or ""),
                str(row.get("target_episode_id") or ""),
                str(row.get("relationship_type") or ""))

    by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    for assertion in assertions:
        key = triple(assertion)
        if key in by_triple:
            raise ValueError("sealed analysis contains a duplicate causal assertion triple")
        by_triple[key] = assertion
    if {triple(edge) for edge in graph_edges} != set(by_triple):
        raise ValueError("Semantica causal edges do not match the sealed analysis assertions")

    hydrated: list[dict[str, Any]] = []
    for edge in graph_edges:
        assertion = by_triple[triple(edge)]
        row = dict(edge)
        for field in ("assertion_id", "evidence_refs", "reasoning", "provenance"):
            if assertion.get(field) not in (None, "", []):
                row[field] = assertion[field]
        hydrated.append(row)
    return hydrated


def _step_attention(episodes: list[dict[str, Any]], *,
                    final_submission: dict[str, Any] | None = None,
                    conclusion_step: bool = False) -> tuple[list[dict[str, Any]],
                                                            list[dict[str, Any]]]:
    priority, technical = [], []
    for episode in episodes:
        for raw in episode.get("review_attention") or []:
            row = {**raw, "episode_id": episode.get("episode_id"),
                   "decision_function": episode.get("decision_function")}
            # A reason recorded on submit_answer is contemporaneous runtime evidence.  Do not
            # ask a human to repair a missing note_decision for the final answer when that reason
            # is already present on the actual submission action.
            if (conclusion_step and row.get("code") == "MISSING_RUNTIME_TESTIMONY"
                    and str((final_submission or {}).get("reasoning") or "").strip()):
                continue
            (priority if row.get("code") in _PRIORITY_REVIEW_CODES else technical).append(row)
    return _unique_rows(priority), _unique_rows(technical)


def _step_state_result(episode: dict[str, Any]) -> dict[str, int]:
    state = episode.get("state_after") or {}
    observed = state.get("observed_state") or {}
    declared = state.get("declared_state") or {}
    return {
        "surfaced_notes": len(observed.get("surfaced_notes") or []),
        "read_notes": len(observed.get("read_notes") or []),
        "findings": len(declared.get("findings") or []),
        "uncertainties": len(declared.get("uncertainties") or []),
    }


def _trace_support(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    refs: list[str] = []
    for episode in episodes:
        source_map = ((episode.get("reconstruction") or {})
                      .get("source_refs_by_field") or {})
        for field in ("material_question", "decision", "decision_rationale",
                      "model_interpretation", "claimed_basis_summary"):
            refs.extend(str(ref) for ref in source_map.get(field) or [])
    unique = list(dict.fromkeys(refs))
    return {"status": "TRACE_CITED" if unique else "NO_FIELD_SOURCE_REFS",
            "ref_count": len(unique), "refs": unique}


def _coverage_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose what the agent inspected without pretending that this proves completeness."""
    listings: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    reads: list[dict[str, Any]] = []
    evidence_notes: list[str] = []
    for episode in episodes:
        observations = {
            str(row.get("event_ref")): row.get("result") or {}
            for row in episode.get("observations") or []
        }
        for action in episode.get("actions") or []:
            tool, args = str(action.get("tool") or ""), action.get("args") or {}
            result = observations.get(str(action.get("event_ref")), {})
            objective = str(args.get("objective") or "").strip() or None
            if tool == "list_documents":
                filters = {key: args.get(key) for key in (
                    "doc_type_contains", "date_from", "date_to")
                    if args.get(key) not in (None, "")}
                documents = result.get("documents") or []
                returned = int(result.get("returned", len(documents)))
                total = result.get("total")
                try:
                    total = int(total)
                except (TypeError, ValueError):
                    total = None
                offset = int(result.get("offset", args.get("offset", 0) or 0))
                limit = int(result.get("limit", args.get("limit", 200) or 200))
                page_complete = (bool(result.get("page_complete"))
                                 if result.get("page_complete") is not None
                                 else total is not None and offset + returned >= total)
                listings.append({
                    "event_ref": action.get("event_ref"),
                    "objective": objective,
                    "filters": filters,
                    "unfiltered": not filters,
                    "documents_returned": returned,
                    "documents_total": total,
                    "offset": offset,
                    "limit": limit,
                    "page_complete": page_complete,
                    "unreturned_count": (max(total - (offset + returned), 0)
                                         if total is not None else None),
                    "type_summary": result.get("types") or [],
                })
            elif tool == "search":
                hits = result.get("hits") or result.get("results") or []
                note_ids = [str(row.get("note_id")) for row in hits if row.get("note_id")]
                unique_note_ids = list(dict.fromkeys(note_ids))
                searches.append({
                    "event_ref": action.get("event_ref"),
                    "query": args.get("query"),
                    "objective": objective,
                    "results_returned": len(hits),
                    "unique_notes_returned": len(unique_note_ids),
                    "duplicate_hits": len(note_ids) - len(unique_note_ids),
                    "note_ids": unique_note_ids,
                })
            elif tool == "read":
                reads.append({
                    "event_ref": action.get("event_ref"),
                    "note_id": args.get("note_id"),
                    "objective": objective,
                    "truncated": result.get("truncated"),
                })
            elif tool == "record_evidence" and args.get("note_id"):
                evidence_notes.append(str(args["note_id"]))

    final_state = (episodes[-1].get("state_after") or {}) if episodes else {}
    observed = final_state.get("observed_state") or {}
    surfaced = list(dict.fromkeys(str(value)
                                  for value in observed.get("surfaced_notes") or []))
    read_notes = list(dict.fromkeys(
        [str(row["note_id"]) for row in reads if row.get("note_id")]
        or [str(value) for value in observed.get("read_notes") or []]
    ))
    return {
        "assessment": "HUMAN_TO_ADJUDICATE",
        "assessment_prompt": (
            "These are the records and searches actually observed; whether they cover every "
            "material source for the task is a human judgment."
        ),
        "surfaced_count": len(surfaced),
        "read_count": len(read_notes),
        "unfiltered_listing_done": any(row["unfiltered"] for row in listings),
        "inventory_page_complete": (
            all(row["page_complete"] for row in listings if row["unfiltered"])
            if any(row["unfiltered"] for row in listings) else None),
        "listings": listings,
        "searches": searches,
        "reads": reads,
        "read_note_ids": read_notes,
        "evidence_note_ids": list(dict.fromkeys(evidence_notes)),
    }


def _decision_dependencies(episodes: list[dict[str, Any]],
                           trace_support: dict[str, Any]) -> dict[str, Any]:
    """Keep the decision's claimed inputs inspectable beside its outcome."""
    testimonies = [row for episode in episodes
                   for row in episode.get("runtime_testimonies") or []]
    candidates = list(dict.fromkeys(
        str(candidate)
        for episode in episodes
        for candidate in ((episode.get("reconstruction") or {}).get("candidate_set") or [])
    ))
    alternatives = list(dict.fromkeys(
        str(value) for testimony in testimonies
        for value in testimony.get("alternatives") or [] if str(value).strip()
    ))
    inferences = list(dict.fromkeys(
        str(testimony.get("provisional_inference")).strip()
        for testimony in testimonies
        if str(testimony.get("provisional_inference") or "").strip()
    ))
    uncertainties = list(dict.fromkeys(
        str(testimony.get("uncertainty")).strip()
        for testimony in testimonies
        if str(testimony.get("uncertainty") or "").strip()
    ))
    citation_resolutions = _unique_rows([
        resolution for testimony in testimonies
        for resolution in testimony.get("citation_resolutions") or []
    ])
    checked_facts = _unique_rows([
        resolution for testimony in testimonies
        for resolution in testimony.get("checked_fact_resolutions") or []
    ])
    reconstructed_basis = list(dict.fromkeys(
        str(value)
        for episode in episodes
        for value in (((episode.get("reconstruction") or {})
                       .get("claimed_basis_summary")) or [])
        if str(value).strip()
    ))
    verified_reference_summary = list(dict.fromkeys(
        str(value)
        for episode in episodes
        for value in (((episode.get("reconstruction") or {})
                       .get("verified_reference_summary")) or [])
        if str(value).strip()
    ))
    prior_finding_rows: list[dict[str, Any]] = []
    for episode in episodes:
        state = episode.get("choice_state_before") or episode.get("state_before") or {}
        prior_finding_rows.extend(
            ((state.get("declared_state") or {}).get("findings") or []))
    prior_findings = _unique_rows(prior_finding_rows)
    return {
        "candidate_set": candidates,
        "alternatives_considered": alternatives,
        "provisional_inferences": inferences,
        "unresolved_uncertainties": uncertainties,
        "citation_resolutions": citation_resolutions,
        "checked_discriminating_facts": checked_facts,
        "trace_reconstructed_basis": reconstructed_basis,
        "verified_reference_summary": verified_reference_summary,
        "available_findings_before_choice": prior_findings,
        "trace_support": trace_support,
    }


def _task_brief(run_dir: Path | None,
                presentation: dict[str, Any] | None) -> dict[str, Any]:
    """Return only the task identity/question needed before a blinded review begins."""
    reference = (presentation or {}).get("task_contract_ref") or {}
    brief = {
        "question": None,
        "contract_id": reference.get("id"),
        "contract_version": reference.get("version"),
        "arm_id": (presentation or {}).get("arm_id"),
    }
    if run_dir is None or presentation is None:
        return brief
    root = Path(run_dir).resolve()
    raw_ref = str(presentation.get("rendered_prompt_artifact_ref") or "").strip()
    if not raw_ref:
        return brief
    prompt_path = (root / raw_ref).resolve(strict=False)
    try:
        prompt_path.relative_to(root)
    except ValueError:
        return brief
    if not prompt_path.is_file():
        return brief
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("QUESTION:"):
            brief["question"] = line.split(":", 1)[1].strip() or None
            break
    return brief


def _concise_step(episodes: list[dict[str, Any]], *, kind: str,
                  final_submission: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(episodes) != 1:
        raise ValueError("one human-review step must contain exactly one atomic decision")
    first, last = episodes[0], episodes[-1]
    testimonies = [row for episode in episodes
                   for row in episode.get("runtime_testimonies") or []]
    # An explicitly linked testimony from an earlier episode is supporting context for
    # this choice, not the current choice itself. Only episode-local testimony may
    # override the reconstructed decision/question/rationale shown to the reviewer.
    local_testimonies = [row for row in testimonies
                         if row.get("link_scope") == "EPISODE_LOCAL"]
    last_testimony = local_testimonies[-1] if local_testimonies else None
    reconstructed = last.get("reconstruction") or {}
    conclusion_step = kind == "conclusion"
    priority, technical = _step_attention(
        episodes, final_submission=final_submission, conclusion_step=conclusion_step)
    trace_support = _trace_support(episodes)
    coverage = _coverage_summary(episodes)
    partial_inventory = [row for row in coverage.get("listings") or []
                         if row.get("unfiltered") and row.get("page_complete") is False]
    if partial_inventory:
        priority = _unique_rows([*priority, {
            "code": "INVENTORY_PAGE_PARTIAL",
            "severity": "REVIEW",
            "title": "The unfiltered note inventory was only partially returned",
            "detail": "The agent learned the chart's total count and type summary, but did not "
                      "receive every exact note row in this listing. Review whether later searches "
                      "and reads adequately covered the unreturned portion.",
            "pages": [{key: row.get(key) for key in (
                "event_ref", "documents_returned", "documents_total", "unreturned_count")}
                for row in partial_inventory],
            "provenance": "DETERMINISTIC_DERIVED",
            "is_error": False,
            "route": "COVERAGE_QUESTION",
        }])
    if trace_support["status"] == "TRACE_CITED":
        supported_missing = [row for row in priority
                             if row.get("code") == "MISSING_RUNTIME_TESTIMONY"]
        priority = [row for row in priority
                    if row.get("code") != "MISSING_RUNTIME_TESTIMONY"]
        technical = _unique_rows([*technical, *supported_missing])
    decisions = _unique_rows([
        {"decision": (episode.get("reconstruction") or {}).get("decision")}
        for episode in episodes
        if (episode.get("reconstruction") or {}).get("decision")
    ])
    basis_sources = list(dict.fromkeys(
        str(source) for testimony in testimonies
        for source in testimony.get("basis_sources") or []))
    clauses = _unique_rows([
        clause for episode in episodes
        for clause in episode.get("relevant_offered_clauses") or []])
    reason = ((last_testimony or {}).get("because")
              or (final_submission or {}).get("reasoning") if conclusion_step
              else (last_testimony or {}).get("because"))
    if not reason:
        reason = (reconstructed.get("decision_rationale")
                  or reconstructed.get("model_interpretation"))
    decision = ((last_testimony or {}).get("decision") or reconstructed.get("decision"))
    if conclusion_step and final_submission:
        decision = final_submission.get("value") or decision
    decision_subject = reconstructed.get("decision_subject")
    actions = [action for episode in episodes for action in episode.get("actions") or []]
    phase, phase_label = _phase_for(last.get("decision_function"), actions=actions)
    return {
        "step_id": str(first.get("episode_id")),
        "audit_unit": "ATOMIC_DECISION",
        "kind": kind,
        "phase": phase,
        "phase_label": phase_label,
        "decision_function": last.get("decision_function"),
        "decision_subject": decision_subject,
        "episode_ids": [str(row.get("episode_id")) for row in episodes],
        "episode_count": len(episodes),
        "question": ((last_testimony or {}).get("facing")
                     or reconstructed.get("material_question")
                     or reconstructed.get("scenario")),
        "question_provenance": ("SELF_REPORTED"
                                if (last_testimony or {}).get("facing")
                                else "MODEL_RECONSTRUCTED"),
        "decision": decision,
        "reason": reason,
        "reason_provenance": ("SELF_REPORTED" if last_testimony
                              else "RUNTIME_SUBMISSION" if conclusion_step
                              and (final_submission or {}).get("reasoning")
                              else "TRACE_RECONSTRUCTED"
                              if trace_support["status"] == "TRACE_CITED"
                              else "MODEL_RECONSTRUCTED"),
        "trace_support": trace_support,
        "grounding_assessment": last.get("grounding_assessment") or {},
        "basis_sources": basis_sources,
        "coverage": coverage,
        "evidence": _evidence_packet(episodes),
        "dependencies": _decision_dependencies(episodes, trace_support),
        "guidelines": clauses,
        "review_attention": priority,
        "technical_attention": technical,
        "decisions": [row["decision"] for row in decisions],
        "state_result": _step_state_result(last),
        "detail_episodes": episodes,
        "link_to_next": None,
    }


def _concise_review_chain(episodes: list[dict[str, Any]], execution: dict[str, Any],
                          causal_edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Human reading order: atomic decisions grouped visually; outcome revealed last."""
    conclusion = execution.get("final_submission")
    steps: list[dict[str, Any]] = []
    final_decision: dict[str, Any] | None = None
    final_index = next((index for index in range(len(episodes) - 1, -1, -1)
                        if episodes[index].get("decision_function") == "what_to_answer"), None)

    for index, episode in enumerate(episodes):
        function = episode.get("decision_function")
        kind = ("conclusion" if function == "what_to_answer"
                else "retrieval" if function == "where_to_look" else "judgment")
        step = _concise_step([episode], kind=kind, final_submission=conclusion)
        if index == final_index:
            final_decision = step
        else:
            steps.append(step)

    ordered = [*steps, *([final_decision] if final_decision else [])]
    step_by_episode = {
        str(episode_id): step
        for step in ordered for episode_id in step.get("episode_ids") or []
    }
    for current in ordered:
        source_ids = set(current["episode_ids"])
        links: list[dict[str, Any]] = []
        for edge in causal_edges:
            if edge.get("source_episode_id") not in source_ids:
                continue
            target = step_by_episode.get(str(edge.get("target_episode_id")))
            if target is None:
                continue
            links.append({
                "relationship_type": edge.get("relationship_type"),
                "target_step_id": target.get("step_id"),
                "target_title": target.get("phase_label"),
                "target_question": target.get("question"),
                "target_decision": target.get("decision"),
                **({"assertion_id": edge.get("assertion_id")}
                   if edge.get("assertion_id") else {}),
                **({"evidence_refs": edge.get("evidence_refs")}
                   if edge.get("evidence_refs") else {}),
                **({"reasoning": edge.get("reasoning")}
                   if edge.get("reasoning") else {}),
                "provenance": edge.get("provenance") or "SEMANTICA_CAUSAL_ASSERTION",
            })
        current["causal_links"] = _unique_rows(links)
    for current, following in zip(ordered, ordered[1:]):
        source_ids, target_ids = set(current["episode_ids"]), set(following["episode_ids"])
        edge = next((row for row in causal_edges
                     if row.get("source_episode_id") in source_ids
                     and row.get("target_episode_id") in target_ids), None)
        current["link_to_next"] = (
            {"kind": "CAUSAL", "relationship_type": edge.get("relationship_type"),
             **({"assertion_id": edge.get("assertion_id")}
                if edge.get("assertion_id") else {}),
             **({"evidence_refs": edge.get("evidence_refs")}
                if edge.get("evidence_refs") else {}),
             **({"reasoning": edge.get("reasoning")}
                if edge.get("reasoning") else {}),
             "provenance": edge.get("provenance") or "SEMANTICA_CAUSAL_ASSERTION"}
            if edge else {"kind": "TEMPORAL_ONLY", "relationship_type": None})

    phases: list[dict[str, Any]] = []
    for step in ordered:
        if not phases or phases[-1]["phase"] != step["phase"]:
            phases.append({
                "phase": step["phase"], "label": step["phase_label"],
                "step_ids": [], "decision_count": 0,
            })
        phases[-1]["step_ids"].append(step["step_id"])
        phases[-1]["decision_count"] += 1

    return {
        "conclusion": conclusion,
        "final_decision": final_decision,
        "steps": steps,
        "phases": phases,
        "priority_review_count": sum(len(row["review_attention"]) for row in steps),
        "technical_attention_count": sum(len(row["technical_attention"]) for row in steps),
    }


def human_review_view(ledger: Any, run_id: str, analysis_id: str | None = None, *,
                      run_dir: Path | None = None,
                      langtrace_ui_base: str | None = None,
                      artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the view model. An absent selection returns alternatives, never mixed episodes."""
    chain = ledger.chain(run_id, analysis_id)
    if chain.get("status") != "OK":
        return {"schema": "acr.human_review.v1", **chain,
                "warning": "select one append-only analysis before reviewing a chain"}
    chosen = str(chain["analysis_id"])
    selected_reader = getattr(ledger, "selected_analysis", None)
    selected = selected_reader(run_id) if callable(selected_reader) else None
    is_selected = str(selected) == chosen if selected is not None else False
    artifact = artifact or ledger.load_analysis_artifact(run_id, chosen)
    if artifact.get("run_id") != run_id or artifact.get("analysis_id") != chosen:
        raise ValueError("supplied analysis artifact does not match the requested review")
    presentation = _presentation(run_dir, str(artifact.get("task_presentation_hash") or ""))
    runner_meta = None
    if run_dir is not None and (Path(run_dir) / "runner_meta.json").exists():
        runner_meta = json.loads(
            (Path(run_dir) / "runner_meta.json").read_text(encoding="utf-8"))
    langtrace_project_id = str((runner_meta or {}).get("langtrace_project_id") or "")
    raw_by_id = {str(row["episode_id"]): row for row in artifact.get("episodes") or []}
    all_cycles = artifact.get("cycles") or []
    finding_consumers = _finding_testimony_consumers(all_cycles)
    testimony_by_ref = {
        str(row.get("testimony_ref")): row for row in _testimonies(all_cycles)
        if row.get("testimony_ref")
    }
    rendered_episodes: list[dict[str, Any]] = []
    for chain_row in chain.get("episodes") or []:
        raw = raw_by_id[str(chain_row["episode_id"])]
        cycles = _cycles_for(artifact, raw)
        choice_cycle = _choice_cycle(artifact, cycles)
        testimonies = _testimonies(cycles)
        local_refs = {str(row.get("testimony_ref")) for row in testimonies}
        linked_refs = list(dict.fromkeys(
            str((action.get("args") or {}).get("decision_testimony_ref"))
            for cycle in cycles for action in cycle.get("actions") or []
            if (action.get("args") or {}).get("decision_testimony_ref")
            and str((action.get("args") or {}).get("decision_testimony_ref")) not in local_refs
        ))
        linked_testimonies = [
            {**testimony_by_ref[ref], "link_scope": "EXPLICIT_ACTION_REFERENCE"}
            for ref in linked_refs if ref in testimony_by_ref
        ]
        testimonies.extend(linked_testimonies)
        local_receipts = [row for row in testimonies
                          if row.get("link_scope") == "EPISODE_LOCAL"
                          and row.get("runtime_receipt_ref")]
        episode_testimony_refs = {
            str(row.get("testimony_ref")) for row in testimonies if row.get("testimony_ref")}
        compound = {ref: finding_consumers[ref] for ref in episode_testimony_refs
                    if len(finding_consumers.get(ref) or []) > 1}
        if compound:
            cross_episode_attention = [{
                "code": "COMPOUND_RUNTIME_TESTIMONY",
                "severity": "REVIEW",
                "title": "One runtime testimony bundled several note judgments",
                "detail": "The same testimony was used to justify multiple independently "
                          "reviewable note standings. Treat each finding as a separate audit "
                          "unit; the old testimony cannot prove each rationale independently.",
                "refs": sorted(compound),
                "finding_event_refs": sorted({event_ref for refs in compound.values()
                                              for event_ref in refs}),
                "provenance": "DETERMINISTIC_DERIVED",
                "is_error": False,
                "route": "INSTRUMENTATION_QUESTION",
            }]
        elif linked_testimonies:
            cross_episode_attention = [{
                "code": "CROSS_EPISODE_TESTIMONY",
                "severity": "REVIEW",
                "title": "Atomic decision relies on an earlier testimony",
                "detail": "This legacy finding did not carry its own complete testimony. Review "
                          "whether the referenced rationale independently supports this choice.",
                "refs": [row["testimony_ref"] for row in linked_testimonies],
                "provenance": "SERVER_FACT",
                "is_error": False,
                "route": "INSTRUMENTATION_QUESTION",
            }]
        else:
            cross_episode_attention = []
        offered_ids = {row.get("rule_id") for row in
                       ((presentation or {}).get("offered_clause_catalog") or [])}
        cited = {str(row.get("ref")) for testimony in testimonies
                 for row in testimony.get("citation_resolutions") or []}
        relevant_clauses = [row for row in
                            ((presentation or {}).get("offered_clause_catalog") or [])
                            if row.get("rule_id") in cited]
        trace_id = str(artifact.get("trace_id") or "")
        base = str(langtrace_ui_base or "").rstrip("/")
        links = [{
            "trace_id": trace_id,
            "event_ref": ref,
            "href": (
                f"{base}/project/{quote(langtrace_project_id, safe='')}/traces"
                f"#{quote(str(ref), safe='')}"
                if base and langtrace_project_id else
                f"{base}/trace/{quote(trace_id, safe='')}#{quote(str(ref), safe='')}"
                if base else f"langtrace://trace/{trace_id}#{ref}"),
        } for ref in raw.get("source_event_ids") or []]
        rendered_episodes.append({
            "episode_id": raw["episode_id"],
            "semantica_decision_id": chain_row.get("decision_id"),
            "audit_unit": raw.get("audit_unit") or "ATOMIC_DECISION",
            "bearing_cycle_id": raw.get("bearing_cycle_id") or (
                choice_cycle.get("cycle_id") if choice_cycle else None),
            "decision_function": raw.get("decision_function"),
            "decision_subject": raw.get("decision_subject"),
            "state_before": cycles[0].get("state_before") if cycles else None,
            "choice_state_before": (
                choice_cycle.get("state_before") if choice_cycle else None),
            "choice_cycle_id": choice_cycle.get("cycle_id") if choice_cycle else None,
            "choice_boundary_provenance": (
                "SERVER_SEALED_RUNTIME_RECEIPT" if local_receipts
                else "MODEL_RECONSTRUCTED_FROM_LEGACY_TRACE"),
            "runtime_receipt_status": ("SEALED" if local_receipts else
                                       "LEGACY_TRACE_DERIVED" if testimonies else "NONE"),
            "runtime_receipt_refs": [row["runtime_receipt_ref"] for row in local_receipts],
            "trigger_event_refs": cycles[0].get("trigger_event_refs") if cycles else [],
            "runtime_testimonies": testimonies,
            "grounding_assessment": _grounding_assessment(
                testimonies,
                decision_function=raw.get("decision_function"),
                decision_subject=raw.get("decision_subject")),
            "uses_model_knowledge": any(
                "own_knowledge" in (row.get("basis_sources") or [])
                for row in testimonies),
            "review_attention": [*_review_attention(testimonies), *cross_episode_attention],
            "offered_clause_count": len(offered_ids),
            "relevant_offered_clauses": relevant_clauses,
            "runtime_findings": _findings(cycles),
            "reconstruction": {
                "provenance": "MODEL_RECONSTRUCTED",
                **{key: raw.get(key) for key in (
                    "material_question", "decision_subject", "scenario", "candidate_set", "decision",
                    "decision_rationale", "model_interpretation",
                    "claimed_basis_summary", "verified_reference_summary", "state_delta",
                    "field_provenance", "source_refs_by_field")},
            },
            "actions": [action for cycle in cycles for action in cycle.get("actions") or []],
            "observations": [observation for cycle in cycles
                             for observation in cycle.get("observations") or []],
            "state_after": cycles[-1].get("state_after") if cycles else None,
            "observed_downstream_refs": raw.get("observed_downstream_refs") or [],
            "hypothesized_impact": raw.get("hypothesized_impact") or [],
            "counterfactual_supported_impact": (
                raw.get("counterfactual_supported_impact") or []),
            "attribution": None,
            "disposition": _disposition(testimonies),
            "raw_langtrace_links": links,
        })
    execution = _execution_summary(all_cycles)
    causal_edges = _hydrate_causal_edges(
        chain.get("causal_edges") or [], artifact.get("causal_assertions") or [])
    # The current append-only narrative revision is a pure function of the sealed
    # analysis artifact. This also lets projection build the view before a local
    # artifact pointer has been persisted on the graph node.
    from acr.mvp.semantica_audit import bundle_id_for

    narrative_bundle_id = bundle_id_for(artifact)
    stability_status = str(artifact.get("stability_status") or "")
    view_mode = ("SELECTED" if is_selected else
                 "STABLE_EXPLICIT" if stability_status == "STABLE_ACROSS_PASSES" else
                 "PROVISIONAL_EXPLICIT")
    return {
        "schema": "acr.human_review.v1", "status": "OK", "run_id": run_id,
        "storage_authority": "SEMANTICA_CONTEXT_GRAPH",
        "analysis_id": chosen,
        "decision_narrative_bundle_id": narrative_bundle_id,
        "analysis_artifact_hash": artifact.get("analysis_artifact_hash"),
        "selected_analysis_id": chosen if is_selected else None,
        "analysis_view_mode": view_mode,
        "trace_id": artifact.get("trace_id"),
        "task_presentation_hash": artifact.get("task_presentation_hash"),
        "taxonomy_version": artifact.get("taxonomy_version"),
        "runtime_receipt_manifest": artifact.get("runtime_receipt_manifest") or {
            "mode": "LEGACY", "sealed_receipt_count": 0},
        "decision_receipt_coverage": artifact.get("decision_receipt_coverage") or {
            "status": "NO_SEALED_DECISIONS",
            "episode_count": len(rendered_episodes), "sealed_episode_count": 0},
        "task_presentation": presentation,
        "task": _task_brief(run_dir, presentation),
        "reconstructor_identity": artifact.get("reconstructor_identity"),
        "reconstructor_attempts": artifact.get("reconstructor_attempts") or [],
        "stability_status": artifact.get("stability_status"),
        "reconstruction_stability": artifact.get("reconstruction_stability"),
        "execution": execution, "process_ladder": _process_ladder(all_cycles),
        "complete_answer": {"status": "NOT_ASSESSED",
                            "reason": "D_applicable and D_checked need attributed semantics"},
        "runner": runner_meta, "causal_edges": causal_edges,
        "review_chain": _concise_review_chain(rendered_episodes, execution, causal_edges),
        "suggested_links": chain.get("suggested_links") or [], "episodes": rendered_episodes,
    }


def render(view: dict[str, Any]) -> str:
    if view.get("status") != "OK":
        return (f"RUN {view.get('run_id')} — NO ANALYSIS SELECTION\n"
                f"available: {', '.join(view.get('available_analysis_ids') or [])}")
    lines = [f"RUN {view['run_id']} | analysis {view['analysis_id']}",
             f"stability: {view.get('stability_status')} "
             f"({view.get('reconstruction_stability')})"]
    for episode in view.get("episodes") or []:
        lines.append(f"\n[{episode['episode_id']}] {episode['decision_function']} "
                     f"— {episode['disposition']}")
        for testimony in episode.get("runtime_testimonies") or []:
            lines.append(f"  SELF_REPORTED: {testimony.get('decision')} "
                         f"[{testimony.get('rule_coverage_claim')}]")
        grounding = episode.get("grounding_assessment") or {}
        lines.append(
            "  GROUNDING: refs="
            f"{grounding.get('reference_resolution_status')} | semantics="
            f"{grounding.get('semantic_entailment_status')} | judgment="
            f"{grounding.get('judgment_mode')}")
        lines.append(f"  MODEL_RECONSTRUCTED: "
                     f"{(episode.get('reconstruction') or {}).get('model_interpretation')}")
        lines.append("  raw: " + ", ".join(
            row["event_ref"] for row in episode.get("raw_langtrace_links") or []))
    return "\n".join(lines)
