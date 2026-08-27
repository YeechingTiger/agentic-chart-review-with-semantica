"""Project one chart-review analysis into Semantica's audit provenance.

``ContextGraph.record_decision`` deliberately contains only a de-identified, comparable
decision signature.  The complete run-local Decision Episode belongs in Semantica's
``ProvenanceManager``: state before and after, the material question, actions and observations,
the basis actually available to the agent, alternatives, the choice, and the reason it continued
or stopped.  This module is only the ACR domain projection into those Semantica primitives; it is
not a second ledger or a second similarity implementation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acr.mvp.task_presentation import content_hash

AUDIT_SCHEMA = "acr.semantica_decision_audit.v1"
ATOMIC_DECISION_SCHEMA = "acr.atomic_decision.v1"
NARRATIVE_SCHEMA = "semantica.decision_narrative.v1"
# Narrative records are immutable in Semantica provenance. Bump this projection
# revision whenever the human-readable mapping changes so corrected views append a
# new bundle instead of rewriting an already-reviewable historical bundle.
NARRATIVE_PROJECTION_REVISION = "10"
PROVENANCE_FILENAME = "semantica-provenance.sqlite3"


def _audit_id(kind: str, identity: Any) -> str:
    return f"acr:{kind}:{content_hash(identity)[:24]}"


def audit_location(ledger_path: Path, artifact: dict[str, Any]) -> tuple[Path, Path]:
    """Return ``(sqlite_path, run_dir)`` without creating another storage authority."""
    artifact_ref = artifact.get("artifact_ref")
    if artifact_ref:
        artifact_parent = Path(str(artifact_ref)).resolve().parent
        run_dir = artifact_parent.parent if artifact_parent.name == "analyses" else artifact_parent
    else:
        run_dir = Path(ledger_path).resolve().parent
    return run_dir / PROVENANCE_FILENAME, run_dir


def bundle_id_for(artifact: dict[str, Any]) -> str:
    artifact_hash = artifact.get("analysis_artifact_hash") or content_hash(artifact)
    return _audit_id("decision-bundle", {
        "run_id": artifact["run_id"],
        "analysis_id": artifact["analysis_id"],
        "analysis_artifact_hash": artifact_hash,
        "narrative_projection_revision": NARRATIVE_PROJECTION_REVISION,
    })


@dataclass(frozen=True)
class AuditProjection:
    provenance_path: Path
    bundle_id: str
    entry_count: int


class _Writer:
    """Idempotent facade over Semantica's append/version-aware provenance API."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    @staticmethod
    def _fingerprint(*, source: str, metadata: dict[str, Any], kwargs: dict[str, Any]) -> str:
        material = {
            "source": source,
            "metadata": metadata,
            "typed": {key: value for key, value in kwargs.items()
                      if key not in {"activity_started_at_time", "activity_ended_at_time"}},
        }
        return content_hash(material)

    def entity(self, entity_id: str, *, source: str, metadata: dict[str, Any],
               **kwargs: Any) -> str:
        fingerprint = self._fingerprint(source=source, metadata=metadata, kwargs=kwargs)
        stored_metadata = {**metadata, "acr_record_hash": fingerprint}
        existing = self.manager.storage.retrieve(entity_id)
        if existing is not None:
            if (existing.metadata or {}).get("acr_record_hash") != fingerprint:
                raise ValueError(
                    f"immutable Semantica provenance identity changed for {entity_id!r}")
            return entity_id
        entry = self.manager.track_entity(
            entity_id=entity_id, source=source, metadata=stored_metadata, **kwargs)
        if entry is None:
            raise RuntimeError(f"Semantica failed to persist provenance entity {entity_id!r}")
        return entity_id

    def relationship(self, relationship_id: str, *, source: str,
                     metadata: dict[str, Any], **kwargs: Any) -> str:
        fingerprint = self._fingerprint(source=source, metadata=metadata, kwargs=kwargs)
        stored_metadata = {**metadata, "acr_record_hash": fingerprint}
        existing = self.manager.storage.retrieve(relationship_id)
        if existing is not None:
            if (existing.metadata or {}).get("acr_record_hash") != fingerprint:
                raise ValueError(
                    f"immutable Semantica provenance identity changed for {relationship_id!r}")
            return relationship_id
        entry = self.manager.track_relationship(
            relationship_id=relationship_id, source=source, metadata=stored_metadata, **kwargs)
        if entry is None:
            raise RuntimeError(
                f"Semantica failed to persist provenance relationship {relationship_id!r}")
        return relationship_id


def _presentation(run_dir: Path, expected_hash: str) -> tuple[dict[str, Any] | None, Path | None]:
    path = run_dir / "task_presentation.json"
    if not path.exists():
        return None, None
    from acr.mvp.task_presentation import ContractSnapshot

    snapshot = ContractSnapshot.from_path(path)
    if snapshot.presentation_hash != expected_hash:
        raise ValueError("Task Presentation does not match the analysis")
    return snapshot.to_dict(), path


def _human_note_label(value: Any) -> str:
    """Turn stable trace note ids into labels a chart reviewer can scan."""
    raw = str(value or "note").strip()
    stem, separator, date = raw.rpartition("_")
    if not separator or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        stem, date = raw, ""
    label = stem.replace("_", " ").replace("-", " ")
    for pattern, replacement in {
        r"\bOnc\b": "Oncology",
        r"\bMed\b": "",
        r"\bMD\b": "",
        r"\bOP\b": "outpatient",
    }.items():
        label = re.sub(pattern, replacement, label)
    label = " ".join(label.split()).strip() or "note"
    return f"{date} — {label}" if date else label


def _human_runtime_ref(value: Any) -> str:
    ref = str(value or "")
    if ref.startswith("search:"):
        return f"search “{ref.split(':', 1)[1]}”"
    if ref.startswith("note:"):
        return _human_note_label(ref.split(":", 1)[1])
    if ref.startswith("finding:"):
        return f"recorded finding {ref.split(':', 1)[1]}"
    if ref.startswith("decision:"):
        return f"runtime decision {ref.split(':', 1)[1]}"
    return ref


def _causal_basis(link: dict[str, Any]) -> str:
    refs = [_human_runtime_ref(value) for value in link.get("evidence_refs") or []]
    return (" Recorded dependency: " + ", ".join(refs) + ".") if refs else ""


def _finding_label(finding: dict[str, Any]) -> str:
    event_time = str(finding.get("event_time") or finding.get("value")
                     or finding.get("date") or "").strip()
    note = _human_note_label(finding.get("note_id"))
    if event_time and event_time not in note:
        note = f"{event_time} — {note}"
    standing = {
        "can_establish": "establishes the requested field",
        "merely_mentions": "does not establish the field by itself",
        "neither": "does not bear on the requested field",
    }.get(str(finding.get("standing") or ""),
          str(finding.get("standing") or "standing not recorded").replace("_", " "))
    assertion = str(finding.get("assertion_class") or "").replace("_", " ").strip()
    return f"{note}: {assertion + '; ' if assertion else ''}{standing}"


def _state_lines(state: dict[str, Any] | None) -> list[str]:
    state = state or {}
    observed = state.get("observed_state") or {}
    declared = state.get("declared_state") or {}
    surfaced = [str(value) for value in observed.get("surfaced_notes") or []]
    read = [str(value) for value in observed.get("read_notes") or []]
    findings = declared.get("findings") or []
    uncertainties = [str(value) for value in declared.get("uncertainties") or []]
    rows = [f"{len(surfaced)} notes surfaced; {len(read)} read"]
    if read:
        sample = "; ".join(_human_note_label(value) for value in read[:8])
        suffix = f" (+{len(read) - 8} more)" if len(read) > 8 else ""
        rows.append("Notes read: " + sample + suffix)
    if findings:
        rows.append(f"{len(findings)} findings already recorded")
        rows.extend(_finding_labels(state)[:6])
    rows.extend(f"Unresolved: {value}" for value in uncertainties)
    return rows


def _finding_labels(state: dict[str, Any] | None) -> list[str]:
    declared = ((state or {}).get("declared_state") or {})
    labels: list[str] = []
    for finding in declared.get("findings") or []:
        labels.append(_finding_label(finding))
    return labels


def _observation_for(event_ref: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    for episode in episodes:
        for observation in episode.get("observations") or []:
            if str(observation.get("event_ref")) == str(event_ref):
                return observation.get("result") or {}
    return {}


def _action_summary(action: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    tool = str(action.get("tool") or "action")
    args = action.get("args") or {}
    result = _observation_for(action.get("event_ref"), episodes)
    objective = str(args.get("objective") or "").strip() or None
    if tool == "list_documents":
        returned = int(result.get("returned", len(result.get("documents") or [])))
        total = result.get("total")
        observation = (f"{returned} of {total} matching metadata row(s) returned"
                       if total is not None else f"{returned} document(s) returned")
        if result.get("page_complete") is False:
            observation += f"; partial page, {result.get('unreturned')} not returned"
        name = "List available documents"
    elif tool == "search":
        hits = result.get("hits") or result.get("results") or []
        notes = {str(row.get("note_id")) for row in hits if row.get("note_id")}
        observation = f"{len(hits)} hit(s) across {len(notes)} unique note(s)"
        name = f"Search: {args.get('query') or 'query not recorded'}"
    elif tool == "read":
        observation = "Read completed" + (" (truncated)" if result.get("truncated") else "")
        name = f"Read {_human_note_label(args.get('note_id'))}"
    elif tool == "record_evidence":
        quote = " ".join(str(result.get("quote") or "").split())
        observation = "Evidence span recorded" + (f': “{quote[:320]}”' if quote else "")
        name = f"Record evidence from {_human_note_label(args.get('note_id'))}"
    elif tool == "record_finding":
        quote = " ".join(str(result.get("quote") or "").split())
        observation = _finding_label(args)
        if quote:
            observation += f' · “{quote[:320]}”'
        name = f"Judge {_human_note_label(args.get('note_id'))}"
    elif tool == "note_decision":
        observation = "Decision testimony recorded"
        name = "State decision basis"
    elif tool == "submit_answer":
        observation = "Submission accepted" if result.get("accepted") else "Submission evaluated"
        name = f"Submit {args.get('status') or 'answer'}"
    else:
        observation = "Tool result recorded"
        name = tool
    return {"name": name, "objective": objective, "observation": observation,
            "event_ref": action.get("event_ref")}


def _basis_rows(step: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies = step.get("dependencies") or {}
    resolution_by_ref = {
        str(row.get("ref")): row for row in dependencies.get("citation_resolutions") or []
        if row.get("ref") is not None
    }
    for row in dependencies.get("checked_discriminating_facts") or []:
        if row.get("ref") is not None:
            resolution_by_ref[str(row["ref"])] = row
    rows: list[dict[str, Any]] = []
    for kind in step.get("basis_sources") or []:
        status = "SELF_REPORTED"
        if kind == "chart":
            status = "OBSERVED"
        elif kind == "own_knowledge":
            status = "SELF_REPORTED_OUTSIDE_SUPPLIED_MATERIAL"
        rows.append({"kind": str(kind), "label": str(kind).replace("_", " "),
                     "status": status, "provenance": "SELF_REPORTED"})
    grounding = step.get("grounding_assessment") or {}
    if grounding:
        rows.append({
            "kind": "grounding_assessment",
            "label": "Reference check and decision authority",
            "status": grounding.get("judgment_mode"),
            "reference_status": grounding.get("reference_resolution_status"),
            "semantic_entailment_status": grounding.get("semantic_entailment_status"),
            "provenance": "MIXED_EXPLICIT_PROVENANCE",
        })
    clause_refs: set[str] = set()
    for clause in step.get("guidelines") or []:
        ref = str(clause.get("rule_id") or "")
        clause_refs.add(ref)
        resolution = resolution_by_ref.get(ref) or {}
        fact = ref.startswith("discriminating_fact.")
        rows.append({
            "kind": "discriminating_fact" if fact else "guideline", "reference": ref,
            "label": _human_clause_label(clause.get("text"), ref),
            "status": ("FACT_CHECK_RECORDED_SEMANTICS_UNREVIEWED" if fact
                       else "REFERENCE_AVAILABLE_APPLICATION_UNREVIEWED"),
            "reference_status": resolution.get("status") or "OFFERED",
            "provenance": "TASK_PRESENTATION",
        })
    for fact in dependencies.get("checked_discriminating_facts") or []:
        if str(fact.get("ref") or "") in clause_refs:
            continue
        rows.append({
            "kind": "discriminating_fact", "reference": fact.get("ref"),
            "label": _human_clause_label(None, str(fact.get("ref") or "")),
            "status": fact.get("status"),
            "provenance": "DETERMINISTIC_DERIVED",
        })
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(content_hash(row), row)
    return list(unique.values())


def _human_clause_label(value: Any, fallback: str) -> str:
    """Render contract structures for a reader while retaining the exact source elsewhere."""
    structured = value
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            structured = json.loads(value)
        except (TypeError, ValueError):
            structured = value
    if isinstance(structured, dict):
        condition = str(structured.get("if") or "").strip()
        consequence = str(structured.get("then") or "").strip()
        turns_on = [str(item).replace("_", " ")
                    for item in structured.get("turns_on") or []]
        parts = []
        if condition:
            parts.append(f"If {condition}")
        if consequence:
            parts.append(f"then {consequence}")
        if turns_on:
            parts.append("check: " + ", ".join(turns_on))
        if parts:
            return "; ".join(parts) + "."
    if value not in (None, ""):
        return str(value)
    label = fallback.rsplit(".", 1)[-1].replace("_", " ").strip()
    return f"Check whether {label}" if label else fallback


def _audit_observation_lines(step: dict[str, Any],
                             details: list[dict[str, Any]]) -> list[str]:
    """Human-scale evidence changes; raw arrays and full note ids stay behind trace links."""
    rows: list[str] = []
    coverage = step.get("coverage") or {}
    for listing in coverage.get("listings") or []:
        returned, total = listing.get("documents_returned"), listing.get("documents_total")
        line = (f"Inventory: {total} matching note(s); {returned} exact metadata row(s) "
                "returned" if total is not None else
                f"Inventory: {returned} exact metadata row(s) returned")
        if listing.get("page_complete") is False:
            line += f" — partial page ({listing.get('unreturned_count')} not returned)"
        rows.append(line)
        types = listing.get("type_summary") or []
        if types:
            top = ", ".join(
                f"{row.get('doc_type')} ({row.get('count')})" for row in types[:6])
            rows.append("Largest note-type groups: " + top
                        + (f" (+{len(types) - 6} more types)" if len(types) > 6 else ""))
    for search in coverage.get("searches") or []:
        rows.append(
            f"Search “{search.get('query')}”: {search.get('results_returned')} hit(s) across "
            f"{search.get('unique_notes_returned')} unique note(s)"
        )
    for item in step.get("evidence") or []:
        quote = " ".join(str(item.get("quote") or "").split())
        identity = _finding_label(item)
        prefix = "Finding" if item.get("kind") == "NOTE_FINDING" else "Evidence"
        rows.append(f"{prefix}: {identity}" + (f' — “{quote[:420]}”' if quote else ""))
    # Rule/reference identifiers are already rendered as labelled basis rows.  Repeating
    # their raw ids here makes the human narrative look like a provenance dump; the ids
    # remain available in the domain projection/atomic audit record and, when the
    # referenced contract clause was offered, as a labelled ``basis`` row.
    rows.extend(str(value) for value in (
        (detail.get("reconstruction") or {}).get("state_delta") for detail in details
    ) if value not in (None, ""))
    return list(dict.fromkeys(rows))


def _safe_trace_link(row: dict[str, Any]) -> dict[str, Any]:
    link = {"label": f"Langtrace {row.get('event_ref')}",
            "ref": row.get("event_ref")}
    href = str(row.get("href") or "")
    if href.startswith(("http://", "https://")):
        link["href"] = href
    return link


def _choice_cycle(artifact: dict[str, Any], cycles: list[dict[str, Any]]) \
        -> dict[str, Any] | None:
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


def _runtime_testimonies(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        results = {str(row.get("event_ref")): row.get("result") or {}
                   for row in cycle.get("observations") or []}
        for action in cycle.get("actions") or []:
            tool = action.get("tool")
            if tool not in {"note_decision", "record_finding"}:
                continue
            args = action.get("args") or {}
            result = results.get(str(action.get("event_ref")), {})
            if tool == "record_finding" and not result.get("testimony_ref"):
                continue
            reported = result.get("self_reported") or {}
            decision = (args.get("decision") if tool == "note_decision" else
                        reported.get("decision") or
                        f"{args.get('note_id')} is {args.get('standing')} for "
                        f"{args.get('field')} ({args.get('assertion_class')})")
            rows.append({
                "event_ref": action.get("event_ref"),
                "testimony_ref": result.get("testimony_ref"),
                "testimony_tool": tool,
                "facing": args.get("facing"),
                "decision": decision,
                "because": args.get("because"),
                "alternatives": args.get("alternatives") or [],
                "basis_sources": args.get("basis_sources") or [],
                "cited_refs": args.get("cited_refs") or [],
                "checked_discriminating_fact_refs": (
                    args.get("checked_discriminating_fact_refs") or []),
                "rule_coverage_claim": args.get("rule_coverage_claim"),
                "provisional_inference": args.get("provisional_inference"),
                "uncertainty": args.get("uncertainty"),
                "citation_resolutions": result.get("citation_resolutions") or [],
                "checked_fact_resolutions": result.get("checked_fact_resolutions") or [],
                "claim_provenance": "SELF_REPORTED",
                "reference_resolution_provenance": "DETERMINISTIC_DERIVED",
            })
    return rows


def _action_observation_pairs(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for cycle in cycles:
        observations = {str(row.get("event_ref")): row
                        for row in cycle.get("observations") or []}
        for action in cycle.get("actions") or []:
            event_ref = str(action.get("event_ref") or "")
            pairs.append({
                "cycle_id": cycle.get("cycle_id"),
                "event_ref": event_ref,
                "action": action,
                "action_occurrence_provenance": "SERVER_FACT",
                "action_content_provenance": (
                    "SELF_REPORTED" if action.get("tool") in {
                        "note_decision", "record_finding"}
                    else "SERVER_FACT"),
                "observation": observations.get(event_ref),
                "observation_provenance": "SERVER_FACT",
            })
    return pairs


def _transition_record(artifact: dict[str, Any], episode: dict[str, Any],
                       cycles: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = list(artifact.get("episodes") or [])
    position = next((index for index, row in enumerate(episodes)
                     if row.get("episode_id") == episode.get("episode_id")), None)
    next_episode = (episodes[position + 1] if position is not None
                    and position + 1 < len(episodes) else None)
    submissions = [action for cycle in cycles for action in cycle.get("actions") or []
                   if action.get("tool") == "submit_answer"]
    accepted = any(
        bool((observation.get("result") or {}).get("accepted"))
        for cycle in cycles for observation in cycle.get("observations") or []
        if any(str(action.get("event_ref")) == str(observation.get("event_ref"))
               and action.get("tool") == "submit_answer"
               for action in cycle.get("actions") or [])
    )
    causal = next((row for row in artifact.get("causal_assertions") or []
                   if row.get("source_episode_id") == episode.get("episode_id")
                   and next_episode is not None
                   and row.get("target_episode_id") == next_episode.get("episode_id")), None)
    if submissions:
        disposition = "STOP_AFTER_ACCEPTED_SUBMISSION" if accepted else "SUBMISSION_ATTEMPTED"
    elif next_episode is not None:
        disposition = "CONTINUE"
    else:
        disposition = "END_OF_RECONSTRUCTED_CHAIN"
    return {
        "disposition": disposition,
        "next_episode_id": next_episode.get("episode_id") if next_episode else None,
        "observed_submission_event_refs": [row.get("event_ref") for row in submissions],
        "accepted_by_runtime_gate": accepted,
        "link_to_next": ({
            "relationship_type": causal.get("relationship_type"),
            "assertion_id": causal.get("assertion_id"),
            "provenance": causal.get("provenance") or "MODEL_RECONSTRUCTED",
        } if causal else ({
            "relationship_type": None,
            "provenance": "TEMPORAL_ORDER_ONLY",
        } if next_episode else None)),
        "provenance": "DETERMINISTIC_DERIVED_FROM_OBSERVED_ORDER_AND_GATE",
        "cannot_establish": "the agent's private reason for continuing or stopping",
    }


def _atomic_decision_record(artifact: dict[str, Any], episode: dict[str, Any],
                            cycles: list[dict[str, Any]]) -> dict[str, Any]:
    """Complete observable audit packet for one material choice, without hidden CoT."""
    choice_cycle = _choice_cycle(artifact, cycles)
    testimonies = _runtime_testimonies(cycles)
    last_testimony = testimonies[-1] if testimonies else None
    before = cycles[0].get("state_before") if cycles else None
    choice_before = choice_cycle.get("state_before") if choice_cycle else before
    after = cycles[-1].get("state_after") if cycles else None
    question = ((last_testimony or {}).get("facing")
                or episode.get("material_question"))
    rationale = ((last_testimony or {}).get("because")
                 or episode.get("decision_rationale")
                 or episode.get("model_interpretation"))
    chosen = ((last_testimony or {}).get("decision") or episode.get("decision"))
    alternatives = list(dict.fromkeys([
        *(str(value) for value in episode.get("candidate_set") or []),
        *(str(value) for row in testimonies for value in row.get("alternatives") or []),
    ]))
    return {
        "schema": ATOMIC_DECISION_SCHEMA,
        "audit_unit": "ATOMIC_DECISION",
        "decision_cardinality": 1,
        "episode_id": episode.get("episode_id"),
        "decision_function": episode.get("decision_function"),
        "decision_subject": episode.get("decision_subject"),
        "time_boundary": {
            "source_cycle_ids": episode.get("source_cycle_ids") or [],
            "source_event_ids": episode.get("source_event_ids") or [],
            "source_seq_start": (cycles[0].get("source_seq_range") or [None])[0]
            if cycles else None,
            "source_seq_end": (cycles[-1].get("source_seq_range") or [None, None])[-1]
            if cycles else None,
            "choice_cycle_id": choice_cycle.get("cycle_id") if choice_cycle else None,
            "provenance": "SERVER_FACT_AND_DETERMINISTIC_BOUNDARY",
        },
        "state": {
            "episode_start": before,
            "immediately_before_choice": choice_before,
            "episode_end": after,
            "observed_state_provenance": "SERVER_FACT",
            "declared_state_provenance": "SELF_REPORTED",
        },
        "material_question": {
            "value": question,
            "provenance": "SELF_REPORTED" if last_testimony else "MODEL_RECONSTRUCTED",
            "source_refs": (([last_testimony.get("testimony_ref")]
                             if last_testimony else
                             (episode.get("source_refs_by_field") or {})
                             .get("material_question", []))),
        },
        "choice": {
            "available_options": alternatives,
            "selected": chosen,
            "runtime_reported_alternatives": [
                value for row in testimonies for value in row.get("alternatives") or []],
            "provenance": ("SELF_REPORTED_AND_MODEL_RECONSTRUCTED"
                           if last_testimony else "MODEL_RECONSTRUCTED"),
            "source_refs": (episode.get("source_refs_by_field") or {}).get("decision", []),
        },
        "rationale": {
            "value": rationale,
            "provenance": "SELF_REPORTED" if last_testimony else "MODEL_RECONSTRUCTED",
            "source_refs": (([last_testimony.get("testimony_ref")]
                             if last_testimony else
                             (episode.get("source_refs_by_field") or {})
                             .get("decision_rationale", []))),
            "not_hidden_chain_of_thought": True,
        },
        "basis": {
            "runtime_testimonies": testimonies,
            "reconstructed_claimed_basis": episode.get("claimed_basis_summary") or [],
            "verified_reference_summary": episode.get("verified_reference_summary") or [],
            "used_own_knowledge": any(
                "own_knowledge" in (row.get("basis_sources") or []) for row in testimonies),
            "provisional_inferences": [row.get("provisional_inference") for row in testimonies
                                       if row.get("provisional_inference")],
            "unresolved_uncertainties": [row.get("uncertainty") for row in testimonies
                                         if row.get("uncertainty")],
        },
        "actions_and_observations": _action_observation_pairs(cycles),
        "state_delta": {
            "value": episode.get("state_delta"),
            "provenance": (episode.get("field_provenance") or {}).get(
                "state_delta", "MODEL_RECONSTRUCTED"),
            "source_refs": (episode.get("source_refs_by_field") or {}).get("state_delta", []),
        },
        "transition": _transition_record(artifact, episode, cycles),
        "downstream": {
            "observed_refs": episode.get("observed_downstream_refs") or [],
            "hypothesized_impact": episode.get("hypothesized_impact") or [],
            "counterfactual_supported_impact": (
                episode.get("counterfactual_supported_impact") or []),
        },
        "reconstruction": {
            "stability_status": episode.get("stability_status"),
            "stability": episode.get("reconstruction_stability"),
            "field_provenance": episode.get("field_provenance") or {},
            "source_refs_by_field": episode.get("source_refs_by_field") or {},
        },
        "human_correctness": "NOT_YET_ADJUDICATED",
    }


_TITLES_BY_SUBJECT = {
    "retrieval_inventory": "Survey available records",
    "retrieval_source": "Choose note types or sources",
    "retrieval_query_batch": "Choose a search query batch",
    "retrieval_document_set": "Choose which notes to open",
    "evidence_item": "Judge one evidence item",
    "evidence_relationship": "Compare evidence candidates",
    "case_scope": "Decide case scope",
    "case_inference": "Evaluate a case inference",
    "case_absence": "Interpret what was not found",
    "case_sufficiency": "Decide whether the evidence is enough",
    "answer_selection": "Choose the answer",
    "other": "Review a material decision",
}


def _narrative_title(step: dict[str, Any], *, is_conclusion: bool) -> str:
    phase = str(step.get("phase_label") or "Decision")
    if step.get("phase") == "EVIDENCE_ASSEMBLY":
        subject = "Choose evidence to preserve"
    elif step.get("decision_function") == "where_to_look" \
            and step.get("decision_subject") == "evidence_item":
        subject = "Choose a specific note or evidence item"
    else:
        subject = _TITLES_BY_SUBJECT.get(
            str(step.get("decision_subject") or ""), "Review a material decision")
    return f"{phase} — {'Final answer' if is_conclusion else subject}"


def _run_outcome_conclusion(view: dict[str, Any]) -> dict[str, Any] | None:
    """Show the observed answer without fabricating a missing answer Decision."""
    submission = (view.get("execution") or {}).get("final_submission") or {}
    if not submission:
        return None
    return {
        "step_id": f"{view.get('analysis_id')}:run-outcome",
        "audit_unit": "RUN_OUTCOME",
        "kind": "conclusion",
        "phase": "ANSWER_FORMATION",
        "phase_label": "Final run outcome",
        "decision_function": None,
        "decision_subject": None,
        "episode_ids": [],
        "question": "What answer did the run submit?",
        "question_provenance": "RUNTIME_EXECUTION",
        "decision": submission.get("value") or submission.get("status"),
        "reason": submission.get("reasoning"),
        "reason_provenance": "RUNTIME_SUBMISSION",
        "basis_sources": [],
        "dependencies": {},
        "guidelines": [],
        "review_attention": [],
        "technical_attention": [],
        "detail_episodes": [],
        "outcome_event_ref": submission.get("event_ref"),
    }


def _narrative_payloads(view: dict[str, Any], *, bundle_id: str,
                        decision_ids: dict[str, str]) -> list[dict[str, Any]]:
    chain = view.get("review_chain") or {}
    steps = list(chain.get("steps") or [])
    conclusion = chain.get("final_decision") or _run_outcome_conclusion(view)
    ordered = [*steps, *([conclusion] if conclusion else [])]
    title = ((view.get("task") or {}).get("question")
             or "Chart review decision chain")
    payloads: list[dict[str, Any]] = []
    for order, step in enumerate(ordered):
        details = list(step.get("detail_episodes") or [])
        first = details[0] if details else {}
        last = details[-1] if details else {}
        before = (last.get("choice_state_before") or first.get("state_before") or {})
        after = last.get("state_after") or {}
        actions = [
            _action_summary(action, details)
            for detail in details for action in detail.get("actions") or []
        ]
        observations = _audit_observation_lines(step, details)
        episode_ids = [str(value) for value in step.get("episode_ids") or []]
        linked_decisions = [decision_ids[value] for value in episode_ids
                            if value in decision_ids]
        outcome_only = step.get("audit_unit") == "RUN_OUTCOME"
        if outcome_only and (episode_ids or linked_decisions):
            raise ValueError("a run outcome cannot masquerade as a Semantica decision")
        if not outcome_only and (len(episode_ids) != 1 or len(linked_decisions) != 1):
            raise ValueError(
                "each human-review narrative must map to exactly one Semantica decision")
        links = [_safe_trace_link(row) for detail in details
                 for row in detail.get("raw_langtrace_links") or []]
        is_conclusion = step.get("kind") == "conclusion"
        next_step = ordered[order + 1] if order + 1 < len(ordered) else None
        if is_conclusion:
            accepted = (((view.get("execution") or {}).get("final_submission") or {})
                        .get("accepted"))
            why_next = "Stop: the final submission was accepted by the runtime gate." \
                if accepted else "Stop: this is the recorded final submission."
        elif next_step is not None and next_step.get("audit_unit") == "RUN_OUTCOME":
            why_next = "Continue to the final submission."
        elif next_step is not None:
            relation = step.get("link_to_next") or {}
            if relation.get("kind") == "CAUSAL":
                why_next = (
                    f"Continue: an explicit {relation.get('relationship_type')} assertion "
                    "connects this step to the next “"
                    f"{next_step.get('phase_label') or 'material'}” decision."
                    f"{_causal_basis(relation)}"
                )
            elif step.get("causal_links"):
                links = step["causal_links"]
                first_link = links[0]
                extra = f" (+{len(links) - 1} more asserted link(s))" \
                    if len(links) > 1 else ""
                why_next = (
                    f"Continue: an explicit {first_link.get('relationship_type')} assertion "
                    "connects this choice to the later decision “"
                    f"{first_link.get('target_title') or first_link.get('target_step_id')}”"
                    f"{extra}.{_causal_basis(first_link)} The next card remains the observed "
                    "execution order."
                )
            else:
                why_next = (
                    "Continue: a later decision was observed in execution order; the trace "
                    "does not assert that this step caused it."
                )
        else:
            why_next = "Continue to the final submission."
        narrative_id = _audit_id("decision-narrative", {
            "bundle_id": bundle_id, "step_id": step.get("step_id"),
            "is_conclusion": is_conclusion,
        })
        payloads.append({
            "entity_id": narrative_id,
            "metadata": {
                "schema": NARRATIVE_SCHEMA, "bundle_id": bundle_id,
                "audit_unit": step.get("audit_unit") or "ATOMIC_DECISION",
                "bundle_title": title,
                "bundle_subtitle": (
                    f"analysis {view.get('analysis_id')} · run {view.get('run_id')}"),
                "order": order, "title": _narrative_title(
                    step, is_conclusion=is_conclusion),
                "phase": step.get("phase"),
                "phase_label": step.get("phase_label"),
                "decision_function": step.get("decision_function"),
                "decision_subject": step.get("decision_subject"),
                "question": step.get("question"),
                "question_provenance": step.get("question_provenance"),
                # A run outcome is an observation, not another material choice.  Giving it
                # an empty reconstructed state avoids the misleading "0 notes" card and
                # preserves the decision/outcome distinction.
                "known_before": [] if outcome_only else _state_lines(before),
                "known_before_boundary": (
                    "NOT_APPLICABLE_RUN_OUTCOME" if outcome_only
                    else "IMMEDIATELY_BEFORE_MATERIAL_CHOICE"
                ),
                "rationale": step.get("reason"),
                "rationale_provenance": step.get("reason_provenance"),
                "actions": actions,
                "observations": observations,
                "candidate_changes": {
                    "before": _finding_labels(before),
                    "after": ((step.get("dependencies") or {}).get("candidate_set")
                              or _finding_labels(after)),
                    "provenance": "OBSERVED_AND_MODEL_RECONSTRUCTED",
                },
                "basis": _basis_rows(step),
                "decision": step.get("decision"),
                "why_next_or_stop": why_next,
                "causal_links": step.get("causal_links") or [],
                "attention": [*(step.get("review_attention") or []),
                              *(step.get("technical_attention") or [])],
                "trace_links": links,
                "outcome_event_ref": step.get("outcome_event_ref"),
                "decision_ids": linked_decisions,
                "episode_ids": episode_ids,
                "is_conclusion": is_conclusion,
                "domain_projection": step,
            },
            "used_entities": linked_decisions,
        })
    return payloads


def project_analysis_audit(*, artifact: dict[str, Any], decision_ids: dict[str, str],
                           human_view: dict[str, Any], provenance_path: Path) \
        -> AuditProjection:
    """Write the complete audit projection through Semantica public APIs."""
    from semantica.provenance import ProvenanceManager

    provenance_path = Path(provenance_path)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    manager = ProvenanceManager(storage_path=str(provenance_path))
    if provenance_path.exists():
        os.chmod(provenance_path, 0o600)
    writer = _Writer(manager)
    run_id, analysis_id = str(artifact["run_id"]), str(artifact["analysis_id"])
    bundle_id = bundle_id_for(artifact)
    run_dir = audit_location(provenance_path, artifact)[1]
    presentation, presentation_path = _presentation(
        run_dir, str(artifact.get("task_presentation_hash") or ""))

    presentation_id: str | None = None
    if artifact.get("task_presentation_hash"):
        presentation_id = _audit_id(
            "task-presentation", artifact["task_presentation_hash"])
        writer.entity(
            presentation_id,
            source=str(presentation_path or f"task-presentation:{artifact['task_presentation_hash']}"),
            metadata={
                "schema": "acr.task_presentation_pointer.v1",
                "presentation_hash": artifact["task_presentation_hash"],
                "presentation": presentation,
            },
                entity_type="task_presentation", activity_id="task_presentation_capture",
                agent_id="acr", agent_type="software_agent", is_automated=True,
                role="input_recorder",
            )

    analysis_entity = _audit_id("analysis-artifact", {
        "run_id": run_id, "analysis_id": analysis_id,
        "hash": artifact["analysis_artifact_hash"],
    })
    writer.entity(
        analysis_entity,
        source=str(artifact.get("artifact_ref") or f"analysis:{artifact['analysis_artifact_hash']}"),
        metadata={
            "schema": AUDIT_SCHEMA, "run_id": run_id, "analysis_id": analysis_id,
            "analysis_artifact_hash": artifact["analysis_artifact_hash"],
            "trace_id": artifact.get("trace_id"),
            "trace_manifest_hash": artifact.get("trace_manifest_hash"),
            "trace_content_hash": artifact.get("trace_content_hash"),
            "task_presentation_hash": artifact.get("task_presentation_hash"),
            "reconstructor_identity": artifact.get("reconstructor_identity"),
            "reconstructor_call": artifact.get("reconstructor_call"),
            "reconstructor_attempts": artifact.get("reconstructor_attempts") or [],
            "stability_status": artifact.get("stability_status"),
            "reconstruction_stability": artifact.get("reconstruction_stability"),
        },
        entity_type="analysis_artifact", activity_id="decision_reconstruction",
        agent_id=str(artifact.get("reconstructor_identity") or "decision-reconstructor"),
        agent_type="software_agent", is_automated=True, role="reconstructor",
        used_entities=[presentation_id] if presentation_id else [], bundle_id=bundle_id,
    )

    cycle_entities: dict[str, str] = {}
    testimony_entities_by_cycle: dict[str, list[str]] = {}
    reference_entities_by_cycle: dict[str, list[str]] = {}
    for cycle in artifact.get("cycles") or []:
        cycle_id = str(cycle["cycle_id"])
        activity_id = _audit_id("react-activity", {
            "run_id": run_id, "analysis_id": analysis_id, "cycle_id": cycle_id})
        component_ids: list[str] = []
        for state_kind in ("state_before", "state_after"):
            state = cycle.get(state_kind) or {}
            entity_id = _audit_id("state", {
                "run_id": run_id, "analysis_id": analysis_id,
                "cycle_id": cycle_id, "state_kind": state_kind})
            writer.entity(
                entity_id, source=analysis_entity,
                metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                          "analysis_id": analysis_id, "cycle_id": cycle_id,
                          "state_kind": state_kind, "state": state,
                          "state_hash": content_hash(state),
                          "observed_provenance": "SERVER_FACT",
                          "declared_provenance": "SELF_REPORTED"},
                entity_type="state_snapshot", activity_id=activity_id,
                agent_id="acr-langtrace-replay", agent_type="software_agent",
                is_automated=True, role="state_replayer", parent_entity_id=analysis_entity,
                bundle_id=bundle_id,
            )
            component_ids.append(entity_id)

        observation_by_event = {
            str(row.get("event_ref")): row for row in cycle.get("observations") or []}
        action_ids: dict[str, str] = {}
        observation_ids: dict[str, str] = {}
        for index, action in enumerate(cycle.get("actions") or []):
            event_ref = str(action.get("event_ref") or f"action:{index}")
            action_id = _audit_id("action", {
                "run_id": run_id, "analysis_id": analysis_id,
                "cycle_id": cycle_id, "event_ref": event_ref, "index": index})
            writer.entity(
                action_id, source=analysis_entity,
                metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                          "analysis_id": analysis_id, "cycle_id": cycle_id,
                          "event_ref": event_ref, "action": action,
                          "provenance": "SERVER_FACT"},
                entity_type="agent_action", activity_id=activity_id,
                agent_id=str(artifact.get("review_agent_identity") or "chart-review-agent"),
                agent_type="software_agent", is_automated=True, role="decision_actor",
                source_location=event_ref, parent_entity_id=analysis_entity,
                bundle_id=bundle_id,
            )
            action_ids[event_ref] = action_id
            component_ids.append(action_id)
            observation = observation_by_event.get(event_ref)
            if observation is not None:
                observation_id = _audit_id("observation", {
                    "run_id": run_id, "analysis_id": analysis_id,
                    "cycle_id": cycle_id, "event_ref": event_ref, "index": index})
                writer.entity(
                    observation_id, source=analysis_entity,
                    metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                              "analysis_id": analysis_id, "cycle_id": cycle_id,
                              "event_ref": event_ref, "observation": observation,
                              "provenance": "SERVER_FACT"},
                    entity_type="tool_observation", activity_id=activity_id,
                    agent_id="acr-toolserver", agent_type="software_agent",
                    is_automated=True, role="observer", source_location=event_ref,
                    parent_entity_id=analysis_entity, used_entities=[action_id],
                    bundle_id=bundle_id,
                )
                observation_ids[event_ref] = observation_id
                component_ids.append(observation_id)

        testimony_ids: list[str] = []
        reference_ids: list[str] = []
        for index, action in enumerate(cycle.get("actions") or []):
            tool = action.get("tool")
            if tool not in {"note_decision", "record_finding"}:
                continue
            event_ref = str(action.get("event_ref") or f"action:{index}")
            result = (observation_by_event.get(event_ref) or {}).get("result") or {}
            if tool == "record_finding" and not result.get("testimony_ref"):
                continue
            args = action.get("args") or {}
            testimony_ref = str(result.get("testimony_ref")
                                or next(iter(cycle.get("decision_testimony_refs") or []),
                                        event_ref))
            resolutions = [*(result.get("citation_resolutions") or []),
                           *(result.get("checked_fact_resolutions") or [])]
            for resolution_index, resolution in enumerate(resolutions):
                ref = str(resolution.get("ref") or "")
                ref_id = _audit_id("basis-reference", {
                    "run_id": run_id, "analysis_id": analysis_id,
                    "cycle_id": cycle_id, "event_ref": event_ref,
                    "testimony_ref": testimony_ref, "resolution_index": resolution_index,
                    "ref": ref, "resolution_hash": content_hash(resolution),
                })
                writer.entity(
                    ref_id, source=analysis_entity,
                    metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                              "analysis_id": analysis_id, "cycle_id": cycle_id,
                              "event_ref": event_ref, "testimony_ref": testimony_ref,
                              "resolution_index": resolution_index,
                              "reference": resolution,
                              "provenance": "DETERMINISTIC_DERIVED"},
                    entity_type="decision_basis_reference", activity_id=activity_id,
                    agent_id="acr-reference-resolver", agent_type="software_agent",
                    is_automated=True, role="reference_resolver",
                    parent_entity_id=analysis_entity, bundle_id=bundle_id,
                )
                reference_ids.append(ref_id)
            testimony_id = _audit_id("decision-testimony", {
                "run_id": run_id, "analysis_id": analysis_id, "ref": testimony_ref})
            reported = result.get("self_reported") or {}
            decision = (args.get("decision") if tool == "note_decision" else
                        reported.get("decision") or
                        f"{args.get('note_id')} is {args.get('standing')} for "
                        f"{args.get('field')} ({args.get('assertion_class')})")
            writer.entity(
                testimony_id, source=analysis_entity,
                metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                          "analysis_id": analysis_id, "cycle_id": cycle_id,
                          "testimony_ref": testimony_ref,
                          "testimony": {
                              "tool": tool,
                              "facing": args.get("facing"), "decision": decision,
                              "because": args.get("because"),
                              "basis_sources": args.get("basis_sources") or [],
                              "cited_refs": args.get("cited_refs") or [],
                              "checked_discriminating_fact_refs": (
                                  args.get("checked_discriminating_fact_refs") or []),
                              "rule_coverage_claim": args.get("rule_coverage_claim"),
                              "provisional_inference": args.get("provisional_inference"),
                              "alternatives": args.get("alternatives") or [],
                              "uncertainty": args.get("uncertainty"),
                          },
                          "resolutions": result,
                          "provenance": "SELF_REPORTED"},
                entity_type="decision_testimony", activity_id=activity_id,
                agent_id=str(artifact.get("review_agent_identity") or "chart-review-agent"),
                agent_type="software_agent", is_automated=True,
                role="decision_testifier", source_location=event_ref,
                parent_entity_id=analysis_entity,
                used_entities=[action_ids[event_ref], *reference_ids],
                bundle_id=bundle_id,
            )
            testimony_ids.append(testimony_id)
            component_ids.append(testimony_id)
        testimony_entities_by_cycle[cycle_id] = list(dict.fromkeys(testimony_ids))
        reference_entities_by_cycle[cycle_id] = list(dict.fromkeys(reference_ids))

        cycle_entity = _audit_id("react-cycle", {
            "run_id": run_id, "analysis_id": analysis_id, "cycle_id": cycle_id})
        writer.entity(
            cycle_entity, source=analysis_entity,
            metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                      "analysis_id": analysis_id, "cycle_id": cycle_id,
                      "cycle": cycle, "provenance": "DETERMINISTIC_DERIVED"},
            entity_type="react_cycle", activity_id=activity_id,
            agent_id="acr-langtrace-replay", agent_type="software_agent",
            is_automated=True, role="cycle_replayer", parent_entity_id=analysis_entity,
            used_entities=list(dict.fromkeys(component_ids)), bundle_id=bundle_id,
            activity_started_at_time=cycle.get("source_event_time"),
            activity_ended_at_time=cycle.get("source_event_time"),
        )
        cycle_entities[cycle_id] = cycle_entity

    episode_by_id = {str(row["episode_id"]): row for row in artifact.get("episodes") or []}
    for episode_id, episode in episode_by_id.items():
        decision_id = decision_ids[episode_id]
        source_cycles = [str(value) for value in episode.get("source_cycle_ids") or []]
        cycle_rows = [row for row in artifact.get("cycles") or []
                      if str(row.get("cycle_id")) in source_cycles]
        testimonies = [entity for cycle_id in source_cycles
                       for entity in testimony_entities_by_cycle.get(cycle_id, [])]
        references = [entity for cycle_id in source_cycles
                      for entity in reference_entities_by_cycle.get(cycle_id, [])]
        has_testimony = bool(testimonies)
        atomic_record = _atomic_decision_record(artifact, episode, cycle_rows)
        writer.entity(
            decision_id, source=analysis_entity,
            metadata={
                "schema": AUDIT_SCHEMA, "run_id": run_id, "analysis_id": analysis_id,
                "acr_episode_id": episode_id, "episode": episode,
                "atomic_decision": atomic_record,
                "component_map": {
                    "time_boundary": "SERVER_FACT_AND_DETERMINISTIC_BOUNDARY",
                    "state_before": "OBSERVED_AND_DECLARED",
                    "choice_state_before": "OBSERVED_AND_DECLARED",
                    "material_question": ("SELF_REPORTED" if has_testimony
                                          else "MODEL_RECONSTRUCTED"),
                    "available_basis": "OBSERVED_AND_SELF_REPORTED",
                    "alternatives_and_choice": (
                        "SELF_REPORTED_AND_MODEL_RECONSTRUCTED" if has_testimony
                        else "MODEL_RECONSTRUCTED"),
                    "decision_rationale": ("SELF_REPORTED" if has_testimony
                                           else "MODEL_RECONSTRUCTED"),
                    "actions_and_observations": "SERVER_FACT",
                    "state_after": "OBSERVED_AND_DECLARED",
                    "continue_or_stop": "DETERMINISTIC_DERIVED",
                    "human_correctness": "NOT_YET_ADJUDICATED",
                },
                "state_before": cycle_rows[0].get("state_before") if cycle_rows else None,
                "material_question": atomic_record["material_question"],
                "state_after": cycle_rows[-1].get("state_after") if cycle_rows else None,
                "field_provenance": episode.get("field_provenance") or {},
                "source_refs_by_field": episode.get("source_refs_by_field") or {},
                "reconstruction_stability": episode.get("reconstruction_stability"),
                "confidence_semantics": "RECONSTRUCTION_STABILITY",
            },
            entity_type="decision", activity_id=_audit_id(
                "reconstruction-activity", {"run_id": run_id, "analysis_id": analysis_id}),
            agent_id=str(artifact.get("reconstructor_identity") or "decision-reconstructor"),
            agent_type="software_agent", is_automated=True, role="decision_reconstructor",
            parent_entity_id=analysis_entity,
            used_entities=list(dict.fromkeys([
                *(cycle_entities[value] for value in source_cycles if value in cycle_entities),
                *testimonies, *references,
            ])), confidence=float(episode.get("reconstruction_stability")
                                  or artifact.get("reconstruction_stability") or 0.0),
            bundle_id=bundle_id,
        )

    for assertion in artifact.get("causal_assertions") or []:
        relationship_id = _audit_id("causal-relationship", {
            "run_id": run_id, "analysis_id": analysis_id,
            "assertion_id": assertion.get("assertion_id")})
        writer.relationship(
            relationship_id, source=analysis_entity,
            metadata={"schema": AUDIT_SCHEMA, "run_id": run_id,
                      "analysis_id": analysis_id, "relationship_domain": "causal",
                      "assertion": assertion,
                      "source_decision_id": decision_ids[str(assertion["source_episode_id"])],
                      "target_decision_id": decision_ids[str(assertion["target_episode_id"])]},
            activity_id="causal_assertion_projection", agent_id="decision-reconstructor",
            agent_type="software_agent", is_automated=True, role="causal_assertion_recorder",
            bundle_id=bundle_id,
        )

    _write_decision_narratives(
        writer=writer, artifact=artifact, human_view=human_view,
        decision_ids=decision_ids, analysis_entity=analysis_entity,
        bundle_id=bundle_id)

    check = manager.check(strict=True)
    chain = manager.verify_chain()
    if not check.get("valid") or not chain.get("valid"):
        raise RuntimeError(
            f"Semantica provenance integrity failed: references={check}; chain={chain}")
    os.chmod(provenance_path, 0o600)
    return AuditProjection(
        provenance_path=provenance_path, bundle_id=bundle_id,
        entry_count=len(manager.audit_log(format="json")),
    )


def _write_decision_narratives(*, writer: _Writer, artifact: dict[str, Any],
                               human_view: dict[str, Any], decision_ids: dict[str, str],
                               analysis_entity: str, bundle_id: str) -> None:
    """Append one revisioned human-readable bundle over existing Semantica Decisions."""
    run_id, analysis_id = str(artifact["run_id"]), str(artifact["analysis_id"])
    writer.entity(
        bundle_id, source=analysis_entity,
        metadata={"schema": "semantica.decision_narrative_bundle.v1",
                  "bundle_id": bundle_id, "run_id": run_id, "analysis_id": analysis_id,
                  "analysis_artifact_hash": artifact["analysis_artifact_hash"]},
        entity_type="decision_narrative_bundle", activity_id="domain_projection",
        agent_id="acr", agent_type="software_agent", is_automated=True,
        role="domain_projector", parent_entity_id=analysis_entity,
        used_entities=list(decision_ids.values()), bundle_id=bundle_id,
    )
    for narrative in _narrative_payloads(
            human_view, bundle_id=bundle_id, decision_ids=decision_ids):
        writer.entity(
            narrative["entity_id"], source=analysis_entity,
            metadata=narrative["metadata"], entity_type="decision_narrative",
            activity_id="human_readable_domain_projection", agent_id="acr",
            agent_type="software_agent", is_automated=True, role="domain_projector",
            parent_entity_id=bundle_id, used_entities=narrative["used_entities"],
            bundle_id=bundle_id,
        )


def project_decision_narratives(*, artifact: dict[str, Any],
                                decision_ids: dict[str, str],
                                human_view: dict[str, Any],
                                provenance_path: Path) -> AuditProjection:
    """Append only the current narrative revision; never rewrite the audit substrate."""
    from semantica.provenance import ProvenanceManager

    provenance_path = Path(provenance_path)
    if not provenance_path.is_file():
        raise ValueError(f"Semantica provenance store is missing: {provenance_path}")
    manager = ProvenanceManager(storage_path=str(provenance_path))
    writer = _Writer(manager)
    run_id, analysis_id = str(artifact["run_id"]), str(artifact["analysis_id"])
    analysis_entity = _audit_id("analysis-artifact", {
        "run_id": run_id, "analysis_id": analysis_id,
        "hash": artifact["analysis_artifact_hash"],
    })
    if manager.storage.retrieve(analysis_entity) is None:
        raise ValueError("cannot project narratives before the analysis provenance exists")
    missing_decisions = [value for value in decision_ids.values()
                         if manager.storage.retrieve(value) is None]
    if missing_decisions:
        raise ValueError(
            f"cannot project narratives before Semantica Decisions exist: {missing_decisions}")
    bundle_id = bundle_id_for(artifact)
    _write_decision_narratives(
        writer=writer, artifact=artifact, human_view=human_view,
        decision_ids=decision_ids, analysis_entity=analysis_entity,
        bundle_id=bundle_id)
    check = manager.check(strict=True)
    chain = manager.verify_chain()
    if not check.get("valid") or not chain.get("valid"):
        raise RuntimeError(
            f"Semantica provenance integrity failed: references={check}; chain={chain}")
    os.chmod(provenance_path, 0o600)
    return AuditProjection(
        provenance_path=provenance_path, bundle_id=bundle_id,
        entry_count=len(manager.audit_log(format="json")),
    )
