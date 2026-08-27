"""Reconstruct Decision Episodes without allowing the reader to rewrite the run.

The canonical input is a complete Langtrace export, including taxonomy-neutral Runtime Decision
Receipts when the reviewed run emitted them. Deterministic code first replays it into fixed ReAct
cycles. Luna assigns the evolving post-run decision taxonomy and interprets cycles without
rewriting receipt-backed testimony. A verifier owns receipt binding, one-decision cardinality,
cycle coverage, contiguity, pre-state boundaries, source references and causal assertions.

Runtime Decision Testimony remains SELF_REPORTED.  Luna's interpretation remains
MODEL_RECONSTRUCTED.  They are shown together, never collapsed into a fictional chain of thought.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from acr.mvp.decision_receipts import (
    LEGACY_RECEIPT_PROVENANCE,
    RUNTIME_DECISION_RECEIPT_SCHEMA,
    receipt_from_action,
)
from acr.mvp.decision_types import (
    DECISION_SUBJECTS,
    DECISION_TAXONOMY_SCHEMA,
    DECISION_TYPES,
    subjects_for,
)
from acr.mvp.langtrace_io import LangtraceReviewTrace
from acr.mvp.task_presentation import content_hash
from acr.mvp.timeline import build_react_cycles, build_trace_completeness

SERVER_FACT = "SERVER_FACT"
SELF_REPORTED = "SELF_REPORTED"
DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
MODEL_RECONSTRUCTED = "MODEL_RECONSTRUCTED"
HUMAN_ADJUDICATED = "HUMAN_ADJUDICATED"

ROLES = ("DECISION_BEARING", "DECISION_SUPPORT", "MECHANICAL")
CAUSAL_TYPES = ("CAUSED", "INFLUENCED", "PRECEDENT_FOR")
ANALYSIS_SCHEMA = "acr.decision_episode_analysis.v3"
VERIFIER_VERSION = "verified-policy-reference-inventory.v6"
MODEL_INPUT_PROJECTION_VERSION = "decision-relevant-cycle-view.v1"
EPISODE_FIELDS = (
    "material_question", "decision_subject", "scenario", "candidate_set", "decision",
    "decision_rationale",
    "model_interpretation",
    "claimed_basis_summary", "verified_reference_summary", "state_delta",
    "observed_downstream_refs", "hypothesized_impact",
    "counterfactual_supported_impact",
)


class ReconstructionError(ValueError):
    pass


class StructuredLLM(Protocol):
    def generate_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]: ...


def _call_record(llm: StructuredLLM, prior_count: int, *,
                 reconstructor_identity: str) -> dict[str, Any]:
    records = getattr(llm, "call_records", None)
    if isinstance(records, list) and len(records) == prior_count + 1 \
            and isinstance(records[-1], dict):
        return dict(records[-1])
    return {
        "call_index": prior_count + 1,
        "requested_model": reconstructor_identity,
        "requested_provider": (reconstructor_identity.split("/", 1)[0]
                               if "/" in reconstructor_identity else None),
        "resolved_model": None, "response_provider": None, "response_id": None,
        "created": None, "usage": None, "identity_status": "UNAVAILABLE_FROM_ADAPTER",
    }


def _source_map_schema(source_refs: list[str]) -> dict[str, Any]:
    # The verifier below is the source-of-truth reference allowlist. Repeating the same enum for
    # every field makes the provider schema larger than the actual decision projection and has
    # caused otherwise valid structured responses to be truncated in transit.
    properties = {field: {"type": "array", "items": {"type": "string"}}
                  for field in EPISODE_FIELDS}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _episode_schema(cycle_ids: list[str], source_refs: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "source_cycle_ids": {"type": "array", "items": {"type": "string", "enum": cycle_ids},
                             "minItems": 1},
        "decision_function": {"type": "string", "enum": list(DECISION_TYPES)},
        "material_question": {"type": "string"},
        "decision_subject": {"type": "string", "enum": list(DECISION_SUBJECTS)},
        "scenario": {"type": "string"},
        "candidate_set": {"type": "array", "items": {"type": "string"}},
        "decision": {"type": "string"},
        "decision_rationale": {"type": "string"},
        "model_interpretation": {"type": "string"},
        "claimed_basis_summary": {"type": "array", "items": {"type": "string"}},
        "verified_reference_summary": {"type": "array", "items": {"type": "string"}},
        "state_delta": {"type": "string"},
        "observed_downstream_refs": {"type": "array", "items": {"type": "string"}},
        "hypothesized_impact": {"type": "array", "items": {"type": "string"}},
        "counterfactual_supported_impact": {
            "type": "array", "items": {"type": "string"}},
        "source_refs_by_field": _source_map_schema(source_refs),
    }
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def extraction_response_format(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_ids = [cycle["cycle_id"] for cycle in cycles]
    source_refs = sorted(_inventory(cycles))
    annotation = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": list(ROLES)},
            "decision_function": {
                "anyOf": [{"type": "string", "enum": list(DECISION_TYPES)}, {"type": "null"}]},
        },
        "required": ["role", "decision_function"],
        "additionalProperties": False,
    }
    annotations = {
        "type": "object",
        "properties": {cycle_id: annotation for cycle_id in cycle_ids},
        "required": cycle_ids,
        "additionalProperties": False,
    }
    causal = {
        "type": "object",
        "properties": {
            "source_episode_index": {"type": "integer", "minimum": 0},
            "target_episode_index": {"type": "integer", "minimum": 0},
            "relationship_type": {"type": "string", "enum": list(CAUSAL_TYPES)},
            "evidence_refs": {"type": "array", "items": {"type": "string"},
                              "minItems": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["source_episode_index", "target_episode_index", "relationship_type",
                     "evidence_refs", "reasoning"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "cycle_annotations": annotations,
            "episodes": {"type": "array", "items": _episode_schema(
                cycle_ids, source_refs)},
            "mechanical_cycle_ids": {
                "type": "array", "items": {"type": "string", "enum": cycle_ids}},
            "causal_links": {"type": "array", "items": causal},
        },
        "required": ["cycle_annotations", "episodes", "mechanical_cycle_ids", "causal_links"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {
        "name": "acr_decision_episodes", "strict": True, "schema": schema}}


def _taxonomy_menu() -> str:
    return "\n".join(f"  - {name}: {definition.about}"
                     for name, definition in DECISION_TYPES.items())


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Keep decision state while replacing accumulated raw-observation copies with counts.

    The immutable full cycles remain in the append-only analysis artifact and verifier. This is
    only the model-facing projection: repeating hundreds of surfaced note ids and every prior
    citation in every cycle adds no new decision signal and can exhaust structured-output limits.
    """
    declared = dict((state or {}).get("declared_state") or {})
    observed = dict((state or {}).get("observed_state") or {})
    compact_observed = {
        key: observed.get(key)
        for key in (
            "evidence_refs", "finding_call_refs", "gates", "read_notes", "result_status",
            "search_refs", "submissions",
        )
        if key in observed
    }
    compact_observed["surfaced_note_count"] = len(observed.get("surfaced_notes") or [])
    compact_observed["citation_resolution_count"] = len(
        observed.get("citation_resolutions") or [])
    return {"declared_state": declared, "observed_state": compact_observed}


def _compact_result(tool: str | None, result: dict[str, Any]) -> dict[str, Any]:
    """Summarize high-cardinality transport payloads without rewriting decision evidence."""
    compact = dict(result or {})
    documents = compact.pop("documents", None)
    if isinstance(documents, list):
        dates = [str(row.get("date")) for row in documents
                 if isinstance(row, dict) and row.get("date")]
        compact["documents_summary"] = {
            "count": len(documents),
            "first_date": min(dates) if dates else None,
            "last_date": max(dates) if dates else None,
        }
    receipt = compact.pop("decision_receipt", None)
    if isinstance(receipt, dict):
        server_facts = receipt.get("server_facts") or {}
        compact["decision_receipt_summary"] = {
            "schema": receipt.get("schema"),
            "receipt_id": receipt.get("receipt_id"),
            "receipt_hash": receipt.get("receipt_hash"),
            "testimony_ref": receipt.get("testimony_ref"),
            "verified_input_refs": server_facts.get("verified_input_refs") or [],
        }
    # The full note text is material to reconstructing standing. Other tool results remain small
    # after inventory and receipt compaction, so retain them verbatim.
    return compact


def model_input_projection(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the reproducible, decision-relevant view sent to the reconstruction model."""
    projected: list[dict[str, Any]] = []
    for cycle in cycles:
        actions = [dict(action) for action in cycle.get("actions") or []]
        tool_by_ref = {str(action.get("event_ref")): str(action.get("tool") or "")
                       for action in actions}
        observations = []
        for observation in cycle.get("observations") or []:
            event_ref = str(observation.get("event_ref") or "")
            observations.append({
                **{key: value for key, value in observation.items() if key != "result"},
                "result": _compact_result(
                    tool_by_ref.get(event_ref), observation.get("result") or {}),
            })
        projected.append({
            key: cycle.get(key)
            for key in (
                "cycle_id", "source_event_ids", "source_seq_range", "source_event_time",
                "structural_kind", "trigger_event_refs", "declared_open_question",
                "decision_testimony_refs", "decision_receipt_refs", "has_decision_testimony",
                "has_decision_receipt",
            )
        } | {
            "state_before": _compact_state(cycle.get("state_before") or {}),
            "actions": actions,
            "observations": observations,
            "state_after": _compact_state(cycle.get("state_after") or {}),
        })
    return {
        "projection_version": MODEL_INPUT_PROJECTION_VERSION,
        "available_source_refs": sorted(_inventory(cycles)),
        "cycles": projected,
    }


def build_prompt(cycles: list[dict[str, Any]]) -> str:
    cycle_ids = [cycle["cycle_id"] for cycle in cycles]
    return f"""You are reconstructing one completed chart-review run for audit. You are not
reviewing the patient, judging clinical correctness, or inventing private chain-of-thought.

Below is a fixed ReAct Cycle skeleton deterministically replayed from a complete Langtrace
export. You may interpret it, but you MUST NOT add, remove, duplicate, reorder, or move a cycle.
Every cycle must occur exactly once: in one Decision Episode or in mechanical_cycle_ids.

For each cycle, choose a role:
  DECISION_BEARING — contains a substantive choice
  DECISION_SUPPORT — action/observation serving an adjacent substantive choice
  MECHANICAL — transport or termination with no substantive review choice

ROLE/FUNCTION COHERENCE IS A HARD WIRE CONSTRAINT, NOT A LABELING PREFERENCE:
  * DECISION_BEARING -> decision_function MUST be exactly one function from the menu below.
  * DECISION_SUPPORT -> decision_function MUST be JSON null.
  * MECHANICAL -> decision_function MUST be JSON null.
The annotation is about that individual cycle.
A support or mechanical cycle does not inherit the enclosing episode's function. For example,
a list_documents cycle can support a
where_to_look episode while its own decision_function remains null.

An Atomic Decision is one material commitment among meaningful alternatives that a reviewer can
judge with one verdict.  If a reviewer could say "the first part was right but the second part was
wrong", those are two decisions.  A Decision Episode is the audit envelope for exactly one Atomic
Decision: its first cycle is exactly one DECISION_BEARING cycle, followed by zero or more
DECISION_SUPPORT cycles that execute or observe that already-made commitment.  A new material
question, a changed commitment, or replanning after a new observation starts a new episode.

Do not equate tool calls with decisions:
  * Choosing a note-type/source scope is one decision when it can change what evidence is seen.
  * A precommitted keyword/query batch is one decision; executing every query in that batch is
    support. A new query chosen after observing results is a new decision.
  * Selectively choosing a note/set to open is one decision. Opening every note in an already
    committed read-all set is support.
  * An offset>0 list_documents call that merely finishes an already-declared inventory is
    MECHANICAL pagination. It is not part of an intervening keyword-search episode.
  * Except for that pagination continuation, an observable switch between inventory, search,
    and read starts a new where_to_look decision. In particular, the first read after search or
    inventory is the auditable choice of which note/set to open, even without runtime testimony.
  * A mandatory operation with no meaningful alternative is MECHANICAL, not a fabricated choice.
  * Every successful record_finding call commits the Standing of one note+Field. Its own cycle is
    DECISION_BEARING with decision_function=standing and begins a separate evidence_item episode.
    On current traces its own args/result are the contemporaneous Decision Testimony. A legacy
    prior/shared testimony is only supporting context and cannot turn independently reviewable
    findings into support cycles or make a compound choice atomic.
  * Standing for each note+Field, sufficiency/stop, and conflict resolution are separate decisions
    when each can be independently right or wrong. Conflict resolution occurs after the individual
    findings and should cite finding:N dependencies when the runtime recorded them.

A cycle with has_decision_receipt=true is a witnessed runtime commitment and MUST be
DECISION_BEARING. The sealed receipt fixes that decision boundary and canonical testimony fields;
it does not choose the decision_function, decision_subject, semantic correctness, or whether the
agent improperly bundled several commitments. Legacy note_decision calls without a sealed receipt
remain testimony rather than automatic episodes. Never merge independent choices merely because
they are sequential or share the same decision function.

Decision functions (one per episode):
{_taxonomy_menu()}

SOURCE AND TIME RULES:
  * material_question names the one unresolved question this episode answers. Prefer the
    contemporaneous Decision Testimony's facing value when present; otherwise reconstruct it
    from cited cycle/state refs and keep it distinct from scenario.
  * decision_subject identifies what this atomic choice acted on. For where_to_look, distinguish
    note inventory, source/note type, one precommitted query batch, and a selected document set.
    `evidence_item` is also coherent when the choice is to retrieve/process one specific note;
    decision_function still distinguishes that retrieval choice from judging the note's standing.
    `other` is the subject escape valve. If decision_function is `other`, retain any known
    controlled subject instead of discarding it merely because the function vocabulary is open.
    For functions with exactly one controlled subject (for example standing -> evidence_item and
    which_wins -> evidence_relationship), the verifier derives that subject from the function;
    only where_to_look and the `other` escape valve require a genuinely semantic subject choice.
  * scenario describes only the state immediately before the material choice. Because the first
    episode cycle bears that choice, its source list must include state_before:<first-cycle-id>;
    never cite an observation from that episode.
  * decision_rationale is a concise audit explanation supported by its own source refs. It is
    never hidden chain-of-thought and must not be presented as SELF_REPORTED unless it cites a
    matching runtime Decision Testimony.
  * source refs may be cycle:<cycle-id>, state_before:<cycle-id>, state_after:<cycle-id>,
    layer1:<seq>, or a runtime testimony ref such as decision:<seq>.
  * observed_downstream_refs must be real later layer1:<seq> refs.
    Its source_refs_by_field entry is a redundant index and is deterministically canonicalized
    to the subset that really occurs after the episode. Invalid current/prior refs are discarded
    and audited rather than forcing a full semantic re-extraction; do not invent different refs.
  * hypothesized_impact is explicitly only a hypothesis.
  * counterfactual_supported_impact must be empty unless the trace contains a real
    counterfactual result; ordinary temporal succession is not counterfactual evidence.
  * causal_links are optional. Emit one only when evidence_refs support CAUSED, INFLUENCED, or
    PRECEDENT_FOR. Sequential adjacency alone is never causal evidence.
  * source_refs_by_field must contain every field, even when its list is empty.

Before returning JSON, audit every cycle annotation: if role is not DECISION_BEARING, replace
its decision_function with null. This consistency check is mandatory. `source_cycle_ids` and
`mechanical_cycle_ids` are redundant transport fields: identify each episode by putting its own
DECISION_BEARING cycle first. The verifier deterministically attaches the immediately following
DECISION_SUPPORT cycles and derives the mechanical set from cycle_annotations, so do not try to
move a support cycle across another bearing or mechanical cycle.

Cycle ids, in immutable order:
{json.dumps(cycle_ids, ensure_ascii=False, indent=2)}

DECISION-RELEVANT FIXED CYCLE PROJECTION:
The full immutable cycles remain verifier input and are stored in the analysis artifact. This
reproducible projection preserves exact actions, read text, testimony, reference resolutions,
state commitments, counts, and stable refs. It summarizes repeated inventory rows and accumulated
state copies only; use available_source_refs exactly and never infer omitted note contents.
{json.dumps(model_input_projection(cycles), ensure_ascii=False, indent=2)}
"""


def _inventory(cycles: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for cycle in cycles:
        cid = str(cycle["cycle_id"])
        refs |= {f"cycle:{cid}", f"state_before:{cid}", f"state_after:{cid}"}
        refs.update(map(str, cycle.get("source_event_ids") or []))
        refs.update(map(str, cycle.get("decision_testimony_refs") or []))
        for observation in cycle.get("observations") or []:
            result = observation.get("result") or {}
            if result.get("finding_ref"):
                refs.add(str(result["finding_ref"]))
            # Guideline, policy, instruction, evidence and discriminating-fact locators are
            # first-class sources once the runtime server resolved them. Excluding them made a
            # correct grounded reconstruction impossible to express.
            refs.update(_verified_runtime_refs(result))
        refs.update(_produced_runtime_refs(cycle))
    return refs


def _verified_runtime_refs(result: dict[str, Any]) -> list[str]:
    """Return refs whose resolution status was established by the runtime server."""
    refs: list[str] = []
    for row in [*(result.get("citation_resolutions") or []),
                *(result.get("checked_fact_resolutions") or [])]:
        if not isinstance(row, dict) or not row.get("ref"):
            continue
        if row.get("verified") is True or row.get("status") == "CLAIMED_AND_VERIFIED":
            refs.append(str(row["ref"]))
    return list(dict.fromkeys(refs))


def _runtime_receipt(cycle: dict[str, Any]) -> dict[str, Any] | None:
    """Return the one validated sealed receipt owned by this fixed tool cycle."""
    results = {str(row.get("event_ref")): row.get("result") or {}
               for row in cycle.get("observations") or []}
    receipts = [receipt for action in cycle.get("actions") or []
                for receipt in [receipt_from_action(
                    action, results.get(str(action.get("event_ref")), {}))]
                if receipt is not None]
    if len(receipts) > 1:
        raise ReconstructionError("one ReAct cycle contains multiple runtime Decision Receipts")
    return receipts[0] if receipts else None


def _runtime_receipt_manifest(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    sealed = [receipt for cycle in cycles for receipt in [_runtime_receipt(cycle)]
              if receipt is not None]
    witnessed = sum(bool(cycle.get("decision_testimony_refs")) for cycle in cycles)
    mode = ("NONE" if witnessed == 0 else
            "SEALED" if sealed and len(sealed) == witnessed else
            "MIXED" if sealed else "LEGACY")
    return {
        "mode": mode,
        "sealed_receipt_count": len(sealed),
        "witnessed_decision_count": witnessed,
        "receipt_schemas": sorted({str(row["schema"]) for row in sealed}),
        "receipt_hashes": [str(row["receipt_hash"]) for row in sealed],
        "provenance": DETERMINISTIC_DERIVED,
    }


def _decision_receipt_coverage(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure sealed runtime provenance over the post-run Decision projection."""
    sealed = [str(row["episode_id"]) for row in episodes if row.get("runtime_receipt_ref")]
    legacy = [str(row["episode_id"]) for row in episodes
              if row.get("runtime_testimony_ref") and not row.get("runtime_receipt_ref")]
    reconstructed = [str(row["episode_id"]) for row in episodes
                     if not row.get("runtime_testimony_ref")]
    total = len(episodes)
    status = ("NO_DECISIONS" if total == 0 else
              "FULLY_SEALED" if len(sealed) == total else
              "PARTIALLY_SEALED" if sealed else "NO_SEALED_DECISIONS")
    return {
        "status": status,
        "episode_count": total,
        "sealed_episode_count": len(sealed),
        "legacy_testimony_episode_count": len(legacy),
        "reconstructed_without_testimony_count": len(reconstructed),
        "sealed_episode_ids": sealed,
        "unsealed_episode_ids": [*legacy, *reconstructed],
        "provenance": DETERMINISTIC_DERIVED,
    }


def _runtime_testimony(cycle: dict[str, Any]) -> dict[str, Any] | None:
    """Recover fields the acting agent explicitly recorded; a reader may not rewrite them.

    The reconstructor still supplies pre-state scenario, interpretation and downstream impact.
    But ``facing``, the chosen outcome, ``because`` and the claimed bases were already captured
    contemporaneously. Treating Luna's paraphrase as a second source of truth creates artificial
    drift and, worse, can misrepresent what the reviewed agent actually committed to.
    """
    sealed_receipt = _runtime_receipt(cycle)
    results = {str(row.get("event_ref")): row.get("result") or {}
               for row in cycle.get("observations") or []}
    for action in cycle.get("actions") or []:
        tool = action.get("tool")
        if tool not in {"note_decision", "record_finding"} or action.get("ok") is not True:
            continue
        args = action.get("args") or {}
        result = results.get(str(action.get("event_ref")), {})
        receipt = sealed_receipt if sealed_receipt is not None \
            and sealed_receipt.get("source_event_ref") == action.get("event_ref") else None
        if receipt is not None:
            testimony = receipt["testimony"]
            server_facts = receipt["server_facts"]
            decision = str(testimony.get("selected") or "")
            alternatives = [str(value) for value in testimony.get("alternatives") or []
                            if str(value)]
            candidates = list(dict.fromkeys(
                [value for value in [decision, *alternatives] if value]))
            return {
                "testimony_ref": str(receipt["testimony_ref"]),
                "receipt_ref": str(receipt["receipt_id"]),
                "receipt_schema": str(receipt["schema"]),
                "receipt_hash": str(receipt["receipt_hash"]),
                "receipt_provenance": str(receipt["provenance"]),
                "tool": str(tool),
                "material_question": str(testimony.get("question") or ""),
                "candidate_set": candidates,
                "decision": decision,
                "decision_rationale": str(testimony.get("rationale") or ""),
                "claimed_basis_summary": [str(value) for value in
                                          testimony.get("basis_sources") or []],
                "verified_reference_summary": [str(value) for value in
                                                server_facts.get("verified_input_refs") or []],
            }
        testimony_ref = result.get("testimony_ref")
        if not testimony_ref:
            testimony_ref = next(iter(cycle.get("decision_testimony_refs") or []), None)
        if not testimony_ref:
            continue

        if tool == "record_finding":
            decision = str(args.get("standing") or "")
        else:
            decision = str(args.get("decision") or "")
        alternatives = [str(value) for value in (args.get("alternatives") or [])
                        if str(value)]
        candidates = list(dict.fromkeys([value for value in [decision, *alternatives] if value]))
        return {
            "testimony_ref": str(testimony_ref),
            "receipt_ref": None,
            "receipt_schema": None,
            "receipt_hash": None,
            "receipt_provenance": LEGACY_RECEIPT_PROVENANCE,
            "tool": str(tool),
            "material_question": str(args.get("facing") or ""),
            "candidate_set": candidates,
            "decision": decision,
            "decision_rationale": str(args.get("because") or ""),
            "claimed_basis_summary": [str(value) for value in
                                      (args.get("basis_sources") or [])],
            "verified_reference_summary": _verified_runtime_refs(result),
        }
    return None


def _runtime_decisions(cycles: list[dict[str, Any]]) -> dict[str, str]:
    """Compatibility index used by callers that only need the witnessed outcome text."""
    out: dict[str, str] = {}
    for cycle in cycles:
        testimony = _runtime_testimony(cycle)
        if testimony:
            out[testimony["testimony_ref"]] = testimony["decision"]
    return out


def _produced_runtime_refs(cycle: dict[str, Any]) -> set[str]:
    """Return references whose successful production is a server-observed fact.

    Search and note references are the same stable locators accepted by runtime citation
    resolution. Findings and Decision Testimonies use the opaque references returned by their
    tools. Merely mentioning a locator in testimony never makes the enclosing episode its
    producer; the successful server result does.
    """
    results = {str(row.get("event_ref")): row.get("result") or {}
               for row in cycle.get("observations") or []}
    refs: set[str] = set()
    for action in cycle.get("actions") or []:
        if action.get("ok") is not True:
            continue
        event_ref = str(action.get("event_ref") or "")
        result = results.get(event_ref)
        if not isinstance(result, dict):
            continue
        args = action.get("args") or {}
        tool = action.get("tool")
        if tool == "search" and isinstance(result.get("hits"), list):
            query = str(args.get("query") or "")
            if query:
                refs.add(f"search:{query}")
        elif tool == "read":
            note_id = str(args.get("note_id") or "")
            if note_id and str(result.get("note_id") or "") == note_id:
                refs.add(f"note:{note_id}")
        elif tool == "record_finding" and result.get("recorded") is True \
                and result.get("finding_ref"):
            refs.add(str(result["finding_ref"]))
        if tool in {"note_decision", "record_finding"} and result.get("testimony_ref"):
            refs.add(str(result["testimony_ref"]))
    return refs


def _runtime_reference_dependencies(episodes: list[dict[str, Any]],
                                    by_cycle_id: dict[str, dict[str, Any]], *,
                                    analysis_id: str) -> list[dict[str, Any]]:
    """Project explicit runtime reference use into auditable Semantica dependencies.

    A retrieval decision owns the searches and reads executed by its support cycles. When a later
    Atomic Decision explicitly cites one uniquely produced ``decision:*``, ``search:*``,
    ``note:*`` or ``finding:*`` reference, the earlier decision changed the later decision's
    available state. That is a deterministic ``INFLUENCED`` dependency. Repeated producers are
    left unlinked because the stable runtime locator cannot identify which execution the consumer
    used. Uncited temporal adjacency remains temporal-only.
    """
    owners: dict[str, set[int]] = {}
    for episode_index, episode in enumerate(episodes):
        for cycle_id in episode.get("source_cycle_ids") or []:
            for ref in _produced_runtime_refs(by_cycle_id[str(cycle_id)]):
                owners.setdefault(ref, set()).add(episode_index)

    dependency_refs: dict[tuple[int, int], set[str]] = {}
    for target_index, episode in enumerate(episodes):
        for cycle_id in episode.get("source_cycle_ids") or []:
            cycle = by_cycle_id[str(cycle_id)]
            results = {str(row.get("event_ref")): row.get("result") or {}
                       for row in cycle.get("observations") or []}
            for action in cycle.get("actions") or []:
                if action.get("tool") not in {"note_decision", "record_finding"}:
                    continue
                result = results.get(str(action.get("event_ref") or ""), {})
                for ref in _verified_runtime_refs(result):
                    prior_owners = {index for index in owners.get(ref, set())
                                    if index < target_index}
                    if len(prior_owners) != 1:
                        continue
                    source_index = next(iter(prior_owners))
                    dependency_refs.setdefault((source_index, target_index), set()).add(ref)

    rows: list[dict[str, Any]] = []
    for source_index, target_index in sorted(dependency_refs):
        refs = sorted(dependency_refs[(source_index, target_index)])
        rows.append({
            "assertion_id": f"{analysis_id}:runtime-dependency:{len(rows) + 1}",
            "source_episode_id": episodes[source_index]["episode_id"],
            "target_episode_id": episodes[target_index]["episode_id"],
            "relationship_type": "INFLUENCED",
            "evidence_refs": refs,
            "reasoning": (
                "The target decision explicitly cited runtime reference(s) produced by the "
                f"source decision's execution: {', '.join(refs)}."),
            "provenance": "DETERMINISTIC_DERIVED_FROM_RUNTIME_REFERENCE",
        })
    return rows


def _validate_source_refs(episode: dict[str, Any], first_cycle: dict[str, Any],
                          last_cycle: dict[str, Any], inventory: set[str]) \
        -> tuple[list[str], list[str]]:
    sources = episode.get("source_refs_by_field")
    if not isinstance(sources, dict) or set(sources) != set(EPISODE_FIELDS):
        raise ReconstructionError("source_refs_by_field must contain every episode field exactly")
    for field, refs in sources.items():
        if field == "observed_downstream_refs":
            # This is a redundant source index. The top-level claim is bounded by event order
            # below and becomes the canonical field value/index.
            continue
        if not isinstance(refs, list) or any(str(ref) not in inventory for ref in refs):
            bad = [str(ref) for ref in (refs if isinstance(refs, list) else [])
                   if str(ref) not in inventory]
            raise ReconstructionError(f"{field} contains unknown source refs: {bad!r}")

    first_id = str(first_cycle["cycle_id"])
    scenario_refs = list(map(str, sources["scenario"]))
    required = f"state_before:{first_id}"
    first_seq = int(first_cycle["source_seq_range"][0])
    if required not in scenario_refs:
        raise ReconstructionError("scenario must cite the first cycle's pre-decision state")
    for ref in scenario_refs:
        if ref.startswith("layer1:") and int(ref.split(":", 1)[1]) >= first_seq:
            raise ReconstructionError("scenario may cite only pre-decision state/events")

    last_seq = int(last_cycle["source_seq_range"][1])
    downstream = list(map(str, episode.get("observed_downstream_refs") or []))
    valid_downstream: list[str] = []
    discarded_downstream: list[str] = []
    for ref in downstream:
        if ref.startswith("layer1:") and ref in inventory \
                and int(ref.split(":", 1)[1]) > last_seq:
            valid_downstream.append(ref)
        else:
            discarded_downstream.append(ref)
    if episode.get("counterfactual_supported_impact"):
        raise ReconstructionError(
            "counterfactual-supported impact needs a separately verified counterfactual artifact")
    return list(dict.fromkeys(valid_downstream)), list(dict.fromkeys(discarded_downstream))


def _canonical_cycle_ownership(cycle_ids: list[str],
                               annotations: dict[str, dict[str, Any]]) \
        -> tuple[list[list[str]], list[str]]:
    """Derive episode envelopes from the model's per-cycle semantic role labels.

    Asking the model both to label every cycle and to copy the same ownership into two arrays
    creates a second, fallible source of truth. Once roles are fixed, each bearing cycle owns the
    contiguous support run immediately after it; mechanical cycles own themselves.
    """
    groups: list[list[str]] = []
    mechanical: list[str] = []
    current: list[str] | None = None
    for cycle_id in cycle_ids:
        role = annotations[cycle_id]["role"]
        if role == "DECISION_BEARING":
            current = [cycle_id]
            groups.append(current)
        elif role == "DECISION_SUPPORT":
            if current is None:
                raise ReconstructionError(
                    f"support cycle {cycle_id} has no preceding decision-bearing cycle")
            current.append(cycle_id)
        else:
            mechanical.append(cycle_id)
            current = None
    return groups, mechanical


def _retrieval_family(cycle: dict[str, Any]) -> str | None:
    """Return the first successful retrieval modality directly visible in one cycle."""
    families = {
        "list_documents": "inventory",
        "search": "search",
        "read": "read",
        "read_document": "read",
    }
    for action in cycle.get("actions") or []:
        if action.get("ok") is True and action.get("tool") in families:
            return families[str(action["tool"])]
    return None


def _is_inventory_pagination(cycle: dict[str, Any]) -> bool:
    """Whether this cycle mechanically continues an inventory from a later page."""
    return any(
        action.get("tool") == "list_documents"
        and action.get("ok") is True
        and isinstance((action.get("args") or {}).get("offset"), int)
        and int((action.get("args") or {})["offset"]) > 0
        for action in cycle.get("actions") or []
    )


def _canonical_annotations(cycles: list[dict[str, Any]],
                           annotations: dict[str, dict[str, Any]]) \
        -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Normalize boundaries whose semantics are directly visible in the fixed trace.

    A successful ``submit_answer`` with no new bearing label executes the immediately preceding
    commitment. Offset pagination executes an already-made inventory commitment. A switch among
    observable retrieval modalities changes the object of the retrieval choice. None of those
    boundaries is a semantic judgment for Luna to make differently on repeated reads.
    """
    canonical = {str(cycle_id): dict(annotation)
                 for cycle_id, annotation in annotations.items()}
    normalizations: list[dict[str, Any]] = []
    active_bearing = False
    active_function: str | None = None
    active_retrieval_family: str | None = None

    def normalize(cycle_id: str, annotation: dict[str, Any], field: str,
                  value: Any, reason: str) -> None:
        previous = annotation.get(field)
        if previous == value:
            return
        normalizations.append({
            "cycle_id": cycle_id,
            "field": field,
            "reader_value": previous,
            "canonical_value": value,
            "reason": reason,
            "provenance": DETERMINISTIC_DERIVED,
        })
        annotation[field] = value

    for cycle in cycles:
        cycle_id = str(cycle["cycle_id"])
        annotation = canonical[cycle_id]
        role = annotation["role"]
        family = _retrieval_family(cycle)
        has_submit = any(
            action.get("tool") == "submit_answer" and action.get("ok") is True
            for action in cycle.get("actions") or [])
        if has_submit and role != "DECISION_BEARING" and active_bearing \
                and role != "DECISION_SUPPORT":
            normalize(cycle_id, annotation, "role", "DECISION_SUPPORT",
                      "successful submit_answer executes the preceding commitment")
            role = "DECISION_SUPPORT"
        elif _is_inventory_pagination(cycle):
            reason = "offset pagination mechanically completes an earlier inventory commitment"
            normalize(cycle_id, annotation, "role", "MECHANICAL", reason)
            normalize(cycle_id, annotation, "decision_function", None, reason)
            role = "MECHANICAL"
        elif family is not None:
            if not active_bearing:
                reason = (
                    "observable retrieval modality switch starts a new auditable "
                    "where_to_look decision")
                normalize(cycle_id, annotation, "role", "DECISION_BEARING", reason)
                normalize(cycle_id, annotation, "decision_function", "where_to_look", reason)
                role = "DECISION_BEARING"
            elif active_function == "where_to_look" \
                    and active_retrieval_family is None:
                # This first retrieval call executes the immediately preceding commitment and
                # establishes which observable modality that commitment owns.
                active_retrieval_family = family
            elif active_function == "where_to_look" \
                    and active_retrieval_family is not None \
                    and family != active_retrieval_family:
                reason = (
                    "observable retrieval modality switch starts a new auditable "
                    "where_to_look decision")
                normalize(cycle_id, annotation, "role", "DECISION_BEARING", reason)
                normalize(cycle_id, annotation, "decision_function", "where_to_look", reason)
                role = "DECISION_BEARING"

        if role == "DECISION_BEARING":
            active_bearing = True
            active_function = annotation["decision_function"]
            active_retrieval_family = family if active_function == "where_to_look" else None
        elif role == "MECHANICAL":
            active_bearing = False
            active_function = None
            active_retrieval_family = None
    return canonical, normalizations


def _retrieval_subject(source_ids: list[str],
                       by_id: dict[str, dict[str, Any]]) -> str | None:
    """Derive a retrieval subject when execution makes the acted-on object unambiguous."""
    actions = [action for cycle_id in source_ids
               for action in (by_id[cycle_id].get("actions") or [])]
    for action in actions:
        tool = action.get("tool")
        if tool == "list_documents":
            filter_keys = {"doc_type", "note_type", "source", "type"}
            filtered = any((action.get("args") or {}).get(key) not in (None, "", [])
                           for key in filter_keys)
            return "retrieval_source" if filtered else "retrieval_inventory"
        if tool == "search":
            return "retrieval_query_batch"
        if tool in {"read", "read_document"}:
            return "retrieval_document_set"
        if tool == "record_evidence":
            return "evidence_item"
    return None


def _retrieval_decision(source_ids: list[str], by_id: dict[str, dict[str, Any]],
                        subject: str) -> tuple[str, list[str]] | None:
    """Derive the executed retrieval choice and its event refs from successful tool facts."""
    actions = [action for cycle_id in source_ids
               for action in (by_id[cycle_id].get("actions") or [])
               if action.get("ok") is True]
    if subject == "retrieval_query_batch":
        selected = [str((action.get("args") or {}).get("query") or "")
                    for action in actions if action.get("tool") == "search"]
        selected = list(dict.fromkeys(value for value in selected if value))
        refs = [str(action["event_ref"]) for action in actions
                if action.get("tool") == "search" and action.get("event_ref")]
        if selected:
            return f"Search query batch: {', '.join(selected)}", refs
    if subject == "retrieval_document_set":
        selected = [str((action.get("args") or {}).get("note_id") or "")
                    for action in actions
                    if action.get("tool") in {"read", "read_document"}]
        selected = list(dict.fromkeys(value for value in selected if value))
        refs = [str(action["event_ref"]) for action in actions
                if action.get("tool") in {"read", "read_document"}
                and action.get("event_ref")]
        if selected:
            return f"Open selected document set: {', '.join(selected)}", refs
    if subject in {"retrieval_inventory", "retrieval_source"}:
        listings = [action for action in actions if action.get("tool") == "list_documents"]
        refs = [str(action["event_ref"]) for action in listings if action.get("event_ref")]
        if listings:
            return ("Run filtered document inventory" if subject == "retrieval_source"
                    else "Run unfiltered document inventory"), refs
    return None


def verify_extraction(raw: dict[str, Any], cycles: list[dict[str, Any]], *,
                      analysis_id: str) -> dict[str, Any]:
    """Fail closed on every fact-ownership invariant before persistence or projection."""
    cycle_ids = [str(cycle["cycle_id"]) for cycle in cycles]
    by_id = {str(cycle["cycle_id"]): cycle for cycle in cycles}
    annotations = raw.get("cycle_annotations")
    if not isinstance(annotations, dict) or set(annotations) != set(cycle_ids):
        raise ReconstructionError("cycle annotations must cover the fixed cycle ids exactly")
    for cycle_id, annotation in annotations.items():
        role = annotation.get("role") if isinstance(annotation, dict) else None
        function = annotation.get("decision_function") if isinstance(annotation, dict) else None
        if role not in ROLES:
            raise ReconstructionError(f"cycle {cycle_id} has invalid role {role!r}")
        if role == "DECISION_BEARING" and function not in DECISION_TYPES:
            raise ReconstructionError(f"decision-bearing cycle {cycle_id} needs a function")
        if role != "DECISION_BEARING" and function is not None:
            raise ReconstructionError(f"non-decision cycle {cycle_id} cannot claim a function")

        cycle = by_id[cycle_id]
        if _runtime_receipt(cycle) is not None and role != "DECISION_BEARING":
            raise ReconstructionError(
                f"cycle {cycle_id} has a sealed runtime Decision Receipt and must preserve its "
                "witnessed decision boundary")
        commits_finding = any(
            action.get("tool") == "record_finding" and action.get("ok") is True
            for action in cycle.get("actions") or [])
        if commits_finding and (role != "DECISION_BEARING" or function != "standing"):
            raise ReconstructionError(
                "every successful record_finding cycle must be a separate standing decision")

    annotations, annotation_normalizations = _canonical_annotations(cycles, annotations)
    raw_episodes = raw.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ReconstructionError("episodes must be a list")
    canonical_groups, mechanical = _canonical_cycle_ownership(cycle_ids, annotations)
    if len(raw_episodes) != len(canonical_groups):
        raise ReconstructionError(
            f"episodes must contain one row per decision-bearing cycle: "
            f"expected {len(canonical_groups)}, got {len(raw_episodes)}")

    inventory = _inventory(cycles)
    episodes: list[dict[str, Any]] = []
    for number, (raw_episode, source_ids) in enumerate(
            zip(raw_episodes, canonical_groups, strict=True), 1):
        claimed_ids = list(map(str, raw_episode.get("source_cycle_ids") or []))
        if claimed_ids != source_ids:
            raise ReconstructionError(
                f"episode {number} source_cycle_ids must equal the canonical contiguous "
                f"envelope {source_ids!r}; got {claimed_ids!r}")
        roles = [annotations[cid]["role"] for cid in source_ids]
        # These are construction invariants now, retained as assertions near the persistence
        # boundary so a future ownership refactor cannot weaken atomicity silently.
        if roles.count("DECISION_BEARING") != 1 or roles[0] != "DECISION_BEARING" \
                or any(role not in {"DECISION_BEARING", "DECISION_SUPPORT"} for role in roles):
            raise ReconstructionError("deterministic cycle ownership violated atomicity")
        function = raw_episode.get("decision_function")
        bearing_id = source_ids[0]
        if function not in DECISION_TYPES \
                or function != annotations[bearing_id]["decision_function"]:
            raise ReconstructionError("episode function must agree with its atomic decision")
        subject = raw_episode.get("decision_subject")
        allowed_subjects = subjects_for(str(function))
        subject_was_derived = False
        subject_derivation = None
        execution_subject = (_retrieval_subject(source_ids, by_id)
                             if function == "where_to_look" else None)
        if execution_subject is not None:
            subject = execution_subject
            subject_was_derived = True
            subject_derivation = "DETERMINISTIC_DERIVED_FROM_EXECUTION"
        elif subject not in allowed_subjects:
            controlled = sorted(value for value in allowed_subjects if value != "other")
            if len(controlled) == 1:
                subject = controlled[0]
                subject_was_derived = True
                subject_derivation = "DETERMINISTIC_DERIVED_FROM_DECISION_FUNCTION"
            else:
                raise ReconstructionError(
                    f"decision subject {subject!r} is not valid for {function!r}")
        valid_downstream, discarded_downstream = _validate_source_refs(
            raw_episode, by_id[source_ids[0]], by_id[source_ids[-1]], inventory)

        sources = {field: list(map(str, raw_episode["source_refs_by_field"][field]))
                   for field in EPISODE_FIELDS}
        # The value is already a list of source refs. Once their existence and temporal
        # direction have been checked above, this second field-level index is pure,
        # deterministic bookkeeping rather than a model judgment.
        sources["observed_downstream_refs"] = valid_downstream
        field_provenance = {field: MODEL_RECONSTRUCTED for field in EPISODE_FIELDS}
        if subject_was_derived:
            field_provenance["decision_subject"] = subject_derivation
        field_provenance["observed_downstream_refs"] = DETERMINISTIC_DERIVED
        values = {field: (subject if field == "decision_subject" else
                          valid_downstream if field == "observed_downstream_refs" else
                          raw_episode.get(field))
                  for field in EPISODE_FIELDS}
        runtime_testimony = _runtime_testimony(by_id[bearing_id])
        if runtime_testimony:
            for field in ("material_question", "candidate_set", "decision",
                          "decision_rationale", "claimed_basis_summary",
                          "verified_reference_summary"):
                values[field] = runtime_testimony[field]
                field_provenance[field] = (SERVER_FACT if field == "verified_reference_summary"
                                           else SELF_REPORTED)
                refs = sources[field]
                if runtime_testimony["testimony_ref"] not in refs:
                    refs.append(runtime_testimony["testimony_ref"])
        elif function == "where_to_look":
            execution_decision = _retrieval_decision(source_ids, by_id, str(subject))
            if execution_decision is not None:
                values["decision"], decision_refs = execution_decision
                field_provenance["decision"] = "DETERMINISTIC_DERIVED_FROM_EXECUTION"
                sources["decision"] = decision_refs
        episodes.append({
            "episode_id": f"{analysis_id}:episode:{number}",
            "source_cycle_ids": source_ids,
            "source_event_ids": [event_id for cid in source_ids
                                 for event_id in by_id[cid].get("source_event_ids") or []],
            "bearing_cycle_id": bearing_id,
            "audit_unit": "ATOMIC_DECISION",
            "decision_function": function,
            **values,
            "field_provenance": field_provenance,
            "source_refs_by_field": sources,
            "runtime_testimony_ref": (runtime_testimony["testimony_ref"]
                                      if runtime_testimony else None),
            "runtime_receipt_ref": (runtime_testimony["receipt_ref"]
                                    if runtime_testimony else None),
            "runtime_receipt_schema": (runtime_testimony["receipt_schema"]
                                       if runtime_testimony else None),
            "runtime_receipt_hash": (runtime_testimony["receipt_hash"]
                                     if runtime_testimony else None),
            "runtime_receipt_provenance": (runtime_testimony["receipt_provenance"]
                                           if runtime_testimony else None),
            "reconstruction_provenance": MODEL_RECONSTRUCTED,
            "discarded_reconstruction_claims": ({
                "observed_downstream_refs": discarded_downstream,
                "reason": "not a real event after this episode",
                "provenance": DETERMINISTIC_DERIVED,
            } if discarded_downstream else {}),
        })

    assertions: list[dict[str, Any]] = []
    for number, link in enumerate(raw.get("causal_links") or [], 1):
        source_index, target_index = link.get("source_episode_index"), link.get("target_episode_index")
        if not isinstance(source_index, int) or not isinstance(target_index, int):
            raise ReconstructionError("causal link indices must be integers")
        if source_index == target_index or not (0 <= source_index < len(episodes)) \
                or not (0 <= target_index < len(episodes)):
            raise ReconstructionError("causal link endpoints must be two existing episodes")
        relationship = link.get("relationship_type")
        evidence_refs = list(map(str, link.get("evidence_refs") or []))
        if relationship not in CAUSAL_TYPES or not evidence_refs:
            raise ReconstructionError("causal link needs a typed relationship and evidence")
        if any(ref not in inventory for ref in evidence_refs):
            raise ReconstructionError("causal link contains an unknown evidence ref")
        assertions.append({
            "assertion_id": f"{analysis_id}:causal:{number}",
            "source_episode_id": episodes[source_index]["episode_id"],
            "target_episode_id": episodes[target_index]["episode_id"],
            "relationship_type": relationship,
            "evidence_refs": evidence_refs,
            "reasoning": str(link.get("reasoning") or ""),
            "provenance": MODEL_RECONSTRUCTED,
        })
    runtime_dependencies = _runtime_reference_dependencies(
        episodes, by_id, analysis_id=analysis_id)
    by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    for assertion in assertions:
        triple = (assertion["source_episode_id"], assertion["target_episode_id"],
                  assertion["relationship_type"])
        if triple in by_triple:
            raise ReconstructionError("duplicate causal relationship between two episodes")
        by_triple[triple] = assertion
    for dependency in runtime_dependencies:
        triple = (dependency["source_episode_id"], dependency["target_episode_id"],
                  dependency["relationship_type"])
        reconstructed = by_triple.get(triple)
        if reconstructed is None:
            assertions.append(dependency)
            by_triple[triple] = dependency
            continue
        # A server-verifiable runtime reference is stronger provenance than the reader's semantic
        # assertion. Keep the reader's claim as supporting context, while persisting one canonical
        # assertion for the one Semantica causal triple.
        dependency["evidence_refs"] = sorted(set(
            dependency["evidence_refs"] + reconstructed["evidence_refs"]))
        dependency["model_reconstructed_support"] = {
            "assertion_id": reconstructed["assertion_id"],
            "reasoning": reconstructed["reasoning"],
            "evidence_refs": reconstructed["evidence_refs"],
        }
        assertions[assertions.index(reconstructed)] = dependency
        by_triple[triple] = dependency
    return {"episodes": episodes, "mechanical_cycle_ids": mechanical,
            "cycle_annotations": annotations,
            "annotation_normalizations": annotation_normalizations,
            "causal_assertions": assertions}


def _write_append_only(directory: Path, artifact: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / f"{artifact['analysis_id']}.json").resolve()
    # The durable artifact must be self-locating.  Downstream Semantica projection
    # uses this reference to recover the owning run directory; assigning it only
    # after serialization silently redirects a later replay to the ledger parent.
    artifact["artifact_ref"] = str(path)
    rendered = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ReconstructionError(f"append-only analysis artifact collision at {path}")
    if not path.exists():
        path.write_text(rendered, encoding="utf-8")
    return path


def _alignment(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"bearing_cycle_id": episode["bearing_cycle_id"],
             "decision_function": episode["decision_function"],
             "decision_subject": episode["decision_subject"],
             "decision": episode["decision"]}
            for episode in artifact["episodes"]]


def reconstruct_run(review: LangtraceReviewTrace, ledger: Any, llm: StructuredLLM, *,
                    passes: int = 1, artifact_dir: Path | None = None,
                    reconstructor_identity: str = "terra",
                    max_attempts_per_pass: int = 3) -> dict[str, Any]:
    if passes < 1:
        raise ValueError("passes must be >= 1")
    if max_attempts_per_pass < 1:
        raise ValueError("max_attempts_per_pass must be >= 1")
    manifest = build_trace_completeness(review)
    cycles = build_react_cycles(review, manifest)
    model_projection = model_input_projection(cycles)
    receipt_manifest = _runtime_receipt_manifest(cycles)
    artifacts: list[dict[str, Any]] = []
    prompt = build_prompt(cycles)
    fallback_call_count = 0
    for pass_index in range(1, passes + 1):
        attempts: list[dict[str, Any]] = []
        attempt_prompt = prompt
        for attempt_index in range(1, max_attempts_per_pass + 1):
            records = getattr(llm, "call_records", None)
            prior_call_count = (len(records) if isinstance(records, list)
                                else fallback_call_count)
            raw = llm.generate_structured(
                attempt_prompt, response_format=extraction_response_format(cycles))
            fallback_call_count += 1
            call_record = _call_record(
                llm, prior_call_count, reconstructor_identity=reconstructor_identity)
            raw_hash = content_hash(raw)
            analysis_id = "analysis-" + content_hash({
                "schema": ANALYSIS_SCHEMA, "verifier_version": VERIFIER_VERSION,
                "taxonomy_version": DECISION_TAXONOMY_SCHEMA,
                "trace_id": review.trace_id,
                "trace_hash": manifest.content_hash, "reconstructor": reconstructor_identity,
                "reconstructor_call": call_record,
                "pass_index": pass_index, "raw_hash": raw_hash,
            })[:20]
            try:
                verified = verify_extraction(raw, cycles, analysis_id=analysis_id)
            except ReconstructionError as exc:
                attempts.append({
                    "attempt_index": attempt_index, "model_call": call_record,
                    "raw_hash": raw_hash, "validation_status": "REJECTED",
                    "validation_error": str(exc),
                })
                if attempt_index == max_attempts_per_pass:
                    raise ReconstructionError(
                        f"pass {pass_index} failed deterministic validation after "
                        f"{max_attempts_per_pass} attempt(s): {exc}") from exc
                attempt_prompt = f"""{prompt}

CORRECTION ATTEMPT {attempt_index + 1} OF {max_attempts_per_pass}
The previous candidate was rejected by the deterministic verifier for exactly this reason:
{exc}

Return the entire corrected JSON object, not a patch. Preserve the fixed cycles and satisfy the
specific invariant above. Re-audit every role/function pair, cycle assignment, temporal source
boundary, and source reference before returning.

REJECTED CANDIDATE (for correction only; it is not accepted evidence):
{json.dumps(raw, ensure_ascii=False, indent=2)}
"""
                continue
            attempts.append({
                "attempt_index": attempt_index, "model_call": call_record,
                "raw_hash": raw_hash, "validation_status": "ACCEPTED",
                "validation_error": None,
            })
            break
        artifact: dict[str, Any] = {
            "schema": ANALYSIS_SCHEMA, "verifier_version": VERIFIER_VERSION,
            "taxonomy_version": DECISION_TAXONOMY_SCHEMA,
            "model_input_projection_version": MODEL_INPUT_PROJECTION_VERSION,
            "model_input_projection_hash": content_hash(model_projection),
            "supported_runtime_receipt_schema": RUNTIME_DECISION_RECEIPT_SCHEMA,
            "runtime_receipt_manifest": receipt_manifest,
            "analysis_id": analysis_id, "run_id": review.run_id,
            "trace_id": review.trace_id, "trace_manifest_hash": manifest.manifest_hash,
            "trace_content_hash": manifest.content_hash,
            "task_presentation_hash": review.task_presentation_hash,
            "task_arm": review.task_arm,
            "review_model": review.review_model,
            "reconstructor_identity": reconstructor_identity, "pass_index": pass_index,
            "reconstructor_call": call_record,
            "reconstructor_attempts": attempts,
            "cycles_hash": content_hash(cycles), "cycles": cycles,
            **verified,
        }
        artifact["decision_receipt_coverage"] = _decision_receipt_coverage(
            artifact["episodes"])
        artifacts.append(artifact)

    alignments = [_alignment(artifact) for artifact in artifacts]
    agrees = all(alignment == alignments[0] for alignment in alignments[1:])
    status = "STABLE_ACROSS_PASSES" if agrees and passes > 1 else (
        "SINGLE_PASS_PROVISIONAL" if passes == 1 else "PROVISIONAL_DRIFT")
    stability = 1.0 if agrees and passes > 1 else (0.5 if passes == 1 else 0.0)
    for artifact in artifacts:
        artifact["stability_status"] = status
        artifact["reconstruction_stability"] = stability
        for episode in artifact["episodes"]:
            episode["stability_status"] = status
            episode["reconstruction_stability"] = stability
        artifact["analysis_artifact_hash"] = content_hash(artifact)
        if artifact_dir is not None:
            _write_append_only(artifact_dir, artifact)
        ledger.project_analysis(artifact)

    selected = (ledger.selected_analysis(review.run_id)
                if hasattr(ledger, "selected_analysis") else None)
    return {
        "schema": "acr.reconstruction_summary.v1", "run_id": review.run_id,
        "case_id": review.patient_id, "trace_id": review.trace_id,
        "trace_completeness": manifest.to_dict(),
        "analyses": [{
            "analysis_id": artifact["analysis_id"],
            "analysis_artifact_hash": artifact["analysis_artifact_hash"],
            "n_episodes": len(artifact["episodes"]),
            "n_mechanical_cycles": len(artifact["mechanical_cycle_ids"]),
            "stability_status": artifact["stability_status"],
            "reconstruction_stability": artifact["reconstruction_stability"],
            "reconstructor_call": artifact["reconstructor_call"],
            "n_attempts": len(artifact["reconstructor_attempts"]),
            "taxonomy_version": artifact["taxonomy_version"],
            "runtime_receipt_mode": artifact["runtime_receipt_manifest"]["mode"],
            "sealed_receipt_count": artifact["runtime_receipt_manifest"][
                "sealed_receipt_count"],
            "decision_receipt_coverage": artifact["decision_receipt_coverage"],
            "artifact_ref": artifact.get("artifact_ref"),
        } for artifact in artifacts],
        "drift": {"passes": passes, "alignment_agrees": agrees,
                  "alignments": alignments},
        "selected_analysis_id": selected,
    }


def render(summary: dict[str, Any]) -> str:
    out = [f"RUN {summary['run_id']} | trace {summary['trace_id']}"]
    manifest = summary.get("trace_completeness") or {}
    out.append(f"trace export: {manifest.get('export_status')} "
               f"({manifest.get('event_count')} Layer-1 events)")
    selected = summary.get("selected_analysis_id")
    out.append(f"selected analysis: {selected or 'NONE — provisional analyses shown separately'}")
    for row in summary.get("analyses") or []:
        out.append(f"- {row['analysis_id']}: {row['n_episodes']} episode(s), "
                   f"{row['stability_status']} (reconstruction stability="
                   f"{row['reconstruction_stability']:.2f})")
    drift = summary.get("drift") or {}
    out.append(f"alignment across {drift.get('passes', 0)} pass(es): "
               f"{'AGREES' if drift.get('alignment_agrees') else 'DRIFT'}")
    return "\n".join(out)
