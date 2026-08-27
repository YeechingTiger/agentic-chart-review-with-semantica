"""The chart, offered to any MCP client, with the gate at submit and the trace at the boundary.

THE ONE OBSERVATION PRINCIPLE. Everything that must be trusted is recorded HERE, server-side,
per tool call — because a model's action only counts if it came through a tool, and this is the
one place every tool call passes regardless of which harness is driving. The harness's own event
stream is archived by the runner as a second layer and is never load-bearing: swap the harness
and Layer 1 (this file's `trace.jsonl`) plus everything downstream of it are unchanged.

WHAT THE GATE REFUSES — three checks, deliberately no more:

  1. a status the contract did not declare submittable      (outcomes.submittable_statuses)
  2. a value-kind answer with no recorded evidence           (a FOUND owes a witness)
  3. an evidence-abstention before an unfiltered listing     (you may not claim "the chart
     does not establish it" without having looked at what the chart contains)

The old runtime's other refusals — forced sampling, elusion bounds, keyword gates — were each
measured and made advisory over there; they do not come back here until something consumes them.

This process serves ONE patient against ONE contract, configured by environment:

  ACR_MVP_SPEC          path to the contract YAML
  ACR_MVP_PATIENT_DIR   path to the patient's document directory
  ACR_MVP_RUN_DIR       where trace.jsonl and result.json are written
  ACR_MVP_TASK_PRESENTATION  immutable ContractSnapshot written by the runner.  It is the
                             only authority for whether a cited rule was actually offered.

stdout speaks JSON-RPC only (the MCP stdio transport); diagnostics go to stderr.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acr.chartstore.corpus import PatientChart
from acr.contract.outcomes import declared_statuses, submittable_statuses
from acr.contract.spec import load_spec
from acr.mvp.decision_receipts import make_runtime_decision_receipt
from acr.mvp.task_presentation import ContractSnapshot, build_task_presentation
from acr.mvp.warrants import BASIS_SOURCES, RULE_COVERAGE_CLAIMS, RunFacts, normalize_basis_sources

PROTOCOL_VERSION_FALLBACK = "2025-06-18"

# MCP tool descriptions are part of what the agent is actually shown. Material operating
# rules therefore need a stable Task Presentation identity, not just prose hidden inside the
# transient tools/list response.
TOOL_SCHEMA_INSTRUCTIONS = {
    "list_documents_inventory_gate": (
        "Call list_documents without filters at least once before asserting that the chart "
        "does not establish something."
    ),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ToolState:
    """What one review has accumulated: evidence, the listing flag, the accepted answer."""

    def __init__(self, task_presentation: ContractSnapshot) -> None:
        self.task_presentation = task_presentation
        self.evidence: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []
        self.listed_documents = False
        self.accepted: dict[str, Any] | None = None
        self.n_decisions = 0
        self.n_searches = 0
        self.n_reads = 0
        self.n_listings = 0
        #: What this run has actually observed, which is what a claimed input is checked
        #: against. Read and merely-surfaced are kept apart on purpose: "cited a document it
        #: read in full" and "cited a document it only saw as a search snippet" are different
        #: warrants, and a guideline about how much reading an answer owes needs to tell them
        #: apart.
        self.documents_read: list[str] = []
        self.documents_seen: set[str] = set()
        self.searches_run: list[str] = []
        self.finding_refs: set[str] = set()
        self.decision_refs: set[str] = set()

    def facts(self) -> RunFacts:
        """What this run has observed, in the shape the citation check reads."""
        return RunFacts(list(self.documents_read), set(self.documents_seen),
                        list(self.searches_run), len(self.evidence),
                        set(self.finding_refs), set(self.decision_refs))

    def snapshot(self) -> dict[str, Any]:
        """Server-side facts at this moment — the context a decision point gets for free,
        so 'clean decision, rich context' does not depend on the model narrating well."""
        # Capped: semantica bounds a metadata value at 1000 chars, and the snapshot must
        # stay small enough to ride along on every decision node.
        return {"n_searches": self.n_searches, "n_reads": self.n_reads,
                "n_listings": self.n_listings, "n_evidence": len(self.evidence),
                "n_findings": len(self.findings),
                "unfiltered_listing_done": self.listed_documents,
                "documents_read": self.documents_read[-10:],
                "searches_run": self.searches_run[-10:],
                "evidence_notes": sorted({e["note_id"] for e in self.evidence})[:15]}


class ChartToolServer:
    def __init__(self, spec_path: Path, patient_dir: Path, run_dir: Path,
                 task_presentation_path: Path | None = None) -> None:
        self.spec = load_spec(spec_path)
        self.chart = PatientChart(patient_dir)
        self.patient_id = patient_dir.name
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        if task_presentation_path is not None:
            self.task_presentation = ContractSnapshot.from_path(task_presentation_path)
        else:
            # Direct unit/stdio use has no runner.  It still gets an explicit snapshot rather
            # than silently treating the canonical spec as if it were presented.
            _, self.task_presentation = build_task_presentation(
                self.spec, run_id=self.run_dir.name, arm_id="detailed",
                operational_preamble="Direct chart-tool invocation.",
                operational_instructions=TOOL_SCHEMA_INSTRUCTIONS)
        self.state = ToolState(self.task_presentation)
        self._acceptance_pending = False
        self._seq = 0
        self._emit(
            "run_meta",
            spec_id=self.spec.spec_id,
            spec_hash=self.spec.spec_hash,
            patient_id=self.patient_id,
            submittable=list(submittable_statuses(self.spec)),
            task_arm=self.task_presentation.arm_id,
            task_presentation_hash=self.task_presentation.presentation_hash,
            live_ledger=None,
        )

    # ------------------------------------------------------------------ trace (Layer 1)
    def _emit(self, kind: str, **payload: Any) -> int:
        self._seq += 1
        rec = {"seq": self._seq, "ts": _now(), "kind": kind, **payload}
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return self._seq


    # ------------------------------------------------------------------ tool surface
    def schemas(self) -> list[dict[str, Any]]:
        opt_str = {"type": "string"}
        objective = {
            "type": "string",
            "description": "which open question this call is meant to resolve (optional, recorded)",
        }
        return [
            {
                "name": "list_documents",
                "description": (
                    "List this patient's documents (metadata only). "
                    + TOOL_SCHEMA_INSTRUCTIONS["list_documents_inventory_gate"]
                    + " Cite instruction:list_documents_inventory_gate if this rule is a basis."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_type_contains": opt_str,
                        "date_from": opt_str,
                        "date_to": opt_str,
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                        "objective": objective,
                    },
                },
            },
            {
                "name": "search",
                "description": "Full-text search across this patient's documents "
                "(notation-tolerant, case-insensitive). You choose the terms.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "doc_type_contains": opt_str,
                        "date_from": opt_str,
                        "date_to": opt_str,
                        "objective": objective,
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read",
                "description": "Read one document by note_id, paginated by character offset.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "objective": objective,
                    },
                    "required": ["note_id"],
                },
            },
            {
                "name": "note_decision",
                "description": "Record concise Decision Testimony BEFORE a decision-bearing "
                "action. This is an auditable explanation, not private chain-of-thought and "
                "not a request to classify the decision. Claims are retained even when an "
                "exact citation is unknown or was not offered.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "facing": {
                            "type": "string",
                            "description": "the situation: what question is open, what you know",
                        },
                        "decision": {
                            "type": "string",
                            "description": "what you decided to do or conclude",
                        },
                        "because": {
                            "type": "string",
                            "description": "the rationale, naming the evidence or rule it rests on",
                        },
                        "alternatives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "the alternatives you considered and set aside",
                        },
                        "basis_sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(BASIS_SOURCES)},
                            "description": "where the rationale came from; orthogonal to whether "
                            "a Task Contract rule directly covers the situation",
                        },
                        "cited_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "exact rule/card/evidence references actually used",
                        },
                        "checked_discriminating_fact_refs": {
                            "type": "array", "items": {"type": "string"},
                            "description": "exact discriminating_fact.<fact_id> values checked",
                        },
                        "rule_coverage_claim": {
                            "type": "string", "enum": list(RULE_COVERAGE_CLAIMS),
                            "description": "how the supplied rules cover this decision",
                        },
                        "provisional_inference": {
                            "type": ["string", "null"],
                            "description": "an assumption added beyond the supplied rules, or null",
                        },
                        "uncertainty": {
                            "type": ["string", "null"],
                            "description": "an unresolved ambiguity or conflict, or null",
                        },
                    },
                    "required": ["facing", "decision", "because", "basis_sources",
                                 "cited_refs", "checked_discriminating_fact_refs",
                                 "rule_coverage_claim"],
                },
            },
            {
                "name": "record_finding",
                "description": "Judge and record how ONE read note stands for ONE requested "
                "field. This call is itself the atomic Decision Testimony: state the situation, "
                "rationale, exact basis, rule coverage, and any inference or uncertainty here. "
                "Do not precede it with a compound note_decision. Standing and assertion class "
                "are self-reported; span resolution is a server fact.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "string"},
                        "field": {"type": "string"},
                        "standing": {"type": "string", "enum": [
                            "can_establish", "merely_mentions", "neither"]},
                        "assertion_class": {"type": "string"},
                        "source_start": {"type": "integer"},
                        "source_end": {"type": "integer"},
                        "event_time": {"type": ["string", "null"]},
                        "record_time": {"type": ["string", "null"]},
                        "carried_forward": {"type": ["boolean", "null"]},
                        "facing": {
                            "type": "string",
                            "description": "the open standing question and facts known before "
                            "this finding is committed",
                        },
                        "because": {
                            "type": "string",
                            "description": "why this note has the selected standing/assertion",
                        },
                        "alternatives": {
                            "type": "array", "items": {"type": "string"},
                            "description": "other plausible standings or assertions considered",
                        },
                        "basis_sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(BASIS_SOURCES)},
                        },
                        "cited_refs": {
                            "type": "array", "items": {"type": "string"},
                            "description": "exact note/rule/finding references actually used",
                        },
                        "checked_discriminating_fact_refs": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "rule_coverage_claim": {
                            "type": "string", "enum": list(RULE_COVERAGE_CLAIMS),
                        },
                        "provisional_inference": {"type": ["string", "null"]},
                        "uncertainty": {"type": ["string", "null"]},
                    },
                    "required": [
                        "note_id", "field", "standing", "assertion_class", "facing",
                        "because", "basis_sources", "cited_refs",
                        "checked_discriminating_fact_refs", "rule_coverage_claim",
                    ],
                },
            },
            {
                "name": "record_evidence",
                "description": "Record a span of a document as evidence. The server resolves the "
                "quote from its own copy; you cannot paste text in. A FOUND answer requires at "
                "least one recorded span.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "supports": {
                            "type": "string",
                            "description": "what this span establishes, in one sentence",
                        },
                        "field": opt_str,
                    },
                    "required": ["note_id", "start", "end", "supports"],
                },
            },
            {
                "name": "submit_answer",
                "description": "Submit the final answer for validation. The verdict comes back; "
                "a refusal names what is missing and you may fix it and submit again.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": list(submittable_statuses(self.spec)),
                        },
                        "value": {
                            "type": ["object", "null"],
                            "description": "field name -> value; null for abstentions",
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["status"],
                },
            },
        ]

    def call(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Returns (payload, is_error). Every call is traced with its full result."""
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            payload: dict[str, Any] = {"error": f"unknown tool {name!r}"}
            self._emit("tool_call", tool=name, args=args, result=payload, ok=False)
            return payload, True
        state_before = self.state.snapshot()
        try:
            payload = handler(**args)
            ok = "error" not in payload
        except TypeError as e:  # bad/missing arguments — the model's error, said plainly
            payload = {"error": f"bad arguments for {name}: {e}"}
            ok = False
        except KeyError as e:  # unknown note_id from PatientChart internals
            payload = {"error": f"unknown note_id {e}"}
            ok = False
        trace_payload = dict(payload)
        if ok:
            receipt = make_runtime_decision_receipt(
                name, args, payload, state_before,
                source_event_ref=f"layer1:{self._seq + 1}")
            if receipt is not None:
                trace_payload["decision_receipt"] = receipt
        self._emit("tool_call", tool=name, args=args, result=trace_payload, ok=ok)
        if name == "submit_answer" and self._acceptance_pending:
            # Emitted here, after the tool_call record, so the trace reads in causal order.
            self._acceptance_pending = False
            self._emit("answer_accepted", status=self.state.accepted["status"],
                       value=self.state.accepted["value"])
        return payload, not ok

    # ------------------------------------------------------------------ handlers
    def _t_list_documents(self, doc_type_contains=None, date_from=None, date_to=None,
                          limit=200, offset=0, objective=None) -> dict[str, Any]:
        limit, offset = int(limit), int(offset)
        page, total = self.chart.list_documents(
            doc_type_contains=doc_type_contains, date_from=date_from, date_to=date_to,
            limit=limit, offset=offset)
        self.state.n_listings += 1
        self.state.documents_seen.update(d.note_id for d in page)
        if not any([doc_type_contains, date_from, date_to]) and offset == 0:
            self.state.listed_documents = True
        returned = len(page)
        return {"documents": [d.to_dict() for d in page], "total": total,
                "returned": returned, "offset": offset, "limit": limit,
                "page_complete": offset + returned >= total,
                "unreturned": max(total - (offset + returned), 0),
                "types": self.chart.type_summary(), "objective": objective,
                "context": self.state.snapshot()}

    def _t_search(self, query, doc_type_contains=None, date_from=None, date_to=None,
                  objective=None) -> dict[str, Any]:
        self.state.n_searches += 1
        self.state.searches_run.append(query)
        hits = self.chart.search(query, doc_type_contains=doc_type_contains,
                                 date_from=date_from, date_to=date_to)
        self.state.documents_seen.update(h.note_id for h in hits)
        return {"hits": [vars(h) for h in hits], "n": len(hits), "objective": objective,
                "context": self.state.snapshot()}

    def _t_read(self, note_id, offset=0, limit=4000, objective=None) -> dict[str, Any]:
        self.state.n_reads += 1
        try:
            page = self.chart.read(note_id, int(offset), int(limit))
        except KeyError:
            return {"error": f"unknown note_id {note_id!r}"}
        if note_id not in self.state.documents_read:
            self.state.documents_read.append(note_id)
        self.state.documents_seen.add(note_id)
        return {**page, "objective": objective, "context": self.state.snapshot()}

    def _testimony_result(self, *, basis_sources: Any, cited_refs: Any,
                          checked_discriminating_fact_refs: Any,
                          rule_coverage_claim: Any,
                          legacy_grounding: Any = None) -> dict[str, Any]:
        """Resolve one atomic self-report against server-owned facts.

        The semantic content remains the agent's claim. Its occurrence, order, available
        context, and reference availability are deterministic facts shared by both generic
        decisions and structured note findings.
        """
        self.state.n_decisions += 1
        if basis_sources is None and legacy_grounding is not None:
            basis_sources = legacy_grounding
        kinds, unrecognised = normalize_basis_sources(basis_sources)
        cited_refs = cited_refs if isinstance(cited_refs, list) else []
        checked = (checked_discriminating_fact_refs
                   if isinstance(checked_discriminating_fact_refs, list) else [])
        citations: list[dict[str, Any]] = []
        for ref in cited_refs:
            raw = str(ref)
            candidate = raw.removeprefix("rule:")
            if candidate.startswith(("decision_rule.", "conflict_rule.", "evidence_rule.",
                                     "answer_check.", "field_", "abstention.",
                                     "proof_obligation.", "discriminating_fact.")):
                citations.append(self.task_presentation.resolve_rule(raw))
            elif raw.startswith(("card:", "instruction:")):
                citations.append(self.task_presentation.resolve_asset(raw))
            else:
                citations.append(self.state.facts().resolve(raw))
        fact_resolutions = [self.task_presentation.resolve_rule(ref) for ref in checked]
        claim = (str(rule_coverage_claim) if rule_coverage_claim in RULE_COVERAGE_CLAIMS
                 else None)
        testimony_ref = f"decision:{self._seq + 1}"
        self.state.decision_refs.add(testimony_ref)
        context = self.state.snapshot()
        out: dict[str, Any] = {"noted": True, "n_decisions": self.state.n_decisions,
                               "testimony_ref": testimony_ref,
                               "basis_sources": kinds,
                               "rule_coverage_claim": claim,
                               "citation_resolutions": citations,
                               "checked_fact_resolutions": fact_resolutions,
                               "context": context}
        if unrecognised:
            out["note"] = f"unrecognised basis source {unrecognised!r} recorded as claimed"
            out["basis_sources_unrecognised"] = unrecognised
        if claim is None:
            out["rule_coverage_note"] = "missing or unrecognised claim recorded, not refused"
        return out

    def _t_note_decision(self, facing, decision, because, basis_sources=None,
                         cited_refs=None, checked_discriminating_fact_refs=None,
                         rule_coverage_claim=None, provisional_inference=None,
                         alternatives=None, uncertainty=None, **legacy: Any) -> dict[str, Any]:
        # Self-reported content on the deterministic channel: WHAT was said is the model's
        # claim, but that it was said, when, in what order, against which server-side state,
        # and whether the information it cites was ever actually observed, are recorded fact.
        # Old artifacts can still be replayed through this boundary, but the current tool schema
        # exposes only the new names. Never upgrade an old `contract` claim into a verified rule.
        return self._testimony_result(
            basis_sources=basis_sources, cited_refs=cited_refs,
            checked_discriminating_fact_refs=checked_discriminating_fact_refs,
            rule_coverage_claim=rule_coverage_claim,
            legacy_grounding=legacy.get("grounding"),
        )

    def _resolve_used(self, used: Any) -> list[dict[str, Any]]:
        """Each claimed input, checked against what this run actually observed — by the same
        `RunFacts` the read-back path uses, so a warrant cannot count as false in the run and
        true in the report."""
        return self.state.facts().resolve_all(used)

    def _t_record_evidence(self, note_id, start, end, supports, field=None) -> dict[str, Any]:
        start, end = int(start), int(end)
        try:
            total = self.chart.read(note_id, 0, 0)["total_chars"]
        except KeyError:
            return {"error": f"unknown note_id {note_id!r}"}
        # Bounds are refused, not clamped: a span the document cannot contain is not evidence,
        # and silently shortening it would record less text than the model believes it cited.
        if start < 0 or end <= start or end > total:
            return {"error": f"span [{start},{end}) is outside the document (0..{total})"}
        quote = self.chart.quote(note_id, start, end)
        ev = {"note_id": note_id, "start": start, "end": end,
              "quote": quote, "supports": supports, "field": field}
        self.state.evidence.append(ev)
        return {"recorded": True, "n_evidence": len(self.state.evidence), "quote": quote}

    def _t_record_finding(self, note_id, field, standing, assertion_class,
                          source_start=None, source_end=None, event_time=None,
                          record_time=None, carried_forward=None,
                          facing=None, because=None, basis_sources=None, cited_refs=None,
                          checked_discriminating_fact_refs=None, rule_coverage_claim=None,
                          provisional_inference=None, alternatives=None, uncertainty=None,
                          decision_testimony_ref=None) -> dict[str, Any]:
        if note_id not in self.state.documents_read:
            return {"error": "record_finding requires the note to have been read first"}
        if standing not in {"can_establish", "merely_mentions", "neither"}:
            return {"error": f"unknown standing {standing!r}"}
        if standing in {"can_establish", "merely_mentions"} and (
                source_start is None or source_end is None):
            return {"error": f"{standing} requires source_start and source_end"}
        if standing == "neither" and assertion_class != "not_applicable":
            return {"error": "neither requires assertion_class='not_applicable'"}
        if standing != "neither" and assertion_class == "not_applicable":
            return {"error": f"{standing} requires a substantive assertion_class"}

        span: list[int] | None = None
        quote: str | None = None
        if source_start is not None or source_end is not None:
            try:
                start, end = int(source_start), int(source_end)
                total = self.chart.read(note_id, 0, 0)["total_chars"]
            except (TypeError, ValueError):
                return {"error": "source_start and source_end must be integers"}
            if start < 0 or end <= start or end > total:
                return {"error": f"span [{start},{end}) is outside the document (0..{total})"}
            span = [start, end]
            quote = self.chart.quote(note_id, start, end)

        # Compatibility note: old direct callers may omit the new audit fields or carry an
        # earlier decision_testimony_ref. The current MCP schema never offers that legacy link;
        # a new run records a self-contained testimony on this same atomic call.
        has_atomic_testimony = all(value is not None for value in (
            facing, because, basis_sources, cited_refs,
            checked_discriminating_fact_refs, rule_coverage_claim))
        testimony: dict[str, Any] = {}
        if has_atomic_testimony:
            testimony = self._testimony_result(
                basis_sources=basis_sources, cited_refs=cited_refs,
                checked_discriminating_fact_refs=checked_discriminating_fact_refs,
                rule_coverage_claim=rule_coverage_claim,
            )

        finding_ref = f"finding:{len(self.state.findings) + 1}"
        finding = {
            "note_id": note_id, "field": field, "standing": standing,
            "assertion_class": assertion_class, "event_time": event_time,
            "record_time": record_time, "carried_forward": carried_forward,
            "span": span, "quote": quote,
            "finding_ref": finding_ref,
            "decision_testimony_ref": (
                testimony.get("testimony_ref") or decision_testimony_ref),
        }
        self.state.findings.append(finding)
        self.state.finding_refs.add(finding_ref)
        out = {
            "recorded": True, "finding_ref": finding_ref,
            "server_fact": {"note_read": True, "span_resolved": span is not None,
                            "span": span},
            "self_reported": {
                "facing": facing,
                "decision": f"{note_id} is {standing} for {field} ({assertion_class})",
                "because": because,
                "standing": standing,
                "assertion_class": assertion_class,
                "alternatives": alternatives or [],
                "provisional_inference": provisional_inference,
                "uncertainty": uncertainty,
            },
            "quote": quote,
        }
        if testimony:
            out |= testimony
        elif decision_testimony_ref:
            out["decision_testimony_ref"] = decision_testimony_ref
            out["instrumentation_status"] = "LEGACY_SHARED_TESTIMONY"
        else:
            out["instrumentation_status"] = "MISSING_ATOMIC_TESTIMONY"
        return out

    def _t_submit_answer(self, status, value=None, reasoning=None) -> dict[str, Any]:
        if self.state.accepted is not None:
            return {"accepted": True, "note": "an answer was already accepted; this run is done"}
        verdict = self._gate(status, value)
        if verdict["accepted"]:
            answer = {
                "status": status, "value": value, "reasoning": reasoning,
                "evidence": self.state.evidence,
                "patient_id": self.patient_id,
                "spec_id": self.spec.spec_id, "spec_hash": self.spec.spec_hash,
                "gate": verdict, "accepted_at": _now(),
            }
            self.state.accepted = answer
            (self.run_dir / "result.json").write_text(
                json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
            self._acceptance_pending = True
        return verdict

    # ------------------------------------------------------------------ the gate
    def _gate(self, status: str, value: Any) -> dict[str, Any]:
        declared = declared_statuses(self.spec)
        if status not in submittable_statuses(self.spec):
            return {"accepted": False, "why": "status not declared submittable by this contract",
                    "missing": [f"status must be one of {list(submittable_statuses(self.spec))}"]}
        kind = declared[status]["kind"]
        if kind == "value" and not self.state.evidence:
            return {"accepted": False, "why": "a value answer owes recorded evidence",
                    "missing": ["call record_evidence on the span that establishes the value"]}
        if kind == "abstain_evidence" and not self.state.listed_documents:
            return {"accepted": False,
                    "why": "an absence claim owes a look at what the chart contains",
                    "missing": ["call list_documents with no filters before abstaining"]}
        return {"accepted": True, "why": "obligations for this status are discharged",
                "kind": kind, "n_evidence": len(self.state.evidence)}


# ---------------------------------------------------------------------- MCP stdio transport
def _rpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def serve(server: ChartToolServer, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}
        if method == "initialize":
            resp = _rpc_result(req_id, {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION_FALLBACK),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "acr-chart", "version": "0.1.0"},
            })
        elif method in ("notifications/initialized", "notifications/cancelled"):
            continue  # notifications get no response
        elif method == "ping":
            resp = _rpc_result(req_id, {})
        elif method == "tools/list":
            resp = _rpc_result(req_id, {"tools": server.schemas()})
        elif method == "tools/call":
            payload, is_error = server.call(params.get("name", ""), params.get("arguments") or {})
            resp = _rpc_result(req_id, {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "isError": is_error,
            })
        elif req_id is None:
            continue  # unknown notification
        else:
            resp = _rpc_error(req_id, -32601, f"method not found: {method}")
        stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        stdout.flush()


def main() -> None:
    spec_path = Path(os.environ["ACR_MVP_SPEC"])
    patient_dir = Path(os.environ["ACR_MVP_PATIENT_DIR"])
    run_dir = Path(os.environ["ACR_MVP_RUN_DIR"])
    presentation = os.environ.get("ACR_MVP_TASK_PRESENTATION")
    server = ChartToolServer(spec_path, patient_dir, run_dir,
                             task_presentation_path=Path(presentation) if presentation else None)
    print(f"acr-chart toolserver: {patient_dir.name} / {server.spec.spec_id}", file=sys.stderr)
    serve(server)
    # The harness closes stdin when the session ends; nothing to finalize here — result.json
    # is written at acceptance time and the runner writes the fallback when there was none.
    time.sleep(0)


if __name__ == "__main__":
    main()
