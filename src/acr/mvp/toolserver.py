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
  ACR_MVP_LEDGER        optional: a semantica ledger to record judgments into LIVE, at the
                        moment each happens. Best-effort by construction — a ledger failure
                        is a stderr line and the run proceeds; the trace can always rebuild
                        the ledger (ingest_run), never the other way around.

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
from acr.mvp.warrants import (GROUNDING_KINDS, INPUT_KINDS, RunFacts,
                              normalize_grounding)

PROTOCOL_VERSION_FALLBACK = "2025-06-18"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ToolState:
    """What one review has accumulated: evidence, the listing flag, the accepted answer."""

    def __init__(self) -> None:
        self.evidence: list[dict[str, Any]] = []
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

    def facts(self) -> RunFacts:
        """What this run has observed, in the shape the citation check reads."""
        return RunFacts(list(self.documents_read), set(self.documents_seen),
                        list(self.searches_run), len(self.evidence))

    def snapshot(self) -> dict[str, Any]:
        """Server-side facts at this moment — the context a decision point gets for free,
        so 'clean decision, rich context' does not depend on the model narrating well."""
        # Capped: semantica bounds a metadata value at 1000 chars, and the snapshot must
        # stay small enough to ride along on every decision node.
        return {"n_searches": self.n_searches, "n_reads": self.n_reads,
                "n_listings": self.n_listings, "n_evidence": len(self.evidence),
                "unfiltered_listing_done": self.listed_documents,
                "documents_read": self.documents_read[-10:],
                "searches_run": self.searches_run[-10:],
                "evidence_notes": sorted({e["note_id"] for e in self.evidence})[:15]}


class ChartToolServer:
    def __init__(self, spec_path: Path, patient_dir: Path, run_dir: Path,
                 ledger_path: Path | None = None) -> None:
        self.spec = load_spec(spec_path)
        self.chart = PatientChart(patient_dir)
        self.patient_id = patient_dir.name
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.state = ToolState()
        self._acceptance_pending = False
        self._seq = 0
        # The ledger is constructed LAZILY, at the first event worth recording — semantica's
        # import costs ~2s, and paying it here would delay the MCP initialize handshake past
        # the model's first tool calls (codex starts the turn without waiting for late
        # registrations; the calls bounce as "unsupported" and the review starts blind).
        self._ledger_path = Path(ledger_path) if ledger_path else None
        self._recorder: Any = None
        self._emit(
            "run_meta",
            spec_id=self.spec.spec_id,
            spec_hash=self.spec.spec_hash,
            patient_id=self.patient_id,
            submittable=list(submittable_statuses(self.spec)),
            live_ledger=str(ledger_path) if ledger_path else None,
        )

    def _live_recorder(self):
        """Synchronous recording, best-effort: judgments reach the ledger at the moment they
        happen. Any failure here degrades to trace-only — never to a failed run."""
        if self._recorder is None and self._ledger_path is not None:
            try:
                from acr.mvp.ledger import RunRecorder, SemanticaLedger
                self._recorder = RunRecorder(SemanticaLedger(self._ledger_path),
                                             run_id=self.run_dir.name,
                                             spec_id=self.spec.spec_id,
                                             case_id=self.patient_id)
            except Exception as e:  # noqa: BLE001 - survive anything here
                print(f"acr-chart toolserver: live ledger disabled ({e})", file=sys.stderr)
                self._ledger_path = None
        return self._recorder

    # ------------------------------------------------------------------ trace (Layer 1)
    def _emit(self, kind: str, **payload: Any) -> int:
        self._seq += 1
        rec = {"seq": self._seq, "ts": _now(), "kind": kind, **payload}
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return self._seq

    def _record_live(self, name: str, args: dict[str, Any], payload: dict[str, Any],
                     ok: bool, seq: int) -> None:
        """Drive the same RunRecorder the replay path drives, one event at a time."""
        recordable = ((name in ("record_evidence", "note_decision") and ok)
                      or (name == "submit_answer" and "accepted" in payload))
        if not recordable:
            return
        recorder = self._live_recorder()
        if recorder is None:
            return
        try:
            if name == "record_evidence":
                recorder.on_evidence(args, payload.get("quote", ""))
            elif name == "note_decision":
                recorder.on_step(args, seq, context=payload.get("context"),
                                 used=payload.get("used"),
                                 grounding=payload.get("grounding"))
            else:
                recorder.on_submission(args, payload, seq)
            recorder.ledger.save()
        except Exception as e:  # noqa: BLE001 - bookkeeping must never fail the run
            print(f"acr-chart toolserver: live ledger write failed ({e})", file=sys.stderr)
            self._recorder, self._ledger_path = None, None

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
                "description": "List this patient's documents (metadata only). Call it without "
                "filters at least once before asserting that the chart does not establish something.",
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
                "description": "Note a decision point BEFORE acting on it: the situation you "
                "face, what you decided, and why. Call it whenever you choose between "
                "alternatives — what to look for next, which document governs, whether the "
                "evidence suffices, whether to stop. Notes are recorded, never judged, and "
                "never refused; they are how your reasoning stays auditable. You are not "
                "asked to classify the decision — that is done later, by a reader.",
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
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "the alternatives you considered and set aside",
                        },
                        "used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "the information this decision rests on, each as a "
                            "reference the record can check: "
                            + "; ".join(f"{k}:<...> ({v.split(' — ')[0]})"
                                        for k, v in INPUT_KINDS.items())
                            + ". Cite what you actually used, not what you could have.",
                        },
                        "grounding": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(GROUNDING_KINDS)},
                            "description": "where this reasoning came from — the one thing "
                            "nobody can reconstruct afterwards, so only you can report it: "
                            + "; ".join(f"{k} = {v}" for k, v in GROUNDING_KINDS.items())
                            + ". `own_knowledge` is never held against you; recording it "
                            "falsely as `contract` is the failure worth avoiding.",
                        },
                    },
                    "required": ["facing", "decision", "because"],
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
        try:
            payload = handler(**args)
            ok = "error" not in payload
        except TypeError as e:  # bad/missing arguments — the model's error, said plainly
            payload = {"error": f"bad arguments for {name}: {e}"}
            ok = False
        except KeyError as e:  # unknown note_id from PatientChart internals
            payload = {"error": f"unknown note_id {e}"}
            ok = False
        seq = self._emit("tool_call", tool=name, args=args, result=payload, ok=ok)
        self._record_live(name, args, payload, ok, seq)
        if name == "submit_answer" and self._acceptance_pending:
            # Emitted here, after the tool_call record, so the trace reads in causal order.
            self._acceptance_pending = False
            self._emit("answer_accepted", status=self.state.accepted["status"],
                       value=self.state.accepted["value"])
            recorder = self._live_recorder()
            if recorder is not None:
                try:
                    recorder.on_result(self.state.accepted)
                    recorder.ledger.save()
                except Exception as e:  # noqa: BLE001
                    print(f"acr-chart toolserver: live ledger write failed ({e})",
                          file=sys.stderr)
                    self._recorder, self._ledger_path = None, None
        return payload, not ok

    # ------------------------------------------------------------------ handlers
    def _t_list_documents(self, doc_type_contains=None, date_from=None, date_to=None,
                          limit=200, offset=0, objective=None) -> dict[str, Any]:
        page, total = self.chart.list_documents(
            doc_type_contains=doc_type_contains, date_from=date_from, date_to=date_to,
            limit=int(limit), offset=int(offset))
        self.state.n_listings += 1
        self.state.documents_seen.update(d.note_id for d in page)
        if not any([doc_type_contains, date_from, date_to]) and int(offset) == 0:
            self.state.listed_documents = True
        return {"documents": [d.to_dict() for d in page], "total": total,
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

    def _t_note_decision(self, facing, decision, because,
                         used=None, options=None, grounding=None) -> dict[str, Any]:
        # Self-reported content on the deterministic channel: WHAT was said is the model's
        # claim, but that it was said, when, in what order, against which server-side state,
        # and whether the information it cites was ever actually observed, are recorded fact.
        del facing, decision, because, options
        self.state.n_decisions += 1
        kinds, unrecognised = normalize_grounding(grounding)
        out: dict[str, Any] = {"noted": True, "n_decisions": self.state.n_decisions,
                               "used": self._resolve_used(used), "grounding": kinds,
                               "context": self.state.snapshot()}
        if unrecognised:
            out["note"] = f"unrecognised grounding {unrecognised!r} recorded as claimed"
            out["grounding_claimed"] = unrecognised
        return out

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
    ledger = os.environ.get("ACR_MVP_LEDGER")
    server = ChartToolServer(spec_path, patient_dir, run_dir,
                             ledger_path=Path(ledger) if ledger else None)
    print(f"acr-chart toolserver: {patient_dir.name} / {server.spec.spec_id}", file=sys.stderr)
    serve(server)
    # The harness closes stdin when the session ends; nothing to finalize here — result.json
    # is written at acceptance time and the runner writes the fallback when there was none.
    time.sleep(0)


if __name__ == "__main__":
    main()
