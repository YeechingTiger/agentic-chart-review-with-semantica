"""Tool schemas + dispatch."""
from __future__ import annotations

import time
from typing import Any

from ..corpus import PatientChart
from ..state import CoverageLedger, Evidence, EvidenceLedger


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool(
        "list_documents",
        "List this patient's documents as METADATA ONLY (type, date, size). Returns no text. "
        "Use this first to see what exists and when. Supports filtering and paging.",
        {
            "doc_type_contains": {"type": "string", "description": "substring filter on document type, e.g. 'Pathology'"},
            "date_from": {"type": "string", "description": "YYYY-MM-DD inclusive lower bound"},
            "date_to": {"type": "string", "description": "YYYY-MM-DD inclusive upper bound"},
            "limit": {"type": "integer", "description": "page size, default 60"},
            "offset": {"type": "integer", "description": "page offset, default 0"},
        },
    ),
    _tool(
        "document_type_summary",
        "Counts and date span for every document type in the chart. The cheapest way to "
        "orient yourself in a large record before deciding what to read.",
        {},
    ),
    _tool(
        "search_notes",
        "Case-insensitive search across the patient's documents. Returns matches with the "
        "note_id and exact character offsets so you can cite them.",
        {
            "query": {"type": "string", "description": "term or phrase; set regex=true for a pattern"},
            "regex": {"type": "boolean"},
            "doc_type_contains": {"type": "string"},
            "date_from": {"type": "string"},
            "date_to": {"type": "string"},
            "max_hits": {"type": "integer", "description": "default 25"},
        },
        ["query"],
    ),
    _tool(
        "read_document",
        "Read a document's text, paginated. Character offsets are stable and citable.",
        {
            "note_id": {"type": "string"},
            "offset": {"type": "integer", "description": "character offset, default 0"},
            "limit": {"type": "integer", "description": "characters to return, default 4000"},
        },
        ["note_id"],
    ),
    _tool(
        "read_section",
        "Read one named section of a document (e.g. IMPRESSION, FINAL DIAGNOSIS, "
        "ASSESSMENT AND PLAN). Much cheaper than reading the whole note. Call with an empty "
        "section to list the sections available.",
        {"note_id": {"type": "string"}, "section": {"type": "string"}},
        ["note_id"],
    ),
    _tool(
        "timeline",
        "Chronological list of documents, optionally filtered by type. Use to establish "
        "sequence — e.g. whether a disease-free interval preceded a later finding.",
        {"doc_type_contains": {"type": "string"}, "limit": {"type": "integer"}},
    ),
    _tool(
        "record_evidence",
        "Record a verbatim quote as evidence. You MUST record evidence before submitting an "
        "answer — the final answer step can only see what is in this ledger. Set stance to "
        "'contradicts' for evidence that cuts against your conclusion; record it anyway.",
        {
            "note_id": {"type": "string"},
            "start": {"type": "integer", "description": "character offset where the quote begins"},
            "end": {"type": "integer", "description": "character offset where the quote ends"},
            "supports": {"type": "string", "description": "which field or assertion this backs"},
            "stance": {"type": "string", "enum": ["supports", "contradicts"]},
        },
        ["note_id", "start", "end", "supports"],
    ),
    _tool(
        "submit_answer",
        "Submit the final answer. status must be FOUND, EVIDENCE_INSUFFICIENT (spec is clear "
        "but the chart lacks the evidence) or SPEC_INSUFFICIENT (the specification does not "
        "cover this case, or the variable is not derivable from notes). A negative or absent "
        "answer is REJECTED unless the specification's proof obligation has been met.",
        {
            "status": {"type": "string", "enum": ["FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"]},
            "value": {"type": "object", "description": "object keyed by the spec's output field names"},
            "reasoning": {"type": "string", "description": "how the decision rules were applied"},
        },
        ["status", "reasoning"],
    ),
]


class Toolbox:
    """Executes tool calls against one patient chart, maintaining both ledgers."""

    def __init__(self, chart: PatientChart, evidence: EvidenceLedger, coverage: CoverageLedger):
        self.chart = chart
        self.evidence = evidence
        self.coverage = coverage
        self.submitted: dict | None = None

    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def dispatch(self, name: str, args: dict) -> tuple[dict, float]:
        t0 = time.time()
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}", "available": [s["function"]["name"] for s in TOOL_SCHEMAS]}, 0.0
        try:
            out = fn(**args)
        except TypeError as e:
            out = {"error": f"bad arguments for {name}: {e}"}
        except KeyError as e:
            out = {"error": f"unknown note_id {e}"}
        except Exception as e:  # noqa: BLE001 - surface tool errors to the model, don't crash the run
            out = {"error": f"{type(e).__name__}: {e}"}
        return out, (time.time() - t0) * 1000

    # -- implementations ---------------------------------------------------------
    def _t_list_documents(self, doc_type_contains=None, date_from=None, date_to=None,
                          limit: int = 60, offset: int = 0) -> dict:
        page, total = self.chart.list_documents(doc_type_contains, date_from, date_to, limit, offset)
        if not any([doc_type_contains, date_from, date_to]):
            self.coverage.listed_documents = True
            self.coverage.total_documents = total
        return {
            "total_matching": total,
            "returned": len(page),
            "offset": offset,
            "more": offset + len(page) < total,
            "documents": [d.to_dict() for d in page],
        }

    def _t_document_type_summary(self) -> dict:
        self.coverage.type_summary_seen = True
        rows = self.chart.type_summary()
        return {"patient_id": self.chart.patient_id, "n_documents": len(self.chart),
                "n_types": len(rows), "types": rows}

    def _t_search_notes(self, query: str, regex: bool = False, doc_type_contains=None,
                        date_from=None, date_to=None, max_hits: int = 25) -> dict:
        self.coverage.note_search(query)
        hits = self.chart.search(query, regex, doc_type_contains, date_from, date_to, max_hits=max_hits)
        return {
            "query": query,
            "n_hits": len(hits),
            "truncated": len(hits) >= max_hits,
            "hits": [h.__dict__ for h in hits],
        }

    def _t_read_document(self, note_id: str, offset: int = 0, limit: int = 4000) -> dict:
        r = self.chart.read(note_id, offset, limit)
        self.coverage.note_read(note_id, r["doc_type"])
        return r

    def _t_read_section(self, note_id: str, section: str = "") -> dict:
        if not section:
            return {"note_id": note_id, "available_sections": self.chart.sections(note_id)}
        r = self.chart.read_section(note_id, section)
        if "error" not in r:
            self.coverage.note_section(note_id, r["section"], r.get("doc_type", ""))
        return r

    def _t_timeline(self, doc_type_contains=None, limit: int = 200) -> dict:
        return {"events": self.chart.timeline(doc_type_contains, limit)}

    def _t_record_evidence(self, note_id: str, start: int, end: int, supports: str,
                           stance: str = "supports") -> dict:
        meta = self.chart._docs.get(note_id)
        if meta is None:
            return {"error": f"unknown note_id {note_id!r}"}
        start, end = max(0, int(start)), max(0, int(end))
        if end <= start:
            return {"error": "end must be greater than start"}
        quote = self.chart.quote(note_id, start, end)
        if not quote.strip():
            return {"error": "that span is empty; re-check the offsets from your search hit"}
        self.evidence.add(Evidence(note_id, meta.doc_type, meta.date.isoformat(), start, end,
                                   quote, supports, "contradicts" if stance == "contradicts" else "supports"))
        return {"recorded": True, "n_evidence": len(self.evidence.items), "quote": quote[:300]}

    def _t_submit_answer(self, status: str, reasoning: str, value: dict | None = None) -> dict:
        self.submitted = {"status": status, "value": value or {}, "reasoning": reasoning}
        return {"received": True, "status": status, "note": "pending validation"}
