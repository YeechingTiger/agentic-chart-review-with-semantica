"""Tool schemas + dispatch."""
from __future__ import annotations

import time
from typing import Any, Sequence

from ..corpus import PatientChart
from ..coverage import CoverageLedger          # the only coverage ledger — see note in state.py
from ..state import Evidence, EvidenceLedger


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
        "read_documents_batch",
        "Read MANY documents in one step. Use this for runtime-drawn validation samples: the "
        "obligation is sized in documents, not in turns, and reading them one per step cannot "
        "fit in any sane budget. Returns a compact excerpt of each.",
        {
            "note_ids": {"type": "array", "items": {"type": "string"},
                         "description": "note_ids to read, typically the ones the runtime drew"},
            "chars_each": {"type": "integer", "description": "excerpt size per document, default 1200"},
        },
        ["note_ids"],
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
        "answer is REJECTED unless the specification's proof obligation has been met. "
        "SPEC_INSUFFICIENT is a report about the SPECIFICATION, not about this chart: it is "
        "rejected unless it names the part at fault, and it may not carry a value.",
        {
            "status": {"type": "string", "enum": ["FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"]},
            "value": {"type": "object", "description": "object keyed by the spec's output field names"},
            "reasoning": {"type": "string", "description": "how the decision rules were applied"},
            # SPEC_INSUFFICIENT only. These three exist because the status alone told the
            # improvement loop nothing: it has to know WHICH text to go and change, and a
            # model asked to infer that from a bare status code will guess the most
            # rewritable paragraph. See graph.SPEC_SECTIONS.
            "spec_section": {"type": "string", "description":
                             "SPEC_INSUFFICIENT only: the part of the specification that "
                             "does not cover this case — one of decision_rule, "
                             "evidence_rules, conflict_rules, when_not_to_use, "
                             "boundary_cases, abstention, fields, proof_obligation, "
                             "data_source"},
            "spec_quote": {"type": "string", "description":
                           "SPEC_INSUFFICIENT only: the sentence you mean, quoted verbatim "
                           "from the specification. Omit it if no such sentence exists — "
                           "that the specification is SILENT is itself the report."},
            "uncovered_fields": {"type": "array", "items": {"type": "string"},
                                 "description": "SPEC_INSUFFICIENT only: which output "
                                                "fields it fails to cover. Omit if the "
                                                "whole answer is affected."},
        },
        ["status", "reasoning"],
    ),
]


class Toolbox:
    """Executes tool calls against one patient chart, maintaining both ledgers."""

    def __init__(self, chart: PatientChart, evidence: EvidenceLedger, coverage: CoverageLedger,
                 known_doc_types: Sequence[str] | None = None):
        self.chart = chart
        self.evidence = evidence
        self.coverage = coverage
        self.submitted: dict | None = None
        # The value domain is CORPUS-WIDE, not this patient's. Getting this wrong inverts the
        # very distinction the check exists for: if the domain were the patient's own types,
        # then "this patient has no pathology" — a finding, and often the answer — would come
        # back as UNKNOWN_DOC_TYPE, a query error. Fall back to the chart only when no corpus
        # vocabulary is supplied, and say so in the error.
        self.known_doc_types: list[str] = sorted(known_doc_types or chart.doc_types)
        self.domain_is_corpus_wide = known_doc_types is not None

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
        bad = self._resolve_doc_type(doc_type_contains)
        if bad:
            return bad
        page, total = self.chart.list_documents(doc_type_contains, date_from, date_to, limit, offset)
        if not any([doc_type_contains, date_from, date_to]):
            self.coverage.listed_documents = True
            self.coverage.total_documents = total
        return {
            "total_matching": total,
            "returned": len(page),
            "offset": offset,
            "more": offset + len(page) < total,
            # The filter was a real type. An empty list here IS a finding: this patient has
            # no such documents. Distinguishing this from UNKNOWN_DOC_TYPE is the whole point.
            "type_filter_valid": True,
            "type_exists_but_empty": bool(doc_type_contains) and total == 0,
            "documents": [d.to_dict() for d in page],
        }

    # -- enumerated value domains -------------------------------------------------
    def _resolve_doc_type(self, needle: str | None) -> dict | None:
        """Reject out-of-domain document types instead of quietly returning nothing.

        Document type is an enumerated value domain — the chart knows all 15 of them. An
        empty result therefore has two possible causes with opposite remedies: the type
        name does not exist (retry with a real one) or it exists and this patient has none
        (that is a finding). Returning `[]` for both is how a typo turns into a silent false
        negative: an agent that asked for "Biopsy", got nothing, and concluded the patient
        has no pathology would be wrong and would have no way to know it.

        Returns an error dict when the value is out of domain, else None.
        """
        if not needle:
            return None
        n = needle.lower().strip()
        if any(n in t.lower() for t in self.known_doc_types):
            return None                      # in the domain; emptiness is then a finding

        # Out of domain. Suggest by shared tokens, then by prefix, then fall back to the
        # types this patient actually has — an empty did_you_mean is honest but useless.
        toks = [t for t in n.replace("_", "-").split("-") if len(t) > 2]
        near = [t for t in self.known_doc_types
                if any(tok in t.lower() for tok in toks)] if toks else []
        if not near:
            near = [t for t in self.known_doc_types if t.lower()[:4] == n[:4]]
        if not near:
            near = self.chart.doc_types[:5]
        return {
            "error": "UNKNOWN_DOC_TYPE",
            "queried": needle,
            "message": ("That string matches no document type in the corpus vocabulary. This "
                        "is NOT evidence that the patient lacks such documents — the name is "
                        "wrong. Retry with a value from known_types. Do not draw any "
                        "conclusion from this result."
                        if self.domain_is_corpus_wide else
                        "That string matches no document type present in THIS patient's chart, "
                        "and no corpus-wide vocabulary was supplied, so absence and misspelling "
                        "cannot be told apart here. Retry with a value from known_types."),
            "domain": "corpus" if self.domain_is_corpus_wide else "patient_chart_only",
            "known_types": self.known_doc_types,
            "did_you_mean": near[:5],
        }

    def _unknown_note(self, note_id: str) -> dict:
        """note_id is an enumerated domain too — a fabricated one must not read as absence."""
        near = [n for n in self.chart._docs if note_id.split("_")[0].lower() in n.lower()][:5]
        return {
            "error": "UNKNOWN_NOTE_ID",
            "queried": note_id,
            "message": ("No such document in this chart. Use a note_id returned by "
                        "list_documents or search_notes; do not construct one."),
            "did_you_mean": near,
        }

    def _t_document_type_summary(self) -> dict:
        self.coverage.type_summary_seen = True
        rows = self.chart.type_summary()
        return {"patient_id": self.chart.patient_id, "n_documents": len(self.chart),
                "n_types": len(rows), "types": rows}

    def _t_search_notes(self, query: str, regex: bool = False, doc_type_contains=None,
                        date_from=None, date_to=None, max_hits: int = 25) -> dict:
        bad = self._resolve_doc_type(doc_type_contains)
        if bad:
            return bad
        hits = self.chart.search(query, regex, doc_type_contains, date_from, date_to, max_hits=max_hits)
        self.coverage.note_search(query, [h.note_id for h in hits])
        return {
            "query": query,
            "n_hits": len(hits),
            "truncated": len(hits) >= max_hits,
            "type_filter_valid": True,
            "hits": [h.__dict__ for h in hits],
        }

    def _t_read_document(self, note_id: str, offset: int = 0, limit: int = 4000) -> dict:
        if note_id not in self.chart._docs:
            return self._unknown_note(note_id)
        r = self.chart.read(note_id, offset, limit)
        self.coverage.note_read(note_id, r["doc_type"])
        return r

    def _t_read_documents_batch(self, note_ids: list[str], chars_each: int = 1200) -> dict:
        docs, unknown = [], []
        for nid in note_ids[:60]:
            if nid not in self.chart._docs:
                unknown.append(nid); continue
            r = self.chart.read(nid, 0, chars_each)
            self.coverage.note_read(nid, r["doc_type"])
            docs.append({"note_id": nid, "doc_type": r["doc_type"], "date": r["date"],
                         "total_chars": r["total_chars"], "text": r["text"]})
        return {"n_read": len(docs), "unknown_note_ids": unknown, "documents": docs}

    def _t_read_section(self, note_id: str, section: str = "") -> dict:
        if note_id not in self.chart._docs:
            return self._unknown_note(note_id)
        if not section:
            return {"note_id": note_id, "available_sections": self.chart.sections(note_id)}
        r = self.chart.read_section(note_id, section)
        if "error" in r:
            # Same distinction as UNKNOWN_DOC_TYPE, one level down: a section name that does
            # not exist is not the same as a section that exists and is empty.
            r["error"] = "UNKNOWN_SECTION"
            r["message"] = ("That section name is not present in this document. This is not "
                            "evidence of anything; pick one of available_sections.")
        else:
            self.coverage.note_section(note_id, r["section"], r.get("doc_type", ""))
        return r

    def _t_timeline(self, doc_type_contains=None, limit: int = 200) -> dict:
        bad = self._resolve_doc_type(doc_type_contains)
        if bad:
            return bad
        return {"events": self.chart.timeline(doc_type_contains, limit), "type_filter_valid": True}

    def _t_record_evidence(self, note_id: str, start: int, end: int, supports: str,
                           stance: str = "supports") -> dict:
        meta = self.chart._docs.get(note_id)
        if meta is None:
            return self._unknown_note(note_id)
        start, end = max(0, int(start)), max(0, int(end))
        if end <= start:
            return {"error": "end must be greater than start"}
        quote = self.chart.quote(note_id, start, end)
        if not quote.strip():
            return {"error": "that span is empty; re-check the offsets from your search hit"}
        self.evidence.add(Evidence(note_id, meta.doc_type, meta.date.isoformat(), start, end,
                                   quote, supports, "contradicts" if stance == "contradicts" else "supports"))
        return {"recorded": True, "n_evidence": len(self.evidence.items), "quote": quote[:300]}

    def _t_submit_answer(self, status: str, reasoning: str, value: dict | None = None,
                         spec_section: str = "", spec_quote: str = "",
                         uncovered_fields: list | None = None) -> dict:
        # The SPEC_INSUFFICIENT fields are carried verbatim and judged by the gate, not here.
        # A toolbox that quietly dropped them would make the gate's rejection unanswerable:
        # the agent would supply a section, be told it supplied none, and have no way to win.
        self.submitted = {"status": status, "value": value or {}, "reasoning": reasoning,
                          "spec_section": spec_section, "spec_quote": spec_quote,
                          "uncovered_fields": list(uncovered_fields or [])}
        return {"received": True, "status": status, "note": "pending validation"}
