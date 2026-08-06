"""Tool schemas + dispatch."""
from __future__ import annotations

import time
from collections.abc import Sequence

from ...chartstore.corpus import PatientChart
from ...contract import outcomes as OUTCOMES
from ...core.state import Evidence, EvidenceLedger
from ...review.coverage import CoverageLedger  # the only coverage ledger — see note in state.py


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


#: The parameter every retrieval tool offers so an action can name what prompted it.
#: NEVER in `required`: a cause is a judgement, and a required judgement becomes a ritual --
#: the model writes "to find the answer" on every call and the field measures nothing. What
#: it is for is the OTHER end: `attribution` reconstructing a run currently has to infer
#: "step 9 read D12 because step 7's search returned it" from adjacency, and reports that
#: inference without marking it as one.
CAUSE_PARAM = "because"

AFTER_PARAM = "after_event"

_CAUSE_PROPERTY = {
    "type": "string",
    "description": ("optional: what prompted this call — the search that surfaced the "
                    "document, the open thread it settles, the stratum it samples. Prose, "
                    "read by a later reader and never checked."),
}

#: The pointer, as a FLAT INTEGER beside the prose rather than a field inside it.
#:
#: The first version made `because` an `anyOf[string, object]` whose object form carried
#: `from.event`. Measured on a fresh run of `tactic-follow-dependency` with the card explicitly asking
#: for it: 0 of 18 calls emitted one. Offered two shapes, the model took the simpler. A flat
#: integer parameter sitting next to the prose is the same information with nothing to choose
#: between, and `because` goes back to being exactly the string every recorded run already
#: contains — so backwards compatibility stops being a special case and becomes the default.
#:
#: Still optional, still never refused. `acr.evaluation.evidence_chain` resolves it and counts;
#: judgement turned into a gate is what this repo measured and removed.
_AFTER_PROPERTY = {
    "type": "integer",
    "description": ("optional but strongly preferred: the `seq` of the EARLIER step that led "
                    "you here — the search that surfaced this document, the deferral you are "
                    "following, the inventory entry you are sampling. It must already have "
                    "happened. This is what makes your reasoning a chain a reader can walk "
                    "instead of a list of unrelated actions."),
}


#: The six retrieval tools. `submit_answer` is not among them: its status enum comes from
#: the contract, so it is built per run by `_submit_answer_tool` below.
_RETRIEVAL_TOOLS: list[dict] = [
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
            "query": {"anyOf": [{"type": "string"},
                                {"type": "array", "items": {"type": "string"}}],
                      "description": ("one keyword, or a LIST of keywords searched in this one "
                                      "call. Prefer the list: five terms in one call cost one "
                                      "call, and hits stay attributed to the term that found "
                                      "them.")},
            "doc_type_contains": {"type": "string"},
            "date_from": {"type": "string"},
            "date_to": {"type": "string"},
            "max_hits": {"type": "integer", "description": "default 25"},
            CAUSE_PARAM: _CAUSE_PROPERTY,
            AFTER_PARAM: _AFTER_PROPERTY,
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
            CAUSE_PARAM: _CAUSE_PROPERTY,
            AFTER_PARAM: _AFTER_PROPERTY,
        },
        ["note_id"],
    ),
    # `timeline` removed: it returned documents in date order, which `list_documents` already
    # gives with `date_from`/`date_to` and a type filter. A second way to ask one question is a
    # second thing to keep honest, and the tool surface should be ordinary — keyword search, and
    # the ability to see every note type and every note date. Nothing invented.
    #
    # `read_section` removed: it addressed headings through an ALL-CAPS regex that reached
    # `FINAL DIAGNOSIS` in 2.3% of the documents containing it and `ADDENDUM` in 0 of 2,401.
    # Use `search_notes` for the offset, then `read_document` with that offset.
    _tool(
        "read_documents_batch",
        "Read MANY documents in one step. Use this for runtime-drawn validation samples: the "
        "obligation is sized in documents, not in turns, and reading them one per step cannot "
        "fit in any sane budget. Returns a compact excerpt of each.",
        {
            "note_ids": {"type": "array", "items": {"type": "string"},
                         "description": "note_ids to read, typically the ones the runtime drew"},
            "chars_each": {"type": "integer", "description": "excerpt size per document, default 1200"},
            CAUSE_PARAM: _CAUSE_PROPERTY,
            AFTER_PARAM: _AFTER_PROPERTY,
        },
        ["note_ids"],
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
            "entity": {"type": "string",
                       "description": ("optional: which specimen, lesion or procedure this "
                                       "quote is ABOUT. Record it when the chart describes "
                                       "more than one; it is how a reader tells a quote about "
                                       "the reported lesion from a quote about another.")},
        },
        ["note_id", "start", "end", "supports"],
    ),
]


def _submit_answer_tool(spec=None) -> dict:
    """`submit_answer`, with its status enum taken from the CONTRACT.

    The enum used to be a literal here. That made the outcome space a property of the tool:
    a contract could not widen it, and one that instructed the model toward a status this
    list did not carry -- STORE.390 does exactly that in its proof obligation -- was asking
    for something the model had no way to send. See `acr.contract.outcomes`.

    The prose is generated from the same declaration for the same reason. A hand-written
    description beside a generated enum is a second statement of the outcome space, free to
    drift from the first, and the one the model actually reads.
    """
    space = OUTCOMES.declared_statuses(spec) if spec is not None else dict(OUTCOMES.DEFAULT_SPACE)
    offered = {n: d for n, d in space.items() if d.get("submittable", True) is not False}
    lines = ["Submit the final answer. `status` must be exactly one of:"]
    lines += [f"  {n} — {d['meaning']}" for n, d in offered.items()]
    if any(d["kind"] == OUTCOMES.KIND_ABSTAIN_EVIDENCE for d in offered.values()):
        lines.append(
            "A negative or absent answer is REJECTED unless the specification's proof "
            "obligation has been met.")
    spec_gap = [n for n, d in offered.items() if d["kind"] == OUTCOMES.KIND_ABSTAIN_SPEC]
    if spec_gap:
        lines.append(
            f"{' / '.join(spec_gap)} reports on the SPECIFICATION, not on this chart: it is "
            "rejected unless it names the part at fault, and it may not carry a value.")
    return _tool(
        "submit_answer",
        "\n".join(lines),
        {
            "status": {"type": "string", "enum": list(offered)},
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
            # THE ANCHOR, DECLARED. Siblings of `value` and deliberately not inside it: there
            # is no registry gold for either, so putting them in `value` would enter them into
            # `field_disagreements` and score the model against a truth that does not exist.
            #
            # Why they are asked for at all. The spec's question says "for the tumour being
            # reported" and nothing resolves WHICH tumour. Measured consequence: three runs
            # answered about the wrong neoplasm -- one coded a sigmoid colon hyperplastic polyp
            # in a lung-cancer chart, two picked the wrong lung lesion where the chart
            # documented an upper-lobe and a middle-lobe mass. In none of those cases could the
            # trace distinguish "did not notice the other lesion" from "noticed it and judged it
            # not the primary" from "the one-row-per-patient gold cannot express two tumours".
            # Enumeration makes those three look different.
            "lesions_considered": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "every distinct neoplasm or mass this chart documents, one entry each, with "
                    "its site/laterality and the note_id that names it. One entry when the "
                    "chart documents one tumour. This is what you looked at, not what you "
                    "answered."),
            },
            "reported_lesion": {
                "type": "string",
                "description": (
                    "which entry in lesions_considered this answer is about, and why each other "
                    "entry is not the reportable primary -- metastasis, second primary, prior "
                    "resected tumour, benign finding, or the same tumour described twice. If "
                    "two entries cannot be resolved into one reportable tumour, say so here "
                    "rather than picking one silently."),
            },
        },
        ["status", "reasoning"],
    )


def build_tool_schemas(spec=None) -> list[dict]:
    """The seven tools, with `submit_answer` bound to this contract's outcome space."""
    return [*_RETRIEVAL_TOOLS, _submit_answer_tool(spec)]


#: The surface with no contract in hand: several tests read the shape rather than a run's, and
#: a caller may need a scratch toolbox before a spec is chosen. Same seven tools, the three
#: statuses `outcomes.DEFAULT_SPACE` declares.
TOOL_SCHEMAS: list[dict] = build_tool_schemas()


class Toolbox:
    """Executes tool calls against one patient chart, maintaining both ledgers."""

    def __init__(self, chart: PatientChart, evidence: EvidenceLedger, coverage: CoverageLedger,
                 known_doc_types: Sequence[str] | None = None, spec=None):
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
        #: What the model said prompted the most recent dispatch. Read by the tracing hook
        #: immediately after the call; cleared each dispatch so a stale cause cannot attach
        #: itself to a later action as a causal link nobody wrote.
        self.last_cause: str = ""
        #: Built once, at construction, from the contract this run is under. Not recomputed
        #: per call: the manifest records the surface the model was offered, and a surface
        #: that could change mid-run would make that record a claim about one moment.
        self._schemas: list[dict] = build_tool_schemas(spec)

    def schemas(self) -> list[dict]:
        return self._schemas

    def dispatch(self, name: str, args: dict) -> tuple[dict, float]:
        t0 = time.time()
        # Stripped centrally, not accepted per tool: one place to change, and no `_t_` method
        # has to grow a parameter it does not use. Cleared every call — see `last_cause`.
        args = dict(args)
        self.last_cause = str(args.pop(CAUSE_PARAM, "") or "")
        self.last_after = args.pop(AFTER_PARAM, None)
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}",
                    "available": [s["function"]["name"] for s in self._schemas]}, 0.0
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

    @staticmethod
    def search_many(chart, query, doc_type_contains=None, date_from=None, date_to=None,
                    max_hits: int = 25) -> dict:
        """One call, one or many keywords, hits kept ATTRIBUTED TO THE TERM THAT FOUND THEM.

        The tool used to take a single string, so covering a chart with five terms cost five
        calls no matter what any policy said — measured at 506 searches over eighteen charts in
        E2, `biopsy` alone eighty-six times, while the same arm opened FEWER documents than a run
        with no card at all. Half of that was a bad card; this is the other half.

        `by_term` rather than a merged pool because which term surfaced a document is the one
        distinction the causal chain can use, and flattening throws it away.

        The cap is PER TERM. Sharing it would let the first term eat the budget and make every
        later term read as "not in this chart", which is the same failure as an unfiltered miss.
        """
        terms = [query] if isinstance(query, str) else list(query or [])
        terms = [str(t) for t in terms if str(t).strip()]
        if not terms:
            return {"error": "NO_TERMS",
                    "detail": ("search_notes needs at least one keyword. An empty list would "
                               "return zero hits, which reads as 'this chart contains nothing'.")}
        by_term = {}
        for t in terms:
            hits = chart.search(t, False, doc_type_contains, date_from, date_to,
                                max_hits=max_hits)
            by_term[t] = {"n_hits": len(hits), "truncated": len(hits) >= max_hits,
                          "hits": [h.__dict__ for h in hits]}
        out = {"terms": terms, "by_term": by_term,
               "n_hits_total": sum(b["n_hits"] for b in by_term.values())}
        if len(terms) == 1:
            # The one-term call keeps its original shape. Every recorded run, every downstream
            # consumer and thirty-three tests read `hits` and `n_hits` off the top level; a new
            # capability that silently changes the old call's return is a breaking change wearing
            # an addition's clothes.
            t = terms[0]
            out.update({"query": t, "n_hits": by_term[t]["n_hits"],
                        "truncated": by_term[t]["truncated"], "hits": by_term[t]["hits"]})
        return out

    def _t_search_notes(self, query, doc_type_contains=None,
                        date_from=None, date_to=None, max_hits: int = 25) -> dict:
        bad = self._resolve_doc_type(doc_type_contains)
        if bad:
            return bad
        out = self.search_many(self.chart, query, doc_type_contains, date_from, date_to,
                               max_hits=max_hits)
        if out.get("error"):
            return out
        for term, block in out["by_term"].items():
            self.coverage.note_search(term, [h["note_id"] for h in block["hits"]])
        out["type_filter_valid"] = True
        return out

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
                unknown.append(nid)
                continue
            r = self.chart.read(nid, 0, chars_each)
            self.coverage.note_read(nid, r["doc_type"])
            docs.append({"note_id": nid, "doc_type": r["doc_type"], "date": r["date"],
                         "total_chars": r["total_chars"], "text": r["text"]})
        return {"n_read": len(docs), "unknown_note_ids": unknown, "documents": docs}

    # `_t_read_section` is gone with `SECTION_RE`. See the note in `acr.chartstore.corpus`: the regex it
    # ran on could address `FINAL DIAGNOSIS` in 2.3% of the documents that contain it and
    # `ADDENDUM` in none of the 2,401 that do, and with no section argument it handed the model
    # an `available_sections` list with the final diagnosis missing from it. `search` returns
    # offsets and `read` takes them, which reaches the same text without a heading vocabulary.

    # `_t_timeline` is gone too, and the handler had to go with the schema rather than after it.
    # `dispatch` resolves `getattr(self, f"_t_{name}")`, so a handler with no schema entry is
    # still callable — the surface the model is OFFERED and the surface it can REACH were two
    # different sets, and the manifest records the first. A run that guessed the name would have
    # used a tool no reader of its manifest could see. `verify_mechanisms.py` M1 compares the two
    # sets so this cannot recur silently.

    def _t_record_evidence(self, note_id: str, start: int, end: int, supports: str,
                           stance: str = "supports", entity: str = "") -> dict:
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
                                   quote, supports,
                                   "contradicts" if stance == "contradicts" else "supports",
                                   entity=str(entity or "")))
        return {"recorded": True, "n_evidence": len(self.evidence.items), "quote": quote[:300]}

    def _t_submit_answer(self, status: str, reasoning: str, value: dict | None = None,
                         spec_section: str = "", spec_quote: str = "",
                         uncovered_fields: list | None = None,
                         lesions_considered: list | None = None,
                         reported_lesion: str = "") -> dict:
        # The SPEC_INSUFFICIENT fields are carried verbatim and judged by the gate, not here.
        # A toolbox that quietly dropped them would make the gate's rejection unanswerable:
        # the agent would supply a section, be told it supplied none, and have no way to win.
        self.submitted = {"status": status, "value": value or {}, "reasoning": reasoning,
                          "spec_section": spec_section, "spec_quote": spec_quote,
                          "uncovered_fields": list(uncovered_fields or []),
                          # Carried verbatim, never validated here. Nothing refuses an answer
                          # over these -- they are a record of the anchor the model chose, and
                          # `len(lesions_considered) > 1` is the signal that the gold row for
                          # this patient may not be able to express the answer at all.
                          "lesions_considered": [str(x) for x in (lesions_considered or [])],
                          "reported_lesion": str(reported_lesion or "")}
        return {"received": True, "status": status, "note": "pending validation"}
