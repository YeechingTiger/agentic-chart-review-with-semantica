"""The single decision on whether a SUBMITTED answer may stand, returned as a recoverable
rejection rather than raised.

THERE IS EXACTLY ONE GATE, AND THIS IS IT
-----------------------------------------
These are module-level functions and not methods on `ChartReviewAgent` because three front
ends must apply the SAME judgement: the langgraph loop (`graph.ChartReviewAgent._gate`), the
MCP server (`mcp_server`), and the deepagents runtime (`deep_runner`). Two gate
implementations that can disagree is the `state.py` two-ledger failure repeated one layer up:
both would compute a verdict, neither would raise when they diverged, and you would be left
with two answers and no way to choose. Nothing outside this module may decide validity for
itself — `ChartReviewAgent` keeps `_gate` / `_check_gate` as thin delegates, and
`run_triggers` reads `check_gate`'s misses to notice a deadlock but never rules on them.

`submit_answer` does not end the run. It is validated: a negative or absent answer is
rejected unless the spec's proof obligation is satisfied by the *computed* coverage ledger.
The rejection, with its reasons, is fed back to the model as an observation. Prompting a
model to "be sure you looked everywhere" is a wish; checking the ledger is a control.

The mirror of this module is `answer_contract`, which holds what an answer owes at EMISSION.
The difference is who is left to hear it: a rejection here goes back into a loop the agent
can act in, so it is returned; a violation there is found after the loop is over, so it
raises.
"""
from __future__ import annotations

from .answer_checks import check_answer_detail, check_field_formats_detail
from .answer_contract import SPEC_SECTIONS
from .corpus import PatientChart
from .coverage import (CoverageLedger, GateResult, admissibility_for_citations, evaluate_gate,
                       keyword_was_searched)
from .coverage_planner import CoveragePlan, OpenThreadLedger
from .spec import ExtractionSpec
from .state import EvidenceLedger


def keyword_hits_among_drawn(spec: ExtractionSpec, coverage: CoverageLedger,
                             chart: PatientChart) -> set[str]:
    """Which drawn documents match the spec's keywords, judged without an LLM.

    A counterweight to inferring relevance from what the agent cited: a drawn document
    that matches the keywords but was never cited is a suspected recognition failure.
    Reported, never gated — keyword matching has its own false positives — but it stops a
    run reporting "0 hits" while having walked past something.
    """
    kws: list[str] = list(getattr(spec.proof_obligation, "required_keywords", []) or [])
    fn = getattr(spec.proof_obligation, "for_negative", {}) or {}
    for st in (fn.get("strata") or []):
        kws.extend(st.get("required_keywords") or [])
    kws = [k for k in {k.lower() for k in kws} if len(k) > 3]
    if not kws:
        return set()
    hits: set[str] = set()
    for ids in coverage.drawn.values():
        for nid in ids:
            if nid not in chart._docs:
                continue
            try:
                txt = chart.read(nid, 0, 20000)["text"].lower()
            except Exception:  # noqa: BLE001 - an unreadable document is not a gate failure
                continue
            if any(k in txt for k in kws):
                hits.add(nid)
    return hits


def check_gate(spec: ExtractionSpec, coverage: CoverageLedger,
               plan: CoveragePlan | None = None) -> GateResult:
    """Evaluate the spec's proof obligation against the stratified ledger.

    COVERAGE IS EVALUATED AGAINST THE FINAL, EXPANDED TERM LIST — that is what `plan` adds.
    Without it, adding a term at run time would be free: the agent could name a term in a
    revision, never run it, and still discharge the spec's shorter list. Making the expanded
    list binding is also the second half of what keeps monotone expansion safe. The FIRST
    list, the spec-declared one, is never used here; it is the falsification baseline and
    lives in `plan.initial_keywords`.
    """
    fn = getattr(spec.proof_obligation, "for_negative", {}) or {}
    gate_spec = dict(fn.get("gate") or {})
    if not gate_spec and fn.get("required_coverage"):
        # A spec written before stratification: fall back to the keyword checks it does
        # declare, rather than silently passing everything.
        gate_spec = {"required_keywords_all_searched": True}
    g = evaluate_gate(gate_spec, coverage.stratum_results())
    for kw in getattr(spec.proof_obligation, "required_keywords", []) or []:
        # One-directional containment; see coverage.keyword_was_searched. The `t in kw` half
        # this used to carry let a search for "t" discharge every required keyword at once.
        if not keyword_was_searched(kw, coverage.searched_terms):
            g.missing.append(f"required search not performed: {kw!r}")
    for kw in (plan.keywords if plan else []):
        if not keyword_was_searched(kw, coverage.searched_terms):
            g.missing.append(f"required search not performed: {kw!r} — this term is in the "
                             "retrieval plan (the spec declared it, or you added it). A term "
                             "you added and did not run widens nothing")
    if not coverage.listed_documents:
        g.missing.append("must list the patient's documents before asserting absence")
    g.verdict = "PASS" if not g.missing else "FAIL"
    return g


def check_threads(threads: OpenThreadLedger | None) -> list[str]:
    """Unsettled threads, as gate refusals. Empty when there are none, or none are known.

    THIS IS THE 8046 ERROR, wired as a control. A histology was coded off a line reading
    "special stains pending" and the addendum that resolved it — in the same file, 353
    characters past where the read stopped — was never chased. Every part of the machinery
    to catch that already existed as advice: the thread-chasing skill, the marker catalogue,
    the `truncated` flag on every read. Advice a model may decline to act on is not a
    control. So an open thread now refuses the answer, and the only ways past are a
    resolution or a dismissal that states a reason — both recorded, so that "the reviewer
    did not notice" and "the reviewer decided it did not matter" stay distinguishable in the
    manifest.
    """
    if threads is None:
        return []
    out = []
    for t in threads.unresolved():
        out.append(f"unsettled thread {t.thread_id}: {t.marker!r} in {t.note_id} "
                   f"({t.doc_type or 'unknown type'}) — {t.obligation}. Resolve it (say where "
                   f"it was settled) or dismiss it with a reason; both are recorded.")
    return out


def _gate_spec_insufficient(spec: ExtractionSpec, submitted: dict, *, tracer=None) -> dict:
    """What SPEC_INSUFFICIENT owes instead of a coverage proof.

    Three refusals, in the order they matter:

    1. NO VALUE. This is the abuse the exit invites: an agent that cannot meet the FOUND
       standard declares the spec inadequate and ships its code anyway. It works, because
       `concordance.variables_from_answer` promotes any populated field to FOUND regardless
       of the answer's status — so the code lands in a denominator having proved nothing.
       This repo has already shipped one exit of exactly that shape.
    2. A NAMED SECTION, from the closed list. "The spec is unclear" with no destination is
       the path of least resistance §6b predicts, and it cannot be routed.
    3. THE AGENT'S OWN WORDS, and any quote it offers must actually be in the spec. A
       fabricated quote would sail straight through refine.py's citation mask, which is the
       one thing standing between a plausible rewrite and an unjustified spec edit.
    """
    value = {k: v for k, v in (submitted.get("value") or {}).items() if str(v).strip()}
    if value:
        if tracer:
            tracer.emit("spec_insufficient_carried_value", severity="warning",
                        fields=sorted(value))
        return {"accepted": False,
                "why": "SPEC_INSUFFICIENT cannot carry a coded value",
                "missing": [
                    f"you set {sorted(value)} while answering SPEC_INSUFFICIENT. Those are "
                    "contradictory claims: one says the specification cannot decide this "
                    "case, the other decides it.",
                    "If the specification does cover the fields you coded, answer FOUND for "
                    "them and meet the evidence standard. If it does not, resubmit "
                    "SPEC_INSUFFICIENT with value omitted.",
                ]}

    forced = spec.data_source == "outside_notes"
    missing: list[str] = []
    section = str(submitted.get("spec_section") or "").strip()
    if section and section not in SPEC_SECTIONS:
        missing.append(f"spec_section={section!r} is not a part of a specification. "
                       f"Choose one of: {', '.join(SPEC_SECTIONS)}")
    elif not section and not forced:
        # Waived when the spec declares `data_source: outside_notes`: the runtime forces this
        # status for every chart there and already knows the section, so demanding one from
        # the agent would reject a run for failing to explain a decision it did not make.
        missing.append("name the part of the specification that does not cover this case in "
                       "spec_section, one of: " + ", ".join(SPEC_SECTIONS))
    if not str(submitted.get("reasoning") or "").strip():
        missing.append("say in your own words what the specification fails to cover. A bare "
                       "status code cannot be acted on by anyone.")
    quote = str(submitted.get("spec_quote") or "").strip()
    if quote and _norm_ws(quote) not in _norm_ws(spec.as_prompt_block()):
        missing.append(f"spec_quote {quote[:120]!r} does not appear in the specification you "
                       "were given. Quote it verbatim, or omit spec_quote entirely — "
                       "'no such sentence exists' is a legitimate and useful report.")
    bad_fields = [f for f in (submitted.get("uncovered_fields") or [])
                  if str(f) not in {x.name for x in (spec.fields or [])}]
    if bad_fields:
        missing.append(f"uncovered_fields names {bad_fields}, which this specification does "
                       f"not declare. Its fields are: "
                       f"{[x.name for x in (spec.fields or [])]}")
    if missing:
        return {"accepted": False,
                "why": "SPEC_INSUFFICIENT is a report about the specification and this one "
                       "cannot be routed to the text responsible",
                "missing": missing}
    return {"accepted": True, "why": "", "missing": []}


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").split()).lower()


def gate_answer(spec: ExtractionSpec, submitted: dict, *, evidence: EvidenceLedger,
                coverage: CoverageLedger, chart: PatientChart, tracer=None,
                threads: OpenThreadLedger | None = None,
                plan: CoveragePlan | None = None) -> dict:
    """The single decision on whether an answer may stand. Returns an acceptance dict.

    Also the one place where WHICH SPEC RULE WAS IN PLAY is knowable for certain, so it is
    the one place that writes it down. Three channels, recorded with their provenance marked
    (see `trace`): the checks that fired are deterministic, the admissibility of each citation
    is deterministic, and the rules the agent names are its own report. Every recording here
    is tracer-guarded and side-effect-free — `mcp_server` calls this function with no tracer
    at all, and attribution must never be able to change a verdict.
    """
    status = submitted.get("status", "")
    value = submitted.get("value") or {}
    if tracer:
        # SELF-REPORTED, and taken from the submission whatever the verdict turns out to be:
        # the rules an agent invoked for an answer that was then rejected are the rules that
        # produced the rejected answer, and dropping them would leave exactly the failures
        # the optimizer cares about with no self-report attached.
        tracer.self_reported_rules([submitted.get("rules_applied"),
                                    submitted.get("reasoning")], where="submit_answer")
        # DETERMINISTIC: which evidence rule admitted or refused each cited document. The
        # stratification already computed it; until now it was recomputed by nobody.
        if evidence.items:
            tracer.evidence_admissibility(admissibility_for_citations(
                spec, coverage, evidence.to_list(),
                [k for k, v in value.items() if str(v if v is not None else "").strip()]))
    if not evidence.items and status == "FOUND":
        return {"accepted": False, "why": "no evidence recorded",
                "missing": ["record at least one verbatim quote with record_evidence before answering FOUND"]}

    # BEFORE any status-specific check, and before the coverage gate. An unsettled thread is
    # a question the chart itself raised and nobody answered, so it bears on a positive
    # (the 8046 case: the interim line WAS the citation) and on a negative (the addendum you
    # never opened is the document that would have changed it) alike. SPEC_INSUFFICIENT is
    # exempt for the same reason it is exempt from coverage: it is a claim about the
    # specification, and no amount of chasing a pending stain can make a silent spec speak.
    if status in ("FOUND", "EVIDENCE_INSUFFICIENT"):
        open_threads = check_threads(threads)
        if open_threads:
            if tracer:
                tracer.emit("open_threads_block_answer", severity="warning",
                            n_open=len(open_threads),
                            thread_ids=[t.thread_id for t in threads.unresolved()])
            return {"accepted": False,
                    "why": "a document in this chart deferred its own conclusion and the "
                           "thread was never settled",
                    "how_to_satisfy": ("chase it: page to the end of the same document, then "
                                       "its section list, then later documents of the "
                                       "addendum types. Then resolve_threads in your next "
                                       "reflection, or dismiss_threads with a reason."),
                    "missing": open_threads}
    if status == "FOUND":
        # Positives were previously accepted unchecked, so `gate_validated: True`
        # on a FOUND answer asserted nothing. These are the spec's own decision
        # rules, applied deterministically rather than hoped for in a prompt.
        #
        # `_detail` rather than the message-only forms: the checker knows which rule fired,
        # on which value, over which quote, and that was being discarded one line later.
        # Attribution reconstructed afterwards from a rejection message is a guess, and a
        # guess is what makes an optimizer rewrite the wrong sentence.
        detail = check_field_formats_detail(spec.fields, value)
        detail += check_answer_detail(getattr(spec, "answer_checks", []) or [],
                                      value, evidence.to_list(), coverage.searched_terms)
        violations = [v.message for v in detail]
        if tracer:
            # Called on clean evaluations too. A rejection streak is only measurable if the
            # evaluations where the rule did NOT fire are observed as well.
            tracer.answer_check_outcome(detail, status=status)
        if violations:
            if tracer:
                tracer.emit("answer_check_failed", severity="warning", violations=violations,
                            rejected_by=sorted({v.rule_id for v in detail}))
            return {"accepted": False,
                    "why": "the answer contradicts the specification's decision rules",
                    "missing": violations}
    if status == "SPEC_INSUFFICIENT":
        # NOT sent to the coverage gate. SPEC_INSUFFICIENT is a statement about the
        # SPECIFICATION, not about this chart, and no amount of reading the chart can
        # discharge it — see the status table in `answer_contract`. What it owes
        # instead is a report the improvement loop can route on, checked here so the
        # agent gets a recoverable rejection through the loop it already understands
        # rather than a crash after the run is over.
        return _gate_spec_insufficient(spec, submitted, tracer=tracer)
    if status == "EVIDENCE_INSUFFICIENT":
        # Runtime-forced validation sampling. Drawn by the sampler, never by the agent:
        # a model choosing which unread documents to check is validating its own
        # judgement with its own judgement.
        # Credit whatever the agent has already read against the outstanding draw before
        # deciding it still owes anything.
        kw_hits = keyword_hits_among_drawn(spec, coverage, chart)
        coverage.resolve_sample_verdicts(evidence.cited_notes(), kw_hits)
        pending = coverage.pending_samples()
        if pending:
            lines = []
            for stratum, docs in pending.items():
                for d in docs:
                    lines.append(f"  {stratum}: {d.note_id} ({d.doc_type}, {d.date})")
            if tracer:
                tracer.emit("forced_sampling", seed=coverage.sampler.seed,
                            counts={k: len(v) for k, v in pending.items()})
            ids = [d.note_id for docs in pending.values() for d in docs]
            return {"accepted": False,
                    "why": "validation sampling not yet done — the runtime has drawn these",
                    "how_to_satisfy": ("call read_documents_batch with note_ids set to the "
                                       "list below, in one step; then record_evidence for "
                                       "any that turn out to be relevant, then resubmit"),
                    "note_ids": ids,
                    "missing": ["these were drawn by the runtime, not chosen by you:"] + lines}
        gate = check_gate(spec, coverage, plan)
        if gate.verdict != "PASS":
            # IS ANYTHING LEFT THAT MORE WORK COULD FIX? If every remaining failure is terminal,
            # refusing again asks the agent for something that does not exist. On the ten-patient
            # real batch of 2026-07-28 that produced this run: rejections 1-2 were sampling
            # draws (satisfied), 5 was a thread (settled), and 3,4,6,7,8 were this line, with the
            # unread-hit count going 13 -> 13 -> 8 -> 3 -> 0 as the agent did the work. The last
            # two failures were an exclusion hit and an elusion bound over cap, neither of which
            # any further reading can change. The ledgers froze at [4, 3, 3], the loop brake
            # fired, and the agent submitted SPEC_INSUFFICIENT -- a claim that the SPECIFICATION
            # is inadequate -- because that was the only exit it had. It was the wrong claim: the
            # spec was fine, this chart's stratification was not.
            #
            # So the abstention is ACCEPTED and labelled. `attach_coverage_claim` will see
            # gate_validated=False, attach no coverage ledger, set route_to_human, and record
            # `negative_basis: COVERAGE_UNREACHABLE`. The answer is "I could not establish the
            # value AND I cannot prove I looked hard enough", which is the truth, and it is
            # distinguishable downstream from both GATE_VALIDATED and SPEC_INSUFFICIENT.
            remaining = [m for m in gate.missing if m not in set(gate.terminal)]
            if gate.terminal and not remaining:
                if tracer:
                    tracer.emit("coverage_unreachable", terminal=list(gate.terminal),
                                n_missing=len(gate.missing))
                return {"accepted": True, "why": "", "missing": [],
                        "coverage_unreachable": list(gate.terminal)}
            return {"accepted": False,
                    "why": "the proof obligation for asserting absence is not yet met",
                    # WHAT WOULD ACTUALLY HELP, first. A terminal item mixed into the list reads
                    # as another instruction and the agent retries it; saying which ones are
                    # dead ends is the difference between eight rejections and three.
                    "how_to_satisfy": (
                        "do the items under `missing`. Any listed under `cannot_be_satisfied_in_"
                        "this_run` are dead ends — do not retry them; they mean this chart's "
                        "stratification was wrong and a different run is needed. If they are the "
                        "ONLY thing left, submit the same answer again and it will be accepted "
                        "and routed to a human."),
                    "missing": remaining or list(gate.missing),
                    "cannot_be_satisfied_in_this_run": list(gate.terminal)}
    return {"accepted": True, "why": "", "missing": []}
