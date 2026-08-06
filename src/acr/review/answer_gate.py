"""The single decision on whether a SUBMITTED answer may stand, returned as a recoverable
rejection rather than raised.

THERE IS EXACTLY ONE GATE, AND THIS IS IT
-----------------------------------------
These are module-level functions and not methods on `ChartReviewAgent` because three front
ends must apply the SAME judgement: the langgraph loop (`graph.ChartReviewAgent._gate`) and
the deepagents runtime (`deep_runner`). An MCP server was a third until 2026-08-05. Two gate
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

from ..chartstore.corpus import PatientChart
from ..contract import outcomes as OUTCOMES
from ..contract.answer_checks import (
    check_answer_detail,
    check_dates_not_after,
    check_field_formats_detail,
)
from ..contract.answer_contract import SPEC_SECTIONS
from ..contract.spec import ExtractionSpec
from ..core.case_context import CaseContext
from ..core.state import EvidenceLedger
from .coverage import (
    CoverageLedger,
    GateResult,
    admissibility_for_citations,
    evaluate_gate,
)
from .coverage_planner import CoveragePlan, OpenThreadLedger, marker_blocks_answer
from .runtime_profiles import (
    ALWAYS_COVERAGE_PROFILE,
    CONDITIONAL_COVERAGE_PROFILE,
    COVERAGE_ALWAYS,
    COVERAGE_ON_NEGATIVE_OR_MISSING,
    DEFAULT_RUNTIME_PROFILE,
    STRATIFIED_COVERAGE_PROFILE,
    coverage_requirement,
    resolve_runtime_policy,
    targeted_negative_basis,
)


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
    # THE REQUIRED-SEARCH REFUSALS ARE GONE. Two loops used to run here: one over the spec's
    # `required_keywords` and one over the plan's expanded list, each refusing the answer for a
    # term the agent had not searched.
    #
    # The baseline design gives the model no keyword list at all. It has a `search` tool, it
    # works out what to look for, and it calls it -- so there is no declared list for a gate to
    # discharge, and a term the model chose not to run is a retrieval decision, not a contract
    # violation. Where a list exists at all it is a measured, development-set artifact offered
    # as a prior, and a prior the model may decline is not a gate.
    #
    # The list that was being enforced here was also measured wrong. From this spec's own
    # provenance: the five terms recall 87.4% of answer-bearing documents over 276,054
    # documents, missing at least one for 31.7% of patients, because the list has `carcinoma`
    # and not `cancer`. And `fit_terms_to_budget` was deleting the model's own proposals --
    # `lobe` and `bronchus` among them -- which `nos_requires_search` then refused the answer
    # for not having searched.
    if not coverage.listed_documents:
        g.missing.append("must list the patient's documents before asserting absence")
    g.verdict = "PASS" if not g.missing else "FAIL"
    return g


def check_threads(threads: OpenThreadLedger | None) -> list[str]:
    """Unsettled threads that may REFUSE an answer. Only the computed ones.

    THIS WAS THE 8046 ERROR, wired as a control: a histology coded off a line reading "special
    stains pending" while the addendum that resolved it sat 353 characters past where the read
    stopped. Every open thread refused the answer, and the only ways past were a resolution or
    a dismissal with a stated reason.

    Measured over every recorded trace on 2026-07-30, that control refused 39 times and 11 of
    those refusals (28%) rejected a tuple that was exactly the registry's answer. The markers
    behind them:

        truncated 111    addendum 40    in consultation 10    additional sections 9
        pending 8        outside facility 5    clinical correlation 4    others 7

    All but `truncated` are substrings scanned across document text, parsed out of a Markdown
    table in a skill file. `addendum` refused 40 times while `read_section("ADDENDUM")` could
    address that heading in 0 of the 2,401 documents containing the word: an obligation whose
    tool could never reach its target.

    So the text-matched markers no longer block. They are still detected, still opened as
    threads, still rendered into the prompt by `OpenThreadLedger.render()` with the settling
    call and thread_id already filled in, and still recorded in the manifest -- which is the
    advisory channel, and it was always there. What went away is the refusal.

    `truncated` still blocks, and the difference is not a matter of degree. It is COMPUTED from
    the character counts of the run's own read against the length that read reported: a fact
    about what this run did rather than a guess about what a word means. It cannot be wrong
    about the corpus, and the agent discharges it by reading to the end -- which
    `OpenThreadLedger` does for it automatically once the document is fully read.
    """
    if threads is None:
        return []
    return [f"unsettled thread {t.thread_id}: {t.marker!r} in {t.note_id} "
            f"({t.doc_type or 'unknown type'}) — {t.obligation}. Resolve it (say where "
            f"it was settled) or dismiss it with a reason; both are recorded."
            for t in threads.unresolved() if marker_blocks_answer(t.marker)]


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


def negative_claim_reasons(
    spec: ExtractionSpec, status: str, value: dict
) -> list[str]:
    """Return the field-level reasons a submission makes a negative-shaped claim.

    ``FOUND`` is not necessarily wholly positive.  A null field and a NOS value both assert
    that something more specific was not established.  The conditional arm must notice
    those claims instead of branching only on the case-level status.
    """
    reasons: list[str] = []
    # BY KIND, not by the literal. `CORPUS_INSUFFICIENT` is the same claim about the same
    # chart as `EVIDENCE_INSUFFICIENT` -- an absence -- and an absence is only true if the
    # chart was searched. A literal test would hand every status a contract adds a free pass
    # through the coverage machinery, which is a wider outcome space and a hole in the gate
    # arriving as one change.
    kind = OUTCOMES.status_kind(spec, status)
    if kind == OUTCOMES.KIND_ABSTAIN_EVIDENCE:
        reasons.append(f"case_status:{status}")
    if kind == OUTCOMES.KIND_VALUE:
        for field in spec.fields:
            raw = value.get(field.name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                reasons.append(f"missing_field:{field.name}")
    # THE `nos_or_unknown` BRANCH IS GONE, and its absence is the point.
    #
    # It inferred "this answer claims something was not established" from the SHAPE of the
    # value: if the coded value appeared in some answer_check's `nos_values`, the submission
    # was treated as a negative claim and the coverage machinery activated against it.
    #
    # A NOS code is a conclusion, not a confession. On this corpus 8000/8010/8046 together are
    # the registry's own answer for 10.8% of patients and C349 for 9.6% -- an abstraction rule
    # frequently resolves to exactly these codes, and `conflict_requires_nos` used to ORDER the
    # agent toward one of them. So the same value could be demanded by one rule and then treated
    # as evidence of an unproven absence by this one. Measured consequence, recorded in
    # COVERAGE_THREE_ARM_PILOT.md: one run submitted the conflict-resolving gold answer exactly
    # ten times and was rejected into a model-call-limit failure.
    #
    # A missing field is still a negative claim -- there is no value there at all, which is a
    # fact about the submission and not an inference from a code table. That half stays.
    return list(dict.fromkeys(reasons))


def _coverage_verdict(
    spec: ExtractionSpec,
    *,
    evidence: EvidenceLedger,
    coverage: CoverageLedger,
    chart: PatientChart,
    plan: CoveragePlan | None,
    tracer=None,
    activation_reasons: list[str],
) -> dict:
    """Evaluate the one coverage proof for either a negative or experimental positive."""
    kw_hits = keyword_hits_among_drawn(spec, coverage, chart)
    coverage.resolve_sample_verdicts(evidence.cited_notes(), kw_hits)
    pending = coverage.pending_samples()
    if pending:
        lines = []
        for stratum, docs in pending.items():
            for d in docs:
                lines.append(f"  {stratum}: {d.note_id} ({d.doc_type}, {d.date})")
        if tracer:
            tracer.emit(
                "forced_sampling",
                seed=coverage.sampler.seed,
                counts={k: len(v) for k, v in pending.items()},
                activation_reasons=activation_reasons,
            )
        ids = [d.note_id for docs in pending.values() for d in docs]
        return {
            "accepted": False,
            "why": "validation sampling not yet done — the runtime has drawn these",
            "how_to_satisfy": (
                "call read_documents_batch with note_ids set to the list below, in one "
                "step; then record_evidence for any that turn out to be relevant, then "
                "resubmit"
            ),
            "note_ids": ids,
            "missing": ["these were drawn by the runtime, not chosen by you:"] + lines,
            "coverage_activated": True,
            "coverage_activation_reasons": activation_reasons,
        }
    gate = check_gate(spec, coverage, plan)
    if gate.verdict != "PASS":
        remaining = [m for m in gate.missing if m not in set(gate.terminal)]
        if gate.terminal and not remaining:
            if tracer:
                tracer.emit(
                    "coverage_unreachable",
                    terminal=list(gate.terminal),
                    n_missing=len(gate.missing),
                    activation_reasons=activation_reasons,
                )
            return {
                "accepted": True,
                "why": "",
                "missing": [],
                "coverage_unreachable": list(gate.terminal),
                "coverage_claim_earned": False,
                "coverage_activated": True,
                "coverage_activation_reasons": activation_reasons,
            }
        return {
            "accepted": False,
            "why": "the configured coverage proof is not yet met",
            "how_to_satisfy": (
                "do the items under `missing`. Any listed under "
                "`cannot_be_satisfied_in_this_run` are dead ends — do not retry them; "
                "they mean this chart's stratification was wrong and a different run is "
                "needed. If they are the ONLY thing left, submit the same answer again and "
                "it will be accepted and routed to a human."
            ),
            "missing": remaining or list(gate.missing),
            "cannot_be_satisfied_in_this_run": list(gate.terminal),
            "coverage_activated": True,
            "coverage_activation_reasons": activation_reasons,
        }
    return {
        "accepted": True,
        "why": "",
        "missing": [],
        # WHAT THE LEDGER OBSERVED BUT DID NOT REFUSE. `evaluate_gate` is advisory by default as
        # of 2026-07-30, so these are the sentences that used to be `missing`: how much of each
        # stratum was reviewed, which required searches never ran, what residual bound the draws
        # earn. They must travel with the acceptance, because an advisory nobody is shown is
        # indistinguishable from an advisory nobody generated -- and the whole point of moving
        # this judgement to the model is that the model can see what the runtime counted.
        # Recorded in the manifest too, so "decided the chart was adequately searched" and "never
        # looked" stay distinguishable after the run.
        "advisories": list(gate.advisories),
        "coverage_claim_earned": True,
        "coverage_activated": True,
        "coverage_activation_reasons": activation_reasons,
    }


def _unwitnessable_after(case: CaseContext | None, chart: PatientChart):
    """The upper bound on any date this case could carry.

    From the case context when the caller supplied one — it may know an extract date the chart
    cannot show — and otherwise derived from the chart itself, because the bound exists whether
    or not anybody thought to pass it and a check that only runs when configured is a check
    that does not run.
    """
    if case is not None:
        return case.unwitnessable_after()
    try:
        docs, _ = chart.list_documents(limit=100_000)
    except Exception:  # noqa: BLE001 - a bound we cannot compute is simply absent
        return None
    dates = [d.date for d in docs if getattr(d, "date", None) is not None]
    return max(dates) if dates else None


def gate_answer(spec: ExtractionSpec, submitted: dict, *, evidence: EvidenceLedger,
                coverage: CoverageLedger, chart: PatientChart, tracer=None,
                threads: OpenThreadLedger | None = None,
                plan: CoveragePlan | None = None,
                coverage_plan: CoveragePlan | None = None,
                coverage_state: dict | None = None,
                case: CaseContext | None = None,
                runtime_profile: str = DEFAULT_RUNTIME_PROFILE) -> dict:
    """The single decision on whether an answer may stand. Returns an acceptance dict.

    Also the one place where WHICH SPEC RULE WAS IN PLAY is knowable for certain, so it is
    the one place that writes it down. Three channels, recorded with their provenance marked
    (see `trace`): the checks that fired are deterministic, the admissibility of each citation
    is deterministic, and the rules the agent names are its own report. Every recording here
    is tracer-guarded and side-effect-free: a caller may pass no tracer at all, and attribution
    must never be able to change a verdict.
    """
    profile_asset, _ = resolve_runtime_policy(runtime_profile)
    status = submitted.get("status", "")
    value = submitted.get("value") or {}
    # THE FALL-THROUGH, CLOSED. Every branch below tests a KIND, and a status the contract
    # does not declare resolves to no kind -- so before this line it matched nothing, reached
    # the unconditional `accepted: True` at the end of this function, and stood having proved
    # nothing at all. The most permissive outcome in the system was the one nobody wrote down.
    kind = OUTCOMES.status_kind(spec, status)
    if kind is None:
        return {"accepted": False,
                "why": "that is not an outcome this specification declares",
                "missing": OUTCOMES.undeclared_status_message(spec, status)}
    if tracer:
        # ONE EVALUATION, counted once, before any branch can return early. It used to be
        # counted inside `answer_check_outcome`, which was called unconditionally so a rejection
        # streak could observe its gaps; that method now runs only when something fired, so the
        # count had silently become a count of rejections. Every `return` below is an evaluation
        # that happened.
        tracer.note_gate_evaluation()
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
    if not evidence.items and kind == OUTCOMES.KIND_VALUE:
        return {"accepted": False, "why": "no evidence recorded",
                "missing": ["record at least one verbatim quote with record_evidence before answering FOUND"]}

    # BEFORE any status-specific check, and before the coverage gate. An unsettled thread is
    # a question the chart itself raised and nobody answered, so it bears on a positive
    # (the 8046 case: the interim line WAS the citation) and on a negative (the addendum you
    # never opened is the document that would have changed it) alike. SPEC_INSUFFICIENT is
    # exempt for the same reason it is exempt from coverage: it is a claim about the
    # specification, and no amount of chasing a pending stain can make a silent spec speak.
    # THREADS ARE ADVISORY. This block refused a positive or an evidence-abstention while a
    # blocking thread was unsettled. It went soft on 2026-08-06, and the reasoning is the arc this
    # file already records: `BLOCKING_MARKERS` had gone from ~21 to `truncated` alone, and
    # `truncated` is discharged AUTOMATICALLY by `OpenThreadLedger` the moment the document is read
    # to its end. What the refusal actually caught was therefore a run that chose not to finish a
    # document — and the answer to that is a recorded gap, not a refusal the agent must argue past
    # with a tool built for the purpose.
    #
    # The thread is still opened, still counted, still in the manifest, and still named in the
    # prompt beside what it obliges. Only the refusal is gone.
    if kind in (OUTCOMES.KIND_VALUE, OUTCOMES.KIND_ABSTAIN_EVIDENCE) and tracer:
        # Recorded, never refused: the trace still carries which declared shapes the answer
        # missed, so the eval plane can count them without re-deriving the spec.
        #
        # NOT through `tracer.answer_check_outcome`. That method emits a `rule_rejection` event
        # and drives the consecutive-rejection streak counter -- machinery about refusals, and
        # nothing here is refused. Logging a rejection for an answer that was accepted is how a
        # manifest starts lying about its own run. Its own event kind, so the eval plane can
        # count instruction-following misses and the rejection accounting stays about rejections.
        shape_detail = check_field_formats_detail(getattr(spec, "fields", []) or [], value)
        if shape_detail:
            tracer.emit(
                "answer_shape_miss",
                severity="warning",
                provenance="DETERMINISTIC",
                source="acr.contract.answer_checks.check_field_formats_detail",
                status=status,
                refused=False,
                violations=[row.to_dict() for row in shape_detail],
                rules=sorted({row.rule_id for row in shape_detail}),
                note=("the declared format/allowable_values are rendered into the prompt; a "
                      "populated field that misses them is an instruction-following failure, "
                      "measured here rather than refused"),
            )
    if kind == OUTCOMES.KIND_VALUE:
        # VALUES NO CASE COULD WITNESS. Two arithmetic checks, both refused rather than
        # recorded, and the reason they may refuse when the format check may not is that no
        # correct answer can fail them. 29 February 2018 is not a notation dispute; a date
        # after the last document in the record has no possible citation. `C34.9` — the
        # refusal that cost this repo twelve correct answers — is a notation dispute, and it
        # stays advisory two blocks above.
        impossible = [v for v in check_field_formats_detail(getattr(spec, "fields", []) or [],
                                                            value)
                      if v.rule_kind == "field_calendar"]
        impossible += check_dates_not_after(getattr(spec, "fields", []) or [], value,
                                            _unwitnessable_after(case, chart))
        if impossible:
            if tracer:
                tracer.emit("impossible_value_refused", severity="warning",
                            provenance="DETERMINISTIC",
                            violations=[v.to_dict() for v in impossible],
                            rules=sorted({v.rule_id for v in impossible}))
            return {"accepted": False,
                    "why": "the answer carries a value no record could witness",
                    "missing": [v.message for v in impossible]}
        # Positives were previously accepted unchecked, so `gate_validated: True`
        # on a FOUND answer asserted nothing. These are the spec's own decision
        # rules, applied deterministically rather than hoped for in a prompt.
        #
        # `_detail` rather than the message-only forms: the checker knows which rule fired,
        # on which value, over which quote, and that was being discarded one line later.
        # Attribution reconstructed afterwards from a rejection message is a guess, and a
        # guess is what makes an optimizer rewrite the wrong sentence.
        # `check_answer_detail` returns [] unconditionally -- `ANSWER_CHECK_KINDS` is empty and a
        # spec that declares one of the removed kinds fails to load. So this block used to call
        # `tracer.answer_check_outcome([])` on every evaluation, which EMITTED A `rule_rejection`
        # EVENT WITH ZERO VIOLATIONS and bumped the consecutive-rejection accounting, for a
        # channel with no rules left in it. That is a manifest reporting gate activity that did
        # not happen.
        #
        # The call is still made, because a spec loaded from a future kind would need it, and
        # because deleting the call and the function together is how the two get out of step.
        # What is gone is the unconditional trace write.
        detail = check_answer_detail(getattr(spec, "answer_checks", []) or [],
                                     value, evidence.to_list(), coverage.searched_terms)
        if detail:
            if tracer:
                tracer.answer_check_outcome(detail, status=status)
                tracer.emit("answer_check_failed", severity="warning",
                            violations=[v.message for v in detail],
                            rejected_by=sorted({v.rule_id for v in detail}))
            return {"accepted": False,
                    "why": "the answer contradicts the specification's decision rules",
                    "missing": [v.message for v in detail]}
    if kind == OUTCOMES.KIND_ABSTAIN_SPEC:
        # NOT sent to the coverage gate. SPEC_INSUFFICIENT is a statement about the
        # SPECIFICATION, not about this chart, and no amount of reading the chart can
        # discharge it — see the status table in `answer_contract`. What it owes
        # instead is a report the improvement loop can route on, checked here so the
        # agent gets a recoverable rejection through the loop it already understands
        # rather than a crash after the run is over.
        verdict = _gate_spec_insufficient(spec, submitted, tracer=tracer)
        verdict["coverage_claim_earned"] = False
        return verdict
    requirement = coverage_requirement(profile_asset.ref)
    reasons: list[str] = []
    if requirement == COVERAGE_ALWAYS and kind in (OUTCOMES.KIND_VALUE,
                                                   OUTCOMES.KIND_ABSTAIN_EVIDENCE):
        reasons = [f"profile:{ALWAYS_COVERAGE_PROFILE}"]
    elif requirement == COVERAGE_ON_NEGATIVE_OR_MISSING:
        # Preserve the historical profile's old branch exactly; the new conditional profile
        # additionally recognises partial and NOS-shaped FOUND answers.
        reasons = (
            negative_claim_reasons(spec, status, value)
            if profile_asset.module_id == CONDITIONAL_COVERAGE_PROFILE
            else (
                [f"case_status:{status}"]
                if profile_asset.module_id == STRATIFIED_COVERAGE_PROFILE
                and kind == OUTCOMES.KIND_ABSTAIN_EVIDENCE
                else []
            )
        )
    if reasons:
        state = coverage_state if coverage_state is not None else {}
        if not state.get("active"):
            state.update(
                {
                    "active": True,
                    "reason": reasons,
                    "trigger_status": status,
                }
            )
            if tracer:
                tracer.emit(
                    "coverage_activated",
                    runtime_profile=profile_asset.ref,
                    trigger_status=status,
                    reasons=reasons,
                )
        return _coverage_verdict(
            spec,
            evidence=evidence,
            coverage=coverage,
            chart=chart,
            plan=coverage_plan or plan,
            tracer=tracer,
            activation_reasons=reasons,
        )
    if kind == OUTCOMES.KIND_ABSTAIN_EVIDENCE:
        # Targeted search is deliberately not a coverage claim.  It still has to establish
        # the patient-level document universe so "nothing found" is not a run that never
        # looked at the chart at all.
        if not coverage.listed_documents:
            return {
                "accepted": False,
                "why": "the targeted-search arm has not established its patient inventory",
                "missing": [
                    "list the patient's documents before concluding that targeted search "
                    "found no admissible evidence"
                ],
                "coverage_claim_earned": False,
            }
        return {
            "accepted": True,
            "why": "",
            "missing": [],
            "coverage_claim_earned": False,
            "negative_basis": targeted_negative_basis(profile_asset.ref),
        }
    return {
        "accepted": True,
        "why": "",
        "missing": [],
        "coverage_claim_earned": False,
    }
