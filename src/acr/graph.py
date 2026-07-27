"""The agent: plan -> act -> reflect -> (act | replan | finalize).

Why a graph and not a plain ReAct loop
--------------------------------------
A plain loop decides "what next" inside the same generation that just read a document,
which makes the stopping decision an afterthought. Here `reflect` is a separate node with
one job: look at what has actually been gathered and rule CONTINUE / REPLAN / SUFFICIENT /
STUCK. Replanning is therefore a first-class, traceable event rather than a drift in the
model's internal monologue.

THE PLAN IS THE COVERAGE PLAN, AND THERE IS ONLY ONE
----------------------------------------------------
This module used to carry a second plan: a prose list of {id, goal, rationale} produced by
a PLAN_PROMPT, rendered into the message list, and read by no code anywhere. Two greps
settled it —

    grep 's["plan"]'                 src/acr/graph.py   -> nothing
    grep 'CoveragePlan|policy_for'   src/acr/graph.py   -> nothing

— so the plan the agent could revise governed nothing, and the plan that governed retrieval
was never consulted by the loop. That is why REPLAN fired 0 times in 291 actions across 37
runs. REPLAN and CONTINUE were mechanically identical: both appended text. A model asked
"does something learned change what should be done next?" about a goal like "find the
pathology report" correctly answers no — the GOAL never changes. What changes is the
RETRIEVAL SCOPE, and the retrieval scope was not in the plan.

So: the prose plan is deleted, `coverage_planner.CoveragePlan` is built once up front, it is
rendered into the agent's messages, it is ENFORCED in `_n_act` (a `sample` type may not be
opened at all unless the runtime's sampler drew it), and it is what reflection revises —
monotonically, in a typed object the runtime applies. REPLAN is no longer a verdict the
model may pick; it is recorded by the runtime when, and only when, a revision actually
changed what the agent may open or must search.

Why the answer is gated in code
-------------------------------
`submit_answer` does not end the run. It is validated: a negative or absent answer is
rejected unless the spec's proof obligation is satisfied by the *computed* coverage ledger.
The rejection, with its reasons, is fed back to the model as an observation. Prompting a
model to "be sure you looked everywhere" is a wish; checking the ledger is a control.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .corpus import PatientChart
from .answer_checks import check_answer_detail, check_field_formats_detail
from .llm import LLMClient, extract_json
from .spec import ExtractionSpec
from .coverage import (SEED_CALLER, SEED_DERIVED, CoverageLedger, ForcedSampler, GateResult,
                       admissibility_for_citations, derive_sample_seed, evaluate_gate,
                       keyword_was_searched, strata_from_spec)
from .coverage_planner import (MONOTONICITY_VS_LEDGER, REFUSED_BUDGET,
                               TRIGGER_UNSETTLED_THREAD, TRIGGERS,
                               CoveragePlan, ExpansionBudget,
                               OpenThreadLedger, PlanRevision, RevisionOutcome, Trigger,
                               documents_by_type, gate_obligation_triggers,
                               load_marker_catalogue, plan_coverage, plan_from_spec,
                               triggers_from_tool_result)
from .state import Budget, EvidenceLedger, RunState
from .tools import Toolbox
from .trace import Tracer, parse_rule_citations, rule_catalog, rule_citation_block

#: The routing vocabulary of `_after_reflect`. REPLAN is still in it and still routes to the
#: plan node, but it is now DERIVED BY THE RUNTIME from an applied revision rather than
#: chosen by the model — see `_n_reflect`. That is the whole fix: when REPLAN was something a
#: supervisor could assert, asserting it changed nothing, so it was never asserted.
VERDICTS = {"CONTINUE", "REPLAN", "SUFFICIENT", "STUCK"}

#: What the model may return. REPLAN is absent on purpose. A model that returns it anyway is
#: recorded and read as CONTINUE: the revision object, not the word, is what replans.
MODEL_VERDICTS = {"CONTINUE", "SUFFICIENT", "STUCK"}

#: Tool calls the retrieval plan is allowed to refuse. Reading only. Searching a `sample`
#: type stays legal — the plan says what must be READ, and a cheap metadata-level search that
#: turns up a hit is exactly the observation that should trigger a promotion.
PLAN_GOVERNED_TOOLS = ("read_document", "read_section", "read_documents_batch")

# ==========================================================================================
# WHAT A COVERAGE CLAIM MEANS, PER STATUS
# ==========================================================================================
# A coverage ledger says one thing: "I searched the universe this spec defines." Whether an
# answer may say that — and whether it MUST — depends on what the answer is a claim ABOUT.
#
#   FOUND                  a claim about this chart, proved by WITNESS. One qualifying
#                          document settles it and the gate checks exactly that, so a
#                          coverage claim would advertise a search nobody verified. Carries
#                          proof_basis, never coverage_attested.
#
#   EVIDENCE_INSUFFICIENT  a claim about this chart that IS a claim about coverage — "the
#                          chart does not support an answer" is only true if the chart was
#                          searched. So it needs the gate, and once gated it must carry the
#                          ledger, or the claim is unauditable.
#
#   SPEC_INSUFFICIENT      NOT a claim about this chart at all. It says "your SPECIFICATION
#                          does not cover this case." Coverage of the chart is beside the
#                          point: searching every document in the record cannot make a
#                          silent spec speak. Demanding a coverage proof for it is a
#                          CATEGORY ERROR, and that category error is what crashed the run —
#                          finalize attached the ledger to every non-FOUND status, then
#                          assert_coverage_claim_is_earned correctly refused it and raised,
#                          and cli.py caught the exception, so the run left a trace and no
#                          manifest. Across 38 real runs SPEC_INSUFFICIENT was reported zero
#                          times and that was written up as the model under-using its
#                          abstention channel. The channel could not be used at all.
#
# So SPEC_INSUFFICIENT is exempted from the coverage gate — and, because a status code with
# nothing attached is useless to the optimizer that reads it, it owes a DIFFERENT proof: the
# spec_gap block below, enforced at emission by assert_spec_gap_is_reported. Exempting it
# from one obligation without imposing the other would turn the crash into a shrug.
# ==========================================================================================

#: The parts of a specification an agent may name as inadequate. Closed, and closed against
#: the SPEC's own structure (see `spec.as_prompt_block`) rather than against free prose: the
#: optimizer in refine.py routes on which text parameter to edit, and "the bit about staging"
#: is not a destination. `data_source` is here because the runtime itself forces
#: SPEC_INSUFFICIENT for `data_source: outside_notes`, and that report needs a section too.
SPEC_SECTIONS = (
    "decision_rule", "evidence_rules", "conflict_rules", "when_not_to_use",
    "boundary_cases", "abstention", "fields", "proof_obligation", "data_source",
)

#: Not a section. The label for a report that named none, which happens on exactly one path:
#: `finalize` authors the answer itself when the agent never called submit_answer, and there
#: is no loop left to reject it into. Recorded as unroutable and counted as degradation
#: rather than crashed on or quietly assigned a plausible section — a guessed destination is
#: worse than a missing one, because the optimizer would go and edit it.
SPEC_SECTION_UNATTRIBUTED = "unattributed"
_REPORTABLE_SECTIONS = SPEC_SECTIONS + (SPEC_SECTION_UNATTRIBUTED,)

#: Why the spec could not answer. Deliberately SMALL, and deliberately not refine.py's
#: verdict vocabulary (SPEC_GAP / SPEC_AMBIGUITY / SPEC_ERROR). Those are the reflection
#: model's classification of a gradient; these three are decidable by the runtime without a
#: model. Duplicating the verdict names here would create a second vocabulary free to drift
#: from the one the optimizer actually branches on.
REMEDY_WRONG_DATA_SOURCE = "WRONG_DATA_SOURCE"   # runtime-forced: the source cannot carry it
REMEDY_CASE_EXCLUDED = "CASE_EXCLUDED"           # the spec says outright it does not apply
REMEDY_SPEC_DOES_NOT_COVER = "SPEC_DOES_NOT_COVER"   # the spec is silent or unclear here
REMEDY_CLASSES = (REMEDY_WRONG_DATA_SOURCE, REMEDY_CASE_EXCLUDED, REMEDY_SPEC_DOES_NOT_COVER)


class CoverageClaimError(AssertionError):
    """An answer advertised a coverage claim it did not earn."""


class SpecGapError(AssertionError):
    """A SPEC_INSUFFICIENT answer was emitted without the report that makes it usable."""


def assert_coverage_claim_is_earned(ans: dict) -> None:
    """`coverage_attested` may appear on exactly one kind of answer.

    A coverage ledger asserts "I searched the defined universe". Only a negative that passed
    the proof obligation has established that. A witness-proved positive never claimed it; a
    give-up and a budget exhaustion never earned it. Attaching the ledger anywhere else makes
    the answer advertise a stronger claim than was verified, and — because it looks exactly
    like a verified one downstream — nothing would catch it.

    Checked at the point of emission rather than left as an intention, since the whole family
    of bugs this guards against consists of intentions that the code did not keep.
    """
    has_ledger = "coverage_attested" in ans
    earned = (ans.get("status") == "EVIDENCE_INSUFFICIENT"
              and ans.get("negative_basis") == "GATE_VALIDATED")
    if has_ledger and not earned:
        raise CoverageClaimError(
            f"coverage_attested attached to status={ans.get('status')!r} "
            f"negative_basis={ans.get('negative_basis')!r} proof_basis={ans.get('proof_basis')!r}; "
            "only a gate-validated EVIDENCE_INSUFFICIENT may carry a coverage claim"
        )
    if earned and not has_ledger:
        raise CoverageClaimError(
            "a gate-validated negative must carry its coverage_attested ledger — "
            "the claim is only auditable if the evidence for it travels with it"
        )


def assert_spec_gap_is_reported(ans: dict) -> None:
    """A SPEC_INSUFFICIENT answer must carry the report the improvement loop routes on.

    This is the mirror of the coverage rule, and it exists for the same reason. Exempting
    SPEC_INSUFFICIENT from the coverage gate is correct, but an exemption on its own leaves
    the channel emitting a bare status code — and a bare status code cannot tell the §6b
    optimizer which field, which sentence, or whether the spec is even at fault. That is a
    channel that reports and cannot be acted on, which is the same defect shape as a check
    that records and cannot refuse.

    It also holds the line the other way: no `value` may ride along. "The spec does not cover
    this" and "here is my answer anyway" are contradictory claims, and the second one would
    otherwise reach L4 as an established fact — `variables_from_answer` promotes any populated
    field to FOUND whatever the answer's status says.
    """
    if ans.get("status") != "SPEC_INSUFFICIENT":
        return
    gap = ans.get("spec_gap")
    if not isinstance(gap, dict) or not gap:
        raise SpecGapError(
            "a SPEC_INSUFFICIENT answer must carry a spec_gap block. This status is the "
            "highest-precision input the spec-improvement loop can receive — the agent "
            "saying, in the one place built for it, that the specification does not cover "
            "this case. Emitting the code alone throws that away."
        )
    if not str(gap.get("reported_by") or "").strip():
        raise SpecGapError("spec_gap is missing 'reported_by'; an agent's report and a "
                           "runtime-forced constant are different signals and must not pool")
    if gap.get("spec_section") not in _REPORTABLE_SECTIONS:
        raise SpecGapError(f"spec_gap.spec_section={gap.get('spec_section')!r} is not one of "
                           f"{list(_REPORTABLE_SECTIONS)}")
    for k in ("agent_words", "agent_words_supplied", "routable"):
        if k not in gap:
            raise SpecGapError(f"spec_gap is missing {k!r}. A report that could not be routed "
                               "must SAY it could not be routed; silence there is what let "
                               "'zero SPEC_INSUFFICIENT in 38 runs' read as a clean result.")
    if ans.get("remedy_class") not in REMEDY_CLASSES:
        raise SpecGapError(f"remedy_class={ans.get('remedy_class')!r} is not one of "
                           f"{list(REMEDY_CLASSES)}")
    if any(str(v).strip() for v in (ans.get("value") or {}).values()):
        raise SpecGapError(
            "a SPEC_INSUFFICIENT answer is carrying a populated value. Declaring the "
            "specification inadequate is not a route past the gate: a coded value that "
            "arrives this way is flattened to FOUND downstream having met no proof standard "
            "at all."
        )


def assert_answer_is_reportable(ans: dict) -> None:
    """Every obligation an answer owes at emission, in one call so no front end owes fewer.

    Three runtimes emit answers (graph, deep_runner, mcp_server). A rule enforced in two of
    them makes the signal silently conditional on which runtime the operator happened to use.
    """
    assert_coverage_claim_is_earned(ans)
    assert_spec_gap_is_reported(ans)


# ------------------------------------------------------- the SPEC_INSUFFICIENT report
#: A spec_section maps to one or more rule-id namespaces in `trace.rule_catalog`. Written out
#: rather than derived by string munging because the two vocabularies are near-misses
#: (`evidence_rules` the section, `evidence_rule.*` the namespace) and a `section + "s"` rule
#: would silently return an empty candidate set the day either name is pluralised.
_SECTION_RULE_NAMESPACES: dict[str, tuple[str, ...]] = {
    "decision_rule": ("decision_rule",),
    "evidence_rules": ("evidence_rule",),
    "conflict_rules": ("conflict_rule",),
    "abstention": ("abstention",),
    "proof_obligation": ("proof_obligation",),
    "fields": ("field_format", "field_allowable_values", "answer_check"),
    # No rule ids exist for these: `when_not_to_use` and `boundary_cases` are prose the
    # catalog does not enumerate, and `data_source` is a scalar. Empty here is a fact about
    # the catalog, and `section_rule_ids_available` says so rather than letting a reader
    # infer that the agent cited nothing.
    "when_not_to_use": (),
    "boundary_cases": (),
    "data_source": (),
    SPEC_SECTION_UNATTRIBUTED: (),
}


def spec_rule_ids(spec: ExtractionSpec, section: str) -> list[str]:
    """Every declared rule identifier in one section of the spec. The candidate set.

    §6b's routing wants to name the RULE, not just the section, and `trace.rule_catalog`
    now mints stable ids for exactly that. This returns the ids that LIVE in the named
    section; which of them the agent actually invoked is a separate question, answered by
    `parse_rule_citations` over the agent's own words and never conflated with this list.
    """
    prefixes = _SECTION_RULE_NAMESPACES.get(section, ())
    if not prefixes:
        return []
    try:
        catalog = rule_catalog(spec)
    except Exception:      # noqa: BLE001 - a spec that cannot enumerate must still report
        return []
    return [r.rule_id for r in catalog
            if r.rule_id.split(".", 1)[0] in prefixes]


def build_spec_gap(spec: ExtractionSpec, submitted: dict, *, reported_by: str,
                   gate_validated: bool) -> tuple[dict, str]:
    """Assemble the (spec_gap, remedy_class) pair for a SPEC_INSUFFICIENT answer.

    One builder for all three front ends. The alternative — each runtime assembling its own —
    is how the same status ends up meaning three different things on disk.
    """
    section = str(submitted.get("spec_section") or "").strip()
    forced = spec.data_source == "outside_notes"
    if forced and section not in SPEC_SECTIONS:
        # The runtime, not the agent, knows why this one abstained, so it names the section
        # itself rather than rejecting an agent that had nothing to name.
        section = "data_source"
    if section not in SPEC_SECTIONS:
        section = SPEC_SECTION_UNATTRIBUTED
    if forced:
        remedy = REMEDY_WRONG_DATA_SOURCE
    elif section == "when_not_to_use":
        # The spec ANTICIPATED this case and excluded it. That is the spec working, not a
        # gap, and filing it as one would have the optimizer "fix" deliberate exclusions.
        remedy = REMEDY_CASE_EXCLUDED
    else:
        remedy = REMEDY_SPEC_DOES_NOT_COVER

    words = str(submitted.get("reasoning") or "").strip()
    if forced and not words:
        words = ("runtime-forced: this spec declares data_source=outside_notes, so no chart "
                 "can answer it and no agent judgement was involved")

    # WHICH RULE. Two different facts, kept apart on purpose. `invoked_rules` is what the
    # agent cited, parsed EXACTLY — a hallucinated identifier is discarded rather than
    # repaired, because a gradient routed at a rule that does not exist would find the spec
    # "silent" about it and propose adding what is already there. `section_rule_ids` is the
    # candidate set the section contains, which is what tells a reader whether citing nothing
    # meant "no rule applied" or "no rule existed to cite".
    #
    # Only the COUNT of misattributions is kept here. `Tracer.rule_attribution` is the
    # authority on those — it counts per identifier and caps the list, so a model emitting a
    # thousand invented ids cannot grow a manifest without bound. A second unbounded copy
    # would defeat that cap and give two numbers that can disagree.
    section_rule_ids = spec_rule_ids(spec, section)
    try:
        known = [r.rule_id for r in rule_catalog(spec)]
    except Exception:      # noqa: BLE001
        known = []
    invoked, misattributed = parse_rule_citations(
        [submitted.get("rules_applied"), submitted.get("reasoning")], known)
    gap = {
        "spec_id": spec.spec_id,
        "spec_hash": spec.spec_hash,
        # WHICH PART of the spec. Closed vocabulary; see SPEC_SECTIONS.
        "spec_section": section,
        # WHICH FIELDS. Empty is a whole-answer claim, said explicitly — an inferred
        # "all of them" would be a runtime guess wearing the agent's authority.
        "uncovered_fields": [str(f) for f in (submitted.get("uncovered_fields") or [])],
        "fields_scope": ("named_fields" if submitted.get("uncovered_fields")
                         else "whole_answer"),
        # THE AGENT'S OWN WORDS, verbatim and unsummarised. §6b's citation mask reads this.
        # The `_supplied` flag is separate because "" and "the agent said nothing" have to be
        # distinguishable — one is a formatting accident, the other is a measurement.
        "agent_words": words,
        "agent_words_supplied": bool(words),
        # The spec sentence it is pointing at, if any. Its ABSENCE is itself informative:
        # refine.py distinguishes a gap (no such sentence exists) from an ambiguity (this
        # sentence, two readings), and it needs the quote to tell them apart.
        "spec_quote": str(submitted.get("spec_quote") or "").strip() or None,
        "invoked_rules": invoked,
        "misattributed_rule_count": len(misattributed),
        "section_rule_ids": section_rule_ids,
        "section_rule_ids_available": bool(section_rule_ids),
        # An agent's report is the signal the loop wants. A runtime-forced one is a constant
        # returned for every chart against this spec — STORE.610 does exactly that by design —
        # and pooling the two would drown 38 runs' worth of real signal in boilerplate.
        "reported_by": reported_by,
        # Recorded, not required. The gate can prove a chart was searched; it has no bearing
        # on whether a specification is silent, so this is context, never a precondition.
        "gate_validated": bool(gate_validated),
        # The one field a consumer must read first. False means the report exists and cannot
        # be acted on; counting those separately is the only way anyone finds out that the
        # channel is degrading again.
        "routable": bool(words) and section in SPEC_SECTIONS,
    }
    return gap, remedy


def strip_value_from_spec_insufficient(ans: dict, tracer=None) -> None:
    """Remove any coded value from a SPEC_INSUFFICIENT answer, loudly.

    Two paths reach here with a value attached, and neither goes through the gate's refusal:
    finalize rewriting a FOUND into SPEC_INSUFFICIENT for `data_source: outside_notes`, and a
    finalize-authored answer produced when the agent never called submit_answer at all.
    Dropping the value silently would be its own small lie, so what was dropped is recorded.
    """
    dropped = {k: v for k, v in (ans.get("value") or {}).items() if str(v).strip()}
    if not dropped:
        ans["value"] = {}
        return
    ans["value"] = {}
    ans["value_withheld"] = sorted(dropped)
    ans["value_withheld_why"] = (
        "SPEC_INSUFFICIENT states the specification cannot decide this case; a value coded "
        "under it met no proof standard and would be read downstream as an established fact"
    )
    if tracer:
        tracer.emit("spec_insufficient_value_withheld", severity="warning",
                    fields=sorted(dropped))


# --------------------------------------------------------------------------- the gate
# Module-level and not methods, because there is now a second front end (the MCP server in
# `mcp_server.py`) that must apply the SAME gate. Two gate implementations that can disagree
# is the `state.py` two-ledger failure repeated one layer up: both would compute a verdict,
# neither would raise when they diverged, and you would be left with two answers and no way
# to choose. `ChartReviewAgent` keeps its `_gate` / `_check_gate` methods as thin delegates.


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
        # discharge it — see the status table at the top of this module. What it owes
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
            return {"accepted": False,
                    "why": "the proof obligation for asserting absence is not yet met",
                    "missing": gate.missing}
    return {"accepted": True, "why": "", "missing": []}


SYSTEM = """You are a cancer-registry abstractor reviewing one patient's chart.

You work exactly the way a careful human abstractor does: first see what documents exist \
and when, then narrow by document type and date, then search, then read only what matters, \
and quote what you find.

Rules that are not negotiable:
- Ground every assertion in a recorded quote. Use record_evidence before you answer.
- Never infer a finding from a document type that cannot establish it. Imaging does not \
establish histology. A restatement in a consult note is weaker than the primary report it \
claims to summarise; when they disagree, cite both and prefer the primary.
- Absence of a mention is not evidence of absence until you have actually looked. The \
specification tells you what "looked" means for this variable.
- If the specification does not cover the case, say SPEC_INSUFFICIENT. If the specification \
is clear but the chart lacks the evidence, say EVIDENCE_INSUFFICIENT. These are different \
answers and choosing the wrong one is an error.
- SPEC_INSUFFICIENT is a report about the SPECIFICATION, so it must name what is wrong with \
it: set spec_section to the part at fault, say in your own words what it fails to cover, \
and quote the sentence you mean if one exists. It must NOT carry a value — if you can code \
a field, answer FOUND for it and prove it.
- The RETRIEVAL PLAN below is binding. Document types it assigns to `sample` are drawn by \
the runtime's sampler and you may not open them yourself; a read of one is refused. That is \
not evidence the type holds nothing. If you find a reason to think it bears on the answer, \
widen the plan at the next supervisor step — the plan may only ever widen, never narrow.
- A document that DEFERS ITS OWN CONCLUSION — "pending", "see addendum", "correlate \
clinically", an outside facility, or a read that came back truncated — has opened a thread, \
and an open thread blocks your answer. Chase it: page to the end of the same document, then \
its section list, then later documents. A deferred conclusion is not evidence for the \
conclusion; it is an instruction about where to look next.

Call one tool at a time and read its result before deciding the next move."""

# NOTE: there is no PLAN_PROMPT any more. The prose plan of {id, goal, rationale} that used
# to live here was read by no code — it was rendered into the message list and that was all.
# The plan is now `coverage_planner.CoveragePlan`, built once up front by
# `coverage_planner.plan_coverage` (or, when no model is available or the planner degrades,
# derived from the spec's own strata by `plan_from_spec`) and rendered by `plan.render()`.
# Keeping both would have left the bug exactly where it was: two plans, one of which matters.

REFLECT_PROMPT = """You are supervising a chart review in progress. Judge only what has \
actually been gathered, and revise the retrieval plan if the observations below say it was \
wrong.

SPECIFICATION QUESTION: {question}

PROOF OBLIGATION FOR A NEGATIVE ANSWER:
{obligation}

THE PLAN (there is only one, and it governs what may be opened):
{plan}

UNSETTLED THREADS:
{threads}

OBSERVATIONS THAT REQUIRE A RESPONSE — these were detected mechanically since the last \
reflection. You are not being asked whether anything happened; these happened. For each one \
either widen the plan or say in `reason` why widening is not warranted:
{triggers}

EVIDENCE RECORDED SO FAR:
{evidence}

COVERAGE SO FAR:
{coverage}

STEPS USED: {step}/{max_steps}
EXPANSION REMAINING: {budget}

HOW THE PLAN MAY BE REVISED — this is enforced in code, not requested:
- ADD a search term. Anything you add becomes a search the gate then REQUIRES to have run.
- PROMOTE a document type toward more reading: sample -> search -> read_all.
- OPEN a thread, or RESOLVE one (say where it was settled) or DISMISS one (say why it does \
not bear on the answer). An unsettled thread blocks submission.
- You may NEVER remove a term, demote a type, or drop a type from the plan. A revision that \
is not a superset of the current plan is REFUSED WHOLE and recorded as refused.

Rule one verdict:
- SUFFICIENT  the recorded evidence already answers the question, or the proof obligation \
for a negative has been met and nothing was found.
- CONTINUE    keep going.
- STUCK       further search is futile; the honest answer is an abstention. Choose this when \
the expansion budget is spent and obligations are still outstanding.
There is no REPLAN verdict. The runtime records a replan when — and only when — your \
revision actually changes what may be opened or what must be searched.

Reply with JSON only:
{{"verdict":"SUFFICIENT|CONTINUE|STUCK","reason":"one or two sentences",
  "revision":{{"add_terms":["..."],
              "promote_types":[{{"type":"<exact type string>","to":"search|read_all"}}],
              "open_threads":[{{"note_id":"...","marker":"...","why":"..."}}],
              "resolve_threads":[{{"thread_id":"...","how":"..."}}],
              "dismiss_threads":[{{"thread_id":"...","reason":"..."}}]}}}}
Send an empty revision object when nothing needs to widen."""

FINALIZE_PROMPT = """{spec_block}

{rule_citations}

You are writing the final answer for patient {patient_id}.

You may use ONLY the evidence below. Anything not in this ledger does not exist for the \
purposes of this answer.

EVIDENCE LEDGER:
{evidence}

COVERAGE ACHIEVED:
{coverage}

{gate_note}

Apply the decision rules to the evidence and produce the answer.

Reply with JSON only:
{{"status":"FOUND|EVIDENCE_INSUFFICIENT|SPEC_INSUFFICIENT",
  "value":{{{value_keys}}},
  "reasoning":"which rules you applied to which evidence",
  "rules_applied":["the identifiers above for the rules you actually used"],
  "evidence_ids":["E1","E2"]}}

If and only if the status is SPEC_INSUFFICIENT, add these and leave every value null — the \
specification cannot both fail to cover the case and decide it:
  "spec_section": one of {spec_sections},
  "spec_quote": the sentence you mean, verbatim from the specification above, or omit it if \
no such sentence exists,
  "uncovered_fields": the output fields it does not cover, or omit for the whole answer"""


class ChartReviewAgent:
    def __init__(
        self,
        spec: ExtractionSpec,
        llm: LLMClient,
        *,
        budget: Budget | None = None,
        reflect_every: int = 3,
        out_dir: str | Path = "runs",
        sample_seed: int | None = None,
        expansion_budget: ExpansionBudget | None = None,
    ):
        self.spec = spec
        self.llm = llm
        self.budget = budget or Budget()
        self.reflect_every = reflect_every
        self.out_dir = Path(out_dir)
        # None does NOT mean "unbounded" and does not mean "some default". It means "price it
        # against the plan", which `ExpansionBudget.priced_against` does with no literal in
        # sight: each cap is "no more than the commitment the plan was already priced at".
        # A caller may override, and the manifest records which of the two happened.
        self.expansion_budget = expansion_budget
        self.expansion_budget_source = ("caller_supplied" if expansion_budget is not None
                                        else "priced_against_plan")
        # Recorded in the trace so an audit can confirm which documents were drawn, and so
        # a run replays deterministically. Two ablation arms must share it to be comparable.
        #
        # None does NOT mean "draw one at random" any more. It means "derive it from
        # (patient, spec_id)", the way `mcp_server` always has — `run()` fills it in once the
        # patient is known. A caller-supplied seed is honoured, but the manifest says so:
        # a seed the caller chose is a seed the caller could have chosen again.
        self.sample_seed = sample_seed
        self.seed_provenance = SEED_CALLER if sample_seed is not None else SEED_DERIVED
        # How many `term_provenance` rows existed before the agent reflected once — i.e. the
        # up-front planner's own proposals. The expansion budget is priced in REFLECTION
        # terms and counted in ALL rows, so the two are reconciled here and nowhere else;
        # see `_price_expansion_budget`. Set in `run()`, defined here because callers that
        # borrow the object without running it (deep_runner, tests) read the budget helpers.
        self._planner_terms = 0
        #: Terms a revision asked for and the budget could not pay for. Kept because partial
        #: application means a term overrun no longer records a refusal, and "the agent hit
        #: the cap" must remain observable — see `_expansion_is_spent`.
        self._terms_deferred: list[str] = []
        self._graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(RunState)
        g.add_node("plan", self._n_plan)
        g.add_node("act", self._n_act)
        g.add_node("reflect", self._n_reflect)
        g.add_node("finalize", self._n_finalize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "act")
        g.add_conditional_edges("act", self._after_act, {"reflect": "reflect", "finalize": "finalize"})
        g.add_conditional_edges("reflect", self._after_reflect,
                                {"act": "act", "plan": "plan", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ the one plan
    def _build_plan(self) -> CoveragePlan:
        """Build the retrieval plan ONCE, before the loop, and record where it came from.

        Up front and frozen-at-birth is the property that makes it auditable: the reviewing
        agent consumes a plan it did not author, and may only widen it. Two sources, and the
        difference is recorded rather than smoothed over — a planner guess must never be
        readable as a curated site binding, which is what `CoveragePlan.source` is for.
        """
        if self.llm is None:
            return plan_from_spec(self.spec, self.chart)
        try:
            p = plan_coverage(self.spec, self.chart, self.llm)
        except Exception as e:      # noqa: BLE001 - a broken planner must not lose the run
            self._plan_fallbacks += 1
            self.tracer.emit("plan_fallback_used", severity="error",
                             error=f"{type(e).__name__}: {e}",
                             message=("the coverage planner raised; falling back to the "
                                      "spec's own strata. The run is NOT exercising the "
                                      "planner and no conclusion about it may be drawn"))
            return plan_from_spec(self.spec, self.chart)
        if not (p.read_all or p.search or p.sample):
            # The planner returned nothing assignable — the reasoning-channel trap that used
            # to be laundered into a one-line prose goal. Loud, countable, and it falls back
            # to a real plan (the spec's strata) rather than to a generic sentence.
            self._plan_fallbacks += 1
            self.tracer.emit("plan_fallback_used", severity="error",
                             message=("the coverage planner produced no usable assignment; "
                                      "falling back to the spec's declared strata"))
            return plan_from_spec(self.spec, self.chart)
        return p

    # ------------------------------------------------------------------ nodes
    def _n_plan(self, s: RunState) -> dict:
        """Render the plan into the working messages. Entered once at START, and again after
        every APPLIED revision — which is what makes a widened scope actually reach the model
        rather than sitting in a Python object nobody shows it."""
        rev = s.get("plan_revisions", 0)
        self.tracer.plan(self.plan.to_dict(), rev)
        # The rule identifiers go in the working prompt, not only in the finalize prompt,
        # because the self-report is collected at submit_answer — and `submit_answer` is
        # reachable from any act step. An agent asked at the last moment to cite identifiers
        # it has never seen will invent them, and we would be measuring our own prompt.
        cite = rule_citation_block(self.spec)
        msgs = s.get("messages") or [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.spec.as_prompt_block()
             + (f"\n\n{cite}" if cite else "")
             + f"\n\nPATIENT: {s['patient_id']}\nBegin. Work the plan; call one tool at a time."},
        ]
        header = ("PLAN (revision %d — the scope was widened; the additions are marked):\n"
                  % rev if rev else "PLAN:\n")
        msgs = msgs + [{"role": "user",
                        "content": header + self.plan.render(self._docs_by_type)}]
        return {"plan": self.plan.to_dict(), "plan_revisions": rev + 1, "messages": msgs}

    def _n_act(self, s: RunState) -> dict:
        msgs = list(s["messages"])
        r = self.llm.chat(msgs, tools=self.toolbox.schemas())
        self.tracer.llm("act", r.content, [c["name"] for c in r.tool_calls], {"total": r.total_tokens})

        if not r.tool_calls:
            msgs.append({"role": "assistant", "content": r.content or ""})
            self._act_no_tool_call += 1
            self.tracer.emit("act_no_tool_call", severity="warning",
                             content_chars=len(r.content or ""),
                             used_reasoning_channel=r.used_reasoning_channel,
                             message="act step produced neither a tool call nor usable text")
            msgs.append({"role": "user", "content":
                         "Continue by calling a tool. If you are ready, call submit_answer."})
            return {"messages": msgs, "step": s.get("step", 0) + 1}

        msgs.append({"role": "assistant", "content": r.content or "",
                     "tool_calls": [{"id": c["id"], "type": "function",
                                     "function": {"name": c["name"],
                                                  "arguments": json.dumps(c["arguments"])}}
                                    for c in r.tool_calls]})
        rejections = list(s.get("rejections", []))
        done = False
        gate_validated = bool(s.get("gate_validated"))
        answer = s.get("answer") or {}

        for c in r.tool_calls:
            refusal = self._plan_refusal(c["name"], c["arguments"] or {})
            if refusal is not None:
                # THE PLAN GOVERNS WHAT MAY BE OPENED. Not as advice in a prompt — as a
                # refusal at dispatch. A plan the agent can ignore is the prose plan again.
                out, ms = refusal, 0.0
                self._plan_refused_opens += 1
                self.tracer.emit("plan_refused_open", severity="warning", tool=c["name"],
                                 blocked=refusal["blocked"])
            else:
                out, ms = self.toolbox.dispatch(c["name"], c["arguments"])
            self.tracer.tool(c["name"], c["arguments"], out, ok="error" not in out, ms=ms)
            self._detect_triggers(c["name"], c["arguments"] or {}, out, s.get("step", 0))

            if c["name"] == "submit_answer":
                verdict = self._gate(self.toolbox.submitted or {})
                if verdict["accepted"]:
                    answer = self.toolbox.submitted or {}
                    done = True
                    gate_validated = True          # the ONLY place this becomes true
                    if self._steps_to_gate_pass is None:
                        # Answers existence and cost in one run: whether the obligation is
                        # reachable at all, and what it costs per patient per variable. If
                        # this lands at 23, a 20-step budget was short by 3 and no second
                        # run is needed to learn that.
                        self._steps_to_gate_pass = s.get("step", 0) + 1
                        self.tracer.emit("gate_passed", step=self._steps_to_gate_pass,
                                         rejections_before=len(rejections))
                    out = {"accepted": True}
                else:
                    rejections.append(verdict)
                    self.tracer.rejected(verdict["why"], verdict["missing"], self.toolbox.submitted)
                    out = {"accepted": False, "why": verdict["why"], "you_must_still": verdict["missing"]}

            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                         "content": json.dumps(out, ensure_ascii=False, default=str)[:6000]})

        return {"messages": msgs, "step": s.get("step", 0) + 1, "done": done,
                "gate_validated": gate_validated,
                "answer": answer, "rejections": rejections,
                "evidence": self.evidence.to_list(), "coverage": self.coverage.to_dict()}

    # -------------------------------------------------------- plan enforcement + triggers
    def _plan_refusal(self, name: str, args: dict) -> dict | None:
        """Refuse a read of a `sample` type the runtime did not draw. Returns None to allow.

        The escape hatch is deliberate and is the whole design: the refusal names the type
        and tells the agent to promote it in the next reflection. An agent that has found a
        reason to open a sampled type gets to open it — by widening the plan on the record,
        which is monotone and auditable, rather than by quietly wandering out of scope.
        """
        if name not in PLAN_GOVERNED_TOOLS:
            return None
        ids = ([args.get("note_id")] if name != "read_documents_batch"
               else list(args.get("note_ids") or []))
        drawn = {n for v in self.coverage.drawn.values() for n in v}
        blocked: list[dict] = []
        for nid in ids:
            meta = self.chart._docs.get(str(nid))
            if meta is None or str(nid) in drawn:
                # Unknown ids fall through to the toolbox, which distinguishes a fabricated
                # note_id from absence — a distinction this guard must not blur. A drawn
                # document is the RUNTIME's choice, never the agent's, so it is always open.
                continue
            if not self.plan.may_open(meta.doc_type):
                blocked.append({"note_id": str(nid), "doc_type": meta.doc_type})
        if not blocked:
            return None
        types = sorted({b["doc_type"] for b in blocked})
        return {
            "error": "OUT_OF_PLAN",
            "blocked": blocked,
            "message": ("The retrieval plan assigns these document types to `sample`: the "
                        "runtime's sampler draws from them and you may not open them "
                        "directly. This is NOT evidence that they hold nothing."),
            "types": types,
            "how_to_proceed": ("if you have a reason to think this type bears on the answer, "
                               "promote it at the next reflection with "
                               f"promote_types=[{{'type': {types[0]!r}, 'to': 'search'}}]. "
                               "The plan may only ever widen, so the promotion is recorded "
                               "and permanent."),
        }

    def _detect_triggers(self, name: str, args: dict, out: dict, step: int) -> None:
        """Mechanical conditions, read off the tool result. No model is asked anything here.

        The old reflect node posed an open question — "does something learned change what
        should be done next?" — whose default answer is no, and which was answered no 291
        times running. These are decidable facts.
        """
        found = triggers_from_tool_result(
            name, args, out if isinstance(out, dict) else {}, plan=self.plan,
            catalogue=self.markers, step=step,
            quote=str((out or {}).get("quote", "")) if name == "record_evidence" else "")
        for t in found:
            if t.kind == TRIGGER_UNSETTLED_THREAD:
                m = self.markers.by_text().get(t.marker)
                th = self.threads.open_thread(
                    note_id=t.note_id, doc_type=t.doc_type, marker=t.marker,
                    obligation=(m.obligation if m else "unsettled"), excerpt=t.observation,
                    step=step)
                if th is None:
                    continue        # already outstanding; re-reading must not multiply debt
            self._record_trigger(t)

    def _record_trigger(self, t: Trigger) -> None:
        """Count one detected trigger, queue it for the next reflection, and trace it.

        `tracer.trigger` and not `tracer.emit("trigger", **t.to_dict())`: the trigger's own
        `kind` collided with the trace envelope's `kind` and every run that detected anything
        died on the spot with a TypeError. One emitter, used by both detection paths, so the
        two cannot drift into two event shapes.
        """
        self._pending_triggers.append(t)
        self._trigger_counts[t.kind] = self._trigger_counts.get(t.kind, 0) + 1
        self.tracer.trigger(**t.to_dict())

    def _gate_triggers(self, step: int) -> None:
        """The fourth trigger: an obligation the CURRENT plan structurally cannot discharge.

        A gate that says "read these search hits" while the plan says "you may not open that
        type" is a deadlock, not a rejection. The old loop would spend the rest of its budget
        in it, which is what a 400k-token run defending an interim answer looks like.
        """
        try:
            g = check_gate(self.spec, self.coverage, self.plan)
        except Exception:      # noqa: BLE001 - trigger detection may never break a run
            return
        unread_types: list[str] = []
        for r in self.coverage.stratum_results():
            for nid in r.hits_unread:
                meta = self.chart._docs.get(nid)
                if meta:
                    unread_types.append(meta.doc_type)
        for t in gate_obligation_triggers(g.missing, plan=self.plan,
                                          unread_hit_types=unread_types, step=step):
            self._record_trigger(t)

    # ------------------------------------------------------------- the expansion budget
    def _price_expansion_budget(self) -> ExpansionBudget:
        """The caps, in the units the plan actually counts them in.

        THE PLANNER'S OWN TERMS ARE NOT THE AGENT'S ALLOWANCE. `priced_against` caps
        `max_terms_added` at the number of SPEC-declared terms — "needing to more than double
        the spec's list is a spec that was wrong" — but `plan.terms_added()` counts every
        `term_provenance` row, and the up-front planner's proposals are rows like any other
        (PLAN_PROMPT asks it for a keyword list, and `plan_coverage` records what it gets as
        the run's first monotone addition, on purpose: a term the planner supplied is
        evidence about the SPEC). Charging those to the agent spent the whole allowance
        before the first reflection — the production path showed `EXPANSION REMAINING:
        terms -3` at step 1, refused the first single-term revision BUDGET_EXHAUSTED, and
        ended the run at step 1 of 12.

        The manifest already keeps `terms_added` and `terms_added_by_reflection` apart. This
        makes the BUDGET keep them apart too, by offsetting the cap by the rows that were
        already there. The alternative — teach the counter to skip `planner_proposal` rows —
        lives in `CoveragePlan.apply_revision`, which this file does not own.
        """
        priced = self.expansion_budget or ExpansionBudget.priced_against(
            self.plan, self._docs_by_type, max_revisions=self.budget.max_plan_revisions)
        if not self._planner_terms:
            return priced
        return _dc_replace(priced,
                           max_terms_added=priced.max_terms_added + self._planner_terms)

    def _expansion_headroom(self) -> dict[str, int]:
        """What is left of each cap, counted exactly the way `apply_revision` prices it."""
        b = self._expansion_budget
        return {"terms": b.max_terms_added - len(self.plan.terms_added()),
                "type_promotions": b.max_type_promotions - len(self.plan.promotion_log),
                "revisions": b.max_revisions - self.plan.revisions_applied}

    def _expansion_budget_report(self) -> dict:
        """The budget as a reader must see it: both halves of the term cap, named."""
        b = self._expansion_budget
        return {**b.to_dict(),
                "source": self.expansion_budget_source,
                "planner_proposed_terms": self._planner_terms,
                "max_terms_added_by_reflection": b.max_terms_added - self._planner_terms,
                "terms_are_counted": (
                    "`max_terms_added` is measured against EVERY term_provenance row, so the "
                    "up-front planner's own proposals are added to the cap rather than "
                    "charged against the agent's reflection allowance"),
                "headroom": self._expansion_headroom()}

    def _n_reflect(self, s: RunState) -> dict:
        step = s.get("step", 0)
        self._gate_triggers(step)
        triggers = list(self._pending_triggers)
        self._pending_triggers = []
        self._reflections += 1
        h = self._expansion_headroom()
        if any(v < 0 for v in h.values()):
            # Unreachable by construction — `apply_revision` refuses anything that would
            # overspend — so it is an error and not a clamp. A negative allowance shown to
            # the supervisor is the defect this budget split exists to remove; hiding one
            # behind max(0, ...) without saying so would put it back invisibly.
            self.tracer.emit("expansion_headroom_negative", severity="error", headroom=h,
                             budget=self._expansion_budget_report(),
                             message=("more expansion has been spent than the budget allows; "
                                      "the remaining allowance shown to the supervisor is "
                                      "clamped at zero and is NOT the true count"))
        remaining = (f"terms {max(h['terms'], 0)}, "
                     f"type promotions {max(h['type_promotions'], 0)}, "
                     f"revisions {max(h['revisions'], 0)}")
        prompt = REFLECT_PROMPT.format(
            question=self.spec.question,
            obligation="\n".join(f"  - {x}" for x in self.spec.proof_obligation.required_coverage) or "  (none)",
            plan=self.plan.render(self._docs_by_type),
            threads=self.threads.render(),
            triggers=("\n".join(f"  - [{t.kind}] {t.observation}"
                                + (f"  candidate terms: {list(t.terms_proposed)}"
                                   if t.terms_proposed else "")
                                + (f"  candidate types: {list(t.types_proposed)}"
                                   if t.types_proposed else "")
                                for t in triggers)
                      or "  (none detected — no observation obliges a revision)"),
            evidence=self.evidence.render(),
            coverage=self.coverage.render(),
            budget=remaining,
            step=step, max_steps=self.budget.max_steps,
        )
        msgs_ref = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        r = self.llm.chat(msgs_ref)
        # require="verdict": the reply may carry more than one JSON object (gpt-5.6-luna
        # leaks a tool-channel preamble object first), and the first one is not the answer.
        j = extract_json(r.content, require="verdict")
        raw_verdict = j.get("verdict")

        # Same trap as the planner, same model, same completion budget — and this prompt is
        # LONGER because it carries the evidence ledger. An unparsed reply silently becoming
        # CONTINUE is indistinguishable from a supervisor that read the evidence and judged
        # "keep going", which makes every CONTINUE in the trace uninterpretable. Retry once,
        # then record the degradation loudly instead of laundering it into a verdict.
        if raw_verdict is None or str(raw_verdict).upper() not in VERDICTS:
            self.tracer.emit("reflect_empty_retry", first_attempt_chars=len(r.content),
                             used_reasoning_channel=r.used_reasoning_channel,
                             completion_tokens=r.completion_tokens, raw_verdict=raw_verdict)
            old = self.llm.cfg.max_tokens
            self.llm.cfg.max_tokens = max(old * 2, 4096)
            try:
                r = self.llm.chat(msgs_ref)
            finally:
                self.llm.cfg.max_tokens = old
            j = extract_json(r.content, require="verdict")
            raw_verdict = j.get("verdict")

        degraded = raw_verdict is None or str(raw_verdict).upper() not in VERDICTS
        if degraded:
            verdict = "CONTINUE"
            self._reflect_fallbacks += 1
            self.tracer.emit("reflect_fallback_used", severity="error", raw_verdict=raw_verdict,
                             message=("the supervisor returned nothing usable after a retry; "
                                      "defaulting to CONTINUE. This verdict carries NO "
                                      "information and no conclusion about reflection "
                                      "behaviour may be drawn from this step"))
        else:
            verdict = str(raw_verdict).upper()
        if verdict == "REPLAN":
            # No longer a verdict the model may assert. Counted, because a model still
            # reaching for the word is a fact about the prompt, and read as CONTINUE: the
            # REVISION replans, not the label. When the label was the mechanism, asserting
            # it changed nothing and so it was never asserted.
            self._model_asserted_replan += 1
            self.tracer.emit("model_asserted_replan", severity="warning",
                             message=("the supervisor returned the REPLAN verdict, which no "
                                      "longer exists; the revision object is what replans"))
            verdict = "CONTINUE"

        # ------------------------------------------------------- apply the typed revision
        # IF THE RUNTIME DOES NOT APPLY IT, IT DID NOT HAPPEN. Everything below mutates the
        # object that actually governs retrieval, or refuses to and records the refusal.
        rev = PlanRevision.from_json(j.get("revision"))
        outcome = None
        deferred: list[str] = []
        salvage = None
        if not rev.is_empty():
            # WHAT OBSERVATION CAUSED IT. The detected triggers when there were any; the
            # supervisor's own stated reason when there were none. An empty observation would
            # make an unprompted widening indistinguishable from a forced one downstream, and
            # the develop plane needs to know which of the two it is holding.
            why = "; ".join(t.observation for t in triggers) or (
                f"unprompted by any detected trigger; supervisor's reason: "
                f"{str(j.get('reason', '')).strip() or '(none given)'}")
            trigger_label = ",".join(sorted({t.kind for t in triggers})) or "unprompted"
            # YOU MAY HAVE 5 OF THE 6. Trim the term list to what the budget can pay for
            # BEFORE it is priced, so one term too many no longer refuses the whole revision.
            to_apply, deferred = self._fit_terms_to_budget(rev)
            outcome = self.plan.apply_revision(
                to_apply, step=step, trigger=trigger_label,
                observation=why[:800],
                budget=self._expansion_budget, threads=self.threads,
                n_docs_by_type=self._docs_by_type,
                known_types=self.toolbox.known_doc_types)
            if deferred:
                self._terms_deferred.extend(deferred)
                outcome.refused.append(
                    f"{REFUSED_BUDGET}: {len(deferred)} of the {len(rev.add_terms)} terms you "
                    f"asked for did not fit the remaining expansion budget and were NOT "
                    f"added: {deferred}")
                self.tracer.emit("revision_partially_applied", severity="warning",
                                 requested_terms=list(rev.add_terms),
                                 applied_terms=list(outcome.terms_added),
                                 deferred_terms=list(deferred),
                                 headroom=self._expansion_headroom(),
                                 message=("the term list was trimmed to the remaining budget "
                                          "rather than the revision being refused whole; the "
                                          "deferred terms are reported back to the agent"))
            self.tracer.emit("plan_revision", applied=outcome.applied,
                             severity=("info" if outcome.applied and not deferred
                                       else "warning"),
                             requested=rev.to_dict(), applied_subset=to_apply.to_dict(),
                             deferred_terms=list(deferred),
                             outcome=outcome.to_dict(),
                             triggers=[t.kind for t in triggers])
            if not outcome.applied:
                self._revisions_refused += 1
                # THE THREAD WORK IS NOT COLLATERAL. Retried on its own below; see
                # `_salvage_thread_work`.
                salvage = self._salvage_thread_work(to_apply, step=step,
                                                    trigger=trigger_label, observation=why)
                # Back to the agent, in the loop it already understands. A refusal the model
                # never sees is a refusal it repeats.
                triggers = triggers + [Trigger(
                    kind="REVISION_REFUSED", step=step,
                    observation=f"{outcome.refusal_class}: {'; '.join(outcome.refused)[:300]}")]

        replanned = bool(outcome and outcome.applied and outcome.changed_retrieval())
        if replanned:
            self._revisions_applied += 1
            # THE runtime-derived REPLAN. It is true exactly when the retrieval scope moved.
            verdict = "REPLAN"

        self.tracer.reflect(verdict, j.get("reason", ""), len(self.evidence.items))
        upd: dict[str, Any] = {"reflection": {"verdict": verdict, "reason": j.get("reason", ""),
                                              "degraded": degraded,
                                              "revision": (outcome.to_dict() if outcome else None),
                                              "triggers": [t.kind for t in triggers]}}
        if replanned:
            upd["plan"] = self.plan.to_dict()
        msgs = list(s["messages"])
        tail = ""
        if outcome is not None and outcome.applied and deferred:
            # PARTIAL APPLICATION IS ONLY HONEST IF IT IS SAID. An agent that asked for six
            # terms, got five and was told nothing would believe it had all six and would
            # never re-ask for the sixth.
            tail = (f"Your revision was APPLIED IN PART ({REFUSED_BUDGET} on the term list "
                    f"only). These terms did NOT fit the remaining expansion budget and were "
                    f"not added: {deferred}. Re-ask for "
                    f"the one that matters most if budget frees up. The plan now reads:\n"
                    + self.plan.render(self._docs_by_type))
        elif outcome is not None and outcome.applied:
            tail = ("Your revision was APPLIED. The plan now reads:\n"
                    + self.plan.render(self._docs_by_type))
        elif outcome is not None:
            tail = (f"Your revision was REFUSED ({outcome.refusal_class}) and the refusal is "
                    f"recorded:\n  - " + "\n  - ".join(outcome.refused))
            if salvage is not None and salvage.applied:
                tail += ("\nThe THREAD work in it was kept and applied — resolved: "
                         f"{salvage.threads_resolved}, dismissed: {salvage.threads_dismissed}, "
                         f"opened: {salvage.threads_opened}. Only the retrieval half was "
                         "refused; do not re-send the thread operations.")
        elif verdict == "SUFFICIENT":
            tail = "If you have what you need, call submit_answer now."
        msgs.append({"role": "user", "content":
                     f"[supervisor] verdict={verdict}. {j.get('reason','')}\n" + tail})
        upd["messages"] = msgs
        return upd

    # ------------------------------------------------- partial application of a revision
    def _fit_terms_to_budget(self, rev: PlanRevision) -> tuple[PlanRevision, list[str]]:
        """Trim `add_terms` to the remaining allowance, keeping the supervisor's own order.

        WHY PARTIAL AND NOT ALL-OR-NOTHING. All-or-nothing is right for MONOTONICITY — a
        revision that also demoted a type must be refused whole, because applying its
        admissible half hands back a plan the agent did not propose. A budget overrun is a
        different kind of failure: nothing about the requested terms is inadmissible, there
        is simply not enough allowance for all of them, and every prefix of the list is a
        plan the agent DID propose. Refusing the whole thing taught the agent nothing, marked
        the plan exhausted forever, and ended the run — over one term.

        The order is the model's own priority order, and the truncation is reported back by
        name, so "you may have 5 of the 6" is a statement the agent can act on. Terms already
        in the plan are kept and cost nothing: `apply_revision` prices only what is new.
        """
        room = max(0, self._expansion_headroom()["terms"])
        kept: list[str] = []
        deferred: list[str] = []
        for t in rev.add_terms:
            if t in self.plan.keywords:
                kept.append(t)
            elif room > 0:
                kept.append(t)
                room -= 1
            else:
                deferred.append(t)
        if not deferred:
            return rev, []
        return _dc_replace(rev, add_terms=tuple(kept)), deferred

    def _salvage_thread_work(self, rev: PlanRevision, *, step: int, trigger: str,
                             observation: str) -> RevisionOutcome | None:
        """Re-apply the thread half of a revision whose retrieval half was refused.

        A refused revision used to discard its thread operations too, so a revision that both
        over-reached on terms and RESOLVED THE THREAD BLOCKING THE ANSWER ended the run twice
        over: budget-exhausted and thread-blocked, with the resolution nowhere. Thread
        bookkeeping is not the retrieval half — `changed_retrieval()` says so itself, and no
        thread operation can violate monotonicity or widen what may be opened — so it does
        not belong to the refusal.

        Re-sent through `apply_revision` rather than applied against the ledger here, so the
        semantics of a resolution (and of an unreasoned dismissal, which is refused) stay
        defined in exactly one place.
        """
        threads_only = PlanRevision(open_threads=rev.open_threads,
                                    resolve_threads=rev.resolve_threads,
                                    dismiss_threads=rev.dismiss_threads)
        if threads_only.is_empty():
            return None
        out = self.plan.apply_revision(
            threads_only, step=step, trigger=trigger, observation=observation[:800],
            budget=self._expansion_budget, threads=self.threads,
            n_docs_by_type=self._docs_by_type, known_types=self.toolbox.known_doc_types)
        self.tracer.emit("thread_work_salvaged", severity="warning", applied=out.applied,
                         threads_opened=out.threads_opened,
                         threads_resolved=out.threads_resolved,
                         threads_dismissed=out.threads_dismissed,
                         refused=out.refused,
                         message=("the retrieval half of this revision was refused; its "
                                  "thread operations were re-applied on their own rather "
                                  "than discarded with it"))
        return out

    def _n_finalize(self, s: RunState) -> dict:
        if s.get("answer") and s.get("done"):
            ans = s["answer"]
        else:
            gate = self._check_gate()
            note = ("The proof obligation for a negative answer is SATISFIED."
                    if gate.verdict == "PASS" else
                    "The proof obligation for a negative answer is NOT satisfied; outstanding: "
                    + "; ".join(gate.missing)
                    + ". You may not assert a confident negative — prefer EVIDENCE_INSUFFICIENT.")
            keys = ", ".join(f'"{f.name}": null' for f in self.spec.fields) or '"value": null'
            prompt = FINALIZE_PROMPT.format(
                spec_block=self.spec.as_prompt_block(), patient_id=s["patient_id"],
                rule_citations=rule_citation_block(self.spec),
                evidence=self.evidence.render(), coverage=self.coverage.render(),
                gate_note=note, value_keys=keys,
                spec_sections=", ".join(SPEC_SECTIONS))
            r = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            self.tracer.llm("finalize", r.content, usage={"total": r.total_tokens})
            ans = extract_json(r.content, require="status")
            # This answer never passes through `gate_answer` — finalize authors it when the
            # agent never called submit_answer — so its self-report has to be collected here
            # or the one path that produces an UNGATED answer is also the one path with no
            # attribution at all, which is precisely backwards.
            self.tracer.self_reported_rules([ans.get("rules_applied"), ans.get("reasoning")],
                                            where="finalize")

        forced_from = None
        if self.spec.data_source == "outside_notes":
            # Recorded before the overwrite: the loop needs to know this SPEC_INSUFFICIENT is
            # a constant the runtime imposes on every chart, not an agent's judgement, and
            # after the assignment that distinction is gone.
            forced_from = ans.get("status")
            ans["status"] = "SPEC_INSUFFICIENT"
        if not ans.get("status"):
            self._finalize_defaults += 1
            self.tracer.emit("finalize_status_defaulted", severity="error",
                             message=("the model produced no status; defaulting to "
                                      "EVIDENCE_INSUFFICIENT. This label is NOT evidence "
                                      "that the agent reached that conclusion"))
            ans["status"] = "EVIDENCE_INSUFFICIENT"

        # Positive and negative findings are proved differently, so they are labelled
        # differently. A negative additionally has three possible bases demanding three
        # different downstream owners: emitting the same EVIDENCE_INSUFFICIENT for all of
        # them is how "the agent gave up" gets filed as "the chart really has nothing".
        if ans.get("status") == "FOUND":
            # Witness proof: one qualifying document settles it, and the gate for FOUND
            # checks exactly that. It never claims the universe was searched. Attaching
            # coverage_attested here would advertise a stronger claim than anything that was
            # verified — the same error as a check that records but cannot refuse, committed
            # on the way out instead of on the way in.
            ans["proof_basis"] = "WITNESS"
            ans["witness_count"] = len(self.evidence.items)
            if not s.get("gate_validated"):
                # Left without going through submit_answer, so not even the witness standard
                # was applied. Different reason from a coverage failure, same consequence.
                ans["proof_basis"] = "UNGATED"
                ans["route_to_human"] = True
                self.tracer.emit("ungated_positive", severity="warning",
                                 termination=getattr(self, "_termination", None))
        elif ans.get("status") == "SPEC_INSUFFICIENT":
            # A statement about the SPECIFICATION, so it gets no negative_basis and no
            # coverage ledger — those describe how well this chart was searched, and the
            # claim is not about this chart. Attaching one is the category error that
            # crashed every run reaching this status. What it does get is the report the
            # §6b optimizer routes on; see build_spec_gap.
            gap, remedy = build_spec_gap(
                self.spec, ans, reported_by=("runtime" if forced_from is not None else "agent"),
                gate_validated=bool(s.get("gate_validated")))
            if forced_from is not None:
                gap["forced_over_status"] = forced_from
            ans["spec_gap"] = gap
            ans["remedy_class"] = remedy
            ans["proof_basis"] = "NOT_APPLICABLE"
            ans["coverage_note"] = ("no coverage claim is made — SPEC_INSUFFICIENT is a "
                                    "statement about the specification, not about this chart")
            # The gate refuses a submitted value, but two paths arrive here without passing
            # it: the outside_notes overwrite above, and a finalize-authored answer written
            # when the agent never called submit_answer.
            strip_value_from_spec_insufficient(ans, self.tracer)
            self.tracer.emit("spec_insufficient_reported",
                             reported_by=gap["reported_by"], remedy_class=remedy,
                             spec_section=gap["spec_section"], routable=gap["routable"],
                             uncovered_fields=gap["uncovered_fields"],
                             has_spec_quote=bool(gap["spec_quote"]))
            if not gap["routable"]:
                # Reachable only when finalize authored the answer itself, so there is no
                # loop left to reject into. Loud and counted: the manifest still lands, the
                # status is still true, and the one thing that makes it USEFUL is missing.
                self._spec_gaps_unroutable += 1
                self.tracer.emit("spec_gap_unroutable", severity="error",
                                 spec_section=gap["spec_section"],
                                 agent_words_supplied=gap["agent_words_supplied"],
                                 message=("SPEC_INSUFFICIENT was reported without naming the "
                                          "part of the specification at fault. The status is "
                                          "recorded; it cannot be routed to any text."))
        elif s.get("gate_validated"):
            ans["negative_basis"] = "GATE_VALIDATED"
            ans["proof_obligation"] = self._check_gate().to_dict()
            ans["coverage_attested"] = self.coverage.to_dict()
        else:
            ans["negative_basis"] = getattr(self, "_termination", None) or "BUDGET_EXHAUSTED"
            ans["route_to_human"] = True
            # Deliberately withholding the ledger: this answer never passed the gate, and
            # attaching it would let the answer read as though it had.
            ans["coverage_note"] = ("no coverage claim is made — this answer did not pass the "
                                    "proof obligation")
            self.tracer.emit("unvalidated_negative", severity="warning",
                             negative_basis=ans["negative_basis"])

        for k in ("spec_section", "spec_quote", "uncovered_fields"):
            # Folded into spec_gap already. Leaving the raw inputs beside the assembled block
            # invites a reader to trust the copy that the gate never validated — and on the
            # finalize-authored path it never did.
            ans.pop(k, None)
        ans["evidence"] = self.evidence.to_list()
        # Travels with the answer, not only in the manifest. An unresolved thread on a
        # finalize-authored answer (the one path with no gate to refuse it) is the exact
        # shape of the 8046 error, and the answer has to carry the fact that it exists.
        if self.threads.threads:
            ans["threads"] = self.threads.to_dict()
            if self.threads.unresolved():
                ans["route_to_human"] = True
                ans["unsettled_threads"] = [t.thread_id for t in self.threads.unresolved()]
                self.tracer.emit("answer_carries_unsettled_threads", severity="warning",
                                 thread_ids=ans["unsettled_threads"], status=ans.get("status"))
        assert_answer_is_reportable(ans)   # enforced at emission, not merely intended
        return {"answer": ans, "done": True}

    # ------------------------------------------------------------------ edges
    def _after_act(self, s: RunState) -> str:
        if s.get("done"):
            return "finalize"
        if self._over_budget(s):
            return "finalize"
        return "reflect" if s.get("step", 0) % self.reflect_every == 0 else "reflect"

    def _after_reflect(self, s: RunState) -> str:
        refl = s.get("reflection") or {}
        v = refl.get("verdict")
        if v is None:
            # Reaching the edge with no verdict at all means the node did not run. Treating
            # that as CONTINUE would hide a broken graph behind normal-looking behaviour.
            self._reflect_fallbacks += 1
            self.tracer.emit("reflect_missing_at_edge", severity="error",
                             message="no verdict present when routing; defaulting to CONTINUE")
            v = "CONTINUE"
        if self._over_budget(s):
            self._termination = "BUDGET_EXHAUSTED"
            return "finalize"
        if v == "SUFFICIENT":
            # NOT to finalize. There must be exactly one route by which an answer is
            # produced, and it runs through submit_answer, because that is where the gate
            # lives. Routing SUFFICIENT to finalize gave the graph a second inbound edge to
            # the answer and the proof obligation was simply skipped — the run still printed
            # a proof_obligation field, computed but unable to refuse, which is a comment
            # wearing the costume of a check. reflect keeps its judgement; it expresses it as
            # "go submit", not as "we are done".
            self.tracer.emit("reflect_sufficient_routed_to_submit",
                             message="supervisor judged the evidence sufficient; routing to "
                                     "submit_answer so the proof obligation still applies")
            return "act"
        if v == "STUCK":
            # A give-up is not an answer and must NOT be asked to prove coverage: it asserts
            # no coverage. It goes straight out, labelled as unvalidated.
            self._termination = "AGENT_GAVE_UP"
            return "finalize"
        if self._expansion_exhausted_with_obligations():
            # EXPANSION HAS A BUDGET, and running out of it is a result, not a nuisance.
            # The alternative — keep looping until max_steps and emit whatever is in hand — is
            # a silent truncation dressed as an answer. This exits labelled, so the manifest
            # carries EXPANSION_BUDGET_EXHAUSTED rather than a shrug, and `finalize` gives it
            # no coverage claim because it never passed the gate.
            self._termination = "EXPANSION_BUDGET_EXHAUSTED"
            self.tracer.emit("expansion_budget_exhausted", severity="warning",
                             budget=self._expansion_budget_report(),
                             terms_added=len(self.plan.terms_added()),
                             terms_deferred=list(self._terms_deferred),
                             promotions=len(self.plan.promotion_log),
                             outstanding=self._outstanding_obligations(),
                             message=("the plan can no longer widen and the proof obligation "
                                      "is still not met. This is EVIDENCE_INSUFFICIENT and it "
                                      "is honest; it is not a pass and it is not a truncation"))
            return "finalize"
        if v == "REPLAN" and s.get("plan_revisions", 0) < self.budget.max_plan_revisions:
            return "plan"
        return "act"

    def _outstanding_obligations(self) -> list[str]:
        try:
            missing = list(check_gate(self.spec, self.coverage, self.plan).missing)
        except Exception:      # noqa: BLE001
            missing = []
        return missing + check_threads(self.threads)

    def _expansion_exhausted_with_obligations(self) -> bool:
        """True when widening is over and the obligations are not discharged.

        Both halves matter. Budget spent with everything discharged is a run that finished;
        obligations outstanding with budget left is a run that should keep going. Only the
        conjunction is a dead end, and a dead end has to be SAID.
        """
        return self._expansion_is_spent() and bool(self._outstanding_obligations())

    def _expansion_is_spent(self) -> bool:
        """Widening is over: the agent has BUMPED INTO the cap and nothing is left to widen.

        The second half is what un-sticks the old rule. `plan.budget_exhausted` is true
        forever after a single budget refusal, so one term too many ended the run even when
        the plan could still promote a type — the widening move that was usually the one
        actually needed. Now a refusal (or a term the budget deferred: with partial
        application, a term overrun no longer records a refusal at all) only ARMS the check,
        and the run stops only when nothing can widen any further:

          * neither a term nor a promotion has any allowance left, or
          * no revisions remain, in which case nothing can be applied whatever the room.

        The first half is kept exactly as it was: a run that never tried to widen is not
        exhausted, however small its budget.
        """
        bumped = (self.plan.budget_exhausted(self._expansion_budget)
                  or bool(self._terms_deferred))
        if not bumped:
            return False
        h = self._expansion_headroom()
        return (h["terms"] <= 0 and h["type_promotions"] <= 0) or h["revisions"] <= 0

    def _over_budget(self, s: RunState) -> bool:
        why = self.budget.exceeded(step=s.get("step", 0),
                                   tokens=self.llm.prompt_tokens + self.llm.completion_tokens,
                                   elapsed=time.time() - self._t0)
        if why:
            self.tracer.emit("budget_exceeded", reason=why)
        return bool(why)

    # ------------------------------------------------------------------ gate
    def _check_gate(self) -> GateResult:
        return check_gate(self.spec, self.coverage, getattr(self, "plan", None))

    def _keyword_hits_among_drawn(self) -> set[str]:
        return keyword_hits_among_drawn(self.spec, self.coverage, self.chart)

    def _gate(self, submitted: dict) -> dict:
        return gate_answer(self.spec, submitted, evidence=self.evidence,
                           coverage=self.coverage, chart=self.chart, tracer=self.tracer,
                           threads=getattr(self, "threads", None),
                           plan=getattr(self, "plan", None))

    # ------------------------------------------------------------------ run
    def run(self, chart: PatientChart, run_id: str | None = None,
            known_doc_types: list[str] | None = None) -> dict:
        self.chart = chart
        self.evidence = EvidenceLedger()
        docs, _ = chart.list_documents(limit=100_000)
        strata = strata_from_spec(self.spec)
        # Derived, not drawn. `ForcedSampler(None)` used to reach for `random.randrange`,
        # which made every unseeded run a fresh roll of the validation draw: rerun until the
        # sample is kind. `--seed` still wins, and `seed_provenance` in the manifest below is
        # how a reader tells the two apart.
        self.effective_seed = (self.sample_seed if self.sample_seed is not None
                               else derive_sample_seed(chart.patient_id, self.spec.spec_id))
        self.coverage = CoverageLedger(docs, strata, ForcedSampler(self.effective_seed))
        # Corpus-wide type vocabulary keeps "this patient has none" (a finding) separable
        # from "no such type" (a typo). Without it the toolbox says so in its own error.
        self.toolbox = Toolbox(chart, self.evidence, self.coverage,
                               known_doc_types=known_doc_types)
        self.tracer = Tracer.create(self.out_dir, run_id)
        self._t0 = time.time()
        self._plan_fallbacks = 0
        self._reflect_fallbacks = 0
        self._finalize_defaults = 0
        self._act_no_tool_call = 0
        self._spec_gaps_unroutable = 0
        self._termination = None
        self._steps_to_gate_pass = None
        self._plan_refused_opens = 0
        self._revisions_applied = 0
        self._revisions_refused = 0
        self._reflections = 0
        self._model_asserted_replan = 0
        self._planner_terms = 0
        self._terms_deferred = []
        self._pending_triggers: list[Trigger] = []
        self._trigger_counts: dict[str, int] = {k: 0 for k in TRIGGERS}

        self.tracer.run_start(patient_id=chart.patient_id, model=self.llm.cfg.model,
                              **self.spec.identity(), n_documents=len(chart),
                              sample_seed=self.effective_seed,
                              seed_provenance=self.seed_provenance)
        # Before any rule can be cited, write down what the rules ARE. An id in a trace whose
        # spec has since been edited is unreadable without the fingerprint that travelled
        # with the run, and "which rule was in play" is the question this whole block exists
        # to answer six months later.
        self.tracer.bind_spec(self.spec)

        # THE ONE PLAN. Built once, before the loop, and never rebuilt: a plan re-derived
        # mid-run would be a fresh model guess with no monotonicity relation to the one the
        # agent has been working against, which is a narrowing wearing a widening's clothes.
        self._docs_by_type = documents_by_type(chart)
        self.threads = OpenThreadLedger()
        self.markers = load_marker_catalogue()
        if self.markers.degraded:
            self.tracer.emit("marker_catalogue_degraded", severity="error",
                             detail=self.markers.degraded,
                             message=("thread detection is running on an incomplete marker "
                                      "set; an unsettled thread may pass unnoticed"))
        self.plan = self._build_plan()
        # Whatever the planner proposed is already in `term_provenance` and is NOT the
        # agent's expansion allowance. Counted before the budget is priced against it.
        self._planner_terms = len(self.plan.term_provenance)
        self._expansion_budget = self._price_expansion_budget()
        self.tracer.emit("retrieval_plan", source=self.plan.source,
                         plan=self.plan.to_dict(),
                         expansion_budget=self._expansion_budget_report(),
                         expansion_budget_source=self.expansion_budget_source,
                         marker_catalogue=self.markers.source,
                         monotonicity_vs_ledger=MONOTONICITY_VS_LEDGER)

        final = self._graph.invoke(
            {"patient_id": chart.patient_id, "spec_id": self.spec.spec_id, "step": 0,
             "max_steps": self.budget.max_steps, "plan_revisions": 0, "rejections": []},
            {"recursion_limit": self.budget.max_steps * 4 + 20},
        )

        answer = final.get("answer", {})
        result = {
            "run_id": self.tracer.run_id,
            "patient_id": chart.patient_id,
            **self.spec.identity(),
            "model": self.llm.cfg.model,
            "answer": answer,
            # THE ONE PLAN, in the shape that governed retrieval. Both term lists are inside
            # it and they are never merged: `initial_keywords` is the spec's, and it is the
            # baseline the develop plane scores against; `keywords` is the final expanded
            # list, and it is what coverage was evaluated against.
            "plan": self.plan.to_dict(),
            "plan_revisions": final.get("plan_revisions", 0),
            # ============================================================================
            # THE REPLAN RATE, AND WHY IT IS IN THE MANIFEST
            # ============================================================================
            # This is the health metric for the PRIOR, not for the agent. High means the
            # assets are underdeveloped — the spec's term list and stratum declarations are
            # being corrected at run time, once per patient, at inference cost. Near zero
            # WITH good assets means the agentic layer has nothing left to do on this
            # criterion and the honest conclusion is to ship a workflow, not an agent.
            # It is here rather than only in the trace because that question is asked over a
            # DIRECTORY of finished runs, and a number recoverable only by replaying JSONL is
            # a number nobody computes. The old REPLAN counter answered it 0/291 times, which
            # read as "the agent is stable" and actually meant "the verdict did nothing".
            "replan": {
                "n_reflections": self._reflections,
                "n_revisions_applied": self._revisions_applied,
                "n_revisions_refused": self._revisions_refused,
                # Applied revisions that CHANGED RETRIEVAL, over reflections. Thread
                # bookkeeping is deliberately not counted: resolving a thread is not a
                # replan, and counting it would reinflate the metric with no-ops.
                "replan_rate": (round(self._revisions_applied / self._reflections, 4)
                                if self._reflections else None),
                # Split, because they are two different facts. Everything the SPEC did not
                # declare counts against the spec; only what a REFLECTION added counts as
                # replanning. The up-front planner's own proposals are in the first number
                # and not the second — a term the planner supplied before the run began is
                # evidence about the spec, not evidence that the agent learned something.
                "terms_added": len(self.plan.terms_added()),
                "terms_added_by_reflection": sum(
                    1 for r in self.plan.term_provenance if r["trigger"] != "planner_proposal"),
                "types_promoted": len(self.plan.promotion_log),
                "triggers_fired": dict(self._trigger_counts),
                "plan_refused_opens": self._plan_refused_opens,
                # A supervisor still reaching for a verdict that no longer exists is a fact
                # about the prompt, not about the chart.
                "model_asserted_replan": self._model_asserted_replan,
            },
            # HARVESTED FROM PRODUCTION, with the trace that produced each one. Every term
            # and every promotion here is a candidate edit to the spec, carrying the step,
            # the trigger that forced it and the observation that caused it — which is the
            # difference between a develop-plane input and a list of words.
            "develop_plane_candidates": {
                "spec_declared_terms": list(self.plan.initial_keywords),
                "terms_added_at_runtime": list(self.plan.term_provenance),
                "types_promoted_at_runtime": list(self.plan.promotion_log),
                "refused_revisions": list(self.plan.refused_revisions),
                # A term the run ASKED FOR and the budget could not pay for is evidence about
                # the spec's list too, and partial application is exactly what stops it from
                # landing in `refused_revisions` (the revision was applied; only the tail of
                # its term list was not). It would otherwise disappear from the harvest.
                "terms_deferred_for_budget": list(self._terms_deferred),
                "what_this_is": ("candidate spec edits observed on a real chart. Score the "
                                 "spec's list against spec_declared_terms, NEVER against the "
                                 "expanded list — a runtime rescue that is folded back into "
                                 "the baseline erases the evidence that the baseline was wrong"),
                "trace": str(self.tracer.path),
            },
            "open_threads": {**self.threads.to_dict(),
                             "marker_catalogue": self.markers.source,
                             "marker_catalogue_degraded": self.markers.degraded or None},
            "expansion_budget": {**self._expansion_budget_report(),
                                 # SPENT, not merely bumped into. A single refusal used to
                                 # read as exhaustion forever; `terms_deferred` is the other
                                 # half of the same fact now that a term overrun trims rather
                                 # than refuses, and both are reported.
                                 "exhausted": self._expansion_is_spent(),
                                 "refused_at_least_once": self.plan.budget_exhausted(
                                     self._expansion_budget),
                                 "terms_deferred": list(self._terms_deferred)},
            "monotonicity_vs_ledger": MONOTONICITY_VS_LEDGER,
            "steps": final.get("step", 0),
            "negative_basis": (final.get("answer") or {}).get("negative_basis"),
            "gate_validated": bool(final.get("gate_validated")),
            # What the spec's provenance permits this run to CLAIM, which is a separate
            # question from whether the gate passed. The gate proves the search was done; it
            # cannot prove the search terms were the right ones, and on this corpus STORE.400's
            # five terms miss the diagnosis for 31.7% of patients while the gate stays clean.
            # `reportable_as_validated` inside this block is the field a downstream filter
            # should read, never `gate_validated` alone.
            "provenance": self.spec.provenance_for_run(
                answer.get("value") or {}, str(answer.get("status") or ""),
                gate_validated=bool(final.get("gate_validated"))),
            # The seed and where it came from, in the manifest, always. A run whose
            # provenance is `caller_supplied` was sampled with a number the operator chose,
            # and a reader who cannot see that cannot tell a reproduced draw from a shopped
            # one. `seed_is_caller_supplied` is the flag; it is redundant with the string on
            # purpose, because a boolean is what a filter over a directory of manifests reads.
            "sample_seed": self.effective_seed,
            "seed_provenance": self.seed_provenance,
            "seed_is_caller_supplied": self.sample_seed is not None,
            # null when the gate never passed; the manifest then says which condition blocked.
            "steps_to_gate_pass": self._steps_to_gate_pass,
            "suspected_recognition_failures": len(self.coverage.suspected_recognition_failures),
            "rejections": final.get("rejections", []),
            # WHICH SPEC RULE WAS IN PLAY. In the manifest as well as the trace, because the
            # §6b loop reads a directory of finished runs: attribution recoverable only by
            # replaying a JSONL file is attribution nobody computes. Two separately named
            # channels inside, never merged — a check that provably fired and a rule the
            # model says it applied must not be readable as the same kind of fact.
            "rule_attribution": self.tracer.rule_attribution(),
            "usage": self.llm.usage(),
            # If this is non-zero the planner degraded and the run's replanning behaviour
            # was never actually exercised — read any conclusion about planning accordingly.
            # Any non-zero entry here means a node degraded silently and the corresponding
            # behaviour was NOT exercised. Read every conclusion against this block first.
            "degradation": {
                "plan_fallbacks": getattr(self, "_plan_fallbacks", 0),
                "reflect_fallbacks": getattr(self, "_reflect_fallbacks", 0),
                "finalize_defaults": getattr(self, "_finalize_defaults", 0),
                "act_no_tool_call": getattr(self, "_act_no_tool_call", 0),
                # A SPEC_INSUFFICIENT nobody can route is the channel half-working, which
                # historically reads as the channel working. It belongs in the block a
                # reader is told to check before believing anything else in the manifest.
                "spec_gaps_unroutable": getattr(self, "_spec_gaps_unroutable", 0),
                # An INT, like every other entry here, because the block's contract is "any
                # non-zero value means a node degraded and the behaviour was not exercised" —
                # and a filter over a directory of manifests reads that contract, not prose.
                # The prose lives beside the thread ledger. Non-zero means the marker set was
                # incomplete, so thread detection — the control that exists to catch the 8046
                # error — was running blind in part.
                "marker_catalogue_incomplete": int(bool(self.markers.degraded)),
            },
            "elapsed_s": round(time.time() - self._t0, 2),
            "trace": str(self.tracer.path),
        }
        self.tracer.run_end(**{k: v for k, v in result.items() if k != "answer"},
                            status=answer.get("status"))
        self.tracer.write_manifest(result)
        return result
