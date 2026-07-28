"""What an answer owes at the moment it is emitted, asserted by every runtime rather than
intended by each.

This is the EMISSION half of answer admissibility, and it is deliberately not the gate.
`answer_gate` decides whether a submission may stand and returns a rejection the agent can
recover from inside its own loop; the rules here RAISE, because they are reached after the
loop is over and there is nobody left to tell. Three runtimes emit answers — `graph`,
`deep_runner`, `mcp_server` — and a rule enforced in two of them makes the signal silently
conditional on which runtime the operator happened to use.

Both halves of the contract are here because they are one rule seen from two sides: an
answer must carry exactly the proof its status claims, and no more. Splitting them would let
one status be exempted from an obligation without being handed the other, which is the
category error this module was written to end (see the status table below).
"""
from __future__ import annotations

from .spec import ExtractionSpec
from .trace import parse_rule_citations, rule_catalog

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


# ---------------------------------------------------------------- the coverage claim
# MOVED HERE FROM `deep_runner`, where it was defined beside one runtime and imported by
# another. Whether an answer may claim coverage is an answer rule, not a property of the
# loop that produced it — and this module is where `assert_answer_is_reportable` already
# lives, which is the check that refuses an unearned claim. A rule and the assertion that
# enforces it belong in one file, or the two can be edited apart.

#: What a manifest says when it makes no coverage claim at all. A key, not an omission:
#: "this run claimed nothing" and "this manifest predates the field" must stay distinguishable
#: to a reader filtering a directory of them.
NO_COVERAGE_CLAIM = "no coverage claim is made — see answer.status"


def attach_coverage_claim(answer: dict, *, gate_validated: bool, ledger: dict,
                          ungated_basis: str) -> None:
    """Everything this runtime says about coverage, derived from the gate and nothing else.

    A coverage ledger asserts "I searched the universe this spec defines". The only thing
    that establishes that is the proof obligation, and the only thing that evaluates the
    proof obligation is `ChartReviewAgent._gate`. So the ledger is attached on the branch the
    gate accepted, and on the other branch the answer says in words that it makes no claim —
    because downstream an unearned ledger is indistinguishable from an earned one, which is
    the entire failure mode.

    Written ONTO THE ANSWER, exactly where `graph._n_finalize` writes it. The manifest used
    to carry a top-level `coverage_attested` and the answer to carry nothing, which put the
    claim outside the reach of `assert_answer_is_reportable` — so the one rule that says who
    may claim coverage was never asked. It is asked now, and it refuses in both directions:
    an unearned ledger raises, and a gate-validated negative WITHOUT its ledger raises too.

    Only EVIDENCE_INSUFFICIENT belongs here. FOUND is proved by witness and never claimed the
    universe was searched; SPEC_INSUFFICIENT is not a claim about this chart at all.
    """
    if gate_validated:
        answer["negative_basis"] = "GATE_VALIDATED"
        answer["coverage_attested"] = ledger
        return
    # `ungated_basis` and not a literal: a negative that never passed the gate still owes the
    # reader WHY it ended, and every value but GATE_VALIDATED routes to a human.
    answer["negative_basis"] = ungated_basis
    answer["route_to_human"] = True
    answer["coverage_note"] = ("no coverage claim is made — this answer did not pass the "
                               "proof obligation")
