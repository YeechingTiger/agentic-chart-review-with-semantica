"""Pick out the statements where a different defensible clinical choice exists, and name who
made each one.

Not every made-up element is a decision. If section 5 listed all of them it would be section
6 with numbers, and a registrar with ten minutes would read neither. The filter is: could a
competent clinician choose otherwise, and would the outputs differ?
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .measurements import SAMPLING_ARITHMETIC, measurement_for
from .prose import (
    format_is_registry_notation,
    human_field,
    human_list,
    lower_first,
    plain,
    sentence,
)
from .signoff import SIGNED, STALE, UNSIGNED
from .statements import (
    MODEL_AUTHORED,
    Element,
    elements,
    for_negative,
    group_sentence,
    n_fields,
    sample_size,
    source_groups,
)


@dataclass(frozen=True)
class Decision:
    """A choice with a defensible alternative, phrased so a clinician can refuse it."""
    element_id: str
    question: str
    choice: str
    who: str
    basis: str
    if_you_disagree: str
    fires: str = ""
    rank: int = 50


def who_decided(el: Element, spec, status: str, record: dict | None) -> str:
    if status == SIGNED and record:
        return f"Confirmed by {record['reviewer']} on {record['signed_at'][:10]}."
    if status == STALE and record:
        return (f"{record['reviewer']} confirmed a different wording on "
                f"{record['signed_at'][:10]}; that approval no longer applies to the text "
                f"below.")
    if el.provenance == MODEL_AUTHORED and not el.recorded:
        return ("Nobody clinical, and the origin is not recorded. Nothing in the file says "
                "where this sentence came from, so it is treated as model-authored — that is "
                "the safe reading, not a checked one.")
    if el.provenance == MODEL_AUTHORED:
        return ("Nobody clinical. The file records that a language model drafted this and no "
                "reviewer has confirmed it."
                + (f" {sentence(el.attributed_to)}" if el.attributed_to else ""))
    if el.provenance == "source_authority":
        doc = (spec.source_authority or {}).get("document", "the cited standard")
        return f"Taken from {doc}." + (f" Recorded by {el.attributed_to}." if el.attributed_to else "")
    if el.provenance == "measured":
        return "Chosen from a measurement on this corpus rather than from a guideline."
    who = f"Recorded provenance: {el.provenance}."
    return who + (f" By {el.attributed_to}." if el.attributed_to else "")


def decisions(spec, source_path: str | Path | None = None,
              els: Sequence[Element] | None = None) -> list[Decision]:
    """The subset of elements where a different defensible clinical choice exists.

    Not every made-up element is a decision. If section 5 listed all of them it would be
    section 6 with numbers, and a registrar with ten minutes would read neither. The filter
    is: could a competent clinician choose otherwise, and would the outputs differ? A
    restatement of a CoC coding rule fails that test; a tolerance, a witness restriction, a
    hedge-acceptance rule, an imputation constant and a settled boundary ruling all pass it.
    """
    els = list(els if els is not None else elements(spec, source_path=source_path))
    by_id = {e.element_id: e for e in els}
    fn = for_negative(spec)
    out: list[Decision] = []

    def push(eid: str, question: str, choice: str, consequence: str, rank: int, fires: str = ""):
        el = by_id.get(eid)
        out.append(Decision(
            element_id=eid, question=plain(question), choice=plain(choice),
            who=who_decided(el, spec, UNSIGNED, None) if el else "Nobody clinical.",
            basis=(el.basis if el else ""), if_you_disagree=plain(consequence),
            fires=fires, rank=rank))

    # 1. A placeholder is the reviewer's question by construction, so it goes first.
    for g in source_groups(spec):
        sched = g.raw.get("surveillance_schedule")
        if isinstance(sched, str) and "PLACEHOLDER" in sched.upper():
            push(f"source-group.{g.index}",
                 "How often must a patient have been seen for us to believe that no "
                 "recurrence happened between visits?",
                 "Undecided. The specification ships an explicit placeholder here and the "
                 "software has no follow-up schedule to work from, so this half of the "
                 "criterion cannot run at all.",
                 "Everything: give a schedule — for example three-monthly for two years, "
                 "then six-monthly to year five, then yearly — and a gap longer than the "
                 "interval becomes a gap in evidence rather than a silent assumption that "
                 "nothing happened. A constant interval is indefensible in both directions: "
                 "too coarse in the first two years, too fine after five.",
                 rank=0)

    # 2. Who may say what. This is the decision that was wrong on P03.
    for g in source_groups(spec):
        eid = f"source-group.{g.index}"
        if g.raw.get("establishes") == []:
            push(eid,
                 "Is it right that nothing in this group of documents can ever contribute to "
                 "any part of the answer?",
                 f"{g.label}. We treat every one of these as incapable of contributing, and "
                 f"we only sample them to check that assumption.",
                 "If any of these can carry a finding that matters — a spine film showing a "
                 "bone metastasis, say — then sampling it will keep rejecting good answers, "
                 "and the honest fix is to move that type into the group we search.",
                 rank=5)
        elif g.establishes and len(g.establishes) < n_fields(spec):
            missing = [f.name for f in spec.fields if f.name not in g.establishes]
            push(eid,
                 f"May {lower_first(g.label)} be used to establish "
                 f"{human_list([human_field(x) for x in g.establishes])} — and is it right "
                 f"that they may not be used for "
                 f"{human_list([human_field(x) for x in missing])}?",
                 f"Yes to the first, no to the second. {group_sentence(g)}",
                 "If you say imaging may not be used to say where the tumour started, the "
                 "answer becomes the unspecified-site code whenever the pathology report "
                 "says only \"right lung\" — which is most of them. If you say it may be "
                 "used for more than this, that evidence is currently being thrown away.",
                 rank=10)

    # 3. The words we must have searched before we are allowed to say "not documented".
    terms_ids = [e.element_id for e in els if e.kind == "search-terms"]
    for eid in terms_ids:
        el = by_id[eid]
        m = measurement_for(spec, list(el.raw))
        push(eid,
             "Are these the words a clinician at this hospital would actually use to state "
             "this answer?",
             el.text,
             "Add a word and more of the chart gets read before we are allowed to say the "
             "answer is absent; remove one and less does. This list is the whole of what "
             "stands between a real answer and a \"not documented\".",
             rank=15,
             fires=(m.text.split("\n\n")[0] if m else ""))

    # 4. The tolerance, shown with the sample size that produces it.
    if "certainty" in by_id:
        cap = (fn.get("gate") or {}).get("max_elusion_upper")
        n = sample_size(spec)
        push("certainty",
             f"Is a residual {cap * 100:.0f}% chance that a relevant document was missed "
             f"acceptable for this variable?" if cap is not None else
             "Is this level of certainty acceptable for this variable?",
             (by_id['certainty'].text + f" Reading {n} documents at random and finding nothing "
              f"is what produces that number." if n else by_id["certainty"].text),
             "Lower it and more documents must be read for every patient; below about 11% it "
             "cannot be met at all at the current sample size, and every answer fails. Raise "
             "it and \"not documented\" gets easier to claim.",
             rank=20, fires=SAMPLING_ARITHMETIC)

    # 5..6. The mechanical checks, grouped by kind. One decision per kind, not per rule:
    # four separate items about undivided T categories are one clinical question wearing four
    # hats, and splitting them is how a ten-minute document becomes a forty-minute one.
    checks = [(e, e.raw) for e in els if e.kind == "check" and isinstance(e.raw, dict)]
    for kind, question, consequence in (
        ("not_less_specific",
         "When the record hedges or is only partly specific, must we still code the more "
         "specific value?",
         "This is the live one, and it has already cost an answer. On 2026-07-28 a run coded "
         "8046 for \"poorly differentiated non-small cell carcinoma\" — the registry's answer — "
         "and the check refused it, because \"small cell\" matched inside \"non-small cell\". "
         "The negation is fixed. The clinical question is not: a report reading \"favor "
         "squamous cell carcinoma\" whose final diagnosis line stays at non-small cell "
         "carcinoma is 8070 if a hedge counts and 8046 if it does not, and nobody clinical has "
         "chosen. The codes are named here and not in the message the agent sees, because a "
         "check that supplies the answer is not checking it — the earlier message ended "
         "'\"favor squamous cell carcinoma\" supports 8070 over 8046' and a run wrote 8070 on a "
         "chart where neither word appears at all."),
        ("code_matches_cited_text",
         "May the coded subsite differ from the lobe the cited evidence names, and if so when?",
         "Remove it and a run can quote \"left lower lobe\" ten times, write \"the primary site "
         "is the left lower lobe\" in its reasoning, and code C342 (middle lobe) — which is what "
         "happened on 2026-07-28, with no check firing. Keep it and a run that believes a later "
         "document overrides the lobe an earlier one names must quote the line that says so "
         "rather than deciding silently. The lobe-to-code mapping (C341 upper, C342 middle, "
         "C343 lower, C340 main bronchus) was recalled by a model and needs checking against "
         "ICD-O-3 topography C34."),
        ("nos_requires_search",
         "Before we are allowed to record an unknown or unspecified value, which parts of the "
         "chart must have been searched?",
         "Drop the requirement and an unknown code becomes available to a run that simply "
         "did not look; add to it and more searching is forced before any unknown is "
         "accepted."),
        ("origin_not_specimen",
         "Should a value be refused when the only quote supporting it describes the specimen "
         "rather than the tumour's origin?",
         "Remove it and the biopsy site gets coded as the site of origin, which is the "
         "commonest single coding error on this variable."),
    ):
        group = [(e, c) for e, c in checks if c.get("kind") == kind]
        if not group:
            continue
        body = "\n".join(f"  - {plain(e.text)}" for e, _ in group)
        push(group[0][0].element_id, question, "Currently:\n" + body, consequence, rank=25)

    # 7. Conventions the spec invents outright (seasonal dates, and anything like them).
    for e in els:
        if e.kind == "convention":
            push(e.element_id,
                 f"Is this an acceptable convention: {lower_first(e.text).rstrip('.')}?",
                 e.text,
                 "This value is invented. It lands in a date field that feeds survival time, "
                 "so choosing a different month moves every affected patient's follow-up by "
                 "up to three months.",
                 rank=30)

    # 8. Rulings the spec declares settled.
    for e in els:
        if e.kind == "boundary":
            push(e.element_id,
                 "Is this ruling correct?",
                 e.text,
                 "The spec tells the run to follow these without further thought, so a wrong "
                 "one is wrong on every patient it touches and will never be questioned.",
                 rank=35)

    # 9. Which patients get no answer at all.
    for e in els:
        if e.kind == "not-for":
            push(e.element_id,
                 "Should these patients really be excluded from this variable?",
                 e.text,
                 "An excluded patient gets no answer and no coverage claim; they drop out of "
                 "every denominator downstream, silently, unless somebody counts them.",
                 rank=40)

    # 10. Which document wins when two disagree — a registrar's judgement, not an engineer's.
    for e in els:
        if e.kind == "conflict":
            push(e.element_id,
                 "When these two disagree, is this the right one to believe?",
                 e.text,
                 "Reverse it and the other document's value is reported instead; both are in "
                 "the chart, so this choice alone decides the answer for those patients.",
                 rank=45)

    # 11. A field whose declared shape rejects every valid value is a decision, not a typo:
    # somebody has to say what the field should actually accept.
    for f in spec.fields:
        if f.format and format_is_registry_notation(f.format):
            push(f"answer.{f.name}",
                 f"What should {human_field(f.name)} actually accept?",
                 f"As shipped, nothing. The file declares its shape as `{f.format}` — registry "
                 f"notation — and the software applies it literally, so it will reject every "
                 f"valid value of this field. Still unfixed on 2026-07-26.",
                 "Nothing downstream can use this field until it is answered. Confirm that "
                 "eight digits with an unknown month or day written as 99 is what the registry "
                 "expects, and the pattern can be written correctly.",
                 rank=1)

    out.sort(key=lambda d: (d.rank, d.element_id))
    return out
