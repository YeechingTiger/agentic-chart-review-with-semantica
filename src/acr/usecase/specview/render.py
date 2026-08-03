"""Lay the reviewable statements out as the markdown document a clinician actually reads.

Ordering is the whole content of this module. Sections 5 and 6 come before the machinery
because they are the two a registrar with ten minutes must reach, the appendix exists so that
no key in the file can be quietly left out, and section 6 is printed in full however long it
gets — capping a long list at ten items with "and 47 more" is exactly the softening it exists
to refuse.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .decisions import decisions, who_decided
from .measurements import CORPUS_HEADER, MEASUREMENTS, SAMPLING_ARITHMETIC, UNMEASURED_NOTE
from .prose import gloss_codes, plain, quoted, sentence
from .signoff import SIGNED, STALE, UNSIGNED, signoff_status
from .statements import (
    MODEL_AUTHORED,
    Element,
    all_terms,
    editorial_problems,
    elements,
    keyword_lists,
    sample_size,
)

SECTION_TITLES = (
    "1. What this answers, and the patients it does not cover",
    "2. What we will accept as proof, and what we will refuse",
    "3. What we do when the chart contradicts itself",
    "4. When we refuse to answer, and what that refusal means downstream",
    "5. DECISIONS WE NEED YOU TO CONFIRM",
    "6. WHAT WE MADE UP",
    "7. HOW OFTEN THIS FIRES",
)


def render_review(spec, source_path: str | Path | None = None,
                  signoffs: Sequence[dict] = ()) -> str:
    els = elements(spec, source_path=source_path)
    ds = decisions(spec, source_path=source_path, els=els)
    L: list[str] = []
    A = L.append

    A(f"# {plain(spec.question).rstrip('.')}")
    A("")
    A(f"A review copy of the extraction specification `{spec.spec_id}` "
      f"(version {spec.spec_version}, content fingerprint `{spec.spec_hash}`).")
    A("")
    A("You are being asked to mark this up. Section 5 is the list of choices we need you to "
      "confirm or overturn; section 6 is everything in here that we invented and nobody "
      "clinical has ever checked. If you read only two sections, read those two.")
    A("")

    _s1(A, spec, els)
    _s2(A, spec, els, signoffs)
    _s3(A, spec, els, signoffs)
    _s4(A, spec, els, signoffs)
    _s5(A, spec, ds, els, signoffs)
    _s6(A, spec, els, signoffs)
    _s7(A, spec)
    _appendix(A, spec, source_path)

    body = "\n".join(L)
    return gloss_codes(body) + "\n"


def _h(A, i: int) -> None:
    A("")
    A(f"## {SECTION_TITLES[i]}")
    A("")


def _pick(els: Sequence[Element], *kinds: str) -> list[Element]:
    return [e for e in els if e.kind in kinds]


def _mark(el: Element, signoffs: Sequence[dict]) -> str:
    status, rec = signoff_status(el, signoffs)
    if status == SIGNED:
        return f"  _[{el.element_id} — confirmed by {rec['reviewer']}, {rec['signed_at'][:10]}]_"
    if status == STALE:
        return (f"  _[{el.element_id} — {rec['reviewer']} confirmed a different wording on "
                f"{rec['signed_at'][:10]}; that approval no longer applies]_")
    return f"  _[{el.element_id}]_"


def _bullets(A, els: Sequence[Element], signoffs: Sequence[dict], empty: str) -> None:
    if not els:
        A(f"_{empty}_")
        A("")
        return
    for e in els:
        A(f"- {sentence(e.text)}{_mark(e, signoffs)}")
    A("")


def _s1(A, spec, els) -> None:
    _h(A, 0)
    if spec.data_source == "outside_notes":
        A("> **This variable is not in the chart at all.** It is a fact about the patient's "
          "relationship to this hospital — where they were diagnosed and where they were "
          "treated — and it lives in registration, referral and billing systems. No amount "
          "of reading notes can establish it. The run is required to refuse it and to report "
          "clues only; a human registrar assigns the code.")
        A("")
    by = {e.element_id: e for e in els}
    A(f"**The question.** {sentence(spec.question)}")
    A("")
    if "authority" in by:
        A(f"{sentence(by['authority'].text)}")
        A("")
    if "guidance" in by:
        A(f"**Standing instruction to the reviewer software.** {sentence(by['guidance'].text)}")
        A("")
    if "proof.not-applicable" in by:
        A(f"{sentence(by['proof.not-applicable'].text)}")
        A("")
    A("**What comes out.**")
    A("")
    for e in _pick(els, "answer"):
        A(f"- {e.text}")
    A("")
    A("**Patients this does not cover.**")
    A("")
    nf = _pick(els, "not-for")
    if nf:
        for e in nf:
            A(f"- {sentence(e.text)}")
    else:
        A("_The specification names no patients it excludes. If there are any, nobody has "
          "written them down, and the run will attempt an answer for every patient it is "
          "given._")
    A("")


def _s2(A, spec, els, signoffs) -> None:
    _h(A, 1)
    by = {e.element_id: e for e in els}
    if "proof.positive" in by:
        A(f"**What is enough to give an answer.** {sentence(by['proof.positive'].text)}"
          f"{_mark(by['proof.positive'], signoffs)}")
        A("")
    wit = [e for e in els if e.element_id.startswith("proof.witness.")]
    if wit:
        A("**And the answer must be quoted from the right kind of document:**")
        A("")
        _bullets(A, wit, signoffs, "")
    A("**What we accept as evidence.**")
    A("")
    _bullets(A, _pick(els, "accept"), signoffs,
             "The specification never says what counts as evidence. Anything the run chooses "
             "to quote will be accepted.")
    A("**What we refuse.**")
    A("")
    _bullets(A, _pick(els, "refuse"), signoffs,
             "The specification never says what does not count.")
    A("**The coding rules the run is told to follow.**")
    A("")
    _bullets(A, _pick(els, "rule"), signoffs, "None are stated.")

    groups = _pick(els, "source-group")
    if groups:
        A("**How much of the chart must be read before we are allowed to say \"it is not "
          "documented\".** Every document in the chart falls into exactly one of these groups.")
        A("")
        for g in groups:
            A(f"- {sentence(g.text)}{_mark(g, signoffs)}")
            for suffix in (".terms", ".sample"):
                sub = by.get(g.element_id + suffix)
                if sub:
                    A(f"    - {sentence(sub.text)}{_mark(sub, signoffs)}")
        A("")
    for eid in ("search-terms", "certainty"):
        if eid in by:
            A(f"- {sentence(by[eid].text)}{_mark(by[eid], signoffs)}")
    if "search-terms" in by or "certainty" in by:
        A("")
    claims = _pick(els, "claim")
    if claims:
        A("**This answer has two halves and both must be proved.**")
        A("")
        _bullets(A, claims, signoffs, "")
    if "hints" in by:
        A(f"{sentence(by['hints'].text)}{_mark(by['hints'], signoffs)}")
        A("")


def _s3(A, spec, els, signoffs) -> None:
    _h(A, 2)
    _bullets(A, _pick(els, "conflict"), signoffs,
             "The specification says nothing about contradictions. Whichever document the run "
             "happens to read first will decide the answer, and nothing records that there "
             "was a disagreement.")


def _s4(A, spec, els, signoffs) -> None:
    _h(A, 3)
    by = {e.element_id: e for e in els}
    A("There are two different refusals and they mean different things to whoever uses this "
      "data:")
    A("")
    _bullets(A, _pick(els, "refusal"), signoffs, "Neither refusal is described.")
    if "proof.negative" in by:
        A(f"**What the run is told at the point of refusing.** "
          f"{sentence(by['proof.negative'].text)}{_mark(by['proof.negative'], signoffs)}")
        A("")
    caut = _pick(els, "caution")
    if caut:
        A("**Why the missing values cannot be ignored downstream.**")
        A("")
        _bullets(A, caut, signoffs, "")
    down = _pick(els, "downstream")
    if down:
        A("**Warnings attached to this variable for anyone analysing it.**")
        A("")
        _bullets(A, down, signoffs, "")
    if not caut and not down:
        A("_Nothing is recorded about what a refusal does to an analysis that uses this "
          "variable. A refused answer is not missing at random — it clusters on the patients "
          "whose care was fragmented — and no warning here means nobody downstream will be "
          "told._")
        A("")


def _s5(A, spec, ds, els, signoffs) -> None:
    _h(A, 4)
    if not ds:
        A("_Nothing in this specification was identified as a clinical choice with a "
          "defensible alternative. Treat that as a finding about this document, not as "
          "reassurance._")
        A("")
        return
    by = {e.element_id: e for e in els}
    A(f"{len(ds)} of them. They are ordered by how much they change, not by where they sit "
      f"in the file. If you have ten minutes, the first three are the ones that move the most "
      f"answers.")
    A("")
    for n, d in enumerate(ds, start=1):
        el = by.get(d.element_id)
        status, rec = signoff_status(el, signoffs) if el else (UNSIGNED, None)
        A(f"### {n}. {d.question}")
        A("")
        A(f"- **What we do now:** {d.choice}")
        A(f"- **Who decided:** {who_decided(el, spec, status, rec) if el else d.who}")
        if d.basis:
            A(f"- **On what basis:** {sentence(d.basis)}")
        else:
            A("- **On what basis:** nothing is recorded.")
        A(f"- **If you decide otherwise:** {d.if_you_disagree}")
        if d.fires:
            A(f"- **How often it matters:** {d.fires}")
        A(f"- To confirm this one: `acr spec signoff --spec {spec.spec_id} --reviewer "
          f"\"your name\" --element {d.element_id}`")
        A("")


def _s6(A, spec, els, signoffs) -> None:
    _h(A, 5)
    mine = [e for e in els if e.provenance == MODEL_AUTHORED]
    silent = [e for e in mine if not e.recorded]
    total = len(els)
    A(f"**{len(mine)} of the {total} statements in this specification have no source outside "
      f"this repository.** A language model wrote them, and the fact that a standards manual "
      f"is named at the top of the file is not evidence that any particular sentence below "
      f"came out of it.")
    A("")
    if silent:
        A(f"Of those, **{len(silent)} have no provenance record at all** — marked _origin not "
          f"recorded_ below. Model-authored is the default the software applies when nobody "
          f"has written anything down, and a default is not a finding: nobody has checked "
          f"where these came from either way. The remaining {len(mine) - len(silent)} carry a "
          f"record that says outright that a model wrote them.")
        A("")
    _s6_findings(A, spec, els)
    if not mine:
        A("_Every element in this specification carries a recorded source. Check the sources, "
          "not this list._")
        A("")
        return
    A("This section is the reason the document exists. It is long because the state of the "
      "specification is that nothing has been attributed, not because the list has been "
      "padded. It shrinks as soon as somebody records where a statement came from — either "
      "by signing it off below, or by naming the section it was taken from in the "
      "`editorial_provenance:` block of the spec file. A label with no section, item or rule "
      "number does not shrink it, and that is deliberate.")
    A("")
    order = ["question", "answer", "authority", "guidance", "rule", "accept", "refuse",
             "proof", "source-group", "search-terms", "sampling", "certainty", "claim",
             "conflict", "refusal", "caution", "downstream", "boundary", "check",
             "convention", "not-for", "hints"]
    labels = {
        "question": "The question itself",
        "answer": "The fields that come out, and what values they may take",
        "authority": "The standard this claims to follow",
        "guidance": "Standing instructions to the software",
        "rule": "Coding rules",
        "accept": "What counts as evidence",
        "refuse": "What does not count as evidence",
        "proof": "What must be proved before an answer or a refusal",
        "source-group": "How the chart is divided up, and which documents may be cited",
        "search-terms": "The words we search for",
        "sampling": "How many documents we read at random",
        "certainty": "How certain we require ourselves to be",
        "claim": "The separate halves of the answer",
        "conflict": "Which document wins when two disagree",
        "refusal": "What each refusal means",
        "caution": "Warnings about missing values",
        "downstream": "Warnings to whoever analyses this",
        "boundary": "Settled rulings on specific situations",
        "check": "Automatic rejections applied to a submitted answer",
        "convention": "Conventions invented outright",
        "not-for": "Patients excluded",
        "hints": "Suggested search terms",
    }
    for kind in order + sorted({e.kind for e in mine} - set(order)):
        group = [e for e in mine if e.kind == kind]
        if not group:
            continue
        A(f"**{labels.get(kind, kind)}**")
        A("")
        for e in group:
            tag = "" if e.recorded else " _(origin not recorded)_"
            A(f"- `{e.element_id}` — {sentence(e.text)}{tag}{_mark(e, signoffs)[2:]}")
        A("")


def _s6_findings(A, spec, els) -> None:
    """Editorial records that do not hold up, printed where the reader can act on them.

    Only the record-shape complaints appear here. The missing-record finding is not repeated
    as a list, because it IS the list below — every statement in section 6 tagged _(origin not
    recorded)_ is one, and printing them twice would bury the four that are different.
    """
    findings = editorial_problems(spec)
    if not findings:
        return
    A(f"**{len(findings)} attribution(s) in this file do not hold up and have been "
      f"ignored.** An attribution that fails its own rules is worse than none: it reads as a "
      f"source to anybody skimming. The statements they named are still listed below as "
      f"unattributed.")
    A("")
    for f in findings:
        A(f"- `{f.element}` — {plain(f.problem)}")
    A("")


def _s7(A, spec) -> None:
    _h(A, 6)
    A(CORPUS_HEADER)
    A("")
    terms = all_terms(spec)
    printed = False

    for m in MEASUREMENTS:
        if m.only_spec and m.only_spec != spec.spec_id:
            continue
        if m.measured_list is not None:
            lists = keyword_lists(spec)
            if not any(tuple(x) == m.measured_list for x in lists):
                A(f"> **A measurement was made on this criterion and no longer describes it.** "
                  f"The numbers below were measured against the required terms "
                  f"{quoted(m.measured_list)}. This specification now requires "
                  f"{quoted(sorted({t for x in lists for t in x})) or 'a different list'}, so "
                  f"the result has been withheld rather than reprinted beside a list it was "
                  f"not measured on. Re-run the measurement.")
                A("")
                printed = True
                continue
        if m.needs_terms and not all(t in terms for t in m.needs_terms):
            continue
        A(m.text)
        A("")
        printed = True

    if sample_size(spec):
        A(SAMPLING_ARITHMETIC)
        A("")
        printed = True
    if not printed or all(m.only_spec != spec.spec_id for m in MEASUREMENTS):
        A(UNMEASURED_NOTE)
        A("")
        shared = sorted(set(terms) & {t for m in MEASUREMENTS if m.measured_list
                                      for t in m.measured_list})
        if shared:
            A(f"This criterion shares {quoted(shared)} with the list that failed.")
            A("")


def _appendix(A, spec, source_path) -> None:
    A("")
    A("## Appendix — where every part of the specification file went")
    A("")
    A("So that nothing in the file can be quietly left out of this document.")
    A("")
    routed = {
        "spec_id": "the header of this document",
        "spec_version": "the header of this document",
        "source_authority": "section 1",
        "data_source": "section 1",
        "question": "section 1 and the title",
        "fields": "section 1, \"what comes out\"",
        "agent_policy": "section 1",
        "when_not_to_use": "section 1 and section 5",
        "decision_rule": "section 2",
        "evidence_rules": "section 2",
        "proof_obligation": "sections 2, 4 and 5",
        "search_hints": "section 2",
        "conflict_rules": "sections 3 and 5",
        "abstention": "section 4",
        "special_codes_not_mar": "section 4",
        "downstream_warning": "section 4",
        "boundary_cases": "section 5",
        "answer_checks": "section 5",
        "date_imputation": "section 5",
        "provenance": "section 6 (it decides what is listed there)",
        "editorial_provenance":
            "section 6 (it decides what is listed there). This is the advisory half: the "
            "runtime never reads it, a missing entry is reported rather than refused, and an "
            "entry that names no section does not take its statement off the list",
        "keyword_field_coverage":
            "nowhere — it is an engineering cross-check that the search terms reach every "
            "field, asserted in the test suite, and carries no clinical decision",
    }
    keys: list[str] = []
    if source_path:
        import yaml
        raw = yaml.safe_load(Path(source_path).read_text(encoding="utf-8")) or {}
        keys = list(raw)
    for k in keys:
        A(f"- `{k}` — {routed.get(k, 'NOT TRANSLATED. This document does not know what this is; read it in the file.')}")
    A("")
