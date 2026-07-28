"""Deterministic checks on a submitted answer.

The decision rules are already in the prompt — `as_prompt_block()` renders every one of
them — and the model still broke two of them on real charts. So this is not a prompting
gap, and another paragraph of instruction is not the fix. Where a rule can be decided
mechanically, decide it mechanically and reject through the same loop the coverage gate
already uses; the run traces show the agent revises correctly when it is told what is
wrong (`rejections: 2` on the synthetic control, then a pass).

Observed failures this exists to catch, both on 2026-07-26:

  P05  coded primary_site C340 (main bronchus) — the BIOPSY site. The report
                    said "Bronchus, distal right main mass, biopsy" and, in the same
                    document, "5 x 6 cm right hilar mass ... obstructing RUL". Rule 1 of
                    the spec: "the site of ORIGIN, not the site of a biopsy".

  P05  coded histology 8046 (NSCLC, NOS) when the pathology read "favor
                    squamous cell carcinoma" and the phrase "squamous cell carcinoma"
                    appeared 12 times. The spec's conflict rule says take the MORE
                    specific reading; it took the less specific one.

  P03  coded primary_site C349 (lung NOS) when "right upper lobe" was
                    documented across seven note types. Not caught here by accident: a NOS
                    code is a positive claim that the subsite is unknown, and that claim is
                    falsifiable against the record.

NO CLINICAL KNOWLEDGE LIVES IN THIS FILE. Which values are "NOS", and which phrases
contradict them, are declared per criterion in the spec under `answer_checks`. That keeps
the checker reusable across criteria and sites, and keeps the oncology in the place a
domain expert can review it.

EVERY REJECTION NAMES ITS RULE. `check_answer` returned bare strings, so the fact of which
check fired — known here, for certain, with no model in the loop — was thrown away one line
after it was computed. §6b's optimizer then has to infer from a message which spec rule was
in play, and a loop that infers attribution from outcomes rewrites text that was never at
fault. `check_answer_detail` returns `Violation`s carrying the rule id, the coded value, the
declared phrase that fired and the quote it fired on; `check_answer` is now a one-line
projection of it, because two implementations of one check is how they drift apart.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .spec import _answer_check_key


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# --------------------------------------------------------------------------- rule identity
# The id functions live here, next to the code that applies the rules, and are imported by
# `acr.trace` rather than re-derived there. Two places minting ids for one rule is the same
# failure as two ledgers counting one run: they agree until they do not, nothing raises, and
# an attribution ends up pointing at a rule nobody can find.

def answer_check_rule_id(chk: Any, position: int | None = None) -> str:
    """`answer_check.<field>.<kind>[.<first nos value>]`, from the check's CONTENT.

    Reuses `spec._answer_check_key` — the identity the provenance channel and `specview`
    already use — so a rejection recorded in a trace names the same string a sign-off names.
    Content, not position, because inserting a check at the top of the list would otherwise
    silently re-point every recorded attribution below it at a different rule.
    """
    if isinstance(chk, dict):
        return f"answer_check.{_answer_check_key(chk)}"
    # A malformed entry still needs an id, or its rejections become unattributable. Position
    # is the only handle left, and it is marked as such.
    return f"answer_check.unparsed#{position if position is not None else 0}"


def field_rule_id(kind: str, name: str) -> str:
    """`field_format.<field>` / `field_allowable_values.<field>`."""
    return f"{kind}.{name}"


@dataclass(frozen=True)
class Violation:
    """One rejection, with everything needed to attribute it without re-reading the run.

    `trigger` and `quote` are the two facts an optimizer needs and a message alone loses:
    which declared phrase fired, and the text it fired on. Without them "the answer
    contradicts the specification's decision rules" is a verdict with no subject, and the
    reflection model is left to guess which of nine `contradicted_by` phrases was involved.
    """

    rule_id: str
    rule_kind: str
    field: str
    coded_value: str
    message: str
    trigger: str = ""
    #: The cited quote the trigger was found in. Chart text, so it stays in the trace (which
    #: already holds every document the agent read) and is referenced by index, not copied,
    #: anywhere it would be a second copy.
    quote: str = ""
    evidence_index: int = -1

    def to_dict(self, with_quote: bool = False) -> dict:
        d = {"rule_id": self.rule_id, "rule_kind": self.rule_kind, "field": self.field,
             "coded_value": self.coded_value, "trigger": self.trigger,
             "evidence_index": self.evidence_index, "message": self.message}
        if with_quote:
            d["quote"] = self.quote
        return d


def _evidence_for(field: str, evidence: Iterable[dict]) -> list[dict]:
    """All cited evidence. The `field` argument is accepted and deliberately ignored.

    An earlier version narrowed to items whose free-text `supports` label contained the
    field name. That silently defeated the whole check, and it did so on the first live
    run: the agent labelled its three quotes

        "Histology and malignant behaviour: ..."                      <- matched
        "Final pathology diagnosis and specimen site: ..."            <- missed
        "Final pathologic diagnosis establishes morphology and ..."   <- missed

    so `histology` scoping kept exactly one quote — about an electronic signature — and
    the answer passed while coding 8046 over a report that said "favor squamous cell
    carcinoma". Substring-matching a model-authored label is the same mistake as
    substring-matching a note type, and it fails the same way.

    The question these checks ask is ledger-wide anyway: does ANYTHING the agent chose to
    cite contradict the value it coded? Narrowing can only lose evidence, never add it, and
    a false pass here is worse than a false rejection — a rejection costs one more step,
    a false pass ships a wrong code with a validated stamp on it.
    """
    return list(evidence)


def _first_quote_containing(items: list[dict], needle: str) -> tuple[int, str]:
    """Which cited quote actually fired the rule. (-1, "") when no single item did."""
    n = _norm(needle)
    for i, e in enumerate(items):
        if n and n in _norm(e.get("quote")):
            return i, str(e.get("quote") or "")
    return -1, ""


def check_answer(checks: list[dict], value: dict, evidence: list[dict],
                 searched: Iterable[str] = ()) -> list[str]:
    """The messages only. Kept as the published signature — `graph`, `refine.blast_radius`
    and a dozen tests call it — with `check_answer_detail` underneath for anything that needs
    to know WHICH rule spoke."""
    return [v.message for v in check_answer_detail(checks, value, evidence, searched)]


def check_answer_detail(checks: list[dict], value: dict, evidence: list[dict],
                        searched: Iterable[str] = ()) -> list[Violation]:
    """Return a list of `Violation`s; empty means the answer passes.

    Check kinds:

      not_less_specific   The coded value is a catch-all/NOS code, but the cited evidence
                          contains a phrase implying something more specific. Coding NOS is
                          an assertion that the specific value is unknown, so a contradicting
                          phrase in the record falsifies it.

      origin_not_specimen Every piece of evidence offered for this field is a specimen or
                          biopsy header. That locates where tissue was taken from, which is
                          not the same claim as where the tumour arose.

      conflict_requires_nos
                          The cited evidence names TWO OR MORE mutually exclusive members of
                          one set, and the answer picked one of them without saying why the
                          other is wrong. This is the mirror image of `not_less_specific` and
                          it needed its own kind: that check protects against coding NOS when
                          the record is specific, and nothing protected against coding
                          specific when the record disagrees with itself. Measured on a real
                          chart — an operative note said the lesion was "coming and arising
                          from the right middle lobe" while the pathology header read "Lung,
                          right lower lobe", both quoted in the SAME answer, and the run coded
                          C342 and passed the gate. The registry coded C349. A coder who
                          cannot tell which lobe codes the NOS value; an agent that quietly
                          picks the first one it read is not abstracting, it is guessing.
    """
    out: list[Violation] = []
    for pos, chk in enumerate(checks or [], start=1):
        field = chk.get("field")
        if not field:
            continue
        coded = str(value.get(field) or "").strip()
        if not coded:
            continue
        items = _evidence_for(field, evidence)
        blob = " ".join(_norm(e.get("quote")) for e in items)
        kind = chk.get("kind", "not_less_specific")
        rid = answer_check_rule_id(chk, pos)

        if kind == "not_less_specific":
            if coded not in (chk.get("nos_values") or []):
                continue
            found = [p for p in (chk.get("contradicted_by") or []) if _norm(p) in blob]
            if found:
                idx, quote = _first_quote_containing(items, found[0])
                out.append(Violation(
                    rule_id=rid, rule_kind=kind, field=field, coded_value=coded,
                    trigger=found[0], quote=quote, evidence_index=idx,
                    message=(
                        f"{field}={coded} is a not-otherwise-specified value, but the evidence "
                        f"you cited contains {found[:4]}. A NOS code asserts the specific value "
                        f"is unknown; the record contradicts that. "
                        + (chk.get("message") or "Code the more specific value, or cite why it "
                                                 "does not apply."))))

        elif kind == "nos_requires_search":
            # A not-otherwise-specified code is a NEGATIVE CLAIM: "the specific value is not
            # documented". This project's whole position is that a negative needs proof the
            # agent looked, so apply it at field granularity too — otherwise NOS is the free
            # escape hatch that abstention is explicitly not allowed to be.
            #
            # This is the only check that can catch patient P03. It coded C349
            # citing pathology that said just "Right lung", while "right upper lobe" sat in
            # seven other note types. Nothing in the cited ledger contradicted C349, so
            # `not_less_specific` passed it — the failure was never looking, and looking is
            # exactly what `searched_terms` records.
            if coded not in (chk.get("nos_values") or []):
                continue
            done = {_norm(t) for t in searched}
            need = [t for t in (chk.get("required_searches") or [])
                    if not any(_norm(t) in d for d in done)]
            if need:
                out.append(Violation(
                    rule_id=rid, rule_kind=kind, field=field, coded_value=coded,
                    # The trigger here is an ABSENCE — the searches never run — so there is
                    # no quote to point at, and recording one would be inventing a witness.
                    trigger=", ".join(need), quote="", evidence_index=-1,
                    message=(
                        f"{field}={coded} claims the specific value is not documented, but you "
                        f"never searched for {need}. Search those terms; if they genuinely return "
                        f"nothing, {coded} is then supported. "
                        + (chk.get("message") or ""))))

        elif kind == "conflict_requires_nos":
            nos = str(chk.get("nos_value") or "").strip()
            groups = [g for g in (chk.get("mutually_exclusive") or []) if g]
            if not nos or len(groups) < 2 or coded == nos:
                # Already coded NOS: the answer has conceded the conflict, which is the whole
                # remedy. Firing here would reject the correct answer.
                continue
            # Which groups the CITED evidence names. Deliberately over the cited quotes only,
            # never over the chart: a check that reads documents the answer did not cite is
            # asking the agent to defend text it never saw.
            present = [(g, next(a for a in g if _norm(a) in blob))
                       for g in groups if any(_norm(a) in blob for a in g)]
            if len(present) > 1:
                names = [alias for _, alias in present]
                idx, quote = _first_quote_containing(items, names[0])
                out.append(Violation(
                    rule_id=rid, rule_kind=kind, field=field, coded_value=coded,
                    trigger=", ".join(names), quote=quote, evidence_index=idx,
                    message=(
                        f"the evidence you cited for {field} names {names}, which cannot all "
                        f"be true of one tumour, and you coded {coded} anyway without saying "
                        f"why the others are wrong. Either cite the statement that settles it "
                        f"— which document is describing the origin and which is describing a "
                        f"specimen, a second site or an error — or code {nos}. "
                        + (chk.get("message") or ""))))

        elif kind == "origin_not_specimen":
            markers = [_norm(m) for m in (chk.get("specimen_markers") or [])]
            if not markers or not items:
                continue
            per_item = [any(m in _norm(e.get("quote")) for m in markers) for e in items]
            if per_item and all(per_item):
                hit = next((m for m in markers if m in _norm(items[0].get("quote"))), "")
                out.append(Violation(
                    rule_id=rid, rule_kind=kind, field=field, coded_value=coded,
                    trigger=hit, quote=str(items[0].get("quote") or ""), evidence_index=0,
                    message=(
                        f"every quote you cited for {field} is a specimen/biopsy header. That "
                        f"establishes where tissue was taken, not where the tumour arose. "
                        + (chk.get("message") or "Cite a statement of the site of ORIGIN."))))
    return out


def check_field_formats(fields, value: dict) -> list[str]:
    """The messages only; `check_field_formats_detail` is the attributable form."""
    return [v.message for v in check_field_formats_detail(fields, value)]


def check_field_formats_detail(fields, value: dict) -> list[Violation]:
    """Enforce the `format` and `allowable_values` the spec already declares per field.

    These were rendered into the prompt and never checked. On 2026-07-26 a deepagents run
    submitted primary_site="C3412" -- four digits where the spec declares C\\d{3} -- and it
    passed the gate, stamped as validated, with zero rejections. A malformed code is not a
    judgement call the model is entitled to make; it is a contract violation, decidable
    without any clinical knowledge, and it should never have reached a human.

    Unlike answer_checks this needs no spec configuration: the constraints are already in
    `fields`. It is always on, for every criterion, at no authoring cost.
    """
    out: list[Violation] = []
    for f in fields or []:
        name = getattr(f, "name", None)
        if not name:
            continue
        raw = value.get(name)
        if raw is None or str(raw).strip() == "":
            continue          # absent is abstention's business, not the format checker's
        s = str(raw).strip()
        fmt = getattr(f, "format", None)
        if fmt:
            try:
                if not re.fullmatch(fmt, s):
                    out.append(Violation(
                        rule_id=field_rule_id("field_format", name), rule_kind="field_format",
                        field=name, coded_value=s, trigger=str(fmt),
                        message=f"{name}={s!r} does not match the required format {fmt!r}."))
            except re.error:
                pass          # a broken pattern in the spec must not block the run
        allowed = getattr(f, "allowable_values", None)
        if allowed and s not in [str(v) for v in allowed]:
            out.append(Violation(
                rule_id=field_rule_id("field_allowable_values", name),
                rule_kind="field_allowable_values", field=name, coded_value=s,
                trigger="", message=(f"{name}={s!r} is not one of the allowed values "
                                     f"{list(allowed)!r}.")))
    return out
