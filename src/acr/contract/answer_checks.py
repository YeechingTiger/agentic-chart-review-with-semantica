"""Format and value-domain checks on a submitted answer. NOTHING CLINICAL.

WHAT WAS HERE, AND WHY IT IS GONE
---------------------------------
This module used to carry five checks that decided clinical questions by matching word lists
against the model's own cited quotes: `not_less_specific`, `nos_requires_search`,
`conflict_requires_nos`, `origin_not_specimen`, `code_matches_cited_text`. Each was added
after a real chart went wrong, each was argued for in a long comment, and every one of them
was measured on 2026-07-30 over every trace this project has ever recorded -- 266 traces, 219
manifests, 202 traces joinable to registry gold, 122 recorded firings:

    rule                       fires   rejected the registry's own value   ever helped
    not_less_specific             22                    22  (100%)                   0
    nos_requires_search           24                    21  ( 88%)                   0
    conflict_requires_nos         67                    18  ( 27%)                  15
    origin_not_specimen            2                     0                           0
    code_matches_cited_text        0                     -                           -

Net across all five: 58 firings destroyed a correct value, 21 preceded a correct one, 39 did
nothing. Counted at the level of a whole submission, 60 of 254 recorded rejections (24%)
refused a tuple that was EXACTLY the registry's, and 12 runs held the exact registry answer
in hand and shipped something else -- 8 of those 12 shipped nothing at all.

The three that never helped once need no further argument. The other two are worth recording
precisely, because both looked defensible right up to the measurement:

`conflict_requires_nos` appeared to break even at 15 helps against 18 harms. It did not. All
15 helps were the same event -- a push to C349 -- because "or code <nos_value>" is the only
remedy its message offers. Its single mechanism is retreat to NOS, and NOS is the registry's
answer for 9.6% of this corpus while C341 alone is 52.7%. It never once helped a run reach a
specific subsite. A rule whose only move pays off at the base rate of the value it moves
toward is a biased coin, not a check.

`origin_not_specimen` fired twice and the model resubmitted the identical value both times.
That is not a check; it is a round trip.

There was also a contradiction between rules that no single rule could show. On CASE009 of the
planning ablation the runtime dropped `lobe` and `bronchus` from the plan for budget
(`fit_terms_to_budget`), and `nos_requires_search` then refused the answer because the run
"never searched for ['lobe', 'bronchus']". One rule punished the model for not running a term
another rule had deleted. That run submitted the registry-correct C341 five times, was
refused five times, and shipped C349.

THE RULE THIS MODULE NOW FOLLOWS
--------------------------------
A wrong clinical value is an instruction-following problem and belongs in the instruction, or
in the evaluation that measures instruction-following. It does not belong in a word list in
code. Word lists do not generalise: every one above was written from one chart and then
applied to every chart, and the measurement above is what that costs.

What survives is the one check with a positive record and no clinical content: the `format`
and `allowable_values` a spec already declares per field. Measured over the same traces, 7
firings, 0 that refused a registry value, 6 that preceded the registry answer. It rejects
`C3412` and `C3432` -- codes that are not codes -- and it decides that without knowing any
oncology.

Two things about it are still wrong and are NOT fixed here, because both are additions rather
than deletions:

  - 4 of its 7 useful firings rejected `C34.9`, `C34.11`, `C34.2` -- the punctuated form
    ICD-O-3 itself writes. That is a notation difference and should be normalised, not
    refused; the round trip is the check's fault, not the model's.
  - `\\d{4}` cannot tell a real morphology from a well-formed invented one. Validity needs a
    code table. A shape regex is not a code table and should stop being described as one.

EVERY REJECTION STILL NAMES ITS RULE. `Violation` carries the rule id, the coded value and
the trigger, so `acr.contract.trace` can attribute a rejection without inferring it from a message.
That machinery was never the problem and it is what made the measurement above possible.
"""
from __future__ import annotations

import calendar as _calendar
import re
from collections.abc import Callable
from dataclasses import dataclass

from .spec import _answer_check_key

#: THE KINDS `check_answer_detail` DISPATCHES ON. Deliberately empty.
#:
#: `acr.contract.spec.bind_provenance` reads this set and REFUSES to load a spec that declares an
#: answer_check kind nothing implements. Emptying it therefore does more than delete code: a
#: spec that still carries an `answer_checks:` block now fails to load instead of quietly
#: declaring rules that no longer run. That is the intended behaviour -- a check listed in
#: YAML, visible in `rule_catalog`, and never firing is indistinguishable from a check that
#: looked and found nothing, which is the failure mode this set was introduced to prevent.
#:
#: Re-adding a kind here means re-adding a word list. Before doing that, read the measurement
#: in the module docstring: five kinds, 122 firings, 58 correct values destroyed.
ANSWER_CHECK_KINDS: frozenset[str] = frozenset()

# --------------------------------------------------------------------------- rule identity
# The id functions live here, next to the code that applies the rules, and are imported by
# `acr.contract.trace` rather than re-derived there. Two places minting ids for one rule is the same
# failure as two ledgers counting one run: they agree until they do not, nothing raises, and
# an attribution ends up pointing at a rule nobody can find.

def answer_check_rule_id(chk, position: int | None = None) -> str:
    """`answer_check.<field>.<kind>[.<first nos value>]`, from the check's CONTENT.

    Retained although no kind is implemented: traces recorded before the clinical checks were
    removed name these ids, and `acr.diagnosis.attribution` still has to resolve them. Minting ids and
    running checks were always separate jobs.
    """
    if isinstance(chk, dict):
        return f"answer_check.{_answer_check_key(chk)}"
    return f"answer_check.unparsed#{position if position is not None else 0}"

def field_rule_id(kind: str, name: str) -> str:
    """`field_format.<field>` / `field_allowable_values.<field>`."""
    return f"{kind}.{name}"

@dataclass(frozen=True)
class Violation:
    """One rejection, with everything needed to attribute it without re-reading the run.

    Field order and defaults are UNCHANGED by the removal of the clinical checks. `acr.contract.trace`
    constructs and serialises these, older traces carry them, and `acr.diagnosis.attribution` reads them,
    so the shape is a recorded format rather than an internal convenience.

    `trigger` and `quote` are set by nothing that survives -- the format checks put the pattern
    in `trigger` and leave `quote` empty -- but they stay in the shape because a trace written
    before 2026-07-30 has them populated and must still deserialise.
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

def check_answer(checks, value: dict, evidence, searched=()) -> list[str]:
    """Always empty. No clinical answer_check kind is implemented -- see the module docstring.

    Kept as a published signature because `refine.blast_radius` scores a candidate rule by
    running the answer through it, and the honest answer to "what would this word list have
    changed" is now "nothing, because word lists are no longer applied".
    """
    return [v.message for v in check_answer_detail(checks, value, evidence, searched)]

def check_answer_detail(checks, value: dict, evidence, searched=()) -> list[Violation]:
    """Always empty. See `ANSWER_CHECK_KINDS` and the module docstring.

    Not deleted outright because `answer_gate.gate_answer` composes it with the format checks
    and `acr.contract.trace` records the composed result; a caller that has to branch on whether the
    function exists is worse than a function that truthfully returns nothing.
    """
    return []

# --------------------------------------------------------------------------- calendars
# THE SECOND STEP, and the reason there is a second step at all.
#
# A regex decides whether each component of a date is a legal TOKEN. It cannot decide whether
# the components together name a day that exists, because that is arithmetic: February has 28
# days except when it has 29, and which one depends on a rule about 4, 100 and 400. The
# pattern this contract shipped with,
#
#     (19|20)\d{2}(0[1-9]|1[0-2]|99)([0-2]\d|3[01]|99)
#
# accepted 29 February 2018, 31 April, and day `00`, and was described in the provenance
# record as the thing that "rejects non-date text" -- true, and not the same claim.
#
# WHERE `99` IS LEGAL is declared here rather than implied by a character class, because the
# four notations mean four different things and one of them has never been decided:
#
#     YYYYMMDD   fully known
#     YYYYMM99   the day is not recorded
#     YYYY9999   neither month nor day is recorded
#     YYYY99DD   UNDECLARED. A known day under an unknown month may be meaningful or may be a
#                transcription slip, nobody in this project has ruled, and a regex that
#                silently admitted or silently refused it would be making the ruling by
#                accident. It is refused, and the message says it is undecided rather than
#                impossible -- the two are different things to tell an author.
#
# The year slot is NEVER 99. `decision_rule[5]` requires an unidentifiable year to be
# APPROXIMATED, so an unknown year has no notation, and it must not acquire one by convention:
# `20999999` -- submitted twice in the E4 floor run and format-valid at the time -- is what a
# model does when it extends the convention one slot too far and nothing stops it. What it
# needed instead was `year_imputed`.
UNKNOWN = "99"

def _partial_date_ccyymmdd(s: str) -> str:
    """Return why this is not a representable partial date, or "" if it is one."""
    if not re.fullmatch(r"\d{8}", s):
        return "must be exactly eight digits, CCYYMMDD"
    yyyy, mm, dd = s[:4], s[4:6], s[6:8]
    if yyyy == "9999":
        return ("the year may not be unknown. A year that cannot be identified is APPROXIMATED "
                "to a real four-digit year and recorded as approximated; there is no notation "
                "for an absent year")
    if int(yyyy) == 0:
        return "year 0000 is not a year"
    if mm == UNKNOWN and dd != UNKNOWN:
        return (f"month is {UNKNOWN} (unknown) but day is {dd}. That notation is NOT DECLARED by "
                f"this contract: whether a known day under an unknown month is meaningful has "
                f"not been decided. Write {yyyy}{UNKNOWN}{UNKNOWN} if neither is known, or "
                f"supply the month")
    if mm != UNKNOWN and not (1 <= int(mm) <= 12):
        return f"month {mm} is not a month"
    if dd == UNKNOWN:
        return ""
    if int(dd) < 1:
        return f"day {dd} is not a day; there is no day zero"
    if mm == UNKNOWN:
        return ""          # unreachable: mm==99 with dd!=99 was refused above
    last = _calendar.monthrange(int(yyyy), int(mm))[1]
    if int(dd) > last:
        return f"{yyyy}-{mm} has {last} days, so day {dd} does not exist"
    return ""

def _earliest_denoted(s: str):
    """The earliest real date a partial-date notation could mean, or None if it means nothing.

    `20189999` is a claim about a year, not about 1 January, but 1 January is the earliest day
    it could denote — so it is the right thing to compare against an upper bound. Comparing the
    literal digits would read `99` as a month and refuse a run for a date it never claimed.
    """
    import datetime
    if not re.fullmatch(r"\d{8}", s):
        return None
    y, m, d = int(s[:4]), s[4:6], s[6:8]
    mm = 1 if m == UNKNOWN else int(m)
    dd = 1 if d == UNKNOWN else int(d)
    if not (1 <= mm <= 12):
        return None
    try:
        return datetime.date(y, mm, dd)
    except ValueError:
        return None

def check_dates_not_after(fields, value: dict, not_after) -> list[Violation]:
    """Dates no case could witness: later than the last document that could report them.

    THE ONE CONTENT CHECK IN THIS MODULE THAT REFUSES, and the test it passes is the one the
    five removed clinical checks failed. `field_format` was demoted to advisory because it
    refused `C34.9` — the punctuated form ICD-O-3 itself writes — and so destroyed correct
    values over notation; 60 of 254 recorded rejections refused a tuple that was exactly the
    registry's. No correct answer can fail THIS one. A document cannot report a diagnosis that
    has not happened, so a date after every document in the chart has no possible witness,
    whatever notation it is written in.

    Applies only to fields declaring the partial-date calendar: it is arithmetic on a declared
    value space, not a judgement about a tumour.
    """
    out: list[Violation] = []
    if not_after is None:
        return out
    for f in fields or []:
        if getattr(f, "calendar", None) != "partial_date_ccyymmdd":
            continue
        raw = value.get(getattr(f, "name", ""))
        if raw is None or str(raw).strip() == "":
            continue
        s = str(raw).strip()
        earliest = _earliest_denoted(s)
        if earliest is None or earliest <= not_after:
            continue
        out.append(Violation(
            rule_id=field_rule_id("date_not_after_record", f.name),
            rule_kind="date_not_after_record", field=f.name, coded_value=s,
            trigger=not_after.isoformat(),
            message=(f"{f.name}={s!r} is after {not_after.isoformat()}, the last document in "
                     f"this record. Nothing in the chart could have reported it, so no "
                     f"citation for it can exist. If the year is an estimate, give a real "
                     f"year and set year_imputed.")))
    return out

#: The calendar kinds `check_field_formats_detail` dispatches on. A field declaring one that
#: is absent here does not load -- same rule as `ANSWER_CHECK_KINDS`, and for the same reason:
#: a declared validator that never runs looks exactly like one that ran and found nothing.
CALENDAR_KINDS: dict[str, Callable[[str], str]] = {
    "partial_date_ccyymmdd": _partial_date_ccyymmdd,
}

class UnknownCalendarKindError(ValueError):
    """A field declares a calendar validator nothing implements."""

def check_field_formats(fields, value: dict) -> list[str]:
    """The messages only; `check_field_formats_detail` is the attributable form."""
    return [v.message for v in check_field_formats_detail(fields, value)]

def check_field_formats_detail(fields, value: dict) -> list[Violation]:
    """Enforce the `format` and `allowable_values` the spec already declares per field.

    The one surviving check, and the only one with a positive measured record: 7 firings over
    every recorded trace, 0 refusals of a registry value, 6 that preceded the registry answer.
    It rejected `C3412` and `C3432` -- four digits where the spec declares `C\\d{3}` -- which
    a run had previously shipped stamped as validated with zero rejections.

    It contains no clinical knowledge. It compares a submitted string against a pattern the
    spec author wrote, which is a contract, not a judgement about a tumour. That is the line
    this module now holds: shape and value domain here, everything clinical in the instruction
    and in the evaluation that measures whether the instruction was followed.

    Needs no spec configuration -- the constraints are already in `fields` -- so it is always
    on, for every criterion, at no authoring cost.
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
        shape_failed = False
        fmt = getattr(f, "format", None)
        if fmt:
            try:
                if not re.fullmatch(fmt, s):
                    shape_failed = True
                    out.append(Violation(
                        rule_id=field_rule_id("field_format", name), rule_kind="field_format",
                        field=name, coded_value=s, trigger=str(fmt),
                        message=f"{name}={s!r} does not match the required format {fmt!r}."))
            except re.error:
                pass          # a broken pattern in the spec must not block the run
        cal = getattr(f, "calendar", None)
        # STEP TWO PRESUPPOSES STEP ONE. `2018-11-07` is one defect — the wrong notation — and
        # reporting it as both a shape miss and a non-existent date would make a single mistake
        # look like two, which is how a rejection message stops being readable.
        if cal and not shape_failed:
            fn = CALENDAR_KINDS.get(str(cal))
            if fn is None:
                raise UnknownCalendarKindError(
                    f"{name}.calendar={cal!r} is not implemented. One of: "
                    f"{', '.join(sorted(CALENDAR_KINDS))}")
            why = fn(s)
            if why:
                out.append(Violation(
                    rule_id=field_rule_id("field_calendar", name), rule_kind="field_calendar",
                    field=name, coded_value=s, trigger=str(cal),
                    message=f"{name}={s!r} is not a date that exists: {why}."))
        allowed = getattr(f, "allowable_values", None)
        if allowed and s not in [str(v) for v in allowed]:
            out.append(Violation(
                rule_id=field_rule_id("field_allowable_values", name),
                rule_kind="field_allowable_values", field=name, coded_value=s,
                trigger="", message=(f"{name}={s!r} is not one of the allowed values "
                                     f"{list(allowed)!r}.")))
    return out
