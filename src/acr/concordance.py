"""L4: guideline concordance, decided by rule and never by a model.

Everything below L4 is spent proving that a negative answer was *earned* — forced sampling,
a stratified ledger, an elusion bound, a gate the agent cannot self-attest. All of it is
thrown away the moment a concordance rate is produced by asking a model whether the care
"looks guideline-concordant". Thresholds, dates, Boolean logic and drug classes are
decidable, so they are decided here, from declared data, deterministically.

**There are no model calls in this module and there is no path from it to one.**
`tests/test_concordance.py::test_no_model_is_reachable_from_this_module` walks the
first-party import closure and fails if `acr.llm`, `acr.graph` or any provider SDK appears.

Three design positions, each taken because of something measured on 2026-07-26:

**NOT_ASSESSABLE is the product, not the fallback.** A recommendation whose inputs are
EVIDENCE_INSUFFICIENT cannot be scored. Folding those cases into the denominator moves the
rate in whichever direction the missingness happens to lean, and the missingness is never
random — `special_codes_not_mar` in the site/histology spec says exactly this: absent
behaviour clusters on outside-hospital and declined biopsies. So an unknown input stops the
evaluation and names itself in `blocking_inputs`.

**A declared input that no rule reads is not an input.** `primary_site: "C3412"` passed the
gate because the spec declared `format: C\\d{3}` and nothing enforced it. The same shape of
bug is available here: a recommendation that lists `required_inputs` its conditions never
mention, or references a variable it never declared. `validate_guideline` refuses to load
either, so the YAML's own documentation is the thing being executed.

**An established absence and an unknown are different facts, and only the gate can tell
them apart.** `status=FOUND, value=null` means the coverage proof closed and the event is
not there; that is a NON_CONCORDANT finding. `status=EVIDENCE_INSUFFICIENT` means nobody
knows; that is NOT_ASSESSABLE. This is the A-vs-B split of the design doc, and it is the
single strongest argument for the coverage apparatus: "we proved it is not documented" is
only sayable because L3 exists. `InputUse.negative_basis` carries the proof forward so L5
can separate a care gap from a documentation gap.

Outcomes. The contract names three; two more exist because collapsing them into the three
would corrupt the very number the file is protecting:

    CONCORDANT            applies, action delivered
    NON_CONCORDANT        applies, action absent, no documented exception   <- the care gap
    NOT_ASSESSABLE        some required input is unknown                    <- not scorable
    NOT_APPLICABLE        determinately outside the population              <- stage I is not
                          a missing datum, and must not dilute the unknowns
    EXCEPTION_DOCUMENTED  applies, action absent, a declared legitimate
                          exception is documented to the same evidentiary
                          standard as the primary variable

`summarise` puts only CONCORDANT and NON_CONCORDANT in the denominator and reports the
other three beside the rate, so the rate can never be read without its exclusions.

NO CLINICAL KNOWLEDGE LIVES IN THIS FILE. Which histologies are NSCLC, which stage groups
trigger adjuvant therapy, which drug classes count and which exceptions are legitimate are
declared in `guidelines/*.yaml`, where an oncologist can review them.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import yaml

Ternary = Literal["TRUE", "FALSE", "UNKNOWN"]
TRUE: Ternary = "TRUE"
FALSE: Ternary = "FALSE"
UNKNOWN: Ternary = "UNKNOWN"

Resolution = Literal["KNOWN", "KNOWN_ABSENT", "UNKNOWN"]
Outcome = Literal[
    "CONCORDANT", "NON_CONCORDANT", "NOT_ASSESSABLE",
    "NOT_APPLICABLE", "EXCEPTION_DOCUMENTED",
]

#: The only two outcomes that may appear in a concordance denominator.
SCORABLE: tuple[Outcome, ...] = ("CONCORDANT", "NON_CONCORDANT")

#: Where a required input is allowed to come from. `not_yet_extractable` is the honest
#: value while the extractor is unbuilt; at runtime such a variable is simply UNKNOWN and
#: the recommendation is NOT_ASSESSABLE, which is the correct answer and not a crash.
INPUT_SOURCES = frozenset({"registry_limited_dataset", "extraction_spec", "not_yet_extractable"})


class GuidelineError(ValueError):
    """The guideline file is not executable as written. Never raised at scoring time."""


class ConcordanceInputError(ValueError):
    """A variable arrived without a status. Guessing one is how a rate gets inflated."""


# ------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class VariableValue:
    """One extracted variable, carrying the status L2/L3 assigned it.

    `status=FOUND` with `value=None` is an *established absence* — the field is nullable,
    the coverage gate closed, and the answer is "there is none". `status=FOUND` with a
    value is a fact. Anything else is an unknown, and unknowns do not get scored.

    `unknown_sentinel` is the third case, and it is set by the engine rather than the
    caller: a value that is present and well-formed but whose *meaning* is "unknown".
    Registries are full of them — stage group `99`, class of case `99`, histology `8000`.
    See `Guideline.unknown_value_codes`.
    """
    status: str = "NOT_EXTRACTED"
    value: Any = None
    negative_basis: str | None = None       # GATE_VALIDATED / AGENT_GAVE_UP / BUDGET_EXHAUSTED
    source: str = ""                        # spec_id, or the registry table it came from
    unknown_sentinel: bool = False

    @property
    def resolution(self) -> Resolution:
        if self.status != "FOUND" or self.unknown_sentinel:
            return "UNKNOWN"
        return "KNOWN_ABSENT" if self.value is None or str(self.value).strip() == "" else "KNOWN"


def _coerce(name: str, raw: Any) -> VariableValue:
    if isinstance(raw, VariableValue):
        return raw
    if not isinstance(raw, dict) or "status" not in raw:
        # A bare scalar has no provenance. Defaulting it to FOUND would let an ungated
        # guess -- the C349-with-zero-searches case -- enter a denominator as if proved.
        raise ConcordanceInputError(
            f"variable {name!r} must be a mapping carrying a status "
            f"(FOUND | EVIDENCE_INSUFFICIENT | SPEC_INSUFFICIENT), got {type(raw).__name__}. "
            f"A value without a status cannot be told apart from a guess."
        )
    return VariableValue(
        status=str(raw.get("status") or "NOT_EXTRACTED"),
        value=raw.get("value"),
        negative_basis=raw.get("negative_basis"),
        source=str(raw.get("source") or ""),
    )


def variables_from_answer(answer: dict, field_names: Sequence[str], *,
                          source: str = "") -> dict[str, VariableValue]:
    """Flatten one extraction answer into per-variable values.

    The status on a run is per *answer*; the value dict is per *field*, and the two do not
    have to agree. This is real and it shipped: `aprime_SYN0002` answered
    EVIDENCE_INSUFFICIENT overall while still coding `primary_site: C186`, because the
    site/histology spec says outright "you may still report primary_site if the site of
    origin is documented". Dropping C186 would discard an established fact; promoting the
    two fields it stayed silent about would invent two.

    So the rule is per field, and it turns on the difference between an assertion and a
    silence — the same distinction the recurrence spec draws with "silence is not remission":

      field present with a value        -> FOUND    (established, whatever the answer said)
      field present and explicitly null -> the answer's status; FOUND here means "there is
                                           none", which is the established absence
      field absent from the value dict  -> the answer's status, but never FOUND — a silence
                                           is not an assertion that there is none

    That last line is why SPEC_INSUFFICIENT survives the flattening. Class of Case comes back
    SPEC_INSUFFICIENT / WRONG_DATA_SOURCE from every chart by design, and downgrading it to
    EVIDENCE_INSUFFICIENT would tell a reader to go looking in the notes again for something
    that is not in the notes.
    """
    value = answer.get("value") or {}
    status = str(answer.get("status") or "EVIDENCE_INSUFFICIENT")
    silent = status if status != "FOUND" else "EVIDENCE_INSUFFICIENT"
    basis = answer.get("negative_basis")
    out: dict[str, VariableValue] = {}
    for name in field_names:
        if name not in value:
            out[name] = VariableValue(silent, None, basis, source)
        elif value[name] is not None and str(value[name]).strip() != "":
            out[name] = VariableValue("FOUND", value[name], basis, source)
        else:
            out[name] = VariableValue(status, None, basis, source)
    return out


# --------------------------------------------------------------------------- guideline
@dataclass(frozen=True)
class ExceptionRule:
    id: str
    label: str = ""
    evidence_standard: str = ""
    when: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    id: str
    title: str = ""
    statement: str = ""
    source: dict = field(default_factory=dict)
    required_inputs: tuple[dict, ...] = ()
    applies_when: tuple[dict, ...] = ()
    satisfied_when: tuple[dict, ...] = ()
    exceptions: tuple[ExceptionRule, ...] = ()

    @property
    def declared_inputs(self) -> list[str]:
        return [str(d.get("name")) for d in self.required_inputs if d.get("name")]

    @property
    def referenced_inputs(self) -> list[str]:
        seen: dict[str, None] = {}
        for cond in list(self.applies_when) + list(self.satisfied_when) + [
                c for e in self.exceptions for c in e.when]:
            for n in _referenced(cond):
                seen[n] = None
        return list(seen)


@dataclass(frozen=True)
class Guideline:
    guideline_id: str
    guideline_version: str = "0.0.0"
    source_authority: dict = field(default_factory=dict)
    value_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: variable -> the codes whose meaning is "unknown". A registry sentinel is a present,
    #: well-formed value that asserts nothing: stage group `99`, class of case `99`,
    #: histology `8000`. Left alone it fails set membership like any other non-member, so a
    #: patient whose stage was never established comes out NOT_APPLICABLE — determinately
    #: outside the population — instead of NOT_ASSESSABLE. That is the inflation this whole
    #: layer exists to refuse, arriving disguised as a value. It is the same bug
    #: `contracts/extensions/note_type_filters.v2.schema.json` was written for: v1 allowed the
    #: literal `unknown` as a list member, which made "the derivation did not run"
    #: indistinguishable from "there is nothing here" and hid a wiring failure for two months.
    unknown_value_codes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    recommendations: tuple[Recommendation, ...] = ()
    raw: dict = field(default_factory=dict)

    @property
    def guideline_hash(self) -> str:
        """A concordance label is only comparable to another under the same hash."""
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def recommendation(self, rec_id: str) -> Recommendation:
        for r in self.recommendations:
            if r.id == rec_id:
                return r
        raise KeyError(rec_id)


def _as_tuple(x: Any) -> tuple:
    if x is None:
        return ()
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def parse_guideline(data: dict) -> Guideline:
    recs = []
    for r in data.get("recommendations") or []:
        recs.append(Recommendation(
            id=str(r.get("id") or ""),
            title=str(r.get("title") or ""),
            statement=str(r.get("statement") or ""),
            source=dict(r.get("source") or {}),
            required_inputs=_as_tuple(r.get("required_inputs")),
            applies_when=_as_tuple(r.get("applies_when")),
            satisfied_when=_as_tuple(r.get("satisfied_when")),
            exceptions=tuple(
                ExceptionRule(
                    id=str(e.get("id") or ""),
                    label=str(e.get("label") or ""),
                    evidence_standard=str(e.get("evidence_standard") or ""),
                    when=_as_tuple(e.get("when")),
                )
                for e in (r.get("exceptions") or [])
            ),
        ))
    return Guideline(
        guideline_id=str(data.get("guideline_id") or ""),
        guideline_version=str(data.get("guideline_version") or "0.0.0"),
        source_authority=dict(data.get("source_authority") or {}),
        value_sets={k: tuple(str(x) for x in v)
                    for k, v in (data.get("value_sets") or {}).items()},
        unknown_value_codes={k: tuple(str(x) for x in v)
                             for k, v in (data.get("unknown_value_codes") or {}).items()},
        recommendations=tuple(recs),
        raw=data,
    )


def load_guideline(path: str | Path, *, validate: bool = True) -> Guideline:
    g = parse_guideline(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if validate:
        bad = validate_guideline(g)
        if bad:
            raise GuidelineError(f"{path}: " + "; ".join(bad))
    return g


# -------------------------------------------------------------------------- validation
def validate_guideline(g: Guideline) -> list[str]:
    """Return a list of violations; empty means the file is executable. Never raises.

    Same contract as `check_field_formats`, and here for the same reason: the C3412 failure
    was a constraint that was *written down* and never run. Note one deliberate divergence —
    `check_field_formats` swallows `re.error` so a typo cannot block a live patient run,
    whereas a bad pattern here is rejected outright. A guideline is authored once, offline,
    under review; a silently disabled rule would quietly move every rate computed from it.
    """
    out: list[str] = []
    if not g.guideline_id:
        out.append("guideline_id is required")
    if not g.recommendations:
        out.append("no recommendations declared")
    everything_read: set[str] = set()
    seen: set[str] = set()
    for r in g.recommendations:
        everything_read.update(r.referenced_inputs)
        p = f"recommendation {r.id or '<unnamed>'}"
        if not r.id:
            out.append("a recommendation has no id")
        elif r.id in seen:
            out.append(f"{p}: duplicate id")
        seen.add(r.id)
        if not r.applies_when:
            out.append(f"{p}: applies_when is empty, so it would apply to every patient")
        if not r.satisfied_when:
            out.append(f"{p}: satisfied_when is empty, so it could never be non-concordant")

        conds = list(r.applies_when) + list(r.satisfied_when)
        ex_ids: set[str] = set()
        for e in r.exceptions:
            if not e.id:
                out.append(f"{p}: an exception has no id")
            elif e.id in ex_ids:
                out.append(f"{p}: duplicate exception id {e.id!r}")
            ex_ids.add(e.id)
            if not e.when:
                out.append(f"{p}: exception {e.id!r} has no condition, so it fires for everyone")
            conds.extend(e.when)
        out += [f"{p}: {m}" for c in conds for m in _validate_condition(c, g.value_sets)]

        declared, referenced = set(r.declared_inputs), set(r.referenced_inputs)
        for n in sorted(referenced - declared):
            out.append(f"{p}: condition reads {n!r}, which required_inputs does not declare")
        for n in sorted(declared - referenced):
            out.append(f"{p}: required_inputs declares {n!r}, which no condition reads")
        for d in r.required_inputs:
            src = d.get("source")
            if src not in INPUT_SOURCES:
                out.append(f"{p}: input {d.get('name')!r} has source {src!r}, "
                           f"not one of {sorted(INPUT_SOURCES)}")
        # A number in a rule is a clinical decision wearing engineering clothes. The
        # recurrence spec refuses to invent a surveillance interval and writes
        # PLACEHOLDER_REQUIRES_CLINICAL_INPUT instead; the same standard applies to a
        # 120-day adjuvant window, which is an operationalisation and not a guideline text.
        if any(_has_threshold(c) for c in conds) and not r.source.get("operationalisation"):
            out.append(f"{p}: uses a numeric threshold but declares no "
                       f"source.operationalisation saying who set it and whether it is signed off")

    # A sentinel list attached to a variable nothing reads is the declared-but-unenforced
    # bug again, and it fails silently in the worst direction: the `99` it was meant to
    # catch goes on being scored as a real value.
    for name, codes in g.unknown_value_codes.items():
        if name not in everything_read:
            out.append(f"unknown_value_codes declares {name!r}, which no condition reads")
        if not codes:
            out.append(f"unknown_value_codes[{name!r}] is empty")
    return out


def _validate_condition(cond: Any, value_sets: dict) -> list[str]:
    if not isinstance(cond, dict):
        return [f"condition must be a mapping, got {type(cond).__name__}"]
    op = cond.get("op")
    if op not in _OPS:
        return [f"unknown op {op!r}; known ops are {sorted(_OPS)}"]
    out: list[str] = []
    if op in ("all_of", "any_of"):
        subs = cond.get("conditions") or []
        if not subs:
            out.append(f"{op} has no conditions")
        for s in subs:
            out += _validate_condition(s, value_sets)
        return out
    if op == "not":
        return _validate_condition(cond.get("condition") or {}, value_sets)
    for key in _VAR_KEYS.get(op, ("var",)):
        if not cond.get(key):
            out.append(f"{op} is missing {key!r}")
    if op in ("in_set", "not_in_set"):
        if cond.get("set") and cond["set"] not in value_sets:
            out.append(f"{op} references undeclared value_set {cond['set']!r}")
        if not cond.get("set") and not cond.get("values"):
            out.append(f"{op} needs either `set` or `values`")
    if op == "matches":
        try:
            re.compile(str(cond.get("pattern", "")))
        except re.error as exc:
            out.append(f"matches has an invalid pattern {cond.get('pattern')!r}: {exc}")
    if op in ("equals", "not_equals") and "value" not in cond:
        out.append(f"{op} is missing 'value'")
    if op in ("at_least", "at_most") and _as_number(cond.get("value")) is None:
        out.append(f"{op} needs a numeric 'value', got {cond.get('value')!r}")
    if op == "days_between" and cond.get("min_days") is None and cond.get("max_days") is None:
        out.append("days_between needs at least one of min_days / max_days")
    return out


def _referenced(cond: Any) -> list[str]:
    if not isinstance(cond, dict):
        return []
    op = cond.get("op")
    if op in ("all_of", "any_of"):
        return [n for s in (cond.get("conditions") or []) for n in _referenced(s)]
    if op == "not":
        return _referenced(cond.get("condition") or {})
    return [str(cond[k]) for k in _VAR_KEYS.get(op, ("var",)) if cond.get(k)]


def _has_threshold(cond: Any) -> bool:
    if not isinstance(cond, dict):
        return False
    op = cond.get("op")
    if op in ("all_of", "any_of"):
        return any(_has_threshold(s) for s in (cond.get("conditions") or []))
    if op == "not":
        return _has_threshold(cond.get("condition") or {})
    return op in ("at_least", "at_most") or cond.get("min_days") is not None \
        or cond.get("max_days") is not None


# ---------------------------------------------------------------------- three-valued core
@dataclass(frozen=True)
class _V:
    """A condition's verdict plus the variables it touched. Pure; combinators only merge."""
    truth: Ternary
    used: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _merge(vs: Sequence[_V], truth: Ternary) -> _V:
    return _V(truth,
              tuple(dict.fromkeys(n for v in vs for n in v.used)),
              tuple(dict.fromkeys(n for v in vs for n in v.unknown)),
              tuple(dict.fromkeys(n for v in vs for n in v.notes)))


def _and(vs: Sequence[_V]) -> _V:
    truth = FALSE if any(v.truth == FALSE for v in vs) else \
        UNKNOWN if any(v.truth == UNKNOWN for v in vs) else TRUE
    return _merge(vs, truth)


def _or(vs: Sequence[_V]) -> _V:
    truth = TRUE if any(v.truth == TRUE for v in vs) else \
        UNKNOWN if any(v.truth == UNKNOWN for v in vs) else FALSE
    return _merge(vs, truth)


def _negate(v: _V) -> _V:
    return _V({TRUE: FALSE, FALSE: TRUE, UNKNOWN: UNKNOWN}[v.truth], v.used, v.unknown, v.notes)


@dataclass(frozen=True)
class _Env:
    variables: dict[str, VariableValue]
    value_sets: dict[str, tuple[str, ...]]


def _norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x if x is not None else "")).strip().lower()


def _as_number(x: Any) -> float | None:
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


_TRUEISH = {"true", "yes", "y", "1"}
_FALSEISH = {"false", "no", "n", "0"}


def _as_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    s = _norm(x)
    return True if s in _TRUEISH else False if s in _FALSEISH else None


_CCYYMMDD = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_ISOISH = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _date_range(raw: Any) -> tuple[dt.date, dt.date] | None:
    """A registry date as the interval of days it could denote, or None if unparseable.

    Registry dates are only as precise as the chart was. STORE writes an unknown month or
    day as `99` — `20100499` is the shipped boundary case for "diagnosed in the spring of
    2010" — so a date is an interval, and a threshold that flips inside that interval has
    not been decided. `days_between` uses both ends and answers UNKNOWN when they disagree,
    which is the difference between reporting a rate and inventing one.
    """
    if isinstance(raw, dt.datetime):
        return (raw.date(), raw.date())
    if isinstance(raw, dt.date):
        return (raw, raw)
    m = _CCYYMMDD.match(str(raw or "").strip()) or _ISOISH.match(str(raw or "").strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 1:
        return None
    if mo in (0, 99):
        return (dt.date(y, 1, 1), dt.date(y, 12, 31))
    if not 1 <= mo <= 12:
        return None
    last = calendar.monthrange(y, mo)[1]
    if d in (0, 99):
        return (dt.date(y, mo, 1), dt.date(y, mo, last))
    if not 1 <= d <= last:
        return None
    return (dt.date(y, mo, d), dt.date(y, mo, d))


def _get(env: _Env, name: str) -> VariableValue:
    return env.variables.get(name) or VariableValue("NOT_EXTRACTED")


def _unary(cond: dict, env: _Env) -> tuple[str, VariableValue]:
    name = str(cond.get("var") or "")
    return name, _get(env, name)


# Which keys of a condition name a variable. Used by validation, by the reference scan and
# by evaluation, so a new op cannot be half-registered.
_VAR_KEYS: dict[str, tuple[str, ...]] = {
    "days_between": ("from", "to"),
    "on_or_before": ("from", "to"),
}


def _op_all_of(cond: dict, env: _Env) -> _V:
    return _and([_eval_condition(c, env) for c in cond.get("conditions") or []])


def _op_any_of(cond: dict, env: _Env) -> _V:
    return _or([_eval_condition(c, env) for c in cond.get("conditions") or []])


def _op_not(cond: dict, env: _Env) -> _V:
    return _negate(_eval_condition(cond.get("condition") or {}, env))


def _settled(name: str, vv: VariableValue, *, absent: Ternary) -> _V | None:
    """Short-circuit the two resolutions that never reach the operator's own logic.

    `absent` is the answer for an ESTABLISHED absence, and the two directions are not
    symmetric. An absence answers "is it X?" with no — it does not answer "what is it?".
    So membership tests take FALSE, while a test whose truth would place the patient
    *inside* a population (`not_in_set`, `not_equals`) takes UNKNOWN: a variable that was
    proved to have no value cannot be evidence that its value is something else.
    """
    if vv.resolution == "UNKNOWN":
        return _V(UNKNOWN, (name,), (name,))
    if vv.resolution == "KNOWN_ABSENT":
        return _V(absent, (name,), (name,) if absent == UNKNOWN else ())
    return None


def _op_equals(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    return _settled(name, vv, absent=FALSE) or \
        _V(TRUE if _norm(vv.value) == _norm(cond.get("value")) else FALSE, (name,))


def _op_not_equals(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    return _settled(name, vv, absent=UNKNOWN) or \
        _V(FALSE if _norm(vv.value) == _norm(cond.get("value")) else TRUE, (name,))


def _members(cond: dict, env: _Env) -> set[str]:
    vals = env.value_sets.get(str(cond["set"])) if cond.get("set") else cond.get("values") or []
    return {_norm(v) for v in vals}


def _op_in_set(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    return _settled(name, vv, absent=FALSE) or \
        _V(TRUE if _norm(vv.value) in _members(cond, env) else FALSE, (name,))


def _op_not_in_set(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    return _settled(name, vv, absent=UNKNOWN) or \
        _V(FALSE if _norm(vv.value) in _members(cond, env) else TRUE, (name,))


def _op_matches(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    settled = _settled(name, vv, absent=FALSE)
    if settled:
        return settled
    hit = re.fullmatch(str(cond.get("pattern", "")), str(vv.value).strip())
    return _V(TRUE if hit else FALSE, (name,))


def _op_is_true(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    settled = _settled(name, vv, absent=FALSE)
    if settled:
        return settled
    b = _as_bool(vv.value)
    if b is None:
        return _V(UNKNOWN, (name,), (name,), (f"{name}={vv.value!r} is not a boolean",))
    return _V(TRUE if b else FALSE, (name,))


def _op_is_false(cond: dict, env: _Env) -> _V:
    return _negate(_op_is_true(cond, env))


def _op_is_present(cond: dict, env: _Env) -> _V:
    name, vv = _unary(cond, env)
    if vv.resolution == "UNKNOWN":
        return _V(UNKNOWN, (name,), (name,))
    return _V(TRUE if vv.resolution == "KNOWN" else FALSE, (name,))


def _op_is_absent(cond: dict, env: _Env) -> _V:
    return _negate(_op_is_present(cond, env))


def _compare(cond: dict, env: _Env, keep: str) -> _V:
    name, vv = _unary(cond, env)
    settled = _settled(name, vv, absent=FALSE)
    if settled:
        return settled
    got, want = _as_number(vv.value), _as_number(cond.get("value"))
    if got is None:
        return _V(UNKNOWN, (name,), (name,), (f"{name}={vv.value!r} is not numeric",))
    ok = got >= want if keep == "at_least" else got <= want
    return _V(TRUE if ok else FALSE, (name,))


def _op_at_least(cond: dict, env: _Env) -> _V:
    return _compare(cond, env, "at_least")


def _op_at_most(cond: dict, env: _Env) -> _V:
    return _compare(cond, env, "at_most")


def _two_dates(cond: dict, env: _Env) -> tuple[_V | None, tuple[str, str], Any, Any]:
    a_name, b_name = str(cond.get("from") or ""), str(cond.get("to") or "")
    a, b = _get(env, a_name), _get(env, b_name)
    used = (a_name, b_name)
    unk = tuple(n for n, v in ((a_name, a), (b_name, b)) if v.resolution == "UNKNOWN")
    if unk:
        return _V(UNKNOWN, used, unk), used, None, None
    if "KNOWN_ABSENT" in (a.resolution, b.resolution):
        # The event provably did not happen, so it provably did not happen in the window.
        # This is the case the coverage gate exists to produce, and it is a finding.
        return _V(FALSE, used, (), ("an endpoint is an established absence",)), used, None, None
    ar, br = _date_range(a.value), _date_range(b.value)
    bad = [n for n, r in ((a_name, ar), (b_name, br)) if r is None]
    if bad:
        return (_V(UNKNOWN, used, tuple(bad), tuple(f"{n} is not a parseable date" for n in bad)),
                used, None, None)
    return None, used, ar, br


def _op_days_between(cond: dict, env: _Env) -> _V:
    short, used, ar, br = _two_dates(cond, env)
    if short is not None:
        return short
    lo, hi = (br[0] - ar[1]).days, (br[1] - ar[0]).days
    floor = cond.get("min_days")
    ceil = cond.get("max_days")
    floor = float("-inf") if floor is None else float(floor)
    ceil = float("inf") if ceil is None else float(ceil)
    if lo >= floor and hi <= ceil:
        return _V(TRUE, used)
    if hi < floor or lo > ceil:
        return _V(FALSE, used)
    note = (f"an imprecise date leaves the interval between {lo} and {hi} days, which "
            f"straddles the {floor}..{ceil} threshold")
    return _V(UNKNOWN, used, used, (note,))


def _op_on_or_before(cond: dict, env: _Env) -> _V:
    short, used, ar, br = _two_dates(cond, env)
    if short is not None:
        return short
    if ar[1] <= br[0]:
        return _V(TRUE, used)
    if ar[0] > br[1]:
        return _V(FALSE, used)
    return _V(UNKNOWN, used, used,
              ("an imprecise date leaves the ordering of the two dates undecided",))


_OPS = {
    "all_of": _op_all_of, "any_of": _op_any_of, "not": _op_not,
    "equals": _op_equals, "not_equals": _op_not_equals,
    "in_set": _op_in_set, "not_in_set": _op_not_in_set,
    "matches": _op_matches,
    "is_true": _op_is_true, "is_false": _op_is_false,
    "is_present": _op_is_present, "is_absent": _op_is_absent,
    "at_least": _op_at_least, "at_most": _op_at_most,
    "days_between": _op_days_between, "on_or_before": _op_on_or_before,
}


def _eval_condition(cond: Any, env: _Env) -> _V:
    if not isinstance(cond, dict):
        raise GuidelineError(f"condition must be a mapping, got {type(cond).__name__}: {cond!r}")
    op = cond.get("op")
    if op not in _OPS:
        raise GuidelineError(f"unknown condition op {op!r}; known ops are {sorted(_OPS)}")
    return _OPS[op](cond, env)


def _eval_all(conds: Sequence[dict], env: _Env) -> _V:
    return _and([_eval_condition(c, env) for c in conds]) if conds else _V(TRUE)


# ------------------------------------------------------------------------------- results
@dataclass(frozen=True)
class InputUse:
    variable: str
    status: str
    value: Any
    resolution: Resolution
    negative_basis: str | None = None
    source: str = ""
    #: True when `status` and `resolution` disagree because the value is a registry
    #: sentinel: L2 really did return `99`, and `99` really does mean nobody knows.
    unknown_sentinel: bool = False

    def to_dict(self) -> dict:
        return {"variable": self.variable, "status": self.status, "value": self.value,
                "resolution": self.resolution, "negative_basis": self.negative_basis,
                "source": self.source, "unknown_sentinel": self.unknown_sentinel}


@dataclass(frozen=True)
class ConcordanceResult:
    recommendation_id: str
    outcome: Outcome
    rule_applied: str
    reason: str
    inputs_used: tuple[InputUse, ...] = ()
    blocking_inputs: tuple[str, ...] = ()
    exception_id: str | None = None
    notes: tuple[str, ...] = ()
    guideline_id: str = ""
    guideline_version: str = ""
    guideline_hash: str = ""

    @property
    def scorable(self) -> bool:
        return self.outcome in SCORABLE

    def to_dict(self) -> dict:
        return {
            "recommendation_id": self.recommendation_id, "outcome": self.outcome,
            "rule_applied": self.rule_applied, "reason": self.reason,
            "inputs_used": [i.to_dict() for i in self.inputs_used],
            "blocking_inputs": list(self.blocking_inputs),
            "exception_id": self.exception_id, "notes": list(self.notes),
            "guideline_id": self.guideline_id, "guideline_version": self.guideline_version,
            "guideline_hash": self.guideline_hash,
            "engine": "acr.concordance/deterministic",
        }


def _inputs_used(env: _Env, *verdicts: _V) -> tuple[InputUse, ...]:
    names = dict.fromkeys(n for v in verdicts for n in v.used)
    out = []
    for n in names:
        vv = _get(env, n)
        out.append(InputUse(n, vv.status, vv.value, vv.resolution, vv.negative_basis,
                            vv.source, vv.unknown_sentinel))
    return tuple(out)


def _bind(variables: dict[str, Any], guideline: Guideline) -> dict[str, VariableValue]:
    """Coerce the caller's dict and demote registry sentinels to unknowns."""
    out: dict[str, VariableValue] = {}
    for k, raw in (variables or {}).items():
        vv = _coerce(k, raw)
        codes = {_norm(c) for c in guideline.unknown_value_codes.get(k, ())}
        if vv.resolution == "KNOWN" and _norm(vv.value) in codes:
            vv = replace(vv, unknown_sentinel=True)
        out[k] = vv
    return out


def assess_one(rec: Recommendation, variables: dict[str, Any], guideline: Guideline
               ) -> ConcordanceResult:
    """Score one recommendation. Pure: same inputs, same output, no I/O, no model."""
    env = _Env(_bind(variables, guideline), guideline.value_sets)
    ident = {"guideline_id": guideline.guideline_id,
             "guideline_version": guideline.guideline_version,
             "guideline_hash": guideline.guideline_hash}

    applies = _eval_all(rec.applies_when, env)
    if applies.truth == FALSE:
        return ConcordanceResult(
            rec.id, "NOT_APPLICABLE", "applies_when",
            "the patient is determinately outside this recommendation's population",
            _inputs_used(env, applies), notes=applies.notes, **ident)
    if applies.truth == UNKNOWN:
        return ConcordanceResult(
            rec.id, "NOT_ASSESSABLE", "applies_when",
            "cannot tell whether this recommendation applies: "
            + ", ".join(applies.unknown) + " is not established",
            _inputs_used(env, applies), applies.unknown, notes=applies.notes, **ident)

    satisfied = _eval_all(rec.satisfied_when, env)
    if satisfied.truth == TRUE:
        return ConcordanceResult(
            rec.id, "CONCORDANT", "satisfied_when",
            "the recommendation applies and the recommended action is documented",
            _inputs_used(env, applies, satisfied), notes=satisfied.notes, **ident)

    # Exceptions are evaluated before an unknown action is reported, because a documented
    # contraindication settles the case whether or not the action can be established. They
    # are NOT evaluated before a satisfied action: care that was delivered is concordant
    # even if a reason not to deliver it also exists.
    fired = [(e, _eval_all(e.when, env)) for e in rec.exceptions]
    for e, v in fired:
        if v.truth == TRUE:
            return ConcordanceResult(
                rec.id, "EXCEPTION_DOCUMENTED", f"exception:{e.id}",
                f"the recommended action is absent, but a declared legitimate exception is "
                f"documented: {e.label or e.id}. This is not a care gap and is excluded from "
                f"the denominator.",
                _inputs_used(env, applies, satisfied, v), exception_id=e.id,
                notes=v.notes, **ident)

    if satisfied.truth == UNKNOWN:
        return ConcordanceResult(
            rec.id, "NOT_ASSESSABLE", "satisfied_when",
            "the recommendation applies, but whether the action was delivered is not "
            "established: " + ", ".join(satisfied.unknown),
            _inputs_used(env, applies, satisfied), satisfied.unknown,
            notes=satisfied.notes, **ident)

    unresolved = [(e, v) for e, v in fired if v.truth == UNKNOWN]
    if unresolved:
        # The action is absent and we cannot rule out a legitimate reason. Calling this a
        # care gap inflates the gap rate exactly the way scoring an unknown input inflates
        # the concordance rate; the design doc's failure D (counting a patient who declined
        # chemotherapy as a care gap) is this branch getting it wrong.
        blocking = tuple(dict.fromkeys(n for _, v in unresolved for n in v.unknown))
        return ConcordanceResult(
            rec.id, "NOT_ASSESSABLE", "exception_status_unknown",
            "the recommended action is absent, but whether a legitimate exception applies is "
            "not established: " + ", ".join(f"{e.id}({','.join(v.unknown)})"
                                            for e, v in unresolved),
            _inputs_used(env, applies, satisfied, *[v for _, v in unresolved]), blocking,
            notes=tuple(n for _, v in unresolved for n in v.notes), **ident)

    return ConcordanceResult(
        rec.id, "NON_CONCORDANT", "satisfied_when",
        "the recommendation applies, the recommended action is not documented, and every "
        "declared exception was ruled out",
        _inputs_used(env, applies, satisfied, *[v for _, v in fired]),
        notes=satisfied.notes, **ident)


def assess(variables: dict[str, Any], guideline: Guideline, *,
           recommendation_ids: Iterable[str] | None = None) -> list[ConcordanceResult]:
    wanted = set(recommendation_ids) if recommendation_ids is not None else None
    return [assess_one(r, variables, guideline) for r in guideline.recommendations
            if wanted is None or r.id in wanted]


def summarise(results: Iterable[ConcordanceResult]) -> dict:
    """Counts and a rate whose denominator is stated, never implied.

    The rate is `None` — not 0.0, not 1.0 — when nothing was scorable, and the three
    excluded outcomes are reported beside it. A caller that wants to quote the rate has to
    carry the exclusions along, which is the whole point of NOT_ASSESSABLE being an outcome
    rather than a dropped row.
    """
    rs = list(results)
    n = {k: sum(1 for r in rs if r.outcome == k) for k in
         ("CONCORDANT", "NON_CONCORDANT", "NOT_ASSESSABLE", "NOT_APPLICABLE",
          "EXCEPTION_DOCUMENTED")}
    denom = n["CONCORDANT"] + n["NON_CONCORDANT"]
    blocking: dict[str, int] = {}
    for r in rs:
        for b in r.blocking_inputs:
            blocking[b] = blocking.get(b, 0) + 1
    return {
        "n_recommendations": len(rs),
        "denominator": denom,
        "concordance_rate": (n["CONCORDANT"] / denom) if denom else None,
        "denominator_excludes": {k: n[k] for k in
                                 ("NOT_ASSESSABLE", "NOT_APPLICABLE", "EXCEPTION_DOCUMENTED")},
        "counts": n,
        "assessable_fraction": (denom / (denom + n["NOT_ASSESSABLE"]))
        if (denom + n["NOT_ASSESSABLE"]) else None,
        "blocking_inputs": dict(sorted(blocking.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
