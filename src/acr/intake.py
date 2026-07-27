"""L0 INTAKE — an arbitrary question becomes a routing decision, or an explicit gap.

`registry_catalog` answers "which spec produces `primary_site`?", by exact string match. That
is the whole of this system's front door today, and it only opens for questions that are
already registry variable names. The questions this platform exists for are not:

    "does this patient have stage II-IIIA resected NSCLC"
    "was EGFR testing done before first-line therapy"
    "did this patient get guideline-concordant adjuvant chemo"

None of the three is a variable. The first is a predicate over variables that already have
specs. The second and third are guideline rules with declared inputs, most of which nothing
in this repo can extract. Treating any of them as "a new variable" is how a platform acquires
an unbounded spec backlog: every question becomes a new spec because nobody checked whether
it was already computable (design doc §8, "Variables cannot be enumerated").

THE RULE THIS MODULE IS BUILT AROUND
------------------------------------
The CLASSIFICATION may be a model call — what a sentence means is judgement, and no amount of
Python decides it. Every CONSEQUENCE is code:

  * a route to a spec is `VariableCatalog.resolve`: exact match, and a miss raises with the
    whole vocabulary attached. A near-miss is never quietly accepted.
  * a composition emits conditions in `concordance`'s own grammar, and every term is checked
    against the specs' declared fields and — through the runtime's own
    `answer_checks.check_field_formats` — their declared `format` and `allowable_values`. A
    term naming a field or a value no spec produces refuses the whole predicate.
  * whether a variable is answerable from notes is read off `spec.data_source`, never off the
    model's opinion. A classifier that calls `class_of_case` an ordinary extraction is
    overruled by `specs/STORE.610.class_of_case.yaml`, which exists precisely to refuse.
  * everything left over becomes a `Gap` carrying a named remedy. Never a default, never an
    invented variable.

The asymmetry is the whole design. A wrong classification produces a report that looks wrong,
which a human reads and corrects in one line. A wrong consequence produces a cohort silently
bound to a spec nobody asked for — the failure `registry_catalog`'s `AmbiguousVariableError`
already exists to prevent, one layer up, for the same reason.

WHY THE GAP LIST IS THE PRODUCT
-------------------------------
Routing `guidelines/nccn_nsclc_subset.yaml` resolves 6 of its 22 distinct inputs. The other
16 are declared `not_yet_extractable` or `registry_limited_dataset` in the guideline itself.
An intake layer that reported "3 recommendations routed" and stopped would be technically
true and operationally a lie: nothing can be scored. The honest unit of output here is the
unresolved list, printed at full length, which is why `Gap` carries a remedy rather than a
severity — a severity gets sorted and ignored; a remedy names who has to do what.

NO CHART IS EVER READ HERE. This module imports no corpus and takes no patient argument, and
`tests/test_intake.py` asserts that structurally rather than trusting the CLI's `--dry-run`
default. A planning layer that *can* reach PHI is a planning layer that eventually will be
asked to, and the flag protecting it is one typo from being off.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

import yaml

from .answer_checks import check_field_formats
# `_Env` and `_eval_all` are concordance's, and they are imported rather than reimplemented
# on purpose. A cohort predicate is three-valued for exactly the reason a guideline rule is:
# `histology = 8000` is a present, well-formed value that means "nobody established this",
# and a second evaluator would be a second place for that logic to drift back into
# two-valued. When it drifts, a patient whose histology was never established silently
# evaluates FALSE — determinately outside the cohort — and the denominator shrinks with no
# artifact recording it.
from .concordance import (Guideline, VariableValue, _Env, _eval_all, _referenced,
                          _validate_condition, load_guideline)
from .registry_catalog import (AmbiguousVariableError, UnknownVariableError, VariableCatalog,
                               VariableResolutionError, normalise_name)

# ---------------------------------------------------------------------------- outcomes
#: Five, and no sixth. "I could not tell" is not a routing outcome — it is the absence of
#: one, and it is reported as an `unclassified` gap so that nothing downstream can mistake it
#: for a decision that was made.
EXISTING_VARIABLE = "EXISTING_VARIABLE"
COMPOSITION = "COMPOSITION"
NEW_VARIABLE = "NEW_VARIABLE"
GUIDELINE_RULE = "GUIDELINE_RULE"
WRONG_DATA_SOURCE = "WRONG_DATA_SOURCE"
OUTCOMES = (EXISTING_VARIABLE, COMPOSITION, NEW_VARIABLE, GUIDELINE_RULE, WRONG_DATA_SOURCE)

#: One line each, and they are shipped to the classifier verbatim. A prompt that paraphrases
#: the outcome names in a docstring somewhere else is a prompt that drifts from the code that
#: enforces them.
OUTCOME_MEANINGS = {
    EXISTING_VARIABLE: "one shipped spec already answers this; name the variable(s)",
    COMPOSITION: "a deterministic predicate over variables that already have specs; emit the "
                 "terms, do not propose a spec",
    NEW_VARIABLE: "a chart review in its own right, with its own evidence rules and its own "
                  "proof obligation; no composition of existing variables produces it",
    GUIDELINE_RULE: "a recommendation in a shipped guideline; name the guideline and the "
                    "recommendation id",
    WRONG_DATA_SOURCE: "not in the clinical notes at any depth — it lives in registration, "
                       "billing, tumour-registry or follow-up systems",
}

# ------------------------------------------------------------------------------- gaps
UNKNOWN_VARIABLE = "unknown_variable"
AMBIGUOUS_VARIABLE = "ambiguous_variable"
UNKNOWN_FIELD_IN_PREDICATE = "unknown_field_in_predicate"
VALUE_OUTSIDE_DECLARED_DOMAIN = "value_outside_declared_domain"
UNDECLARED_VALUE_SET = "undeclared_value_set"
MALFORMED_TERM = "malformed_term"
UNSATISFIABLE_TERM = "unsatisfiable_term"
NOT_YET_EXTRACTABLE = "not_yet_extractable"
OUTSIDE_NOTES = "outside_notes"
NO_SPEC_DECLARES_THE_REFUSAL = "no_spec_declares_the_refusal"
NO_SENTINEL_POLICY = "no_sentinel_policy"
GUIDELINE_SOURCE_DISAGREES_WITH_CATALOG = "guideline_source_disagrees_with_catalog"
SPEC_AUTHORING_REQUIRED = "spec_authoring_required"
ROUTE_TARGET_MISSING = "route_target_missing"
UNCLASSIFIED = "unclassified"

#: Gaps that mean no consequence could be built at all, as opposed to gaps that describe a
#: consequence which was built and is incomplete. Only these make the decision `refused` and
#: only these make `acr ask` exit non-zero. The distinction matters because the NCCN routing
#: produces sixteen gaps and is still a correct, useful, exit-zero answer; a predicate naming
#: a field no spec declares produces one gap and is not an answer at all.
REFUSING_KINDS = frozenset({UNKNOWN_VARIABLE, AMBIGUOUS_VARIABLE, UNKNOWN_FIELD_IN_PREDICATE,
                            VALUE_OUTSIDE_DECLARED_DOMAIN, UNDECLARED_VALUE_SET,
                            MALFORMED_TERM, UNSATISFIABLE_TERM, UNCLASSIFIED})

PLACEHOLDER = "PLACEHOLDER_REQUIRES_HUMAN_INPUT"

#: Where a NEW_VARIABLE route sends the human. Checked for existence at route time, because a
#: route to a skill that is not in the tree is the "declared but never run" failure this repo
#: keeps finding — a constraint written down, wired to nothing, and believed for months.
SPEC_AUTHORING_SKILL = "skills/spec-authoring"


class IntakeError(ValueError):
    """Base for every way a question fails to become a routing decision."""


class PredicateRefused(IntakeError):
    """Someone tried to evaluate a predicate the checker refused, or one it could not complete."""


@dataclass(frozen=True)
class Gap:
    """Something the routing could not close, and the specific thing a human must do about it.

    `remedy` is mandatory and is prose addressed to a person. An earlier shape had a
    `severity` int instead; severities get sorted, skimmed and ignored, and the gap list
    became a scoreboard nobody acted on. A remedy cannot be skimmed — it either names a next
    action or it visibly does not.
    """

    kind: str
    subject: str
    detail: str
    remedy: str
    context: tuple[str, ...] = ()

    @property
    def refusing(self) -> bool:
        return self.kind in REFUSING_KINDS

    def to_dict(self) -> dict:
        return {"kind": self.kind, "subject": self.subject, "detail": self.detail,
                "remedy": self.remedy, "context": list(self.context),
                "refusing": self.refusing}


# ------------------------------------------------------------------------- classification
@dataclass(frozen=True)
class Classification:
    """What the classifier believes the question means. Advisory in every field.

    Nothing here is trusted. `outcome` is reconciled against `spec.data_source`, `variables`
    are put through `VariableCatalog.resolve`, `predicate` terms are checked term by term,
    and `missing_inputs` is the classifier's only sanctioned way to mention something the
    vocabulary does not contain — a name invented anywhere else in this object becomes a
    refusing gap rather than a route.
    """

    outcome: str | None = None
    rationale: str = ""
    classifier: str = "none"
    variables: tuple[str, ...] = ()
    predicate: tuple[dict, ...] = ()
    guideline_id: str = ""
    recommendation_ids: tuple[str, ...] = ()
    proposed_variable: dict = field(default_factory=dict)
    missing_inputs: tuple[str, ...] = ()
    model_calls: int = 0
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "rationale": self.rationale,
                "classifier": self.classifier, "variables": list(self.variables),
                "predicate": [dict(t) for t in self.predicate],
                "guideline_id": self.guideline_id,
                "recommendation_ids": list(self.recommendation_ids),
                "proposed_variable": dict(self.proposed_variable),
                "missing_inputs": list(self.missing_inputs),
                "model_calls": self.model_calls}


class Classifier(Protocol):
    """Question + vocabulary -> Classification. The only place a model may be consulted."""

    def classify(self, question: str, vocab: dict) -> Classification: ...


def vocabulary(catalog: VariableCatalog, guidelines: Sequence[Guideline] = ()) -> dict:
    """Everything a classifier is allowed to know. Deliberately not the chart.

    Allowable value lists are truncated at twenty. `histology` in the NCCN value sets runs to
    forty-one codes and `class_of_case` to twenty-four; shipping all of them teaches the
    classifier to answer with codes rather than with variable names, and the codes it invents
    from that pattern all look plausible. Truncation is visible (`…`), so the classifier can
    see it is not being shown a closed world.
    """
    variables = []
    for e in catalog.entries():
        av = list(e.allowable_values or ())
        variables.append({
            "name": e.name, "spec_id": e.spec_id, "type": e.type, "format": e.format,
            "allowable_values": av[:20] + (["…"] if len(av) > 20 else []),
            "data_source": e.data_source,
            "description": re.sub(r"\s+", " ", e.description).strip()[:240],
        })
    return {
        "variables": variables,
        "guidelines": [{"guideline_id": g.guideline_id,
                        "recommendations": [{"id": r.id, "title": r.title}
                                            for r in g.recommendations]}
                       for g in guidelines],
        "outcomes": dict(OUTCOME_MEANINGS),
    }


class ExactNameClassifier:
    """Zero model calls. Resolves a question that is already a name, and nothing else.

    `acr ask primary_site` and `acr ask NCCN-NSCLC-SUBSET` should not cost a model call or a
    network round trip, and routing the shipped guideline must work on a machine with no
    credentials at all — which is also what lets the test suite exercise the whole consequence
    path offline.

    It does NOT try to be clever about prose. There is no keyword table here and there must
    not be one: "does this patient have stage II-IIIA resected NSCLC" contains the substring
    `stage`, and a keyword classifier would route it to the stage spec, drop the histology and
    the resection, and return an answer to a question nobody asked. Substring matching on a
    name a human typed is the same mistake that filed `Fine-Needle-Report` outside
    `["Pathology", "Cytology"]`. Prose gets `None`, and `None` becomes a gap that says to pass
    `--model`.
    """

    name = "exact-name"

    def __init__(self, catalog: VariableCatalog, guidelines: Sequence[Guideline] = ()):
        self.catalog = catalog
        self.guidelines = list(guidelines)

    def classify(self, question: str, vocab: dict) -> Classification:
        key = normalise_name(question)
        for g in self.guidelines:
            if key == normalise_name(g.guideline_id):
                return Classification(
                    GUIDELINE_RULE, f"{question!r} is the id of a shipped guideline",
                    self.name, guideline_id=g.guideline_id,
                    recommendation_ids=tuple(r.id for r in g.recommendations))
            for r in g.recommendations:
                if key == normalise_name(r.id):
                    return Classification(
                        GUIDELINE_RULE, f"{question!r} is a recommendation id in "
                                        f"{g.guideline_id}",
                        self.name, guideline_id=g.guideline_id, recommendation_ids=(r.id,))
        if key in self.catalog.known_aliases():
            return Classification(
                EXISTING_VARIABLE, f"{question!r} is a name the catalogue already indexes",
                self.name, variables=(question.strip(),))
        return Classification(
            None,
            "not a variable name, a spec id, a STORE item, a guideline id or a "
            "recommendation id — reading a sentence is judgement and needs a model",
            self.name)


class StubClassifier:
    """A fixed answer, for tests and for reproducing a model classification without the model.

    Every test in `tests/test_intake.py` uses this. A suite that needs a network call is a
    suite that rots: it goes red on a credential rotation, someone marks it skip, and the
    consequence path — which is the part that must never break — stops being tested at all.
    """

    name = "stub"

    def __init__(self, answers: dict[str, Classification] | Classification):
        self._one = answers if isinstance(answers, Classification) else None
        self._many = {} if self._one else {normalise_name(k): v for k, v in answers.items()}

    def classify(self, question: str, vocab: dict) -> Classification:
        c = self._one or self._many.get(normalise_name(question))
        if c is None:
            return Classification(None, "the stub has no answer for this question", self.name)
        return replace(c, classifier=c.classifier or self.name)


_CLASSIFIER_SYSTEM = """\
You are the intake router of a clinical chart-review platform. You are given ONE question and \
the platform's complete vocabulary. You classify the question. You do not answer it, you never \
see a patient chart, and you must not invent a variable name.

Reply with a single JSON object:

  outcome              one of the five names below, exactly
  rationale            one sentence: what the question is asking for
  variables            names taken VERBATIM from the vocabulary. If the name you want is not \
in the vocabulary, do not put it here.
  predicate            for COMPOSITION only: a list of condition terms, grammar below
  guideline_id         for GUIDELINE_RULE only
  recommendation_ids   for GUIDELINE_RULE only
  proposed_variable    for NEW_VARIABLE only: {"name","question","why_not_composable"}
  missing_inputs       every input the question needs that the vocabulary does not contain. \
This is the ONLY place a name that is not in the vocabulary may appear.

The five outcomes:
%(outcomes)s

Condition grammar (this is the platform's own rule-engine grammar; anything else is rejected):
  {"op":"equals","var":NAME,"value":LITERAL}
  {"op":"not_equals","var":NAME,"value":LITERAL}
  {"op":"in_set","var":NAME,"values":[LITERAL,...]}
  {"op":"not_in_set","var":NAME,"values":[LITERAL,...]}
  {"op":"matches","var":NAME,"pattern":REGEX}
  {"op":"is_present","var":NAME} / {"op":"is_absent","var":NAME}
  {"op":"at_least","var":NAME,"value":NUMBER} / {"op":"at_most",...}
  {"op":"any_of","conditions":[...]} / {"op":"all_of","conditions":[...]} / {"op":"not","condition":{...}}
  {"op":"days_between","from":NAME,"to":NAME,"min_days":N,"max_days":N}
  {"op":"on_or_before","from":NAME,"to":NAME}
The terms are ANDed. Every literal must be one the variable's declared format or \
allowable_values permits; a literal that is not is rejected and the whole predicate is refused.

Choose COMPOSITION over NEW_VARIABLE whenever the question is decidable by a predicate over \
existing variables, even if some inputs are missing — put the missing ones in missing_inputs. \
Choose NEW_VARIABLE only when the question needs its own chart review with its own evidence \
rules ("was this patient a candidate for surgery?"). Choose WRONG_DATA_SOURCE when the answer \
is not in clinical notes at any depth.
"""


class ModelClassifier:
    """One model call per question, and the call sees the question and the vocabulary only.

    `extract_json(..., require="outcome")` rather than `LLMClient.json_chat`, because
    gpt-5.6-luna leaks its tool-call channel into the text channel and emits a preamble object
    before the answer — `llm.extract_json` documents the trace. "First parseable object wins"
    returns the preamble, whose `.get("outcome")` is None, and the caller then records a
    perfectly good classification as a failure.
    """

    def __init__(self, client, name: str = ""):
        self.client = client
        self.name = name or f"model:{getattr(getattr(client, 'cfg', None), 'model', '?')}"

    def messages(self, question: str, vocab: dict) -> list[dict]:
        outcomes = "\n".join(f"  {k}: {v}" for k, v in OUTCOME_MEANINGS.items())
        return [
            {"role": "system", "content": _CLASSIFIER_SYSTEM % {"outcomes": outcomes}},
            {"role": "user", "content": "VOCABULARY:\n"
                                        + json.dumps(vocab, indent=1, ensure_ascii=False)
                                        + f"\n\nQUESTION: {question}"},
        ]

    def classify(self, question: str, vocab: dict) -> Classification:
        from .llm import extract_json          # local: keeps litellm off the import path of
                                               # every offline test that never calls a model
        resp = self.client.chat(self.messages(question, vocab))
        data = extract_json(resp.content, require="outcome")
        out = str(data.get("outcome") or "").strip().upper()
        return Classification(
            outcome=out if out in OUTCOMES else None,
            rationale=str(data.get("rationale") or "")
                      or (f"the classifier returned outcome {data.get('outcome')!r}, which is "
                          f"not one of {list(OUTCOMES)}" if out not in OUTCOMES else ""),
            classifier=self.name,
            variables=_strs(data.get("variables")),
            predicate=tuple(t for t in (data.get("predicate") or []) if isinstance(t, dict)),
            guideline_id=str(data.get("guideline_id") or ""),
            recommendation_ids=_strs(data.get("recommendation_ids")),
            proposed_variable=dict(data.get("proposed_variable") or {}),
            missing_inputs=_strs(data.get("missing_inputs")),
            model_calls=1,
            raw=data,
        )


def _strs(x: Any) -> tuple[str, ...]:
    if x is None:
        return ()
    seq = x if isinstance(x, (list, tuple)) else [x]
    return tuple(str(v).strip() for v in seq if str(v).strip())


# --------------------------------------------------------------------------- composition
@dataclass(frozen=True)
class Term:
    """One condition of a predicate, plus the check the catalogue put it through."""

    condition: dict
    variables: tuple[str, ...] = ()
    spec_ids: tuple[str, ...] = ()
    ok: bool = True
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"condition": dict(self.condition), "variables": list(self.variables),
                "spec_ids": list(self.spec_ids), "ok": self.ok,
                "problems": list(self.problems)}


@dataclass(frozen=True)
class PredicateVerdict:
    truth: str
    unknown: tuple[str, ...] = ()
    used: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"truth": self.truth, "unknown": list(self.unknown),
                "used": list(self.used), "notes": list(self.notes)}


@dataclass(frozen=True)
class CompositionPredicate:
    """A checkable expression over variables that already have specs. Not a spec.

    Two flags, and they mean different things:

      `checkable` false — a term named a field or a value no spec produces. There is nothing
        here to run and `evaluate` refuses outright.
      `complete` false — every emitted term checks out, but the classifier declared inputs the
        vocabulary does not contain, so the predicate answers a WEAKER question than the one
        asked. "stage II-IIIA *resected* NSCLC" without `surgical_resection_extent` is
        "stage II-IIIA NSCLC", and running it silently would hand back a cohort that is
        larger than the one requested with nothing on the artifact saying so. `evaluate`
        refuses that too, unless the caller passes `accept_partial=True` and thereby writes
        the decision down.
    """

    terms: tuple[Term, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    unknown_value_codes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    value_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def checkable(self) -> bool:
        return bool(self.terms) and all(t.ok for t in self.terms)

    @property
    def complete(self) -> bool:
        return self.checkable and not self.missing_inputs

    @property
    def variables(self) -> list[str]:
        return list(dict.fromkeys(n for t in self.terms for n in t.variables))

    def conditions(self) -> list[dict]:
        """The terms in `concordance`'s grammar, ready to paste into a guideline's
        `applies_when`. Emitted even when the predicate is refused — a reader needs to see
        the thing that was rejected, and the refusal lives on `checkable`, not on absence."""
        return [dict(t.condition) for t in self.terms]

    def expression(self) -> str:
        """The predicate as one line a human can check against the sentence they typed."""
        parts = [_render(t.condition) for t in self.terms]
        if not parts:
            return "<no terms>"
        text = " AND ".join(parts)
        if self.missing_inputs:
            text += " AND <NOT COMPUTABLE: " + ", ".join(self.missing_inputs) + ">"
        return text

    def evaluate(self, values: dict[str, Any], *, accept_partial: bool = False
                 ) -> PredicateVerdict:
        """Three-valued, over `concordance`'s engine. UNKNOWN is a real answer, not a FALSE.

        Values are the same shape `assess` takes: {name: {"status": ..., "value": ...}}. A
        bare scalar is rejected there, on purpose — a value with no status cannot be told
        apart from a guess.
        """
        if not self.checkable:
            raise PredicateRefused(
                "this predicate was refused by the catalogue check and has no meaning: "
                + "; ".join(p for t in self.terms for p in t.problems))
        if self.missing_inputs and not accept_partial:
            raise PredicateRefused(
                f"this predicate cannot express {', '.join(self.missing_inputs)}, which no "
                f"spec produces, so it answers a weaker question than the one asked. Pass "
                f"accept_partial=True to run it anyway and record that you did.")
        env = _Env(_bind_values(values, self.unknown_value_codes), dict(self.value_sets))
        v = _eval_all(self.conditions(), env)
        return PredicateVerdict(v.truth, tuple(v.unknown), tuple(v.used), tuple(v.notes))

    def to_dict(self) -> dict:
        return {"expression": self.expression(), "conditions": self.conditions(),
                "terms": [t.to_dict() for t in self.terms],
                "missing_inputs": list(self.missing_inputs),
                "checkable": self.checkable, "complete": self.complete,
                "variables": self.variables}


def _bind_values(values: dict[str, Any], sentinels: dict[str, tuple[str, ...]]
                 ) -> dict[str, VariableValue]:
    """Coerce caller values and demote registry sentinels, exactly as `concordance._bind` does.

    Not imported from there only because `_bind` takes a whole `Guideline`, and a composition
    has none. Same three lines, same reason: `pathologic_stage_group = 99` is present,
    well-formed and means "nobody staged this patient", so it must reach the engine as UNKNOWN
    rather than as a value that happens not to be in {IIA, IIB, IIIA}.
    """
    from .concordance import _coerce, _norm            # local: same-module private helpers
    out: dict[str, VariableValue] = {}
    for k, raw in (values or {}).items():
        vv = _coerce(k, raw)
        codes = {_norm(c) for c in sentinels.get(k, ())}
        if vv.resolution == "KNOWN" and _norm(vv.value) in codes:
            vv = replace(vv, unknown_sentinel=True)
        out[k] = vv
    return out


def _render(cond: dict) -> str:
    op = cond.get("op")
    if op in ("all_of", "any_of"):
        joiner = " AND " if op == "all_of" else " OR "
        return "(" + joiner.join(_render(c) for c in cond.get("conditions") or []) + ")"
    if op == "not":
        return "NOT " + _render(cond.get("condition") or {})
    var = cond.get("var", "?")
    if op in ("in_set", "not_in_set"):
        vals = cond.get("values") or []
        body = ("{" + ", ".join(str(v) for v in vals) + "}") if vals else f"@{cond.get('set')}"
        return f"{var} {'in' if op == 'in_set' else 'not in'} {body}"
    if op == "matches":
        return f"{var} matches /{cond.get('pattern')}/"
    if op in ("equals", "not_equals"):
        return f"{var} {'==' if op == 'equals' else '!='} {cond.get('value')!r}"
    if op in ("at_least", "at_most"):
        return f"{var} {'>=' if op == 'at_least' else '<='} {cond.get('value')}"
    if op in ("is_true", "is_false", "is_present", "is_absent"):
        return f"{op.replace('_', ' ')}({var})"
    if op == "days_between":
        return (f"days({cond.get('from')} -> {cond.get('to')}) in "
                f"[{cond.get('min_days')}, {cond.get('max_days')}]")
    if op == "on_or_before":
        return f"{cond.get('from')} <= {cond.get('to')}"
    return json.dumps(cond, sort_keys=True)


def _field_of(catalog: VariableCatalog, name: str) -> tuple[str, Any] | None:
    """(spec_id, OutputField) for an exact field name, or None. No prefix, no fuzz."""
    for sid in sorted(catalog.specs):
        for f in catalog.specs[sid].fields:
            if f.name == name:
                return sid, f
    return None


def _literals(cond: dict, value_sets: dict[str, tuple[str, ...]]) -> list[Any]:
    op = cond.get("op")
    if op in ("equals", "not_equals", "at_least", "at_most"):
        return [cond.get("value")]
    if op in ("in_set", "not_in_set"):
        if cond.get("values"):
            return list(cond["values"])
        return list(value_sets.get(str(cond.get("set")), ()))
    return []


def check_predicate(terms: Sequence[dict], catalog: VariableCatalog, *,
                    value_sets: dict[str, tuple[str, ...]] | None = None,
                    unknown_value_codes: dict[str, tuple[str, ...]] | None = None,
                    missing_inputs: Sequence[str] = ()
                    ) -> tuple[CompositionPredicate, list[Gap]]:
    """Check a proposed predicate against the specs, term by term. Nothing here is judgement.

    Four checks, and each one closes a failure this repo has already paid for:

      1. the variable must be an EXACT field name of exactly one spec. `check_guideline_bindings`
         exists because `guidelines/` once named `ajcc_pathologic_stage` where the spec's field
         is `pathologic_stage_group`. Nothing errored; the variable simply never arrived, every
         case came back NOT_ASSESSABLE naming a variable the operator believed was requested,
         and the concordance denominator went quietly to zero.
      2. every literal must satisfy that field's declared `format` and `allowable_values`, and
         it is checked by calling `answer_checks.check_field_formats` — the same function the
         gate runs on a submitted answer. A predicate may not ask for a value the runtime
         would reject if an agent produced it. That function was itself written after
         `primary_site="C3412"` — four digits against a declared `C\\d{3}` — passed the gate
         stamped as validated.
      3. a `matches` pattern must compile, and where the field declares `allowable_values` at
         least one member must match it. A regex matching nothing is a term that silently
         empties the cohort.
      4. the grammar itself is checked by `concordance._validate_condition`, so a term this
         layer accepts is a term the rule engine can execute. Two validators would drift, and
         the one that drifts is always the copy.

    Returns the predicate (with per-term verdicts attached even when refused) and the gaps.
    """
    vs = dict(value_sets or {})
    gaps: list[Gap] = []
    checked: list[Term] = []

    for raw in terms:
        problems: list[str] = []
        if not isinstance(raw, dict):
            checked.append(Term({"op": "?"}, ok=False,
                                problems=(f"term is not a mapping: {raw!r}",)))
            gaps.append(Gap(MALFORMED_TERM, str(raw), "a predicate term must be a mapping",
                            "re-run the classification; if it repeats, the grammar block in "
                            "the intake prompt is not being followed"))
            continue

        problems += _validate_condition(raw, vs)
        if raw.get("op") in ("in_set", "not_in_set") and raw.get("set") and str(raw["set"]) not in vs:
            gaps.append(Gap(UNDECLARED_VALUE_SET, str(raw["set"]),
                            f"the term references value set {raw['set']!r}, which no loaded "
                            f"guideline declares",
                            "pass --guideline for the file that declares it, or inline the "
                            "members with `values`"))

        names = list(dict.fromkeys(_referenced(raw)))
        spec_ids: list[str] = []
        for n in names:
            hit = _field_of(catalog, n)
            if hit is None:
                problems.append(f"no shipped spec declares a field named {n!r}")
                gaps.append(Gap(
                    UNKNOWN_FIELD_IN_PREDICATE, n,
                    f"the predicate reads {n!r}; the catalogue in "
                    f"{catalog.directory or 'specs'} produces "
                    f"{', '.join(catalog.known_names())}",
                    "either the classifier misnamed an existing field — fix the name — or the "
                    "variable does not exist, in which case this is a NEW_VARIABLE and needs a "
                    f"spec ({SPEC_AUTHORING_SKILL}), not a composition term",
                    context=tuple(catalog.known_names())))
                continue
            sid, fobj = hit
            spec_ids.append(sid)
            for lit in _literals(raw, vs):
                if lit is None or str(lit).strip() == "":
                    continue
                # The gate's own format checker, on one field and one value. Note what it
                # inherits: STORE.390 declares `format: "CCYYMMDD"`, which is registry
                # notation and not a regex, so a real date literal fails here. That is the
                # spec being wrong, not the question — and it is exactly why the message
                # below names the spec and the pattern instead of just saying "invalid".
                for bad in check_field_formats([fobj], {n: lit}):
                    problems.append(bad)
                    gaps.append(Gap(
                        VALUE_OUTSIDE_DECLARED_DOMAIN, f"{n}={lit!r}", bad,
                        f"either the literal is wrong, or {sid} declares a domain that does "
                        f"not cover it — check the field's format/allowable_values before "
                        f"changing the question",
                        context=(sid,)))
            if raw.get("op") == "matches" and getattr(fobj, "allowable_values", None):
                pat = str(raw.get("pattern", ""))
                try:
                    if not any(re.fullmatch(pat, str(v)) for v in fobj.allowable_values):
                        problems.append(
                            f"pattern {pat!r} matches none of {n}'s allowable values")
                        gaps.append(Gap(
                            UNSATISFIABLE_TERM, f"{n} matches /{pat}/",
                            f"no allowable value of {n} in {sid} matches this pattern, so the "
                            f"term can never be true and the cohort is empty by construction",
                            "fix the pattern, or use in_set with the values you mean",
                            context=(sid,)))
                except re.error:
                    pass                        # already reported by _validate_condition

        checked.append(Term(dict(raw), tuple(names), tuple(dict.fromkeys(spec_ids)),
                            ok=not problems, problems=tuple(problems)))

    pred = CompositionPredicate(tuple(checked), tuple(missing_inputs),
                                dict(unknown_value_codes or {}), vs)

    # One gap, not one per variable. Every field with a registry sentinel that nothing
    # declares will evaluate that sentinel as an ordinary non-member: a patient coded
    # `pathologic_stage_group = 99` comes out "not in the cohort" rather than "we do not know",
    # which is the same inflation `Guideline.unknown_value_codes` was added to refuse, arriving
    # here instead. It is a note rather than a refusal because this layer genuinely cannot tell
    # a sentinel from a value — only a clinical author can.
    unpoliced = [v for v in pred.variables if v not in pred.unknown_value_codes]
    if unpoliced:
        gaps.append(Gap(
            NO_SENTINEL_POLICY, ", ".join(unpoliced),
            "no loaded guideline declares unknown_value_codes for these, so a registry "
            "sentinel (stage 99, histology 8000, site C809) will evaluate as an ordinary "
            "non-member and quietly drop the patient out of the cohort instead of leaving "
            "them UNKNOWN",
            "declare the sentinels under unknown_value_codes in the guideline that will use "
            "this predicate, then re-run"))

    for name in missing_inputs:
        gaps.append(Gap(
            SPEC_AUTHORING_REQUIRED, name,
            f"the question needs {name!r} and no shipped spec produces it, so the predicate "
            f"answers a weaker question than the one asked",
            f"author a spec for {name!r} ({SPEC_AUTHORING_SKILL}), or accept the weaker "
            f"predicate explicitly"))
    return pred, gaps


# ------------------------------------------------------------------------ spec skeletons
#: The questions a spec cannot be finished without. Each is bound to the key its answer goes
#: into, so the human is told where the answer lands and not merely that one is wanted.
OPEN_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("question", "What exactly is being asked, in one sentence a registrar would recognise?"),
    ("data_source", "Is this answerable from clinical notes at all, or does it live in "
                    "registration/billing/registry systems? `outside_notes` forces every run "
                    "to SPEC_INSUFFICIENT by design — see STORE.610."),
    ("fields[].allowable_values", "What is the value domain, and is there a "
                                  "not-otherwise-specified value? A NOS code is a positive "
                                  "claim that the specific value is unknown."),
    ("decision_rule", "What is the decision boundary, including the cases that look like it "
                      "and are not?"),
    ("evidence_rules.counts_as_evidence", "WHAT COUNTS AS EVIDENCE: which document types, "
                                          "and what wording in them, may establish this?"),
    ("evidence_rules.does_not_count", "WHAT DOES NOT COUNT: name the tempting sources that "
                                      "are inadmissible — imaging impressions, a problem-list "
                                      "entry, a clinician restating an outside report."),
    ("conflict_rules", "ON CONFLICT: two documents disagree — which wins, and why? Most "
                       "specific? Most recent? The pathology over the clinic note?"),
    ("proof_obligation.for_positive.witness", "Which strata may a citation for a positive "
                                              "answer come from? Prose here is a wish; only "
                                              "`witness` is read by the gate."),
    ("proof_obligation.for_negative", "WHEN TO ABSTAIN, and what must demonstrably have been "
                                      "searched and read before an absence may be asserted."),
    ("abstention", "Which abstention applies when: SPEC_INSUFFICIENT (the spec does not cover "
                   "this case) versus EVIDENCE_INSUFFICIENT (the spec is clear, the chart is "
                   "not). They are different answers and are used differently downstream."),
    ("source_authority", "Which manual, which edition, which item number — or an explicit "
                         "statement that this is not a registry item."),
)


@dataclass(frozen=True)
class SpecSkeleton:
    """A spec with every judgement slot left empty, plus the questions that fill them.

    The skeleton is deliberately NOT LOADABLE. `data_source` carries the placeholder string
    and `spec.ExtractionSpec` declares it `Literal["notes", "outside_notes"]`, so pydantic
    refuses the file the moment anything tries to run it. That is a hard stop rather than a
    convention: this repo's standing fact is that all four original specs were written by a
    language model in one commit and no registrar has read a line of any of them. A
    model-authored draft that loads is a model-authored draft that runs.
    """

    variable: str
    question: str
    yaml_text: str
    open_questions: tuple[dict, ...] = ()
    route: str = SPEC_AUTHORING_SKILL
    why_not_composable: str = ""

    def to_dict(self) -> dict:
        return {"variable": self.variable, "question": self.question,
                "yaml_text": self.yaml_text,
                "open_questions": [dict(q) for q in self.open_questions],
                "route": self.route, "why_not_composable": self.why_not_composable}


def spec_skeleton(variable: str, question: str, *, why_not_composable: str = "") -> SpecSkeleton:
    """Emit the draft and the question list. No file is written and none may be."""
    slug = normalise_name(variable) or "unnamed_variable"
    body = {
        "spec_id": f"DRAFT.{slug}",
        "spec_version": "0.0.0",
        "status": "DRAFT_REQUIRES_HUMAN_COMPLETION",
        "source_authority": {"document": PLACEHOLDER, "items": [PLACEHOLDER]},
        "data_source": PLACEHOLDER,
        "question": question.strip() or PLACEHOLDER,
        "fields": [{"name": slug, "type": "string", "nullable": True,
                    "allowable_values": [PLACEHOLDER], "description": PLACEHOLDER}],
        "decision_rule": [PLACEHOLDER],
        "evidence_rules": {"counts_as_evidence": [PLACEHOLDER],
                           "does_not_count": [PLACEHOLDER]},
        "conflict_rules": [{"if": PLACEHOLDER, "then": PLACEHOLDER}],
        "proof_obligation": {
            "for_positive": {"statement": PLACEHOLDER, "witness": {slug: [PLACEHOLDER]}},
            "for_negative": {"mode": PLACEHOLDER, "required_coverage": [PLACEHOLDER],
                             "required_keywords": [PLACEHOLDER],
                             "required_doc_types_read": [PLACEHOLDER]},
        },
        "abstention": {"SPEC_INSUFFICIENT": PLACEHOLDER, "EVIDENCE_INSUFFICIENT": PLACEHOLDER},
    }
    header = (
        f"# DRAFT SPEC — generated by acr.intake for a question no shipped spec answers.\n"
        f"# It does not load: `data_source` is a placeholder and spec.ExtractionSpec declares\n"
        f"# it Literal['notes','outside_notes'], so pydantic refuses this file until a human\n"
        f"# has answered the questions below. Do not fix that by guessing the value.\n"
        f"#\n# Route: {SPEC_AUTHORING_SKILL}\n"
    )
    text = header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=96)
    return SpecSkeleton(slug, question.strip(), text,
                        tuple({"key": k, "question": q} for k, q in OPEN_QUESTIONS),
                        why_not_composable=why_not_composable)


def unfinished_placeholders(text_or_data: str | dict) -> list[str]:
    """Dotted paths of every slot still holding the placeholder. Empty means finished.

    Not decoration: the way a draft becomes a live spec is that someone fills it in, and the
    way that goes wrong is that they fill in nine of eleven slots and ship it. This returns
    the two.
    """
    data = yaml.safe_load(text_or_data) if isinstance(text_or_data, str) else text_or_data
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and PLACEHOLDER in node:
            out.append(path)

    walk(data, "")
    return out


# ------------------------------------------------------------------------------ routing
@dataclass(frozen=True)
class RoutedInput:
    """One input, and the outcome code assigned it. Same five outcomes at every depth."""

    name: str
    outcome: str
    spec_id: str = ""
    data_source: str = ""
    declared_source: str = ""
    recommendation_ids: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "outcome": self.outcome, "spec_id": self.spec_id,
                "data_source": self.data_source, "declared_source": self.declared_source,
                "recommendation_ids": list(self.recommendation_ids), "note": self.note}


@dataclass(frozen=True)
class RoutingDecision:
    """One question in, one decision out. `outcome` is the CODE's, never the classifier's.

    `classified_as` keeps what the model said, because the two disagreeing is information: a
    classifier that keeps calling `class_of_case` an ordinary extraction is a prompt to fix,
    and it is only visible if both verdicts survive to the report.
    """

    question: str
    outcome: str | None
    classified_as: str | None = None
    classifier: str = "none"
    rationale: str = ""
    resolved: tuple[RoutedInput, ...] = ()
    predicate: CompositionPredicate | None = None
    skeleton: SpecSkeleton | None = None
    guideline_id: str = ""
    recommendation_ids: tuple[str, ...] = ()
    gaps: tuple[Gap, ...] = ()
    model_calls: int = 0

    @property
    def refused(self) -> bool:
        return self.outcome is None or any(g.refusing for g in self.gaps)

    @property
    def unresolved(self) -> list[RoutedInput]:
        return [r for r in self.resolved if r.outcome != EXISTING_VARIABLE]

    def spec_ids(self) -> list[str]:
        return list(dict.fromkeys(r.spec_id for r in self.resolved
                                  if r.spec_id and r.outcome == EXISTING_VARIABLE))

    def to_dict(self) -> dict:
        return {
            "question": self.question, "outcome": self.outcome,
            "classified_as": self.classified_as, "classifier": self.classifier,
            "rationale": self.rationale, "refused": self.refused,
            "resolved": [r.to_dict() for r in self.resolved],
            "spec_ids": self.spec_ids(),
            "predicate": self.predicate.to_dict() if self.predicate else None,
            "skeleton": self.skeleton.to_dict() if self.skeleton else None,
            "guideline_id": self.guideline_id,
            "recommendation_ids": list(self.recommendation_ids),
            "gaps": [g.to_dict() for g in self.gaps],
            "model_calls": self.model_calls,
            "reads_charts": False,
        }


def _route_one_name(name: str, catalog: VariableCatalog, *, declared_source: str = "",
                    rec_ids: Sequence[str] = ()) -> tuple[RoutedInput, list[Gap]]:
    """A single variable name -> (RoutedInput, gaps). The one place a name becomes a route.

    Exact match, and the data-source verdict comes off the spec. A classifier that thinks
    `class_of_case` is an ordinary extraction does not get to be right about that: STORE.610
    declares `data_source: outside_notes` and `graph.py` forces every run of it to
    SPEC_INSUFFICIENT at finalize, so the honest outcome is WRONG_DATA_SOURCE whatever the
    sentence looked like.
    """
    gaps: list[Gap] = []
    try:
        res = catalog.resolve(name)
    except AmbiguousVariableError as e:
        gaps.append(Gap(AMBIGUOUS_VARIABLE, name, str(e),
                        f"name the spec_id you mean: {', '.join(e.spec_ids)}",
                        context=tuple(e.spec_ids)))
        return RoutedInput(name, NEW_VARIABLE, note="ambiguous", declared_source=declared_source,
                           recommendation_ids=tuple(rec_ids)), gaps
    except UnknownVariableError as e:
        gaps.append(Gap(
            UNKNOWN_VARIABLE, name,
            f"nothing in {catalog.directory or 'specs'} produces {name!r}"
            + (f"; did you mean {', '.join(e.suggestions.get(name, []))}?"
               if e.suggestions.get(name) else ""),
            f"if the name is a typo, fix it; if the variable really does not exist, this is a "
            f"NEW_VARIABLE and needs a spec ({SPEC_AUTHORING_SKILL})",
            context=tuple(e.known)))
        return RoutedInput(name, NEW_VARIABLE, note="no spec produces this name",
                           declared_source=declared_source,
                           recommendation_ids=tuple(rec_ids)), gaps
    except VariableResolutionError as e:                      # empty request, etc.
        gaps.append(Gap(UNKNOWN_VARIABLE, name, str(e), "name a variable"))
        return RoutedInput(name, NEW_VARIABLE, note=str(e),
                           recommendation_ids=tuple(rec_ids)), gaps

    v = res.variables[0]
    if v.data_source != "notes":
        gaps.append(Gap(
            OUTSIDE_NOTES, v.name,
            f"{v.spec_id} declares data_source={v.data_source}: every agent run of it is "
            f"forced to SPEC_INSUFFICIENT / WRONG_DATA_SOURCE at finalize, so a chart review "
            f"pays for a full pass to arrive at a constant",
            f"supply {v.name} from the registry limited dataset and merge it with "
            f"`acr concord --extra-variables`; do not extract it",
            context=(v.spec_id,)))
        return RoutedInput(v.name, WRONG_DATA_SOURCE, v.spec_id, v.data_source,
                           declared_source, tuple(rec_ids)), gaps
    return RoutedInput(v.name, EXISTING_VARIABLE, v.spec_id, v.data_source,
                       declared_source, tuple(rec_ids)), gaps


def route_guideline(guideline: Guideline, catalog: VariableCatalog, *,
                    recommendation_ids: Sequence[str] = (), question: str = "",
                    skills_dir: str | Path = "skills") -> RoutingDecision:
    """Decompose a guideline rule into its declared inputs and route every one of them.

    Zero model calls, at any depth. The input names are already written down in the guideline
    YAML — there is nothing to interpret, so interpreting it would be paying a model to read
    a list. The recursion terminates because every `required_input` carries a declared
    `source`, and each of the three sources maps to exactly one of the five outcomes:

        extraction_spec           -> resolve through the catalogue (EXISTING_VARIABLE, or
                                     WRONG_DATA_SOURCE if the spec says outside_notes)
        registry_limited_dataset  -> WRONG_DATA_SOURCE
        not_yet_extractable       -> NEW_VARIABLE

    The two cross-checks below exist because the guideline and the specs are two places that
    each declare where a variable comes from, and two declarations drift. `check_guideline_bindings`
    already covers one direction (an extraction_spec input naming a field no spec has); these
    cover the other (a guideline still calling something unextractable that now has a spec, and
    a guideline routing to the registry for something a spec claims to extract from notes).
    """
    recs = [r for r in guideline.recommendations
            if not recommendation_ids or r.id in set(recommendation_ids)]
    wanted = set(recommendation_ids)
    gaps: list[Gap] = []
    for missing in sorted(wanted - {r.id for r in guideline.recommendations}):
        gaps.append(Gap(ROUTE_TARGET_MISSING, missing,
                        f"{guideline.guideline_id} declares no recommendation {missing!r}",
                        "check the id against the guideline: "
                        + ", ".join(r.id for r in guideline.recommendations)))

    by_name: dict[str, RoutedInput] = {}
    for rec in recs:
        for decl in rec.required_inputs:
            if not isinstance(decl, dict):
                continue
            name = str(decl.get("name") or "").strip()
            if not name:
                continue
            src = str(decl.get("source") or "")
            item = str(decl.get("item") or "")
            prior = by_name.get(name)
            if prior is not None:
                by_name[name] = replace(
                    prior, recommendation_ids=tuple(dict.fromkeys(prior.recommendation_ids
                                                                  + (rec.id,))))
                continue

            if src == "extraction_spec":
                ri, g = _route_one_name(name, catalog, declared_source=src, rec_ids=(rec.id,))
                by_name[name] = ri
                gaps.extend(g)
                continue

            hit = _field_of(catalog, name)
            if src == "registry_limited_dataset":
                note = f"{item} — declared as coming from the registry limited dataset"
                if hit and catalog.specs[hit[0]].data_source == "notes":
                    gaps.append(Gap(
                        GUIDELINE_SOURCE_DISAGREES_WITH_CATALOG, name,
                        f"{guideline.guideline_id} routes {name!r} to the registry, but "
                        f"{hit[0]} declares data_source=notes and produces a field of that "
                        f"name — two declarations of where one variable comes from",
                        "settle it in one place: either drop the spec field or change the "
                        "guideline's source, then re-run",
                        context=(hit[0],)))
                by_name[name] = RoutedInput(name, WRONG_DATA_SOURCE,
                                            spec_id=(hit[0] if hit else ""),
                                            declared_source=src, recommendation_ids=(rec.id,),
                                            note=note)
                gaps.append(Gap(
                    OUTSIDE_NOTES, name,
                    f"{note}. No chart review can produce it; "
                    + (f"{hit[0]} exists precisely to refuse it" if hit
                       else "no spec exists to refuse it either"),
                    f"join it from the registry extract and merge with "
                    f"`acr concord --extra-variables`"
                    + ("" if hit else
                       f"; and consider a refusing spec like STORE.610 so the refusal is a "
                       f"shipped artifact rather than a runtime opinion"),
                    context=tuple(x for x in [hit[0] if hit else ""] if x)))
                if not hit:
                    gaps.append(Gap(
                        NO_SPEC_DECLARES_THE_REFUSAL, name,
                        f"nothing in {catalog.directory or 'specs'} declares that {name!r} is "
                        f"not in the notes, so the refusal lives only in this guideline",
                        f"author a refusing spec on the STORE.610 pattern "
                        f"({SPEC_AUTHORING_SKILL})"))
                continue

            # not_yet_extractable, or anything else the guideline invented.
            if hit:
                gaps.append(Gap(
                    GUIDELINE_SOURCE_DISAGREES_WITH_CATALOG, name,
                    f"{guideline.guideline_id} marks {name!r} not_yet_extractable, but {hit[0]} "
                    f"now declares a field of that name — the guideline is stale and the "
                    f"recommendation is blocking on an input that has since arrived",
                    f"change the input's source to extraction_spec with spec_id {hit[0]} and "
                    f"re-run; this changes the guideline hash, which is correct",
                    context=(hit[0],)))
            by_name[name] = RoutedInput(name, NEW_VARIABLE, spec_id=(hit[0] if hit else ""),
                                        declared_source=src or "unspecified",
                                        recommendation_ids=(rec.id,),
                                        note=item or "no spec exists")
            gaps.append(Gap(
                NOT_YET_EXTRACTABLE, name,
                f"{guideline.guideline_id} needs {name!r}" + (f" ({item})" if item else "")
                + " and declares it not_yet_extractable"
                + (f"; it is required by {', '.join(sorted({rec.id}))}" if rec.id else ""),
                f"author a spec for it ({SPEC_AUTHORING_SKILL}), or accept that every "
                f"recommendation reading it is NOT_ASSESSABLE",
                context=(rec.id,)))

    gaps += _skill_gaps(gaps, skills_dir)
    resolved = tuple(by_name.values())
    return RoutingDecision(
        question=question or guideline.guideline_id,
        outcome=GUIDELINE_RULE,
        classified_as=GUIDELINE_RULE,
        classifier="deterministic",
        rationale=f"{guideline.guideline_id} declares {len(recs)} recommendation(s) over "
                  f"{len(resolved)} distinct inputs; each input's `source` decides its route",
        resolved=resolved,
        guideline_id=guideline.guideline_id,
        recommendation_ids=tuple(r.id for r in recs),
        gaps=tuple(gaps),
        model_calls=0,
    )


def route(question: str, catalog: VariableCatalog, *, classifier: Classifier | None = None,
          guidelines: Sequence[Guideline] = (), skills_dir: str | Path = "skills"
          ) -> RoutingDecision:
    """One question -> one routing decision. The classifier judges; everything after is code."""
    guidelines = list(guidelines)
    clf = classifier or ExactNameClassifier(catalog, guidelines)
    cls = clf.classify(question, vocabulary(catalog, guidelines))
    gaps: list[Gap] = []

    if cls.outcome is None:
        # Not a sixth outcome. No decision was made, and the report says so rather than
        # picking the nearest one — a routing layer that guesses is the substring matcher
        # again, at a higher level and with more authority.
        return RoutingDecision(
            question, None, None, cls.classifier, cls.rationale, model_calls=cls.model_calls,
            gaps=(Gap(UNCLASSIFIED, question, cls.rationale or "no classification",
                      "pass --model to classify prose, or ask by exact variable name, spec "
                      "id, STORE item, guideline id or recommendation id"),))

    # ------------------------------------------------------------------ GUIDELINE_RULE
    if cls.outcome == GUIDELINE_RULE:
        g = next((x for x in guidelines
                  if normalise_name(x.guideline_id) == normalise_name(cls.guideline_id)), None)
        if g is None and len(guidelines) == 1 and not cls.guideline_id:
            g = guidelines[0]
        if g is None:
            return RoutingDecision(
                question, GUIDELINE_RULE, cls.outcome, cls.classifier, cls.rationale,
                model_calls=cls.model_calls,
                gaps=(Gap(ROUTE_TARGET_MISSING, cls.guideline_id or "(unnamed)",
                          f"the classification names guideline {cls.guideline_id!r}, which is "
                          f"not loaded"
                          + (f"; loaded: {', '.join(x.guideline_id for x in guidelines)}"
                             if guidelines else "; no guideline files were loaded"),
                          "pass --guidelines pointing at the directory that holds it"),))
        d = route_guideline(g, catalog, recommendation_ids=cls.recommendation_ids,
                            question=question, skills_dir=skills_dir)
        return replace(d, classified_as=cls.outcome, classifier=cls.classifier,
                       rationale=cls.rationale or d.rationale, model_calls=cls.model_calls)

    # ------------------------------------------------------------------ COMPOSITION
    if cls.outcome == COMPOSITION:
        vs: dict[str, tuple[str, ...]] = {}
        uvc: dict[str, tuple[str, ...]] = {}
        for g in guidelines:
            vs.update(g.value_sets)
            uvc.update(g.unknown_value_codes)
        pred, pgaps = check_predicate(cls.predicate, catalog, value_sets=vs,
                                      unknown_value_codes=uvc,
                                      missing_inputs=cls.missing_inputs)
        gaps += pgaps
        resolved: list[RoutedInput] = []
        for n in pred.variables:
            ri, g2 = _route_one_name(n, catalog)
            # A variable already reported as unknown by the term check does not need a second
            # gap saying the same thing in different words.
            resolved.append(ri)
            gaps += [x for x in g2 if x.kind != UNKNOWN_VARIABLE]
        for n in cls.missing_inputs:
            resolved.append(RoutedInput(n, NEW_VARIABLE, note="named by the classifier as an "
                                                              "input the vocabulary lacks"))
        if not cls.predicate:
            gaps.append(Gap(MALFORMED_TERM, question,
                            "the classification says COMPOSITION but emitted no predicate "
                            "terms, so there is nothing to check or run",
                            "re-run the classification; a composition without an expression is "
                            "the outcome that lets judgement hide"))
        return RoutingDecision(
            question, COMPOSITION, cls.outcome, cls.classifier, cls.rationale,
            tuple(resolved), pred, None, "", (), tuple(gaps) + tuple(_skill_gaps(gaps, skills_dir)),
            cls.model_calls)

    # ------------------------------------------------------------------ NEW_VARIABLE
    if cls.outcome == NEW_VARIABLE:
        pv = cls.proposed_variable or {}
        name = str(pv.get("name") or (cls.variables[0] if cls.variables else "")).strip()
        if not name:
            name = normalise_name(question)[:60] or "unnamed_variable"
        hit = _field_of(catalog, name)
        if hit:
            # The classifier proposed a spec for something already shipped. Do not author it.
            ri, g2 = _route_one_name(name, catalog)
            gaps += g2
            gaps.append(Gap(
                SPEC_AUTHORING_REQUIRED, name,
                f"the classification asked for a new spec, but {hit[0]} already declares a "
                f"field named {name!r} — writing a second one is how the same variable gets "
                f"two disagreeing answers for one patient",
                f"route to {hit[0]} instead, or rename the proposed variable if it really is "
                f"a different thing",
                context=(hit[0],)))
            return RoutingDecision(question, ri.outcome, cls.outcome, cls.classifier,
                                   cls.rationale, (ri,), None, None, "", (),
                                   tuple(gaps), cls.model_calls)
        sk = spec_skeleton(name, str(pv.get("question") or question),
                           why_not_composable=str(pv.get("why_not_composable") or ""))
        gaps.append(Gap(
            SPEC_AUTHORING_REQUIRED, name,
            f"no shipped spec answers this, and it is not a predicate over ones that do; "
            f"{len(sk.open_questions)} questions must be answered by a human before the "
            f"skeleton is a spec",
            f"take the skeleton to {SPEC_AUTHORING_SKILL} and answer the open questions; the "
            f"draft deliberately does not load until they are"))
        return RoutingDecision(
            question, NEW_VARIABLE, cls.outcome, cls.classifier, cls.rationale,
            (RoutedInput(name, NEW_VARIABLE, note="skeleton emitted"),), None, sk, "", (),
            tuple(gaps) + tuple(_skill_gaps(gaps, skills_dir)), cls.model_calls)

    # ------------------------------------------- EXISTING_VARIABLE and WRONG_DATA_SOURCE
    names = list(cls.variables) or ([question.strip()] if cls.outcome == EXISTING_VARIABLE
                                    else [])
    resolved = []
    for n in names:
        ri, g2 = _route_one_name(n, catalog)
        resolved.append(ri)
        gaps += g2
    for n in cls.missing_inputs:
        resolved.append(RoutedInput(n, NEW_VARIABLE, note="not in the vocabulary"))
        gaps.append(Gap(SPEC_AUTHORING_REQUIRED, n,
                        f"the classification needs {n!r}, which no shipped spec produces",
                        f"author a spec for it ({SPEC_AUTHORING_SKILL})"))

    if not resolved:
        # WRONG_DATA_SOURCE with nothing to hang it on. STORE.610 is the worked example of
        # why this matters: the refusal is only enforceable when a spec carries it, because
        # `graph.py` reads `data_source`, not a report.
        gaps.append(Gap(
            NO_SPEC_DECLARES_THE_REFUSAL, question,
            "the classification says this is not in the notes, but names no variable, so "
            "nothing in the tree records the refusal and the next person asks again",
            f"author a refusing spec on the STORE.610 pattern ({SPEC_AUTHORING_SKILL})"))
        return RoutingDecision(question, cls.outcome, cls.outcome, cls.classifier,
                               cls.rationale, (), None, None, "", (), tuple(gaps),
                               cls.model_calls)

    # The outcome is the code's: if any resolved variable is outside_notes, that is what the
    # route is, whatever the classifier said.
    outcome = (WRONG_DATA_SOURCE if any(r.outcome == WRONG_DATA_SOURCE for r in resolved)
               else NEW_VARIABLE if all(r.outcome == NEW_VARIABLE for r in resolved)
               else EXISTING_VARIABLE)
    return RoutingDecision(question, outcome, cls.outcome, cls.classifier, cls.rationale,
                           tuple(resolved), None, None, "", (),
                           tuple(gaps) + tuple(_skill_gaps(gaps, skills_dir)),
                           cls.model_calls)


def _skill_gaps(gaps: Sequence[Gap], skills_dir: str | Path) -> list[Gap]:
    """One extra gap when a route points at a skill that is not in the tree.

    Every NEW_VARIABLE outcome tells a human to go to `skills/spec-authoring`. If that
    directory does not exist the instruction is decoration, and decoration that looks like a
    process is this repo's most-repeated failure: a constraint written down, wired to nothing,
    and believed. Checked once per decision, not once per gap.
    """
    if not any(SPEC_AUTHORING_SKILL in g.remedy or SPEC_AUTHORING_SKILL in g.detail
               for g in gaps):
        return []
    root = Path(skills_dir)
    target = root / Path(SPEC_AUTHORING_SKILL).name / "SKILL.md"
    if target.is_file():
        return []
    return [Gap(ROUTE_TARGET_MISSING, SPEC_AUTHORING_SKILL,
                f"this decision routes a human to {SPEC_AUTHORING_SKILL}, and {target} does "
                f"not exist — the route is a sentence, not a destination",
                f"create {target}, or change the route to name whoever actually authors specs")]


def load_guidelines(directory: str | Path) -> list[Guideline]:
    """Every `*.yaml` in a directory, validated. A directory that is not there is not an error.

    Guidelines are optional to intake: routing a variable name needs none, and a machine that
    has not been given any should still be able to run `acr ask primary_site`. An invalid one
    IS an error — `load_guideline` raises — because a guideline that half-parses would route
    half its inputs and report the other half as absent.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    return [load_guideline(p) for p in sorted(d.glob("*.yaml"))]
