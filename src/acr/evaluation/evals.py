"""The deterministic evaluation plane: what may be judged, what fired, and what changed.

PART 1 — THE PRECEDENCE REGISTRY (`REGISTRY`, `judge_ruling`, `assert_judge_allowed`)
--------------------------------------------------------------------------------------
One row per evaluable dimension, declaring whether a DETERMINISTIC check exists. Where one
exists a model judge is FORBIDDEN — refused, not discouraged. `acr.evaluation.judge` calls
`assert_judge_allowed()` and gets a `JudgeForbidden` carrying the reason.

Three of the standard LLM-judge metrics have strictly stronger exact equivalents here:

    hallucination     -> the quote is sliced out of the document by offset
    correctness       -> does the code equal the answer-key code?        `==`
    task completion   -> the coverage gate verdict, already computed and stored

Approximating a quantity you can compute exactly is a downgrade on every axis: slower,
costlier, irreproducible, and it disagrees with itself run to run. But the reason it is
forbidden rather than merely discouraged is narrower and worse:

    A task-completion judge LAUNDERS ABSTENTION. On a chart with no admissible evidence the
    correct answer is EVIDENCE_INSUFFICIENT, and a judge asked "did the agent complete the
    task?" scores that correct answer as a failure. Optimising against that number teaches
    the agent to guess — and it teaches it on a subpopulation that is not missing at random.
    `special_codes_not_mar` in the site/histology spec says so outright: the absent values
    cluster on outside-hospital and declined biopsies. The metric's error is concentrated
    exactly where the clinical stakes are, and no aggregate will show it.

THE FENCE IS PER SUB-QUESTION, NOT PER DIMENSION. `evidence_support` and `step_efficiency`
each have a half code decides exactly and a half only a reader can answer, so each is a
split parent naming two rows. A per-dimension fence has to pick one answer for the whole
dimension and is wrong either way: it forbids "does this quote actually support this value",
or it lets a model re-decide admissibility the gate already computed.

THIS REGISTRY IS THE ONE NAMESPACE. `acr.evaluation.judge` advertises dimensions; `assets/evaluators/*.yaml`
declare one each; both are checked against these rows. They were not, and the two sides had
zero names in common (see `unknown_dimensions`, and the seam test in tests/test_judge.py).

The registry is DATA in one auditable place rather than policy scattered through
if-statements, because a rule spread over five call sites has five places to forget it. An
unregistered dimension raises `UnknownDimension`: the default is "nobody has decided", never
"a judge may proceed". `_validate_registry()` runs at import and refuses a row that claims
determinism without naming the code that decides it.

PART 2 — RUNTIME ABNORMAL-BEHAVIOUR DETECTORS (`run_detectors`)
---------------------------------------------------------------
Each was first found by hand-auditing traces after the fact. As detectors they are
deterministic, free, and fire while the run is still cheap to stop:

  * `zero_document_read` — submitted an answer having read no documents
  * `degenerate_search` — a term that cannot fail; a keyword matcher where searching "t"
    discharged every required keyword has already shipped in this repo
  * `patient_crossover` — touched a patient other than the one asked about: an IRB
    incident, not a bug report
  * `rejection_loop` — the same answer_check firing again on a byte-identical answer. One
    run was rejected twice and then burned a 400k-token budget without revising.
  * `resource_out_of_band` / `resource_unmeasured` — tokens or turns against a declared band

Each returns a `Finding` carrying its evidence, never a bare boolean: "abnormal=True" cannot
be triaged, argued with, or re-checked six weeks later.

PART 3 — THE REGRESSION HARNESS (`score`, `save_baseline`, `compare`)
----------------------------------------------------------------------
672 tests and not one calls a model, which is right — but it means reconstructing a run's
numbers is a grep across manifests, and that is how "SPEC_INSUFFICIENT fired 0 times" got
explained backwards as model behaviour when the channel was in fact crashing. A zero from a
broken counter and a zero from a well-behaved agent are byte-identical until something
scores them side by side.

`score()` READS manifests and traces and runs nothing: no model, no corpus, no tool.
`tests/test_evals.py::test_no_model_is_reachable_from_this_module` walks the first-party
import closure and fails if `acr.core.llm`, `acr.graph` or a provider SDK ever appears here.

It emits per-field rates AND PER-INSTANCE rows, and `compare()` returns REGRESSION on any
per-instance or per-subgroup drop even when every headline rate rose — a change that lifts
the mean while destroying one subgroup is precisely what a mean is built to hide.

Missing is never zero. Unknown tokens, unpriced runs and unkeyed instances are reported as
None beside their own counters; folding them into a mean moves the number in whatever
direction the missingness happens to lean.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

from ..core import site as _site

# ============================================================ PART 1: precedence registry


class UnknownDimension(KeyError):
    """Asked about an unregistered dimension. Fail closed: no judge may proceed."""


class JudgeForbidden(RuntimeError):
    """A model judge was requested where an exact deterministic check already exists."""


class DimensionIsSplit(UnknownDimension):
    """Asked about a parent dimension that has a deterministic half AND a judged half.

    Refused rather than answered, because either answer is wrong: "deterministic" kills the
    judged half, "judged" re-answers the half that has an exact answer. The caller names a
    sub-question. Subclasses `UnknownDimension` so every fail-closed caller stays closed.
    """


@dataclass(frozen=True)
class Dimension:
    """One registry row: name, deterministic?, method, verifier, judge metric replaced, why.

    `verifier` names the code that already decides this, so a reviewer can go read it. A row
    claiming determinism with no verifier is a promise rather than a check.

    `sub_questions` makes THE FENCE PER SUB-QUESTION rather than per dimension. Evidence
    support and step efficiency each have a half code decides exactly and a half only a
    reader can answer; a per-dimension fence has to pick one and is wrong either way — it
    either forbids the legitimate judged half or licenses a model to re-answer arithmetic.
    A row with sub_questions is a PARENT: it names its halves and is not itself evaluable.
    """
    name: str
    deterministic: bool
    method: str
    verifier: str | None
    replaces_judge_metric: str | None
    why: str
    sub_questions: tuple[str, ...] = ()


#: THE REGISTRY. Data, in one place, reviewable in one screen; field order as declared above.
#: The long-form argument for the first three rows is in this module's docstring.
REGISTRY: dict[str, Dimension] = {d.name: d for d in (
    Dimension("hallucination", True, "exact substring: the quote is sliced out of the "
              "document by offset, so it cannot be model-authored",
              "acr.review.tools.toolbox.Toolbox._t_record_evidence", "faithfulness judge",
              "enforced at the tool boundary, before an estimator could be wrong"),
    Dimension("correctness", True, "exact match of the coded value against the key value",
              "acr.evaluation.evals.score", "answer-correctness judge",
              "a judge calling C341 'basically right' against a key of C349 has absorbed "
              "the entire error being measured"),
    Dimension("task_completion", True, "the gate verdict recorded on the run",
              "acr.review.answer_gate.check_gate", "task-completion / goal-achievement judge",
              "THE LOAD-BEARING ROW: a completion judge scores a correct abstention as a "
              "failure, and correct abstentions are not missing at random"),
    Dimension("answer_format_validity", True, "the spec's declared per-field `format` regex "
              "and `allowable_values`", "acr.contract.answer_checks.check_field_formats",
              "schema-adherence judge",
              "primary_site='C3412' shipped gate-validated against a declared C\\d{3}"),
    Dimension("rule_compliance", True, "the spec's declared answer_checks",
              "acr.contract.answer_checks.check_answer", "instruction-following judge",
              "these rules were in the prompt when the model broke them; a second model "
              "reading them back is not the fix"),
    Dimension("abstention_correctness", True, "abstained-vs-key cross tabulation; a null "
              "key value means abstention is right", "acr.evaluation.evals.score", "hedging judge",
              "its own dimension so a correct abstention is never filed as a failure"),
    Dimension("patient_isolation", True, "every patient identifier in the run equals the "
              "one asked about", "acr.evaluation.evals.detect_patient_crossover", None,
              "an IRB incident decided by string equality"),
    # SPLIT dimensions. The parent is not evaluable by anything; each half is registered
    # separately so the fence can refuse one and permit the other in the same breath.
    Dimension("evidence_support", False, "SPLIT — name a sub-question", None, None,
              "may this document class establish this field is decidable from the spec; "
              "does this quote actually support this value is not, and one fence over both "
              "either forbids the second or licenses a model to re-decide the first",
              sub_questions=("evidence_support.deterministic", "evidence_support.judged")),
    Dimension("evidence_support.deterministic", True,
              "the stratum's `establishes` rule — the spec's evidence_rules in the form code "
              "reads", "acr.review.coverage.admissibility_for_citations", "groundedness judge",
              "'Radiology can localise a mass; it cannot establish histology' is a rule the "
              "gate already applied; asking a model to reapply it is a downgrade"),
    Dimension("step_efficiency", False, "SPLIT — name a sub-question", None, None,
              "turns, documents and dollars are counters; whether the spend was reasonable "
              "GIVEN WHAT THE RUN KNEW AT THE TIME is not a counter",
              sub_questions=("step_efficiency.deterministic", "step_efficiency.judged")),
    Dimension("step_efficiency.deterministic", True,
              "recorded counters against a declared band", "acr.evaluation.evals.detect_resource_band",
              "efficiency judge",
              "arithmetic — and an unmeasured counter reports as unmeasured, not as cheap"),
    # Judge PERMITTED below. No exact equivalent exists, and inventing a fake one would be
    # the mirror-image error: a bogus exact check is worse than an honest estimate.
    #
    # THE THREE `acr.evaluation.judge` ADVERTISES ARE HERE BY NAME. They were not, and every one of them
    # raised UnknownDimension against this registry: the judge could not run on anything it
    # claimed to support, and it failed closed, so nothing complained. Two agents each did
    # their half correctly and no test crossed the seam. `tests/test_judge.py::
    # test_every_dimension_the_judge_advertises_is_known_to_the_registry` crosses it now.
    Dimension("l5_explanation_quality", False,
              "judge permitted; a SCREEN in front of the human adjudicator, never instead",
              None, None,
              "L5 asks WHY a case is non-concordant and there is no label for why, by "
              "construction; validating that layer needs a human on a stratified sample"),
    Dimension("trajectory_quality", False,
              "judge permitted; ADVISORY, never a headline rate", None, None,
              "was the read order sensible, was the addendum chased, was the effort "
              "proportionate — no answer key exists, and the alternative to a judge here is "
              "nobody looking at trajectories at all"),
    Dimension("bad_case_triage", False, "judge permitted; ordering only, never a verdict",
              None, None,
              "the pool of known-bad cases is larger than the human hours; ordering it is a "
              "preference, and getting the order wrong costs reading time and nothing else"),
    Dimension("evidence_support.judged", False,
              "judge permitted; the half admissibility cannot reach", None, None,
              "'is this admissible quote the RIGHT quote for this value, and does anything "
              "in what was never opened contradict it' is not decidable from the record; "
              "kept as a sub-question so nobody reaches for a grounding judge by renaming "
              "grounding, and so the admissible-document-class half stays code's"),
    Dimension("step_efficiency.judged", False,
              "judge permitted; ADVISORY, paired with the counters, never averaged into them",
              None, None,
              "a 40-turn run on a chart that turned out to be hard is not the same failure "
              "as a 40-turn run on a chart whose answer was on page one, and the band "
              "cannot tell them apart"),
    Dimension("rationale_quality", False, "judge permitted; ADVISORY, never a headline rate",
              None, None, "whether the prose would help a human abstractor redo the work is "
              "a matter of reading, and it makes no claim about the patient"),
    Dimension("spec_ambiguity_triage", False, "judge permitted; routes spec-vs-behaviour",
              None, None, "whether a miscode came from an ambiguous rule or an ignored one "
              "requires reading the rule; acr.improvement.refine treats the verdict as a proposal"),
)}

#: THE RECONCILIATION TABLE. Two names for one question is how the two halves of this plane
#: drifted apart in the first place, so a superseded name resolves here instead of surviving
#: as a second spelling. Kept rather than deleted: a name that once appeared in a report has
#: to keep resolving, or last month's numbers stop being readable.
ALIASES: dict[str, str] = {
    # "is this real quote the RIGHT quote" is verbatim the judged half of evidence support.
    "evidence_relevance": "evidence_support.judged",
    # counters against a band is verbatim the deterministic half of step efficiency.
    "cost_and_turns": "step_efficiency.deterministic",
    # AgentLoop's preset name for what this repo calls trajectory quality.
    "tool_selection_rationality": "trajectory_quality",
}


def _validate_registry() -> None:
    # A row claiming determinism without naming the deciding code is how this registry rots
    # into decoration. Asserted at import so it cannot rot quietly.
    for d in REGISTRY.values():
        if d.deterministic != bool(d.verifier):
            raise ValueError(f"registry row {d.name!r}: deterministic={d.deterministic} but "
                             f"verifier={d.verifier!r}; each implies the other")
        if d.deterministic and d.sub_questions:
            raise ValueError(f"registry row {d.name!r}: a split parent cannot itself be "
                             f"deterministic; the deterministic half is a sub-question")
        for sq in d.sub_questions:
            # A parent naming a half that does not exist is a fence with a hole in it: the
            # refusal message would send a reader to a dimension nobody registered.
            if sq not in REGISTRY:
                raise ValueError(f"registry row {d.name!r} names sub-question {sq!r}, which "
                                 f"is not registered")
            if not sq.startswith(f"{d.name}."):
                raise ValueError(f"sub-question {sq!r} must be named <parent>.<half>")
        # Every dotted row must be claimed by a parent. Otherwise `evidence_support.judged`
        # could exist while `evidence_support` still answered "judge permitted" for the whole
        # dimension, which is the per-dimension fence coming back in through the side door.
        if "." in d.name and d.name not in {s for p in REGISTRY.values()
                                            for s in p.sub_questions}:
            raise ValueError(f"registry row {d.name!r} looks like a sub-question but no "
                             f"parent row declares it")
    for old, new in ALIASES.items():
        if old in REGISTRY:
            raise ValueError(f"{old!r} is both a registry row and an alias; one name, one row")
        if new not in REGISTRY:
            raise ValueError(f"alias {old!r} -> {new!r}, which is not registered")


_validate_registry()


def resolve_dimension(dimension: str) -> str:
    """Fold a superseded name onto its canonical row. Case and padding folded first.

    ` Evidence_Relevance ` must not miss a registry keyed on `evidence_relevance` — a
    precedence check bypassed by a typo is the same as no precedence check.
    """
    d = str(dimension or "").strip().lower()
    return ALIASES.get(d, d)


@dataclass(frozen=True)
class JudgeRuling:
    dimension: str
    allowed: bool
    deterministic_method: str | None
    use_instead: str | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def judge_ruling(dimension: str) -> JudgeRuling:
    """THE query `acr.evaluation.judge` consults. A ruling with its reason, never a bare bool.

    A bare False leaves the caller nothing to show the operator, who then just reruns the
    eval with the judge 'turned back on'. The reason travels with the refusal.
    """
    name = resolve_dimension(dimension)
    d = REGISTRY.get(name)
    if d is None:
        raise UnknownDimension(
            f"{dimension!r} is not in the precedence registry. Register it — with its exact "
            f"check, or with an explicit statement that none exists — before any judge runs "
            f"on it; an unregistered dimension is not a permitted one. Known: "
            f"{', '.join(sorted(REGISTRY))}")
    if d.sub_questions:
        raise DimensionIsSplit(
            f"{name!r} is split per sub-question and cannot be ruled on as a whole. Declare "
            f"one of {list(d.sub_questions)}: " + "; ".join(
                f"{s} is " + (f"decided by {REGISTRY[s].verifier}" if REGISTRY[s].deterministic
                              else "judge-permitted") for s in d.sub_questions)
            + f". {d.why}.")
    return (JudgeRuling(d.name, False, d.method, d.verifier, d.why) if d.deterministic
            else JudgeRuling(d.name, True, None, None, d.why))


def assert_judge_allowed(dimension: str) -> JudgeRuling:
    """Raise `JudgeForbidden` when an exact check exists. Call before spending a token."""
    r = judge_ruling(dimension)
    if not r.allowed:
        raise JudgeForbidden(f"a model judge is forbidden for {dimension!r}: an exact check "
                             f"exists ({r.deterministic_method}). Use {r.use_instead} — "
                             f"{r.reason}.")
    return r


def judgeable_dimensions() -> tuple[str, ...]:
    """The canonical rows a judge or an evaluator YAML may declare. Split parents excluded."""
    return tuple(n for n, d in REGISTRY.items() if not d.deterministic and not d.sub_questions)


def unknown_dimensions(names: Iterable[str]) -> dict[str, str]:
    """`{advertised name: why the registry will not accept it}` — empty when the seam holds.

    THE SEAM CHECK. `acr.evaluation.judge` advertised three dimensions and this registry knew none of
    them; every call it claimed to support raised UnknownDimension. It failed closed, so it
    was safe and useless, and no test on either side could see it because each module was
    correct alone. Anything that publishes a dimension list runs this against the registry.
    """
    out: dict[str, str] = {}
    for n in names:
        try:
            r = judge_ruling(n)
        except UnknownDimension as exc:      # covers DimensionIsSplit
            out[str(n)] = str(exc)
            continue
        if not r.allowed:
            out[str(n)] = (f"registered, but a deterministic evaluator exists "
                           f"({r.deterministic_method}); use {r.use_instead}")
    return out


class PrecedenceGate:
    """The registry in the query shape `acr.evaluation.judge` requires: dimension -> verifier | None.

    `acr.evaluation.judge` takes this as a parameter rather than importing it, so this class is the one
    object that makes the real registry answerable to the judge's protocol. Before it, the
    only thing satisfying that protocol was a test double, which is exactly why the seam
    could be wrong in production while every test on both sides passed.

    An unregistered or split dimension RAISES. The judge turns a raising registry into a
    refusal, which is the correct reading of "nobody has decided".
    """

    def deterministic_evaluator_for(self, dimension: str) -> str | None:
        r = judge_ruling(dimension)          # raises UnknownDimension / DimensionIsSplit
        return None if r.allowed else r.use_instead

    def __repr__(self) -> str:
        return "acr.evaluation.evals.PrecedenceGate()"


def precedence_gate() -> PrecedenceGate:
    return PrecedenceGate()


# ================================================== PART 2: abnormal-behaviour detectors

IRB, CRITICAL, WARN = "IRB", "CRITICAL", "WARN"
_SEVERITY_ORDER = {IRB: 0, CRITICAL: 1, WARN: 2}

#: The site's person-id shape, configured in ONE place. Findings land in eval reports that live
#: in the tree, so identifiers are masked on the way out — see tests/test_no_phi_in_tree.py.
#: `evals` is otherwise stdlib-only by design (pinned by tests/test_evals.py), so the compiled
#: pattern is imported rather than the module: one name, and no package edge worth checking.

READ_TOOLS = {"read_document", "read_documents_batch"}
SEARCH_TOOLS = {"search_notes", "search_documents", "search"}
#: Terms that match everything. A search that cannot fail is not evidence that you looked.
UNIVERSAL_TERMS = {".", ".*", ".+", ".?", "*", "%", "^", "$", r"\w", r"\w*", r"\w+", r"\s*"}
ABSTAIN_STATUSES = {"EVIDENCE_INSUFFICIENT", "NO_ANSWER", "SPEC_INSUFFICIENT", "ABSTAIN"}


def _query_terms(query: object) -> list[str]:
    """One search argument -> the terms it actually searched for.

    A list is the batched form and each element is its own term; a scalar is one term; `None` is
    no term. Stringifying a list here is the defect this function exists to prevent.
    """
    if query is None:
        return []
    if isinstance(query, (list, tuple, set)):
        # EVERY ELEMENT, filtered by nothing. The first version dropped whitespace-only terms
        # (`if str(t).strip() or t == ""`), which HID them from `detect_degenerate_search` — whose
        # first branch is `"empty" if not s`, i.e. that term is exactly what it exists to report.
        # It also made this disagree with `contract.behaviour._query_terms` on `["  "]`, in the same
        # changeset that added a test to keep the two identical.
        return [str(t) for t in query]
    return [str(query)]



#: Environment variable holding the pseudonymisation key. Kept out of the tree, beside the
#: provider credentials.
#: `core.site` owns it: `audit` needs the identical answer and may not import this plane.
PSEUDONYM_KEY_ENV = _site.PSEUDONYM_KEY_ENV


def pseudonymise(value: str) -> str:
    """One identifier's keyed pseudonym. The joinable form, shared with `acr audit run`."""
    return _site.fingerprint(value)


def pseudonym_basis() -> str:
    """`hmac` when a key is available, `constant` otherwise. Recorded in every baseline."""
    return "hmac" if os.environ.get(PSEUDONYM_KEY_ENV) else "constant"


def mask_person_ids(obj: Any) -> Any:
    """Replace any real person_id with an opaque token, structure preserved.

    TWO MASKS, AND THE DIFFERENCE IS NOT COSMETIC. The constant token protects the identifier
    and destroys DISTINCTNESS: ten patients masked this way share one `instance_id`, and
    `_outcome_index` is a dict, so nine of them vanish and the tenth answers for the batch.
    That is how a real 10-patient before/after reported `0 regressions` while two instances
    had visibly left a good outcome — and why no test caught it, since a synthetic `SYN0001`
    matches no configured pattern and never collides.

    With a key set, each id becomes its own token, stable across processes so two baselines
    can be joined, and not invertible without the key. Without one the old behaviour stands,
    because a pseudonym that is merely a hash of a 12-digit number behind a known prefix is
    a lookup table, not a protection. `compare` refuses rather than guess when it sees the
    collision, so the unkeyed path is slow, never wrong.
    """
    pattern = _site.PERSON_ID
    if pattern is None:
        # No identifier shape configured for this site, so there is nothing to mask and nothing to
        # claim. `core/site.require_person_id_pattern` refuses the case where that is unsafe.
        return obj
    key = os.environ.get(PSEUDONYM_KEY_ENV)
    if not key:
        return json.loads(pattern.sub("<person_id:redacted>", json.dumps(obj, default=str)))

    def tok(m: re.Match[str]) -> str:
        # `site.fingerprint`, so a `<person:…>` token in an eval report and a fingerprint in an
        # audit report are the same string for the same identifier.
        return f"<person:{_site.fingerprint(m.group(0))}>"

    return json.loads(pattern.sub(tok, json.dumps(obj, default=str)))


def _num(v: Any, cast):
    # bool is an int in Python, and a True token count is a bug better surfaced as absent.
    return cast(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


#: `run-YYYYMMDD-HHMMSS-hex`, the id `agent.py` mints for a single run.
_RUN_ID_DATE = re.compile(r"^run-(\d{4})(\d{2})(\d{2})-")
#: `<arm>__YYYYMMDDTHHMMSSZ__<code_sha>`, the batch directory the launcher mints.
_BATCH_DIR_DATE = re.compile(r"__(\d{4})(\d{2})(\d{2})T\d{6}Z__")


def _run_date(run_id: str, source: str = "") -> str:
    """`2026-07-27`, or "" when nothing recorded one.

    TWO SOURCES, FIRST-WINS, never combined. `run_id` is not reliably a run id: over this tree's
    509 manifests, 493 record the PATIENT id there (`SYN0007`) and only 16 carry the timestamped
    form, so the batch directory name is where the date is for almost everything already recorded.
    The runtime's own stamp outranks the path when both are present.

    Adding the two would be the `searched_terms` defect again — that read the manifest AND the trace
    and concatenated them, so every term counted twice and `detect_degenerate_search` reported 10
    findings over a tree containing 5.

    "" not today's date. Substituting the clock would stamp a two-week-old baseline with the day
    somebody happened to read it, and every downstream reader would take that as measured.
    """
    if m := _RUN_ID_DATE.match(run_id or ""):
        return "-".join(m.groups())
    if m := _BATCH_DIR_DATE.search(source or ""):
        return "-".join(m.groups())
    return ""


@dataclass
class Finding:
    """A detector's output. Never a bare boolean — `abnormal=True` cannot be triaged."""
    detector: str
    severity: str
    message: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return mask_person_ids({"detector": self.detector, "severity": self.severity,
                                "message": self.message, "evidence": self.evidence})


@dataclass
class RunRecord:
    """A manifest, plus the trace beside it if one survives.

    The trace is optional on purpose: manifests outlive traces in `runs/`, and a detector
    that needed both would go quiet rather than report. Where a detector can only decide
    from the trace it says so in its evidence (`revision_observable: false`) instead of
    handing back a clean bill it did not earn.
    """
    manifest: dict
    trace: list[dict] = field(default_factory=list)
    source: str = ""

    @classmethod
    def from_manifest(cls, path: str | Path) -> RunRecord:
        p, tr = Path(path), []
        tp = p.with_name(p.name.replace(".manifest.json", ".jsonl"))
        if tp.is_file() and tp != p:
            tr = [json.loads(ln) for ln in tp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return cls(json.loads(p.read_text(encoding="utf-8")), tr, str(p))

    # Trivial accessors, one line each. Every one yields None/"" for "not recorded", never a
    # zero: a zero is a claim about the run, and absence is not that claim.
    patient_id = property(lambda s: str(s.manifest.get("patient_id") or ""))
    spec_id = property(lambda s: str(s.manifest.get("spec_id") or ""))
    spec_hash = property(lambda s: str(s.manifest.get("spec_hash") or ""))
    #: The rest of the identity block the runtime writes, which this plane read none of until
    #: 2026-08-04. `BaselineKey` was four strings an operator typed on the command line, so the
    #: run's own statement of which arm it was had no reader at all — `experiment_config_hash`
    #: was computed for every run and consumed by nothing.
    code_sha = property(lambda s: str(s.manifest.get("code_sha") or ""))
    model = property(lambda s: str(s.manifest.get("model") or ""))
    #: WHICH ARM, as one value: a hash over spec_hash + runtime profile + prompt_assets + model +
    #: temperature + seed + call ceiling + code_sha, assembled in `agent.py` where a reader can see
    #: what counts. This is the only field that moves when two arms differ ONLY in their prompt —
    #: the case where `commit`, `spec_hash`, `model` and `date` are all identical and the arms are
    #: not. Empty for the 294 of this tree's 509 manifests written before it existed.
    experiment_config_hash = property(
        lambda s: str(s.manifest.get("experiment_config_hash") or ""))
    run_id = property(lambda s: str(s.manifest.get("run_id") or ""))
    #: The calendar day this run started, from the run id or the batch directory it sits in.
    #: Derived rather than typed: `--date` was a free-text option no artifact could contradict.
    run_date = property(lambda s: _run_date(s.run_id, s.source))
    answer = property(lambda s: s.manifest.get("answer") or {})
    status = property(lambda s: str(s.answer.get("status") or ""))
    #: The outcome KIND the contract gave this status, recorded at emission by the runtime.
    #: Empty for a manifest written before 2026-08-02, which is why `abstained` falls back to
    #: the literal set rather than treating an unrecorded kind as a value-carrying answer.
    status_kind = property(lambda s: str(s.answer.get("status_kind") or ""))
    #: Whether this run's fields may be scored as CODED. Anything that is not the
    #: value-carrying kind coded nothing, whatever it is spelled: a contract may declare more
    #: than one abstention, and a hardcoded set of names would score the ones it had not heard
    #: of as if they had answered.
    abstained = property(lambda s: (s.status_kind != "value") if s.status_kind
                         else s.status in ABSTAIN_STATUSES)
    value = property(lambda s: s.answer.get("value") or {})
    gate_validated = property(lambda s: bool(s.manifest.get("gate_validated")))
    rejections = property(lambda s: list(s.manifest.get("rejections") or []))
    evidence = property(lambda s: list(s.manifest.get("evidence")
                                       or s.answer.get("evidence") or []))
    total_tokens = property(lambda s: _num((s.manifest.get("usage") or {}).get("total_tokens"),
                                           int))
    #: Read, never derived from tokens: a price constant belongs to the caller, and a wrong
    #: one silently rescales every cost number in the report.
    #:
    #: `spend.usd` FIRST, because that is where the number actually is. This read
    #: `manifest["cost_usd"]` alone, a key no manifest this repo has ever written contains --
    #: the priced ceiling in `spend.py` writes `spend: {usd: ..., priced: true, ...}`. So every
    #: baseline reported `cost None` with `n_cost_unknown` equal to the whole cohort while each
    #: manifest carried its own price: the ten-patient real batch of 2026-07-28 summed to
    #: $3.5247 and scored as unmeasured. `spend.usd` is itself None for an unpriced model (never 0.0),
    #: so an unknown price still reads as unknown here.
    cost_usd = property(
        lambda s: _num((s.manifest.get("spend") or {}).get("usd"), float)
    )
    #: Trace-first, manifest as fallback. `None` means nobody counted, which is not zero.
    n_documents_read = property(lambda s: sum(
        len((e.get("args") or {}).get("note_ids") or []) or 1
        for e in s.tool_calls(READ_TOOLS)) if s.trace else
        _num(s._coverage().get("n_read"), int))
    #: Every term this run searched, one per element. FLATTENED, because `search_notes` is BATCHED
    #: — `toolbox.py` accepts a list and records one coverage entry per term — and 953 of the 3,988
    #: search events in this tree carry a list-valued `query`. This used to be
    #: `str(args["query"])`, so `str(["bx", "adenocarcinoma"])` was ONE opaque term:
    #: `detect_degenerate_search` reported 1 finding over 509 manifests while
    #: `coverage_state.searched_terms` showed five degenerate terms, and no test caught it because
    #: every fixture in `tests/test_evals.py` writes a scalar.
    searched_terms = property(lambda s: s._searched_terms())

    def _searched_terms(self) -> list[str]:
        """Every term this run searched, ONCE each occurrence, trace-first.

        TRACE OR MANIFEST, NOT BOTH. This concatenated the two, which was harmless only while the
        manifest half read `coverage_attested` — a key the runtime never writes — so in practice
        only the trace contributed. Fixing the fallback to `coverage_state` (502 of 509 manifests
        carry it) made both halves non-empty and every term counted TWICE: a real run yielded 12
        terms where 6 were searched, and `detect_degenerate_search` reported 10 findings over this
        tree where 5 occurrences exist. Two sources for one fact, added together.

        The trace wins because it is per-CALL: it preserves that a term was searched three times,
        which is what a rejection-loop or repeat-search reading needs. `coverage_state` is the
        run's own summary and is the only source when no trace survives.
        """
        from_trace = [str(t) for e in self.tool_calls(SEARCH_TOOLS)
                      for t in _query_terms((e.get("args") or {}).get("query"))]
        if from_trace:
            return from_trace
        return [str(t) for t in (self._coverage().get("searched_terms") or [])]

    def _coverage(self) -> dict:
        """The manifest's coverage block, under whichever name wrote it.

        `coverage_state` is what the runtime writes (502 of 509 manifests here);
        `coverage_attested` is a legacy name carried by 13 older ones and by every fixture in
        `tests/test_evals.py` — which is why the fallback read a key production never wrote and a
        trace-less manifest reported zero terms searched. Both are read; neither is guessed at.
        """
        for name in ("coverage_state", "coverage_attested"):
            block = self.manifest.get(name)
            if isinstance(block, dict) and block:
                return block
        return {}

    def tool_calls(self, names: Iterable[str] | None = None) -> list[dict]:
        want = set(names or ())
        return [ev for ev in self.trace if ev.get("kind") == "tool"
                and (not want or str(ev.get("tool") or "").split(".")[-1] in want)]

    @property
    def turns(self) -> int | None:
        u = self.manifest.get("usage") or {}
        for v in (u.get("llm_calls"), u.get("turns"), self.manifest.get("steps")):
            if _num(v, int) is not None:
                return int(v)
        return sum(1 for ev in self.trace if ev.get("kind") == "llm") if self.trace else None


def detect_zero_document_read(run: RunRecord) -> list[Finding]:
    """Submitted an answer without opening a document.

    Not hypothetical: a run can list documents, read the METADATA, and answer from note
    types alone. That answer can even be right, which is why accuracy will not catch it.
    """
    n = run.n_documents_read
    if not run.status or n is None or n > 0:
        return []
    # Quotes with zero reads is the stronger claim: they were not obtained from the record.
    return [Finding("zero_document_read", CRITICAL,
                    f"submitted status={run.status!r} having read 0 documents",
                    {"status": run.status, "value": run.value, "n_documents_read": 0,
                     "n_evidence_cited": len(run.evidence), "gate_validated": run.gate_validated,
                     "quotes_without_reads": bool(run.evidence), "source": run.source})]


def detect_degenerate_search(run: RunRecord, *, min_term_chars: int) -> list[Finding]:
    """A search term that cannot fail.

    `min_term_chars` is required and has no default. What counts as degenerate is a policy
    choice about this corpus's vocabulary, and a default would be the same class of mistake
    as the bug it catches: a keyword matcher here counted every required keyword discharged
    because the agent searched "t".
    """
    if min_term_chars is None or int(min_term_chars) < 1:
        raise ValueError(f"min_term_chars is required and must be >= 1 (no defensible "
                         f"default exists); got {min_term_chars!r}")
    out: list[Finding] = []
    for term in run.searched_terms:
        s = term.strip()
        why = ("empty" if not s else
               "matches every document" if s in UNIVERSAL_TERMS else
               "no alphanumeric character" if not any(c.isalnum() for c in s) else
               f"shorter than the declared minimum of {int(min_term_chars)} characters"
               if len(s) < int(min_term_chars) else "")
        if why:
            out.append(Finding("degenerate_search", CRITICAL,
                               f"search term {s!r} is degenerate: {why}",
                               {"term": s, "reason": why, "min_term_chars": int(min_term_chars),
                                "all_terms": run.searched_terms, "source": run.source}))
    return out


def detect_patient_crossover(run: RunRecord, *, expected_patient: str) -> list[Finding]:
    """Any patient identifier in the run that is not the one asked about.

    An IRB incident, reported as one. Identifiers are masked into the finding so that filing
    the incident does not itself write PHI into the tree.
    """
    if not expected_patient:
        raise ValueError("expected_patient is required: crossover cannot be decided against "
                         "an unspecified patient")
    mentions = [(run.manifest.get("patient_id"), "manifest.patient_id")]
    for ev in run.trace:
        src = {**(ev.get("args") if isinstance(ev.get("args"), dict) else {}), **ev}
        mentions += [(src[k], f"trace[{ev.get('seq')}].{ev.get('tool') or ev.get('kind')}.{k}")
                     for k in ("patient", "patient_id") if k in src]
    seen: dict[str, list[str]] = {}
    for pid, where in mentions:
        p = str(pid or "").strip()
        if p and p != str(expected_patient):
            seen.setdefault(p, []).append(where)
    if not seen:
        return []
    return [Finding("patient_crossover", IRB,
                    f"{len(seen)} patient identifier(s) other than the one asked about "
                    f"appear in this run",
                    {"expected_patient": expected_patient, "other_patients": sorted(seen),
                     "sites": seen, "source": run.source})]


def _rejection_signature(ev: Mapping[str, Any]) -> str:
    why = ev.get("why") or ev.get("check") or ""
    return re.sub(r"\s+", " ", " ".join(str(w) for w in why)
                  if isinstance(why, (list, tuple)) else str(why)).strip().lower()[:200]


def detect_rejection_loop(run: RunRecord, *, max_repeats: int) -> list[Finding]:
    """The same answer_check firing again against an unrevised answer.

    A rejection the agent RESPONDS to is the loop working as designed; the traces show it
    revises correctly once told what is wrong. The pathology is the repeat with no revision:
    one run was rejected twice and then spent a 400k-token budget without changing a field.

    `max_repeats` is required and has no default — how many identical rejections constitute
    a loop is a budget decision, not a fact about the code.
    """
    if max_repeats is None or int(max_repeats) < 2:
        raise ValueError(f"max_repeats is required and must be >= 2 — one rejection is not a "
                         f"loop, and no default is defensible; got {max_repeats!r}")
    traced = [ev for ev in run.trace if ev.get("kind") == "answer_rejected"]
    # Manifest-only runs do not store the attempted answer, so a repeat there is suggestive,
    # not proven. Say that in the finding rather than staying silent or over-claiming.
    observable = bool(traced)
    items = ([(_rejection_signature(e), json.dumps(e.get("attempted"), sort_keys=True,
                                                   default=str)) for e in traced]
             if traced else [(_rejection_signature(r), "") for r in run.rejections])
    out: list[Finding] = []
    # groupby, not a global count: two identical rejections either side of a real revision
    # are the loop working. Only an unbroken run of them is the pathology.
    for (sig, _), grp in groupby(items):
        n = len(list(grp))
        if n >= int(max_repeats):
            out.append(Finding(
                "rejection_loop", CRITICAL if observable else WARN,
                f"the same rejection fired {n} times in a row" + (
                    " with a byte-identical answer" if observable
                    else " (revision unverifiable: no trace)"),
                {"signature": sig, "repeats": n, "max_repeats": int(max_repeats),
                 "revision_observable": observable, "total_rejections": len(items),
                 "tokens_after_loop": run.total_tokens, "source": run.source}))
    return out


def _band(name: str, b: tuple[int, int] | None) -> tuple[int, int]:
    # An unbanded run cannot be out of band, which is not the same as being in one.
    if b is None or int(b[0]) > int(b[1]):
        raise ValueError(f"{name} is required and must be (lo, hi) with lo <= hi; got {b!r}")
    return int(b[0]), int(b[1])


def detect_resource_band(run: RunRecord, *, token_band: tuple[int, int],
                         turn_band: tuple[int, int]) -> list[Finding]:
    """Tokens and turns against declared bands. Both required, no defaults.

    A LOW value fires too: a run that used 300 tokens did not do the work, and the cheap run
    is the one an aggregate cost report congratulates you for.

    An unmeasured counter is its own finding. Reporting 0 tokens for a run whose usage was
    never recorded is how a broken channel reads as good behaviour — the same shape as
    "SPEC_INSUFFICIENT fired 0 times".
    """
    out: list[Finding] = []
    for label, val, band in (("tokens", run.total_tokens, _band("token_band", token_band)),
                             ("turns", run.turns, _band("turn_band", turn_band))):
        ev = {"metric": label, "value": val, "band": list(band), "source": run.source}
        if val is None:
            out.append(Finding("resource_unmeasured", WARN,
                               f"{label} were never recorded, so the band is unchecked", ev))
        elif val < band[0] or val > band[1]:
            ev["side"] = "below" if val < band[0] else "above"
            out.append(Finding("resource_out_of_band", WARN,
                               f"{label}={val} is {ev['side']} the declared band "
                               f"[{band[0]}, {band[1]}]", ev))
    return out


def detect_uncaused_reads(run: RunRecord) -> list[Finding]:
    """How much of this run's reading is causally unexplained in its own record.

    Not a violation — `because` is optional and a model that omits it has broken no rule.
    This counts, because the number is what tells a reader how much of an attribution report
    over this run rests on adjacency rather than on the record. A run at 0% caused reads can
    still be diagnosed; the diagnosis is just weaker, and it should say so.

    Silent on a run with no reads at all: `detect_zero_document_read` owns that case, and two
    detectors reporting one fact reads as two problems.
    """
    reads = run.tool_calls(READ_TOOLS)
    if not reads:
        return []
    uncaused = [e for e in reads if not str(e.get("because") or "").strip()]
    if not uncaused:
        return []
    # WARN, and it is the mildest tier this module has — there is no informational level, and
    # inventing one here would put a fourth string into `_SEVERITY_ORDER`'s blind spot, where
    # unknown severities sort to 9 by accident rather than by decision.
    return [Finding(
        "uncaused_read", WARN,
        f"{len(uncaused)} of {len(reads)} reads record no cause; attribution over this run "
        f"must infer their motivation from adjacency",
        {"n_reads": len(reads), "n_uncaused": len(uncaused),
         "caused_fraction": round(1 - len(uncaused) / len(reads), 3), "source": run.source},
    )]


def _evidence_of(run: RunRecord) -> list[dict]:
    """The recorded evidence set, mappings only. One accessor, so three checks cannot disagree.

    WHERE the set lives is already settled by `RunRecord.evidence` above, and this delegates to
    it rather than repeating the lookup: `agent.py` writes the ledger TWICE — top level at
    :1205 and inside the answer at :987 — while `run_manifest.build_manifest` writes only the
    answer copy, so a second accessor that happened to check one key would report "no evidence"
    for whole classes of manifest. "No evidence" is the gate's conclusion, not this audit's.

    All this adds is dropping non-mapping rows, so a hand-edited manifest costs one detector a
    skipped item instead of raising AttributeError partway through a batch.
    """
    return [e for e in run.evidence if isinstance(e, dict)]


def audit_evidence_set(run: RunRecord) -> list[Finding]:
    """Structural defects in the evidence set AS A SET, not in any one item.

    Every existing check is per-item: does this quote re-read at its offsets, is this span
    non-empty. DeepEvidence's evidence-graph audit reports set-level numbers instead — a 0.6%
    duplication rate, ≥99% relation correctness — and nothing here had ever counted the
    equivalent. Each finding carries a RATE, not a flag, because a check that can only say
    "there is duplication" cannot be compared against a baseline or tracked across arms.

    WHAT THIS DELIBERATELY DOES NOT CHECK, AND WHY — measured 2026-07-31
    --------------------------------------------------------------------
    The first version of this function grouped evidence by `supports` and reported, per group,
    an orphaned contradiction and a single-witness field. It rested on reading `supports` as a
    FIELD KEY. It is not one: `record_evidence` declares it as "which field **or assertion**
    this backs", and on twelve real runs the model wrote a sentence every time — "2023-04-12
    cytology was suspicious for adenocarcinoma but recommended tissue confirmation". The model
    was following the contract; the checks were not reading it.

    Grouped on free prose, every group holds exactly one row, so `single_witness_field` fired on
    12 of 12 runs and `orphan_contradiction` on 8 — not because anything was wrong but because
    the grouping key is unique by construction. A check that cannot come back clean measures
    nothing and trains a reader to skip its whole severity class. Both are gone rather than
    softened: with no machine-readable link from a span to a spec field, neither question is
    computable from what a manifest records, and a check that guesses at one is the mistake
    `DETERMINISTIC_RULES_REMOVED.md` already costed out once.

    Silent on an empty ledger: that is the gate's case, and it already refuses the answer.
    """
    items = _evidence_of(run)
    if not items:
        return []
    out: list[Finding] = []

    # OVERLAPPING SPANS IN ONE DOCUMENT. Grouped by note, NOT by `supports` — two char ranges
    # that overlap inside one document are the same text recorded twice whatever prose each row
    # carries, and grouping by prose made this under-fire for the same reason the deleted checks
    # over-fired. The ledger de-duplicates identical (note, start, end, supports, entity)
    # tuples; it cannot see that chars 0-40 and 10-50 are largely one sentence.
    by_note: dict[str, list[dict]] = {}
    for e in items:
        by_note.setdefault(str(e.get("note_id") or ""), []).append(e)
    pairs = 0
    for rows in by_note.values():
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if int(a.get("start", 0)) < int(b.get("end", 0)) and \
                   int(b.get("start", 0)) < int(a.get("end", 0)):
                    pairs += 1
    if pairs:
        out.append(Finding(
            "evidence_span_overlap", WARN,
            f"{pairs} overlapping span pair(s) within one document; the same text is recorded "
            f"more than once",
            {"n_evidence": len(items), "n_overlapping_pairs": pairs,
             "overlap_rate": round(pairs / len(items), 3), "source": run.source}))
    return out


#: NO DETECTOR READS `entity`, AND THAT IS A MEASURED DECISION — 2026-07-31.
#:
#: Two were written and both were removed after one twelve-run batch.
#:
#: `entity_answer_mismatch` compared each anchor against the answer's `reported_lesion` by
#: exact equality and raised CRITICAL when none matched. It raised CRITICAL on 12 of 12 runs,
#: wrongly every time: the two sides are not the same kind of string. An anchor is a label
#: ("Right upper-lobe lung adenocarcinoma"); `reported_lesion` is prose the model writes to
#: explain its choice ("…, the sole documented reportable primary. The 2023-04-12 aspirate and
#: 2023-04-27 biopsy describe one lesion."). Equality there can only fail.
#:
#: `multiple_anchored_entities` then counted DISTINCT anchor labels instead, on the theory that
#: two labels mean two things. It fired on 12 of 12 and was right about 1:
#:
#:     SYN0001  Right upper-lobe lung adenocarcinoma / …lung mass/adenocarcinoma   one lesion
#:     SYN0002  sigmoid colon carcinoma / …lesion / …mass                          one lesion
#:     SYN0003  Pancreatic head adenocarcinoma / Pancreatic head mass              one lesion
#:     SYN0005  Urinary bladder mass / …mucosal lesion / …carcinoma in situ        one lesion
#:     SYN0004  Left breast UOQ primary tumour / Pulmonary metastases              TWO, real
#:
#: Four in five are one lesion under a name that moved as the workup did — which is how a chart
#: is written: the mass becomes a carcinoma when pathology returns. So the count measures
#: PHRASING DRIFT, not entity count, and a check that cannot come back clean on a correct run
#: measures nothing while training a reader to skip its severity class.
#:
#: Separating "sigmoid colon mass" from "sigmoid colon carcinoma" requires deciding that two
#: phrasings name one thing. That is clinical judgement, and `DETERMINISTIC_RULES_REMOVED.md`
#: is this tree's record of what writing clinical judgement into Python costs.
#:
#: The FIELD stays and earns its place without a detector: SYN0004's two labels tell a reader
#: at a glance that the answer's evidence spans a primary and its metastases, and the
#: attribution agent reads the same thing. A prerequisite for any future check here is a tool
#: contract that asks for a STABLE label per thing rather than a fresh description per quote —
#: `record_evidence` does not ask for that today, and until it does, distinct-label count is
#: not a measurement of anything.
_ENTITY_HAS_NO_DETECTOR = True


@dataclass(frozen=True)
class DetectorConfig:
    """Every threshold, declared by the caller. No field has a default.

    Omitting one is a TypeError at the call site, which is the point: thresholds belong
    where a reviewer reads them, not buried here where they become folklore.
    """
    min_term_chars: int
    max_rejection_repeats: int
    token_band: tuple[int, int]
    turn_band: tuple[int, int]

    def __post_init__(self) -> None:
        for n in ("min_term_chars", "max_rejection_repeats", "token_band", "turn_band"):
            if getattr(self, n) is None:
                raise ValueError(f"DetectorConfig.{n} is required and has no default")


def detect_value_domain_violation(run: RunRecord, *, spec=None) -> list[Finding]:
    """A submitted value that is not a code in the table the run was SHOWN. Advisory.

    THE MODULE SAID THIS WAS COUNTED AND NOTHING COUNTED IT. `contract/code_tables.py`'s docstring
    names three jobs, the second being "`check_values()` returns typed problems for the evaluation
    plane to **count**" — and `check_values` had zero non-test callers. The table was loaded
    (fail-closed on a typo), rendered into the prompt, and recorded in the manifest: all INPUT-side.
    Nothing compared what came back.

    The two failures that module records as its own motivation both passed uncounted: a run coding
    morphology `7205` (not an ICD-O-3 code) and one writing "C341 is the right middle lobe" and
    coding accordingly (C341 is the UPPER lobe). `score` reports MISMATCH for those with no
    indication the value was not a code at all, and on an unkeyed variable reports nothing.

    WARN, NEVER CRITICAL, and that is not timidity. This repo removed five deterministic content
    checks after they destroyed 58 correct values against 21 helps, and demoted the coverage gate
    after ~150 rejections of which 27 refused the registry's exact tuple. A count is a finding; a
    refusal is a regression waiting to be measured.

    `spec` IS PASSED IN, not read from the manifest: the manifest records the table's identity, not
    the table, and re-deriving a spec from a manifest inside a detector is how an evaluator starts
    disagreeing with the run it is evaluating.
    """
    domain = str(getattr(spec, "value_domain", "") or "").strip() if spec is not None else ""
    if not domain or run.abstained or not run.value:
        # No declared domain, or nothing coded. Reporting "conformant" here would be a claim about
        # a table that does not exist.
        return []
    from ..contract.code_tables import CodeTableError, check_values, load_table
    try:
        table = load_table(domain)
    except CodeTableError:
        return []
    fields = {ax.field for ax in table.axes.values()}
    by_axis = {name: run.value.get(ax.field) for name, ax in table.axes.items()
               if ax.field in run.value}
    if not by_axis:
        return []
    # `by_axis` is built FROM `table.axes`, so `check_values`'s unknown-axis refusal cannot fire
    # here. An earlier version caught `CodeTableError` and emitted a `value_domain_unusable`
    # finding — a new finding kind that nothing could ever produce, which is the defect this whole
    # changeset is about. The refusal is kept as an assertion instead: if it ever raises, the
    # invariant above has broken and a silent `return []` would hide it.
    problems = check_values(by_axis, table=table)
    if not problems:
        return []
    return [Finding(
        "value_domain_violation", WARN,
        f"{len(problems)} submitted value(s) are not codes in {table.table_id}",
        {"value_domain": domain, "table_id": table.table_id,
         "problems": [pr.to_dict() for pr in problems],
         "checked_fields": sorted(fields & set(run.value)),
         "source": run.source})]


def run_detectors(run: RunRecord, *, config: DetectorConfig,
                  expected_patient: str | None = None, spec=None) -> list[Finding]:
    """All detectors, most severe first. `expected_patient` defaults to the manifest's own.

    `spec` is optional and only the value-domain detector uses it: a caller that has the contract
    gets code conformance counted, and one that does not gets everything else rather than an error.
    """
    exp = expected_patient or run.patient_id
    out = detect_zero_document_read(run)
    out += detect_degenerate_search(run, min_term_chars=config.min_term_chars)
    out += detect_patient_crossover(run, expected_patient=exp) if exp else []
    out += detect_rejection_loop(run, max_repeats=config.max_rejection_repeats)
    out += detect_resource_band(run, token_band=config.token_band, turn_band=config.turn_band)
    out += detect_uncaused_reads(run)
    out += audit_evidence_set(run)
    out += detect_value_domain_violation(run, spec=spec)
    return sorted(out, key=lambda f: _SEVERITY_ORDER.get(f.severity, 9))


# ==================================================== PART 3: the regression harness

EXACT = "EXACT"
MISMATCH = "MISMATCH"
ABSTAINED_CORRECT = "ABSTAINED_CORRECT"
ABSTAINED_MISSED = "ABSTAINED_MISSED"
ANSWERED_OVER_ABSTAIN = "ANSWERED_OVER_ABSTAIN"
NO_KEY = "NO_KEY"
#: The two outcomes worth rewarding. Losing a correct abstention counts as a regression: it
#: means the run started guessing on a chart with nothing admissible to find.
GOOD_OUTCOMES = (EXACT, ABSTAINED_CORRECT)
#: Outcomes whose denominator is "the key names a value" — the exact-match denominator.
KEYED_VALUED = (EXACT, MISMATCH, ABSTAINED_MISSED)


@dataclass
class FieldOutcome:
    field: str
    coded: str | None
    key: str | None
    outcome: str


@dataclass
class InstanceResult:
    """One run scored. The row an aggregate would have hidden."""
    instance_id: str
    patient_id: str
    spec_id: str
    spec_hash: str
    status: str
    gate_validated: bool
    outcomes: list[FieldOutcome]
    turns: int | None
    total_tokens: int | None
    cost_usd: float | None
    subgroups: list[str]
    findings: list[dict]

    def to_dict(self) -> dict:
        return mask_person_ids(asdict(self))


#: More than one value across the scored runs. A baseline whose key carries this is a MIXTURE of
#: arms, and `compare` refuses it as an endpoint — the shape that reported a two-spec-hash tree as
#: a clean +5.6 points.
MIXED = "MIXED"

#: The identity a manifest records about its own arm, and the `BaselineKey` field each one answers.
#: `date` is not here: it comes from the run id, not from a manifest field.
IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("code_sha", "commit"),
    ("spec_hash", "spec_hash"),
    ("model", "model"),
    ("experiment_config_hash", "experiment_config_hash"),
)


@dataclass(frozen=True)
class BaselineKey:
    """What a baseline is comparable ACROSS. Every part matters, so every part is stored.

    `experiment_config_hash` is the part that was missing, and its absence was not cosmetic. The
    other four are all identical between two arms that differ only in their PROMPT — a skill card,
    a retrieval prior, a runtime profile — so `compare` reported `key_differences: []` and
    `verdict: OK` for two runs whose system prompts differed by a whole card. The runtime had
    already computed the discriminating value for every run; nothing read it.

    `basis` says where the key came from. `derived` means the manifests were asked; `declared`
    means an operator typed it and the runs were not consulted, which is every baseline recorded
    before 2026-08-04.
    """
    commit: str
    spec_hash: str
    model: str
    date: str
    #: Additive with a default so every baseline already on disk still loads and every existing
    #: caller still constructs. An empty value means NOT RECORDED, which `compare` reports as a
    #: file that cannot say rather than as an arm that changed.
    experiment_config_hash: str = ""
    basis: str = "declared"

    def as_str(self) -> str:
        # The arm hash is appended only when there is one. A trailing `|` on the 500-odd baselines
        # that predate this field would assert a value they do not have, and this string is what a
        # human pastes into a note.
        base = f"{self.commit}|{self.spec_hash}|{self.model}|{self.date}"
        return f"{base}|{self.experiment_config_hash}" if self.experiment_config_hash else base

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_mixture(self) -> bool:
        """True when any part of this key describes more than one arm."""
        return bool(self.mixed_fields)

    @property
    def mixed_fields(self) -> list[str]:
        return sorted(k for k, v in self.to_dict().items() if v == MIXED)


def manifest_identity(runs: Sequence[RunRecord]) -> dict:
    """What the RUNS say about which arm they are. Data, never a refusal.

    Per field: the unanimous value or `MIXED`, the distinct values seen, and how many runs recorded
    nothing. Scoring a heterogeneous tree is legitimate — `eval score` over all of `runs/` is the
    only way to sweep the detectors across everything recorded — so heterogeneity is reported here
    and refused one step later, in `compare`, where a mixture cannot be one end of a delta.

    `n_unrecorded` is load-bearing and not a diagnostic afterthought: 294 of this tree's 509
    manifests carry no `experiment_config_hash`, and a reader who cannot see that count cannot tell
    a baseline whose arm is established from one whose arm is an inference from three coarser
    fields.
    """
    fields: dict[str, dict] = {}
    for name, _ in IDENTITY_FIELDS:
        seen = [getattr(r, name) for r in runs]
        present = sorted({v for v in seen if v})
        fields[name] = {
            "value": present[0] if len(present) == 1 else (MIXED if present else ""),
            "mixed": len(present) > 1,
            "values": present,
            "n_unrecorded": sum(1 for v in seen if not v),
        }
    dates = sorted({r.run_date for r in runs if r.run_date})
    return {"n_runs": len(runs), "fields": fields, "dates": dates,
            "n_undated": sum(1 for r in runs if not r.run_date)}


def _identity_date(identity: Mapping[str, Any]) -> str:
    """One day, a range, or "". A range is stated as a range: collapsing a nine-day batch to its
    first day is the same class of claim as naming one arm for a mixture."""
    dates = list(identity.get("dates") or [])
    if not dates:
        return ""
    return dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}"


def derive_baseline_key(runs: Sequence[RunRecord]) -> BaselineKey:
    """The key the runs themselves state. The default, so nobody has to type four strings.

    Refuses an empty run set rather than returning a key of four empty strings: two such keys
    compare equal to each other and to nothing real, so `compare` would call two unrelated empty
    baselines the same configuration.
    """
    if not runs:
        raise ValueError("no runs to derive a baseline key from — an empty key would be four "
                         "empty strings, and two of those compare equal")
    ident = manifest_identity(runs)
    got = {slot: ident["fields"][name]["value"] for name, slot in IDENTITY_FIELDS}
    return BaselineKey(date=_identity_date(ident), basis="derived", **got)


def reconcile_baseline_key(key: BaselineKey, runs: Sequence[RunRecord]) -> list[str]:
    """Every part of `key` that the manifests CONTRADICT. Empty means the claim holds.

    A refusal that can only fire on a wrong claim, which is the property the five deterministic
    content checks this repo deleted did not have — they destroyed 58 correct values against 21
    helps. Two ways this one stays incapable of refusing something correct:

      * A field NO manifest recorded contradicts nothing. Silence is not disagreement, and reading
        it as disagreement would refuse every manifest written before `code_sha` reached the
        identity block, plus every fabricated fixture in `tests/test_evals.py`.
      * A field the runs report as MIXED is not a contradiction of the operator's string. It is
        reported once, by `key_basis`, because the remedy differs: a wrong `--model` is a typo to
        fix, a mixed tree is a batch to re-run.

    What it does catch, measured: `--model TOTALLY-WRONG-MODEL` was accepted in silence and written
    into the baseline, where every downstream reader took it as the model that produced the numbers.
    """
    ident = manifest_identity(runs)
    out = []
    for name, slot in IDENTITY_FIELDS:
        claimed, f = getattr(key, slot), ident["fields"][name]
        if claimed and not f["mixed"] and f["value"] and claimed != f["value"]:
            out.append(f"{slot}: declared {claimed!r}, but the runs recorded "
                       f"{f['value']!r} ({len(runs) - f['n_unrecorded']} run(s))")
    date = _identity_date(ident)
    if key.date and date and key.date != date:
        out.append(f"date: declared {key.date!r}, but the run ids say {date!r}")
    return out


def _norm_value(v: Any) -> str | None:
    return (str(v).strip() or None) if v is not None else None


def _rate(num: int, den: int) -> float | None:
    # None, not 0.0: an empty denominator is "not measured", and 0.0 reads as "measured and
    # terrible". Those two lead to opposite actions.
    return round(num / den, 4) if den else None


def _key_row(answer_key: Mapping[str, Any], iid: str, run: RunRecord) -> dict | None:
    """This run's key row, or None when the key says nothing about THIS RUN.

    THE BARE FALLBACK WAS UNCONDITIONAL, and that made `n_unkeyed` structurally incapable of
    reporting a coverage miss. `tools/answer_key_from_corpus.py` writes bare patient ids, so the
    composite `patient__spec` lookup never hit, the bare lookup always did, and `run.spec_id` was
    compared to nothing: a run of a DIFFERENT CONTRACT scored against this key as a wrong answer.
    Measured on this tree's `runs/`, scoring a spec-390 key over all 509 manifests: 8 cross-spec
    runs scored as ABSTAINED_MISSED and dragged the published exact-match from the spec's actual
    75.8% to 74.6%, with `n_unkeyed: 0` asserting complete coverage.

    A row carrying no `spec_id` is accepted on the bare id — every answer key written before the
    producer emitted one is still readable, and refusing them would make every recorded baseline
    unscoreable. A row that DOES carry one must match.
    """
    row = answer_key.get(iid)
    if row is not None:
        return dict(row) if isinstance(row, Mapping) else None
    row = answer_key.get(run.patient_id)
    if not isinstance(row, Mapping):
        return None
    # `spec_ids` (plural) when a key covers more than one contract id for ONE variable, which is
    # what an ABLATION arm is: `STORE.400_522_523.site_histology_behavior.UNSTRATIFIED` has the same
    # correct answers and a different retrieval policy, and the corpus has no separate ground truth
    # for it. The strict single-id check made all 3 real UNSTRATIFIED manifests unkeyed and dropped
    # `exact_match_den` from 3 to 2 — the arm's own comparison silently became "nothing changed".
    declared = {str(x) for x in (row.get("spec_ids") or []) if str(x)}
    if row.get("spec_id"):
        declared.add(str(row["spec_id"]))
    if declared and run.spec_id and run.spec_id not in declared:
        return None
    return dict(row)


def _outcome_for(coded: str | None, kv: str | None, keyed: bool) -> str:
    if not keyed:
        return NO_KEY
    if kv is None:
        # The key says abstention is correct. Answering is the error here — and it is the
        # error a task-completion judge would have scored as the success.
        return ABSTAINED_CORRECT if coded is None else ANSWERED_OVER_ABSTAIN
    return ABSTAINED_MISSED if coded is None else (EXACT if coded == kv else MISMATCH)


def _field_row(f: str, outcomes: list[str], n_gate: int) -> dict:
    c, n = outcomes.count, len(outcomes)
    keyed_abstain, valued = (c(ABSTAINED_CORRECT) + c(ANSWERED_OVER_ABSTAIN),
                             sum(c(o) for o in KEYED_VALUED))
    return {"field": f, "n": n,
            "exact_match_rate": _rate(c(EXACT), valued), "exact_match_den": valued,
            "abstention_rate": _rate(c(ABSTAINED_CORRECT) + c(ABSTAINED_MISSED), n),
            "correct_abstention_rate": _rate(c(ABSTAINED_CORRECT), keyed_abstain),
            "correct_abstention_den": keyed_abstain,
            "gate_validated_rate": _rate(n_gate, n)}


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _mean(vals: list[float]) -> float | None:
    return round(statistics.fmean(vals), 2) if vals else None


@dataclass
class ScoreReport:
    key: BaselineKey
    per_field: dict[str, dict]
    per_instance: list[InstanceResult]
    totals: dict
    #: What the manifests said about their own arm, and every part of `key` they contradict. Both
    #: are computed by `score` rather than by the command, because `analyze_arms.py`,
    #: `measure_controller_value.py` and every future reader call `score` directly — a guard that
    #: lived only in `cli_eval` would be absent on all of those paths. Same reasoning as
    #: `prompt_asset_manifest` living in `run_manifest` rather than in `agent`.
    key_basis: dict = field(default_factory=dict)
    key_contradictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"baseline_key": self.key.to_dict(), "baseline_key_str": self.key.as_str(),
                "key_basis": self.key_basis, "key_contradictions": self.key_contradictions,
                # Whether the instance ids in this file can identify an instance at all. On
                # `constant` they cannot, and every per-instance consumer has to say so
                # rather than read the collapsed index.
                "pseudonym_basis": pseudonym_basis(),
                "per_field": self.per_field, "totals": self.totals,
                "per_instance": [r.to_dict() for r in self.per_instance]}

    def table(self) -> str:
        head = f"{'field':<26}{'n':>5}{'exact':>9}{'abstain':>9}{'gate':>8}"
        t = self.totals
        rows = [head, "-" * len(head)]
        rows += [f"{d['field']:<26}{d['n']:>5}{_pct(d['exact_match_rate']):>9}"
                 f"{_pct(d['abstention_rate']):>9}{_pct(d['gate_validated_rate']):>8}"
                 for d in self.per_field.values()]
        return "\n".join(rows + [
            (f"\ninstances {t['n_instances']}  turns(mean) {t['turns_mean']}  tokens(mean) "
             f"{t['tokens_mean']}  cost {t['cost_usd_total']}  findings {t['n_findings']} "
             f"{t['findings_by_severity']}"),
            (f"unmeasured: tokens {t['n_tokens_unknown']}  cost {t['n_cost_unknown']}  "
             f"unkeyed instances {t['n_unkeyed']}")])


def score(runs: Sequence[RunRecord], answer_key: Mapping[str, Mapping[str, Any]], *,
          fields: Sequence[str], key: BaselineKey,
          detector_config: DetectorConfig | None = None, spec=None) -> ScoreReport:
    """Score manifests against an answer key. Reads only; runs nothing.

    `answer_key` maps instance id (or bare patient id) -> {"fields": {name: value_or_None},
    "subgroups": [...]}. A field whose key value is None asserts that ABSTENTION IS THE
    CORRECT ANSWER — the one case an LLM completion judge gets backwards, encoded as data.
    """
    if not fields:
        raise ValueError("fields is required: scoring whatever keys the model happened to "
                         "emit makes the denominator depend on the model")
    by_field: dict[str, list[str]] = {f: [] for f in fields}
    instances, n_unkeyed = [], 0
    for run in runs:
        iid = f"{run.patient_id}__{run.spec_id}"
        row = _key_row(answer_key, iid, run)
        n_unkeyed += int(row is None)
        kf = (row or {}).get("fields") or {}
        abstained = run.abstained
        outcomes = []
        for f in fields:
            coded = None if abstained else _norm_value(run.value.get(f))
            kv = _norm_value(kf.get(f))
            o = _outcome_for(coded, kv, row is not None and f in kf)
            outcomes.append(FieldOutcome(f, coded, kv, o))
            by_field[f].append(o)
        findings = (run_detectors(run, config=detector_config, spec=spec)
                    if detector_config else [])
        instances.append(InstanceResult(
            str((row or {}).get("instance_id") or iid), run.patient_id, run.spec_id,
            run.spec_hash, run.status, run.gate_validated, outcomes, run.turns,
            run.total_tokens, run.cost_usd,
            [str(g) for g in ((row or {}).get("subgroups") or [])],
            [fd.to_dict() for fd in findings]))

    # `None` RATES READ AS "NO CHANGE" to a human and to `compare`, whose NOT_COMPARABLE guard only
    # catches colliding instance ids. A key that covers none of these runs is a wrong pairing, and
    # it must not present as a clean result with empty denominators.
    all_unkeyed = bool(instances) and n_unkeyed == len(instances)
    n_gate = sum(1 for r in instances if r.gate_validated)
    costs, toks, turns = ([r.cost_usd for r in instances if r.cost_usd is not None],
                          [float(r.total_tokens) for r in instances if r.total_tokens is not None],
                          [float(r.turns) for r in instances if r.turns is not None])
    sev = dict(Counter(fd["severity"] for r in instances for fd in r.findings))
    totals = {"n_instances": len(instances), "n_unkeyed": n_unkeyed,
              #: True when the key speaks about NONE of these runs. Every rate is then `None`, and
              #: `None` is indistinguishable from "measured and unchanged" downstream.
              "all_instances_unkeyed": all_unkeyed,
              "turns_mean": _mean(turns), "tokens_mean": _mean(toks),
              "cost_usd_total": round(sum(costs), 4) if costs else None,
              "n_tokens_unknown": len(instances) - len(toks),
              "n_cost_unknown": len(instances) - len(costs),
              "n_findings": sum(len(r.findings) for r in instances),
              "findings_by_severity": sev, "by_subgroup": _subgroup_rates(instances)}
    return ScoreReport(key, {f: _field_row(f, o, n_gate) for f, o in by_field.items()},
                       instances, totals,
                       key_basis=manifest_identity(runs),
                       key_contradictions=reconcile_baseline_key(key, runs))


def _subgroup_rates(instances: Sequence[InstanceResult]) -> dict:
    """Exact-match rate per (subgroup, field). The slice a mean is built to hide."""
    acc: dict[str, dict[str, list[int]]] = {}
    for r in instances:
        for g in r.subgroups or ["_all"]:
            for o in r.outcomes:
                if o.outcome in KEYED_VALUED:
                    acc.setdefault(g, {}).setdefault(o.field, []).append(int(o.outcome == EXACT))
    return {g: {f: {"n": len(v), "exact_match_rate": round(sum(v) / len(v), 4)}
                for f, v in fs.items()} for g, fs in acc.items()}


def save_baseline(report: ScoreReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p


def load_baseline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pair(b: float | None, a: float | None) -> dict:
    return {"before": b, "after": a,
            "delta": round(a - b, 4) if b is not None and a is not None else None}


def _outcome_index(report: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {(i.get("instance_id", ""), o.get("field", "")): o.get("outcome", "")
            for i in (report.get("per_instance") or []) for o in (i.get("outcomes") or [])}


def collided_instance_ids(report: Mapping[str, Any]) -> list[str]:
    """Instance ids that appear more than once in one baseline.

    `_outcome_index` is a dict keyed on (instance_id, field). A repeated id therefore does
    not raise and does not warn — the later row overwrites the earlier one and the batch
    silently shrinks to its last member. Ten real patients whose ids were all masked to the
    same constant produced a three-key index and a per-instance verdict of `0 regressions`
    over a comparison that contained two.
    """
    seen = Counter(str(i.get("instance_id", "")) for i in (report.get("per_instance") or []))
    return sorted(k for k, n in seen.items() if n > 1)


#: The parts of a key that describe the ARM. `basis` is excluded: how a key was obtained is not a
#: property of the configuration, and reporting "derived -> declared" as a difference would announce
#: an arm change every time a baseline recorded before 2026-08-04 is compared against a new one.
_ARM_PARTS = ("commit", "spec_hash", "model", "date", "experiment_config_hash")


def _key_differences(bk: Mapping[str, Any], ak: Mapping[str, Any]) -> list[str]:
    """Every part of the arm that moved, with "cannot say" kept distinct from "changed".

    A baseline recorded before `experiment_config_hash` existed carries no value for it. "The arm
    changed" and "one of these files cannot say" are different claims and only one of them is a
    reason to re-run, so the two are never collapsed into a single `'' -> 'a1b2'` line that reads
    like a configuration change.
    """
    out = []
    for k in _ARM_PARTS:
        b, a = bk.get(k) or "", ak.get(k) or ""
        if b == a:
            continue
        if not b or not a:
            side = "before" if not b else "after"
            out.append(f"{k}: not recorded in the {side} baseline (the other says "
                       f"{(a or b)!r}) — a file that cannot say, not an arm that changed")
        else:
            out.append(f"{k}: {b!r} -> {a!r}")
    return out


def _identity_problems(baseline: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """`(mixed fields, contradictions)` for one baseline.

    Reads `key_basis` when present and falls back to the key itself, so a baseline written by a
    caller that assembled its own report — or one recorded before `key_basis` existed and later
    hand-edited — is still checked on the evidence it does carry.
    """
    fields = (baseline.get("key_basis") or {}).get("fields") or {}
    mixed = sorted(n for n, f in fields.items() if (f or {}).get("mixed"))
    if not mixed:
        mixed = sorted(k for k, v in (baseline.get("baseline_key") or {}).items() if v == MIXED)
    return mixed, [str(c) for c in (baseline.get("key_contradictions") or [])]


def compare(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
    """Two baselines into a delta: per field AND per instance AND per subgroup.

    The verdict is REGRESSION if ANY instance left a good outcome or ANY subgroup rate fell,
    even when every headline rate rose. That is the only configuration in which this harness
    is worth having: the aggregate improvement is what gets shipped, and the subgroup
    collapse is what reaches a patient.

    A key mismatch is reported, never silently tolerated. Comparing across spec hashes is
    legitimate — it is the main reason to compare at all — but the reader has to be told,
    because "accuracy fell" and "the question changed" look identical in the numbers.
    """
    bk, ak = dict(before.get("baseline_key") or {}), dict(after.get("baseline_key") or {})
    diffs = _key_differences(bk, ak)
    bf, af = before.get("per_field") or {}, after.get("per_field") or {}
    per_field = {f: _pair((bf.get(f) or {}).get("exact_match_rate"),
                          (af.get(f) or {}).get("exact_match_rate"))
                 for f in sorted(set(bf) | set(af))}

    # A MIXTURE IS NOT AN ENDPOINT. Measured: a run tree spanning two spec hashes scored as one
    # clean baseline and this function reported +5.6 points. `tools/analyze_arms.py:191` refuses the
    # identical shape in prose ("refusing to compare: these arms ran on N different spec versions");
    # the evaluation plane averaged it, because nothing here had ever read what the manifests said
    # about their own arm.
    # Checked before the collision guard: a collision breaks only the per-instance arm, whereas a
    # mixture means neither file describes one configuration and no column is interpretable.
    b_mix, b_con = _identity_problems(before)
    a_mix, a_con = _identity_problems(after)
    if b_mix or a_mix or b_con or a_con:
        return {"before_key": bk, "after_key": ak, "key_differences": diffs,
                "per_field": per_field, "regressions": [], "improvements": [],
                "subgroup_regressions": [], "verdict": "NOT_COMPARABLE",
                "not_comparable": {
                    "reason": ("a baseline does not describe one configuration"
                               if (b_mix or a_mix) else
                               "a baseline claims an identity its own runs contradict"),
                    # Every `not_comparable` shape carries `reason`/`detail`/`why`/`remedy`, and
                    # the shape-specific fields sit beside them. The CLI printer read the collision
                    # shape's own keys, so adding this second shape raised `KeyError: n_colliding`
                    # at the moment it first fired — the consumer-with-no-producer defect, inverted.
                    "detail": (f"{', '.join(sorted(set(b_mix) | set(a_mix)))} differ within "
                               f"{' and '.join(s for s, m in (('before', b_mix), ('after', a_mix)) if m)}"
                               if (b_mix or a_mix) else "; ".join(b_con + a_con)),
                    "mixed_fields": sorted(set(b_mix) | set(a_mix)),
                    "mixed_in": ([s for s, m in (("before", b_mix), ("after", a_mix)) if m]),
                    "contradictions": b_con + a_con,
                    "why": ("every rate in a mixed baseline is an average over two or more arms, "
                            "so a delta against it prices the difference between the arms and the "
                            "difference between the mixtures as one number"),
                    "remedy": ("score each arm separately — one `eval score --runs` per arm "
                               "directory — and compare those. `key_basis.fields` names the "
                               "field that moved and the values it took.")}}

    # THE PER-INSTANCE ARM IS EITHER SOUND OR ABSENT, never approximate. A collided id means
    # the index cannot tell two patients apart, so the arm that the docstring above calls the
    # only reason to have this harness is not computable — and a `0 regressions` printed from
    # a collapsed index is worse than no number, because it reads as evidence of safety.
    collisions = sorted(set(collided_instance_ids(before)) | set(collided_instance_ids(after)))
    if collisions:
        return {"before_key": bk, "after_key": ak, "key_differences": diffs,
                "per_field": per_field, "regressions": [], "improvements": [],
                "subgroup_regressions": [], "verdict": "NOT_COMPARABLE",
                "not_comparable": {
                    "reason": "instance ids are not unique within a baseline",
                    "detail": f"{len(collisions)} colliding id(s), basis={pseudonym_basis()}",
                    "colliding_ids": collisions[:10],
                    "n_colliding": len(collisions),
                    "why": ("person ids were masked to a single constant token, so every "
                            "patient shares one instance_id and the per-instance index keeps "
                            "one row per field"),
                    "remedy": (f"set {PSEUDONYM_KEY_ENV} to a secret and re-run `eval score` "
                               "on both arms; each id then masks to its own stable token, "
                               "joinable across baselines and not invertible without the key"),
                    "pseudonym_basis": pseudonym_basis()}}

    b_out, a_out = _outcome_index(before), _outcome_index(after)
    regressions, improvements = [], []
    for k in sorted(set(b_out) | set(a_out)):
        b, a = b_out.get(k), a_out.get(k)
        row = {"instance_id": k[0], "field": k[1], "before": b, "after": a}
        if b in GOOD_OUTCOMES and a not in GOOD_OUTCOMES:
            regressions.append(row)
        elif a in GOOD_OUTCOMES and b not in GOOD_OUTCOMES:
            improvements.append(row)

    bs = (before.get("totals") or {}).get("by_subgroup") or {}
    subs = [{"subgroup": g, "field": f, "n_after": cell["n"],
             **_pair(((bs.get(g) or {}).get(f) or {}).get("exact_match_rate"),
                     cell["exact_match_rate"])}
            for g, fs in (((after.get("totals") or {}).get("by_subgroup") or {}).items())
            for f, cell in fs.items()
            if ((bs.get(g) or {}).get(f) or {}).get("exact_match_rate") is not None
            and cell["exact_match_rate"] < bs[g][f]["exact_match_rate"]]
    return {"before_key": bk, "after_key": ak, "key_differences": diffs,
            "per_field": per_field, "regressions": regressions, "improvements": improvements,
            "subgroup_regressions": subs,
            "verdict": "REGRESSION" if (regressions or subs) else "OK"}
