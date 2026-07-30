"""The deterministic evaluation plane: what may be judged, what fired, and what changed.

PART 1 — THE PRECEDENCE REGISTRY (`REGISTRY`, `judge_ruling`, `assert_judge_allowed`)
--------------------------------------------------------------------------------------
One row per evaluable dimension, declaring whether a DETERMINISTIC check exists. Where one
exists a model judge is FORBIDDEN — refused, not discouraged. `acr.judge` calls
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

THIS REGISTRY IS THE ONE NAMESPACE. `acr.judge` advertises dimensions; `evaluators/*.yaml`
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
import closure and fails if `acr.llm`, `acr.graph` or a provider SDK ever appears here.

It emits per-field rates AND PER-INSTANCE rows, and `compare()` returns REGRESSION on any
per-instance or per-subgroup drop even when every headline rate rose — a change that lifts
the mean while destroying one subgroup is precisely what a mean is built to hide.

Missing is never zero. Unknown tokens, unpriced runs and unkeyed instances are reported as
None beside their own counters; folding them into a mean moves the number in whatever
direction the missingness happens to lean.
"""
from __future__ import annotations

import hashlib
import hmac
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
              "acr.tools.toolbox.Toolbox._t_record_evidence", "faithfulness judge",
              "enforced at the tool boundary, before an estimator could be wrong"),
    Dimension("correctness", True, "exact match of the coded value against the key value",
              "acr.evals.score", "answer-correctness judge",
              "a judge calling C341 'basically right' against a key of C349 has absorbed "
              "the entire error being measured"),
    Dimension("task_completion", True, "the gate verdict recorded on the run",
              "acr.answer_gate.check_gate", "task-completion / goal-achievement judge",
              "THE LOAD-BEARING ROW: a completion judge scores a correct abstention as a "
              "failure, and correct abstentions are not missing at random"),
    Dimension("answer_format_validity", True, "the spec's declared per-field `format` regex "
              "and `allowable_values`", "acr.answer_checks.check_field_formats",
              "schema-adherence judge",
              "primary_site='C3412' shipped gate-validated against a declared C\\d{3}"),
    Dimension("rule_compliance", True, "the spec's declared answer_checks",
              "acr.answer_checks.check_answer", "instruction-following judge",
              "these rules were in the prompt when the model broke them; a second model "
              "reading them back is not the fix"),
    Dimension("abstention_correctness", True, "abstained-vs-key cross tabulation; a null "
              "key value means abstention is right", "acr.evals.score", "hedging judge",
              "its own dimension so a correct abstention is never filed as a failure"),
    Dimension("patient_isolation", True, "every patient identifier in the run equals the "
              "one asked about", "acr.evals.detect_patient_crossover", None,
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
              "reads", "acr.coverage.admissibility_for_citations", "groundedness judge",
              "'Radiology can localise a mass; it cannot establish histology' is a rule the "
              "gate already applied; asking a model to reapply it is a downgrade"),
    Dimension("step_efficiency", False, "SPLIT — name a sub-question", None, None,
              "turns, documents and dollars are counters; whether the spend was reasonable "
              "GIVEN WHAT THE RUN KNEW AT THE TIME is not a counter",
              sub_questions=("step_efficiency.deterministic", "step_efficiency.judged")),
    Dimension("step_efficiency.deterministic", True,
              "recorded counters against a declared band", "acr.evals.detect_resource_band",
              "efficiency judge",
              "arithmetic — and an unmeasured counter reports as unmeasured, not as cheap"),
    # Judge PERMITTED below. No exact equivalent exists, and inventing a fake one would be
    # the mirror-image error: a bogus exact check is worse than an honest estimate.
    #
    # THE THREE `acr.judge` ADVERTISES ARE HERE BY NAME. They were not, and every one of them
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
              "requires reading the rule; acr.refine treats the verdict as a proposal"),
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
    """THE query `acr.judge` consults. A ruling with its reason, never a bare bool.

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

    THE SEAM CHECK. `acr.judge` advertised three dimensions and this registry knew none of
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
    """The registry in the query shape `acr.judge` requires: dimension -> verifier | None.

    `acr.judge` takes this as a parameter rather than importing it, so this class is the one
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
        return "acr.evals.PrecedenceGate()"


def precedence_gate() -> PrecedenceGate:
    return PrecedenceGate()


# ================================================== PART 2: abnormal-behaviour detectors

IRB, CRITICAL, WARN = "IRB", "CRITICAL", "WARN"
_SEVERITY_ORDER = {IRB: 0, CRITICAL: 1, WARN: 2}

#: The real corpus's person_id shape. Findings land in eval reports that live in the tree,
#: so identifiers are masked on the way out — see tests/test_no_phi_in_tree.py.
_PERSON_ID = re.compile(r"1168\d{12}")

READ_TOOLS = {"read_document", "read_documents_batch"}
SEARCH_TOOLS = {"search_notes", "search_documents", "search"}
#: Terms that match everything. A search that cannot fail is not evidence that you looked.
UNIVERSAL_TERMS = {".", ".*", ".+", ".?", "*", "%", "^", "$", r"\w", r"\w*", r"\w+", r"\s*"}
ABSTAIN_STATUSES = {"EVIDENCE_INSUFFICIENT", "NO_ANSWER", "SPEC_INSUFFICIENT", "ABSTAIN"}


#: Environment variable holding the pseudonymisation key. Kept out of the tree, beside the
#: provider credentials.
PSEUDONYM_KEY_ENV = "ACR_PSEUDONYM_KEY"


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
    does not match `_PERSON_ID` and never collides.

    With a key set, each id becomes its own token, stable across processes so two baselines
    can be joined, and not invertible without the key. Without one the old behaviour stands,
    because a pseudonym that is merely a hash of a 12-digit number behind a known prefix is
    a lookup table, not a protection. `compare` refuses rather than guess when it sees the
    collision, so the unkeyed path is slow, never wrong.
    """
    key = os.environ.get(PSEUDONYM_KEY_ENV)
    if not key:
        return json.loads(_PERSON_ID.sub("<person_id:redacted>", json.dumps(obj, default=str)))

    def tok(m: "re.Match[str]") -> str:
        digest = hmac.new(key.encode(), m.group(0).encode(), hashlib.sha256).hexdigest()[:12]
        return f"<person:{digest}>"

    return json.loads(_PERSON_ID.sub(tok, json.dumps(obj, default=str)))


def _num(v: Any, cast):
    # bool is an int in Python, and a True token count is a bug better surfaced as absent.
    return cast(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


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
    answer = property(lambda s: s.manifest.get("answer") or {})
    status = property(lambda s: str(s.answer.get("status") or ""))
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
        _num((s.manifest.get("coverage_attested") or {}).get("n_read"), int))
    searched_terms = property(lambda s: [
        str(t) for t in ((s.manifest.get("coverage_attested") or {}).get("searched_terms") or [])
    ] + [str((e.get("args") or {}).get("query")) for e in s.tool_calls(SEARCH_TOOLS)
         if (e.get("args") or {}).get("query") is not None])

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


def run_detectors(run: RunRecord, *, config: DetectorConfig,
                  expected_patient: str | None = None) -> list[Finding]:
    """All detectors, most severe first. `expected_patient` defaults to the manifest's own."""
    exp = expected_patient or run.patient_id
    out = detect_zero_document_read(run)
    out += detect_degenerate_search(run, min_term_chars=config.min_term_chars)
    out += detect_patient_crossover(run, expected_patient=exp) if exp else []
    out += detect_rejection_loop(run, max_repeats=config.max_rejection_repeats)
    out += detect_resource_band(run, token_band=config.token_band, turn_band=config.turn_band)
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


@dataclass(frozen=True)
class BaselineKey:
    """What a baseline is comparable ACROSS. Every part matters, so every part is stored."""
    commit: str
    spec_hash: str
    model: str
    date: str

    def as_str(self) -> str:
        return f"{self.commit}|{self.spec_hash}|{self.model}|{self.date}"

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_value(v: Any) -> str | None:
    return (str(v).strip() or None) if v is not None else None


def _rate(num: int, den: int) -> float | None:
    # None, not 0.0: an empty denominator is "not measured", and 0.0 reads as "measured and
    # terrible". Those two lead to opposite actions.
    return round(num / den, 4) if den else None


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

    def to_dict(self) -> dict:
        return {"baseline_key": self.key.to_dict(), "baseline_key_str": self.key.as_str(),
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
          detector_config: DetectorConfig | None = None) -> ScoreReport:
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
        row = answer_key.get(iid) or answer_key.get(run.patient_id)
        n_unkeyed += int(row is None)
        kf = (row or {}).get("fields") or {}
        abstained = run.status in ABSTAIN_STATUSES
        outcomes = []
        for f in fields:
            coded = None if abstained else _norm_value(run.value.get(f))
            kv = _norm_value(kf.get(f))
            o = _outcome_for(coded, kv, row is not None and f in kf)
            outcomes.append(FieldOutcome(f, coded, kv, o))
            by_field[f].append(o)
        findings = run_detectors(run, config=detector_config) if detector_config else []
        instances.append(InstanceResult(
            str((row or {}).get("instance_id") or iid), run.patient_id, run.spec_id,
            run.spec_hash, run.status, run.gate_validated, outcomes, run.turns,
            run.total_tokens, run.cost_usd,
            [str(g) for g in ((row or {}).get("subgroups") or [])],
            [fd.to_dict() for fd in findings]))

    n_gate = sum(1 for r in instances if r.gate_validated)
    costs, toks, turns = ([r.cost_usd for r in instances if r.cost_usd is not None],
                          [float(r.total_tokens) for r in instances if r.total_tokens is not None],
                          [float(r.turns) for r in instances if r.turns is not None])
    sev = dict(Counter(fd["severity"] for r in instances for fd in r.findings))
    totals = {"n_instances": len(instances), "n_unkeyed": n_unkeyed,
              "turns_mean": _mean(turns), "tokens_mean": _mean(toks),
              "cost_usd_total": round(sum(costs), 4) if costs else None,
              "n_tokens_unknown": len(instances) - len(toks),
              "n_cost_unknown": len(instances) - len(costs),
              "n_findings": sum(len(r.findings) for r in instances),
              "findings_by_severity": sev, "by_subgroup": _subgroup_rates(instances)}
    return ScoreReport(key, {f: _field_row(f, o, n_gate) for f, o in by_field.items()},
                       instances, totals)


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
    diffs = [f"{k}: {bk.get(k)!r} -> {ak.get(k)!r}"
             for k in ("commit", "spec_hash", "model", "date") if bk.get(k) != ak.get(k)]
    bf, af = before.get("per_field") or {}, after.get("per_field") or {}
    per_field = {f: _pair((bf.get(f) or {}).get("exact_match_rate"),
                          (af.get(f) or {}).get("exact_match_rate"))
                 for f in sorted(set(bf) | set(af))}

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
