"""Agent-as-a-judge, fenced in. A judged number is an OPINION and never a gate.

WHY THE FENCE IS THE POINT
--------------------------
A model scoring model output is the right instrument in exactly one situation: the thing
being measured has no ground truth and no mechanical decision procedure. Everywhere else it
is strictly worse than the code that already exists, and worse in a way that does not show
up in the number — it agrees with the deterministic checker most of the time, so nobody
notices the cases where it does not, and a check that was enforced becomes a check that is
usually enforced. This repo already decided that question layer by layer: L4 concordance is
a rule engine, `answer_checks` rejects mechanically, `coverage.evaluate_gate` counts strata.
None of those may be re-decided by a judge, so `judge()` asks the precedence registry first
and REFUSES. There is no flag, no policy value and no keyword that turns the refusal off;
the refusal is not configurable because a configurable fence is a fence that will be opened
once, at 2am, by whoever is trying to make a run finish.

The refusal is also fail-closed. A registry that is absent, that raises, or that does not
implement the query shape below is not "no deterministic evaluator exists" — it is "we do
not know", and judging on "we do not know" is how a deterministic dimension gets quietly
re-scored by a model after a refactor moves the registry.

WHERE JUDGING LEGITIMATELY BELONGS — the only three dimensions this module will accept:

  l5_explanation_quality  `explain.py` says it in its own docstring: there is no label for
                          WHY a case is non-concordant, and validating that layer needs human
                          adjudication on a stratified sample. The judge is a SCREEN placed in
                          front of that human — it decides reading order, never the verdict.

  trajectory_quality      Was the read order sensible, were unsettled threads chased, was the
                          effort proportionate. No answer key exists for any of it, and the
                          alternative to a judge is not a better measurement, it is nobody
                          looking at trajectories at all.

  bad_case_triage         A pool of known-bad cases is larger than the human hours available.
                          Ordering that pool is a preference, not a fact; getting the order
                          wrong costs reading time and nothing else.

ANSWER-KEY ISOLATION IS IN THE TYPE, NOT IN A DOCSTRING
-------------------------------------------------------
Two of those dimensions are contaminated by the key. Show a trajectory judge that the final
answer was right and it stops scoring the path: every route to a correct answer reads as
sensible, every route to a wrong one reads as sloppy, and the "trajectory score" becomes a
noisy copy of accuracy — which is already measured deterministically and better. Show an L5
explanation judge the registry truth and it scores whether the explanation named the cause
the key implies, which is precisely the judgement the human is there to make.

So a blinded dimension does not take a packet that has anywhere to put a key: `BlindPacket`
has no `answer_key` field, is frozen, and `blind_packet()` refuses artifacts carrying a
key-bearing name at any depth. `judge()` refuses a `KeyedPacket` for a blinded dimension.
Triage is the exception and it is the honest one — the cases in a bad-case pool are bad
BECAUSE a deterministic evaluator disagreed with the key, so withholding the disagreement
would leave nothing to triage.

EVIDENCE CLASS TRAVELS WITH THE NUMBER
--------------------------------------
Every verdict is stamped JUDGED, every deterministic measurement is stamped DETERMINISTIC,
and `aggregate()` raises rather than average across the two. One mean over both classes is
the failure this module exists to prevent: it produces a single defensible-looking score in
which the model's opinion of an explanation is indistinguishable from a counted stratum.
`combine_explicitly()` will produce a combined number, but only when asked, only with a
weight the caller states, and it keeps both sides visible in the output.

MULTIPLE LENSES, NOT REPETITION
-------------------------------
Asking the same question three times measures the model's temperature. Each dimension is
therefore decomposed into distinct questions aimed at distinct failure modes, and the lens
questions are checked for distinctness so a copy-paste cannot turn a panel back into a poll.

THE DIMENSION NAMES ARE `acr.evals`' NAMES, AND A TEST CROSSES THE SEAM
-----------------------------------------------------------------------
The three dimensions below were once this module's private vocabulary and the precedence
registry had never heard of any of them: every dimension this module advertised raised
`UnknownDimension` against the registry, so it could not run on anything it claimed to
support. It failed closed — safe, and useless. Two agents each built their half correctly
and nothing checked the seam. `tests/test_judge.py::
test_every_dimension_the_judge_advertises_is_known_to_the_registry` fails if that reopens.

AN EVALUATOR IS A SPEC FILE, NOT CODE
--------------------------------------
A new evaluation need is met by writing a YAML in `evaluators/`, never by editing this
module — the same artifact class as `specs/` and `guidelines/`. `load_evaluator()` holds the
five load-time refusals; see the section at the bottom of this file. The one that matters
most is that an evaluator ships with a case it MUST FAIL, because an evaluator that scores
everything the same is indistinguishable from a clean system.

The model is a thin seam (`JudgeModel`); nothing here imports a provider or opens a socket.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

# ============================================================================== refusals
class JudgeRefusal(Exception):
    """Base: the judge declined. Never catch this to retry with different arguments."""


class DeterministicEvaluatorExists(JudgeRefusal):
    """The dimension is decided by code. A model opinion may not stand in for it."""


class DimensionNotJudgeable(JudgeRefusal):
    """Not one of the three dimensions where judging is legitimate."""


class RegistryUnavailable(JudgeRefusal):
    """The precedence registry could not be queried, so precedence is unknown."""


class AnswerKeyLeak(JudgeRefusal):
    """The key reached a dimension it would contaminate."""


class JudgeCannotGate(JudgeRefusal):
    """A judged verdict was used to accept, adopt, validate or gate something."""


class MixedEvidence(JudgeRefusal):
    """A judged number and a deterministic number were about to be averaged together."""


# ============================================================================ evidence class
#: Strings, not booleans. `deterministic=False` on a dict that lost the field reads as
#: "judged" when it actually means "nobody said", and unstamped must never render as either.
EV_DETERMINISTIC = "DETERMINISTIC"
EV_JUDGED = "JUDGED"

# ============================================================================== dimensions
#: Every name here is a row in `acr.evals.REGISTRY`, checked by the seam test. A dimension
#: this module advertises that the registry does not know is a judge that cannot run.
DIM_L5_EXPLANATION_QUALITY = "l5_explanation_quality"
DIM_TRAJECTORY_QUALITY = "trajectory_quality"
DIM_BAD_CASE_TRIAGE = "bad_case_triage"
#: The judged halves of two SPLIT dimensions. The fence is per sub-question: the other half
#: of each is deterministic and forbidden here, and naming the parent is refused outright.
DIM_EVIDENCE_SUPPORT_JUDGED = "evidence_support.judged"
DIM_STEP_EFFICIENCY_JUDGED = "step_efficiency.judged"

JUDGEABLE_DIMENSIONS = (DIM_L5_EXPLANATION_QUALITY, DIM_TRAJECTORY_QUALITY,
                        DIM_BAD_CASE_TRIAGE, DIM_EVIDENCE_SUPPORT_JUDGED,
                        DIM_STEP_EFFICIENCY_JUDGED)

#: Blinded because the key would answer the question the judge is being asked.
KEY_BLINDED_DIMENSIONS = (DIM_L5_EXPLANATION_QUALITY, DIM_TRAJECTORY_QUALITY,
                          DIM_EVIDENCE_SUPPORT_JUDGED, DIM_STEP_EFFICIENCY_JUDGED)
#: Triage only. The disagreement with truth IS the material being sorted.
KEY_PERMITTED_DIMENSIONS = (DIM_BAD_CASE_TRIAGE,)

#: Artifact names that carry the key. Checked recursively, because the leak that happens in
#: practice is a nested `{"case": {"truth": ...}}` packet assembled three call frames away.
KEY_BEARING_NAMES = ("answer_key", "answer_keys", "gold", "gold_standard", "ground_truth",
                     "truth", "truth_value", "label", "labels", "registry_truth",
                     "expected", "expected_value", "correct_value", "key")

#: A rendering budget for the prompt, NOT a decision threshold. Every number that decides
#: something in this module is a required parameter with no default.
PACKET_CHAR_BUDGET = 6000

SCORE_MIN, SCORE_MAX = 0.0, 1.0

NOT_VALIDATED = "NOT_VALIDATED"
FOR_HUMAN_REVIEW = "FOR_HUMAN_REVIEW"

JUDGED_NOTICE = (
    "JUDGED evidence: a model's opinion, conditioned on the judge model and date recorded "
    "on this verdict. It may rank, screen or flag for a human. It may not gate, adopt or "
    "validate anything, and it is not a substitute for the human adjudication the sampled "
    "layer requires."
)

#: Field names a Verdict may never carry — `if v.passed` must not be writable against it.
DECISION_FIELD_NAMES = ("pass", "passed", "fail", "failed", "accept", "accepted", "reject",
                        "rejected", "ok", "approved", "valid", "validated", "gate", "adopt")

PERMITTED_USES = ("RANK", "SCREEN", "FLAG_FOR_HUMAN")
FORBIDDEN_USES = ("GATE", "ADOPT", "VALIDATE", "ACCEPT", "REJECT", "PUBLISH", "AUTO_MERGE",
                  "MARK_REVIEWED", "SIGN_OFF")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(dimension: Any) -> str:
    """Case and whitespace folded BEFORE the registry lookup — otherwise ` Histology `
    misses a registry keyed on `histology` and the precedence check is bypassed by a typo."""
    return str(dimension or "").strip().lower()


# ================================================================================ the seams
@runtime_checkable
class PrecedenceRegistry(Protocol):
    """The query shape this module requires of `acr.evals`' precedence registry.

    One method, one question: does a deterministic evaluator exist for this dimension?
    Return the evaluator (or any truthy handle) if one does, None if none does. Whether it
    is currently enabled, configured or passing is deliberately not asked — EXISTENCE is
    what decides precedence, so a disabled evaluator still forbids judging rather than
    silently handing the dimension to a model.

    `_lookup` also accepts `evaluator_for(dimension)` and a plain Mapping, so this module
    does not block on the exact name the sibling module lands with.
    """

    def deterministic_evaluator_for(self, dimension: str) -> Any | None: ...


class JudgeModel(Protocol):
    """The whole model seam. `model_id` is mandatory: a verdict whose producing model is
    unknown cannot be re-checked when the model changes underneath us, and models change."""

    model_id: str

    def ask(self, prompt: str) -> Mapping[str, Any]: ...


def _lookup(registry: Any, dimension: str) -> Any | None:
    """Ask the registry, and fail CLOSED on anything unexpected.

    An absent or broken registry is not evidence that a dimension is judgeable. Treating it
    as such is how a dimension that L4 decides mechanically ends up scored by a model after
    somebody moves a module.
    """
    if registry is None:
        raise RegistryUnavailable(
            "no precedence registry supplied; precedence is unknown and judging is refused. "
            "Pass acr.evals' registry (see PrecedenceRegistry for the required query shape).")
    for name in ("deterministic_evaluator_for", "evaluator_for"):
        fn = getattr(registry, name, None)
        if callable(fn):
            try:
                return fn(dimension)
            except Exception as exc:      # an unknown state must not read as "none"
                raise RegistryUnavailable(
                    f"precedence registry raised on {dimension!r}: {exc!r}") from exc
    if isinstance(registry, Mapping):
        return registry.get(dimension)
    raise RegistryUnavailable(f"{type(registry).__name__} implements no known precedence "
                              f"query (deterministic_evaluator_for / evaluator_for / Mapping)")


# ================================================================================== packets
@dataclass(frozen=True)
class BlindPacket:
    """What a blinded judge sees. There is no field here that can hold the key."""

    trace: tuple[dict, ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    subject_id: str = ""


@dataclass(frozen=True)
class KeyedPacket:
    """Triage only. `judge()` refuses this for every blinded dimension."""

    trace: tuple[dict, ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    answer_key: Mapping[str, Any] = field(default_factory=dict)
    subject_id: str = ""


def _scan_for_key(obj: Any, path: str = "artifacts") -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if _norm(k) in KEY_BEARING_NAMES:
                raise AnswerKeyLeak(
                    f"{path}.{k} is a key-bearing name; a blinded dimension may not see it. "
                    f"Judge the trajectory or the explanation on its own terms, or use "
                    f"{DIM_BAD_CASE_TRIAGE}, which is allowed the key.")
            _scan_for_key(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scan_for_key(v, f"{path}[{i}]")


def blind_packet(trace: Sequence[Mapping[str, Any]] = (),
                 artifacts: Mapping[str, Any] | None = None,
                 subject_id: str = "") -> BlindPacket:
    """Build a blinded packet, refusing key material smuggled in as an artifact."""
    arts = dict(artifacts or {})
    _scan_for_key(arts)
    _scan_for_key(list(trace), "trace")
    return BlindPacket(trace=tuple(dict(e) for e in trace), artifacts=arts,
                       subject_id=str(subject_id))


def keyed_packet(trace: Sequence[Mapping[str, Any]] = (),
                 artifacts: Mapping[str, Any] | None = None,
                 answer_key: Mapping[str, Any] | None = None,
                 subject_id: str = "") -> KeyedPacket:
    return KeyedPacket(trace=tuple(dict(e) for e in trace), artifacts=dict(artifacts or {}),
                       answer_key=dict(answer_key or {}), subject_id=str(subject_id))


# =================================================================================== lenses
@dataclass(frozen=True)
class Lens:
    key: str
    question: str
    catches: str          # the failure mode this lens exists to catch, and no other lens does


#: The BUILT-IN panels, for the dimensions this module has always exposed through `judge()`.
#: A NEW evaluation need does not come here: it is a YAML in `evaluators/`, loaded by
#: `load_evaluator()` and run by `run_evaluator()`. Adding a lens to this dict is a code
#: change, a deploy, and a domain expert who cannot read or sign what they asked for.
LENSES: dict[str, tuple[Lens, ...]] = {
    DIM_L5_EXPLANATION_QUALITY: (
        Lens("cause_grounding",
             "For each cause the explanation asserts, name the specific ledger fact or cited "
             "document in the packet that supports it. Which assertions have none?",
             "a fluent explanation resting on facts that are not in the packet"),
        Lens("alternatives_addressed",
             "Which of the other candidate causes does the explanation fail to rule out, and "
             "what in the packet would have let it rule them out?",
             "a plausible cause chosen without the three rivals being considered"),
        Lens("certainty_calibration",
             "Does the stated confidence exceed what the coverage proof in the packet "
             "supports? Quote the sentence that overclaims, if any.",
             "correct cause, unearned certainty — the failure grounding cannot see"),
    ),
    DIM_TRAJECTORY_QUALITY: (
        Lens("read_order",
             "In the order the documents were actually read, was anything decisive left until "
             "after speculative reading? Give the step numbers.",
             "the right answer reached by an expensive route nobody would repeat"),
        Lens("unsettled_threads",
             "Name every contradiction, hedge or unresolved hint raised during the run that "
             "was never followed up.",
             "a dropped thread, invisible in a run that still finished"),
        Lens("effort_proportionality",
             "Was the effort spent proportionate to the difficulty visible in the packet? "
             "Say whether it was over-spent or under-spent and on which steps.",
             "both over- and under-spend, which a quality-only lens scores identically"),
    ),
    # The judged half of a SPLIT dimension. Admissibility — may this document class establish
    # this field — is already decided by acr.coverage.admissibility_for_citations and is
    # forbidden here. What is left is what no exact check can pose.
    DIM_EVIDENCE_SUPPORT_JUDGED: (
        Lens("quote_states_the_value",
             "Does the cited passage STATE the coded value, or does it merely mention the "
             "topic nearby? Quote the words that carry the value, or say there are none.",
             "an admissible document cited at the wrong sentence — real quote, wrong claim"),
        Lens("contradiction_in_what_was_never_opened",
             "In the documents listed as NOT READ, is there anything that contradicts, "
             "supersedes or amends the cited evidence? Name the document and the sentence.",
             "the care gap that is really a documentation gap: the record disagreed with "
             "itself and the run only ever saw one side"),
        Lens("hedge_read_as_fact",
             "Was a hedged phrase — favor, suspicious for, cannot exclude, consistent with — "
             "converted into a definite coded value? Quote the hedge.",
             "the registrar-error mode: 'favor squamous' coded as squamous, which a "
             "supports/does-not-support question scores as supported"),
    ),
    # The other judged half. The counters are acr.evals.detect_resource_band's and stay
    # deterministic; only "given what it knew AT THE TIME" is asked here.
    DIM_STEP_EFFICIENCY_JUDGED: (
        Lens("reasonable_given_what_was_known",
             "Taking each step in order and using ONLY what was visible before it, was the "
             "next step a reasonable thing to do? Name the first step that was not.",
             "hindsight scoring: a step that looks wasteful only because we now know the "
             "answer was elsewhere, which the turn counter also cannot separate"),
        Lens("cheaper_step_available",
             "Was there a cheaper step already available that would have settled the same "
             "question earlier? Name it and the step it should have replaced.",
             "expensive route to a right answer, invisible to a band the run stayed inside"),
        Lens("spend_after_it_was_settled",
             "After which step was the answer already established by admissible evidence, "
             "and what was spent after that point?",
             "the tail of a run, where cost accrues and nothing is learned"),
    ),
    DIM_BAD_CASE_TRIAGE: (
        Lens("human_informative",
             "Would a human reading this case learn something not already stated in the "
             "packet? What specifically?",
             "cases that are merely confirmed-wrong, and waste the scarce reading hour"),
        Lens("systematic_or_idiosyncratic",
             "Does this failure look like a class that would recur across patients, or a "
             "one-off? Name the class if there is one.",
             "one-offs crowding out a repeating failure worth fixing once"),
        Lens("actionability",
             "If the diagnosis in this case is right, what would change — a spec rule, an "
             "evidence rule, a prompt, or nothing?",
             "interesting-but-unfixable cases ranked above boring-but-fixable ones"),
    ),
}


def _lenses_for(dimension: str) -> tuple[Lens, ...]:
    lenses = LENSES[dimension]
    # Repetition is not a panel. Three askings of one question measure sampling noise; three
    # questions aimed at three failure modes measure three things.
    if len({ln.question for ln in lenses}) != len(lenses):
        raise ValueError(f"{dimension}: duplicate lens question — repetition, not a panel")
    return lenses


# ================================================================================== verdicts
@dataclass(frozen=True)
class LensReading:
    lens: str
    score: float | None            # None = the model returned nothing usable; NOT zero
    observation: str = ""
    concerns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """One judged opinion. Frozen, stamped, and impossible to read as a decision.

    There is deliberately no `passed`, `accepted` or `ok` field: a caller cannot write
    `if verdict.passed` against a field that does not exist, and `__bool__` refuses so that
    `if verdict:` cannot become a gate by accident either.
    """

    dimension: str
    subject_id: str
    judge_model: str
    judged_at: str
    lens_readings: tuple[LensReading, ...]
    score: float | None
    concerns: tuple[str, ...] = ()
    incomplete: bool = False
    evidence_class: str = EV_JUDGED
    notice: str = JUDGED_NOTICE

    def __bool__(self) -> bool:
        raise JudgeCannotGate(
            "a Verdict has no truth value. Truthiness here would be a gate wearing the "
            "costume of an if-statement; use rank(), screen_for_human() or .score.")

    def to_dict(self) -> dict:
        return {"dimension": self.dimension, "subject_id": self.subject_id,
                "evidence_class": self.evidence_class, "judge_model": self.judge_model,
                "judged_at": self.judged_at, "score": self.score, "notice": self.notice,
                "incomplete": self.incomplete, "concerns": list(self.concerns),
                "validation_status": NOT_VALIDATED,
                "lens_readings": [{"lens": r.lens, "score": r.score, "concerns":
                                   list(r.concerns), "observation": r.observation}
                                  for r in self.lens_readings]}


# ==================================================================================== judge
#: Trace keys the judge is shown. An allowlist, not a denylist: the prompt is built from
#: named fields, so an attribute bolted onto an event or a packet subclass cannot ride into
#: the model's context unnoticed.
TRACE_KEYS_SHOWN = ("seq", "kind", "tool", "args", "result", "verdict", "reason", "content",
                    "plan")


def _render(packet: BlindPacket | KeyedPacket) -> str:
    body: dict[str, Any] = {
        "artifacts": packet.artifacts,
        "trace": [{k: v for k, v in e.items() if k in TRACE_KEYS_SHOWN} for e in packet.trace]}
    if isinstance(packet, KeyedPacket):
        body["answer_key"] = packet.answer_key
    return json.dumps(body, indent=1, ensure_ascii=False, default=str)[:PACKET_CHAR_BUDGET]


def _prompt(dimension: str, lens: Lens, rendered: str) -> str:
    return (f"You are screening one case for a human reviewer. Dimension: {dimension}.\n"
            f"You are NOT deciding anything: your output orders a human's reading queue.\n\n"
            f"QUESTION (answer only this one):\n{lens.question}\n\n"
            f"THE CASE:\n{rendered}\n\n"
            f'Reply with JSON only: {{"score": <{SCORE_MIN}-{SCORE_MAX}, higher is better>, '
            f'"observation": "<what you saw, citing the packet>", "concerns": ["..."]}}')


def _read_lens(lens: Lens, raw: Any) -> LensReading:
    m: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    obs = str(m.get("observation", ""))[:800]
    concerns = tuple(str(c)[:300] for c in (m.get("concerns") or []))
    try:
        score: float | None = float(m["score"])
    except (TypeError, ValueError, KeyError):
        score = None
    # An unusable or out-of-range reply is NOT a zero. A fabricated zero would sort this case
    # to the front of the human queue as if the judge had found something, and would drag any
    # mean computed over the lenses; None keeps "the judge failed here" visible downstream.
    if score is None or not SCORE_MIN <= score <= SCORE_MAX:
        return LensReading(lens.key, None, obs, concerns)
    return LensReading(lens.key, score, obs, concerns)


def judge(dimension: str, packet: BlindPacket | KeyedPacket, *,
          registry: Any, model: JudgeModel) -> Verdict:
    """Ask the panel of lenses about one case, or refuse.

    There is no `force`, no `policy`, no `**kwargs`: every override anyone might reach for
    has to be a TypeError, because an override that merely logs a warning gets used.
    """
    dim = _norm(dimension)

    # Precedence FIRST, before the judgeable-dimension allowlist. The allowlist is a module
    # constant and therefore monkeypatchable; the registry answer is not, so a dimension that
    # code already decides is refused even by a caller who has widened the allowlist.
    existing = _lookup(registry, dim)
    if existing is not None:
        raise DeterministicEvaluatorExists(
            f"{dim!r} is decided by {existing!r}. A judged opinion may not stand in for a "
            f"deterministic evaluator, and there is no argument to this function that "
            f"changes that. If the evaluator is wrong, fix the evaluator.")

    if dim not in JUDGEABLE_DIMENSIONS:
        raise DimensionNotJudgeable(
            f"{dim!r} is not one of {JUDGEABLE_DIMENSIONS}. Judging is for dimensions with no "
            f"ground truth BY CONSTRUCTION, not ones that merely lack an evaluator yet — "
            f"write the evaluator.")

    # Nominal, not duck-typed: an object that merely looks like a packet could carry a key
    # past the isolation check below.
    if not isinstance(packet, (BlindPacket, KeyedPacket)):
        raise TypeError("packet must be a BlindPacket or KeyedPacket")
    if dim in KEY_BLINDED_DIMENSIONS:
        if isinstance(packet, KeyedPacket):
            raise AnswerKeyLeak(
                f"{dim} is judged blind: with the key in hand it becomes a noisy restatement "
                f"of accuracy, already measured deterministically. Use blind_packet().")
        # Re-scanned here and not only in `blind_packet()`: BlindPacket is a plain dataclass
        # and can be constructed directly, so the factory is a convenience and this is the
        # enforcement point.
        _scan_for_key(packet.artifacts)
        _scan_for_key(list(packet.trace), "trace")

    model_id = str(getattr(model, "model_id", "") or "")
    if not model_id:
        raise ValueError("the judge model must report a model_id: a judged number is "
                         "conditioned on the model that produced it, and an unattributable "
                         "verdict cannot be re-checked when the model changes")

    rendered = _render(packet)
    readings = tuple(_read_lens(ln, model.ask(_prompt(dim, ln, rendered)))
                     for ln in _lenses_for(dim))
    scored = [r.score for r in readings if r.score is not None]
    return Verdict(
        dimension=dim, subject_id=str(getattr(packet, "subject_id", "")),
        judge_model=model_id, judged_at=_now(), lens_readings=readings,
        score=(sum(scored) / len(scored)) if scored else None,
        concerns=tuple(c for r in readings for c in r.concerns),
        incomplete=len(scored) != len(readings))


# ============================================================== measurements and the split
@dataclass(frozen=True)
class Measurement:
    """A number that knows what kind of number it is."""

    dimension: str
    score: float
    evidence_class: str
    source: str = ""

    @classmethod
    def from_verdict(cls, v: Verdict) -> "Measurement":
        if v.score is None:
            raise ValueError(f"{v.dimension}: no lens produced a usable score, so there "
                             f"is no measurement to carry forward")
        return cls(v.dimension, v.score, EV_JUDGED, f"{v.judge_model}@{v.judged_at}")


def deterministic_measurement(dimension: str, score: float, *, registry: Any) -> Measurement:
    """Stamp DETERMINISTIC — and prove the claim against the same registry `judge()` asks.

    The fence has two sides. Refusing to judge a deterministic dimension is useless if a
    judged number can be relabelled DETERMINISTIC on the way into a report.
    """
    dim = _norm(dimension)
    ev = _lookup(registry, dim)
    if ev is None:
        raise DeterministicEvaluatorExists(
            f"nothing deterministic evaluates {dim!r}, so this number cannot be stamped "
            f"{EV_DETERMINISTIC}. Register the evaluator, or carry it as {EV_JUDGED}.")
    return Measurement(dim, float(score), EV_DETERMINISTIC, source=repr(ev))


def aggregate(measurements: Sequence[Measurement]) -> dict:
    """Mean of measurements that share an evidence class. Raises on a mixed set.

    This is the silent-averaging fence. One mean over both classes produces a single
    defensible-looking score in which a counted stratum and a model's opinion are
    indistinguishable, and no reader downstream can take them apart again.
    """
    if not measurements:
        raise ValueError("nothing to aggregate")
    classes = {m.evidence_class for m in measurements}
    if len(classes) > 1:
        raise MixedEvidence(
            f"refusing to average across {sorted(classes)}. Call combine_explicitly() with a "
            f"stated weight if you really want one number; the split stays in the output.")
    cls = classes.pop()
    return {"evidence_class": cls, "n": len(measurements),
            "mean": sum(m.score for m in measurements) / len(measurements),
            "dimensions": sorted(m.dimension for m in measurements)}


def combine_explicitly(measurements: Sequence[Measurement], *, judged_weight: float) -> dict:
    """One number, on request, with the split preserved beside it.

    `judged_weight` has no default on purpose: how much a model's opinion should count
    against a counted fact is a judgement call the caller must make in the open, in the
    diff, where a reviewer can argue with it.
    """
    if judged_weight is None or not SCORE_MIN <= float(judged_weight) <= SCORE_MAX:
        raise ValueError(f"judged_weight must be given, in [{SCORE_MIN}, {SCORE_MAX}]")
    w = float(judged_weight)
    det = [m for m in measurements if m.evidence_class == EV_DETERMINISTIC]
    jud = [m for m in measurements if m.evidence_class == EV_JUDGED]
    unknown = sorted({m.evidence_class for m in measurements
                      if m.evidence_class not in (EV_DETERMINISTIC, EV_JUDGED)})
    if unknown:
        raise MixedEvidence(f"unstamped or unknown evidence class: {unknown}")
    if not det and not jud:
        raise ValueError("nothing to combine")
    d_mean = (sum(m.score for m in det) / len(det)) if det else None
    j_mean = (sum(m.score for m in jud) / len(jud)) if jud else None
    if d_mean is None:
        combined, note = j_mean, "combined number is JUDGED throughout — no deterministic input"
    elif j_mean is None:
        combined, note = d_mean, "combined number is DETERMINISTIC throughout"
    else:
        combined, note = (1 - w) * d_mean + w * j_mean, "combined number is part opinion"
    return {
        "combined": combined, "judged_weight": w, "caveat": note,
        "deterministic": {"n": len(det), "mean": d_mean,
                          "dimensions": sorted(m.dimension for m in det)},
        "judged": {"n": len(jud), "mean": j_mean,
                   "dimensions": sorted(m.dimension for m in jud),
                   "sources": sorted({m.source for m in jud})},
        "validation_status": NOT_VALIDATED if jud else "", "notice": JUDGED_NOTICE if jud else ""}


# ================================================================ what a verdict may be used for
def apply_verdict(verdict: Verdict, use: str) -> dict:
    """The only door out of this module, and it opens onto three uses.

    Fail-closed on an unknown use: a new verb nobody thought about here is more likely to be
    a gate than a ranking, and refusing costs one line in this tuple.
    """
    u = str(use or "").strip().upper()
    if u in FORBIDDEN_USES:
        raise JudgeCannotGate(
            f"{u} is a decision. This verdict is {EV_JUDGED} evidence from "
            f"{verdict.judge_model}; it may {', '.join(PERMITTED_USES)} and nothing else. "
            f"Whatever {u} would have done needs a deterministic evaluator or a human.")
    if u not in PERMITTED_USES:
        raise JudgeCannotGate(f"unknown use {u!r}; permitted uses are {PERMITTED_USES}")
    return {"use": u, "evidence_class": EV_JUDGED, "disposition": FOR_HUMAN_REVIEW,
            "validation_status": NOT_VALIDATED, "verdict": verdict.to_dict()}


def rank(verdicts: Sequence[Verdict], *, worst_first: bool) -> list[Verdict]:
    """Order verdicts. Unscored cases sort to the human's end of the list either way —
    'the judge could not read this' is a reason for a person to look, not a reason to skip."""
    if worst_first is None:
        raise ValueError("worst_first must be stated")
    return sorted(verdicts, key=lambda v: (v.score is not None,
                                           v.score if v.score is not None else 0.0),
                  reverse=not worst_first)


def screen_for_human(verdicts: Sequence[Verdict], *, flag_at_or_below: float,
                     queue_size: int) -> dict:
    """Build a reading queue. A SCREEN in front of the human, never instead of one.

    Nothing is discarded silently: `n_below_cutoff` and `n_not_shown` are reported so a
    queue truncated to the hours available cannot be read as "and the rest were fine".
    """
    if flag_at_or_below is None or not SCORE_MIN <= float(flag_at_or_below) <= SCORE_MAX:
        raise ValueError(f"flag_at_or_below must be given, in [{SCORE_MIN}, {SCORE_MAX}]")
    if queue_size is None or int(queue_size) < 1:
        raise ValueError("queue_size must be given and at least 1")
    cut, size = float(flag_at_or_below), int(queue_size)
    flagged = [v for v in verdicts if v.score is None or v.score <= cut]
    ordered = rank(flagged, worst_first=True)
    return {
        "queue": [{"subject_id": v.subject_id, "dimension": v.dimension, "score": v.score,
                   "incomplete": v.incomplete, "concerns": list(v.concerns),
                   "judge_model": v.judge_model, "judged_at": v.judged_at,
                   "evidence_class": EV_JUDGED, "disposition": FOR_HUMAN_REVIEW}
                  for v in ordered[:size]],
        "n_judged": len(verdicts), "n_below_cutoff": len(flagged),
        "n_not_shown": max(0, len(flagged) - size), "flag_at_or_below": cut,
        "evidence_class": EV_JUDGED, "validation_status": NOT_VALIDATED,
        "notice": JUDGED_NOTICE}


# ================================================ AN EVALUATOR IS A SPEC FILE, NOT CODE
# A new evaluation need is a new YAML in `evaluators/`. Nobody deploys code to ask a new
# question, and a domain expert can read and sign the whole evaluator without reading Python.
#
# Five refusals happen at LOAD, before a token is spent. Each is a defect this repo has
# already shipped in another form:
#
#   1. THE PRECEDENCE FENCE, PER SUB-QUESTION. A dimension that code decides is refused with
#      the deterministic method named. The fence is `acr.evals.REGISTRY` — data — not an `if`
#      buried here. Per SUB-QUESTION because evidence_support and step_efficiency each have a
#      deterministic half and a judged half, and a per-dimension fence kills the legitimate
#      one. Naming a split parent is refused and told which half to declare.
#   2. CONTEXT IS DECLARED AND NOTHING ELSE IS INJECTED. `expected_output` on a dimension it
#      would contaminate fails to load, so withholding the answer key is a property of the
#      file that a reviewer can check by reading it.
#   3. TOOL SCOPE IS DECLARED AND BOUNDED. An agent-judge that can open documents IS A PHI
#      ACCESS PATH. It may open only the chart of the patient whose trace it is judging — the
#      same cross-patient rule `acr.evals.detect_patient_crossover` enforces on the agent,
#      now on its auditor, at load AND at every tool call.
#   4. AT LEAST ONE must_pass AND ONE must_fail CASE, OR IT DOES NOT LOAD. The one that
#      matters most. The recurring defect here is the check that cannot fail: a gate that
#      never read its flag, a matcher that "t" satisfied, a renderer that reported every
#      element unattributed, a trigger pipeline that had never executed. An evaluator that
#      scores everything the same looks exactly like a clean system. Requiring a case it must
#      REJECT makes "decorative" a load error — and `certify_evaluator()` runs the cases, so
#      it is not merely a list of ids either.
#   5. A JUDGE RUN IS A RUN: traced, cost-accounted, rate-limited, `cost_class` declared, so
#      a fleet of document-opening judges cannot quietly cost more than the reviews it audits.

class EvaluatorSpecInvalid(JudgeRefusal):
    """The YAML is not a loadable evaluator. Refused at load, before any spend."""


class EvaluatorCannotFail(JudgeRefusal):
    """No case it must reject, or it did not reject the case it declared it would."""


class ToolScopeViolation(JudgeRefusal):
    """A judge tool was declared or invoked outside the patient under review."""


class JudgeBudgetExceeded(JudgeRefusal):
    """The judge fleet hit its declared call or cost ceiling. Refused, not throttled."""


COST_TRACE_ONLY = "trace_only"
COST_READS_DOCUMENTS = "reads_documents"
COST_RERUNS_SEARCHES = "reruns_searches"
#: Cheapest first. An evaluator may declare a class at least as expensive as its tools and
#: context imply, never cheaper — an under-declared cost class is how a budget is blown by a
#: file that read as free.
COST_CLASSES = (COST_TRACE_ONLY, COST_READS_DOCUMENTS, COST_RERUNS_SEARCHES)

#: The whole context vocabulary. Anything not on this list cannot be requested, so a new
#: injection channel has to be opened here, in code review, and not in a YAML file.
CONTEXT_VARIABLES = ("trace", "cited_evidence", "coverage_ledger", "spec", "gate_verdict",
                     "documents_not_read", "expected_output")
#: The one that makes an agent-judge worth paying for — and it cannot be free.
CONTEXT_MIN_COST = {"documents_not_read": COST_READS_DOCUMENTS}
#: The answer key, by its context name.
CONTEXT_ANSWER_KEY = "expected_output"

#: Tool -> the cheapest cost class that can honestly declare it.
EVALUATOR_TOOLS = {"read_document": COST_READS_DOCUMENTS,
                   "read_section": COST_READS_DOCUMENTS,
                   "list_documents": COST_READS_DOCUMENTS,
                   "search_notes": COST_RERUNS_SEARCHES}
#: The ONLY permissible scope. Not a default and not a maximum — the only accepted value, so
#: `scope: cohort` is a load error rather than a policy discussion held once, at 2am.
TOOL_SCOPE_PATIENT_UNDER_REVIEW = "patient_under_review"

SCALES = {"binary": 2, "3-point": 3, "5-point": 5}
PROMPT_FIELDS = ("role", "dimensions", "scale", "checklist")
EVALUATOR_FIELDS = ("evaluator_id", "dimension", "cost_class", "prompt", "context", "tools",
                    "output", "must_pass", "must_fail")
#: score and reason are required of every evaluator: a score with no reason cannot be argued
#: with by the human it was produced for, and an unarguable number gets believed.
REQUIRED_OUTPUT = ("score", "reason")
OPTIONAL_OUTPUT = ("cot",)


def _str_list(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise EvaluatorSpecInvalid(f"{what} must be a list, got {type(value).__name__}")
    items = tuple(str(v).strip() for v in value)
    if not items or any(not i for i in items):
        raise EvaluatorSpecInvalid(f"{what} must be a non-empty list of non-empty strings")
    if len(set(items)) != len(items):
        raise EvaluatorSpecInvalid(f"{what} has a duplicate entry: {sorted(items)}")
    return items


@dataclass(frozen=True)
class ToolGrant:
    """One declared tool and the scope it is bounded to. Both are load-checked."""

    name: str
    scope: str


@dataclass(frozen=True)
class EvaluatorSpec:
    """A loaded `evaluators/*.yaml`. Frozen: nothing widens a grant after the load checks."""

    evaluator_id: str
    dimension: str
    cost_class: str
    role: str
    prompt_dimensions: tuple[str, ...]
    scale: str
    checklist: tuple[str, ...]
    context: tuple[str, ...]
    tools: tuple[ToolGrant, ...]
    output: Mapping[str, str]
    must_pass: tuple[str, ...]
    must_fail: tuple[str, ...]
    source: str = ""

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)

    @property
    def sees_answer_key(self) -> bool:
        return CONTEXT_ANSWER_KEY in self.context

    def to_dict(self) -> dict:
        return {"evaluator_id": self.evaluator_id, "dimension": self.dimension,
                "cost_class": self.cost_class, "scale": self.scale,
                "context": list(self.context),
                "tools": [{"name": t.name, "scope": t.scope} for t in self.tools],
                "must_pass": list(self.must_pass), "must_fail": list(self.must_fail),
                "source": self.source}


def _check_dimension(dimension: str, registry: Any) -> str:
    """ENFORCEMENT 1. The precedence fence, at load, per sub-question.

    `_lookup` is the same query `judge()` makes, so an evaluator file and a direct call
    cannot disagree about what is fenced. A registry that raises — including on a split
    parent, which `acr.evals` refuses to rule on as a whole — becomes a refusal here rather
    than an assumption that nothing deterministic exists.
    """
    dim = _norm(dimension)
    if not dim:
        raise EvaluatorSpecInvalid("dimension is required")
    existing = _lookup(registry, dim)       # RegistryUnavailable on unknown/split/broken
    if existing is not None:
        raise DeterministicEvaluatorExists(
            f"evaluator declares {dim!r}, which is decided by {existing!r}. Use that method; "
            f"a judge may never re-answer a question that has an exact answer. If this "
            f"evaluator means the adjacent question that has no exact answer, declare that "
            f"sub-question instead — the fence is per sub-question, not per dimension.")
    return dim


def _check_context(context: Any, dimension: str, cost_class: str) -> tuple[str, ...]:
    """ENFORCEMENT 2. Declared, closed, and the answer key withheld by the file itself."""
    ctx = _str_list(context, "context")
    unknown = [c for c in ctx if c not in CONTEXT_VARIABLES]
    if unknown:
        raise EvaluatorSpecInvalid(
            f"context {unknown} is not injectable. Permitted: {list(CONTEXT_VARIABLES)}. A "
            f"new channel into a judge's prompt is a code change, reviewed, not a YAML line.")
    if CONTEXT_ANSWER_KEY in ctx and dimension not in KEY_PERMITTED_DIMENSIONS:
        raise AnswerKeyLeak(
            f"{CONTEXT_ANSWER_KEY} is refused for {dimension!r}: with the key in hand this "
            f"evaluator becomes a noisy restatement of accuracy, which is already measured "
            f"deterministically and better. Only {list(KEY_PERMITTED_DIMENSIONS)} may see it, "
            f"because there the disagreement with the key IS the material being sorted.")
    for c in ctx:
        need = CONTEXT_MIN_COST.get(c)
        if need and COST_CLASSES.index(cost_class) < COST_CLASSES.index(need):
            raise EvaluatorSpecInvalid(
                f"context {c!r} cannot be assembled under cost_class {cost_class!r}: reading "
                f"what the agent did not read means opening documents. Declare {need!r}.")
    return ctx


def _check_tools(tools: Any, cost_class: str) -> tuple[ToolGrant, ...]:
    """ENFORCEMENT 3. Declared, bounded to the patient under review, priced honestly."""
    rows = list(tools or [])
    # Checked before the per-tool price, or this branch is unreachable: every grantable tool
    # costs at least `reads_documents`, so a trace_only file would always be refused with a
    # message about pricing rather than the one that says what is actually wrong.
    if cost_class == COST_TRACE_ONLY and rows:
        raise EvaluatorSpecInvalid(f"cost_class {COST_TRACE_ONLY!r} grants no tools, but "
                                   f"{len(rows)} are declared")
    grants: list[ToolGrant] = []
    for row in rows:
        # A bare string is a tool with no scope, which is exactly the declaration that must
        # not be possible: an unscoped document reader is an unbounded PHI access path.
        if not isinstance(row, Mapping) or "name" not in row:
            raise EvaluatorSpecInvalid(
                f"each tool must be a mapping with `name` and `scope`, got {row!r}. A tool "
                f"without a declared scope is an unbounded PHI access path.")
        name, scope = str(row.get("name") or "").strip(), str(row.get("scope") or "").strip()
        if name not in EVALUATOR_TOOLS:
            raise EvaluatorSpecInvalid(f"tool {name!r} is not grantable to a judge. "
                                       f"Permitted: {sorted(EVALUATOR_TOOLS)}")
        if scope != TOOL_SCOPE_PATIENT_UNDER_REVIEW:
            raise ToolScopeViolation(
                f"tool {name!r} declares scope {scope!r}; the only permitted scope is "
                f"{TOOL_SCOPE_PATIENT_UNDER_REVIEW!r}. The auditor is held to the same "
                f"cross-patient rule as the agent it audits: one trace, one chart.")
        need = EVALUATOR_TOOLS[name]
        if COST_CLASSES.index(cost_class) < COST_CLASSES.index(need):
            raise EvaluatorSpecInvalid(f"tool {name!r} requires cost_class at least {need!r}, "
                                       f"but {cost_class!r} was declared")
        grants.append(ToolGrant(name, scope))
    if len({g.name for g in grants}) != len(grants):
        raise EvaluatorSpecInvalid("a tool is granted twice; one grant, one scope")
    if cost_class != COST_TRACE_ONLY and not grants:
        raise EvaluatorSpecInvalid(
            f"cost_class {cost_class!r} says this evaluator opens the record, but it declares "
            f"no tools. Declare the tools or declare {COST_TRACE_ONLY!r} — a cost class that "
            f"overstates access is as unreviewable as one that understates it.")
    return tuple(grants)


def _check_cases(data: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """ENFORCEMENT 4. A case it must pass AND a case it must REJECT, or it does not load."""
    for fieldname in ("must_pass", "must_fail"):
        if not data.get(fieldname):
            raise EvaluatorCannotFail(
                f"{fieldname} is empty. Every evaluator ships with at least one case it must "
                f"pass and at least one it must FAIL. An evaluator that scores everything the "
                f"same is indistinguishable from a clean system, and this repo has shipped "
                f"four checks that could not fail; a declared rejection makes 'decorative' a "
                f"load error instead of a discovery six weeks later.")
    passes = _str_list(data["must_pass"], "must_pass")
    fails = _str_list(data["must_fail"], "must_fail")
    both = sorted(set(passes) & set(fails))
    if both:
        raise EvaluatorCannotFail(f"case(s) {both} are declared must_pass AND must_fail; that "
                                  f"evaluator cannot be wrong, which is the defect itself")
    return passes, fails


def _check_output(output: Any) -> Mapping[str, str]:
    if not isinstance(output, Mapping):
        raise EvaluatorSpecInvalid("output must be a mapping, e.g. "
                                   "{score: required, reason: required, cot: optional}")
    out = {str(k).strip(): str(v).strip() for k, v in output.items()}
    unknown = sorted(set(out) - set(REQUIRED_OUTPUT) - set(OPTIONAL_OUTPUT))
    if unknown:
        raise EvaluatorSpecInvalid(f"output field(s) {unknown} are not part of a verdict")
    for f in REQUIRED_OUTPUT:
        if out.get(f) != "required":
            raise EvaluatorSpecInvalid(
                f"output.{f} must be 'required'. A score with no reason cannot be argued with "
                f"by the human it exists to inform, and an unarguable number gets believed.")
    if out.get("cot") not in (None, "optional"):
        raise EvaluatorSpecInvalid("output.cot may only be 'optional'")
    return out


def parse_evaluator(data: Mapping[str, Any], *, registry: Any, source: str = "",
                    expect_id: str | None = None) -> EvaluatorSpec:
    """The five load-time refusals, in one place, over already-parsed YAML."""
    if not isinstance(data, Mapping):
        raise EvaluatorSpecInvalid(f"{source or 'evaluator'}: top level must be a mapping")
    unknown = sorted(set(map(str, data)) - set(EVALUATOR_FIELDS))
    if unknown:
        # A typo'd key is silent otherwise: `contexts:` would declare nothing and the
        # evaluator would run with an empty prompt rather than refusing to load.
        raise EvaluatorSpecInvalid(f"{source or 'evaluator'}: unknown field(s) {unknown}; "
                                   f"permitted: {list(EVALUATOR_FIELDS)}")
    eid = str(data.get("evaluator_id") or "").strip()
    if not eid:
        raise EvaluatorSpecInvalid(f"{source or 'evaluator'}: evaluator_id is required")
    if expect_id is not None and eid != expect_id:
        # Two files that disagree with their own ids make the id unusable as a citation in a
        # report, and let a file quietly shadow another.
        raise EvaluatorSpecInvalid(f"{source}: evaluator_id {eid!r} must equal the filename "
                                   f"stem {expect_id!r}")
    cost_class = str(data.get("cost_class") or "").strip()
    if cost_class not in COST_CLASSES:
        raise EvaluatorSpecInvalid(f"{eid}: cost_class must be one of {list(COST_CLASSES)}, "
                                   f"got {cost_class!r}. A judge run is a run and its class "
                                   f"is what the budget is planned against.")
    dim = _check_dimension(data.get("dimension"), registry)

    prompt = data.get("prompt")
    if not isinstance(prompt, Mapping):
        raise EvaluatorSpecInvalid(f"{eid}: prompt must be a mapping with {list(PROMPT_FIELDS)}")
    missing = [f for f in PROMPT_FIELDS if not prompt.get(f)]
    if missing:
        raise EvaluatorSpecInvalid(f"{eid}: prompt is missing {missing}")
    scale = str(prompt["scale"]).strip()
    if scale not in SCALES:
        raise EvaluatorSpecInvalid(f"{eid}: scale must be one of {sorted(SCALES)}, got "
                                   f"{scale!r}")
    must_pass, must_fail = _check_cases(data)
    return EvaluatorSpec(
        evaluator_id=eid, dimension=dim, cost_class=cost_class,
        role=str(prompt["role"]).strip(),
        prompt_dimensions=_str_list(prompt["dimensions"], f"{eid}: prompt.dimensions"),
        scale=scale, checklist=_str_list(prompt["checklist"], f"{eid}: prompt.checklist"),
        context=_check_context(data.get("context"), dim, cost_class),
        tools=_check_tools(data.get("tools"), cost_class),
        output=_check_output(data.get("output")),
        must_pass=must_pass, must_fail=must_fail, source=source)


def load_evaluator(path: str | Path, *, registry: Any) -> EvaluatorSpec:
    """Load one `evaluators/*.yaml`. Refuses rather than degrades."""
    import yaml  # local: this module stays importable without a YAML parser present

    p = Path(path)
    return parse_evaluator(yaml.safe_load(p.read_text(encoding="utf-8")) or {},
                           registry=registry, source=str(p), expect_id=p.stem)


def load_evaluators(directory: str | Path, *, registry: Any) -> dict[str, EvaluatorSpec]:
    """Load every evaluator in a directory. One bad file refuses the whole load.

    Not skip-and-warn: a directory that silently drops the evaluator with the failing case
    and keeps the three that pass everything is the exact shape of the defect enforcement 4
    exists to catch.
    """
    d = Path(directory)
    if not d.is_dir():
        raise EvaluatorSpecInvalid(f"{d} is not a directory")
    out: dict[str, EvaluatorSpec] = {}
    for p in sorted(d.glob("*.yaml")):
        spec = load_evaluator(p, registry=registry)
        if spec.evaluator_id in out:
            raise EvaluatorSpecInvalid(f"duplicate evaluator_id {spec.evaluator_id!r}")
        out[spec.evaluator_id] = spec
    if not out:
        raise EvaluatorSpecInvalid(f"no evaluators found in {d}")
    return out


# ------------------------------------------------------- ENFORCEMENT 5: a judge run is a run
@dataclass(frozen=True)
class JudgeRunRecord:
    """One evaluator pass, in the shape the run manifests use. The trace of a judge run."""

    evaluator_id: str
    dimension: str
    cost_class: str
    subject_id: str
    judge_model: str
    judged_at: str
    n_model_calls: int
    cost_usd: float
    context_injected: tuple[str, ...]
    tools_granted: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"evaluator_id": self.evaluator_id, "dimension": self.dimension,
                "cost_class": self.cost_class, "subject_id": self.subject_id,
                "judge_model": self.judge_model, "judged_at": self.judged_at,
                "n_model_calls": self.n_model_calls, "cost_usd": self.cost_usd,
                "context_injected": list(self.context_injected),
                "tools_granted": list(self.tools_granted), "evidence_class": EV_JUDGED,
                "validation_status": NOT_VALIDATED}


class JudgeLedger:
    """Rate limit and cost accounting for a judge fleet. Every ceiling is stated by the caller.

    A fleet of document-opening judges can cost more than the reviews it is auditing, and it
    does so one cheap-looking call at a time. `charge()` runs BEFORE the model call: a limit
    enforced after the spend is a report, not a limit.
    """

    def __init__(self, *, max_calls: int, max_cost_usd: float,
                 cost_per_call_usd: Mapping[str, float]):
        if max_calls is None or int(max_calls) < 1:
            raise ValueError("max_calls is required and must be >= 1")
        if max_cost_usd is None or float(max_cost_usd) <= 0:
            raise ValueError("max_cost_usd is required and must be > 0")
        missing = [c for c in COST_CLASSES if c not in (cost_per_call_usd or {})]
        if missing:
            # No default price. A missing one would be read as free, and the class that gets
            # forgotten is `reads_documents`, which is the expensive one.
            raise ValueError(f"cost_per_call_usd must price every cost class; missing "
                             f"{missing} (no default price exists — an unpriced class reads "
                             f"as free and it is the document-opening one that gets omitted)")
        self.max_calls, self.max_cost_usd = int(max_calls), float(max_cost_usd)
        self.price = {k: float(v) for k, v in cost_per_call_usd.items()}
        self.records: list[JudgeRunRecord] = []
        self.n_calls, self.cost_usd = 0, 0.0

    def charge(self, spec: EvaluatorSpec, n_calls: int) -> float:
        cost = self.price[spec.cost_class] * int(n_calls)
        if self.n_calls + int(n_calls) > self.max_calls:
            raise JudgeBudgetExceeded(
                f"{spec.evaluator_id}: {self.n_calls + int(n_calls)} judge calls would exceed "
                f"the declared ceiling of {self.max_calls}")
        if self.cost_usd + cost > self.max_cost_usd:
            raise JudgeBudgetExceeded(
                f"{spec.evaluator_id}: ${self.cost_usd + cost:.4f} of judging would exceed the "
                f"declared ceiling of ${self.max_cost_usd:.4f} (cost_class "
                f"{spec.cost_class!r} at ${self.price[spec.cost_class]:.4f}/call)")
        self.n_calls += int(n_calls)
        self.cost_usd = round(self.cost_usd + cost, 6)
        return cost

    def record(self, rec: JudgeRunRecord) -> JudgeRunRecord:
        self.records.append(rec)
        return rec

    def report(self) -> dict:
        by_class: dict[str, dict] = {}
        for r in self.records:
            row = by_class.setdefault(r.cost_class, {"n_runs": 0, "n_calls": 0, "cost_usd": 0.0})
            row["n_runs"] += 1
            row["n_calls"] += r.n_model_calls
            row["cost_usd"] = round(row["cost_usd"] + r.cost_usd, 6)
        return {"n_runs": len(self.records), "n_calls": self.n_calls,
                "cost_usd": round(self.cost_usd, 6), "max_calls": self.max_calls,
                "max_cost_usd": self.max_cost_usd, "by_cost_class": by_class,
                "runs": [r.to_dict() for r in self.records]}


def scoped_tool_broker(spec: EvaluatorSpec, *, patient_under_review: str, backend: Any):
    """ENFORCEMENT 3 at call time. The declared scope, enforced on every invocation.

    Declaring a scope in YAML that nothing checks is the renderer that reported every element
    unattributed. This is the check: an ungranted tool is refused, and any call naming a
    patient other than the one whose trace is being judged is refused as a cross-patient
    access — the auditor held to the rule the runtime detectors hold the agent to.
    """
    if not patient_under_review:
        raise ToolScopeViolation("patient_under_review is required: a scope cannot be "
                                 "enforced against an unspecified patient")
    granted = set(spec.tool_names)

    def call(tool: str, **kwargs: Any) -> Any:
        if tool not in granted:
            raise ToolScopeViolation(f"{spec.evaluator_id} did not declare {tool!r}; declared: "
                                     f"{sorted(granted)}")
        for k in ("patient", "patient_id", "person_id", "subject_id"):
            if k in kwargs and str(kwargs[k]) != str(patient_under_review):
                raise ToolScopeViolation(
                    f"{spec.evaluator_id} tried to open {k}={kwargs[k]!r} while judging "
                    f"{patient_under_review!r}. An agent-judge that can open documents is a "
                    f"PHI access path: one trace, one chart.")
        return getattr(backend, tool)(**kwargs)

    return call


# ------------------------------------------------------------------ running a loaded evaluator
def build_context(spec: EvaluatorSpec, available: Mapping[str, Any]) -> dict:
    """ENFORCEMENT 2 at injection time. Exactly the declared variables, no more, no less.

    `available` is what the harness could supply; what is INJECTED is the intersection with
    the declaration — and a declared variable the harness cannot supply is an error rather
    than an empty section, because a judge silently asked to compare against a
    `documents_not_read` that never arrived will answer anyway.
    """
    missing = [c for c in spec.context if c not in available]
    if missing:
        raise EvaluatorSpecInvalid(f"{spec.evaluator_id} declares context {missing}, which the "
                                   f"harness did not supply; refusing to judge without it")
    return {c: available[c] for c in spec.context}


def _evaluator_prompt(spec: EvaluatorSpec, context: Mapping[str, Any]) -> tuple[str, bool]:
    """Render the declared context, and SAY SO when it did not all fit.

    Silent truncation is the decorative-pass hazard in its worst form here: an evaluator
    shown the first half of `documents_not_read` answers "nothing there contradicts the
    citation" and it reads exactly like a clean chart. The budget stays — a prompt has to end
    somewhere — but the fact of truncation travels into the prompt AND onto the verdict.
    """
    full = json.dumps(context, indent=1, ensure_ascii=False, default=str)
    truncated = len(full) > PACKET_CHAR_BUDGET
    body = full[:PACKET_CHAR_BUDGET] + (
        f"\n\n[TRUNCATED at {PACKET_CHAR_BUDGET} characters of {len(full)}. You have NOT been "
        f"shown all of the declared context. Do not report an absence — 'nothing there "
        f"contradicts it' is not something you can currently establish.]" if truncated else "")
    n = SCALES[spec.scale]
    cot = ', "cot": "<reasoning>"' if spec.output.get("cot") == "optional" else ""
    return (f"{spec.role}\n\nYou are screening one case for a human reviewer. Dimension: "
            f"{spec.dimension} (evaluator {spec.evaluator_id}).\nYou are NOT deciding "
            f"anything: your output orders a human's reading queue.\n\n"
            "WHAT TO ASSESS:\n" + "\n".join(f"- {d}" for d in spec.prompt_dimensions) +
            "\n\nCHECKLIST, IN ORDER:\n" +
            "\n".join(f"{i}. {c}" for i, c in enumerate(spec.checklist, start=1)) +
            f"\n\nTHE CASE (only the declared context is shown):\n{body}\n\n"
            f'Reply with JSON only: {{"score": <{spec.scale}: '
            f'{"0 or 1" if n == 2 else f"1..{n}"}, higher is better>, '
            f'"reason": "<why, citing the case>"{cot}}}', truncated)


def _normalised_score(spec: EvaluatorSpec, raw: Any) -> float | None:
    """The declared scale onto [0, 1]. Anything off-scale is None, never zero.

    A fabricated zero sorts the case to the front of the human queue as though the judge had
    found something, and drags any mean computed over it. "The judge failed here" has to stay
    visible; that is the same rule `_read_lens` follows.
    """
    n = SCALES[spec.scale]
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if spec.scale == "binary":
        return v if v in (0.0, 1.0) else None
    return (v - 1) / (n - 1) if 1 <= v <= n else None


def run_evaluator(spec: EvaluatorSpec, available_context: Mapping[str, Any], *,
                  registry: Any, model: JudgeModel, ledger: JudgeLedger,
                  subject_id: str) -> tuple[Verdict, JudgeRunRecord]:
    """Run one loaded evaluator: fence re-checked, context closed, charged, traced.

    The fence is re-asserted here and not trusted from load time: a spec object can outlive
    the registry it was loaded against, and "it was allowed when we loaded it" is how a
    dimension that acquired a deterministic evaluator keeps being judged.
    """
    _check_dimension(spec.dimension, registry)
    model_id = str(getattr(model, "model_id", "") or "")
    if not model_id:
        raise ValueError("the judge model must report a model_id: a judged number is "
                         "conditioned on the model that produced it")
    context = build_context(spec, available_context)
    prompt, truncated = _evaluator_prompt(spec, context)
    ledger.charge(spec, 1)                 # before the call: a limit charged after is a report
    reply = model.ask(prompt)
    m: Mapping[str, Any] = reply if isinstance(reply, Mapping) else {}
    score = _normalised_score(spec, m.get("score"))
    reason = str(m.get("reason", ""))[:800]
    concerns = ([] if score is not None else
                [f"{spec.evaluator_id}: no usable score on the declared {spec.scale} scale"])
    if truncated:
        # Carried onto the verdict, not only into the prompt: a score formed on a partial
        # packet must not be readable downstream as a complete look at the case.
        concerns.append(f"{spec.evaluator_id}: the declared context did not fit in "
                        f"{PACKET_CHAR_BUDGET} characters and was truncated")
    verdict = Verdict(
        dimension=spec.dimension, subject_id=str(subject_id), judge_model=model_id,
        judged_at=_now(),
        lens_readings=(LensReading(spec.evaluator_id, score, reason),),
        score=score, concerns=tuple(concerns),
        incomplete=score is None or truncated)
    rec = ledger.record(JudgeRunRecord(
        evaluator_id=spec.evaluator_id, dimension=spec.dimension, cost_class=spec.cost_class,
        subject_id=str(subject_id), judge_model=model_id, judged_at=verdict.judged_at,
        n_model_calls=1, cost_usd=ledger.price[spec.cost_class],
        context_injected=spec.context, tools_granted=spec.tool_names))
    return verdict, rec


def certify_evaluator(spec: EvaluatorSpec, *, run_case, pass_at_or_above: float,
                      fail_at_or_below: float) -> dict:
    """EXECUTE the declared cases. Enforcement 4 with the pipeline actually run.

    Listing case ids is a promise; running them is the check. Both thresholds are required
    and no default exists — where the line sits is a property of this evaluator's scale and
    its purpose, and a default here would be the same magic number the rest of this repo
    refuses. Three ways to fail:

      * a must_pass case scored below the line — the evaluator rejects good work;
      * a must_fail case NOT rejected — the evaluator cannot fail, which is the defect;
      * every case scored identically — decorative. A constant scorer satisfies any
        threshold pair you pick if you only check the two ends separately, and it is exactly
        what a clean system looks like from the outside.
    """
    if pass_at_or_above is None or fail_at_or_below is None:
        raise ValueError("pass_at_or_above and fail_at_or_below are both required; no "
                         "default line exists for an evaluator's own scale")
    hi, lo = float(pass_at_or_above), float(fail_at_or_below)
    if not (SCORE_MIN <= lo < hi <= SCORE_MAX):
        raise ValueError(f"need {SCORE_MIN} <= fail_at_or_below < pass_at_or_above <= "
                         f"{SCORE_MAX}; got {lo} and {hi}. Overlapping lines let one score "
                         f"satisfy both, which certifies a constant scorer.")
    rows, problems = [], []
    for case_id, expect in ([(c, "pass") for c in spec.must_pass] +
                            [(c, "fail") for c in spec.must_fail]):
        v = run_case(case_id)
        # A Verdict, not a bare number: certification has to exercise the path that will run
        # in production, stamp and all. A float here would let a case "pass" without the
        # evaluator ever having been assembled, prompted or charged.
        if not isinstance(v, Verdict):
            raise TypeError(f"run_case({case_id!r}) must return a Verdict, got "
                            f"{type(v).__name__}")
        s = v.score
        rows.append({"case_id": case_id, "expected": expect, "score": s})
        if s is None:
            problems.append(f"{case_id}: no usable score, so the case decided nothing")
        elif expect == "pass" and s < hi:
            problems.append(f"{case_id}: must_pass scored {s} < {hi}")
        elif expect == "fail" and s > lo:
            problems.append(f"{case_id}: must_fail scored {s} > {lo} — this evaluator did not "
                            f"reject the case it declared it would reject")
    scores = {r["score"] for r in rows}
    if len(scores) == 1:
        problems.append(f"every case scored {scores.pop()!r}: this evaluator separates "
                        f"nothing, and an evaluator that scores everything the same looks "
                        f"exactly like a clean system")
    if problems:
        raise EvaluatorCannotFail(f"{spec.evaluator_id} is not certified: " + "; ".join(problems))
    return {"evaluator_id": spec.evaluator_id, "dimension": spec.dimension,
            "pass_at_or_above": hi, "fail_at_or_below": lo, "cases": rows,
            "evidence_class": EV_JUDGED, "validation_status": NOT_VALIDATED}
