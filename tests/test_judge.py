"""The fence around the judge, tested from the outside as an attacker would.

Three properties, and every test here is one of them:

  1. PRECEDENCE. If code decides a dimension, a model may not. The interesting tests are not
     the happy refusal — they are the ways a caller would try to get around it: a `force=`
     kwarg, a permissive policy object, an evaluator marked disabled, a widened allowlist, a
     differently-cased dimension name, a registry that is missing or broken. Every one must
     end in a refusal, because a fence with one gap is a fence nobody has to climb.

  2. ISOLATION. The blinded dimensions must not see the key, enforced in the type rather
     than in a docstring. The test that matters is the leak that actually happens: nobody
     passes `answer_key=`, they nest the truth three levels down in an artifacts dict
     assembled somewhere else, so the scan is recursive and the prompt is inspected for the
     sentinel afterwards.

  3. NO GATING. A judged verdict may rank, screen and flag. It may not accept, adopt or
     validate, it has no boolean, and it has no field a caller could read as a decision.

No model is called anywhere in this file: `StubJudge` records prompts and replays scripted
JSON, which is also what makes the prompt-inspection tests possible.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from acr.evaluation import evals as E
from acr.evaluation.judge import (
    CONTEXT_VARIABLES,
    COST_CLASSES,
    DECISION_FIELD_NAMES,
    DIM_BAD_CASE_TRIAGE,
    DIM_EVIDENCE_SUPPORT_JUDGED,
    DIM_L5_EXPLANATION_QUALITY,
    DIM_TRAJECTORY_QUALITY,
    EV_DETERMINISTIC,
    EV_JUDGED,
    EVALUATOR_TOOLS,
    FORBIDDEN_USES,
    JUDGEABLE_DIMENSIONS,
    LENSES,
    NOT_VALIDATED,
    PERMITTED_USES,
    AnswerKeyLeak,
    BlindPacket,
    DeterministicEvaluatorExists,
    DimensionNotJudgeable,
    EvaluatorCannotFail,
    EvaluatorSpecInvalid,
    JudgeBudgetExceeded,
    JudgeCannotGate,
    JudgeLedger,
    JudgeRefusal,
    Measurement,
    MixedEvidence,
    RegistryUnavailable,
    ToolScopeViolation,
    Verdict,
    aggregate,
    apply_verdict,
    blind_packet,
    build_context,
    certify_evaluator,
    combine_explicitly,
    deterministic_measurement,
    judge,
    keyed_packet,
    load_evaluator,
    load_evaluators,
    parse_evaluator,
    rank,
    run_evaluator,
    scoped_tool_broker,
    screen_for_human,
)

# A truth value that must never reach a blinded judge's prompt. Not a patient id, not note
# text — a synthetic sentinel, so the assertion is exact.
SENTINEL = "SENTINEL_TRUTH_C349_DO_NOT_SHOW"


# ------------------------------------------------------------------------------- doubles
class Registry:
    """The precedence registry shape `judge()` requires of `acr.evaluation.evals`.

    The sibling module had not landed when this was written; the query it must answer is
    `deterministic_evaluator_for(dimension) -> evaluator | None`, and `judge()` also accepts
    `evaluator_for` or a plain Mapping.
    """

    def __init__(self, **evaluators):
        self._by_dim = dict(evaluators)

    def deterministic_evaluator_for(self, dimension):
        return self._by_dim.get(dimension)


class StubJudge:
    """The whole model seam. Records prompts; never opens a socket."""

    def __init__(self, reply=None, model_id="stub-judge/v0"):
        self.model_id = model_id
        self.reply = reply if reply is not None else {"score": 0.8, "observation": "fine",
                                                      "concerns": []}
        self.prompts: list[str] = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.reply(prompt) if callable(self.reply) else self.reply


EMPTY = Registry()          # nothing is deterministically evaluated
TRACE = ({"seq": 0, "kind": "tool", "tool": "search_notes", "args": {"q": "histology"}},
         {"seq": 1, "kind": "reflect", "verdict": "CONTINUE", "reason": "one hit only"})


def _verdict(score, subject="P-synthetic-1", dim=DIM_TRAJECTORY_QUALITY):
    stub = StubJudge({"score": score, "observation": "o", "concerns": ["c"]})
    return judge(dim, blind_packet(TRACE, subject_id=subject), registry=EMPTY, model=stub)


# ============================================================ 1. precedence and refusal
@pytest.mark.parametrize("dim", JUDGEABLE_DIMENSIONS)
def test_the_three_no_ground_truth_dimensions_are_judgeable(dim):
    packet = (keyed_packet(TRACE, answer_key={"histology": "8070"})
              if dim == DIM_BAD_CASE_TRIAGE else blind_packet(TRACE))
    v = judge(dim, packet, registry=EMPTY, model=StubJudge())
    assert v.evidence_class == EV_JUDGED and v.score == pytest.approx(0.8)


def test_a_dimension_with_a_deterministic_evaluator_is_refused():
    """L4 concordance is a rule engine. Asking a model to re-score it is the whole hazard."""
    reg = Registry(guideline_concordance="acr.contract.concordance.score_guideline")
    with pytest.raises(DeterministicEvaluatorExists, match="concordance"):
        judge("guideline_concordance", blind_packet(TRACE), registry=reg, model=StubJudge())


def test_refusal_wins_even_for_a_dimension_that_is_also_on_the_judgeable_list():
    """Precedence is checked before the allowlist, so overlap resolves toward the code."""
    reg = Registry(**{DIM_TRAJECTORY_QUALITY: "some.evaluator"})
    with pytest.raises(DeterministicEvaluatorExists):
        judge(DIM_TRAJECTORY_QUALITY, blind_packet(TRACE), registry=reg, model=StubJudge())


def test_a_dimension_that_merely_lacks_an_evaluator_is_still_not_judgeable():
    with pytest.raises(DimensionNotJudgeable, match="write the evaluator"):
        judge("histology_accuracy", blind_packet(TRACE), registry=EMPTY, model=StubJudge())


def test_no_model_is_called_when_the_judge_refuses():
    """A refusal that still spent a completion is a refusal that costs money to enforce."""
    stub = StubJudge()
    for reg, dim, exc in ((Registry(x="e"), "x", DeterministicEvaluatorExists),
                          (EMPTY, "unknown_dim", DimensionNotJudgeable),
                          (None, DIM_TRAJECTORY_QUALITY, RegistryUnavailable)):
        with pytest.raises(exc):
            judge(dim, blind_packet(TRACE), registry=reg, model=stub)
    assert stub.prompts == []


# ------------------------------------------------------- every override I could think of
def test_override_by_keyword_force():
    with pytest.raises(TypeError):
        judge("x", blind_packet(), registry=Registry(x="e"), model=StubJudge(), force=True)


@pytest.mark.parametrize("kw", ["allow_judged_override", "policy", "strict", "ignore_registry",
                                "evidence_class", "gate"])
def test_override_by_any_other_keyword(kw):
    """There is no kwargs bag to smuggle a policy through; each of these is a TypeError."""
    with pytest.raises(TypeError):
        judge("x", blind_packet(), registry=Registry(x="e"), model=StubJudge(), **{kw: True})


def test_override_by_a_permissive_registry_object():
    """A registry advertising `allow_judge` is ignored: only the lookup result is read."""
    class Permissive(Registry):
        allow_judge = True
        judge_overrides_deterministic = True

    with pytest.raises(DeterministicEvaluatorExists):
        judge("x", blind_packet(), registry=Permissive(x="e"), model=StubJudge())


def test_override_by_marking_the_evaluator_disabled():
    """Existence forbids judging, not enabledness — otherwise disabling a check silently
    hands the dimension to a model, which is the opposite of what disabling should mean."""
    class Disabled:
        enabled = False
        disabled = True
        def __repr__(self): return "disabled-evaluator"

    with pytest.raises(DeterministicEvaluatorExists):
        judge("x", blind_packet(), registry=Registry(x=Disabled()), model=StubJudge())


def test_override_by_widening_the_module_allowlist(monkeypatch):
    """A caller who appends to JUDGEABLE_DIMENSIONS still cannot judge a decided dimension."""
    monkeypatch.setattr("acr.evaluation.judge.JUDGEABLE_DIMENSIONS",
                        JUDGEABLE_DIMENSIONS + ("guideline_concordance",))
    with pytest.raises(DeterministicEvaluatorExists):
        judge("guideline_concordance", blind_packet(),
              registry=Registry(guideline_concordance="e"), model=StubJudge())


@pytest.mark.parametrize("spelling", ["  Guideline_Concordance ", "GUIDELINE_CONCORDANCE",
                                      "guideline_concordance\n"])
def test_override_by_casing_or_padding_the_dimension_name(spelling):
    """Normalise before the lookup, or the fence is bypassed by pressing shift."""
    with pytest.raises(DeterministicEvaluatorExists):
        judge(spelling, blind_packet(), registry=Registry(guideline_concordance="e"),
              model=StubJudge())


def test_override_by_omitting_the_registry():
    with pytest.raises(TypeError):
        judge(DIM_TRAJECTORY_QUALITY, blind_packet(), model=StubJudge())


@pytest.mark.parametrize("bad", [None, object(), "not-a-registry"])
def test_unknown_or_broken_registry_fails_closed(bad):
    """Absent precedence information is not permission. `object()` implements no query."""
    with pytest.raises(RegistryUnavailable):
        judge(DIM_TRAJECTORY_QUALITY, blind_packet(), registry=bad, model=StubJudge())


def test_a_registry_that_raises_fails_closed():
    class Broken:
        def deterministic_evaluator_for(self, dimension):
            raise RuntimeError("registry backend down")

    with pytest.raises(RegistryUnavailable, match="raised"):
        judge(DIM_TRAJECTORY_QUALITY, blind_packet(), registry=Broken(), model=StubJudge())


def test_a_mapping_and_an_alternate_method_name_both_satisfy_the_protocol():
    """Named so the sibling module can land with either shape without a change here."""
    class Alt:
        def evaluator_for(self, dimension): return "e" if dimension == "x" else None

    for reg in ({"x": "e"}, Alt()):
        with pytest.raises(DeterministicEvaluatorExists):
            judge("x", blind_packet(), registry=reg, model=StubJudge())
        judge(DIM_TRAJECTORY_QUALITY, blind_packet(TRACE), registry=reg, model=StubJudge())


def test_relabelling_a_judged_number_as_deterministic_is_refused():
    """The other side of the fence: laundering an opinion into the deterministic column."""
    with pytest.raises(DeterministicEvaluatorExists, match="cannot be stamped"):
        deterministic_measurement(DIM_TRAJECTORY_QUALITY, 0.9, registry=EMPTY)
    m = deterministic_measurement("x", 0.9, registry=Registry(x="e"))
    assert m.evidence_class == EV_DETERMINISTIC


# ================================================================== 2. answer-key isolation
def test_a_blinded_dimension_refuses_a_keyed_packet():
    for dim in (DIM_L5_EXPLANATION_QUALITY, DIM_TRAJECTORY_QUALITY):
        with pytest.raises(AnswerKeyLeak, match="blind"):
            judge(dim, keyed_packet(TRACE, answer_key={"histology": SENTINEL}),
                  registry=EMPTY, model=StubJudge())


def test_the_blind_packet_type_has_nowhere_to_put_a_key():
    """Enforced in the signature: there is no field, and the dataclass is frozen."""
    assert "answer_key" not in {f.name for f in dataclasses.fields(BlindPacket)}
    with pytest.raises(TypeError):
        BlindPacket(trace=(), artifacts={}, answer_key={"histology": SENTINEL})
    with pytest.raises(dataclasses.FrozenInstanceError):
        blind_packet().answer_key = {"histology": SENTINEL}


@pytest.mark.parametrize("artifacts", [
    {"ground_truth": {"histology": SENTINEL}},
    {"case": {"registry": {"truth": SENTINEL}}},
    {"cases": [{"id": "c1", "expected_value": SENTINEL}]},
    {"Answer_Key": {"histology": SENTINEL}},
])
def test_key_material_nested_in_artifacts_is_caught(artifacts):
    """The leak that happens in practice is nested and unintentional, so the scan recurses."""
    with pytest.raises(AnswerKeyLeak):
        blind_packet(TRACE, artifacts=artifacts)


def test_the_scan_runs_at_judge_time_not_only_in_the_factory():
    """BlindPacket can be built directly, so the factory is a convenience, not the fence."""
    smuggled = BlindPacket(trace=(), artifacts={"case": {"truth": SENTINEL}})
    with pytest.raises(AnswerKeyLeak):
        judge(DIM_TRAJECTORY_QUALITY, smuggled, registry=EMPTY, model=StubJudge())


def test_no_key_material_reaches_the_prompt_of_a_blinded_judge():
    """The end-to-end property the type discipline exists to buy."""
    stub = StubJudge()
    judge(DIM_L5_EXPLANATION_QUALITY,
          blind_packet(TRACE, artifacts={"scaffold": {"causes": ["B_DOCUMENTATION_GAP"]}}),
          registry=EMPTY, model=stub)
    assert stub.prompts and not any(SENTINEL in p for p in stub.prompts)


def test_a_packet_subclass_cannot_ride_an_extra_field_into_the_prompt():
    """`_render` is an allowlist of named fields, so a bolted-on attribute is not serialised."""
    @dataclasses.dataclass(frozen=True)
    class Sneaky(BlindPacket):
        smuggled: str = SENTINEL

    stub = StubJudge()
    judge(DIM_TRAJECTORY_QUALITY, Sneaky(trace=TRACE), registry=EMPTY, model=stub)
    assert not any(SENTINEL in p for p in stub.prompts)


def test_triage_is_the_one_dimension_allowed_the_key_and_actually_gets_it():
    """A bad-case pool is bad because a deterministic evaluator disagreed with the key;
    hiding the disagreement would leave nothing to sort."""
    stub = StubJudge()
    judge(DIM_BAD_CASE_TRIAGE, keyed_packet(TRACE, answer_key={"histology": SENTINEL}),
          registry=EMPTY, model=stub)
    assert all(SENTINEL in p for p in stub.prompts)


# ===================================================================== 3. lenses and stamping
@pytest.mark.parametrize("dim", JUDGEABLE_DIMENSIONS)
def test_each_dimension_asks_distinct_questions_rather_than_one_question_repeatedly(dim):
    lenses = LENSES[dim]
    assert len(lenses) >= 3
    assert len({ln.question for ln in lenses}) == len(lenses)
    assert len({ln.catches for ln in lenses}) == len(lenses)


def test_one_model_call_per_lens_and_each_carries_its_own_question():
    stub = StubJudge()
    v = judge(DIM_TRAJECTORY_QUALITY, blind_packet(TRACE), registry=EMPTY, model=stub)
    lenses = LENSES[DIM_TRAJECTORY_QUALITY]
    assert len(stub.prompts) == len(lenses) == len(v.lens_readings)
    for ln, prompt in zip(lenses, stub.prompts):
        assert ln.question in prompt


def test_every_verdict_records_the_judge_model_and_the_date():
    """A judged number is conditioned on the model that produced it, and models change."""
    v = _verdict(0.5)
    assert v.judge_model == "stub-judge/v0"
    assert v.judged_at.startswith("20") and "T" in v.judged_at
    assert v.to_dict()["judge_model"] == "stub-judge/v0"


def test_a_model_that_cannot_name_itself_is_refused():
    with pytest.raises(ValueError, match="model_id"):
        judge(DIM_TRAJECTORY_QUALITY, blind_packet(TRACE), registry=EMPTY,
              model=StubJudge(model_id=""))


@pytest.mark.parametrize("reply", [{"observation": "no score"}, {"score": "high"},
                                   {"score": 7.0}, "not json at all"])
def test_an_unusable_lens_reply_is_none_and_never_a_zero(reply):
    """A fabricated zero would sort the case to the front of the human queue as if the judge
    had found something, and would drag any mean computed over the lenses."""
    v = judge(DIM_TRAJECTORY_QUALITY, blind_packet(TRACE), registry=EMPTY,
              model=StubJudge(reply))
    assert all(r.score is None for r in v.lens_readings)
    assert v.score is None and v.incomplete is True


def test_a_verdict_is_frozen_so_its_stamp_cannot_be_edited():
    v = _verdict(0.4)
    for fieldname, value in (("evidence_class", EV_DETERMINISTIC), ("score", 1.0),
                             ("judge_model", "something-else")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(v, fieldname, value)


def test_a_judged_number_cannot_be_silently_averaged_into_a_deterministic_one():
    det = deterministic_measurement("histology_accuracy", 0.9, registry=Registry(
        histology_accuracy="acr.contract.answer_checks"))
    jud = Measurement.from_verdict(_verdict(0.4))
    with pytest.raises(MixedEvidence, match="combine_explicitly"):
        aggregate([det, jud])
    assert aggregate([det])["evidence_class"] == EV_DETERMINISTIC
    assert aggregate([jud])["mean"] == pytest.approx(0.4)


def test_a_combined_number_must_be_asked_for_and_keeps_the_split_visible():
    det = deterministic_measurement("histology_accuracy", 1.0,
                                    registry=Registry(histology_accuracy="e"))
    jud = Measurement.from_verdict(_verdict(0.0))
    with pytest.raises(TypeError):
        combine_explicitly([det, jud])                      # weight is not optional
    with pytest.raises(ValueError):
        combine_explicitly([det, jud], judged_weight=1.5)
    out = combine_explicitly([det, jud], judged_weight=0.25)
    assert out["combined"] == pytest.approx(0.75)
    assert out["deterministic"]["mean"] == 1.0 and out["judged"]["mean"] == 0.0
    assert out["judged"]["sources"] and out["validation_status"] == NOT_VALIDATED


def test_an_unstamped_measurement_cannot_enter_a_combination():
    with pytest.raises(MixedEvidence, match="unknown evidence class"):
        combine_explicitly([Measurement("d", 0.5, "")], judged_weight=0.5)


# ======================================================== 4. a verdict cannot gate anything
def test_a_verdict_has_no_field_a_caller_could_read_as_a_decision():
    """`if verdict.passed` must not be writable, because there is no such field to read."""
    assert not {f.name for f in dataclasses.fields(Verdict)} & set(DECISION_FIELD_NAMES)


def test_a_verdict_has_no_truth_value():
    """`if verdict:` is a gate wearing the costume of an if-statement."""
    v = _verdict(0.9)
    with pytest.raises(JudgeCannotGate, match="no truth value"):
        bool(v)
    with pytest.raises(JudgeCannotGate):
        if v:                                # the whole point: this must not be writable
            pass


@pytest.mark.parametrize("use", FORBIDDEN_USES)
def test_every_decision_use_is_refused(use):
    with pytest.raises(JudgeCannotGate, match="decision"):
        apply_verdict(_verdict(0.95), use)


@pytest.mark.parametrize("use", ["approve", "auto_adopt", "", "score_and_merge"])
def test_an_unrecognised_use_fails_closed(use):
    with pytest.raises(JudgeCannotGate):
        apply_verdict(_verdict(0.95), use)


@pytest.mark.parametrize("use", PERMITTED_USES)
def test_the_permitted_uses_all_end_at_a_human(use):
    out = apply_verdict(_verdict(0.2), use)
    assert out["disposition"] == "FOR_HUMAN_REVIEW"
    assert out["validation_status"] == NOT_VALIDATED
    assert out["evidence_class"] == EV_JUDGED


def test_a_high_score_adopts_nothing_and_validates_nothing():
    """The strongest possible verdict still buys no standing anywhere."""
    v = _verdict(1.0)
    assert v.to_dict()["validation_status"] == NOT_VALIDATED
    assert "not a substitute for the human" in v.notice
    for use in ("ADOPT", "VALIDATE", "ACCEPT"):
        with pytest.raises(JudgeCannotGate):
            apply_verdict(v, use)


def test_screening_orders_a_queue_and_hides_nothing():
    verdicts = [_verdict(s, subject=f"S{i}") for i, s in enumerate((0.9, 0.1, 0.3, 0.2))]
    out = screen_for_human(verdicts, flag_at_or_below=0.5, queue_size=2)
    assert [q["subject_id"] for q in out["queue"]] == ["S1", "S3"]
    assert out["n_judged"] == 4 and out["n_below_cutoff"] == 3 and out["n_not_shown"] == 1
    assert all(q["disposition"] == "FOR_HUMAN_REVIEW" for q in out["queue"])
    assert out["validation_status"] == NOT_VALIDATED


def test_screening_thresholds_are_required_and_have_no_defaults():
    """No magic numbers: whoever sets the human's cutoff says so in the diff."""
    verdicts = [_verdict(0.4)]
    with pytest.raises(TypeError):
        screen_for_human(verdicts, queue_size=1)
    with pytest.raises(TypeError):
        screen_for_human(verdicts, flag_at_or_below=0.5)
    with pytest.raises(ValueError):
        screen_for_human(verdicts, flag_at_or_below=2.0, queue_size=1)
    with pytest.raises(ValueError):
        screen_for_human(verdicts, flag_at_or_below=0.5, queue_size=0)
    with pytest.raises(ValueError):
        rank(verdicts, worst_first=None)


def test_a_case_the_judge_could_not_read_goes_to_the_human_rather_than_being_dropped():
    """Fail toward the person. 'The judge produced nothing' is a reason to look, not to skip."""
    unscored = judge(DIM_L5_EXPLANATION_QUALITY, blind_packet(TRACE, subject_id="S-broken"),
                     registry=EMPTY, model=StubJudge({"observation": "unparseable"}))
    good = _verdict(0.95, subject="S-fine", dim=DIM_L5_EXPLANATION_QUALITY)
    out = screen_for_human([good, unscored], flag_at_or_below=0.5, queue_size=5)
    assert [q["subject_id"] for q in out["queue"]] == ["S-broken"]
    assert out["queue"][0]["incomplete"] is True
    assert rank([good, unscored], worst_first=True)[0].subject_id == "S-broken"
    with pytest.raises(ValueError, match="no measurement"):
        Measurement.from_verdict(unscored)


# ============================================== 5. THE SEAM: judge.py vs the real registry
# These are the tests that were missing. Every test above uses the `Registry` double, which
# is why both modules could be internally correct while sharing zero dimension names: the
# judge advertised three dimensions, `acr.evaluation.evals` had never heard of any of them, and every
# call the judge claimed to support raised UnknownDimension. It failed closed, so it was safe
# and useless. Nothing here uses a double.
ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = ROOT / "evaluators"
GATE = E.precedence_gate()          # the REAL registry, in the shape judge() requires
PRICES = {"trace_only": 0.01, "reads_documents": 0.25, "reruns_searches": 0.40}


def _ledger(max_calls=50, max_cost_usd=20.0):
    return JudgeLedger(max_calls=max_calls, max_cost_usd=max_cost_usd,
                       cost_per_call_usd=PRICES)


def test_every_dimension_the_judge_advertises_is_known_to_the_registry():
    """THE SEAM TEST. Fails if this module advertises a dimension `acr.evaluation.evals` cannot rule on.

    Same shape as the mismatch caught earlier between two other modules, where one read a key
    the other never wrote: two agents, two correct halves, nobody checking across. A judge
    whose entire advertised surface is refused by the registry is not a safe judge, it is a
    dead one, and only a test that holds both modules at once can see it.
    """
    problems = E.unknown_dimensions(JUDGEABLE_DIMENSIONS)
    assert problems == {}, (
        f"acr.evaluation.judge advertises {sorted(problems)} which acr.evaluation.evals.REGISTRY will not permit. "
        f"Reconcile the names in one namespace — the registry — not in two lists: {problems}")


def test_the_real_registry_answers_the_protocol_the_judge_requires():
    """Not the double: `acr.evaluation.evals.PrecedenceGate` itself, queried the way `judge()` queries."""
    assert isinstance(GATE, E.PrecedenceGate)
    for dim in JUDGEABLE_DIMENSIONS:
        assert GATE.deterministic_evaluator_for(dim) is None, dim
    assert GATE.deterministic_evaluator_for("task_completion") == "acr.review.answer_gate.check_gate"


@pytest.mark.parametrize("dim", JUDGEABLE_DIMENSIONS)
def test_the_judge_actually_runs_against_the_real_registry(dim):
    """The test that would have failed before: every advertised dimension, real registry."""
    packet = (keyed_packet(TRACE, answer_key={"histology": "8070"})
              if dim == DIM_BAD_CASE_TRIAGE else blind_packet(TRACE))
    v = judge(dim, packet, registry=GATE, model=StubJudge())
    assert v.evidence_class == EV_JUDGED and v.score == pytest.approx(0.8)


def test_a_dimension_the_real_registry_never_heard_of_fails_closed():
    """An unregistered name is 'nobody has decided', which is not 'a judge may proceed'."""
    with pytest.raises(RegistryUnavailable, match="precedence registry"):
        judge("trajectory_quality_v2", blind_packet(TRACE), registry=GATE, model=StubJudge())


def test_the_deterministic_half_is_refused_by_the_real_registry():
    with pytest.raises(DeterministicEvaluatorExists, match="detect_resource_band"):
        judge("step_efficiency.deterministic", blind_packet(TRACE), registry=GATE,
              model=StubJudge())


def test_naming_a_split_parent_is_refused_rather_than_answered():
    """Either whole-dimension answer is wrong, so the registry refuses to give one."""
    with pytest.raises(RegistryUnavailable, match="split"):
        judge("evidence_support", blind_packet(TRACE), registry=GATE, model=StubJudge())


# ==================================== 6. AN EVALUATOR IS A SPEC FILE — the five refusals
def _yaml(**over) -> dict:
    """A minimal loadable evaluator. Each test below breaks exactly one field."""
    base = {
        "evaluator_id": "synthetic-evaluator",
        "dimension": DIM_TRAJECTORY_QUALITY,
        "cost_class": "trace_only",
        "prompt": {"role": "You screen one trajectory for a human.",
                   "dimensions": ["was the read order sensible"],
                   "scale": "3-point",
                   "checklist": ["list the steps", "name the first unreasonable one"]},
        "context": ["trace"],
        "tools": [],
        "output": {"score": "required", "reason": "required", "cot": "optional"},
        "must_pass": ["SYN-CASE-good"],
        "must_fail": ["SYN-CASE-bad"],
    }
    base.update(over)
    return base


def _spec(**over):
    return parse_evaluator(_yaml(**over), registry=GATE, source="synthetic")


# --- the shipped files ---------------------------------------------------------------
def test_the_shipped_evaluators_load_against_the_real_registry():
    specs = load_evaluators(EVALUATORS, registry=GATE)
    assert len(specs) >= 2
    for s in specs.values():
        assert s.cost_class in COST_CLASSES and s.must_pass and s.must_fail
        assert set(s.context) <= set(CONTEXT_VARIABLES)


def test_one_shipped_evaluator_opens_the_documents_the_agent_did_not_read():
    """The capability that justifies an agent-judge at all, and no deterministic check can
    pose it: is there anything in what was never opened that contradicts the citation?"""
    specs = load_evaluators(EVALUATORS, registry=GATE)
    flagship = [s for s in specs.values() if "documents_not_read" in s.context]
    assert flagship, "no shipped evaluator reads what the agent did not read"
    s = flagship[0]
    assert s.dimension == DIM_EVIDENCE_SUPPORT_JUDGED
    assert s.cost_class != "trace_only" and s.tool_names
    assert all(t.scope == "patient_under_review" for t in s.tools)


def test_a_new_evaluation_need_is_a_new_file_and_not_a_code_change():
    """The whole point of enforcement: a YAML nobody wrote code for loads and runs."""
    spec = _spec(evaluator_id="brand-new-question", dimension="step_efficiency.judged")
    ledger = _ledger()
    v, rec = run_evaluator(spec, {"trace": list(TRACE)}, registry=GATE,
                           model=StubJudge({"score": 3, "reason": "settled by step 2"}),
                           ledger=ledger, subject_id="P-synthetic-1")
    assert v.score == pytest.approx(1.0) and v.evidence_class == EV_JUDGED
    assert rec.evaluator_id == "brand-new-question" and rec.cost_class == "trace_only"


# --- 1. the precedence fence, at load, PER SUB-QUESTION --------------------------------
@pytest.mark.parametrize("dim,expect", [("correctness", "evals.score"),
                                        ("task_completion", "answer_gate.check_gate"),
                                        ("hallucination", "record_evidence")])
def test_an_evaluator_for_a_deterministic_dimension_is_refused_naming_the_method(dim, expect):
    with pytest.raises(DeterministicEvaluatorExists, match=expect):
        _spec(dimension=dim)


def test_the_fence_is_per_sub_question_not_per_dimension():
    """A per-dimension fence would have to kill one of these two, and is wrong either way."""
    assert _spec(dimension="evidence_support.judged").dimension == "evidence_support.judged"
    with pytest.raises(DeterministicEvaluatorExists, match="admissibility_for_citations"):
        _spec(dimension="evidence_support.deterministic")
    assert _spec(dimension="step_efficiency.judged").dimension == "step_efficiency.judged"
    with pytest.raises(DeterministicEvaluatorExists, match="detect_resource_band"):
        _spec(dimension="step_efficiency.deterministic")


def test_declaring_the_parent_of_a_split_dimension_does_not_load():
    with pytest.raises(JudgeRefusal, match="split"):
        _spec(dimension="evidence_support")


def test_an_unregistered_dimension_does_not_load():
    with pytest.raises(JudgeRefusal):
        _spec(dimension="vibes")


def test_the_fence_is_re_checked_at_run_time_and_not_trusted_from_load():
    """A spec outlives the registry it loaded against; 'it was allowed then' is not a check."""
    spec = _spec(dimension=DIM_TRAJECTORY_QUALITY)
    later = Registry(**{DIM_TRAJECTORY_QUALITY: "acr.evaluation.evals.some_new_exact_check"})
    stub = StubJudge()
    with pytest.raises(DeterministicEvaluatorExists):
        run_evaluator(spec, {"trace": []}, registry=later, model=stub, ledger=_ledger(),
                      subject_id="P-1")
    assert stub.prompts == []                       # refused before a token was spent


# --- 2. context is declared, and nothing else is injected ------------------------------
def test_the_answer_key_is_refused_on_a_dimension_it_would_contaminate():
    with pytest.raises(AnswerKeyLeak, match="expected_output"):
        _spec(dimension=DIM_L5_EXPLANATION_QUALITY,
              context=["trace", "expected_output"])


def test_the_answer_key_is_permitted_only_where_the_disagreement_is_the_material():
    spec = _spec(dimension=DIM_BAD_CASE_TRIAGE, context=["trace", "expected_output"])
    assert spec.sees_answer_key
    # ...and the same line in the shipped L5 file would be a load error, which is the
    # property: withholding the key is checkable by READING the file.
    l5 = load_evaluator(EVALUATORS / "l5-explanation-screen.yaml", registry=GATE)
    assert "expected_output" not in l5.context


def test_an_undeclared_context_variable_cannot_be_requested():
    with pytest.raises(EvaluatorSpecInvalid, match="not injectable"):
        _spec(context=["trace", "registry_truth"])


def test_only_declared_context_is_injected_even_when_more_is_available():
    spec = _spec(context=["trace"])
    ctx = build_context(spec, {"trace": [1], "expected_output": SENTINEL,
                               "coverage_ledger": {"n": 3}})
    assert ctx == {"trace": [1]}
    stub = StubJudge()
    run_evaluator(spec, {"trace": [1], "expected_output": SENTINEL}, registry=GATE,
                  model=stub, ledger=_ledger(), subject_id="P-1")
    assert stub.prompts and not any(SENTINEL in p for p in stub.prompts)


def test_a_declared_context_the_harness_cannot_supply_refuses_rather_than_judging_blank():
    """An empty section is answered anyway; a missing one has to stop the run."""
    spec = _spec(dimension=DIM_EVIDENCE_SUPPORT_JUDGED,
                 cost_class="reads_documents",
                 tools=[{"name": "read_document", "scope": "patient_under_review"}],
                 context=["trace", "documents_not_read"])
    with pytest.raises(EvaluatorSpecInvalid, match="documents_not_read"):
        build_context(spec, {"trace": []})


def test_reading_what_was_never_opened_cannot_be_declared_free():
    with pytest.raises(EvaluatorSpecInvalid, match="reads_documents"):
        _spec(dimension=DIM_EVIDENCE_SUPPORT_JUDGED, cost_class="trace_only",
              context=["trace", "documents_not_read"])


# --- 3. tool scope is declared and bounded --------------------------------------------
@pytest.mark.parametrize("scope", ["cohort", "any_patient", "", "all"])
def test_a_tool_scoped_beyond_the_patient_under_review_does_not_load(scope):
    with pytest.raises(ToolScopeViolation, match="patient_under_review"):
        _spec(cost_class="reads_documents",
              tools=[{"name": "read_document", "scope": scope}])


def test_a_tool_without_a_declared_scope_does_not_load():
    with pytest.raises(EvaluatorSpecInvalid, match="unbounded PHI access path"):
        _spec(cost_class="reads_documents", tools=["read_document"])


def test_a_tool_the_judge_may_not_hold_does_not_load():
    with pytest.raises(EvaluatorSpecInvalid, match="not grantable"):
        _spec(cost_class="reads_documents",
              tools=[{"name": "registry_truth", "scope": "patient_under_review"}])
    assert "registry_truth" not in EVALUATOR_TOOLS


def test_a_cost_class_that_understates_or_overstates_access_does_not_load():
    with pytest.raises(EvaluatorSpecInvalid, match="reruns_searches"):
        _spec(cost_class="reads_documents",
              tools=[{"name": "search_notes", "scope": "patient_under_review"}])
    with pytest.raises(EvaluatorSpecInvalid, match="grants no tools"):
        _spec(cost_class="trace_only",
              tools=[{"name": "read_document", "scope": "patient_under_review"}])
    with pytest.raises(EvaluatorSpecInvalid, match="declares no tools"):
        _spec(cost_class="reads_documents", tools=[])


def test_the_declared_scope_is_enforced_on_every_call_and_not_only_in_the_yaml():
    """A declared scope nothing checks is the renderer that reported every element
    unattributed. The auditor gets the cross-patient rule the agent gets."""
    spec = _spec(cost_class="reads_documents",
                 tools=[{"name": "read_document", "scope": "patient_under_review"}])

    class Backend:
        def __init__(self):
            self.opened = []

        def read_document(self, **kw):
            self.opened.append(kw)
            return "note text"

        def search_notes(self, **kw):
            raise AssertionError("ungranted tool reached the backend")

    backend = Backend()
    call = scoped_tool_broker(spec, patient_under_review="SYN0001", backend=backend)
    assert call("read_document", patient_id="SYN0001", note_id="n1") == "note text"
    with pytest.raises(ToolScopeViolation, match="PHI access path"):
        call("read_document", patient_id="SYN0002", note_id="n1")
    with pytest.raises(ToolScopeViolation, match="did not declare"):
        call("search_notes", patient_id="SYN0001", query="histology")
    assert backend.opened == [{"patient_id": "SYN0001", "note_id": "n1"}]
    with pytest.raises(ToolScopeViolation, match="unspecified patient"):
        scoped_tool_broker(spec, patient_under_review="", backend=backend)


# --- 4. a case it must pass AND a case it must FAIL ------------------------------------
@pytest.mark.parametrize("field", ["must_pass", "must_fail"])
def test_an_evaluator_without_both_kinds_of_case_does_not_load(field):
    with pytest.raises(EvaluatorCannotFail, match=field):
        _spec(**{field: []})


def test_a_case_declared_both_pass_and_fail_does_not_load():
    with pytest.raises(EvaluatorCannotFail, match="cannot be wrong"):
        _spec(must_pass=["SYN-CASE-x"], must_fail=["SYN-CASE-x"])


def _runner(spec, scores, ledger):
    """Run the declared cases through the real path, scripted per case id."""
    def run_case(case_id):
        stub = StubJudge(lambda prompt: {"score": scores[case_id], "reason": "scripted"})
        v, _ = run_evaluator(spec, {c: [{"case_id": case_id}] for c in spec.context},
                             registry=GATE, model=stub, ledger=ledger, subject_id=case_id)
        return v
    return run_case


def test_certification_runs_the_cases_rather_than_trusting_the_list():
    spec = _spec()
    out = certify_evaluator(spec, run_case=_runner(spec, {"SYN-CASE-good": 3,
                                                          "SYN-CASE-bad": 1}, _ledger()),
                            pass_at_or_above=0.9, fail_at_or_below=0.1)
    assert [r["score"] for r in out["cases"]] == [1.0, 0.0]
    assert out["validation_status"] == NOT_VALIDATED


def test_an_evaluator_that_does_not_reject_its_must_fail_case_is_refused():
    """The defect this enforcement exists for: a check that cannot fail."""
    spec = _spec()
    with pytest.raises(EvaluatorCannotFail, match="did not reject"):
        certify_evaluator(spec, run_case=_runner(spec, {"SYN-CASE-good": 3,
                                                        "SYN-CASE-bad": 2}, _ledger()),
                          pass_at_or_above=0.9, fail_at_or_below=0.1)


def test_an_evaluator_that_scores_everything_the_same_is_refused():
    """A constant scorer looks exactly like a clean system, and passes any single-ended
    threshold check you write. Certification compares the cases against each other."""
    spec = _spec()
    with pytest.raises(EvaluatorCannotFail, match="separates nothing"):
        certify_evaluator(spec, run_case=_runner(spec, {"SYN-CASE-good": 3,
                                                        "SYN-CASE-bad": 3}, _ledger()),
                          pass_at_or_above=0.5, fail_at_or_below=0.4)


def test_certification_thresholds_are_required_and_cannot_overlap():
    spec = _spec()
    run = _runner(spec, {"SYN-CASE-good": 3, "SYN-CASE-bad": 1}, _ledger())
    with pytest.raises(TypeError):
        certify_evaluator(spec, run_case=run)                       # no default line exists
    with pytest.raises(ValueError, match="constant scorer"):
        certify_evaluator(spec, run_case=run, pass_at_or_above=0.4, fail_at_or_below=0.6)


def test_an_unscorable_case_certifies_nothing():
    spec = _spec()
    with pytest.raises(EvaluatorCannotFail, match="decided nothing"):
        certify_evaluator(spec, run_case=_runner(spec, {"SYN-CASE-good": "n/a",
                                                        "SYN-CASE-bad": 1}, _ledger()),
                          pass_at_or_above=0.9, fail_at_or_below=0.1)


def test_one_undeclared_case_refuses_the_whole_directory(tmp_path):
    """Skip-and-warn would drop exactly the evaluator with the failing case and keep the
    three that pass everything — the defect wearing a clean report."""
    (tmp_path / "decorative-evaluator.yaml").write_text(
        json.dumps({**_yaml(evaluator_id="decorative-evaluator"), "must_fail": []}),
        encoding="utf-8")
    (tmp_path / "fine-evaluator.yaml").write_text(
        json.dumps(_yaml(evaluator_id="fine-evaluator")), encoding="utf-8")
    with pytest.raises(EvaluatorCannotFail):
        load_evaluators(tmp_path, registry=GATE)


def test_the_filename_and_the_evaluator_id_must_agree(tmp_path):
    p = tmp_path / "one-name.yaml"
    p.write_text(json.dumps(_yaml(evaluator_id="another-name")), encoding="utf-8")
    with pytest.raises(EvaluatorSpecInvalid, match="filename stem"):
        load_evaluator(p, registry=GATE)


def test_a_typo_in_a_field_name_refuses_rather_than_declaring_nothing():
    """`contexts:` would otherwise declare no context at all and still run."""
    bad = _yaml()
    bad["contexts"] = bad.pop("context")
    with pytest.raises(EvaluatorSpecInvalid, match="unknown field"):
        parse_evaluator(bad, registry=GATE)


@pytest.mark.parametrize("output", [{"score": "required"},
                                    {"score": "required", "reason": "optional"},
                                    {"score": "optional", "reason": "required"},
                                    {"score": "required", "reason": "required",
                                     "cot": "required"}])
def test_a_score_without_a_required_reason_does_not_load(output):
    with pytest.raises(EvaluatorSpecInvalid):
        _spec(output=output)


def test_a_scale_the_runtime_cannot_normalise_does_not_load():
    with pytest.raises(EvaluatorSpecInvalid, match="scale"):
        _spec(prompt={**_yaml()["prompt"], "scale": "10-point"})


def test_an_off_scale_reply_is_unscored_and_never_zero():
    """A fabricated zero sorts the case to the front of the queue as if something was found."""
    spec = _spec()
    v, _ = run_evaluator(spec, {"trace": []}, registry=GATE,
                         model=StubJudge({"score": 7, "reason": "off scale"}),
                         ledger=_ledger(), subject_id="P-1")
    assert v.score is None and v.incomplete and v.concerns


# --- 5. a judge run is a run: traced, cost-accounted, rate-limited ---------------------
def test_every_judge_run_is_traced_with_its_cost_class_and_model():
    spec = _spec()
    ledger = _ledger()
    run_evaluator(spec, {"trace": []}, registry=GATE, model=StubJudge(), ledger=ledger,
                  subject_id="P-1")
    rep = ledger.report()
    assert rep["n_runs"] == 1 and rep["cost_usd"] == pytest.approx(PRICES["trace_only"])
    row = rep["runs"][0]
    assert row["cost_class"] == "trace_only" and row["judge_model"] == "stub-judge/v0"
    assert row["judged_at"] and row["evidence_class"] == EV_JUDGED
    assert row["context_injected"] == ["trace"] and row["tools_granted"] == []
    assert rep["by_cost_class"]["trace_only"]["n_runs"] == 1


def test_the_call_ceiling_is_charged_before_the_model_is_called():
    """A limit enforced after the spend is a report, not a limit."""
    spec = _spec()
    ledger = _ledger(max_calls=1)
    stub = StubJudge()
    run_evaluator(spec, {"trace": []}, registry=GATE, model=stub, ledger=ledger,
                  subject_id="P-1")
    with pytest.raises(JudgeBudgetExceeded, match="ceiling of 1"):
        run_evaluator(spec, {"trace": []}, registry=GATE, model=stub, ledger=ledger,
                      subject_id="P-2")
    assert len(stub.prompts) == 1


def test_a_fleet_of_document_opening_judges_hits_the_cost_ceiling():
    """cost_class is declared so this cannot cost more than the reviews it audits, quietly."""
    spec = _spec(cost_class="reads_documents",
                 tools=[{"name": "read_document", "scope": "patient_under_review"}],
                 dimension=DIM_EVIDENCE_SUPPORT_JUDGED)
    ledger = _ledger(max_cost_usd=0.30)
    run_evaluator(spec, {"trace": []}, registry=GATE, model=StubJudge(), ledger=ledger,
                  subject_id="P-1")
    with pytest.raises(JudgeBudgetExceeded, match="reads_documents"):
        run_evaluator(spec, {"trace": []}, registry=GATE, model=StubJudge(), ledger=ledger,
                      subject_id="P-2")


@pytest.mark.parametrize("kw", [{"max_calls": 0}, {"max_cost_usd": 0},
                                {"cost_per_call_usd": {"trace_only": 0.01}}])
def test_the_budget_has_no_defaults(kw):
    args = {"max_calls": 10, "max_cost_usd": 1.0, "cost_per_call_usd": PRICES, **kw}
    with pytest.raises(ValueError):
        JudgeLedger(**args)
    with pytest.raises(TypeError):
        JudgeLedger(max_calls=10, max_cost_usd=1.0)                 # prices are not optional


def test_a_context_too_large_to_show_is_marked_incomplete_rather_than_answered_anyway():
    """The worst decorative pass available here: shown half of `documents_not_read`, a judge
    reports 'nothing there contradicts the citation' and it reads as a clean chart."""
    spec = _spec(dimension=DIM_EVIDENCE_SUPPORT_JUDGED, cost_class="reads_documents",
                 tools=[{"name": "read_document", "scope": "patient_under_review"}],
                 context=["trace", "documents_not_read"])
    huge = [{"note_id": f"n{i}", "text": "x" * 200} for i in range(200)]
    stub = StubJudge({"score": 3, "reason": "nothing contradicts it"})
    v, _ = run_evaluator(spec, {"trace": [], "documents_not_read": huge}, registry=GATE,
                         model=stub, ledger=_ledger(), subject_id="P-1")
    assert "TRUNCATED" in stub.prompts[0]
    assert v.incomplete and any("truncated" in c for c in v.concerns)
