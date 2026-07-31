"""§6b — the reflective optimizer over text parameters.

The property under test is never "the optimizer improves the spec". It is that the optimizer
CANNOT do the three things that would make its improvements meaningless:

  * accept a verdict that blames the spec without citing it (gradients pool in the most
    editable prose, and prose is the most editable thing here);
  * turn "this rule is wrong" into an automatic edit (that edits the loss, and the result
    looks exactly like an accuracy gain);
  * validate a revision on the cases it was written from, or on so few cases that "no
    regression" means "we could not have seen one".

Every fixture below is fabricated. No corpus is read, no model is called, and the reflection
seam is a canned stub — a test that reached a model would be spending money to assert a
control-flow property.
"""

# ---------------------------------------------------------------------------------------------
# TESTS REMOVED 2026-07-30, with the rules they specified.
#
# `answer_checks` carried five checks that decided clinical questions by matching word lists
# against the model's own cited quotes. Measured over every trace this project has recorded
# (266 traces, 202 joinable to registry gold, 122 firings):
#
#   not_less_specific        22 fires   22 rejected the registry's own value    0 ever helped
#   nos_requires_search      24 fires   21 rejected the registry's own value    0 ever helped
#   conflict_requires_nos    67 fires   18 rejected the registry's own value   15 "helped",
#                                       all 15 of them the same push to the NOS code
#   origin_not_specimen       2 fires    0                                      0
#   code_matches_cited_text   0 fires    -                                      -
#
# `fit_terms_to_budget` deleted 103 search terms the model had proposed for itself, and on
# CASE009 it deleted `lobe` and `bronchus` while `nos_requires_search` refused the answer for
# never having searched them. The required-keyword gate enforced a list measured at 87.4%
# recall over 276,054 documents.
#
# A test that pins a rule in place is part of the rule, so these went with them:
#   - test_a_mechanical_rule_gets_a_real_number_by_replay
#
# Nothing replaced them here. A wrong clinical value is an instruction-following failure and is
# measured as one. tests/test_answer_checks.py holds what survives: field `format` and
# `allowable_values`, the only check with a positive measured record.
# ---------------------------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from acr.improvement import refine as R

# --------------------------------------------------------------------------- fixtures
#: A miniature spec, standing in for the rendered prose the citation mask checks against.
#: Deliberately short: the mask's job is a substring check, and a long fixture would hide
#: which sentence a test depends on.
SPEC_TEXT = """
Record the ICD-O-3 topography code for the site of ORIGIN, not the site of a biopsy.
Where the record is more specific than the summary line, take the more specific reading.
A cytology report may establish behaviour only when corroborated by tissue.
"""
SPEC_ID = "SYN.400.site"


def case(cid="SYN0001", *, surfaced=True, adjudication=R.ADJUDICATED_KEY_CORRECT,
         subgroup="squamous", **kw) -> R.FailureCase:
    return R.FailureCase(case_id=cid, spec_id=SPEC_ID, field="histology", coded_value="8046",
                         key_value="8070", establishing_evidence_surfaced=surfaced,
                         answer_key_adjudication=adjudication, subgroup=subgroup,
                         invoked_rules=("conflict_rules[0]",), **kw)


def router(v: R.ReflectionVerdict | None, *, cid="SYN0001") -> R.GradientRouter:
    stub = R.StubReflector({cid: v} if v is not None else {})
    return R.GradientRouter({SPEC_ID: SPEC_TEXT}, stub)


def route(v: R.ReflectionVerdict, c: R.FailureCase | None = None) -> R.Routing:
    c = c or case()
    return router(v, cid=c.case_id).route(c)


GAP = R.ReflectionVerdict(
    verdict=R.SPEC_GAP, parameter_id="spec_rules",
    rationale="nothing addresses a hedged diagnosis awaiting stains",
    missing_sentence="A hedged diagnosis such as 'favor squamous' with stains pending is "
                     "coded to the favoured histology.",
    proposed_text="A hedged diagnosis such as 'favor squamous' with stains pending is "
                  "coded to the favoured histology.")

AMBIGUITY = R.ReflectionVerdict(
    verdict=R.SPEC_AMBIGUITY, parameter_id="spec_rules", rationale="two readings",
    quoted_passage="take the more specific reading",
    readings=("more specific among the stated diagnoses",
              "more specific including hedged descriptors such as 'favor'"),
    proposed_text="Where the record is more specific than the summary line, including hedged "
                  "descriptors, take the more specific reading.")

NOT_SPEC = R.ReflectionVerdict(
    verdict=R.NOT_A_SPEC_FIX, parameter_id="skill", rationale="the rule was present, ignored",
    quoted_passage="take the more specific reading",
    proposed_text="Before coding an NOS histology, restate the most specific phrase found.")

ERROR = R.ReflectionVerdict(
    verdict=R.SPEC_ERROR, parameter_id="spec_rules",
    rationale="cytology alone should establish behaviour in this setting",
    quoted_passage="A cytology report may establish behaviour only when corroborated by tissue",
    proposed_text="A cytology report may establish behaviour.")


# =========================================================== 1. the parameter registry
def test_the_six_design_table_parameters_are_registered_with_their_owners():
    """The table is the contract. Who may update each is the only thing that differs between
    parameters, so a wrong policy here is the whole design silently inverted."""
    expected = {
        "keyword_list": R.AUTO_ON_CERTIFICATION,
        "document_type_policy": R.CLINICIAN_SIGNS,
        "skill": R.AUTO_ON_HELDOUT_GAIN,
        "spec_rules": R.CLINICIAN_SIGNS,
        "agent_system_prompt": R.ENGINEER,
        "answer_check_rejection_messages": R.ENGINEER,
    }
    assert set(R.DESIGN_TABLE_PARAMETER_IDS) == set(expected)
    for pid, policy in expected.items():
        assert R.get_parameter(pid).update_policy == policy


def test_the_rejection_messages_are_a_registered_parameter():
    """The row nobody has ever looked at. A run was rejected twice for coding 8046 over
    "favor squamous" and then burned a 400k-token budget without revising; a bad rejection
    message explains that at least as well as a bad spec. Registering it is most of the fix,
    because it makes the text visible as something that can be wrong."""
    p = R.get_parameter("answer_check_rejection_messages")
    assert p.file.endswith("*.yaml") and p.path_within == "answer_checks[*].message"
    assert p.in_objective is False  # the wording is not the target; the check's rule is


def test_registry_is_data_not_scattered_policy():
    """One auditable place, and immutable. A policy that can be mutated at a call site is a
    policy that will be, and the audit then describes a file nobody runs."""
    assert isinstance(R.PARAMETER_REGISTRY, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        R.PARAMETER_REGISTRY[0].update_policy = R.AUTO_ON_CERTIFICATION
    for p in R.PARAMETER_REGISTRY:
        assert p.file and p.path_within and p.kind and p.why


def test_registry_invariants_hold_and_forbid_auto_updates_inside_the_objective():
    R.registry_invariants()
    assert sum(p.in_objective for p in R.PARAMETER_REGISTRY) == 1
    assert R.get_parameter("spec_rules").in_objective is True


def test_unknown_parameter_names_the_vocabulary():
    with pytest.raises(R.UnknownParameterError) as e:
        R.get_parameter("prompt")
    assert "spec_rules" in str(e.value)


# ================================================= 2. routing — the first two cuts
def test_retrieval_failure_never_reaches_the_reflector():
    """Cut 1 is not about text. If the establishing evidence never surfaced, no sentence
    would have helped — and asking a reflection model anyway invites a spec verdict on a
    retrieval failure, which is how §6c's bug becomes a spec edit."""
    def explode(*_a, **_kw):  # a reflector that must not be called
        raise AssertionError("the reflector was consulted on a retrieval failure")

    r = R.GradientRouter({SPEC_ID: SPEC_TEXT}, explode).route(case(surfaced=False))
    assert (r.verdict, r.destination) == (R.RETRIEVAL_FAILURE, R.TO_RETRIEVAL_6C)
    assert r.in_denominator is True and r.parameter_id is None


def test_a_wrong_answer_key_leaves_the_denominator():
    r = route(GAP, case(adjudication=R.ADJUDICATED_KEY_WRONG))
    assert (r.verdict, r.destination) == (R.ANSWER_KEY_WRONG, R.TO_ADJUDICATED_OUT)
    assert r.in_denominator is False


def test_an_unadjudicated_key_is_unresolved_rather_than_routed():
    """Routing before adjudication attributes a registry error to the text, and the text then
    grows a rule to accommodate a wrong answer."""
    r = route(GAP, case(adjudication=R.NOT_ADJUDICATED))
    assert r.destination == R.TO_UNRESOLVED and "adjudicated" in r.rejected_reason


def test_missing_spec_text_raises_rather_than_skipping_the_mask():
    with pytest.raises(R.RefineError, match="citation mask cannot run"):
        R.GradientRouter({}, R.StubReflector({"SYN0001": GAP})).route(case())


# ======================================== 2b. the citation mask — the four paths
def test_spec_gap_with_the_sentence_that_should_exist_is_accepted():
    r = route(GAP)
    assert (r.verdict, r.destination) == (R.SPEC_GAP, R.TO_PROPOSAL)
    assert r.change_class == R.FORM
    assert r.citation["missing_sentence"].startswith("A hedged diagnosis")


def test_spec_ambiguity_needs_the_passage_and_both_readings():
    r = route(AMBIGUITY)
    assert (r.verdict, r.destination) == (R.SPEC_AMBIGUITY, R.TO_PROPOSAL)
    assert r.citation["quoted_passage"] in SPEC_TEXT
    assert len(r.citation["readings"]) == 2


def test_not_a_spec_fix_must_quote_the_rule_that_already_covers_it():
    """The leaf that keeps this honest. Without it every wrong answer becomes a spec edit and
    the spec fills with restatements of rules that were present and ignored."""
    r = route(NOT_SPEC)
    assert (r.verdict, r.destination, r.parameter_id) == (R.NOT_A_SPEC_FIX, R.TO_PROPOSAL, "skill")
    assert r.change_class is None  # not a gradient at the spec, so no FORM/CONTENT


@pytest.mark.parametrize("verdict,why", [
    (R.ReflectionVerdict(R.SPEC_GAP, "spec_rules", "silent"), "missing_sentence"),
    (R.ReflectionVerdict(R.SPEC_GAP, "spec_rules", "silent",
                         missing_sentence="take the more specific reading"), "already present"),
    (R.ReflectionVerdict(R.SPEC_AMBIGUITY, "spec_rules", "unclear",
                         readings=("a", "b")), "quoted_passage"),
    (R.ReflectionVerdict(R.SPEC_AMBIGUITY, "spec_rules", "unclear",
                         quoted_passage="a sentence the spec never contains",
                         readings=("a", "b")), "does not appear"),
    (R.ReflectionVerdict(R.SPEC_AMBIGUITY, "spec_rules", "unclear",
                         quoted_passage="take the more specific reading",
                         readings=("only one",)), "BOTH readings"),
    (R.ReflectionVerdict(R.NOT_A_SPEC_FIX, "skill", "present and ignored"), "quoted_passage"),
    (R.ReflectionVerdict(R.NOT_A_SPEC_FIX, "skill", "present and ignored",
                         quoted_passage="a sentence the spec never contains"), "does not appear"),
    (R.ReflectionVerdict(R.SPEC_ERROR, "spec_rules", "wrong"), "quoted_passage"),
])
def test_an_uncited_verdict_is_rejected_and_returns_unresolved(verdict, why):
    """No quote, no verdict. The router returns unresolved rather than guessing, because a
    plausible uncited revision is the cheapest thing a reflection model can produce."""
    r = route(verdict)
    assert (r.verdict, r.destination) == (R.UNRESOLVED, R.TO_UNRESOLVED)
    assert why in r.rejected_reason
    assert r.change_class is None


def test_rejection_reasons_tell_the_reflector_what_to_do():
    """Rejecting a verdict is the same act as rejecting an answer, and the registry exists
    because unactionable rejections burn budgets. So the mask's messages are held to it."""
    for v in (R.ReflectionVerdict(R.SPEC_GAP, "spec_rules", "silent"),
              R.ReflectionVerdict(R.NOT_A_SPEC_FIX, "skill", "ignored")):
        msg = R.citation_defect(v, SPEC_TEXT)
        assert len(msg.split()) > 8 and re.search(r"\b(Name|Quote)\b", msg)


# ============================================== 3. the spec asymmetry, enforced
def test_form_and_content_are_derived_from_the_verdict_not_supplied():
    assert R.spec_change_class(R.SPEC_GAP) == R.FORM
    assert R.spec_change_class(R.SPEC_AMBIGUITY) == R.FORM
    assert R.spec_change_class(R.SPEC_ERROR) == R.CONTENT
    with pytest.raises(R.RefineError):
        R.spec_change_class(R.NOT_A_SPEC_FIX)


def test_content_escalates_as_a_question_and_can_never_become_an_edit():
    """THE test. "This rule is wrong" can come from the standard or from a clinician; it
    cannot come from the data, because the spec defines what a correct answer is and editing
    it edits the loss. Classifying CONTENT as FORM to get an edit through is the single most
    damaging thing this loop could do — the result looks like an accuracy improvement."""
    r = route(ERROR)
    assert (r.verdict, r.change_class, r.destination) == (
        R.SPEC_ERROR, R.CONTENT, R.TO_CLINICIAN_QUESTION)

    q = R.escalate(r)
    assert q.quoted_passage in SPEC_TEXT
    assert q.evidence["coded_value"] == "8046" and q.evidence["answer_key"] == "8070"
    assert "proposed_text" not in q.to_dict()  # nothing to accept, only something to answer

    # 1. The direct route is closed.
    with pytest.raises(R.ContentEscalationRequired):
        R.Proposal(parameter_id="spec_rules", case_id="SYN0001", verdict=R.SPEC_ERROR,
                   citation={}, proposed_text="A cytology report may establish behaviour.",
                   blast_radius=R.BlastRadius.not_computable("prose"), change_class=R.CONTENT)

    # 2. So is laundering it as FORM-with-no-class, or as an unclassified spec edit.
    for klass in (None, "form", "Form", ""):
        with pytest.raises(R.RefineError):
            R.Proposal(parameter_id="spec_rules", case_id="SYN0001", verdict=R.SPEC_ERROR,
                       citation={}, proposed_text="x",
                       blast_radius=R.BlastRadius.not_computable("prose"), change_class=klass)

    # 3. And the batcher emits a question, never a batch element.
    batches, questions, _ = R.assemble([r], lambda _r: R.BlastRadius.not_computable("prose"))
    assert batches == [] and len(questions) == 1


def test_no_argument_anywhere_can_widen_the_content_door():
    """"It must not be expressible by any argument, flag or policy value." So: no public
    callable in this module takes a parameter that would let a caller force one through."""
    forbidden = re.compile(r"force|override|allow_content|as_form|auto_apply|bypass|unsafe")
    for name, obj in vars(R).items():
        if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", None) != R.__name__:
            continue  # imported helpers are not this module's surface
        if inspect.isclass(obj) and issubclass(obj, Exception):
            continue  # exceptions carry no policy
        for pname in inspect.signature(obj).parameters:
            assert not forbidden.search(pname), f"{name}({pname}) is a door into the objective"
    # And no update policy admits automatic edits to a parameter inside the objective.
    assert R.get_parameter("spec_rules").update_policy == R.CLINICIAN_SIGNS


def test_a_form_proposal_to_the_spec_is_still_only_a_proposal():
    """The spec is clinician-owned, so even the allowed gradient throttles to human speed."""
    p = R.Proposal("spec_rules", "SYN0001", R.SPEC_GAP, {}, "a new sentence",
                   R.BlastRadius.not_computable("prose"), change_class=R.FORM)
    assert p.may_apply_automatically is False
    assert R.Proposal("skill", "SYN0001", R.NOT_A_SPEC_FIX, {}, "a step",
                      R.BlastRadius.not_computable("prose")).may_apply_automatically is True


# ============================================================== 4. blast radius
CHECK = {"field": "histology", "kind": "not_less_specific", "nos_values": ["8046"],
         "contradicted_by": ["favor squamous"], "message": "take the more specific reading"}


def coded(cid, quote, value="8046"):
    return R.CodedCase(cid, {"histology": value}, ({"quote": quote, "supports": "histology"},))


def test_a_keyword_is_priced_by_grep():
    br = R.blast_radius_for("keyword_list", mechanism="keyword", term="Favor Squamous",
                            note_texts=["... favor squamous ...", "adenocarcinoma", ""])
    assert (br.computable, br.n_cases_changed, br.n_cases_examined) == (True, 1, 3)


def test_prose_says_so_in_words_because_an_absent_number_reads_as_zero():
    br = R.blast_radius_for("spec_rules")
    d = br.to_dict()
    assert d["computable"] is False and d["n_cases_changed"] is None
    assert "re-running" in d["basis"]
    assert set(d) == {"computable", "n_cases_changed", "n_cases_examined", "basis"}


def test_every_batch_element_carries_a_blast_radius_field():
    batches, _, _ = R.assemble([route(GAP)], lambda _r: R.BlastRadius.not_computable("prose"))
    for el in batches[0].to_dict()["elements"]:
        assert "blast_radius" in el and el["blast_radius"]["basis"]


# ================================================ 5. batching and acceptance
def three_routings():
    rs = []
    for cid, v in (("SYN0001", GAP), ("SYN0002", AMBIGUITY), ("SYN0003", NOT_SPEC)):
        rs.append(route(v, case(cid, subgroup="squamous" if cid != "SYN0003" else "adeno")))
    return rs


def test_failures_accumulate_into_one_batch_per_parameter_with_evidence_attached():
    """Accumulate, do not act on one: a text-parameter change moves every patient, and one
    acceptance test is all the budget affords."""
    batches, questions, leftover = R.assemble(
        three_routings(), lambda _r: R.BlastRadius.not_computable("prose"))
    by_param = {b.parameter_id: b for b in batches}
    assert set(by_param) == {"spec_rules", "skill"} and not questions and not leftover
    assert len(by_param["spec_rules"].proposals) == 2
    d = by_param["spec_rules"].to_dict()
    assert d["update_policy"] == R.CLINICIAN_SIGNS and d["n_elements"] == 2
    # each element keeps its own evidence, so a clinician disposing of the batch can see which
    # failure each sentence came from
    assert {e["case_id"] for e in d["elements"]} == {"SYN0001", "SYN0002"}
    assert all(e["citation"] for e in d["elements"])


def test_a_verdict_with_no_proposed_text_stays_unresolved():
    v = R.ReflectionVerdict(R.NOT_A_SPEC_FIX, "skill", "ignored",
                            quoted_passage="take the more specific reading")
    batches, _, leftover = R.assemble([route(v)], lambda _r: R.BlastRadius.not_computable("p"))
    assert not batches and leftover[0].rejected_reason == "verdict carries no proposed text"


PLAN_ARGS = {"baseline_accuracy": 0.85, "detectable_regression_pp": 5.0,
             "z_alpha": 1.96, "z_power": 0.84, "cost_per_case_usd": 0.32}


def test_the_powered_sample_size_is_the_order_the_cadence_argument_assumes():
    """The design says a few hundred per arm and $100-200 per candidate per arm. At a
    two-sided 0.05 and 80% power the honest number is ~900 per arm, so the real bill is
    nearer $290 per arm — the design's band assumed 300-600. Recorded rather than tuned:
    fitting the z's to reproduce a quoted price is how a power calculation becomes decoration.
    The conclusion the number supports (batch, do not test one change at a time) only gets
    stronger."""
    n = R.required_per_arm_n(baseline_accuracy=0.85, detectable_regression_pp=5.0,
                             z_alpha=1.96, z_power=0.84)
    assert 800 <= n <= 1000
    assert 100 <= n * 0.32 <= 400
    # A one-sided test at the same power is cheaper, and the caller chooses by passing z.
    assert R.required_per_arm_n(baseline_accuracy=0.85, detectable_regression_pp=5.0,
                                z_alpha=1.645, z_power=0.84) < n


def test_thresholds_are_required_and_raise_when_absent():
    with pytest.raises(TypeError):
        R.required_per_arm_n(baseline_accuracy=0.85, detectable_regression_pp=5.0)  # no z's
    with pytest.raises(R.MissingThresholdError):
        R.required_per_arm_n(baseline_accuracy=0.85, detectable_regression_pp=5.0,
                             z_alpha=1.96, z_power=None)
    with pytest.raises(R.MissingThresholdError):
        R.read_per_instance([R.PerInstanceResult("SYN0001", "a", True, True)],
                            max_tolerated_subgroup_drop_pp=None)


def _batch_and_sets(n_val=1000):
    batch = R.assemble([route(GAP)], lambda _r: R.BlastRadius.not_computable("prose"))[0][0]
    return batch, ["SYN0001"], [f"V{i:04d}" for i in range(n_val)]


def test_plan_names_two_arms_a_price_and_a_per_instance_read():
    batch, diag, val = _batch_and_sets()
    plan = R.plan_validation(batch, diagnosis_case_ids=diag, validation_case_ids=val,
                             test_case_ids=["T0001"], **PLAN_ARGS)
    d = plan.to_dict()
    assert d["arms"] == ["control", "candidate"] and d["status"].startswith("NOT RUN")
    assert d["estimated_cost_usd"] == pytest.approx(2 * plan.per_arm_n * 0.32, rel=1e-6)
    assert "PER INSTANCE" in d["read_as"]
    assert set(R.per_instance_result_shape()) >= {"case_id", "subgroup", "control_correct",
                                                  "candidate_correct"}


@pytest.mark.parametrize("diag,val,test", [
    (["SYN0001", "V0001"], [f"V{i:04d}" for i in range(1000)], ["T0001"]),   # diag ∩ val
    (["SYN0001"], [f"V{i:04d}" for i in range(1000)], ["V0007"]),            # val ∩ test
    (["SYN0001", "T0001"], [f"V{i:04d}" for i in range(1000)], ["T0001"]),   # diag ∩ test
])
def test_overlapping_sets_are_refused(diag, val, test):
    """Proposing from failures on a set and scoring on the same set is fitting, and it
    reports as a gain."""
    batch, _, _ = _batch_and_sets()
    with pytest.raises(R.OverlappingSetsError):
        R.plan_validation(batch, diagnosis_case_ids=diag, validation_case_ids=val,
                          test_case_ids=test, **PLAN_ARGS)


def test_a_batch_element_outside_the_diagnosis_set_is_refused():
    batch, _, val = _batch_and_sets()
    with pytest.raises(R.OverlappingSetsError, match="not in the diagnosis set"):
        R.plan_validation(batch, diagnosis_case_ids=["OTHER"], validation_case_ids=val,
                          test_case_ids=[], **PLAN_ARGS)


def test_an_underpowered_validation_set_is_refused():
    """A run too small to see a 5-point regression that then shows no regression has not
    shown there is none, and it costs the same $100-200 to learn nothing."""
    batch, diag, _ = _batch_and_sets()
    with pytest.raises(R.RefineError, match="underpowered|per arm"):
        R.plan_validation(batch, diagnosis_case_ids=diag,
                          validation_case_ids=[f"V{i:04d}" for i in range(20)],
                          test_case_ids=[], **PLAN_ARGS)


def test_a_lifted_mean_does_not_carry_a_destroyed_subgroup():
    """The failure an average is built to hide, which is why the result shape is per instance."""
    results = ([R.PerInstanceResult(f"A{i}", "adeno", False, True) for i in range(20)]
               + [R.PerInstanceResult(f"S{i}", "squamous", True, False) for i in range(5)]
               + [R.PerInstanceResult(f"S{i}", "squamous", True, True) for i in range(5, 15)])
    reading = R.read_per_instance(results, max_tolerated_subgroup_drop_pp=2.0)
    assert reading.mean_delta_pp > 0                      # the average says accept
    assert reading.per_subgroup_delta_pp["squamous"] < 0  # one subgroup was destroyed
    assert reading.regressed_subgroups == ("squamous",)
    assert reading.accept is False


def test_a_uniform_gain_is_accepted_and_an_empty_run_is_refused():
    ok = [R.PerInstanceResult(f"A{i}", "adeno", False, True) for i in range(10)]
    assert R.read_per_instance(ok, max_tolerated_subgroup_drop_pp=2.0).accept is True
    with pytest.raises(R.RefineError):
        R.read_per_instance([], max_tolerated_subgroup_drop_pp=2.0)


# ==================================================================== safety rails
def test_a_real_person_id_cannot_enter_a_routing_input():
    """The pseudonym map lives outside this tree. A case assembled straight from the real
    corpus must fail loudly here rather than travel into a proposal document."""
    shaped_like_a_real_id = "1168" + "0" * 12  # the shape, built at runtime, never written
    with pytest.raises(R.PhiInFailureCaseError):
        case(shaped_like_a_real_id)


def test_the_reflection_seam_is_unbuilt_and_says_why():
    with pytest.raises(NotImplementedError, match="spends money"):
        R.llm_reflector()


def test_unknown_adjudication_values_raise():
    with pytest.raises(R.RefineError):
        case(adjudication="probably fine")
