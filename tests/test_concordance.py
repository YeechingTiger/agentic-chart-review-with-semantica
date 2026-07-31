"""L4 concordance: the rule engine, and the two ways a concordance rate gets faked.

A concordance rate is a fraction, and both halves of it are attackable:

  numerator   scoring a patient whose care was correct because the guideline never applied,
              or because a legitimate exception was documented and ignored
  denominator scoring a patient whose inputs are unknown, in either direction — as
              concordant (inflates) or as a care gap (deflates)

Every test below is one of those two attacks, run against the engine. The reference cohort
in `test_unknown_inputs_never_enter_the_denominator` is the whole argument in five patients:
the honest rate is 1/2, and both naive alternatives (4/5 and 1/5) are asserted to be wrong.

The other thing under test is that this layer is a rule engine at all. L2 and L3 spend their
entire budget earning the right to say "we proved it is not documented"; a model asked to
judge concordance in free text would hand that back. `test_no_model_is_reachable` walks the
first-party import closure of `acr.contract.concordance` and fails if a provider SDK, `acr.core.llm` or
`acr.graph` appears anywhere in it.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from acr.contract.concordance import (
    ConcordanceInputError,
    ConcordanceResult,
    Guideline,
    GuidelineError,
    VariableValue,
    assess,
    assess_one,
    load_guideline,
    parse_guideline,
    summarise,
    validate_guideline,
    variables_from_answer,
)
from acr.contract.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
GUIDELINE = ROOT / "assets" / "guidelines" / "nccn_nsclc_subset.yaml"

ADJ = "NSCLC-ADJ-SYSTEMIC-II-IIIA"
BIO = "NSCLC-BIOMARKER-BEFORE-FIRST-LINE"
LOCAL = "NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY"


@pytest.fixture(scope="module")
def guideline():
    return load_guideline(GUIDELINE)


def found(value, **kw) -> dict:
    return {"status": "FOUND", "value": value, **kw}


def insufficient() -> dict:
    return {"status": "EVIDENCE_INSUFFICIENT", "value": None}


def proven_absent() -> dict:
    """FOUND with a null value: the coverage gate closed and there is nothing there."""
    return {"status": "FOUND", "value": None, "negative_basis": "GATE_VALIDATED"}


def adjuvant_case(**overrides) -> dict:
    """Resected stage IIB squamous NSCLC who received a platinum doublet 53 days later."""
    v = {
        "primary_site": found("C341"),
        "histology": found("8070"),
        "behavior": found("3"),
        "class_of_case": found("10", source="registry"),
        "pathologic_stage_group": found("IIB"),
        "surgical_resection_extent": found("lobectomy"),
        "surgical_margins": found("negative"),
        "adjuvant_systemic_therapy_class": found("platinum_doublet_chemotherapy"),
        "date_of_definitive_surgery": found("20220310"),
        "date_of_first_adjuvant_systemic_therapy": found("20220502"),
        "date_of_death": proven_absent(),
        "ecog_performance_status_after_surgery": found("1"),
        "patient_refused_adjuvant_systemic_therapy": found(False),
        "contraindication_to_systemic_therapy": found(False),
        "clinical_trial_enrollment": found(False),
    }
    v.update(overrides)
    return v


def untreated_adjuvant_case(**overrides) -> dict:
    """The same patient, with the adjuvant therapy proved absent rather than delivered."""
    return adjuvant_case(
        adjuvant_systemic_therapy_class=proven_absent(),
        date_of_first_adjuvant_systemic_therapy=proven_absent(),
        **overrides)


def one(guideline, variables, rec_id=ADJ) -> ConcordanceResult:
    return assess_one(guideline.recommendation(rec_id), variables, guideline)


# --------------------------------------------------------------- the four ordinary verdicts
def test_concordant(guideline):
    r = one(guideline, adjuvant_case())
    assert r.outcome == "CONCORDANT"
    assert r.rule_applied == "satisfied_when"
    assert r.blocking_inputs == ()
    used = {i.variable for i in r.inputs_used}
    assert {"pathologic_stage_group", "adjuvant_systemic_therapy_class"} <= used


def test_non_concordant_when_the_absence_is_proved(guideline):
    """The payoff of the whole coverage apparatus, at the layer that consumes it.

    `date_of_first_adjuvant_systemic_therapy` is FOUND-with-null: L3's gate closed and the
    therapy is not there. That is a care gap. Had it come back EVIDENCE_INSUFFICIENT the
    engine would refuse to score it — see the next test — and the difference between those
    two answers is exactly the difference between "not done" and "not documented", which
    top-k retrieval cannot express and a coverage proof can.
    """
    r = one(guideline, untreated_adjuvant_case())
    assert r.outcome == "NON_CONCORDANT"
    basis = {i.variable: i.negative_basis for i in r.inputs_used}
    assert basis["date_of_first_adjuvant_systemic_therapy"] == "GATE_VALIDATED"
    resolutions = {i.variable: i.resolution for i in r.inputs_used}
    assert resolutions["date_of_first_adjuvant_systemic_therapy"] == "KNOWN_ABSENT"


def test_non_concordant_when_the_wrong_drug_class_was_given(guideline):
    """Drug classes are decidable, so they are decided. Single-agent adjuvant chemotherapy
    is not the recommended regimen and the value set deliberately omits it."""
    r = one(guideline, adjuvant_case(
        adjuvant_systemic_therapy_class=found("single_agent_chemotherapy")))
    assert r.outcome == "NON_CONCORDANT"


def test_non_concordant_when_therapy_started_outside_the_window(guideline):
    r = one(guideline, adjuvant_case(
        date_of_first_adjuvant_systemic_therapy=found("20221102")))   # 237 days
    assert r.outcome == "NON_CONCORDANT"


def test_not_applicable_is_not_the_same_as_not_assessable(guideline):
    """Stage IA is a fact, not a missing datum. Filing it under NOT_ASSESSABLE would put
    every correctly-excluded patient into the pile that means "we could not tell"."""
    r = one(guideline, adjuvant_case(pathologic_stage_group=found("IA2")))
    assert r.outcome == "NOT_APPLICABLE"
    assert r.rule_applied == "applies_when"
    assert r.blocking_inputs == ()


def test_a_determinate_exclusion_and_an_unknown_one_are_different_answers(guideline):
    """8041 is small cell — a different disease, so NOT_APPLICABLE is a true statement.
    8010 is "carcinoma, NOS", which does not say whether the disease is small cell, and the
    recommendation turns entirely on that. Both codes fail NSCLC set membership identically,
    so membership alone cannot tell them apart; `unknown_value_codes` is what does."""
    assert one(guideline, adjuvant_case(histology=found("8041"))).outcome == "NOT_APPLICABLE"
    assert one(guideline, adjuvant_case(histology=found("8046"))).outcome == "CONCORDANT"

    nos = one(guideline, adjuvant_case(histology=found("8010")))
    assert nos.outcome == "NOT_ASSESSABLE"
    assert "histology" in nos.blocking_inputs


def test_a_registry_sentinel_is_an_unknown_wearing_a_value(guideline):
    """Stage `99` and class of case `99` are in the shipped specs' own allowable_values, so
    they will arrive. Scored as ordinary non-members they would report the patient as
    determinately outside the population — the inflation this layer exists to refuse,
    arriving disguised as a value rather than as a missing field.

    `status` stays FOUND because that is what L2 truthfully returned; `resolution` is
    UNKNOWN because that is what the code means. `unknown_sentinel` is why the two differ,
    so the disagreement is inspectable rather than a discrepancy someone has to explain.
    """
    for var in ("pathologic_stage_group", "class_of_case"):
        r = one(guideline, adjuvant_case(**{var: found("99")}))
        assert r.outcome == "NOT_ASSESSABLE", var
        assert var in r.blocking_inputs
        used = {i.variable: i for i in r.inputs_used}[var]
        assert (used.status, used.resolution, used.unknown_sentinel) == \
            ("FOUND", "UNKNOWN", True)
        assert used.value == "99", "the sentinel is reported, not erased"


def test_the_sentinel_codes_are_values_the_specs_can_actually_emit(guideline):
    """A sentinel list guessed rather than read off the spec would silently never fire."""
    stage = load_spec(ROOT / "assets" / "specs" / "STORE.700_880.stage.yaml")
    allowable = {f.name: [str(v) for v in (f.allowable_values or [])] for f in stage.fields}
    for var in ("clinical_stage_group", "pathologic_stage_group"):
        assert set(guideline.unknown_value_codes[var]) <= set(allowable[var]), var
        assert not set(guideline.unknown_value_codes[var]) & set(
            guideline.value_sets["stage_I"] + guideline.value_sets["stage_II_IIIA"]
            + guideline.value_sets["stage_IV"]), f"{var}: a sentinel is also a scored value"

    coc = load_spec(ROOT / "assets" / "specs" / "STORE.610.class_of_case.yaml")
    coc_allowable = {str(v) for f in coc.fields for v in (f.allowable_values or [])}
    assert set(guideline.unknown_value_codes["class_of_case"]) <= coc_allowable


# ------------------------------------------------- NOT_ASSESSABLE, which is the whole point
@pytest.mark.parametrize("blocked", [
    "pathologic_stage_group",      # applicability
    "surgical_margins",
    "histology",
])
def test_unknown_applicability_input_blocks_scoring(guideline, blocked):
    r = one(guideline, adjuvant_case(**{blocked: insufficient()}))
    assert r.outcome == "NOT_ASSESSABLE"
    assert r.rule_applied == "applies_when"
    assert blocked in r.blocking_inputs


def test_unknown_action_input_blocks_scoring(guideline):
    """Applicability is settled; whether the therapy happened is not. Not a care gap."""
    r = one(guideline, adjuvant_case(adjuvant_systemic_therapy_class=insufficient()))
    assert r.outcome == "NOT_ASSESSABLE"
    assert r.rule_applied == "satisfied_when"
    assert "adjuvant_systemic_therapy_class" in r.blocking_inputs


def test_unknown_exception_status_blocks_scoring(guideline):
    """Cause D of the design doc, and the one most often botched in the literature.

    The therapy is provably absent, so the arithmetic says care gap — but nobody has
    established whether the patient refused. Reporting a care gap here inflates the gap
    rate the same way scoring an unknown input inflates the concordance rate. The exception
    is an extracted variable and is held to the same standard as the primary ones.
    """
    r = one(guideline, untreated_adjuvant_case(
        patient_refused_adjuvant_systemic_therapy=insufficient()))
    assert r.outcome == "NOT_ASSESSABLE"
    assert r.rule_applied == "exception_status_unknown"
    assert "patient_refused_adjuvant_systemic_therapy" in r.blocking_inputs
    assert "patient_refused" in r.reason


def test_a_ruled_out_exception_does_not_block_scoring(guideline):
    """The mirror image: FOUND=false is the gate saying it looked and there was no refusal.
    That is what makes NON_CONCORDANT sayable at all."""
    assert one(guideline, untreated_adjuvant_case()).outcome == "NON_CONCORDANT"


def test_spec_insufficient_input_is_unknown_not_absent(guideline):
    """Class of Case is SPEC_INSUFFICIENT from notes by construction — the shipped spec sets
    `data_source: outside_notes` and graph.py forces WRONG_DATA_SOURCE. Every treatment
    recommendation here is facility-attributed, so without a registry feed for [610] the
    honest answer is that none of them can be scored from notes alone."""
    spec = load_spec(ROOT / "assets" / "specs" / "STORE.610.class_of_case.yaml")
    assert spec.data_source == "outside_notes"
    from_notes = variables_from_answer(
        {"status": "SPEC_INSUFFICIENT", "remedy_class": "WRONG_DATA_SOURCE", "value": {}},
        [f.name for f in spec.fields], source=spec.spec_id)
    assert from_notes["class_of_case"].status == "SPEC_INSUFFICIENT"
    assert from_notes["class_of_case"].resolution == "UNKNOWN"

    r = one(guideline, adjuvant_case(class_of_case=from_notes["class_of_case"]))
    assert r.outcome == "NOT_ASSESSABLE"
    assert "class_of_case" in r.blocking_inputs
    statuses = {i.variable: i.status for i in r.inputs_used}
    assert statuses["class_of_case"] == "SPEC_INSUFFICIENT"


def test_a_variable_nobody_extracted_is_unknown_not_false(guideline):
    """No staging spec exists yet. The recommendation must come back unscorable and say so,
    not quietly treat a missing stage as "not stage II" and drop the patient."""
    v = adjuvant_case()
    del v["pathologic_stage_group"]
    r = one(guideline, v)
    assert r.outcome == "NOT_ASSESSABLE"
    assert "pathologic_stage_group" in r.blocking_inputs
    assert [i.status for i in r.inputs_used
            if i.variable == "pathologic_stage_group"] == ["NOT_EXTRACTED"]


def test_unknown_inputs_never_enter_the_denominator(guideline):
    """The reference cohort. Five patients, two scorable, honest rate 1/2.

    Folding the three unknowns in either direction gives 4/5 or 1/5, and both are asserted
    wrong here because both are the mistake this layer exists to refuse.
    """
    cohort = [
        adjuvant_case(),                                                    # CONCORDANT
        untreated_adjuvant_case(),                                          # NON_CONCORDANT
        adjuvant_case(pathologic_stage_group=insufficient()),          # unknown stage
        adjuvant_case(adjuvant_systemic_therapy_class=insufficient()),      # unknown action
        untreated_adjuvant_case(
            patient_refused_adjuvant_systemic_therapy=insufficient()),      # unknown exception
    ]
    results = [one(guideline, c) for c in cohort]
    assert [r.outcome for r in results] == [
        "CONCORDANT", "NON_CONCORDANT",
        "NOT_ASSESSABLE", "NOT_ASSESSABLE", "NOT_ASSESSABLE"]

    s = summarise(results)
    assert s["denominator"] == 2
    assert s["concordance_rate"] == pytest.approx(0.5)
    assert s["denominator_excludes"]["NOT_ASSESSABLE"] == 3
    assert s["assessable_fraction"] == pytest.approx(2 / 5)
    assert s["concordance_rate"] not in (pytest.approx(4 / 5), pytest.approx(1 / 5))
    assert s["blocking_inputs"]["pathologic_stage_group"] == 1


def test_a_cohort_with_nothing_scorable_has_no_rate(guideline):
    """None, not 0.0 and not 1.0. A rate over an empty denominator is not a small number."""
    s = summarise([one(guideline, adjuvant_case(histology=insufficient()))])
    assert s["denominator"] == 0
    assert s["concordance_rate"] is None
    assert s["counts"]["NOT_ASSESSABLE"] == 1


# ------------------------------------------------------ legitimate exceptions are not gaps
@pytest.mark.parametrize("override,expected_id", [
    ({"ecog_performance_status_after_surgery": found("3")},
     "performance_status_precludes_systemic_therapy"),
    ({"patient_refused_adjuvant_systemic_therapy": found(True)}, "patient_refused"),
    ({"contraindication_to_systemic_therapy": found("yes")}, "contraindicating_comorbidity"),
    ({"clinical_trial_enrollment": found(True)}, "therapeutic_clinical_trial"),
    ({"date_of_death": found("20220415")}, "died_before_window_closed"),
])
def test_documented_exception_is_a_distinct_outcome(guideline, override, expected_id):
    r = one(guideline, untreated_adjuvant_case(**override))
    assert r.outcome == "EXCEPTION_DOCUMENTED"
    assert r.exception_id == expected_id
    assert r.rule_applied == f"exception:{expected_id}"
    assert "not a care gap" in r.reason


def test_documented_exceptions_are_excluded_from_the_denominator(guideline):
    results = [
        one(guideline, adjuvant_case()),
        one(guideline, untreated_adjuvant_case()),
        one(guideline, untreated_adjuvant_case(
            patient_refused_adjuvant_systemic_therapy=found(True))),
    ]
    s = summarise(results)
    assert s["counts"]["EXCEPTION_DOCUMENTED"] == 1
    assert s["denominator"] == 2                        # the refusal is out of the fraction
    assert s["concordance_rate"] == pytest.approx(0.5)
    assert s["denominator_excludes"]["EXCEPTION_DOCUMENTED"] == 1


def test_care_delivered_beats_an_exception(guideline):
    """A patient who was offered a reason not to treat and was treated anyway is concordant.
    Exceptions are only consulted once the action is known to be absent."""
    r = one(guideline, adjuvant_case(clinical_trial_enrollment=found(True)))
    assert r.outcome == "CONCORDANT"


def test_an_exception_settles_a_case_whose_action_is_unknown(guideline):
    """The other ordering: a documented refusal removes the patient from the denominator
    whether or not the therapy record can be found, so the unknown never has to be raised."""
    r = one(guideline, adjuvant_case(
        adjuvant_systemic_therapy_class=insufficient(),
        patient_refused_adjuvant_systemic_therapy=found(True)))
    assert r.outcome == "EXCEPTION_DOCUMENTED"


# ----------------------------------------------------------------- dates and value classes
def test_a_disjunctive_action_accepts_either_modality(guideline):
    """SABR for a medically inoperable stage I patient is correct care. Requiring surgery
    specifically would score every one of them as a care gap."""
    base = {
        "primary_site": found("C342"), "histology": found("8140"), "behavior": found("3"),
        "class_of_case": found("11"), "clinical_stage_group": found("IA2"),
        "date_of_initial_diagnosis": found("20100612"),
        "date_of_first_definitive_local_therapy": found("20100901"),
        "surgical_resection_extent": proven_absent(),
        "radiotherapy_intent": found("definitive"),
        "ecog_performance_status": found("2"),
        "patient_refused_local_therapy": found(False),
        "comfort_care_or_hospice_before_treatment": found(False),
        "clinical_trial_enrollment": found(False),
    }
    assert one(guideline, base, LOCAL).outcome == "CONCORDANT"
    surgical = dict(base, surgical_resection_extent=found("wedge_resection"),
                    radiotherapy_intent=proven_absent())
    assert one(guideline, surgical, LOCAL).outcome == "CONCORDANT"
    neither = dict(base, radiotherapy_intent=found("palliative"))
    assert one(guideline, neither, LOCAL).outcome == "NON_CONCORDANT"


def test_an_imprecise_date_that_straddles_a_threshold_is_not_a_decision(guideline):
    """`20100499` is the shipped STORE.390 boundary case for "diagnosed in the spring".

    The month is known and the day is not, so the interval to therapy on 2010-10-15 is
    somewhere between 168 and 197 days and the 180-day bound falls inside it. Picking a day
    and reporting the answer would make the verdict an artefact of the imputation. Moving
    the therapy to 2010-09-01 puts the whole interval under the bound and the same imprecise
    diagnosis date decides cleanly.
    """
    base = {
        "primary_site": found("C342"), "histology": found("8140"), "behavior": found("3"),
        "class_of_case": found("11"), "clinical_stage_group": found("IA2"),
        "date_of_initial_diagnosis": found("20100499"),
        "surgical_resection_extent": found("lobectomy"),
        "radiotherapy_intent": proven_absent(),
        "ecog_performance_status": found("0"),
        "patient_refused_local_therapy": found(False),
        "comfort_care_or_hospice_before_treatment": found(False),
        "clinical_trial_enrollment": found(False),
    }
    straddles = one(guideline, dict(
        base, date_of_first_definitive_local_therapy=found("20101015")), LOCAL)
    assert straddles.outcome == "NOT_ASSESSABLE"
    assert any("straddles" in n for n in straddles.notes)

    decided = one(guideline, dict(
        base, date_of_first_definitive_local_therapy=found("20100901")), LOCAL)
    assert decided.outcome == "CONCORDANT"


def test_biomarker_ordering_and_the_never_treated_trigger(guideline):
    base = {
        "primary_site": found("C343"), "histology": found("8140"), "behavior": found("3"),
        "class_of_case": found("10"), "clinical_stage_group": found("IVA"),
        "date_of_first_systemic_therapy": found("2019-12-02"),      # ISO is accepted too
        "date_of_egfr_result": found("20191125"),
        "date_of_alk_result": found("20191125"),
        "tissue_insufficient_for_molecular_testing": found(False),
        "patient_refused_molecular_testing": found(False),
    }
    assert one(guideline, base, BIO).outcome == "CONCORDANT"

    late = dict(base, date_of_alk_result=found("20200110"))
    assert one(guideline, late, BIO).outcome == "NON_CONCORDANT"

    squamous = dict(base, histology=found("8070"))
    assert one(guideline, squamous, BIO).outcome == "NOT_APPLICABLE"
    adenosquamous = dict(base, histology=found("8560"))
    assert one(guideline, adenosquamous, BIO).outcome == "CONCORDANT"

    # The recommendation is about the ordering of two events and one of them did not happen.
    never = dict(base, date_of_first_systemic_therapy=proven_absent())
    assert one(guideline, never, BIO).outcome == "NOT_APPLICABLE"

    unsent = dict(base, date_of_egfr_result=proven_absent())
    assert one(guideline, unsent, BIO).outcome == "NON_CONCORDANT"


def _one_rule_guideline(condition: dict) -> Guideline:
    """A guideline with a single applicability condition, for probing one operator."""
    return parse_guideline({
        "guideline_id": "PROBE",
        "value_sets": {"squamous": ["8070", "8071"]},
        "recommendations": [{
            "id": "R",
            "source": {"paraphrase": True},
            "required_inputs": [{"name": "histology", "source": "registry_limited_dataset"},
                                {"name": "acted", "source": "registry_limited_dataset"}],
            "applies_when": [condition],
            "satisfied_when": [{"op": "is_true", "var": "acted"}],
        }],
    })


def test_an_absent_value_cannot_place_a_patient_inside_a_population():
    """`not_in_set` on an established absence is UNKNOWN; `in_set` on it is FALSE.

    "There is no histology on record" answers "is it squamous?" with no. It does not answer
    "is it non-squamous?" — and the non-squamous population is the one a molecular-testing
    recommendation scores. Treating the two as mirror images reads an open-world fact as a
    closed-world one, which is the same error as reading an empty search result as a finding.

    Probed one operator at a time on purpose. In the shipped biomarker rule the neighbouring
    `in_set nsclc_histology` is FALSE for the same input and correctly dominates the
    conjunction, so the asymmetry is invisible there.
    """
    absent = {"histology": proven_absent(), "acted": found(True)}

    negative = _one_rule_guideline({"op": "not_in_set", "var": "histology", "set": "squamous"})
    r = assess(absent, negative)[0]
    assert r.outcome == "NOT_ASSESSABLE"
    assert r.blocking_inputs == ("histology",)

    positive = _one_rule_guideline({"op": "in_set", "var": "histology", "set": "squamous"})
    assert assess(absent, positive)[0].outcome == "NOT_APPLICABLE"

    # With a value present the two are ordinary mirror images again.
    assert assess({"histology": found("8140"), "acted": found(True)}, negative)[0].outcome \
        == "CONCORDANT"
    assert assess({"histology": found("8070"), "acted": found(True)}, negative)[0].outcome \
        == "NOT_APPLICABLE"


def test_a_malformed_code_falls_out_of_the_population_and_that_is_l3s_job(guideline):
    """`C3412` — four digits — was submitted on 2026-07-26 and passed the gate because the
    declared `format: C\\d{3}` was rendered into the prompt and never enforced.

    Here it simply fails `C34\\d` and the patient leaves the population, which is the wrong
    answer for a lung case with a typo. The engine cannot tell a typo from a different organ
    and must not try. The defence is upstream, in `check_field_formats`, which now rejects
    it before it can reach this layer — this test exists to record that the seam is load
    bearing, so that turning that check off has a visible cost here.
    """
    from acr.contract.answer_checks import check_field_formats
    spec = load_spec(ROOT / "assets" / "specs" / "STORE.400_522_523.site_histology_behavior.yaml")
    assert check_field_formats(spec.fields, {"primary_site": "C3412"}), \
        "L3 must still reject the malformed code that L4 would silently drop"
    assert one(guideline, adjuvant_case(primary_site=found("C3412"))).outcome == "NOT_APPLICABLE"


# --------------------------------------------------------------------- engine properties
def test_bare_values_without_a_status_are_refused(guideline):
    with pytest.raises(ConcordanceInputError) as exc:
        one(guideline, adjuvant_case(behavior="3"))
    assert "status" in str(exc.value)


def test_assess_is_pure(guideline):
    """Same inputs, same output; key order irrelevant; the caller's dict is not touched."""
    v = adjuvant_case()
    before = copy.deepcopy(v)
    first = [r.to_dict() for r in assess(v, guideline)]
    second = [r.to_dict() for r in assess(v, guideline)]
    shuffled = [r.to_dict() for r in assess(dict(reversed(list(v.items()))), guideline)]
    assert first == second == shuffled
    assert v == before
    assert [r["recommendation_id"] for r in first] == [ADJ, BIO, LOCAL]
    assert all(r["engine"] == "acr.contract.concordance/deterministic" for r in first)


def test_every_result_carries_the_guideline_it_was_scored_under(guideline):
    r = one(guideline, adjuvant_case())
    assert r.guideline_id == "NCCN-NSCLC-SUBSET"
    assert r.guideline_hash == guideline.guideline_hash and len(r.guideline_hash) == 16

    # A label is only comparable to another label under the same hash — the same rule the
    # spec loader applies, and the reason a threshold lives in the YAML and not in code.
    edited = copy.deepcopy(guideline.raw)
    edited["recommendations"][0]["satisfied_when"][1]["max_days"] = 90
    assert parse_guideline(edited).guideline_hash != guideline.guideline_hash


FORBIDDEN_MODULES = {
    "llm", "graph", "deep_runner", "cli",                      # first-party paths to a model
    "openai", "anthropic", "litellm", "langchain", "langgraph", "deepagents",
    "requests", "httpx", "urllib", "socket", "http",           # and to a network
}


def _first_party_closure(start: Path) -> dict[str, set[str]]:
    """Every module `acr.contract.concordance` reaches, and what each one imports."""
    seen: dict[str, set[str]] = {}
    queue = [start]
    while queue:
        path = queue.pop()
        if path.name in seen:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if node.level:                        # from .x import y
                    imports.add(root)
                    sibling = path.parent / f"{root}.py"
                    if sibling.exists():
                        queue.append(sibling)
                else:
                    imports.add(root)
        seen[path.name] = imports
    return seen


def test_no_model_is_reachable_from_this_module():
    """L4 is a rule engine. Not by convention, and not only in the docstring.

    An LLM judging concordance in free text would give back everything L2 and L3 paid for:
    the verdict would stop being reproducible from the recorded inputs, and there would be
    no way to tell a rule change from a model change when a rate moved.
    """
    closure = _first_party_closure(ROOT / "src" / "acr" / "contract" / "concordance.py")
    assert "concordance.py" in closure
    offenders = {mod: sorted(imports & FORBIDDEN_MODULES)
                 for mod, imports in closure.items() if imports & FORBIDDEN_MODULES}
    assert not offenders, f"a model or network is reachable from acr.contract.concordance: {offenders}"


# ------------------------------------------------------------------- the guideline itself
def test_the_shipped_guideline_is_executable(guideline):
    assert validate_guideline(guideline) == []
    assert len(guideline.recommendations) == 3
    for r in guideline.recommendations:
        assert r.statement.strip(), f"{r.id} has no human-readable statement"
        assert r.source.get("paraphrase") is True, \
            f"{r.id} must not present itself as verbatim guideline text"
        assert r.exceptions, f"{r.id} declares no legitimate exceptions"


def test_declared_inputs_must_be_the_inputs_the_rules_read(guideline):
    """The C3412 lesson applied to this file: a constraint written down and never run is
    not a constraint. `required_inputs` is the recommendation's documentation, so it is
    executed against the conditions rather than trusted."""
    for r in guideline.recommendations:
        assert sorted(r.declared_inputs) == sorted(r.referenced_inputs), r.id

    undeclared = copy.deepcopy(guideline.raw)
    undeclared["recommendations"][0]["required_inputs"] = \
        undeclared["recommendations"][0]["required_inputs"][:-1]
    bad = validate_guideline(parse_guideline(undeclared))
    assert any("required_inputs does not declare" in m for m in bad), bad

    unread = copy.deepcopy(guideline.raw)
    unread["recommendations"][0]["required_inputs"].append(
        {"name": "smoking_status", "source": "registry_limited_dataset"})
    assert any("no condition reads" in m for m in validate_guideline(parse_guideline(unread)))


@pytest.mark.parametrize("mutate,expected", [
    (lambda g: g["recommendations"][0]["applies_when"].append({"op": "roughly_equals",
                                                               "var": "behavior"}),
     "unknown op"),
    (lambda g: g["recommendations"][0]["applies_when"][1].__setitem__("set", "nsclc_histolgy"),
     "undeclared value_set"),
    (lambda g: g["recommendations"][0]["applies_when"][0].__setitem__("pattern", "C34(\\d"),
     "invalid pattern"),
    (lambda g: g["recommendations"][0]["source"].pop("operationalisation"),
     "source.operationalisation"),
    (lambda g: g["recommendations"][0]["exceptions"][0].__setitem__("when", []),
     "fires for everyone"),
    (lambda g: g["recommendations"][0].__setitem__("satisfied_when", []),
     "never be non-concordant"),
    (lambda g: g["unknown_value_codes"].__setitem__("stage_group", ["99"]),
     "no condition reads"),
])
def test_validation_refuses_an_unexecutable_guideline(guideline, mutate, expected):
    """A typo'd op or value-set name must fail loudly at load. `check_field_formats`
    swallows a broken regex so a live patient run cannot be blocked by a spec typo; a
    guideline is authored offline under review, so here the same typo is fatal."""
    data = copy.deepcopy(guideline.raw)
    mutate(data)
    violations = validate_guideline(parse_guideline(data))
    assert any(expected in m for m in violations), violations


def test_load_guideline_raises_rather_than_scoring_with_a_broken_rule(tmp_path):
    import yaml
    data = yaml.safe_load(GUIDELINE.read_text(encoding="utf-8"))
    data["recommendations"][0]["applies_when"][1]["set"] = "does_not_exist"
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(GuidelineError):
        load_guideline(p)
    assert load_guideline(p, validate=False).guideline_id == "NCCN-NSCLC-SUBSET"


def test_every_input_declares_where_it_would_come_from(guideline):
    """None of these recommendations can be scored end-to-end today, and the YAML says which
    inputs are why: every treatment variable is `not_yet_extractable`. That is the design
    doc's build order ("do not build 4 and 5 before 3") written as a checkable claim rather
    than a caveat in prose, and at runtime it costs nothing — an unbuilt extractor produces
    NOT_ASSESSABLE naming itself, which is the correct answer.

    A `spec_id` that names no shipped spec is the interesting failure: it would read as a
    live binding while the variable silently never arrives.
    """
    by_source: dict[str, set[str]] = {}
    for r in guideline.recommendations:
        for d in r.required_inputs:
            by_source.setdefault(d["source"], set()).add(d["name"])
    assert by_source["registry_limited_dataset"] >= {"class_of_case"}
    assert by_source["extraction_spec"] == {
        "primary_site", "histology", "behavior", "date_of_initial_diagnosis",
        "clinical_stage_group", "pathologic_stage_group"}
    assert "adjuvant_systemic_therapy_class" in by_source["not_yet_extractable"]

    shipped = {load_spec(p).spec_id for p in (ROOT / "assets" / "specs").glob("*.yaml")}
    for r in guideline.recommendations:
        for d in r.required_inputs:
            if d["source"] == "extraction_spec":
                assert d.get("spec_id") in shipped, f"{r.id}/{d['name']} -> {d.get('spec_id')}"


def test_a_bound_spec_really_does_emit_the_field_the_guideline_names(guideline):
    """The seam that has no type checker. `variables_from_answer` keys on spec field names,
    so a guideline variable bound to `extraction_spec` must BE a field of that spec —
    otherwise the flattening produces a name nothing reads and the rule blocks forever on a
    variable that was extracted successfully under a different name."""
    fields: dict[str, set[str]] = {}
    for p in (ROOT / "assets" / "specs").glob("*.yaml"):
        s = load_spec(p)
        fields[s.spec_id] = {f.name for f in s.fields}
    for r in guideline.recommendations:
        for d in r.required_inputs:
            if d["source"] == "extraction_spec":
                assert d["name"] in fields[d["spec_id"]], \
                    f"{r.id}: {d['name']!r} is not a field of {d['spec_id']}"


# --------------------------------------------------------------- the extraction → L4 seam
def test_a_partial_answer_flattens_per_field_not_per_answer():
    """The real shape from `runs/aprime_SYN0002`: EVIDENCE_INSUFFICIENT overall, with
    `primary_site: C186` established anyway, because the spec permits exactly that. The
    field that carries a value is FOUND; the two the answer stayed silent about are not."""
    spec = load_spec(ROOT / "assets" / "specs" / "STORE.400_522_523.site_histology_behavior.yaml")
    v = variables_from_answer(
        {"status": "EVIDENCE_INSUFFICIENT", "negative_basis": "GATE_VALIDATED",
         "value": {"primary_site": "C186"}},
        [f.name for f in spec.fields], source=spec.spec_id)
    assert v["primary_site"] == VariableValue("FOUND", "C186", "GATE_VALIDATED", spec.spec_id)
    assert v["histology"].resolution == "UNKNOWN"
    assert v["behavior"].status == "EVIDENCE_INSUFFICIENT"
    # A silence is not an assertion that there is none — the recurrence spec's rule, applied
    # to a value dict instead of a chart.
    assert v["histology"].resolution != "KNOWN_ABSENT"


def test_an_explicit_null_under_a_found_answer_is_an_established_absence():
    v = variables_from_answer(
        {"status": "FOUND", "value": {"recurrence_type": "00", "recurrence_date": None}},
        ["recurrence_type", "recurrence_date"])
    assert v["recurrence_date"].resolution == "KNOWN_ABSENT"
    assert v["recurrence_type"].resolution == "KNOWN"
