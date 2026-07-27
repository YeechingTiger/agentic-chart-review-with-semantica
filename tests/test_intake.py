"""L0 intake: a question becomes a routing decision whose consequences are all code.

The property under test is NOT "the router classifies well". Classification is judgement and
a model does it; a suite that asserted on classification quality would be asserting on a
prompt and would go red every time a provider ships a new checkpoint. What is asserted here
is the half that must never move:

  * a route to a spec is exact-match and a miss is loud;
  * a composition that names a field or a value no spec produces is REFUSED, not narrowed,
    not defaulted, not silently dropped;
  * whether a variable is answerable from notes comes off `spec.data_source` and overrules
    the classifier;
  * everything unresolved is a Gap with a remedy, and the guideline routing prints all of
    them.

Every test runs offline. `no_network` below makes that structural rather than aspirational:
`litellm.completion` is replaced with a function that fails the test, so a future edit that
reaches for a provider goes red here instead of going red in CI six weeks later on a
credential rotation — at which point someone marks it skip and the consequence path stops
being tested at all.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import acr.intake as I
from acr.cli import app
from acr.concordance import _validate_condition, load_guideline, parse_guideline
from acr.registry_catalog import VariableCatalog
from acr.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
GUIDELINES = ROOT / "guidelines"
SHB = "STORE.400_522_523.site_histology_behavior"
STG = "STORE.700_880.stage"
runner = CliRunner()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any provider call from anywhere in this file is a test failure, not a slow test."""
    import litellm

    def boom(*a, **k):
        raise AssertionError("a test in test_intake.py called a model; intake tests are offline")

    monkeypatch.setattr(litellm, "completion", boom)


@pytest.fixture(scope="module")
def cat() -> VariableCatalog:
    return VariableCatalog.from_directory(SPECS)


@pytest.fixture(scope="module")
def guides() -> list:
    return I.load_guidelines(GUIDELINES)


def stub(**kw) -> I.StubClassifier:
    return I.StubClassifier(I.Classification(**kw))


# ------------------------------------------------------------------ this layer reads no chart
def test_intake_cannot_reach_a_patient_chart():
    """Structural, because `--dry-run` is one typo in one script from being off.

    A planning layer that *can* open a note will eventually be asked to, by someone who
    reasons that it already has the patient id in hand. The guarantee worth having is that
    the code to do it is not present: no corpus, no chart, no note text, at all.
    """
    src = inspect.getsource(I)
    for forbidden in ("from .corpus", "import corpus", "PatientChart", "read_text("):
        assert forbidden not in src, (
            f"acr.intake mentions {forbidden!r}; this layer plans and must not be able to read PHI"
        )
    assert "patient" not in {p for p in inspect.signature(I.route).parameters}


def test_the_report_says_on_its_face_that_no_chart_was_read(cat, guides):
    d = I.route("primary_site", cat, guidelines=guides)
    assert d.to_dict()["reads_charts"] is False


# ------------------------------------------------------------------ five outcomes, no sixth
def test_there_are_exactly_five_outcomes_and_unclassified_is_not_one_of_them(cat):
    assert len(I.OUTCOMES) == 5
    assert set(I.OUTCOME_MEANINGS) == set(I.OUTCOMES)
    d = I.route("was EGFR testing done before first-line therapy", cat)
    assert d.outcome is None, "an unreadable question must not be assigned a routing outcome"
    assert d.refused
    assert [g.kind for g in d.gaps] == [I.UNCLASSIFIED]
    assert "--model" in d.gaps[0].remedy


# ------------------------------------------------------------------ EXISTING_VARIABLE
@pytest.mark.parametrize("question,spec_id", [
    ("primary_site", SHB),
    ("Primary Site", SHB),
    ("pathologic_stage_group", STG),
    ("STORE.390", "STORE.390.date_of_initial_diagnosis"),
])
def test_an_exact_name_routes_to_its_spec_with_no_model_call(cat, question, spec_id):
    d = I.route(question, cat)
    assert d.outcome == I.EXISTING_VARIABLE
    assert d.model_calls == 0
    assert spec_id in d.spec_ids()
    assert not d.gaps


def test_prose_containing_a_variable_name_is_not_substring_matched(cat, guides):
    """`stage II-IIIA resected NSCLC` contains `stage`. A keyword router would send it to the
    stage spec, drop the histology and the resection, and answer a question nobody asked —
    the same failure that filed `Fine-Needle-Report` outside ["Pathology", "Cytology"]."""
    d = I.route("does this patient have stage II-IIIA resected NSCLC", cat, guidelines=guides)
    assert d.outcome is None
    assert d.spec_ids() == []


def test_a_classifier_naming_a_variable_no_spec_produces_is_refused_with_the_vocabulary(cat):
    """Fails loudly on a miss. `ajcc_pathologic_stage` is the real drift: the guideline layer
    once named it where the spec's field is `pathologic_stage_group`, nothing errored, and
    every case came back NOT_ASSESSABLE naming a variable the operator believed was extracted."""
    d = I.route("q", cat, classifier=stub(outcome=I.EXISTING_VARIABLE,
                                          variables=("ajcc_pathologic_stage",)))
    assert d.refused
    g = next(g for g in d.gaps if g.kind == I.UNKNOWN_VARIABLE)
    assert "pathologic_stage_group" in g.context, "the error must carry the real vocabulary"
    assert "pathologic_stage_group" in g.detail, "and suggest the near miss it did not accept"
    assert d.spec_ids() == [], "a miss must not resolve to a spec anyway"


# ------------------------------------------------------------------ WRONG_DATA_SOURCE
def test_the_spec_overrules_the_classifier_on_where_a_variable_lives(cat):
    """STORE.610 is the worked example. A classifier that reads `class_of_case` as an ordinary
    extraction does not get to be right: the spec declares `data_source: outside_notes` and
    `graph.py` forces every run of it to SPEC_INSUFFICIENT at finalize."""
    d = I.route("class_of_case", cat,
                classifier=stub(outcome=I.EXISTING_VARIABLE, variables=("class_of_case",)))
    assert d.classified_as == I.EXISTING_VARIABLE
    assert d.outcome == I.WRONG_DATA_SOURCE, "the route is the code's, not the classifier's"
    g = next(g for g in d.gaps if g.kind == I.OUTSIDE_NOTES)
    assert "STORE.610.class_of_case" in g.detail
    assert "do not extract it" in g.remedy


def test_a_refusal_with_no_spec_behind_it_is_itself_a_gap(cat):
    """A WRONG_DATA_SOURCE verdict that lives only in a report is forgotten by the next person
    who asks. STORE.610 exists so the refusal is a shipped, hashed artifact."""
    d = I.route("what was billed for this admission", cat,
                classifier=stub(outcome=I.WRONG_DATA_SOURCE))
    assert d.outcome == I.WRONG_DATA_SOURCE
    assert any(g.kind == I.NO_SPEC_DECLARES_THE_REFUSAL for g in d.gaps)


# ------------------------------------------------------------------ COMPOSITION
STAGE_II_IIIA_ADENO = (
    {"op": "matches", "var": "primary_site", "pattern": r"C34\d"},
    {"op": "equals", "var": "histology", "value": "8140"},
    {"op": "equals", "var": "behavior", "value": "3"},
    {"op": "in_set", "var": "pathologic_stage_group", "values": ["IIA", "IIB", "IIIA"]},
)


def _composition(cat, guides, terms=STAGE_II_IIIA_ADENO, missing=()):
    return I.route("does this patient have stage II-IIIA lung adeno", cat, guidelines=guides,
                   classifier=stub(outcome=I.COMPOSITION, predicate=terms,
                                   missing_inputs=missing))


def test_a_composition_emits_a_checkable_expression_and_no_spec(cat, guides):
    d = _composition(cat, guides)
    assert d.outcome == I.COMPOSITION
    assert d.skeleton is None, "a composition must not propose a spec; that is the backlog bug"
    assert d.predicate.checkable and d.predicate.complete
    assert d.predicate.expression() == (
        r"primary_site matches /C34\d/ AND histology == '8140' AND behavior == '3' "
        "AND pathologic_stage_group in {IIA, IIB, IIIA}")
    assert set(d.spec_ids()) == {SHB, STG}


def test_the_emitted_conditions_are_the_rule_engine_s_own_grammar(cat, guides):
    """Not a lookalike. If the predicate needed translating before `concordance` could run it,
    the translation would be a second grammar and the two would drift."""
    d = _composition(cat, guides)
    for cond in d.predicate.conditions():
        assert _validate_condition(cond, {}) == []


def test_a_composition_evaluates_three_valued_so_a_registry_sentinel_is_not_an_exclusion(
        cat, guides):
    """`pathologic_stage_group = 99` means nobody staged the patient. Evaluated two-valued it
    fails set membership like any non-member and the patient leaves the cohort as though their
    stage had been established as something else — the inflation `unknown_value_codes` exists
    to refuse, arriving one layer earlier."""
    d = _composition(cat, guides)
    found = {"primary_site": {"status": "FOUND", "value": "C341"},
             "histology": {"status": "FOUND", "value": "8140"},
             "behavior": {"status": "FOUND", "value": "3"},
             "pathologic_stage_group": {"status": "FOUND", "value": "IIB"}}
    assert d.predicate.evaluate(found).truth == "TRUE"

    sentinel = dict(found, pathologic_stage_group={"status": "FOUND", "value": "99"})
    v = d.predicate.evaluate(sentinel)
    assert v.truth == "UNKNOWN"
    assert v.unknown == ("pathologic_stage_group",)

    out_of_cohort = dict(found, pathologic_stage_group={"status": "FOUND", "value": "IVA"})
    assert d.predicate.evaluate(out_of_cohort).truth == "FALSE"


def test_variables_with_no_declared_sentinel_policy_are_reported_not_assumed(cat, guides):
    """The NCCN file declares sentinels for histology and the two stage groups and not for
    primary_site or behavior, so exactly those two are named. Silence here would mean C809
    (unknown primary) quietly evaluating as an ordinary non-member."""
    d = _composition(cat, guides)
    g = next(g for g in d.gaps if g.kind == I.NO_SENTINEL_POLICY)
    assert set(g.subject.split(", ")) == {"primary_site", "behavior"}
    assert not g.refusing, "an undeclared sentinel is a warning about meaning, not a refusal"


def test_a_predicate_naming_a_field_no_spec_produces_is_refused_whole(cat, guides):
    d = _composition(cat, guides, terms=(
        {"op": "equals", "var": "histology", "value": "8140"},
        {"op": "equals", "var": "ajcc_pathologic_stage", "value": "IIB"}))
    assert d.refused
    assert d.predicate.checkable is False
    assert any(g.kind == I.UNKNOWN_FIELD_IN_PREDICATE for g in d.gaps)
    with pytest.raises(I.PredicateRefused):
        d.predicate.evaluate({"histology": {"status": "FOUND", "value": "8140"}})


def test_a_predicate_naming_a_value_the_spec_would_reject_is_refused(cat, guides):
    """`C3412` is the value that passed the gate stamped as validated against a declared
    `C\\d{3}`; `answer_checks.check_field_formats` was written for it and is the function this
    check calls. A predicate may not ask for a value the runtime would reject if an agent
    produced it."""
    d = _composition(cat, guides, terms=(
        {"op": "equals", "var": "primary_site", "value": "C3412"},
        {"op": "equals", "var": "behavior", "value": "9"}))
    assert d.refused
    subjects = {g.subject for g in d.gaps if g.kind == I.VALUE_OUTSIDE_DECLARED_DOMAIN}
    assert subjects == {"primary_site='C3412'", "behavior='9'"}


def test_a_pattern_that_can_never_match_is_refused_rather_than_run(cat, guides):
    """A term matching none of the field's allowable values empties the cohort by
    construction, and an empty cohort reports as a clean run with n=0."""
    d = _composition(cat, guides, terms=({"op": "matches", "var": "behavior", "pattern": "Z+"},))
    assert d.refused
    assert any(g.kind == I.UNSATISFIABLE_TERM for g in d.gaps)


def test_an_incomplete_composition_refuses_to_run_until_the_weakening_is_written_down(
        cat, guides):
    """"stage II-IIIA *resected* NSCLC" minus `surgical_resection_extent` is "stage II-IIIA
    NSCLC" — a larger cohort than the one asked for, with nothing on the artifact saying so."""
    d = _composition(cat, guides, missing=("surgical_resection_extent", "surgical_margins"))
    assert d.predicate.checkable is True
    assert d.predicate.complete is False
    assert "<NOT COMPUTABLE: surgical_resection_extent, surgical_margins>" in \
        d.predicate.expression()
    assert not d.refused, "an incomplete predicate is honest, not broken"

    found = {"primary_site": {"status": "FOUND", "value": "C341"},
             "histology": {"status": "FOUND", "value": "8140"},
             "behavior": {"status": "FOUND", "value": "3"},
             "pathologic_stage_group": {"status": "FOUND", "value": "IIB"}}
    with pytest.raises(I.PredicateRefused) as ei:
        d.predicate.evaluate(found)
    assert "surgical_resection_extent" in str(ei.value)
    assert d.predicate.evaluate(found, accept_partial=True).truth == "TRUE"
    assert {g.subject for g in d.gaps if g.kind == I.SPEC_AUTHORING_REQUIRED} == \
        {"surgical_resection_extent", "surgical_margins"}


def test_a_composition_with_no_terms_is_refused(cat, guides):
    """The outcome that lets judgement hide: "it is a composition" with nothing to check."""
    d = _composition(cat, guides, terms=())
    assert d.refused
    assert any(g.kind == I.MALFORMED_TERM for g in d.gaps)


def test_a_named_value_set_resolves_only_from_a_loaded_guideline(cat, guides):
    term = ({"op": "in_set", "var": "histology", "set": "nsclc_histology"},)
    assert _composition(cat, guides, terms=term).predicate.checkable is True
    bare = I.route("q", cat, classifier=stub(outcome=I.COMPOSITION, predicate=term))
    assert bare.refused
    assert any(g.kind == I.UNDECLARED_VALUE_SET for g in bare.gaps)


# ------------------------------------------------------------------ NEW_VARIABLE
@pytest.fixture
def new_var(cat):
    return I.route("was this patient a candidate for surgery", cat, classifier=stub(
        outcome=I.NEW_VARIABLE, proposed_variable={
            "name": "surgical_candidacy",
            "question": "Was this patient a candidate for surgical resection?",
            "why_not_composable": "operability is a judgement over comorbidity and PFTs; no "
                                  "combination of shipped variables expresses it"}))


def test_a_new_variable_emits_a_skeleton_and_routes_to_spec_authoring(new_var):
    assert new_var.outcome == I.NEW_VARIABLE
    assert new_var.predicate is None, "a judgemental variable is not a composition"
    assert new_var.skeleton.route == I.SPEC_AUTHORING_SKILL
    assert any(g.kind == I.SPEC_AUTHORING_REQUIRED for g in new_var.gaps)


def test_the_skeleton_deliberately_does_not_load(new_var, tmp_path):
    """A model-authored draft that loads is a model-authored draft that runs. All four original
    specs in this repo were written by a language model in one commit and no registrar has read
    a line of any of them; the placeholder in `data_source` makes pydantic the stop, rather
    than a convention someone has to remember."""
    p = tmp_path / "draft.yaml"
    p.write_text(new_var.skeleton.yaml_text, encoding="utf-8")
    with pytest.raises(Exception) as ei:
        load_spec(p)
    assert "data_source" in str(ei.value)


def test_the_skeleton_names_every_judgement_a_human_still_owes(new_var):
    keys = {q["key"] for q in new_var.skeleton.open_questions}
    assert {"evidence_rules.counts_as_evidence", "evidence_rules.does_not_count",
            "conflict_rules", "proof_obligation.for_negative", "abstention"} <= keys
    left = I.unfinished_placeholders(new_var.skeleton.yaml_text)
    assert "data_source" in left
    assert any(k.startswith("evidence_rules.does_not_count") for k in left)
    assert I.unfinished_placeholders({"a": {"b": "done"}}) == []


def test_a_skeleton_is_never_proposed_for_a_variable_that_already_has_a_spec(cat):
    """Two specs for one variable is how one patient gets two disagreeing answers — the
    two-ledger failure `state.py` already had to remove once."""
    d = I.route("code the histology", cat, classifier=stub(
        outcome=I.NEW_VARIABLE, proposed_variable={"name": "histology"}))
    assert d.skeleton is None
    assert d.outcome == I.EXISTING_VARIABLE
    g = next(g for g in d.gaps if g.kind == I.SPEC_AUTHORING_REQUIRED)
    assert SHB in g.context


# ------------------------------------------------------------------ GUIDELINE_RULE
def test_the_shipped_nccn_subset_routes_and_the_answer_is_mostly_gaps(cat, guides):
    """The measured state of the tree, asserted so it cannot quietly improve or rot. Six of
    twenty-seven inputs resolve. Reporting "3 recommendations routed" would be true and
    useless: nothing in this guideline can be scored."""
    d = I.route("NCCN-NSCLC-SUBSET", cat, guidelines=guides)
    assert d.outcome == I.GUIDELINE_RULE
    assert d.model_calls == 0, "input names are already written down; reading a list is not judgement"
    assert len(d.recommendation_ids) == 3
    by = {}
    for r in d.resolved:
        by.setdefault(r.outcome, []).append(r.name)
    assert sorted(by[I.EXISTING_VARIABLE]) == [
        "behavior", "clinical_stage_group", "date_of_initial_diagnosis", "histology",
        "pathologic_stage_group", "primary_site"]
    assert sorted(by[I.WRONG_DATA_SOURCE]) == ["class_of_case", "date_of_death"]
    assert len(by[I.NEW_VARIABLE]) == 19
    assert len(d.resolved) == 27
    assert sum(1 for g in d.gaps if g.kind == I.NOT_YET_EXTRACTABLE) == 19
    assert not d.refused, "an honest gap list is a successful routing, not a failure"


def test_every_guideline_input_carries_a_remedy_naming_who_does_what(cat, guides):
    d = I.route("NCCN-NSCLC-SUBSET", cat, guidelines=guides)
    assert d.gaps
    for g in d.gaps:
        assert g.remedy.strip(), f"gap {g.kind}/{g.subject} has no remedy"
        assert g.subject.strip() and g.detail.strip()


def test_one_recommendation_can_be_routed_alone(cat, guides):
    d = I.route("NSCLC-BIOMARKER-BEFORE-FIRST-LINE", cat, guidelines=guides)
    assert d.recommendation_ids == ("NSCLC-BIOMARKER-BEFORE-FIRST-LINE",)
    assert {r.name for r in d.resolved} == {
        "primary_site", "histology", "behavior", "class_of_case", "clinical_stage_group",
        "date_of_first_systemic_therapy", "date_of_egfr_result", "date_of_alk_result",
        "tissue_insufficient_for_molecular_testing", "patient_refused_molecular_testing"}


def test_naming_a_recommendation_the_guideline_does_not_have_is_reported(cat, guides):
    d = I.route_guideline(guides[0], cat, recommendation_ids=("NSCLC-NO-SUCH-RULE",))
    assert any(g.kind == I.ROUTE_TARGET_MISSING for g in d.gaps)


def test_a_guideline_naming_the_registry_for_something_a_spec_extracts_is_a_contradiction(
        cat, tmp_path):
    """Two places declare where a variable comes from, so the two can drift. This is the
    direction `check_guideline_bindings` does not cover."""
    g = parse_guideline({
        "guideline_id": "SYNTH", "recommendations": [{
            "id": "R1",
            "required_inputs": [{"name": "histology", "source": "registry_limited_dataset"},
                                {"name": "primary_site", "source": "not_yet_extractable"}],
            "applies_when": [{"op": "is_present", "var": "histology"}],
            "satisfied_when": [{"op": "is_present", "var": "primary_site"}]}]})
    d = I.route_guideline(g, cat)
    kinds = [x.kind for x in d.gaps if x.kind == I.GUIDELINE_SOURCE_DISAGREES_WITH_CATALOG]
    assert len(kinds) == 2, "both directions of the drift must be reported"
    stale = next(x for x in d.gaps if "stale" in x.detail)
    assert SHB in stale.context


def test_a_route_to_a_skill_that_is_not_in_the_tree_is_reported_not_printed_as_advice(
        cat, guides, tmp_path):
    """Every NEW_VARIABLE outcome tells a human to go to `skills/spec-authoring`. An
    instruction pointing at a directory that does not exist is decoration that looks like a
    process — this repo's most-repeated failure."""
    d = I.route("NCCN-NSCLC-SUBSET", cat, guidelines=guides, skills_dir=tmp_path)
    assert any(g.kind == I.ROUTE_TARGET_MISSING and g.subject == I.SPEC_AUTHORING_SKILL
               for g in d.gaps)

    made = tmp_path / "spec-authoring"
    made.mkdir()
    (made / "SKILL.md").write_text("---\nname: spec-authoring\ndescription: x\n---\n",
                                   encoding="utf-8")
    d2 = I.route("NCCN-NSCLC-SUBSET", cat, guidelines=guides, skills_dir=tmp_path)
    assert not any(g.kind == I.ROUTE_TARGET_MISSING for g in d2.gaps)


def test_a_guideline_that_is_not_loaded_is_a_gap_not_a_guess(cat):
    d = I.route("q", cat, classifier=stub(outcome=I.GUIDELINE_RULE, guideline_id="NCCN-OTHER"))
    assert any(g.kind == I.ROUTE_TARGET_MISSING for g in d.gaps)
    assert d.resolved == ()


def test_load_guidelines_tolerates_an_absent_directory_but_not_a_broken_file(tmp_path):
    assert I.load_guidelines(tmp_path / "nope") == []
    (tmp_path / "bad.yaml").write_text("guideline_id: X\nrecommendations: []\n", encoding="utf-8")
    with pytest.raises(Exception):
        I.load_guidelines(tmp_path)


# ------------------------------------------------------------------ the model boundary
class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeClient:
    """Records what the classifier sent and returns a canned completion. No provider."""

    def __init__(self, content):
        self.content, self.sent, self.calls = content, [], 0
        self.cfg = type("C", (), {"model": "fake/one"})()

    def chat(self, messages, tools=None):
        self.sent.append(messages)
        self.calls += 1
        return FakeResp(self.content)


def test_the_classifier_prompt_carries_the_vocabulary_and_nothing_about_a_patient(cat, guides):
    c = FakeClient("{}")
    mc = I.ModelClassifier(c)
    msgs = mc.messages("does this patient have stage II-IIIA lung adeno",
                       I.vocabulary(cat, guides))
    blob = json.dumps(msgs)
    assert "pathologic_stage_group" in blob and "NCCN-NSCLC-SUBSET" in blob
    assert all(k in blob for k in I.OUTCOMES)
    assert "corpus" not in blob and "note_id" not in blob


def test_a_leaked_preamble_object_does_not_lose_the_classification(cat, guides):
    """gpt-5.6-luna leaks its tool-call channel into the text channel and emits a preamble
    object before the answer; `llm.extract_json` documents the trace. "First parseable object
    wins" returns the preamble, whose outcome is None, and a perfectly good classification is
    recorded as a failure."""
    c = FakeClient('{"search":"stage"}\n{"outcome":"COMPOSITION","rationale":"r",'
                   '"predicate":[{"op":"equals","var":"histology","value":"8140"}]}')
    cls = I.ModelClassifier(c).classify("q", I.vocabulary(cat, guides))
    assert cls.outcome == I.COMPOSITION
    assert cls.model_calls == 1 and c.calls == 1
    d = I.route("q", cat, classifier=I.StubClassifier(cls), guidelines=guides)
    assert d.outcome == I.COMPOSITION and d.predicate.checkable


def test_an_outcome_outside_the_five_is_not_accepted(cat, guides):
    c = FakeClient('{"outcome":"MAYBE","rationale":""}')
    cls = I.ModelClassifier(c).classify("q", I.vocabulary(cat, guides))
    assert cls.outcome is None
    assert "MAYBE" in cls.rationale
    assert I.route("q", cat, classifier=I.StubClassifier(cls)).refused


def test_the_vocabulary_truncates_long_value_lists_visibly(cat, guides):
    v = I.vocabulary(cat, guides)
    coc = next(e for e in v["variables"] if e["name"] == "class_of_case")
    assert len(coc["allowable_values"]) == 21 and coc["allowable_values"][-1] == "…"


# ------------------------------------------------------------------ the CLI
def test_ask_routes_a_name_and_exits_zero(tmp_path):
    r = runner.invoke(app, ["ask", "primary_site", "--out", str(tmp_path / "r.json")])
    assert r.exit_code == 0, r.output
    doc = json.loads((tmp_path / "r.json").read_text())
    assert doc["schema"] == "acr.routing/1"
    assert doc["routing"]["outcome"] == I.EXISTING_VARIABLE
    assert doc["routing"]["model_calls"] == 0
    assert doc["routing"]["reads_charts"] is False


def test_ask_prints_the_whole_nccn_gap_list(tmp_path):
    r = runner.invoke(app, ["ask", "NCCN-NSCLC-SUBSET", "--out", str(tmp_path / "r.json")])
    assert r.exit_code == 0, r.output
    gaps = json.loads((tmp_path / "r.json").read_text())["routing"]["gaps"]
    assert len(gaps) >= 22
    assert "23 GAP" in r.output or f"{len(gaps)} GAP" in r.output
    assert "not_yet_extractable" in r.output


def test_ask_exits_non_zero_when_a_route_is_refused():
    r = runner.invoke(app, ["ask", "does this patient have resected NSCLC"])
    assert r.exit_code == 2
    assert "unclassified" in r.output


def test_no_dry_run_is_refused_and_names_the_command_that_does_the_work():
    """The flag exists so the default is visible. There is no second mode: wiring execution in
    here would put a command that reads PHI behind a flag whose default is the only safe
    value, and defaults get overridden in scripts."""
    r = runner.invoke(app, ["ask", "primary_site", "--no-dry-run"])
    assert r.exit_code == 2
    assert "acr extract" in r.output


def test_ask_json_output_is_parseable(tmp_path):
    r = runner.invoke(app, ["ask", "class_of_case", "--json"])
    assert r.exit_code == 0, r.output
    assert '"WRONG_DATA_SOURCE"' in r.output
