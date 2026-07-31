"""L4.5: how a guideline rule knows which variables it reads — in both directions.

The property under test is not "a dependency graph gets built". It is that neither
direction is allowed to be quiet, because both have a silent failure mode that produces a
number rather than an error:

  FORWARD    an input the guideline declares and nothing can supply is a GAP that stops the
             rule. It must not arrive as an absent variable and let the remaining conditions
             decide the case anyway — that is how `any_of` returns CONCORDANT for a
             recommendation whose action variable was never wired up.

  EXCEPTIONS a recommendation with no exceptions must say so and say WHY. "There are none"
             and "we forgot" are byte-identical YAML unless something forces them apart, and
             the difference between them is a patient who declined chemotherapy counted as a
             care gap. That is failure D of the design doc and the single most common way a
             concordance study produces a wrong and damaging number.

  BACKWARD   editing a spec must make every concordance result computed under the old spec
             LOOK wrong. A stale CONCORDANT is worse than no answer, because no answer
             announces itself and a stale CONCORDANT does not.

Nothing here asserts that the shipped guideline has few gaps. It has 21, and a test that
tolerated fewer would be pressure to hide them.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

# src/acr/deps.py landed on 2026-07-27; the `pytest.importorskip` guard that stood here while
# it did not exist is gone, as its own comment instructed. A skipped specification is a
# specification nobody is held to.
import acr.contract.deps as D
from acr.commands.cli import CONCORD_SCHEMA as CLI_CONCORD_SCHEMA
from acr.commands.cli import EXTRACT_SCHEMA as CLI_EXTRACT_SCHEMA
from acr.commands.cli import app
from acr.contract.concordance import _VAR_KEYS, load_guideline, parse_guideline
from acr.contract.registry_catalog import VariableCatalog

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "assets" / "specs"
GUIDELINE = ROOT / "assets" / "guidelines" / "nccn_nsclc_subset.yaml"

runner = CliRunner()

REAL_RECS = ("NSCLC-ADJ-SYSTEMIC-II-IIIA", "NSCLC-BIOMARKER-BEFORE-FIRST-LINE",
             "NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY")


@pytest.fixture(scope="module")
def cat() -> VariableCatalog:
    return VariableCatalog.from_directory(SPECS)


@pytest.fixture(scope="module")
def shipped(cat) -> "D.GuidelineDeps":
    return D.build_dependencies(load_guideline(GUIDELINE), cat)


# --------------------------------------------------------------------- synthetic fixtures
def _toy_guideline(**rec_overrides) -> dict:
    """A one-recommendation guideline that loads, so a test can break exactly one thing.

    Deliberately not a copy of the NCCN file: a test that has to edit 20KB of real clinical
    content to change one field ends up asserting against whatever the other workflow last
    wrote into that file.
    """
    rec = {
        "id": "TOY-1",
        "title": "toy",
        "required_inputs": [
            {"name": "histology", "source": "extraction_spec",
             "spec_id": "STORE.400_522_523.site_histology_behavior", "item": "STORE [522]"},
            {"name": "treatment_given", "source": "not_yet_extractable", "item": "n/a"},
        ],
        "applies_when": [{"op": "in_set", "var": "histology", "set": "toy_set"}],
        "satisfied_when": [{"op": "is_true", "var": "treatment_given"}],
        "exceptions": [{"id": "refused", "label": "declined",
                        "when": [{"op": "is_true", "var": "treatment_given"}]}],
    }
    rec.update(rec_overrides)
    return {"guideline_id": "TOY", "guideline_version": "0.0.1",
            "value_sets": {"toy_set": ["8140"]}, "recommendations": [rec]}


def _toy_deps(cat, **rec_overrides) -> "D.GuidelineDeps":
    return D.build_dependencies(parse_guideline(_toy_guideline(**rec_overrides)), cat)


# ===================================================================== FORWARD: resolution
def test_every_declared_input_lands_in_exactly_one_bucket(shipped):
    """resolved + gaps partitions required_inputs. A declared input that appears in neither
    is an input the impact analysis will never notice changing."""
    for rd in shipped.per_recommendation:
        declared = [str(d.get("name")) for d in shipped.guideline.recommendation(
            rd.recommendation_id).required_inputs]
        assert sorted(rd.resolved + [g["name"] for g in rd.gaps]) == sorted(declared)
        assert len(set(rd.resolved) & {g["name"] for g in rd.gaps}) == 0


def test_the_shipped_guideline_declares_thirtyeight_inputs_of_which_twentyone_are_gaps(shipped):
    """The honest number, pinned. This assertion exists to be BROKEN UPWARDS by someone
    building an extractor, and to fail loudly if a gap is ever quietly reclassified as
    resolved without an extractor behind it."""
    total = sum(len(rd.inputs) for rd in shipped.per_recommendation)
    gaps = sum(len(rd.gaps) for rd in shipped.per_recommendation)
    assert (total, gaps) == (38, 21)
    assert all(g["kind"] == D.GAP_NOT_YET_EXTRACTABLE
               for rd in shipped.per_recommendation for g in rd.gaps)


def test_an_extraction_sourced_input_resolves_to_exactly_one_spec_field(shipped):
    """`registry_catalog` is the only router. Re-deriving spec_id -> field here would be a
    second copy free to disagree with the one `extract` actually runs."""
    by_name = {i.name: i for rd in shipped.per_recommendation for i in rd.inputs}
    assert by_name["pathologic_stage_group"].spec_id == "STORE.700_880.stage"
    assert by_name["primary_site"].spec_id == "STORE.400_522_523.site_histology_behavior"
    assert by_name["date_of_initial_diagnosis"].spec_id == "STORE.390.date_of_initial_diagnosis"
    for i in by_name.values():
        if i.source == D.SOURCE_EXTRACTION_SPEC and i.resolved:
            assert i.spec_hash and len(i.spec_hash) == 16


def test_a_registry_limited_dataset_item_resolves_without_a_spec(shipped):
    """`class_of_case` is not a gap. STORE.610 exists and answers SPEC_INSUFFICIENT by
    design; the value comes from the registry feed, which is a supply route, not a hole."""
    adj = shipped.for_recommendation("NSCLC-ADJ-SYSTEMIC-II-IIIA")
    coc = next(i for i in adj.inputs if i.name == "class_of_case")
    assert coc.resolved and coc.source == D.SOURCE_REGISTRY and coc.spec_id is None
    assert "class_of_case" in adj.resolved


def test_a_misspelled_spec_field_is_a_gap_naming_the_spec_it_missed(cat):
    """The C3412-shaped bug at the binding layer: `ajcc_pathologic_stage` is written down,
    nothing produces it, and without this check every case comes back NOT_ASSESSABLE naming
    a variable the operator believes was extracted."""
    d = _toy_deps(cat, required_inputs=[
        {"name": "ajcc_pathologic_stage", "source": "extraction_spec",
         "spec_id": "STORE.700_880.stage", "item": "x"},
        {"name": "treatment_given", "source": "not_yet_extractable", "item": "n/a"}],
        applies_when=[{"op": "is_present", "var": "ajcc_pathologic_stage"}])
    gap = next(g for g in d.for_recommendation("TOY-1").gaps
               if g["name"] == "ajcc_pathologic_stage")
    assert gap["kind"] == D.GAP_NO_SUCH_SPEC_FIELD
    assert "pathologic_stage_group" in gap["detail"]


def test_a_declared_spec_id_that_does_not_own_the_field_is_a_gap(cat):
    """The name resolves, but to a different spec than the guideline says. Trusting the name
    and ignoring the declaration would bind the rule to a spec no reviewer approved."""
    d = _toy_deps(cat, required_inputs=[
        {"name": "histology", "source": "extraction_spec",
         "spec_id": "STORE.390.date_of_initial_diagnosis", "item": "STORE [522]"},
        {"name": "treatment_given", "source": "not_yet_extractable", "item": "n/a"}])
    gap = next(g for g in d.for_recommendation("TOY-1").gaps if g["name"] == "histology")
    assert gap["kind"] == D.GAP_SPEC_ID_MISMATCH
    assert "STORE.400_522_523.site_histology_behavior" in gap["detail"]


def test_an_input_bound_to_an_outside_notes_spec_is_a_gap_not_a_resolution(cat):
    """STORE.610 is `data_source: outside_notes`, so every agent run over it is forced to
    SPEC_INSUFFICIENT / WRONG_DATA_SOURCE at finalize. Calling that binding `resolved`
    promises a variable that provably never arrives."""
    d = _toy_deps(cat, required_inputs=[
        {"name": "class_of_case", "source": "extraction_spec",
         "spec_id": "STORE.610.class_of_case", "item": "STORE [610]"},
        {"name": "treatment_given", "source": "not_yet_extractable", "item": "n/a"}],
        applies_when=[{"op": "is_present", "var": "class_of_case"}])
    gap = next(g for g in d.for_recommendation("TOY-1").gaps if g["name"] == "class_of_case")
    assert gap["kind"] == D.GAP_SPEC_CANNOT_ANSWER_FROM_NOTES
    assert "registry_limited_dataset" in gap["detail"]


def test_an_unrecognised_source_is_a_gap_and_never_a_pass(cat):
    """`parse_guideline` does not validate, so a typo'd source can reach here. Defaulting it
    to anything is the resolver inventing a supply route."""
    d = _toy_deps(cat, required_inputs=[
        {"name": "histology", "source": "chart_review_by_hand", "item": "x"},
        {"name": "treatment_given", "source": "not_yet_extractable", "item": "n/a"}])
    gap = next(g for g in d.for_recommendation("TOY-1").gaps if g["name"] == "histology")
    assert gap["kind"] == D.GAP_UNKNOWN_SOURCE


# ============================================================ FORWARD: predicate classes
def test_every_declared_input_carries_at_least_one_predicate_class(shipped):
    for rd in shipped.per_recommendation:
        for i in rd.inputs:
            assert i.predicate_classes, f"{rd.recommendation_id}:{i.name} is read by nothing"


def test_classes_split_eligibility_action_timing_exception(shipped):
    adj = shipped.for_recommendation("NSCLC-ADJ-SYSTEMIC-II-IIIA").predicate_classes
    assert adj["primary_site"] == [D.ELIGIBILITY]
    assert adj["pathologic_stage_group"] == [D.ELIGIBILITY]
    assert adj["adjuvant_systemic_therapy_class"] == [D.ACTION]
    assert adj["patient_refused_adjuvant_systemic_therapy"] == [D.EXCEPTION]
    assert adj["date_of_first_adjuvant_systemic_therapy"] == [D.ACTION, D.TIMING]


def test_a_variable_read_by_two_blocks_carries_both_classes(shipped):
    """`date_of_definitive_surgery` opens the adjuvant window in satisfied_when AND anchors
    the died-before-the-window exception. Assigning a shared variable to one block would
    drop a driving variable from the other, which is how L5 reports cause B eliminated when
    the missing thing was the whole question. Same reasoning as the comment in `cli.concord`.
    """
    adj = shipped.for_recommendation("NSCLC-ADJ-SYSTEMIC-II-IIIA").predicate_classes
    assert adj["date_of_definitive_surgery"] == [D.ACTION, D.TIMING, D.EXCEPTION]
    assert adj["date_of_death"] == [D.TIMING, D.EXCEPTION]


def test_a_date_endpoint_inside_applies_when_is_eligibility_not_timing(shipped):
    """`is_present(date_of_first_systemic_therapy)` is a population test — the trigger event
    happened — not a window. Classifying by the variable's type instead of the op it sits in
    would call it timing and put it in the wrong half of the L5 split."""
    bio = shipped.for_recommendation("NSCLC-BIOMARKER-BEFORE-FIRST-LINE").predicate_classes
    assert bio["date_of_first_systemic_therapy"] == [D.ELIGIBILITY, D.ACTION, D.TIMING]
    assert bio["date_of_egfr_result"] == [D.ACTION, D.TIMING]


def test_classes_reach_inside_any_of(shipped):
    """`satisfied_when` for stage I is an `any_of`. A scan that stopped at the top level
    would classify neither branch and report both variables as read by nothing."""
    s1 = shipped.for_recommendation("NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY").predicate_classes
    assert s1["radiotherapy_intent"] == [D.ACTION]
    assert s1["surgical_resection_extent"] == [D.ACTION]


def test_every_temporal_op_in_the_engine_is_known_to_this_module():
    """Tripwire, not a duplicate. `concordance._VAR_KEYS` is the engine's list of ops that
    name their variables with from/to; every one of them today is a date comparison. If a
    new op is added there and not here it would be classified as an ordinary predicate and
    its variables would silently lose the `timing` class."""
    assert set(_VAR_KEYS) <= D.TEMPORAL_OPS


# ================================================================= THE EXCEPTION RULE
def test_all_three_shipped_recommendations_declare_their_exceptions(shipped):
    got = {rd.recommendation_id: list(rd.exceptions_declared) for rd in shipped.per_recommendation}
    assert set(got) == set(REAL_RECS)
    assert all(v for v in got.values())
    assert "patient_refused" in got["NSCLC-ADJ-SYSTEMIC-II-IIIA"]


def test_a_recommendation_with_no_exceptions_and_no_reason_refuses_to_load(cat, tmp_path):
    """The whole point. Silence here is indistinguishable from "we forgot", and the cost of
    getting it wrong is a patient who declined treatment counted as a care gap."""
    doc = _toy_guideline()
    doc["recommendations"][0].pop("exceptions")
    doc["recommendations"][0]["satisfied_when"] = [{"op": "is_present", "var": "treatment_given"}]
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(D.UndeclaredExceptionsError) as ei:
        D.load_guideline_deps(p, specs_dir=SPECS)
    assert "TOY-1" in str(ei.value)
    assert D.NONE_DECLARED_KEY in str(ei.value)


def test_an_empty_exceptions_list_is_the_same_silence_and_is_refused(cat, tmp_path):
    doc = _toy_guideline(exceptions=[])
    doc["recommendations"][0]["satisfied_when"] = [{"op": "is_present", "var": "treatment_given"}]
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(D.UndeclaredExceptionsError):
        D.load_guideline_deps(p, specs_dir=SPECS)


def test_none_declared_with_a_real_reason_loads(cat, tmp_path):
    doc = _toy_guideline()
    doc["recommendations"][0].pop("exceptions")
    doc["recommendations"][0]["satisfied_when"] = [{"op": "is_present", "var": "treatment_given"}]
    doc["recommendations"][0][D.NONE_DECLARED_KEY] = (
        "This recommendation scores whether a test RESULT existed before therapy started; a "
        "patient cannot decline the ordering of two dates, so there is no refusal exception.")
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    d = D.load_guideline_deps(p, specs_dir=SPECS)
    rd = d.for_recommendation("TOY-1")
    assert rd.exceptions_declared == ()
    assert "cannot decline" in rd.exceptions_none_declared_reason


def test_none_declared_accepts_a_mapping_so_a_reviewer_can_be_named(cat, tmp_path):
    doc = _toy_guideline()
    doc["recommendations"][0].pop("exceptions")
    doc["recommendations"][0]["satisfied_when"] = [{"op": "is_present", "var": "treatment_given"}]
    doc["recommendations"][0][D.NONE_DECLARED_KEY] = {
        "reason": "No exception can apply: the recommendation is about the ORDER of two "
                  "recorded dates and neither date is a decision a patient makes.",
        "reviewed_by": "thoracic oncology, pending"}
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert D.load_guideline_deps(p, specs_dir=SPECS).for_recommendation(
        "TOY-1").exceptions_none_declared_reason.startswith("No exception can apply")


@pytest.mark.parametrize("shrug", ["", "none", "N/A", "none_declared", "  ", "TBD"])
def test_a_shrug_is_not_a_reason(cat, tmp_path, shrug):
    """"none" restates the field name. The reason has to say why none is CORRECT, because
    that sentence is the only thing a clinical reviewer can disagree with."""
    doc = _toy_guideline()
    doc["recommendations"][0].pop("exceptions")
    doc["recommendations"][0]["satisfied_when"] = [{"op": "is_present", "var": "treatment_given"}]
    doc["recommendations"][0][D.NONE_DECLARED_KEY] = shrug
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(D.UndeclaredExceptionsError) as ei:
        D.load_guideline_deps(p, specs_dir=SPECS)
    assert "reason" in str(ei.value)


def test_declaring_exceptions_and_none_declared_at_once_is_a_contradiction(cat, tmp_path):
    doc = _toy_guideline()
    doc["recommendations"][0][D.NONE_DECLARED_KEY] = (
        "There are no exceptions to this recommendation under any circumstance whatsoever.")
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(D.UndeclaredExceptionsError) as ei:
        D.load_guideline_deps(p, specs_dir=SPECS)
    assert "both" in str(ei.value).lower()


def test_the_sentinel_may_not_be_written_into_the_exceptions_key_itself(cat, tmp_path):
    """`exceptions: none_declared` is the natural spelling and it is REFUSED, with the reason
    attached: `concordance.parse_guideline` iterates `exceptions`, so a bare string there
    makes the file unloadable by the scorer — it would iterate the characters of the word.
    A grammar that only this module can read is a second guideline format."""
    doc = _toy_guideline(exceptions=D.NONE_DECLARED)
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(D.UndeclaredExceptionsError) as ei:
        D.load_guideline_deps(p, specs_dir=SPECS)
    assert D.NONE_DECLARED_KEY in str(ei.value)
    assert "parse_guideline" in str(ei.value)


def test_the_shipped_guideline_still_loads_through_the_deps_loader():
    """The exception rule must not be satisfiable only by toys."""
    d = D.load_guideline_deps(GUIDELINE, specs_dir=SPECS)
    assert [rd.recommendation_id for rd in d.per_recommendation] == list(REAL_RECS)


# ================================================== a gap stops the rule from evaluating
def _known(v):
    return {"status": "FOUND", "value": v}


NSCLC_ELIGIBLE = {
    "primary_site": _known("C341"), "histology": _known("8140"), "behavior": _known("3"),
    "class_of_case": _known("10"), "clinical_stage_group": _known("IA"),
    "pathologic_stage_group": _known("IIB"),
    "date_of_initial_diagnosis": _known("2021-01-04"),
}


def test_a_gapped_recommendation_can_never_come_back_concordant(shipped):
    """The failure this gate exists for. `satisfied_when` for stage I is an `any_of`: feed
    it a resection and the branch is TRUE, so the rule reports CONCORDANT while
    `date_of_first_definitive_local_therapy` — a declared input with no extractor — was
    never wired. Care that was never checked must not be scored as care that was delivered.
    """
    variables = dict(NSCLC_ELIGIBLE)
    variables["surgical_resection_extent"] = _known("lobectomy")
    variables["date_of_first_definitive_local_therapy"] = _known("2021-02-01")
    r = {x.recommendation_id: x for x in D.gated_assess(variables, shipped)}
    stage1 = r["NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY"]
    assert stage1.outcome == "NOT_ASSESSABLE"
    assert stage1.rule_applied == D.RULE_DEPENDENCY_GAP
    assert "radiotherapy_intent" in stage1.blocking_inputs
    assert not any(x.outcome in ("CONCORDANT", "NON_CONCORDANT", "EXCEPTION_DOCUMENTED")
                   for x in r.values())


def test_the_gap_verdict_names_every_unresolved_input_not_just_the_first(shipped):
    variables = dict(NSCLC_ELIGIBLE)
    out = {x.recommendation_id: x for x in D.gated_assess(variables, shipped)}
    for rd in shipped.per_recommendation:
        got = out[rd.recommendation_id]
        if got.outcome == "NOT_APPLICABLE":
            continue
        assert set(g["name"] for g in rd.gaps) <= set(got.blocking_inputs)


def test_a_determinate_not_applicable_survives_a_gap_and_says_the_gap_was_there(shipped):
    """The one verdict a gap may not overturn, and the reason is arithmetic, not policy: a
    gapped variable never arrives, an absent variable is UNKNOWN, and UNKNOWN cannot produce
    FALSE under `_and`. So a NOT_APPLICABLE was settled by a variable that DID arrive, and
    no unbuilt extractor can make small cell carcinoma into NSCLC. Forcing NOT_ASSESSABLE
    here would move determinately-out-of-population patients into the unknown pile, which
    inflates the unknowns exactly the way scoring them would inflate the rate.
    """
    variables = dict(NSCLC_ELIGIBLE) | {"histology": _known("8041")}   # small cell
    out = {x.recommendation_id: x for x in D.gated_assess(variables, shipped)}
    adj = out["NSCLC-ADJ-SYSTEMIC-II-IIIA"]
    assert adj.outcome == "NOT_APPLICABLE"
    assert any(D.RULE_DEPENDENCY_GAP in n for n in adj.notes)


def test_gated_assess_agrees_with_the_engine_when_there_are_no_gaps(cat):
    """The gate adds a refusal; it must not add an opinion. With every input resolved the
    verdict has to be byte-for-byte the engine's."""
    from acr.contract.concordance import assess
    doc = _toy_guideline(required_inputs=[
        {"name": "histology", "source": "extraction_spec",
         "spec_id": "STORE.400_522_523.site_histology_behavior", "item": "STORE [522]"},
        {"name": "behavior", "source": "registry_limited_dataset", "item": "STORE [523]"}],
        satisfied_when=[{"op": "equals", "var": "behavior", "value": "3"}],
        exceptions=[{"id": "x", "when": [{"op": "equals", "var": "behavior", "value": "2"}]}])
    g = parse_guideline(doc)
    d = D.build_dependencies(g, cat)
    assert d.for_recommendation("TOY-1").gaps == []
    v = {"histology": _known("8140"), "behavior": _known("3")}
    assert [r.to_dict() for r in D.gated_assess(v, d)] == [r.to_dict() for r in assess(v, g)]


def test_a_gap_variable_that_arrives_anyway_is_reported_as_a_stale_declaration(shipped):
    """An operator can supply a `not_yet_extractable` input through --extra-variables. The
    data is real; the DECLARATION is now wrong, and a wrong declaration is what the whole
    forward direction is enforcing. Say so rather than either trusting or discarding it."""
    variables = dict(NSCLC_ELIGIBLE) | {"date_of_egfr_result": _known("2021-01-02")}
    out = {x.recommendation_id: x for x in D.gated_assess(variables, shipped)}
    notes = " ".join(out["NSCLC-BIOMARKER-BEFORE-FIRST-LINE"].notes)
    assert "date_of_egfr_result" in notes and "not_yet_extractable" in notes


# ======================================================== BACKWARD: variable -> rules
def test_backward_map_points_from_a_spec_to_the_recommendations_that_read_it(shipped):
    back = shipped.backward()
    assert set(back["STORE.700_880.stage"]["recommendations"]) == set(REAL_RECS)
    assert set(back["STORE.390.date_of_initial_diagnosis"]["recommendations"]) == {
        "NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY"}
    assert back["STORE.700_880.stage"]["inputs"] == [
        ["NSCLC-ADJ-SYSTEMIC-II-IIIA", "pathologic_stage_group"],
        ["NSCLC-BIOMARKER-BEFORE-FIRST-LINE", "clinical_stage_group"],
        ["NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY", "clinical_stage_group"]]


def test_a_spec_no_recommendation_reads_is_reported_as_unused_not_missing(shipped):
    """`STORE.1860_1880.first_recurrence` ships and no NCCN rule reads it. Raising KeyError
    would make "nothing depends on this" indistinguishable from "you typo'd the spec id"."""
    u = shipped.for_spec("STORE.1860_1880.first_recurrence")
    assert u.recommendations == () and u.current_hash
    with pytest.raises(KeyError):
        shipped.for_spec("STORE.9999.not_a_spec")


# ======================================================== BACKWARD: hash invalidation
def _fake_pipeline(tmp_path: Path, specs_dir: Path, guideline_path: Path,
                   *, inline_provenance: bool) -> Path:
    """An extract.json + concord.json pair in the shapes `cli.py` actually writes.

    Patient ids are P01..P05. No real id, date or note text goes into a file, in the repo or
    out of it.
    """
    from acr.contract.spec import load_specs
    specs = load_specs(specs_dir)
    g = load_guideline(guideline_path)
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    extract = {
        "schema": CLI_EXTRACT_SCHEMA,
        "specs": {sid: s.identity() for sid, s in specs.items()},
        "patients": [{"patient_id": p, "variables": {}, "answers": {}} for p in
                     ("P01", "P02", "P03", "P04", "P05")],
    }
    (run / "extract.json").write_text(json.dumps(extract, indent=2), encoding="utf-8")
    concord = {
        "schema": CLI_CONCORD_SCHEMA,
        "guideline": {"path": str(guideline_path), "guideline_id": g.guideline_id,
                      "guideline_version": g.guideline_version,
                      "guideline_hash": g.guideline_hash},
        "extract_input": str(run / "extract.json"),
        "recommendations": {r.id: {"title": r.title} for r in g.recommendations},
        "patients": [{"patient_id": p,
                      "results": [{"recommendation_id": r.id, "outcome": "CONCORDANT"}
                                  for r in g.recommendations]}
                     for p in ("P01", "P02", "P03", "P04", "P05")],
    }
    if inline_provenance:
        concord[D.PROVENANCE_KEY] = D.provenance_block(g, specs)
    (run / "concord.json").write_text(json.dumps(concord, indent=2), encoding="utf-8")
    return run / "concord.json"


@pytest.fixture()
def sandbox(tmp_path):
    """A copy of assets/specs/ and the guideline that a test may edit. The repo's own files are
    owned by another workflow and must not be touched to prove a point about hashing."""
    sd = tmp_path / "assets" / "specs"
    shutil.copytree(SPECS, sd)
    for junk in sd.glob("*/"):
        shutil.rmtree(junk)
    gp = tmp_path / "guideline.yaml"
    shutil.copy(GUIDELINE, gp)
    return sd, gp


def test_the_schema_strings_this_module_scans_for_are_the_ones_the_cli_writes():
    """Tripwire. deps.py cannot import cli.py (cli.py imports deps.py), so the two schema
    constants are written twice and could drift; a drifted constant makes every artifact
    invisible, and invisible reads as "nothing is stale"."""
    assert (D.CONCORD_SCHEMA, D.EXTRACT_SCHEMA) == (CLI_CONCORD_SCHEMA, CLI_EXTRACT_SCHEMA)


@pytest.mark.parametrize("inline", [True, False])
def test_an_untouched_artifact_is_current(sandbox, tmp_path, inline):
    sd, gp = sandbox
    art = _fake_pipeline(tmp_path, sd, gp, inline_provenance=inline)
    d = D.load_guideline_deps(gp, specs_dir=sd)
    v = D.classify_artifact(art, d)
    assert v.verdict == D.CURRENT, v.reason


@pytest.mark.parametrize("inline", [True, False])
def test_editing_a_spec_invalidates_every_concordance_result_computed_from_it(
        sandbox, tmp_path, inline):
    """The headline property. The concord.json is not rewritten, not deleted and not
    touched — the same bytes on disk must stop reading as current the moment the spec they
    were computed from changes."""
    sd, gp = sandbox
    art = D.classify_artifact(_fake_pipeline(tmp_path, sd, gp, inline_provenance=inline),
                              D.load_guideline_deps(gp, specs_dir=sd))
    assert art.verdict == D.CURRENT

    target = sd / "STORE.700_880.stage.yaml"
    doc = yaml.safe_load(target.read_text(encoding="utf-8"))
    doc.setdefault("search_hints", []).append("restaging note")
    target.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    after = D.classify_artifact(tmp_path / "run" / "concord.json",
                                D.load_guideline_deps(gp, specs_dir=sd))
    assert after.verdict == D.STALE
    assert [c["id"] for c in after.changed] == ["STORE.700_880.stage"]
    assert after.affected_results == 15          # 5 patients x 3 recommendations
    assert after.affected_outcomes == {"CONCORDANT": 15}


def test_editing_the_guideline_invalidates_too(sandbox, tmp_path):
    sd, gp = sandbox
    _fake_pipeline(tmp_path, sd, gp, inline_provenance=True)
    doc = yaml.safe_load(gp.read_text(encoding="utf-8"))
    doc["recommendations"][0]["source"]["operationalisation"][
        "max_days_surgery_to_adjuvant"] = 90
    gp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    after = D.classify_artifact(tmp_path / "run" / "concord.json",
                                D.load_guideline_deps(gp, specs_dir=sd))
    assert after.verdict == D.STALE
    assert any(c["kind"] == "guideline" for c in after.changed)


def test_an_artifact_whose_provenance_cannot_be_reached_is_unverifiable_not_current(
        sandbox, tmp_path):
    """The dangerous middle state. The concord.json records the guideline hash inline but
    reaches its spec hashes through extract_input; move or delete the extract and the spec
    identity is simply gone. Reporting that as CURRENT is the exact lie this module exists
    to stop, and reporting it as STALE would cry wolf. It gets its own verdict."""
    sd, gp = sandbox
    art = _fake_pipeline(tmp_path, sd, gp, inline_provenance=False)
    (tmp_path / "run" / "extract.json").unlink()
    v = D.classify_artifact(art, D.load_guideline_deps(gp, specs_dir=sd))
    assert v.verdict == D.UNVERIFIABLE
    assert "extract" in v.reason


def test_an_artifact_missing_one_of_the_specs_its_rules_read_is_unverifiable(sandbox, tmp_path):
    """Provenance that covers three of four specs cannot certify the fourth. Checking only
    what happens to be recorded would let a spec drop out of the manifest and take its own
    staleness check with it."""
    sd, gp = sandbox
    art = _fake_pipeline(tmp_path, sd, gp, inline_provenance=True)
    doc = json.loads(art.read_text(encoding="utf-8"))
    doc[D.PROVENANCE_KEY]["specs"].pop("STORE.700_880.stage")
    art.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    v = D.classify_artifact(art, D.load_guideline_deps(gp, specs_dir=sd))
    assert v.verdict == D.UNVERIFIABLE
    assert "STORE.700_880.stage" in v.reason


def test_impact_of_a_spec_scans_the_tree_and_reports_each_artifact(sandbox, tmp_path):
    sd, gp = sandbox
    _fake_pipeline(tmp_path, sd, gp, inline_provenance=True)
    d = D.load_guideline_deps(gp, specs_dir=sd)
    imp = D.impact_of_spec("STORE.700_880.stage", d, tmp_path)
    assert imp.recommendations == REAL_RECS
    assert len(imp.artifacts) == 1 and imp.artifacts[0].verdict == D.CURRENT

    target = sd / "STORE.700_880.stage.yaml"
    doc = yaml.safe_load(target.read_text(encoding="utf-8"))
    doc["spec_version"] = "0.2.0"
    target.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    imp2 = D.impact_of_spec("STORE.700_880.stage", D.load_guideline_deps(gp, specs_dir=sd),
                            tmp_path)
    assert [a.verdict for a in imp2.artifacts] == [D.STALE]
    assert imp2.stale_results == 15


def test_a_renamed_artifact_is_still_found_by_its_schema(sandbox, tmp_path):
    """`concord --out` takes any filename. A filename glob would miss it, and a missed
    artifact is reported as nothing rather than as stale."""
    sd, gp = sandbox
    art = _fake_pipeline(tmp_path, sd, gp, inline_provenance=True)
    art.rename(art.with_name("2021_cohort_scored.json"))
    d = D.load_guideline_deps(gp, specs_dir=sd)
    assert len(D.impact_of_spec("STORE.700_880.stage", d, tmp_path).artifacts) == 1


def test_the_manifest_records_the_hash_of_every_spec_and_of_the_guideline(shipped):
    m = shipped.manifest()
    assert m["schema"] == D.DEPS_SCHEMA
    assert m["guideline"]["guideline_hash"] == shipped.guideline.guideline_hash
    assert set(m["specs"]) == {p.stem.replace(".yaml", "") and yaml.safe_load(
        p.read_text(encoding="utf-8"))["spec_id"] for p in SPECS.glob("*.yaml")}
    assert all(len(v["spec_hash"]) == 16 for v in m["specs"].values())


# ===================================================================== the CLI
def test_acr_deps_prints_every_gap_in_full():
    """21 gaps, all 21 on stdout. A command that printed "21 gaps" and a top-5 would be
    summarising the honest output into something more encouraging than it is."""
    r = runner.invoke(app, ["deps", "--guideline", str(GUIDELINE), "--specs", str(SPECS)])
    assert r.exit_code == 0, r.output
    d = D.load_guideline_deps(GUIDELINE, specs_dir=SPECS)
    for rd in d.per_recommendation:
        for g in rd.gaps:
            assert g["name"] in r.output, f"{g['name']} was not printed"


def test_acr_deps_json_is_the_whole_graph():
    r = runner.invoke(app, ["deps", "--guideline", str(GUIDELINE), "--specs", str(SPECS),
                            "--json"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.stdout)
    assert set(doc["forward"]["per_recommendation"]) == set(REAL_RECS)
    per = doc["forward"]["per_recommendation"]["NSCLC-ADJ-SYSTEMIC-II-IIIA"]
    assert len(per["gaps"]) == 9 and len(per["resolved"]) == 6
    assert per["predicate_classes"]["date_of_definitive_surgery"] == ["action", "timing",
                                                                     "exception"]
    assert doc["exceptions_declared_per_rec"]["NSCLC-BIOMARKER-BEFORE-FIRST-LINE"] == [
        "insufficient_tissue", "patient_refused_testing"]


def test_acr_deps_fail_on_gap_exits_nonzero():
    r = runner.invoke(app, ["deps", "--guideline", str(GUIDELINE), "--specs", str(SPECS),
                            "--fail-on-gap"])
    assert r.exit_code == 1


def test_acr_deps_refuses_a_guideline_with_an_undeclared_exception_list(tmp_path):
    doc = _toy_guideline()
    doc["recommendations"][0].pop("exceptions")
    doc["recommendations"][0]["satisfied_when"] = [{"op": "is_present", "var": "treatment_given"}]
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    r = runner.invoke(app, ["deps", "--guideline", str(p), "--specs", str(SPECS)])
    assert r.exit_code == 2
    assert "TOY-1" in r.output


def test_acr_deps_spec_reports_impact_and_exits_nonzero_when_something_went_stale(
        sandbox, tmp_path):
    sd, gp = sandbox
    _fake_pipeline(tmp_path, sd, gp, inline_provenance=True)
    args = ["deps", "--guideline", str(gp), "--specs", str(sd),
            "--spec", "STORE.700_880.stage", "--runs", str(tmp_path)]
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "CURRENT" in r.output

    target = sd / "STORE.700_880.stage.yaml"
    doc = yaml.safe_load(target.read_text(encoding="utf-8"))
    doc["spec_version"] = "0.3.0"
    target.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    r2 = runner.invoke(app, args)
    assert r2.exit_code == 1
    assert "STALE" in r2.output
    for rec in REAL_RECS:
        assert rec in r2.output


def test_acr_deps_out_writes_a_manifest_carrying_both_hashes(tmp_path):
    out = tmp_path / "deps.json"
    r = runner.invoke(app, ["deps", "--guideline", str(GUIDELINE), "--specs", str(SPECS),
                            "--out", str(out)])
    assert r.exit_code == 0, r.output
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["guideline"]["guideline_hash"] and m["specs"]["STORE.700_880.stage"]["spec_hash"]
    assert m["forward"]["per_recommendation"]["NSCLC-STAGE-I-DEFINITIVE-LOCAL-THERAPY"]["gaps"]


def test_deps_reaches_no_model(shipped):
    """Same guarantee `concordance` makes, for the same reason: this is a rule layer. If a
    model can be reached from here, someone will eventually ask it to guess a binding."""
    import importlib
    import sys
    seen, stack = set(), ["acr.contract.deps"]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        mod = sys.modules.get(name) or importlib.import_module(name)
        for attr in vars(mod).values():
            m = getattr(attr, "__module__", None) or getattr(attr, "__name__", None)
            if isinstance(m, str) and m.startswith("acr."):
                stack.append(m)
    assert "acr.core.llm" not in seen and "acr.graph" not in seen
