"""The linter has to fail on the specs we shipped, or it is decoration.

Every test below is written against a fault that is either live in `specs/` today or was live
in this repository's history. Two of them are load-bearing:

  * `format: CCYYMMDD` COMPILES. It is a valid Python regex that matches exactly one string,
    the literal "CCYYMMDD", so `check_field_formats` rejects every date STORE.390 and
    STORE.1860_1880 can legally produce. A linter that only calls `re.compile` passes it.
  * commit 173f453 shipped a gate no sample could pass. The arithmetic that decides that is
    one line, and nothing in the tree had ever run it.

Synthetic specs are built through `ExtractionSpec(...)` rather than `load_spec`, deliberately:
the loader enforces provenance, and a fixture that had to carry a provenance block for every
field would be too expensive to write, so the faults would go untested. Nothing here reads a
corpus, an answer key, or a model.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from acr import speclint
from acr.cli import app
from acr.spec import ExtractionSpec, load_spec

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "specs").glob("*.yaml")) + sorted((ROOT / "specs" / "ablation").glob("*.yaml"))
DIAG = ROOT / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
RECUR = ROOT / "specs" / "STORE.1860_1880.first_recurrence.yaml"
STAGE = ROOT / "specs" / "STORE.700_880.stage.yaml"
SHB = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
COC = ROOT / "specs" / "STORE.610.class_of_case.yaml"

runner = CliRunner()


def mkspec(**kw) -> ExtractionSpec:
    base = {"spec_id": "TEST.synthetic", "question": "q?", "fields": [], "proof_obligation": {}}
    base.update(kw)
    return ExtractionSpec(**base)


def checks(findings, check: str, severity: str = speclint.FAIL) -> list:
    return [f for f in findings if f.check == check and f.severity == severity]


# ------------------------------------------------------------------- F1 formal: the format
def test_registry_notation_is_flagged_even_though_it_compiles():
    """The whole point. `re.compile("CCYYMMDD")` succeeds; the field is still unfillable."""
    s = mkspec(fields=[{"name": "d", "type": "string", "format": "CCYYMMDD"}])
    found = checks(speclint.lint_spec(s), speclint.F1)
    assert found, "a format with no regex metacharacter matches one literal string"
    assert "CCYYMMDD" in found[0].message


def test_both_shipped_date_specs_are_caught_today():
    """Named in the brief, and the reason the linter exists. If this test ever passes by the
    specs being fixed, delete it — do not weaken it."""
    for path, field in ((DIAG, "date_of_initial_diagnosis"), (RECUR, "recurrence_date")):
        found = checks(speclint.lint_spec(load_spec(path)), speclint.F1)
        assert any(field in f.where for f in found), f"{path.name}: {field} not flagged"


def test_an_uncompilable_format_is_flagged():
    s = mkspec(fields=[{"name": "x", "format": "C(\\d{3}"}])
    assert checks(speclint.lint_spec(s), speclint.F1)


def test_a_field_with_neither_format_nor_allowable_values_is_flagged():
    s = mkspec(fields=[{"name": "x", "type": "string"}])
    assert checks(speclint.lint_spec(s), speclint.F1)


def test_a_real_regex_with_allowable_values_passes_f1():
    s = mkspec(fields=[{"name": "x", "format": r"C\d{3}"},
                       {"name": "y", "allowable_values": ["0", "1"]}])
    assert not checks(speclint.lint_spec(s), speclint.F1)


# ------------------------------------------------------------ F2 formal: stratum totality
def test_strata_that_leave_documents_unclassified_are_flagged():
    """`assign_strata` drops an unmatched document on the floor: it is never read, never
    sampled, and contributes no elusion, so its silence is invisible."""
    s = mkspec(proof_obligation={"for_negative": {
        "mode": "stratified_exclusion",
        "strata": [{"name": "can_establish", "match": {"doc_type_matches": ["Pathology"]}}]}})
    assert checks(speclint.lint_spec(s), speclint.F2)


def test_a_rest_stratum_discharges_totality():
    s = mkspec(proof_obligation={"for_negative": {
        "mode": "stratified_exclusion",
        "strata": [{"name": "can_establish", "match": {"doc_type_matches": ["Pathology"]}},
                   {"name": "rest", "match": {"rest": True}}]}})
    assert not checks(speclint.lint_spec(s), speclint.F2)


def test_the_recurrence_witness_claim_leaves_documents_unclassified():
    """Live: claim `disease_free_interval_existed` declares one stratum and no `rest`."""
    found = checks(speclint.lint_spec(load_spec(RECUR)), speclint.F2)
    assert any("disease_free_interval_existed" in f.where for f in found)


# --------------------------------------------------------- F3 formal: establishes is real
def test_establishes_naming_an_undeclared_field_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}],
               proof_obligation={"for_negative": {"strata": [
                   {"name": "s", "match": {"rest": True}, "establishes": ["a", "typo_field"]}]}})
    found = checks(speclint.lint_spec(s), speclint.F3)
    assert found and "typo_field" in found[0].message


def test_the_shipped_specs_declare_no_phantom_establishes():
    for p in SPECS:
        assert not checks(speclint.lint_spec(load_spec(p)), speclint.F3), p.name


# --------------------------------------------------- F4 formal: keyword-field reachability
def test_a_field_no_required_term_reaches_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]},
                       {"name": "b", "allowable_values": ["1"]}],
               keyword_field_coverage={"a": ["stage"]},
               proof_obligation={"for_negative": {"required_keywords": ["stage"]}})
    found = checks(speclint.lint_spec(s), speclint.F4)
    assert any("b" in f.where for f in found)


def test_coverage_claimed_by_a_term_that_is_not_required_is_flagged():
    """A mapping may only discharge a field with a term the gate actually enforces."""
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}],
               keyword_field_coverage={"a": ["never_searched"]},
               proof_obligation={"for_negative": {"required_keywords": ["stage"]}})
    assert checks(speclint.lint_spec(s), speclint.F4)


def test_the_stage_spec_keyword_mapping_validates():
    """STORE.700_880 is the one spec that declares the mapping; generalising it must not
    break the case it was generalised from."""
    assert not checks(speclint.lint_spec(load_spec(STAGE)), speclint.F4)


def test_an_empty_required_keyword_list_under_an_enforcing_gate_is_flagged():
    """`required_keywords_all_searched: true` over an empty list enforces nothing. Live in
    all four of the originally shipped specs."""
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}],
               proof_obligation={"for_negative": {
                   "required_keywords": [],
                   "gate": {"required_keywords_all_searched": True}}})
    assert checks(speclint.lint_spec(s), speclint.F4)


# ------------------------------------------------------------- F5 formal: evidence closure
def test_a_field_no_stratum_establishes_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]},
                       {"name": "orphan", "allowable_values": ["1"]}],
               evidence_rules={"counts_as_evidence": ["a path report"]},
               proof_obligation={"for_negative": {"strata": [
                   {"name": "s", "match": {"rest": True}, "establishes": ["a"]}]}})
    found = checks(speclint.lint_spec(s), speclint.F5)
    assert any("orphan" in f.where for f in found)


def test_no_counts_as_evidence_at_all_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}])
    assert checks(speclint.lint_spec(s), speclint.F5)


def test_a_spec_that_declares_itself_unanswerable_is_a_note_not_a_failure():
    """STORE.610 says the variable is not a property of the chart. That is a finding to
    print, not a fault to fail on — but it must still be printed."""
    fs = speclint.lint_spec(load_spec(COC))
    assert not checks(fs, speclint.F5)
    assert checks(fs, speclint.F5, speclint.NOTE)


# --------------------------------------------------------- F6 formal: abstention totality
def test_a_boundary_answer_in_no_value_space_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1", "2"]}],
               boundary_cases=[{"case": "x", "answer": "banana"}])
    found = checks(speclint.lint_spec(s), speclint.F6)
    assert found and "banana" in found[0].message


def test_an_abstention_answer_is_inside_the_outcome_space():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1", "2"]}],
               boundary_cases=[{"case": "x", "answer": "EVIDENCE_INSUFFICIENT for a"}])
    assert not checks(speclint.lint_spec(s), speclint.F6)


def test_the_stage_spec_boundary_answer_T4_is_in_no_field():
    """Live: `answer: "T4"` carries no c/p prefix, so no field in the spec can hold it —
    inside the one spec that exists to stop the c/p categories being conflated."""
    found = checks(speclint.lint_spec(load_spec(STAGE)), speclint.F6)
    assert any("T4" in f.message for f in found)


# ------------------------------------------------------------- F7 formal: conflict matrix
def test_two_strata_establishing_one_field_with_no_ordering_is_reported():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}],
               evidence_rules={"counts_as_evidence": ["anything"]},
               proof_obligation={"for_negative": {"strata": [
                   {"name": "path", "match": {"doc_type_matches": ["Pathology"]},
                    "establishes": ["a"]},
                   {"name": "notes", "match": {"doc_type_matches": ["Progress-Note"]},
                    "establishes": ["a"]}]}})
    found = checks(speclint.lint_spec(s), speclint.F7)
    assert found and "path" in found[0].message and "notes" in found[0].message


def test_a_conflict_rule_naming_both_sources_discharges_the_pair():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}],
               evidence_rules={"counts_as_evidence": ["anything"]},
               conflict_rules=[{"if": "a pathology report and a progress-note disagree",
                                "then": "prefer the pathology"}],
               proof_obligation={"for_negative": {"strata": [
                   {"name": "path", "match": {"doc_type_matches": ["Pathology"]},
                    "establishes": ["a"]},
                   {"name": "notes", "match": {"doc_type_matches": ["Progress-Note"]},
                    "establishes": ["a"]}]}})
    assert not checks(speclint.lint_spec(s), speclint.F7)


# -------------------------------------------------------- F8 formal: gate satisfiability
def test_the_zero_hit_bound_is_the_closed_form_the_ledger_uses():
    from acr.coverage import clopper_pearson_upper
    for n in (10, 24, 25, 29):
        assert speclint.bound_at_n(n, 0.95) == pytest.approx(clopper_pearson_upper(0, n, 0.95))


def test_min_n_is_the_smallest_n_that_actually_reaches_the_bound():
    for bound in (0.20, 0.12, 0.10, 0.05):
        n = speclint.min_n_for_bound(bound, 0.95)
        assert speclint.bound_at_n(n, 0.95) <= bound
        assert speclint.bound_at_n(n - 1, 0.95) > bound


def test_the_shipped_numbers_25_and_0_12_are_satisfiable_by_0_007():
    assert speclint.min_n_for_bound(0.12, 0.95) == 24
    assert speclint.bound_at_n(25, 0.95) == pytest.approx(0.1129, abs=5e-5)
    assert 0.12 - speclint.bound_at_n(25, 0.95) < 0.008


def test_a_0_10_cap_on_25_draws_is_unsatisfiable_and_says_so():
    """The shape of commit 173f453: the gate cannot pass however much work is done, and the
    rejection message never says the number is the reason."""
    s = mkspec(proof_obligation={"for_negative": {
        "confidence": 0.95,
        "strata": [{"name": "rest", "match": {"rest": True}, "min_sample": 25}],
        "gate": {"max_elusion_upper": 0.10}}})
    found = checks(speclint.lint_spec(s), speclint.F8)
    assert found, "an unpassable gate must be a failure, not a warning"
    assert "29" in found[0].message, "the message must name the n that would work"


def test_a_satisfiable_gate_reports_the_margin_without_failing():
    s = mkspec(proof_obligation={"for_negative": {
        "confidence": 0.95,
        "strata": [{"name": "rest", "match": {"rest": True}, "min_sample": 25}],
        "gate": {"max_elusion_upper": 0.12}}})
    assert not checks(speclint.lint_spec(s), speclint.F8)
    rows = speclint.gate_rows(mkspec(proof_obligation={"for_negative": {
        "confidence": 0.95,
        "strata": [{"name": "rest", "match": {"rest": True}, "min_sample": 25}],
        "gate": {"max_elusion_upper": 0.12}}}))
    assert rows and rows[0]["binding_n"] == 25 and rows[0]["min_n_required"] == 24


def test_the_binding_n_is_the_weakest_sampled_stratum():
    """The gate compares the cap against the WORST stratum, so a single small sample sets
    the bound however large the others are."""
    s = mkspec(proof_obligation={"for_negative": {
        "confidence": 0.95,
        "strata": [{"name": "a", "match": {"rest": True}, "min_sample": 60},
                   {"name": "b", "match": {"doc_type_matches": ["X"]},
                    "min_sample_of_misses": 10}],
        "gate": {"max_elusion_upper": 0.12}}})
    assert speclint.gate_rows(s)[0]["binding_n"] == 10
    assert checks(speclint.lint_spec(s), speclint.F8)


def test_a_declared_bound_with_no_declared_confidence_is_a_failure_not_a_default():
    """A confidence this linter picked would be a number nobody chose, applied to a gate
    somebody has to defend."""
    s = mkspec(proof_obligation={"for_negative": {
        "strata": [{"name": "rest", "match": {"rest": True}, "min_sample": 25}],
        "gate": {"max_elusion_upper": 0.12}}})
    assert checks(speclint.lint_spec(s), speclint.F8)


def test_the_thresholds_are_required_parameters():
    with pytest.raises(TypeError):
        speclint.min_n_for_bound(0.12)          # type: ignore[call-arg]
    with pytest.raises(TypeError):
        speclint.bound_at_n(25)                 # type: ignore[call-arg]


# ------------------------------------------------------- F9 formal: answer_check integrity
def test_an_answer_check_on_an_undeclared_field_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1"]}],
               answer_checks=[{"field": "ghost", "kind": "not_less_specific",
                               "nos_values": ["1"]}])
    assert checks(speclint.lint_spec(s), speclint.F9)


def test_a_nos_value_outside_the_declared_domain_is_flagged():
    s = mkspec(fields=[{"name": "a", "allowable_values": ["1", "2"]}],
               answer_checks=[{"field": "a", "kind": "not_less_specific",
                               "nos_values": ["77"]}])
    found = checks(speclint.lint_spec(s), speclint.F9)
    assert found and "77" in found[0].message


def test_the_shipped_answer_checks_are_consistent_with_their_fields():
    for p in (STAGE, SHB):
        assert not checks(speclint.lint_spec(load_spec(p)), speclint.F9), p.name


# --------------------------------------------------------------- tiers 2, 3 and 4 boundary
def test_tier2_needs_a_corpus_and_will_not_invent_one():
    with pytest.raises(TypeError):
        speclint.tier2_checks(mkspec())         # type: ignore[call-arg]


def test_tier2_is_named_but_not_run_by_a_plain_lint():
    out = speclint.render_report([load_spec(SHB)], corpus=None, answer_key=None,
                                 tier3_enabled=False)
    assert "TIER 2" in out and "NOT RUN" in out
    for name in speclint.TIER2_CHECKS:
        assert name in out


def test_tier3_refuses_without_both_the_flag_and_the_key():
    with pytest.raises(speclint.AnswerKeyRefused):
        speclint.tier3_checks(mkspec(), answer_key=None, enabled=True)
    with pytest.raises(speclint.AnswerKeyRefused):
        speclint.tier3_checks(mkspec(), answer_key="key.json", enabled=False)


def test_an_answer_key_is_not_reachable_from_a_plain_lint():
    """The failure this prevents: a lint that quietly scores itself against the key it is
    supposed to be independent of."""
    out = speclint.render_report([load_spec(SHB)], corpus=None, answer_key=None,
                                 tier3_enabled=False)
    assert "REFUSED" in out


def test_tier4_is_a_list_of_things_the_linter_does_not_check():
    assert len(speclint.TIER4_HUMAN) >= 5
    out = speclint.render_report([load_spec(SHB)], corpus=None, answer_key=None,
                                 tier3_enabled=False)
    for item in speclint.TIER4_HUMAN:
        assert item.split(".")[0][:40] in out


# ------------------------------------------------------------------------------ end to end
def test_every_shipped_spec_lints_without_crashing():
    for p in SPECS:
        speclint.lint_spec(load_spec(p))


def test_the_linter_is_not_a_no_op_on_what_we_shipped():
    total = sum(len([f for f in speclint.lint_spec(load_spec(p))
                     if f.severity == speclint.FAIL]) for p in SPECS)
    assert total > 0, "a clean report over these six specs would mean the linter checks nothing"


def test_the_cli_exits_non_zero_on_a_tier1_failure():
    r = runner.invoke(app, ["spec", "lint", str(DIAG)])
    assert r.exit_code != 0
    assert "CCYYMMDD" in r.stdout


def test_the_cli_lints_a_directory_and_prints_the_gate_table():
    r = runner.invoke(app, ["spec", "lint", str(ROOT / "specs")])
    assert "GATE SATISFIABILITY" in r.stdout
    assert "0.1129" in r.stdout, "the bound actually earned by the declared n must be printed"


def test_the_cli_will_not_take_an_answer_key_without_the_flag():
    r = runner.invoke(app, ["spec", "lint", str(COC), "--answer-key", "nonexistent.json"])
    assert r.exit_code != 0


def test_the_chooser_table_prices_a_bound_the_caller_supplies():
    """The declared 25 and 0.12 are model-invented, so the table has to be able to price the
    alternatives a human is choosing between — and to invent none of them itself."""
    out = speclint.render_report([load_spec(SHB)], corpus=None, answer_key=None,
                                 tier3_enabled=False, bounds=[0.05], sizes=[59])
    assert "0.05" in out and "n >= 59" in out
    plain = speclint.render_report([load_spec(SHB)], corpus=None, answer_key=None,
                                   tier3_enabled=False)
    assert "0.05" not in plain
