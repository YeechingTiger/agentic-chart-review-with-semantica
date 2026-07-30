"""The ICD-O-3 code table: a fact table, advisory, and honest about its own provenance.

The two failures it exists for, both measured on a real run on 2026-07-30 and neither catchable
by a regex or a word list:

  `C187 / 7205 / 0` — a run found a real sigmoid colon hyperplastic polyp in a lung-cancer
  patient's chart and coded it. `7205` is not an ICD-O-3 morphology, a hyperplastic polyp has no
  morphology at all because it is not a neoplasm, and behaviour 0 is not reportable. `\\d{4}`
  accepted `7205`, so `check_field_formats` passed it and `answer_shape_miss` never fired.

  "ICD-O-3 topography C341 is right middle lobe" — asserted by a run that then coded C341 over
  evidence reading "right middle lobe". C341 is the upper lobe. A table decides that; a model's
  recollection did not.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acr.icdo3 import (
    BEHAVIOR_NOT_IN_TABLE,
    CODES_DIR,
    EXCLUDED_BY_SPEC,
    MALFORMED,
    NOT_REPORTABLE_BEHAVIOR,
    OUT_OF_TABLE_SCOPE,
    UNKNOWN_MORPHOLOGY,
    CodeTableError,
    behavior_digit,
    check_codes,
    load_table,
    normalize_code,
    prompt_block,
)


@pytest.fixture(scope="module")
def t():
    return load_table("icdo3_colorectal")


def kinds(problems):
    return {p.kind for p in problems}


# ------------------------------------------------------- the answer nothing could catch
def test_the_case004_answer_is_caught_on_all_three_fields(t):
    """`C187 / 7205 / 0`: out of table scope, not a morphology, not reportable."""
    ks = kinds(check_codes("C187", "7205", "0", table=t))
    assert UNKNOWN_MORPHOLOGY in ks, "7205 is not an ICD-O-3 morphology"
    assert NOT_REPORTABLE_BEHAVIOR in ks, "behaviour 0 is benign and not reportable"
    # C187 IS a real colorectal code, so it must not be reported as invalid — the error was
    # answering about the wrong tumour, and this table has no way to know that.
    assert OUT_OF_TABLE_SCOPE not in ks


def test_a_four_digit_number_is_not_a_morphology_code(t):
    """The open gap the removed `field_format` could never close: `\\d{4}` passes anything."""
    for invented in ("7205", "9999", "1234", "8999"):
        assert UNKNOWN_MORPHOLOGY in kinds(check_codes(None, invented, None, table=t)), invented


def test_a_benign_polyp_is_named_when_its_code_and_behaviour_are_given(t):
    """8210/0 is a legal ICD-O-3 code and not a reportable neoplasm. Both facts matter."""
    p = next(x for x in check_codes(None, "8210", "0", table=t)
             if x.kind == NOT_REPORTABLE_BEHAVIOR)
    assert "adenomatous polyp" in p.message.lower()
    assert UNKNOWN_MORPHOLOGY not in kinds(check_codes(None, "8210", "3", table=t)), (
        "8210/3 — carcinoma arising in an adenomatous polyp — IS reportable")


def test_hyperplastic_polyp_has_no_code_at_all(t):
    """Not `code: null` as an oversight — as the finding. It is not a neoplasm."""
    row = next(r for r in t.not_reportable if r["term"] == "Hyperplastic polyp")
    assert row["code"] is None
    assert "not a neoplasm" in row["why"]


# ------------------------------------------------------- notation is folded, not refused
@pytest.mark.parametrize("raw,want", [
    ("C18.7", "C187"), ("c18.7", "C187"), ("C187", "C187"), (" C18.7 ", "C187"),
    ("8140/3", "8140"), ("8140", "8140"), ("C34.11", "C3411"),
])
def test_the_punctuated_form_the_manual_writes_is_folded(raw, want):
    """4 of the removed `field_format` check's 6 useful firings rejected `C34.9`/`C34.11`/`C34.2`
    — the form ICD-O-3 itself uses. It was creating the round trips it then resolved."""
    assert normalize_code(raw) == want


def test_the_behaviour_digit_is_split_out_not_merged():
    assert behavior_digit("8140/3") == "3"
    assert behavior_digit("8140") is None
    assert normalize_code("8140/3") == "8140", "the behaviour digit must not join the morphology"


# ------------------------------------------------------- scope is a finding, not an error
def test_a_lung_code_is_out_of_scope_and_not_invalid(t):
    """This repository's corpus is a LUNG registry — all 1,788 gold topographies are C34x. A
    colorectal table must say 'wrong table' rather than 'wrong answer', or a missing table looks
    like a wrong answer on every case."""
    p = next(x for x in check_codes("C341", "8140", "3", table=t) if x.field == "primary_site")
    assert p.kind == OUT_OF_TABLE_SCOPE
    assert "not evidence that" in p.message
    assert "wrong table is loaded" in p.message


def test_a_malformed_topography_is_distinguished_from_an_out_of_scope_one(t):
    assert kinds(check_codes("C18", None, None, table=t)) == {MALFORMED}
    assert kinds(check_codes("lung", None, None, table=t)) == {MALFORMED}


def test_a_correct_colorectal_answer_has_no_problems(t):
    assert check_codes("C187", "8140", "3", table=t) == []
    assert check_codes("C209", "8480", "3", table=t) == []
    assert check_codes("C211", "8070", "3", table=t) == []


def test_an_absent_field_is_not_a_problem(t):
    """Abstention is not this module's business."""
    assert check_codes(None, None, None, table=t) == []
    assert check_codes("C187", "", "  ", table=t) == []


def test_an_unknown_behaviour_digit_is_its_own_finding(t):
    assert kinds(check_codes(None, None, "7", table=t)) == {BEHAVIOR_NOT_IN_TABLE}


# ------------------------------------------------------- NOS codes are real answers
def test_the_nos_codes_are_in_the_table_and_flagged_not_excluded(t):
    """8000/8010/8046 together are the registry's own answer for 10.8% of this corpus, and C349
    for 9.6%. A table that omitted them would recreate the deleted `not_less_specific`."""
    for c in ("8000", "8010", "8046"):
        assert c in t.morphology
        assert c in t.nos_morphology
    assert check_codes(None, "8010", "3", table=t) == [], "coding NOS is not a problem"


def test_nos_topography_is_marked(t):
    assert "C189" in t.nos_topography and "C210" in t.nos_topography
    assert check_codes("C189", None, None, table=t) == []


# ------------------------------------------------------- the prompt rendering
def test_the_prompt_block_renders_the_whole_domain(t):
    """A model shown 12 of 40 morphologies codes into the 12."""
    b = prompt_block(t)
    for c in t.morphology:
        assert c in b, f"{c} missing from the prompt block"
    for c in t.topography:
        assert c in b


def test_the_prompt_block_states_its_own_provenance(t):
    """Recalled, not transcribed, and nobody has signed it. The model has to be able to weigh it
    against a pathology report it has actually read."""
    b = " ".join(prompt_block(t).split())
    assert "recalled by a language model, not transcribed" in b
    assert "no registrar has checked it" in b
    assert "say so in your reasoning" in b


def test_the_prompt_block_carries_the_not_reportable_list_and_the_safeguards(t):
    b = prompt_block(t)
    assert "Hyperplastic polyp" in b and "no ICD-O-3 morphology exists" in b
    assert "NOT reportable" in b
    assert "benign polyp is not the reportable tumour" in b


# ------------------------------------------------------- provenance discipline in the YAML
def test_the_yaml_declares_itself_model_recalled_and_unbound():
    """Same standard the four code tables already in `specs/` are held to: a table recalled by a
    model must say so and must name what a human should check it against."""
    d = yaml.safe_load((CODES_DIR / "icdo3_colorectal.yaml").read_text(encoding="utf-8"))
    sa = d["source_authority"]
    assert sa["origin"] == "model_recalled"
    assert sa["version_binding"] == "NOT_BOUND"
    assert "no clinical or registrar sign-off" in sa["status"]
    # Read the PARSED fields, not the file text. The comments say the same thing and
    # `yaml.safe_load` drops all of them, so a bare `origin: model_recalled` with its basis only
    # in a comment is the labels-without-basis marking `specs/` added `provenance:` to replace.
    assert "RECALLED BY A LANGUAGE MODEL" in " ".join(sa["basis"].split())
    assert "not a transcription of ICD-O-3" in " ".join(sa["basis"].split())
    assert "casefinding manual" in " ".join(sa["what_a_human_must_check"].split()), \
        "reportability is a registry policy question and the file must say who settles it"


def test_a_missing_table_raises_and_names_what_exists():
    with pytest.raises(CodeTableError) as e:
        load_table("icdo3_does_not_exist")
    assert "available:" in str(e.value)


# ==========================================================================================
# THE LUNG TABLE — the extraction corpus, and the failures it was built for
# ==========================================================================================
GT = Path("/N/project/computable_phenotype/acr_real/ground_truth.csv")


@pytest.fixture(scope="module")
def lung():
    return load_table("icdo3_lung")


def test_the_default_table_is_lung_because_that_is_what_this_repo_extracts(lung):
    assert load_table().table_id == lung.table_id == "ICDO3-LUNG"


def test_the_subsite_digits_a_run_got_wrong(lung):
    """A run asserted "ICD-O-3 topography C341 is right middle lobe" and coded C341 over evidence
    reading "right middle lobe". C341 is the UPPER lobe. This is the whole reason for a table."""
    assert lung.topography_name("C341") == "Upper lobe, lung"
    assert lung.topography_name("C342") == "Middle lobe, lung"
    assert lung.topography_name("C343") == "Lower lobe, lung"
    assert lung.topography_name("C340") == "Main bronchus"
    assert "NOS" in (lung.topography_name("C349") or "")


def test_the_left_lung_has_no_middle_lobe(lung):
    """A run coded C342 while its cited evidence read "left lower lobe" nine times; the registry
    coded C343. Recorded as an anatomical fact, not as a lobe-word regex."""
    d = yaml.safe_load((CODES_DIR / "icdo3_lung.yaml").read_text(encoding="utf-8"))
    lat = d["laterality"]
    assert "C342" not in lat["left_lung_lobes"]
    assert "C342" in lat["right_lung_lobes"]
    imp = next(r for r in lat["impossible"] if r["subsite"] == "C342")
    assert imp["side"] == "left" and "no left middle lobe" in imp["why"]


def test_the_two_blastomas_are_different_diseases(lung):
    """A run coded 8973 where the registry coded 8972. Both are real codes, so only a table with
    both in it and a note between them can surface the confusion."""
    assert "pulmonary blastoma" in (lung.morphology_name("8972") or "").lower()
    assert "pleuropulmonary blastoma" in (lung.morphology_name("8973") or "").lower()
    assert lung.morphology_name("8972") != lung.morphology_name("8973")
    # Neither is a code-level error: the run's mistake was reading the pathology, and this table
    # must not pretend otherwise.
    assert check_codes("C349", "8973", "3", table=lung) == []


def test_the_solid_adenocarcinoma_code_a_run_missed_is_in_the_table(lung):
    """CASE003: registry 8230, run coded 8140. Both legal; the table cannot decide which is right
    but it can stop 8230 from looking invented."""
    assert lung.morphology_name("8230") is not None
    assert check_codes("C341", "8230", "3", table=lung) == []


def test_a_carcinoid_is_malignant_and_reportable(lung):
    """The clinical habit of calling a typical carcinoid benign is not ICD-O-3's position."""
    assert check_codes("C341", "8240", "3", table=lung) == []
    assert lung.reportable_behavior("3") is True


def test_the_nos_codes_that_the_removed_check_pushed_away_from_are_clean(lung):
    """8000/8010/8046 are the registry's answer for 10.8% of this corpus and C349 for 9.6%. The
    deleted `not_less_specific` refused exactly these, 22 firings out of 22."""
    for h in ("8000", "8010", "8046"):
        assert check_codes("C349", h, "3", table=lung) == []


def test_a_haematopoietic_gold_is_a_scope_boundary_not_a_table_gap(lung):
    """Six patients in this corpus have a lymphoma gold histology, and the spec's
    `when_not_to_use` excludes haematopoietic neoplasms. Those are SPEC_INSUFFICIENT cases; a
    table that reported them as UNKNOWN_MORPHOLOGY would make a scope boundary look like an
    incomplete table."""
    p = next(x for x in check_codes("C349", "9680", "3", table=lung) if x.field == "histology")
    assert p.kind == EXCLUDED_BY_SPEC
    assert "SPEC_INSUFFICIENT" in p.message
    assert "not a coding error" in p.message
    for c in ("9591", "9699", "9702"):
        assert {x.kind for x in check_codes(None, c, None, table=lung)} == {EXCLUDED_BY_SPEC}


def test_a_colorectal_code_is_out_of_scope_against_the_lung_table(lung):
    p = next(x for x in check_codes("C187", "8140", "3", table=lung) if x.field == "primary_site")
    assert p.kind == OUT_OF_TABLE_SCOPE


@pytest.mark.skipif(not GT.is_file(), reason="registry gold is outside the repository")
def test_the_table_validates_against_every_registry_answer_in_the_corpus(lung):
    """THE REGRESSION TEST THAT MATTERS, and it is free: deterministic string matching against
    1,788 operator-confirmed registry answers, no model in the loop.

    The first draft of this table scored 1762/1788 and the 26 misses were real codes the registry
    uses — 8033 sarcomatoid carcinoma, 8256/8257 minimally invasive adenocarcinoma, 8550, 8574,
    8141, 8002, 8023, 8144, 8800, 9180 — plus six haematopoietic cases the spec excludes. This
    test is what found them, and it is what will find the next omission.
    """
    import csv
    rows = list(csv.DictReader(GT.open(encoding="utf-8")))
    problems = {}
    for r in rows:
        ps = check_codes(r["gt_primary_site"], r["gt_histology"], r["gt_behavior"], table=lung)
        for p in ps:
            problems.setdefault(p.kind, []).append(p.value)

    # Every remaining problem must be the declared scope boundary. Anything else is a real gap.
    unexpected = {k: sorted(set(v)) for k, v in problems.items() if k != EXCLUDED_BY_SPEC}
    assert not unexpected, (
        f"the table does not cover codes the registry actually uses: {unexpected}. "
        f"Add them; a value domain that rejects the registry's own answers is worse than none.")
    assert len(problems.get(EXCLUDED_BY_SPEC, [])) == 6, (
        "six haematopoietic cases were measured in this corpus; a change here means the cohort "
        "or the spec's exclusions moved and the accuracy denominator moved with them")


# ==========================================================================================
# THE SEAM: the Task Contract declares the table, the prompt renders it
# ==========================================================================================
def test_the_lung_spec_declares_its_code_table():
    """Which code system a value belongs to is part of what the answer MEANS, so the spec says
    it. The runtime does not guess from the corpus or from the field names."""
    from acr.spec import load_spec as _ls
    spec = _ls("specs/STORE.400_522_523.site_histology_behavior.yaml")
    assert spec.value_domain == "icdo3_lung"


def test_a_spec_with_no_code_system_gets_no_block():
    """The date and class-of-case variables have no ICD-O-3 domain. Handing them a wall of lung
    morphologies to ignore is prompt bloat, not helpfulness."""
    from acr.icdo3 import code_domain_block
    from acr.spec import load_specs as _lss
    blocks = {sid: code_domain_block(sp) for sid, sp in _lss("specs").items()}
    assert blocks["STORE.400_522_523.site_histology_behavior"]
    assert not blocks["STORE.390.date_of_initial_diagnosis"]
    assert not blocks["STORE.610.class_of_case"]


def test_a_declared_table_that_does_not_exist_stops_the_spec_from_loading(tmp_path):
    """FAIL CLOSED ON A TYPO. A missing table would otherwise render an empty domain and the run
    would look exactly like one that had been given the codes — the same failure as a skill that
    silently supplies no guidance while the manifest reports that it did."""
    from acr.icdo3 import CodeTableError
    from acr.spec import load_spec as _ls
    p = tmp_path / "S.2.yaml"
    p.write_text(
        "spec_id: S.2\nspec_version: 0.1.0\ndata_source: notes\nquestion: q\n"
        "value_domain: icdo3_atlantis\n"
        "fields:\n  - name: primary_site\n    type: string\n"
        "decision_rule: [r]\nevidence_rules:\n  counts_as_evidence: [anything]\n",
        encoding="utf-8")
    with pytest.raises(CodeTableError) as e:
        _ls(p)
    assert "available:" in str(e.value)


def test_the_rendered_block_contains_the_subsite_facts_a_run_got_wrong():
    """End to end: the thing the model will actually read has C341 next to 'Upper lobe'."""
    from acr.icdo3 import code_domain_block
    from acr.spec import load_spec as _ls
    b = code_domain_block(_ls("specs/STORE.400_522_523.site_histology_behavior.yaml"))
    assert "C341  Upper lobe, lung" in b
    assert "C342  Middle lobe, lung" in b
    assert "C343  Lower lobe, lung" in b
    assert "left lung has no middle lobe" in " ".join(b.split())
    assert "8972" in b and "8973" in b
