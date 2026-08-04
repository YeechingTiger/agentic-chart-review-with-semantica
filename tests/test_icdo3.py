"""The ICD-O-3 code tables: one use case's asset, read through the general loader.

What this file tests is the **cancer asset**, not the framework. The division of labour:
`test_code_tables.py` asserts that the loader works without knowing topography/morphology/behavior
(a lipid panel loads into it too); here we assert that the lung table really does say C341 is the
upper lobe. The first is the framework, the second is this use case's facts.

The two real failures it exists for, measured 2026-07-30, are neither of them catchable by a regex
or a word list:

  `C187 / 7205 / 0` — one run found a real sigmoid hyperplastic polyp in a lung cancer patient's
  chart and coded it. `7205` is not an ICD-O-3 morphology code, a hyperplastic polyp has no
  morphology code at all (it is not a neoplasm), and behaviour 0 is not reportable. `\\d{4}`
  accepted `7205`, so `check_field_formats` let it through and `answer_shape_miss` never fired.

  "ICD-O-3 topography C341 is right middle lobe" — one run asserted exactly that, and then coded
  C341 while its cited evidence said "right middle lobe". C341 is the **upper** lobe. This is the
  kind of thing a table can settle and the model's recall could not.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acr.contract.code_tables import (
    CODES_DIR,
    EXCLUDED_BY_SPEC,
    MALFORMED,
    NOT_ADMISSIBLE,
    NOT_IN_TABLE,
    OUT_OF_TABLE_SCOPE,
    CodeTableError,
    check_values,
    load_table,
    prompt_block,
)
from acr.core import site


@pytest.fixture(scope="module")
def t():
    return load_table("icdo3_colorectal")


@pytest.fixture(scope="module")
def lung():
    return load_table("icdo3_lung")


def kinds(problems):
    return {p.kind for p in problems}


def triple(site=None, histology=None, behavior=None) -> dict:
    """The three positional arguments of the old `check_codes(site, histology, behavior)`, which are
    now three **axis names**.

    The field names (primary_site / histology / behavior) are read from each axis's `field`
    declaration, so problems are still reported against the right field while no cancer word is left
    in the function signature.
    """
    return {"topography": site, "morphology": histology, "behavior": behavior}


# ------------------------------------------------------- THE ANSWER THAT NOTHING CAUGHT
def test_the_case004_answer_is_caught_on_all_three_fields(t):
    """`C187 / 7205 / 0`: not a morphology code, behaviour not admissible, and C187 itself fine."""
    ks = kinds(check_values(triple("C187", "7205", "0"), table=t))
    assert NOT_IN_TABLE in ks, "7205 is not an ICD-O-3 morphology"
    assert NOT_ADMISSIBLE in ks, "behaviour 0 is benign and not reportable"
    # C187 is a real colorectal code, so it must never be reported as invalid — that run's error was
    # answering about the wrong tumour, and this table has no way to know that.
    assert OUT_OF_TABLE_SCOPE not in ks


def test_a_four_digit_number_is_not_a_morphology_code(t):
    """The hole the deleted `field_format` could never close: `\\d{4}` lets anything through."""
    for invented in ("7205", "9999", "1234", "8999"):
        assert NOT_IN_TABLE in kinds(check_values(triple(histology=invented), table=t)), invented


def test_a_benign_polyp_is_named_when_its_code_and_behaviour_are_given(t):
    """8210/0 is a legal ICD-O-3 code and is not a reportable neoplasm. Both facts matter."""
    p = next(x for x in check_values(triple(histology="8210", behavior="0"), table=t)
             if x.kind == NOT_ADMISSIBLE)
    assert "adenomatous polyp" in p.message.lower()
    assert NOT_IN_TABLE not in kinds(check_values(triple(histology="8210", behavior="3"), table=t)), (
        "8210/3 — carcinoma arising in an adenomatous polyp — IS reportable")


def test_hyperplastic_polyp_has_no_code_at_all(t):
    """Not a forgotten `code: null` — this **is** the finding: it is not a neoplasm.

    After the migration that shows up as an exclusion row with no `axis_values` — an exclusion row
    that matches no code at all would match everything if it took part in matching.
    """
    row = next(r for r in t.exclusions if r["term"] == "Hyperplastic polyp")
    assert "axis_values" not in row
    assert "not a neoplasm" in row["why"]


# ------------------------------------------------------- NOTATION IS FOLDED, NOT REFUSED
@pytest.mark.parametrize("raw,want", [
    ("C18.7", "C187"), ("c18.7", "C187"), ("C187", "C187"), (" C18.7 ", "C187"),
    ("8140/3", "8140"), ("8140", "8140"), ("C34.11", "C3411"),
])
def test_the_punctuated_form_the_manual_writes_is_folded(t, raw, want):
    """4 of the 6 useful firings of the deleted `field_format` check were refusing
    `C34.9`/`C34.11`/`C34.2` — which is how ICD-O-3 itself writes them. It was manufacturing a round
    trip that it then solved itself.

    The rule now lives in the table's `normalization:`, not in a module constant.
    """
    assert t.normalize(raw) == want


def test_the_behaviour_digit_is_split_out_not_merged(t):
    assert t.trailing_part("8140/3") == "3"
    assert t.trailing_part("8140") is None
    assert t.normalize("8140/3") == "8140", "the behaviour digit must not join the morphology"


# ------------------------------------------------------- SCOPE IS A FINDING, NOT AN ERROR
def test_a_lung_code_is_out_of_scope_and_not_invalid(t):
    """This repository's corpus is a **lung** registry — all 1,788 gold topographies are C34x. The
    colorectal table has to say "the wrong table is loaded" rather than "the answer is wrong", or
    one wrongly loaded table makes every case look like a wrong answer."""
    p = next(x for x in check_values(triple("C341", "8140", "3"), table=t)
             if x.field == "primary_site")
    assert p.kind == OUT_OF_TABLE_SCOPE
    assert "not evidence that" in p.message
    assert "wrong table is loaded" in p.message


def test_a_malformed_topography_is_distinguished_from_an_out_of_scope_one(t):
    assert kinds(check_values(triple("C18"), table=t)) == {MALFORMED}
    assert kinds(check_values(triple("lung"), table=t)) == {MALFORMED}


def test_a_correct_colorectal_answer_has_no_problems(t):
    assert check_values(triple("C187", "8140", "3"), table=t) == []
    assert check_values(triple("C209", "8480", "3"), table=t) == []
    assert check_values(triple("C211", "8070", "3"), table=t) == []


def test_an_absent_field_is_not_a_problem(t):
    """Abstention is not this module's business."""
    assert check_values(triple(), table=t) == []
    assert check_values(triple("C187", "", "  "), table=t) == []


def test_an_unknown_behaviour_digit_is_its_own_finding(t):
    assert kinds(check_values(triple(behavior="7"), table=t)) == {NOT_IN_TABLE}


# ------------------------------------------------------- NOS CODES ARE REAL ANSWERS
def test_the_nos_codes_are_in_the_table_and_flagged_not_excluded(t):
    """8000/8010/8046 together are the registry's own answer for 10.8% of the cases in this corpus,
    and C349 for 9.6%. A table that left them out would rebuild the deleted `not_less_specific`."""
    morph = t.axes["morphology"]
    for c in ("8000", "8010", "8046"):
        assert c in morph.codes
        assert c in morph.unspecified
    assert check_values(triple(histology="8010", behavior="3"), table=t) == [], \
        "coding NOS is not a problem"


def test_nos_topography_is_marked(t):
    unspec = t.axes["topography"].unspecified
    assert "C189" in unspec and "C210" in unspec
    assert check_values(triple("C189"), table=t) == []


# ------------------------------------------------------- PROMPT RENDERING
def test_the_prompt_block_renders_the_whole_domain(t):
    """A model shown only 12 of 40 morphology codes will code into those 12."""
    b = prompt_block(t)
    for c in t.axes["morphology"].codes:
        assert c in b, f"{c} missing from the prompt block"
    for c in t.axes["topography"].codes:
        assert c in b


def test_the_prompt_block_states_its_own_provenance(t):
    """Recalled by a language model, not transcribed, signed off by nobody. The model has to be able
    to argue with it from the pathology report it actually read.

    After the migration those three sentences live in the table's `warnings:` and not in Python —
    they are assertions about **this table**.
    """
    b = " ".join(prompt_block(t).split())
    assert "recalled by a language model, not transcribed" in b
    assert "no registrar has checked it" in b
    assert "say so in your reasoning" in b


def test_the_prompt_block_carries_the_exclusions_and_the_safeguards(t):
    b = prompt_block(t)
    assert "Hyperplastic polyp" in b and "no code exists" in b
    assert "NOT admissible" in b
    assert "benign polyp is not the reportable tumour" in b


# ------------------------------------------------------- PROVENANCE DISCIPLINE IN THE YAML
def test_the_yaml_declares_itself_model_recalled_and_unbound():
    """The same standard as the four specs in `assets/specs/`: a model-recalled table has to say so
    itself, and has to write down what a human should check it against."""
    d = yaml.safe_load((CODES_DIR / "icdo3_colorectal.yaml").read_text(encoding="utf-8"))
    sa = d["source_authority"]
    assert sa["origin"] == "model_recalled"
    assert sa["version_binding"] == "NOT_BOUND"
    assert "no clinical or registrar sign-off" in sa["status"]
    # Read the **parsed** fields, not the file text. A comment says the same thing and
    # `yaml.safe_load` throws every one of them away, so an `origin: model_recalled` whose only
    # basis is a comment is exactly the "labelled with nothing behind it" that adding `provenance:`
    # to `assets/specs/` replaces.
    assert "RECALLED BY A LANGUAGE MODEL" in " ".join(sa["basis"].split())
    assert "not a transcription of ICD-O-3" in " ".join(sa["basis"].split())
    assert "casefinding manual" in " ".join(sa["what_a_human_must_check"].split()), \
        "reportability is a registry policy question and the file must say who settles it"


def test_a_missing_table_raises_and_names_what_exists():
    with pytest.raises(CodeTableError) as e:
        load_table("icdo3_does_not_exist")
    assert "available:" in str(e.value)


def test_there_is_no_default_table_any_more():
    """The old `load_table(name="icdo3_lung")` had a default, so "forgot to declare a value domain"
    and "declared lung" looked the same.

    Loading the wrong table makes every case look like a wrong answer (that is the reason
    OUT_OF_TABLE_SCOPE exists), so the table name has to be said by the spec. This test replaces the
    one that used to assert "the default table is lung" — that default was removed on purpose.
    """
    with pytest.raises(TypeError):
        load_table()                                    # type: ignore[call-arg]


# ==========================================================================================
# The lung table — the extraction corpus, and the failures it was built for
# ==========================================================================================
GT = Path("$ACR_REAL_CORPUS/ground_truth.csv")


def test_the_subsite_digits_a_run_got_wrong(lung):
    """One run asserted "ICD-O-3 topography C341 is right middle lobe" and coded C341 while its
    evidence said "right middle lobe". C341 is the **upper** lobe. That is the entire reason for
    wanting a table."""
    topo = lung.axes["topography"]
    assert topo.name_of("C341") == "Upper lobe, lung"
    assert topo.name_of("C342") == "Middle lobe, lung"
    assert topo.name_of("C343") == "Lower lobe, lung"
    assert topo.name_of("C340") == "Main bronchus"
    assert "NOS" in (topo.name_of("C349") or "")


def test_the_left_lung_has_no_middle_lobe(lung):
    """A run coded C342 while its cited evidence said "left lower lobe", nine times; the registry
    coded C343. Recorded as an anatomical fact, not as a lobe-name regex."""
    d = yaml.safe_load((CODES_DIR / "icdo3_lung.yaml").read_text(encoding="utf-8"))
    lat = d["laterality"]
    assert "C342" not in lat["left_lung_lobes"]
    assert "C342" in lat["right_lung_lobes"]
    imp = next(r for r in lat["impossible"] if r["subsite"] == "C342")
    assert imp["side"] == "left" and "no left middle lobe" in imp["why"]


def test_the_two_blastomas_are_different_diseases(lung):
    """One run coded 8973 where the registry coded 8972. Both are real codes, so this confusion only
    surfaces from a table that carries both and writes down the difference between them."""
    morph = lung.axes["morphology"]
    assert "pulmonary blastoma" in (morph.name_of("8972") or "").lower()
    assert "pleuropulmonary blastoma" in (morph.name_of("8973") or "").lower()
    assert morph.name_of("8972") != morph.name_of("8973")
    # Neither one is a code-level error: that run's mistake was in reading the pathology, and this
    # table is not allowed to pretend otherwise.
    assert check_values(triple("C349", "8973", "3"), table=lung) == []


def test_the_solid_adenocarcinoma_code_a_run_missed_is_in_the_table(lung):
    """CASE003: the registry has 8230, the run coded 8140. Both are legal; the table cannot settle
    which is right, but it can keep 8230 from looking invented."""
    assert lung.axes["morphology"].name_of("8230") is not None
    assert check_values(triple("C341", "8230", "3"), table=lung) == []


def test_a_carcinoid_is_malignant_and_reportable(lung):
    """Calling a typical carcinoid benign is a clinical habit, not ICD-O-3's position."""
    assert check_values(triple("C341", "8240", "3"), table=lung) == []
    assert lung.axes["behavior"].is_admissible("3") is True


def test_the_nos_codes_that_the_removed_check_pushed_away_from_are_clean(lung):
    """8000/8010/8046 are the registry's answer for 10.8% of this corpus and C349 for 9.6%. These
    are exactly what the deleted `not_less_specific` refused — all 22 of its firings."""
    for h in ("8000", "8010", "8046"):
        assert check_values(triple("C349", h, "3"), table=lung) == []


def test_a_haematopoietic_gold_is_a_scope_boundary_not_a_table_gap(lung):
    """Six patients in this corpus have a lymphoma as their gold histology, and the spec's
    `when_not_to_use` excludes haematopoietic neoplasms. Those are SPEC_INSUFFICIENT cases; a table
    that reported them as NOT_IN_TABLE would turn a scope boundary into something that looks like an
    incomplete table."""
    p = next(x for x in check_values(triple("C349", "9680", "3"), table=lung)
             if x.field == "histology")
    assert p.kind == EXCLUDED_BY_SPEC
    assert "SPEC_INSUFFICIENT" in p.message
    assert "not a coding error" in p.message
    for c in ("9591", "9699", "9702"):
        assert {x.kind for x in check_values(triple(histology=c), table=lung)} == {
            EXCLUDED_BY_SPEC}


def test_a_colorectal_code_is_out_of_scope_against_the_lung_table(lung):
    p = next(x for x in check_values(triple("C187", "8140", "3"), table=lung)
             if x.field == "primary_site")
    assert p.kind == OUT_OF_TABLE_SCOPE


@pytest.mark.skipif(not GT.is_file(), reason="registry gold is outside the repository")
def test_the_table_validates_against_every_registry_answer_in_the_corpus(lung):
    """The regression test that actually matters, and it is free: deterministic string matching
    against 1,788 operator-confirmed registry answers, with no model in the loop.

    The first draft of this table scored 1762/1788, and all 26 misses were codes the registry really
    uses — 8033 spindle cell carcinoma, 8256/8257 minimally invasive adenocarcinoma, 8550, 8574,
    8141, 8002, 8023, 8144, 8800, 9180 — plus the six haematopoietic cases the spec excludes. This
    test is what found them, and it is what will find the next miss.
    """
    import csv
    rows = list(csv.DictReader(GT.open(encoding="utf-8")))
    problems: dict[str, list[str]] = {}
    for r in rows:
        for p in check_values(
                triple(r["gt_primary_site"], r["gt_histology"], r["gt_behavior"]), table=lung):
            problems.setdefault(p.kind, []).append(p.value)

    # Every remaining problem must be that declared scope boundary. Anything else is a real gap.
    unexpected = {k: sorted(set(v)) for k, v in problems.items() if k != EXCLUDED_BY_SPEC}
    assert not unexpected, (
        f"the table does not cover codes the registry actually uses: {unexpected}. "
        f"Add them; a value domain that rejects the registry's own answers is worse than none.")
    assert len(problems.get(EXCLUDED_BY_SPEC, [])) == 6, (
        "six haematopoietic cases were measured in this corpus; a change here means the cohort "
        "or the spec's exclusions moved and the accuracy denominator moved with them")


# ==========================================================================================
# The seam: the Task Contract declares the table, the prompt renders it
# ==========================================================================================
def test_the_lung_spec_declares_its_code_table():
    """Which code system a value belongs to is part of what the answer **means**, so the spec says
    it. The runtime does not guess it from the corpus or from a field name."""
    from acr.contract.spec import load_spec as _ls
    spec = _ls(site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml")
    assert spec.value_domain == "icdo3_lung"


def test_a_spec_with_no_code_system_gets_no_block():
    """Date and class-of-case variables have no ICD-O-3 value domain. Stuffing a wall of lung
    morphology codes at them to be ignored is prompt bloat."""
    from acr.contract.code_tables import code_domain_block
    from acr.contract.spec import load_specs as _lss
    blocks = {sid: code_domain_block(sp) for sid, sp in _lss(str(site.specs_root())).items()}
    assert blocks["STORE.400_522_523.site_histology_behavior"]
    assert not blocks["STORE.390.date_of_initial_diagnosis"]
    assert not blocks["STORE.610.class_of_case"]


def test_a_declared_table_that_does_not_exist_stops_the_spec_from_loading(tmp_path):
    """A typo means FAIL CLOSED. A missing table would otherwise render an empty value domain, and
    the run would look exactly like one that was given the codes — the same failure as a skill that
    silently supplies no guidance at all while the manifest reports that it did."""
    from acr.contract.code_tables import CodeTableError as CTE
    from acr.contract.spec import load_spec as _ls
    p = tmp_path / "S.2.yaml"
    p.write_text(
        "spec_id: S.2\nspec_version: 0.1.0\ndata_source: notes\nquestion: q\n"
        "value_domain: icdo3_atlantis\n"
        "fields:\n  - name: primary_site\n    type: string\n"
        "decision_rule: [r]\nevidence_rules:\n  counts_as_evidence: [anything]\n",
        encoding="utf-8")
    with pytest.raises(CTE) as e:
        _ls(p)
    assert "available:" in str(e.value)


def test_the_rendered_block_contains_the_subsite_facts_a_run_got_wrong():
    """End to end: in the text the model really reads, C341 has 'Upper lobe' written next to it."""
    from acr.contract.code_tables import code_domain_block
    from acr.contract.spec import load_spec as _ls
    b = code_domain_block(_ls(site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"))
    assert "C341  Upper lobe, lung" in b
    assert "C342  Middle lobe, lung" in b
    assert "C343  Lower lobe, lung" in b
    assert "left lung has no middle lobe" in " ".join(b.split())
    assert "8972" in b and "8973" in b
