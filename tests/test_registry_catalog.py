"""L0 resolution: a requested name becomes a spec and a field, or the command stops.

The property under test is not "the resolver finds things". It is that the resolver never
returns a request it did not fully satisfy. A cohort extract that is quietly short one
column produces a concordance run whose denominator went to zero for a reason nobody was
told, and every number downstream is then wrong in a way no artifact records.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acr.concordance import load_guideline, variables_from_answer
from acr.registry_catalog import (AmbiguousVariableError, UnknownVariableError,
                                  VariableCatalog, VariableResolutionError,
                                  check_guideline_bindings, normalise_name)
from acr.spec import load_specs

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
GUIDELINE = ROOT / "guidelines" / "nccn_nsclc_subset.yaml"

# Three tests here briefly skipped on UnprovenancedElementError, because specs/*.yaml carried
# no `provenance:` records and so would not load at all. They do now, as of 2026-07-27, and
# the guards are deleted rather than left in place: a catalogue that cannot read the shipped
# specs is a broken catalogue, and that has to be a failure.


@pytest.fixture(scope="module")
def cat() -> VariableCatalog:
    return VariableCatalog.from_directory(SPECS)


# ------------------------------------------------------------------ the vocabulary
def test_every_field_of_every_shipped_spec_is_resolvable(cat):
    """The catalogue's vocabulary is the union of the specs' fields — no more, no less."""
    expected = {f.name for s in load_specs(SPECS).values() for f in s.fields}
    assert set(cat.known_names()) == expected
    for name in expected:
        res = cat.resolve(name)
        assert res.names == [name]


def test_resolved_names_are_what_variables_from_answer_keys_on(cat):
    """The resolver's output feeds `variables_from_answer` directly. If it emitted anything
    other than a real field name the variable would never arrive and L4 would block on it
    forever — the failure the stage spec's rebinding already had to fix once."""
    res = cat.resolve("primary_site,histology,behavior")
    sid = res.spec_ids[0]
    answer = {"status": "FOUND", "value": {"primary_site": "C341", "histology": "8140",
                                           "behavior": "3"}}
    got = variables_from_answer(answer, res.fields_for(sid), source=sid)
    assert set(got) == {"primary_site", "histology", "behavior"}
    assert all(v.status == "FOUND" for v in got.values())


# ------------------------------------------------------------------ failing loudly
def test_unknown_variable_raises_with_the_whole_vocabulary(cat):
    with pytest.raises(UnknownVariableError) as ei:
        cat.resolve("tx1_date")
    assert ei.value.unknown == ["tx1_date"]
    assert "primary_site" in ei.value.known
    assert "known variables:" in str(ei.value)


def test_every_unknown_is_reported_not_just_the_first(cat):
    """Failing on the first typo makes the operator rerun a cohort extract once per typo."""
    with pytest.raises(UnknownVariableError) as ei:
        cat.resolve("site,histology,tx1_date")
    assert ei.value.unknown == ["site", "tx1_date"]


def test_a_partly_known_request_resolves_nothing(cat):
    """The load-bearing one: never return the known subset. A short extract that says so
    nowhere is worse than no extract."""
    with pytest.raises(UnknownVariableError):
        cat.resolve("histology,not_a_variable")


def test_empty_request_is_an_error_not_an_empty_resolution(cat):
    with pytest.raises(VariableResolutionError):
        cat.resolve("")


def test_no_substring_or_prefix_matching(cat):
    """`primary` and `site` are not `primary_site`. Substring matching on a name a human
    typed is the mechanism that filed `Fine-Needle-Report` outside ["Pathology","Cytology"]
    and swept `Speech-Language-Pathology-Note` in."""
    for near in ("primary", "site", "hist", "stag", "clinical"):
        with pytest.raises(UnknownVariableError):
            cat.resolve(near)


def test_suggestions_never_resolve_anything(cat):
    """difflib appears in the error text and nowhere else."""
    with pytest.raises(UnknownVariableError) as ei:
        cat.resolve("histolgy")
    assert "histology" in ei.value.suggestions["histolgy"]


# ------------------------------------------------------------------ normalisation
@pytest.mark.parametrize("typed", ["primary_site", "Primary Site", "primary-site",
                                   "  PRIMARY_SITE  "])
def test_the_same_name_typed_by_different_people(cat, typed):
    assert cat.resolve(typed).names == ["primary_site"]


def test_normalise_folds_only_case_and_separators():
    assert normalise_name("Primary Site") == "primary_site"
    assert normalise_name("primary") != normalise_name("primary_site")


# ------------------------------------------------------------------ aliases
@pytest.mark.parametrize("alias,expected_kind", [
    ("STORE.400_522_523.site_histology_behavior", "spec_id"),
    ("site_histology_behavior", "spec_stem"),
    ("400", "store_item"),
    ("store.522", "store_item"),
])
def test_spec_level_aliases_expand_to_every_field(cat, alias, expected_kind):
    res = cat.resolve(alias)
    assert res.names == ["primary_site", "histology", "behavior"]
    assert {v.matched_on for v in res.variables} == {expected_kind}


def test_a_store_item_reaches_the_spec_not_a_single_field(cat):
    """Nothing in a spec says which field answers which STORE item — that binding is
    declared in the guideline. Guessing it here would silently drop two of three."""
    assert len(cat.resolve("522").variables) == 3


def test_stem_that_is_also_its_only_field_is_not_an_ambiguity(cat):
    """STORE.610's stem `class_of_case` is also its only field name. Same spec, so the two
    targets collapse rather than colliding."""
    res = cat.resolve("class_of_case")
    assert res.names == ["class_of_case"]
    assert res.spec_ids == ["STORE.610.class_of_case"]


def test_no_alias_is_invented_in_python(cat):
    """`site -> primary_site` is a naming judgement. It belongs in the spec YAML, where a
    domain expert can read it, not in the enforced layer where nobody can."""
    assert not any("aliases" in (s.model_extra or {}) for s in cat.specs.values())
    with pytest.raises(UnknownVariableError):
        cat.resolve("site")


def test_a_declared_alias_is_honoured(tmp_path):
    """The extension point exists and works, at the documented cost of a changed spec_hash."""
    src = yaml.safe_load((SPECS / "STORE.400_522_523.site_histology_behavior.yaml"
                          ).read_text(encoding="utf-8"))
    src["aliases"] = ["tumour_coding"]
    src["fields"][0]["aliases"] = ["site"]
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(src), encoding="utf-8")
    cat = VariableCatalog.from_directory(tmp_path)
    assert cat.resolve("site").names == ["primary_site"]
    assert cat.resolve("tumour_coding").names == ["primary_site", "histology", "behavior"]
    assert cat.resolve("site").variables[0].matched_on == "declared_alias"


# ------------------------------------------------------------------ ambiguity
def test_the_shipped_catalogue_has_no_ambiguous_name(cat):
    for alias, spec_ids in cat.known_aliases().items():
        assert len(spec_ids) == 1, f"{alias} reaches {spec_ids}"


def test_two_specs_declaring_one_field_is_an_error_naming_both(tmp_path):
    """First-match-wins is safe in `assign_strata` because the spec author declared the
    order. Nothing declares an order over specs."""
    for name in ("STORE.400_522_523.site_histology_behavior.yaml",
                 "ablation/STORE.400_522_523.unstratified.yaml"):
        (tmp_path / Path(name).name).write_text((SPECS / name).read_text(encoding="utf-8"),
                                                encoding="utf-8")
    cat = VariableCatalog.from_directory(tmp_path)
    with pytest.raises(AmbiguousVariableError) as ei:
        cat.resolve("primary_site")
    assert len(ei.value.spec_ids) == 2
    assert "STORE.400_522_523.site_histology_behavior.UNSTRATIFIED" in ei.value.spec_ids


def test_the_scan_is_not_recursive_and_that_is_why(cat):
    """`specs/ablation/` holds a second copy of three field names under another spec_id. A
    recursive scan would make `primary_site`, `histology` and `behavior` permanently
    ambiguous — an ablation arm is selected by path, never by name."""
    assert (SPECS / "ablation").is_dir()
    assert "STORE.400_522_523.site_histology_behavior.UNSTRATIFIED" not in cat.specs
    assert cat.resolve("primary_site").spec_ids == ["STORE.400_522_523.site_histology_behavior"]


# ------------------------------------------------------------------ the unit of work
def test_many_fields_of_one_spec_are_one_run(cat):
    """The whole reason L0 groups by spec: three variables, one pass over the chart. Running
    per variable would pay for the chart three times and could code three different sites."""
    res = cat.resolve("primary_site,histology,behavior")
    assert res.spec_ids == ["STORE.400_522_523.site_histology_behavior"]
    assert len(res.fields_for(res.spec_ids[0])) == 3


def test_one_variable_named_twice_is_one_column(cat):
    """`histology` and `STORE.522` are the same request typed two ways."""
    res = cat.resolve("histology,522")
    assert res.names.count("histology") == 1
    assert res.spec_ids == ["STORE.400_522_523.site_histology_behavior"]


def test_outside_notes_variables_resolve_but_are_flagged(cat):
    """STORE.610 is forced to SPEC_INSUFFICIENT at finalize. The resolver reports that and
    does not refuse: the variable really is known, and the refusal is the run's to make."""
    res = cat.resolve("class_of_case")
    assert [v.name for v in res.not_from_notes()] == ["class_of_case"]
    assert res.variables[0].data_source == "outside_notes"
    assert cat.resolve("primary_site").not_from_notes() == []


def test_resolution_round_trips_to_json(cat):
    d = cat.resolve("stage").to_dict()
    assert d["requested"] == ["stage"]
    assert d["spec_ids"] == ["STORE.700_880.stage"]
    assert all(set(v) == {"requested", "name", "spec_id", "matched_on", "data_source"}
               for v in d["variables"])


# ------------------------------------------------------------------ guideline bindings
def test_the_shipped_guideline_binds_to_real_fields(cat):
    assert check_guideline_bindings(cat, load_guideline(GUIDELINE)) == []


def test_a_renamed_field_is_caught_before_it_blocks_forever(cat, tmp_path):
    """The measured failure: a guideline naming `ajcc_pathologic_stage` when the spec's field
    is `pathologic_stage_group`. Nothing errors, the variable never arrives, and every case
    comes back NOT_ASSESSABLE naming a variable the operator believes was extracted."""
    data = yaml.safe_load(GUIDELINE.read_text(encoding="utf-8"))
    data["unknown_value_codes"]["ajcc_pathologic_stage"] = \
        data["unknown_value_codes"].pop("pathologic_stage_group")
    for rec in data["recommendations"]:
        for d in rec["required_inputs"]:
            if d["name"] == "pathologic_stage_group":
                d["name"] = "ajcc_pathologic_stage"
        for c in rec["applies_when"]:
            if c.get("var") == "pathologic_stage_group":
                c["var"] = "ajcc_pathologic_stage"
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    bad = check_guideline_bindings(cat, load_guideline(p))
    assert any("ajcc_pathologic_stage" in b and "no spec" in b for b in bad)


def test_a_wrong_spec_id_binding_is_caught(cat, tmp_path):
    data = yaml.safe_load(GUIDELINE.read_text(encoding="utf-8"))
    for d in data["recommendations"][0]["required_inputs"]:
        if d["name"] == "histology":
            d["spec_id"] = "STORE.700_880.stage"
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    bad = check_guideline_bindings(cat, load_guideline(p))
    assert any("declares spec_id" in b for b in bad)


def test_binding_check_returns_a_list_and_never_raises(cat):
    """Same contract as `check_field_formats` and `validate_guideline`."""
    class Empty:
        recommendations = ()
    assert check_guideline_bindings(cat, Empty()) == []
