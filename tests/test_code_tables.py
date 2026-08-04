"""The code table loader has to be able to express a code system that is not about cancer.

WHY THIS IS A TEST AND NOT A REFACTOR
-------------------------------------
`icdo3.load_table` required every table to carry the three keys `topography` / `morphology` /
`behavior`, `prompt_block` hard-coded those three words as its section headings, and
`_TOPO = C\\d{3}` and `_MORPH = \\d{4}` were module constants. So a LOINC lipid panel or an RxNorm
drug class **simply could not be expressed**: `load_table` raised `CodeTableError` outright, and
`spec.load_spec` is fail-closed, so the whole spec load failed.

That welded the framework to one use case, and it is a thing only visible from the angle of "can a
second domain be loaded into it" — which is why the first test here is a lipid panel, and why it
existed before the code did.

The equivalence of the three cancer tables is pinned just as hard: the migration is mechanical, but
a mechanical migration drops lines too.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODES = ROOT / "assets" / "codes"

#: A code table with nothing to do with cancer, written the way an asset should be. Two axes, each
#: carrying its own label, field name and code shape.
LIPID_TABLE = {
    "table_id": "loinc_lipid_panel",
    "table_version": "1",
    "source_authority": {"scope": "fasting lipid panel, LOINC 2.77"},
    "normalization": {"uppercase": False, "strip_pattern": r"\s"},
    "axes": {
        "analyte": {
            "label": "ANALYTE (LOINC)",
            "field": "analyte_code",
            "code_shape": r"\d{4,5}-\d",
            "shape_description": "four or five digits, a hyphen, one check digit",
            "codes": {
                "2093-3": {"name": "Cholesterol, total"},
                "2571-8": {"name": "Triglyceride"},
                "13457-7": {"name": "LDL cholesterol, calculated",
                            "aliases": ["LDL-C calc"]},
            },
            "unspecified": ["2093-3"],
        },
        "interpretation": {
            "label": "INTERPRETATION",
            "field": "flag",
            "codes": {
                "H": {"name": "above reference range", "admissible": True},
                "N": {"name": "within reference range", "admissible": True},
                "X": {"name": "not reported", "admissible": False},
            },
        },
    },
    "guidance": ["A calculated LDL is not admissible when triglycerides exceed 400 mg/dL."],
}


def _write(tmp_path: pathlib.Path, doc: dict, name: str) -> pathlib.Path:
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False),
                                           encoding="utf-8")
    return tmp_path / f"{name}.yaml"


# ============================================================ THE POINT OF THE WHOLE CHANGE
def test_a_code_system_with_no_cancer_axes_loads(tmp_path):
    """As things stand today, `icdo3.load_table` raises CodeTableError here."""
    from acr.contract.code_tables import load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))
    assert t.table_id == "loinc_lipid_panel"
    assert list(t.axes) == ["analyte", "interpretation"]
    assert t.axes["analyte"].codes["2093-3"]["name"] == "Cholesterol, total"


def test_the_prompt_block_uses_the_tables_own_axis_labels(tmp_path):
    """The section headings come from the YAML, not from Python.

    The old `prompt_block` hard-coded the three headings TOPOGRAPHY / MORPHOLOGY / BEHAVIOUR and the
    sentence "if a diagnosis has no ICD-O-3 morphology then the finding is not a reportable
    neoplasm". A lipid panel handed those headings is nonsense.
    """
    from acr.contract.code_tables import load_table, prompt_block
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    block = prompt_block(load_table("loinc_lipid_panel", codes_dir=str(tmp_path)))
    assert "ANALYTE (LOINC)" in block and "INTERPRETATION" in block
    assert "2093-3" in block and "Cholesterol, total" in block
    assert "LDL-C calc" in block                      # aliases reach the prompt
    for cancer in ("TOPOGRAPHY", "MORPHOLOGY", "BEHAVIOUR", "neoplasm", "ICD-O-3", "tumour"):
        assert cancer not in block, f"{cancer!r} appears in a lipid panel's prompt block"


def test_an_axis_order_declared_in_yaml_is_the_render_order(tmp_path):
    """Axis order is the asset's decision. The other way round makes a reading order like "site
    before morphology" impossible to express."""
    from acr.contract.code_tables import load_table, prompt_block
    doc = {**LIPID_TABLE, "axes": {k: LIPID_TABLE["axes"][k]
                                   for k in ("interpretation", "analyte")}}
    _write(tmp_path, doc, "reordered")
    block = prompt_block(load_table("reordered", codes_dir=str(tmp_path)))
    assert block.index("INTERPRETATION") < block.index("ANALYTE (LOINC)")


def test_normalization_is_declared_by_the_table_not_by_the_module(tmp_path):
    """`C18.7 -> C187` is ICD-O-3's notation, not every code system's.

    The hyphen in the lipid panel's LOINC codes is **part of the code**, and a normalisation
    function with `[.\\s]` and `split_on='/'` hard-coded into it corrupts another system's codes —
    so the rules travel with the table.
    """
    from acr.contract.code_tables import load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))
    assert t.normalize("13457-7") == "13457-7"          # the hyphen is kept
    assert t.normalize(" 2093-3 ") == "2093-3"          # whitespace stripping is declared
    assert t.normalize("ldl") == "ldl"                  # uppercasing is declared off


def test_a_table_with_no_axes_is_refused(tmp_path):
    """An empty table is more dangerous than a missing one: it renders an empty value domain, and
    the run then looks exactly like one that was given the codes.

    Same reason `acr.contract.skills` refuses a missing skill.
    """
    from acr.contract.code_tables import CodeTableError, load_table
    _write(tmp_path, {"table_id": "empty", "axes": {}}, "empty")
    with pytest.raises(CodeTableError, match="no axes"):
        load_table("empty", codes_dir=str(tmp_path))


def test_a_missing_table_names_what_is_available(tmp_path):
    from acr.contract.code_tables import CodeTableError, load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    with pytest.raises(CodeTableError, match="loinc_lipid_panel"):
        load_table("nope", codes_dir=str(tmp_path))


# ============================================================ CHECKED BY AXIS NAME, NOT FIELD NAME
def test_check_values_is_keyed_by_axis_not_by_cancer_field_names(tmp_path):
    """The old signature was `check_codes(site, histology, behavior)` — three cancer field names
    written into the parameters.

    Generalised, callers pass values by **axis name** and the field name is read from the axis's
    `field`, so the same function can check a lipid panel.
    """
    from acr.contract.code_tables import check_values, load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))

    assert check_values({"analyte": "2093-3", "interpretation": "N"}, table=t) == []

    bad_shape = check_values({"analyte": "cholesterol"}, table=t)
    assert [p.kind for p in bad_shape] == ["MALFORMED"]
    assert bad_shape[0].field == "analyte_code"        # the field name comes from the axis
    assert "four or five digits" in bad_shape[0].message

    unknown = check_values({"analyte": "9999-9"}, table=t)
    assert [p.kind for p in unknown] == ["NOT_IN_TABLE"]

    not_admissible = check_values({"interpretation": "X"}, table=t)
    assert [p.kind for p in not_admissible] == ["NOT_ADMISSIBLE"]

    # An empty value is a matter of abstention, not a matter for the code table.
    assert check_values({"analyte": "", "interpretation": None}, table=t) == []


def test_an_unknown_axis_name_is_refused_rather_than_ignored(tmp_path):
    """Silently ignoring a misspelled axis name amounts to reporting "the check passed" while
    nothing whatever was checked."""
    from acr.contract.code_tables import CodeTableError, check_values, load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))
    with pytest.raises(CodeTableError, match="analytee"):
        check_values({"analytee": "2093-3"}, table=t)


# ============================================================ EQUIVALENCE OF THE THREE REAL TABLES
REAL = {"icdo3_lung": {"topography": 8, "morphology": 55, "behavior": 6},
        "icdo3_colorectal": {"topography": 16, "morphology": 37, "behavior": 6},
        "icdo3_multisite": {"topography": 24, "morphology": 69, "behavior": 6}}


@pytest.mark.parametrize("name", sorted(REAL))
def test_the_real_tables_survive_migration_with_every_code(name: str):
    """A mechanical migration drops lines too. Each axis's entry count is pinned to the value
    measured before the migration."""
    from acr.contract.code_tables import load_table
    t = load_table(name)
    got = {ax: len(a.codes) for ax, a in t.axes.items()}
    assert got == REAL[name], f"{name} has a different entry count after the migration: {got}"


@pytest.mark.parametrize("name", sorted(REAL))
def test_the_real_tables_keep_their_icdo3_notation_folding(name: str):
    """`C18.7 -> C187` and `8140/3 -> 8140` must still be here — they are 4 of the 6 useful firings
    of the old check."""
    from acr.contract.code_tables import load_table
    t = load_table(name)
    assert t.normalize("C18.7") == "C187"
    assert t.normalize("c34.1") == "C341"
    assert t.normalize("8140/3") == "8140"


def test_the_lung_table_still_knows_which_lobe_c341_is():
    """Regression: one run wrote "C341 is the right middle lobe" and coded on that basis, and C341
    is the **upper** lobe.

    This is one of the reasons the whole code table exists, and generalising it must not drop it.
    """
    from acr.contract.code_tables import load_table
    t = load_table("icdo3_lung")
    assert "upper" in (t.axes["topography"].codes["C341"].get("name") or "").lower()


def test_the_cancer_prompt_block_still_carries_its_own_warnings():
    """The table's own warnings (no registrar has checked it; behaviour is a field of its own) now
    live in the YAML, and they must still be in the prompt."""
    from acr.contract.code_tables import load_table, prompt_block
    block = prompt_block(load_table("icdo3_lung"))
    assert "TOPOGRAPHY" in block and "MORPHOLOGY" in block
    assert "C341" in block
    low = block.lower()
    assert "registrar" in low or "signed off" in low or "no registrar" in low
