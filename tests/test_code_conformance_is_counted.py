"""`check_values` had no caller, while its module docstring said the evaluation plane counts it.

`contract/code_tables.py` opens by naming three jobs the module does, the second being:

    2. `check_values()` 返回带类型的问题给评测面**计数**，永不用于拒绝。

`check_values` had ZERO non-test callers in `src/` or `tools/`. The table is loaded (fail-closed on
a typo), rendered into the prompt, and recorded in the manifest — all INPUT-side. Nothing compared
the value a run emitted back against the table it was shown.

The two failures the module's own docstring records as its motivation both pass uncounted today: a
run coding morphology `7205` (not an ICD-O-3 code at all) and one writing "C341 is the right middle
lobe" and coding accordingly (C341 is the UPPER lobe). `eval score` reports MISMATCH with no
indication that the value was not a code, and on an unkeyed variable reports nothing.

`prompt_block` — the neighbouring function with the same "no caller in src" shape — IS reached, so a
spot check finds the module wired in and `check_values` free-rides on the impression.

ADVISORY, and that is not a compromise: this repo removed five deterministic content checks after
they destroyed 58 correct values against 21 helps. A count is a finding; a refusal is a regression.
"""

from __future__ import annotations

import json

import pytest

from acr.contract.code_tables import load_table
from acr.contract.spec import load_spec
from acr.core import site
from acr.evaluation import evals as E

SPEC_WITH_DOMAIN = site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"
CFG = E.DetectorConfig(min_term_chars=4, max_rejection_repeats=2,
                       token_band=(1_000, 200_000), turn_band=(3, 60))


@pytest.fixture
def spec():
    if not SPEC_WITH_DOMAIN.is_file():
        pytest.skip("the value-domain spec is not in this checkout")
    s = load_spec(SPEC_WITH_DOMAIN)
    if not getattr(s, "value_domain", ""):
        pytest.skip("that spec no longer declares a value_domain")
    return s


def _run(tmp_path, value: dict, spec_id: str):
    m = tmp_path / "r.manifest.json"
    m.write_text(json.dumps({
        "patient_id": "P1", "spec_id": spec_id,
        "answer": {"status": "FOUND", "status_kind": "value", "value": value},
        "coverage_state": {"n_read": 3, "searched_terms": ["carcinoma"]},
    }), encoding="utf-8")
    return E.RunRecord.from_manifest(m)


def test_the_population_is_real(spec):
    """Guards the rest: a spec with no value domain makes every test here inert."""
    table = load_table(spec.value_domain)
    assert table.axes


def test_a_malformed_code_is_counted(tmp_path, spec):
    """`7205` — the exact value from the failure the module records as its motivation."""
    run = _run(tmp_path, {"histology": "7205"}, spec.spec_id)
    found = E.detect_value_domain_violation(run, spec=spec)
    assert [f.detector for f in found] == ["value_domain_violation"]
    assert found[0].evidence["problems"][0]["value"] == "7205"


def test_a_conforming_code_is_not_counted(tmp_path, spec):
    """VACUOUS IN ITS FIRST FORM. `CodeTable` has no `codes` attribute — the code lists live on each
    AXIS (`axis.codes`) — so `getattr(table, "codes", {})` was `{}`, `good` was `None`, and the test
    submitted an EMPTY value dict. `detect_value_domain_violation` returns `[]` for an empty value
    before it consults the table at all, so the assertion held no matter what the detector did."""
    table = load_table(spec.value_domain)
    axis = next(a for a in table.axes.values() if a.code_shape and a.codes)
    good = sorted(axis.codes)[0]
    run = _run(tmp_path, {axis.field: good}, spec.spec_id)
    assert good, "the axis must offer a real code, or this test is back to asserting nothing"
    assert E.detect_value_domain_violation(run, spec=spec) == []


def test_the_conforming_case_would_fail_if_the_detector_were_inverted(tmp_path, spec):
    """The mutation the test above must be sensitive to: a real code and a fake one must differ."""
    table = load_table(spec.value_domain)
    axis = next(a for a in table.axes.values() if a.code_shape and a.codes)
    good, bad = sorted(axis.codes)[0], "ZZZ999"
    assert E.detect_value_domain_violation(_run(tmp_path, {axis.field: good}, spec.spec_id),
                                          spec=spec) == []
    assert E.detect_value_domain_violation(_run(tmp_path, {axis.field: bad}, spec.spec_id),
                                          spec=spec) != []


def test_it_never_refuses_only_reports(tmp_path, spec):
    """The severity that matters. This repo removed five deterministic content checks after they
    destroyed 58 correct values against 21 helps; a code check that could reject an answer would be
    the sixth."""
    run = _run(tmp_path, {"histology": "7205"}, spec.spec_id)
    found = E.detect_value_domain_violation(run, spec=spec)
    assert found and all(f.severity != E.CRITICAL for f in found), \
        "a value-domain finding must be advisory"


def test_no_spec_means_no_finding_and_no_claim(tmp_path):
    """A run of a contract with no declared domain has nothing to check, and must not report
    'conformant' — there is no table to be conformant to."""
    run = _run(tmp_path, {"histology": "7205"}, "SPEC.without.domain")
    assert E.detect_value_domain_violation(run, spec=None) == []


def test_the_detector_runs_in_the_default_bundle(tmp_path, spec):
    """The whole point: it must be REACHED, not merely importable."""
    run = _run(tmp_path, {"histology": "7205"}, spec.spec_id)
    names = [f.detector for f in E.run_detectors(run, config=CFG, spec=spec)]
    assert "value_domain_violation" in names
