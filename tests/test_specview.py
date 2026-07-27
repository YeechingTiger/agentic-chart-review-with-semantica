"""The reviewer-facing render of a spec, and the sign-off that freezes a reviewer's assent.

Everything here is a property of the DOCUMENT a physician receives, not of the YAML. That
distinction is the whole point: the YAML is already tested to death by test_stage_spec and
test_gate_must_reject, and none of those tests can tell you whether a thoracic oncologist
could read the thing. So the assertions below are about readability as a mechanical
property -- no regex reaches the page, no stratum name reaches the page, no ICD-O code
arrives without its name -- plus the two claims the document makes about itself: that the
made-up list is complete, and that a sign-off dies when the text it approved changes.

`spec review` is not tested for prose quality; nothing can be. It is tested for the failures
that would make the prose a lie.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from acr.cli import app
from acr.spec import load_spec
from acr.specview import (JARGON, MODEL_AUTHORED, SECTION_TITLES, SIGNED, STALE, UNSIGNED,
                          decisions, elements, load_signoffs, record_signoff, render_review,
                          signoff_status)

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "specs").glob("*.yaml"))
SHB = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
STAGE = ROOT / "specs" / "STORE.700_880.stage.yaml"
DXDATE = ROOT / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
RECUR = ROOT / "specs" / "STORE.1860_1880.first_recurrence.yaml"
COC = ROOT / "specs" / "STORE.610.class_of_case.yaml"

runner = CliRunner()


def review(path: Path, **kw) -> str:
    return render_review(load_spec(path), source_path=path, **kw)


# --------------------------------------------------------------------- it renders at all
@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_shipped_spec_renders_with_all_seven_sections_in_order(path: Path):
    """A spec that cannot be rendered is a spec no clinician will ever review."""
    doc = review(path)
    at = [doc.find(f"## {t}") for t in SECTION_TITLES]
    missing = [t for t, i in zip(SECTION_TITLES, at) if i < 0]
    assert not missing, f"{path.name} is missing section(s): {missing}"
    assert at == sorted(at), f"{path.name} renders its sections out of order"


# ------------------------------------------------------------- nothing engineering leaks
@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_no_field_format_regex_reaches_the_reviewer(path: Path):
    """`cT(X|0|is|1mi|1a|...)` is not a fact about lung cancer. A registrar who hits one
    stops reading, and the sentence they stopped before is where the clinical content was.

    `CCYYMMDD` is exempt, and is asserted the other way round below: it is not a regex, it is
    registry notation that the software mistakes for one, and paraphrasing it away would
    conceal a field that rejects every value it is given.
    """
    doc = review(path)
    for f in load_spec(path).fields:
        if f.format and any(ch in f.format for ch in r"\[](){}|+*?^$."):
            assert f.format not in doc, f"{path.name}: raw pattern {f.format!r} is in the document"
    for metachar in (r"\d", r"(?", r"[A-Z]", r"{3}", r"{4}"):
        assert metachar not in doc, f"{path.name}: regex fragment {metachar!r} is in the document"


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_no_stratum_or_gate_jargon_reaches_the_reviewer(path: Path):
    """`may_mention`, `max_elusion_upper` and `Clopper-Pearson` are the vocabulary of the
    people who wrote the bug. The reviewer is being asked to catch it."""
    low = review(path).lower()
    found = sorted(w for w in JARGON if w.lower() in low)
    assert not found, f"{path.name}: jargon reached the reviewer: {found}"


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_icdo_code_carries_its_name(path: Path):
    """C349 means nothing to a reader who is being asked whether C349 is the right answer."""
    doc = review(path)
    bare = []
    for m in re.finditer(r"\b(C\d{3}|[89]\d{3})\b", doc):
        tail = doc[m.end():m.end() + 2]
        if not tail.startswith(" ("):
            bare.append((m.group(0), doc[max(0, m.start() - 60):m.end() + 20]))
    assert not bare, f"{path.name}: unglossed code(s): {[b[0] for b in bare]}\n{bare[:3]}"


# ------------------------------------------------------- the CCYYMMDD bug must be visible
def test_registry_notation_masquerading_as_a_pattern_is_reported_loudly():
    """STORE.390 declares `format: CCYYMMDD`, which check_field_formats applies with
    re.fullmatch -- so it rejects 20100612 and every other valid date. It is still unfixed.
    A review document that renders this field as though it worked is worse than no document.
    """
    doc = review(DXDATE)
    assert "CCYYMMDD" in doc, "the broken pattern must be named, not paraphrased away"
    assert "every" in doc.lower() and "reject" in doc.lower()
    ids = [d.element_id for d in decisions(load_spec(DXDATE), source_path=DXDATE)]
    assert any(i.startswith("answer.date_of_initial_diagnosis") for i in ids), \
        "a field that can never validate is a decision for the reviewer, not a footnote"


def test_a_working_pattern_is_not_reported_as_broken():
    """STORE.700's formats are real regexes. Crying wolf on them would train the reader to
    skip the warning that matters."""
    doc = review(STAGE)
    assert "CCYYMMDD" not in doc
    assert "reject every valid value" not in doc


# ------------------------------------------------------------------- WHAT WE MADE UP
def _fixture(tmp_path: Path, extra: dict | None = None) -> Path:
    body = {
        "spec_id": "TEST.1.thing",
        "question": "Is the thing present?",
        "fields": [{"name": "thing", "type": "string", "description": "the thing"}],
        "decision_rule": ["Code the thing when the thing is documented."],
        "conflict_rules": [{"if": "two notes disagree", "then": "prefer the later one"}],
        "abstention": {"EVIDENCE_INSUFFICIENT": "nothing in the chart says."},
    }
    body.update(extra or {})
    p = tmp_path / "TEST.1.thing.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return p


def test_an_unattributed_element_is_model_authored_and_appears_in_what_we_made_up(tmp_path):
    """The default must be `we made this up`. The opposite default lets a physician sign a
    fabricated rule believing the CoC manual wrote it, which is the failure this whole
    document exists to prevent."""
    p = _fixture(tmp_path)
    els = elements(load_spec(p), source_path=p)
    rule = next(e for e in els if e.element_id == "rule.1")
    assert rule.provenance == MODEL_AUTHORED
    doc = review(p)
    made_up = doc.split(f"## {SECTION_TITLES[5]}")[1].split(f"## {SECTION_TITLES[6]}")[0]
    assert "rule.1" in made_up


def test_declared_provenance_takes_an_element_out_of_what_we_made_up(tmp_path):
    p = _fixture(tmp_path, {"provenance": {"rule.1": "source_authority"}})
    els = elements(load_spec(p), source_path=p)
    assert next(e for e in els if e.element_id == "rule.1").provenance == "source_authority"
    doc = review(p)
    made_up = doc.split(f"## {SECTION_TITLES[5]}")[1].split(f"## {SECTION_TITLES[6]}")[0]
    assert "rule.1" not in made_up


def _made_up(p: Path) -> str:
    doc = review(p)
    return doc.split(f"## {SECTION_TITLES[5]}")[1].split(f"## {SECTION_TITLES[6]}")[0]


def test_an_editorial_citation_takes_an_element_out_of_what_we_made_up(tmp_path):
    """The property the test above is reaching for, through the channel that carries it.

    `decision_rule` is not an enforced element and never will be -- no code path branches on
    it -- so the enforced block cannot hold its provenance without inventing a fake runtime
    dependency. `editorial_provenance` is the channel for exactly this, and a record with a
    real locator in its basis is what shrinks section 6.
    """
    p = _fixture(tmp_path, {"editorial_provenance": [{
        "element": "rule.1", "origin": "store_manual",
        "basis": "CoC STORE 2025, Primary Site [400], coding rule 1 — transcribed verbatim"}]})
    els = elements(load_spec(p), source_path=p)
    rule = next(e for e in els if e.element_id == "rule.1")
    assert rule.provenance == "store_manual" and rule.recorded
    assert "rule.1" not in _made_up(p)


def test_a_label_with_no_locator_does_not_shrink_what_we_made_up(tmp_path):
    """The other half, and the one that matters more.

    `source_authority` -- a bare unverified label naming the document at the top of the file
    -- was the ONLY marking these specs had, and letting it take a sentence off this list is
    the laundering both this module and `acr.spec` exist to stop. It stays on the list, and
    the bad attribution is reported rather than honoured.
    """
    p = _fixture(tmp_path, {"editorial_provenance": [{
        "element": "rule.1", "origin": "store_manual",
        "basis": "it is the kind of thing the manual says"}]})
    rule = next(e for e in elements(load_spec(p), source_path=p) if e.element_id == "rule.1")
    assert rule.provenance == MODEL_AUTHORED and not rule.recorded
    made_up = _made_up(p)
    assert "rule.1" in made_up
    assert "do not hold up" in made_up, "a refused attribution must be reported, not dropped"


@pytest.mark.parametrize("junk", [
    {"editorial_provenance": [{"element": "rule.1", "origin": "not_an_origin", "basis": "x"}]},
    {"editorial_provenance": [{"element": "boundary.99", "origin": "model_authored",
                               "basis": "no external source"}]},
    {"editorial_provenance": {"rule.1": "source_authority"}},
    {"editorial_provenance": ["a bare string somebody was part-way through typing"]},
])
def test_a_half_written_editorial_block_still_loads(tmp_path, junk):
    """Advisory means advisory. An author annotating a file one rule at a time must be able to
    load and render it at every intermediate state, or the channel goes unused -- which is how
    the only marking these specs had ended up being one nobody could check."""
    p = _fixture(tmp_path, junk)
    load_spec(p)
    assert review(p), "a malformed advisory record must not stop the document rendering"


def test_a_missing_editorial_record_is_reported_as_a_finding(tmp_path):
    """Missing is the normal state and it must still be visible. `origin_not_recorded` is not
    the same claim as `model_authored`: one says nobody wrote anything down, the other says
    somebody wrote down that a model did it."""
    from acr.spec import ORIGIN_NOT_RECORDED
    from acr.specview import element_ids, provenance_findings
    p = _fixture(tmp_path)
    spec = load_spec(p)
    ids = element_ids(spec, source_path=p)
    missing = [f.element for f in provenance_findings(spec, source_path=p)
               if f.problem == ORIGIN_NOT_RECORDED]
    assert set(missing) == set(ids), "every unattributed statement is a finding"
    assert "origin not recorded" in _made_up(p)


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_the_findings_agree_with_the_document(path: Path):
    """Two accounts of what is attributed is worse than one. The elements reported as
    `origin not recorded` must be exactly the ones section 6 tags that way — the enforced
    block attributes a keyword list under a runtime path, and the document shows it under an
    id of its own."""
    from acr.spec import ORIGIN_NOT_RECORDED
    from acr.specview import provenance_findings
    spec = load_spec(path)
    els = elements(spec, source_path=path)
    reported = {f.element for f in provenance_findings(spec, source_path=path)
                if f.problem == ORIGIN_NOT_RECORDED}
    assert reported == {e.element_id for e in els if not e.recorded}


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_what_we_made_up_lists_every_model_authored_element_and_never_elides(path: Path):
    """`do not soften it, do not bury it, do not omit it when it is long` -- asserted, because
    the temptation to cap a long list at ten items with `and 47 more` is exactly the
    softening the section exists to refuse."""
    spec = load_spec(path)
    els = elements(spec, source_path=path)
    mine = [e for e in els if e.provenance == MODEL_AUTHORED]
    doc = review(path)
    made_up = doc.split(f"## {SECTION_TITLES[5]}")[1].split(f"## {SECTION_TITLES[6]}")[0]
    absent = [e.element_id for e in mine if e.element_id not in made_up]
    assert not absent, f"{path.name}: {len(absent)} made-up element(s) missing: {absent[:8]}"
    assert str(len(mine)) in made_up, "the count must be stated"
    assert not re.search(r"\.\.\.\s*and \d+ more", made_up)


# --------------------------------------------------------------- DECISIONS WE NEED YOU TO
def _decisions_text(path: Path) -> str:
    doc = review(path)
    return doc.split(f"## {SECTION_TITLES[4]}")[1].split(f"## {SECTION_TITLES[5]}")[0]


def test_store400_surfaces_the_radiology_localisation_decision():
    """The live one. `establishes: [primary_site]` on the imaging stratum is a clinical
    ruling -- radiology may say where the tumour started -- written as a YAML key, which is
    why no clinician has ever seen it. P03 was coded lung NOS because the earlier
    version said the opposite."""
    t = _decisions_text(SHB).lower()
    assert "imaging" in t or "radiolog" in t
    assert "where" in t and ("start" in t or "origin" in t)


def test_store400_surfaces_the_hedged_wording_decision():
    """`favor squamous cell carcinoma` -> 8070 or 8046? The spec has already decided. A
    pathologist may well disagree, and cannot disagree with a `contradicted_by` list."""
    t = _decisions_text(SHB).lower()
    assert "favor squamous" in t
    assert "8070" in t


def test_store400_surfaces_the_residual_miss_tolerance_as_a_number_a_clinician_can_refuse():
    t = _decisions_text(SHB)
    assert "12" in t and "%" in t
    assert "25" in t, "the sample size that produces the 12% must be shown with it"


def test_a_placeholder_awaiting_clinical_input_is_the_first_decision_listed():
    """STORE.1860_1880 ships `surveillance_schedule: PLACEHOLDER_REQUIRES_CLINICAL_INPUT`.
    Anything else at the top of the list would bury the one item that is unarguably the
    reviewer's to answer."""
    ds = decisions(load_spec(RECUR), source_path=RECUR)
    assert ds, "the recurrence spec has decisions"
    assert "undecided" in ds[0].choice.lower(), (
        f"the first decision is {ds[0].element_id}: {ds[0].question!r}, which is not the one "
        f"the specification itself says it cannot answer")
    assert "how often" in ds[0].question.lower()


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_decision_says_who_chose_it_and_what_changes_if_the_reviewer_disagrees(path):
    for d in decisions(load_spec(path), source_path=path):
        assert d.question.strip().endswith("?"), f"{d.element_id}: not phrased as a question"
        assert d.choice.strip(), f"{d.element_id}: no current choice stated"
        assert d.who.strip(), f"{d.element_id}: nobody is named as having chosen"
        assert d.if_you_disagree.strip(), f"{d.element_id}: no consequence stated"


# ------------------------------------------------------------------------------ sign-off
def test_signoff_records_reviewer_date_and_the_element_hash(tmp_path):
    spec = load_spec(SHB)
    rec = record_signoff(tmp_path, spec, "check.1", reviewer="A. Registrar",
                         source_path=SHB, note="agreed")
    assert rec["reviewer"] == "A. Registrar"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", rec["signed_at"])
    el = next(e for e in elements(spec, source_path=SHB) if e.element_id == "check.1")
    assert rec["element_hash"] == el.element_hash
    assert rec["spec_hash"] == spec.spec_hash
    assert load_signoffs(tmp_path, spec.spec_id) == [rec]


def test_editing_the_signed_text_invalidates_the_signoff(tmp_path):
    """The one property that makes a sign-off worth collecting."""
    original = SHB.read_text(encoding="utf-8")
    edited = tmp_path / "edited.yaml"
    spec = load_spec(SHB)
    signs = [record_signoff(tmp_path, spec, "rule.3", reviewer="A. Registrar", source_path=SHB)]
    assert signoff_status(next(e for e in elements(spec, source_path=SHB)
                               if e.element_id == "rule.3"), signs)[0] == SIGNED

    edited.write_text(original.replace(
        "Code behaviour 3 if any malignant invasion is present, no matter how limited.",
        "Code behaviour 3 only when the invasion is measured and reported."), encoding="utf-8")
    after = load_spec(edited)
    el = next(e for e in elements(after, source_path=edited) if e.element_id == "rule.3")
    assert signoff_status(el, signs)[0] == STALE
    doc = render_review(after, source_path=edited, signoffs=signs)
    assert "A. Registrar" in doc and "no longer" in doc.lower()


def test_an_unsigned_element_is_not_reported_as_confirmed(tmp_path):
    spec = load_spec(SHB)
    el = next(e for e in elements(spec, source_path=SHB) if e.element_id == "rule.1")
    assert signoff_status(el, [])[0] == UNSIGNED


def test_a_signoff_does_not_carry_across_specs(tmp_path):
    """Two specs share the sentence about coding the site of origin. A registrar approving
    it for one criterion has not approved it for the other."""
    a = load_spec(SHB)
    rec = record_signoff(tmp_path, a, "rule.1", reviewer="A. Registrar", source_path=SHB)
    b = load_spec(STAGE)
    assert all(signoff_status(e, [rec])[0] != SIGNED for e in elements(b, source_path=STAGE))


# ---------------------------------------------------------------------- HOW OFTEN IT FIRES
def test_a_measurement_is_withdrawn_when_the_thing_it_measured_has_changed(tmp_path):
    """The keyword miss rate was measured against five specific terms. Reprinting it beside a
    sixth term would be a fabricated number wearing a measured number's clothes."""
    doc = review(SHB)
    assert "31.7" in doc

    raw = yaml.safe_load(SHB.read_text(encoding="utf-8"))
    for s in raw["proof_obligation"]["for_negative"]["strata"]:
        if s.get("required_keywords"):
            s["required_keywords"] = s["required_keywords"] + ["cancer"]
    p = tmp_path / "changed.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    changed = review(p)
    section = changed.split(f"## {SECTION_TITLES[6]}")[1]
    assert "no longer describes" in section.lower() or "has since changed" in section.lower()


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_an_unmeasured_spec_says_so_rather_than_borrowing_a_number(path: Path):
    """Silence here reads as `fine`. Only one of the five criteria has ever been measured."""
    section = review(path).split(f"## {SECTION_TITLES[6]}")[1]
    assert section.strip(), f"{path.name}: section 7 is empty"
    assert "276,054" in section or "276054" in section, "the corpus it was measured on"


# -------------------------------------------------------------------------- nothing is dropped
@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_an_untranslatable_section_is_reported_rather_than_silently_dropped(path: Path):
    """A renderer that quietly ignores a key it does not understand tells the reviewer the
    spec contains nothing else. `date_imputation` and `keyword_field_coverage` are both
    extra keys no shipped model field declares."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc = review(path)
    for key in raw:
        assert key.replace("_", " ") in doc.replace("_", " ") or key in doc, \
            f"{path.name}: top-level key {key!r} is nowhere in the review document"


def test_a_spec_that_is_not_in_the_notes_at_all_says_so_first():
    """STORE.610 is not answerable from a chart. A reviewer who reads four sections of
    evidence rules before learning that has been wasted."""
    doc = review(COC)
    first = doc.split(f"## {SECTION_TITLES[1]}")[0].lower()
    assert "not" in first and ("registration" in first or "billing" in first)


# ------------------------------------------------------------------------------------ CLI
def test_cli_review_writes_the_document(tmp_path):
    out = tmp_path / "review.md"
    r = runner.invoke(app, ["spec", "review", str(SHB), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.read_text(encoding="utf-8").startswith("# ")


def test_cli_signoff_appends_and_shows_up_in_the_next_render(tmp_path):
    led = tmp_path / "signoffs"
    r = runner.invoke(app, ["spec", "signoff", "--spec", str(SHB), "--reviewer", "A. Registrar",
                            "--element", "check.1", "--signoffs", str(led)])
    assert r.exit_code == 0, r.output
    rows = [json.loads(x) for x in
            (led / "STORE.400_522_523.site_histology_behavior.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["reviewer"] == "A. Registrar"

    out = tmp_path / "review.md"
    r = runner.invoke(app, ["spec", "review", str(SHB), "--out", str(out),
                            "--signoffs", str(led)])
    assert r.exit_code == 0, r.output
    assert "A. Registrar" in out.read_text(encoding="utf-8")


def test_cli_signoff_refuses_an_element_that_does_not_exist(tmp_path):
    r = runner.invoke(app, ["spec", "signoff", "--spec", str(SHB), "--reviewer", "X",
                            "--element", "rule.999", "--signoffs", str(tmp_path)])
    assert r.exit_code != 0
    assert "rule.1" in r.output, "an error about an unknown id must show the ids that exist"
