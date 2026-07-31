"""The four causes of non-concordance must stay four, and A must never be guessed from B.

A care gap and a documentation gap look identical from inside a chart: in both, the record
does not show the recommended care. The only thing that separates them is a proof that the
absence is real rather than a retrieval failure, and that proof is the stratified coverage
ledger this project already computes. So the rule under test is narrow and absolute — no
gate-validated coverage proof, no choice between A and B — and it is enforced in code
because the case where a model most wants to pick one is exactly the case where it must not.

The other three cases these tests pin down, each a way the split quietly collapses back into
one number:

  * a gate PASS that proves nothing. `evaluate_gate` skips every stratum check when the
    stratum is absent, so a spec declaring no strata passes it vacuously and the answer still
    comes out `negative_basis: GATE_VALIDATED`. Reading that field alone files an unstratified
    abstention as a documentation-gap finding.
  * C eliminated by silence. Registry truth covers 1,788 of 8,894 patients; treating "no
    truth row" as "no disagreement" turns a coverage limitation into a clean bill of health
    for the extraction layer.
  * D eliminated by silence. Never having searched for a patient refusal is not evidence
    there was none, and counting a patient who declined treatment as a care gap is the
    failure the whole fourth cause exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acr.cli import app
from acr.corpus import Corpus
from acr.coverage import CoverageLedger, ForcedSampler, clopper_pearson_upper, evaluate_gate
from acr.evaluation.explain import (A_CARE_GAP, B_DOCUMENTATION_GAP, BOUND_BY_DIGEST, BOUND_BY_REFERENCE,
                         C_EXTRACTION_ERROR, CANNOT_DISTINGUISH, D_JUSTIFIED_EXCEPTION,
                         DEFAULT_MAX_ELUSION_UPPER, ELIMINATED, OPEN, SUPPORTED, UNBOUND,
                         ArtifactBindingError, ExplanationClaimError, VariableResult,
                         artifact_digest, assert_cause_is_earned, assess_coverage_proof,
                         mark_binding, resolve_bound_extract, scaffold_explanation)

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


# ---------------------------------------------------------------------------- builders
def _stratified_ledger(elusion: float = 0.11, *, complete: bool = True,
                       reviewed: int = 4) -> dict:
    """A ledger of the shape a real gate-validated stratified negative carries."""
    return {
        "mode": "stratified_exclusion",
        "sample_seed": 1234,
        "universe": {"n_documents": 293, "n_types": 17},
        "listed_documents": True,
        "searched_terms": ["chemotherapy", "carboplatin", "infusion"],
        "n_read": 31,
        "strata": [
            {"name": "can_establish", "N": 4, "reviewed": reviewed, "complete": complete,
             "elusion_upper": 0.0},
            {"name": "may_mention", "N": 120, "misses_sampled": 25, "miss_sample_hits": 0,
             "elusion_upper": elusion},
            {"name": "cannot_establish", "N": 169, "sampled": 25, "sample_hits": 0,
             "elusion_upper": elusion},
        ],
        "suspected_recognition_failures": [],
    }


def _absent(name: str = "adjuvant_chemotherapy", **over) -> VariableResult:
    ans = {"status": "EVIDENCE_INSUFFICIENT", "value": {name: None},
           "negative_basis": "GATE_VALIDATED", "coverage_attested": _stratified_ledger(),
           "evidence": []}
    ans.update(over)
    return VariableResult.from_answer(name, ans)


def _unproven(name: str = "adjuvant_chemotherapy") -> VariableResult:
    """The common real case: the agent stopped looking. Two of three shipped bases say so."""
    return _absent(name, negative_basis="BUDGET_EXHAUSTED", coverage_attested=None)


def _found(name: str = "first_course_treatment", value: dict | None = None,
           **over) -> VariableResult:
    ans = {"status": "FOUND", "value": value or {name: "surgery_alone"},
           "proof_basis": "WITNESS",
           "evidence": [{"note_id": "Op-Note_2022-09-06", "doc_type": "Op-Note",
                         "date": "2022-09-06", "quote": "lobectomy performed",
                         "supports": name, "stance": "supports"}]}
    ans.update(over)
    return VariableResult.from_answer(name, ans)


def _scaffold(driving, **over):
    kw = {"case_id": "PT1", "recommendation_id": "NCCN-NSCLC-ADJ-1",
          "concordance": "NON_CONCORDANT", "driving_variables": driving}
    kw.update(over)
    return scaffold_explanation(**kw)


# ------------------------------------------------------- the hard rule: A vs B needs proof
def test_without_a_coverage_proof_a_and_b_cannot_be_distinguished():
    s = _scaffold([_unproven()])
    assert s.verdict == CANNOT_DISTINGUISH
    assert s.standing(A_CARE_GAP) == OPEN
    assert s.standing(B_DOCUMENTATION_GAP) == OPEN


@pytest.mark.parametrize("cause", [A_CARE_GAP, B_DOCUMENTATION_GAP])
def test_cannot_distinguish_refuses_both_a_and_b(cause):
    """The enforcement, not just the label. A model that read the packet and picked anyway
    gets an error rather than a confident wrong owner."""
    s = _scaffold([_unproven()])
    with pytest.raises(ExplanationClaimError, match="same observation"):
        assert_cause_is_earned(s, cause)
    assert_cause_is_earned(s, CANNOT_DISTINGUISH)   # the only answer available


def test_a_proven_absence_makes_b_assertable():
    s = _scaffold([_absent()])
    assert s.standing(B_DOCUMENTATION_GAP) == SUPPORTED
    assert s.verdict != CANNOT_DISTINGUISH
    assert_cause_is_earned(s, B_DOCUMENTATION_GAP)


def test_a_proven_absence_still_does_not_prove_the_care_never_happened():
    """B is a finding about the chart. A is a finding about the world, and no ledger reaches
    it — care delivered outside the reporting facility looks exactly like this."""
    s = _scaffold([_absent()])
    assert s.standing(A_CARE_GAP) == OPEN
    assert any("outside the reporting facility" in b
               for c in s.causes if c.cause == A_CARE_GAP for b in c.because)


def test_a_documented_departure_eliminates_b_and_supports_a():
    """Nothing is missing from the chart; the chart records the wrong thing."""
    s = _scaffold([_found()])
    assert s.standing(B_DOCUMENTATION_GAP) == ELIMINATED
    assert s.standing(A_CARE_GAP) == SUPPORTED
    with pytest.raises(ExplanationClaimError, match="eliminated by the coverage ledger"):
        assert_cause_is_earned(s, B_DOCUMENTATION_GAP)


def test_a_is_never_eliminated():
    """Not an oversight. The only evidence that would rule out a care gap is evidence the
    care happened, and that would have scored CONCORDANT at L4 — so it can never appear in
    a case this module sees."""
    for driving in ([_found()], [_absent()], [_unproven()]):
        assert _scaffold(driving).standing(A_CARE_GAP) != ELIMINATED


def test_one_unproven_absence_among_several_still_blocks_the_choice():
    s = _scaffold([_absent("adjuvant_chemotherapy"), _unproven("radiotherapy")])
    assert s.verdict == CANNOT_DISTINGUISH


# ------------------------------------------------------- what makes a proof inadequate
def test_a_vacuous_gate_pass_is_not_a_documentation_proof():
    """Measured hazard, reproduced here against the live gate rather than asserted.

    `evaluate_gate` treats a missing stratum as a satisfied check, so a spec that declares no
    strata PASSES with nothing reviewed and nothing sampled. Run
    `aprime_SYN0002__20260726T035724Z` shipped exactly that: negative_basis GATE_VALIDATED,
    mode unstratified, strata []. It is a legitimate L2 abstention and it is not a proof.
    """
    docs, _ = Corpus(ROOT / "corpus" / "patients").chart("SYN0002").list_documents(limit=100_000)
    led = CoverageLedger(docs, [], ForcedSampler(1234))
    led.listed_documents = True
    assert evaluate_gate({}, led.stratum_results()).verdict == "PASS", "precondition"
    assert led.to_dict()["mode"] == "unstratified"

    proof = assess_coverage_proof(_absent(coverage_attested=led.to_dict()))
    assert proof.adequate is False
    assert any("unstratified" in m for m in proof.missing)
    assert _scaffold([_absent(coverage_attested=led.to_dict())]).verdict == CANNOT_DISTINGUISH


def test_an_incomplete_exhaustive_stratum_is_not_a_proof():
    v = _absent(coverage_attested=_stratified_ledger(complete=False, reviewed=1))
    proof = assess_coverage_proof(v)
    assert proof.adequate is False
    assert any("not exhaustively reviewed (1/4)" in m for m in proof.missing)


def test_the_elusion_bound_is_enforced_against_the_shipped_cap():
    """0.12 is what all three gated specs declare, and a 25-document zero-hit draw clears it
    while a 20-document one does not. The cap is achievable, not decorative."""
    assert clopper_pearson_upper(0, 25) < DEFAULT_MAX_ELUSION_UPPER
    assert clopper_pearson_upper(0, 20) > DEFAULT_MAX_ELUSION_UPPER

    assert assess_coverage_proof(
        _absent(coverage_attested=_stratified_ledger(clopper_pearson_upper(0, 25)))).adequate
    bad = assess_coverage_proof(
        _absent(coverage_attested=_stratified_ledger(clopper_pearson_upper(0, 20))))
    assert bad.adequate is False
    assert any("exceeds 0.12" in m for m in bad.missing)


def test_an_unsampled_remainder_is_unbounded_not_perfect():
    """`evaluate_gate` maxes elusion with default=0.0, reading "nothing sampled" as "nothing
    eludes". Fine as one check among several; not fine as the sole basis for a causal claim."""
    led = _stratified_ledger()
    led["strata"] = [s for s in led["strata"] if s["name"] == "can_establish"]
    proof = assess_coverage_proof(_absent(coverage_attested=led))
    assert proof.adequate is False and proof.worst_elusion_upper == 1.0


@pytest.mark.parametrize("basis", ["AGENT_GAVE_UP", "BUDGET_EXHAUSTED", None])
def test_a_negative_that_stopped_looking_never_proves_absence(basis):
    v = _absent(negative_basis=basis)
    assert assess_coverage_proof(v).adequate is False


# ------------------------------------------------------- C stays live without ground truth
def test_c_is_open_when_no_registry_truth_was_supplied():
    s = _scaffold([_found()])
    assert s.standing(C_EXTRACTION_ERROR) == OPEN
    assert_cause_is_earned(s, C_EXTRACTION_ERROR)


def test_c_is_open_for_the_variables_that_have_no_truth_row():
    """Truth for one variable must not launder the other. Registry coverage is 20% of the
    cohort, so this is the normal case, not the edge case."""
    s = _scaffold([_found("stage", {"stage": "IIIA"}), _found("histology", {"histology": "8070"})],
                  registry_truth={"stage": {"stage": "IIIA"}, "histology": None})
    c = next(x for x in s.causes if x.cause == C_EXTRACTION_ERROR)
    assert c.standing == OPEN and any("histology" in b for b in c.because)


def test_c_is_supported_when_the_registry_disagrees_but_is_not_called_proof():
    s = _scaffold([_found("histology", {"histology": "8046"})],
                  registry_truth={"histology": {"histology": "8070"}})
    c = next(x for x in s.causes if x.cause == C_EXTRACTION_ERROR)
    assert c.standing == SUPPORTED
    assert any("does NOT prove the extraction wrong" in b for b in c.because)


def test_c_is_eliminated_only_when_truth_exists_for_every_driving_variable():
    s = _scaffold([_found("histology", {"histology": "8070"})],
                  registry_truth={"histology": {"histology": "8070"}})
    assert s.standing(C_EXTRACTION_ERROR) == ELIMINATED
    with pytest.raises(ExplanationClaimError):
        assert_cause_is_earned(s, C_EXTRACTION_ERROR)


# ------------------------------------------------------- D needs its own evidence
def test_d_is_open_when_the_exception_catalogue_was_never_queried():
    s = _scaffold([_found()])
    d = next(x for x in s.causes if x.cause == D_JUSTIFIED_EXCEPTION)
    assert d.standing == OPEN and any("never queried" in b for b in d.because)


def test_a_witnessed_exception_supports_d_but_leaves_a_open_for_scope():
    """Whether a documented refusal covers THIS recommendation is judgement, and judgement
    does not survive being hard-coded. Code records the tension; the skill resolves it."""
    s = _scaffold([_found()], exception_results=[
        _found("patient_refusal", {"patient_refusal": "declined adjuvant chemotherapy"})])
    assert s.standing(D_JUSTIFIED_EXCEPTION) == SUPPORTED
    assert s.standing(A_CARE_GAP) != ELIMINATED
    assert any("if an adjudicator" in b for c in s.causes if c.cause == A_CARE_GAP
               for b in c.because)


def test_an_ungated_exception_cannot_carry_d():
    """Same standard as the primary variable: a FOUND that never passed the witness gate is
    not documentation of a refusal."""
    s = _scaffold([_found()], exception_results=[
        _found("patient_refusal", {"patient_refusal": "declined"}, proof_basis="UNGATED")])
    d = next(x for x in s.causes if x.cause == D_JUSTIFIED_EXCEPTION)
    assert d.standing == OPEN and any("never passed the witness gate" in b for b in d.because)


def test_d_is_eliminated_only_by_a_coverage_proof_of_its_own():
    s = _scaffold([_found()], exception_results=[_absent("patient_refusal"),
                                                 _absent("performance_status")])
    assert s.standing(D_JUSTIFIED_EXCEPTION) == ELIMINATED
    s2 = _scaffold([_found()], exception_results=[_absent("patient_refusal"),
                                                  _unproven("performance_status")])
    assert s2.standing(D_JUSTIFIED_EXCEPTION) == OPEN


# ------------------------------------------------------- shape, scope and honesty
def test_the_variable_is_absent_per_field_not_per_answer():
    """STORE.400_522_523 answers three fields at once. A rule about histology must not read
    a populated primary_site as presence."""
    ans = {"status": "EVIDENCE_INSUFFICIENT",
           "value": {"primary_site": "C186", "histology": None, "behavior": None},
           "negative_basis": "GATE_VALIDATED", "coverage_attested": _stratified_ledger()}
    assert VariableResult.from_answer("store400", ans, "histology").is_absent is True
    assert VariableResult.from_answer("store400", ans, "primary_site").is_absent is False


def test_explain_refuses_anything_that_is_not_non_concordant():
    """NOT_ASSESSABLE is a first-class outcome. Explaining it folds unscoreable cases into a
    rate they do not belong in."""
    for verdict in ("CONCORDANT", "NOT_ASSESSABLE"):
        with pytest.raises(ValueError, match="only on NON_CONCORDANT"):
            _scaffold([_found()], concordance=verdict)


def test_a_case_with_no_driving_variables_is_a_caller_bug_not_an_empty_answer():
    with pytest.raises(ValueError, match="no driving variables"):
        _scaffold([])


def test_the_packet_forbids_the_choice_the_scaffold_could_not_make():
    p = _scaffold([_unproven()]).packet
    assert any("Do NOT choose between A" in f for f in p["forbidden"])
    assert any("single non-concordance rate" in f for f in p["forbidden"])


def test_every_serialised_scaffold_says_it_is_unvalidated():
    """L5 has no ground truth in this project. The disclaimer travels with the output rather
    than living only in a docstring nobody downstream reads."""
    d = _scaffold([_absent()]).to_dict()
    assert "NO ground truth" in d["validation_status"]
    assert "NO ground truth" in d["packet"]["validation_status"]


def test_the_four_causes_keep_four_distinct_owners():
    owners = {c.cause: c.owner for c in _scaffold([_absent()]).causes}
    assert len(set(owners.values())) == 4
    assert owners[D_JUSTIFIED_EXCEPTION].startswith("nobody")


def test_render_names_a_standing_and_an_owner_for_every_cause():
    out = _scaffold([_unproven()]).render()
    assert out.splitlines()[0].endswith(CANNOT_DISTINGUISH)
    for cause in (A_CARE_GAP, B_DOCUMENTATION_GAP, C_EXTRACTION_ERROR, D_JUSTIFIED_EXCEPTION):
        assert cause in out


# ==================================================== the artifacts the verdict was made from
"""A four-cause verdict is a function of its inputs, so an input nobody pinned down is a
verdict anybody can choose.

Measured on this tree, `c1b5914-dirty`, entirely on synthetic fixtures: one concord.json, two
extract.json files, `acr explain --extract <the other one>` — the case went from
CANNOT_DISTINGUISH to OPEN_TO_ADJUDICATION with B_DOCUMENTATION_GAP SUPPORTED, exit code 0,
no warning printed, and the only trace in explain.json was `extract_input` pointing serenely
at the swapped file. That is an owner assigned to a records department on evidence chosen
after the fact.

The binding is by CONTENT, never by path. Two things force that. A path proves nothing about
what is at the end of it — `acr extract` overwrites extract.json in place, so the recorded
path can silently come to name a later artifact than the one L4 scored. And the legitimate
use of --extract is relocation, where the content is right and only the path is stale; a
filename check would reject exactly that case and accept the dangerous one.
"""
VAR = "adjuvant_systemic_therapy_class"
SPEC = "FAKE.tx"
STAMP = "2026-07-26T21:00:00+00:00"

GATED = {"status": "EVIDENCE_INSUFFICIENT", "value": {VAR: None},
         "negative_basis": "GATE_VALIDATED", "coverage_attested": _stratified_ledger(),
         "evidence": []}
GAVE_UP = {"status": "EVIDENCE_INSUFFICIENT", "value": {VAR: None},
           "negative_basis": "BUDGET_EXHAUSTED", "coverage_attested": None, "evidence": []}


def _extract_doc(answer: dict, *, created_utc: str = STAMP, pid: str = "P1") -> dict:
    """One patient, one spec, in the shape `acr extract` writes."""
    return {
        "schema": "acr.extract/1", "created_utc": created_utc, "code_sha": "test",
        "corpus": "corpus/patients", "specs_dir": "specs", "cohort": "c.csv",
        "model": "test", "sample_seed": 7, "n_failed_runs": 0,
        "specs": {SPEC: {"spec_id": SPEC}},
        "patients": [{"patient_id": pid, "runs": [], "errors": [],
                      "answers": {SPEC: answer},
                      "variables": {VAR: {"status": answer["status"], "value": None,
                                          "negative_basis": answer.get("negative_basis"),
                                          "source": SPEC, "spec_id": SPEC, "output_field": VAR,
                                          "gate_validated": True, "proof_basis": None}}}],
    }


def _concord_doc(extract_path: Path, *, created_utc: str = STAMP, pid: str = "P1") -> dict:
    return {
        "schema": "acr.concord/1", "created_utc": "2026-07-26T22:00:00+00:00",
        "code_sha": "test", "engine": "acr.concordance/deterministic",
        "guideline": {"path": "g.yaml", "guideline_id": "G", "guideline_version": "1",
                      "guideline_hash": "0" * 16},
        "guideline_binding_warnings": [],
        "extract_input": str(extract_path.resolve()), "extract_created_utc": created_utc,
        "extra_variables": None, "prefer": "error",
        "recommendations": {"REC-1": {"title": "t", "rule_inputs": [VAR],
                                      "exception_inputs": []}},
        "patients": [{"patient_id": pid, "summary": {}, "results": [
            {"recommendation_id": "REC-1", "outcome": "NON_CONCORDANT",
             "rule_applied": "satisfied_when", "reason": "absent",
             "inputs_used": [{"variable": VAR, "status": "EVIDENCE_INSUFFICIENT", "value": None,
                              "negative_basis": None, "source": SPEC}],
             "blocking_inputs": [], "exception_id": None, "notes": []}]}],
        "summary": {},
    }


@pytest.fixture
def chain(tmp_path: Path) -> Path:
    """concord.json + the extract it was computed from + a swapped extract that proves B."""
    honest, swapped = tmp_path / "extract.json", tmp_path / "other_extract.json"
    honest.write_text(json.dumps(_extract_doc(GAVE_UP)), encoding="utf-8")
    swapped.write_text(json.dumps(_extract_doc(GATED, created_utc="2026-07-27T09:00:00+00:00")),
                       encoding="utf-8")
    (tmp_path / "concord.json").write_text(json.dumps(_concord_doc(honest)), encoding="utf-8")
    return tmp_path


def _load(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _resolve(d: Path, **kw):
    return resolve_bound_extract(_load(d / "concord.json"), load=_load, **kw)


# ------------------------------------------------------------------ the digest itself
def test_a_digest_follows_the_content_not_the_name_or_the_formatting(tmp_path):
    """Two properties, and both are load-bearing. A copy under another name is the same
    artifact — that is what makes a relocated run bindable at all. A re-indented copy is also
    the same artifact, because every stage here re-serialises what it reads and a byte hash
    would call that tampering."""
    doc = _extract_doc(GAVE_UP)
    assert artifact_digest(doc) == artifact_digest(json.loads(json.dumps(doc, indent=4)))

    changed = _extract_doc(GATED)
    assert artifact_digest(changed) != artifact_digest(doc)
    # one counter, deep inside the ledger that decides B, is enough to move it
    nudged = json.loads(json.dumps(changed))
    nudged["patients"][0]["answers"][SPEC]["coverage_attested"]["strata"][0]["reviewed"] = 3
    assert artifact_digest(nudged) != artifact_digest(changed)


# ------------------------------------------------------------------ the defect
def test_a_swapped_extract_is_refused_by_content_hash(chain):
    """THE DEFECT. Same concord.json, a different extract, and the four-cause verdict moves.
    Reproduced end to end below; this is the unit that stops it."""
    with pytest.raises(ArtifactBindingError) as e:
        _resolve(chain, override_path=str(chain / "other_extract.json"))
    msg = str(e.value)
    assert artifact_digest(_load(chain / "extract.json")) in msg
    assert artifact_digest(_load(chain / "other_extract.json")) in msg
    assert "--allow-unbound-extract" in msg


def test_the_extract_concord_named_binds_and_the_run_records_the_digest(chain):
    ext, b = _resolve(chain)
    assert b.bound is True and b.basis == BOUND_BY_REFERENCE and b.overridden is False
    assert b.expected_digest == b.actual_digest == artifact_digest(_load(chain / "extract.json"))
    assert ext["patients"][0]["answers"][SPEC]["negative_basis"] == "BUDGET_EXHAUSTED"


def test_an_extract_overwritten_since_the_concord_run_is_refused(chain):
    """No --extract anywhere. `acr extract` writes extract.json into a run directory and a
    rerun overwrites it; the path in concord.json then names an artifact L4 never scored."""
    (chain / "extract.json").write_text(
        json.dumps(_extract_doc(GATED, created_utc="2026-07-27T09:00:00+00:00")),
        encoding="utf-8")
    with pytest.raises(ArtifactBindingError, match="no longer the extract"):
        _resolve(chain)
    _, b = _resolve(chain, allow_unbound=True)
    assert b.bound is False and b.basis == UNBOUND


def test_a_relocated_identical_copy_is_still_bound(chain, tmp_path):
    """The reason --extract survives at all: artifacts get copied off the cluster and the
    absolute path in concord.json dies. Identical content is the same evidence, so this run
    is bound — and a filename check would have rejected precisely this case."""
    moved = tmp_path / "archive" / "extract.json"
    moved.parent.mkdir()
    moved.write_text((chain / "extract.json").read_text(encoding="utf-8"), encoding="utf-8")
    _, b = _resolve(chain, override_path=str(moved))
    assert b.bound is True and b.overridden is True and b.relocated is True
    assert b.used_path == str(moved.resolve()) and b.recorded_path.endswith("extract.json")


def test_a_recorded_digest_beats_re_reading_the_file(chain, tmp_path):
    """Forward compatibility with the upstream half of this fix. When concord.json carries the
    digest of the extract it read, explain can bind a relocated artifact with the original
    gone — which re-reading the named file can never do."""
    doc = _load(chain / "concord.json")
    doc["extract_digest"] = artifact_digest(_load(chain / "extract.json"))
    moved = tmp_path / "archive.json"
    moved.write_text((chain / "extract.json").read_text(encoding="utf-8"), encoding="utf-8")
    (chain / "extract.json").unlink()
    _, b = resolve_bound_extract(doc, load=_load, override_path=str(moved))
    assert b.bound is True and b.basis == BOUND_BY_DIGEST


def test_an_extract_that_cannot_be_found_still_stops_the_run(chain):
    """Unchanged behaviour, kept because the reason is the sharpest one in the file: with no
    ledgers every case reports CANNOT_DISTINGUISH, and a fabricated one is indistinguishable
    from a real one."""
    (chain / "extract.json").unlink()
    with pytest.raises(ArtifactBindingError, match="falsely report CANNOT_DISTINGUISH"):
        _resolve(chain)


def test_a_concord_that_names_no_extract_is_unbindable_not_bound(chain):
    doc = _load(chain / "concord.json")
    doc.pop("extract_input")
    with pytest.raises(ArtifactBindingError, match="falsely report CANNOT_DISTINGUISH"):
        resolve_bound_extract(doc, load=_load)
    _, b = resolve_bound_extract(doc, load=_load, override_path=str(chain / "extract.json"),
                                 allow_unbound=True)
    assert b.bound is False
    assert any("does not name" in r for r in b.because)


def test_the_binding_names_patients_the_swapped_extract_does_not_carry(chain):
    """The quiet half of the same defect: `by_patient.get(pid, {})` falls back to the
    concord-recorded value with no ledger attached, so a patient missing from the overridden
    extract silently becomes CANNOT_DISTINGUISH rather than an error."""
    other = _extract_doc(GATED, pid="SOMEBODY-ELSE")
    p = chain / "third.json"
    p.write_text(json.dumps(other), encoding="utf-8")
    _, b = _resolve(chain, override_path=str(p), allow_unbound=True)
    assert b.absent_patients == ["P1"]


# ------------------------------------------------------------------ what an override produces
def test_an_allowed_override_brands_every_scaffold_it_produced(chain):
    """An override may not produce output a downstream reader can mistake for a bound one.
    The stamp goes on the run, on each scaffold, and inside the packet an adjudicating agent
    reads — because that agent never sees the top of the file."""
    _, b = _resolve(chain, override_path=str(chain / "other_extract.json"), allow_unbound=True)
    out = {"schema": "acr.evaluation.explain/1",
           "cases": [{"case_id": "P1", "scaffold": _scaffold([_absent()]).to_dict()},
                     {"case_id": "P2", "scaffold": None, "not_explainable": "x"}]}
    mark_binding(out, b)

    assert out["inputs_bound"] is False and out["extract_binding"]["bound"] is False
    sc = out["cases"][0]["scaffold"]
    assert sc["extract_binding"]["actual_digest"] == b.actual_digest
    assert sc["validation_status"].startswith("UNBOUND")
    assert "NO ground truth" in sc["validation_status"]        # the older warning survives
    assert sc["packet"]["validation_status"].startswith("UNBOUND")
    assert any("UNBOUND" in f for f in sc["packet"]["forbidden"])
    assert out["cases"][1]["scaffold"] is None                 # nothing to brand, no crash


def test_a_bound_run_is_never_branded_unbound(chain):
    _, b = _resolve(chain)
    out = {"cases": [{"case_id": "P1", "scaffold": _scaffold([_absent()]).to_dict()}]}
    mark_binding(out, b)
    sc = out["cases"][0]["scaffold"]
    assert out["inputs_bound"] is True
    assert sc["validation_status"].startswith("L5 has NO ground truth")
    assert not any("UNBOUND" in f for f in sc["packet"]["forbidden"])
    assert sc["extract_binding"]["basis"] == BOUND_BY_REFERENCE


# ------------------------------------------------------------------ through the real CLI
def test_cli_swapping_the_extract_no_longer_flips_the_verdict_in_silence(chain):
    """The reproduction, exactly as it was measured, now refused.

    Before: exit 0, B_DOCUMENTATION_GAP SUPPORTED, no warning. The bound run is asserted in
    the same test so the refusal cannot be passing because explain is broken outright."""
    r = runner.invoke(app, ["explain", "--input", str(chain / "concord.json")])
    assert r.exit_code == 0, r.output
    bound = json.loads((chain / "explain.json").read_text(encoding="utf-8"))
    assert bound["verdicts"] == {CANNOT_DISTINGUISH: 1}
    assert bound["inputs_bound"] is True

    r = runner.invoke(app, ["explain", "--input", str(chain / "concord.json"),
                            "--extract", str(chain / "other_extract.json"),
                            "--out", str(chain / "swapped.json")])
    assert r.exit_code == 2
    assert not (chain / "swapped.json").exists(), "a refused run must not leave an artifact"
    assert json.loads((chain / "explain.json").read_text(encoding="utf-8")) == bound


def test_cli_the_truth_file_that_moved_cause_c_is_named_in_the_output(chain, tmp_path):
    """The other input that moves a standing. `--truth` flips C from OPEN to SUPPORTED, and
    `registry_truth_supplied: true` was the entire audit trail — one bit, no file, no digest.
    It cannot be BOUND, because nothing upstream says which registry snapshot was meant; what
    it can be is attributable, so two runs that disagree about C can be told apart."""
    t = tmp_path / "truth.json"
    t.write_text(json.dumps({"P1": {VAR: "platinum_doublet"}}), encoding="utf-8")
    r = runner.invoke(app, ["explain", "--input", str(chain / "concord.json"),
                            "--truth", str(t)])
    assert r.exit_code == 0, r.output
    doc = json.loads((chain / "explain.json").read_text(encoding="utf-8"))
    assert doc["registry_truth_supplied"] is True                  # the old bit survives
    assert doc["registry_truth"]["digest"] == artifact_digest({"P1": {VAR: "platinum_doublet"}})
    assert doc["registry_truth"]["path"] == str(t.resolve())
    assert doc["registry_truth"]["bound"] is False                 # honest about what it is not

    r = runner.invoke(app, ["explain", "--input", str(chain / "concord.json")])
    doc = json.loads((chain / "explain.json").read_text(encoding="utf-8"))
    assert doc["registry_truth"] == {"supplied": False, "path": "", "digest": ""}


def test_cli_an_acknowledged_override_runs_but_brands_what_it_produced(chain):
    r = runner.invoke(app, ["explain", "--input", str(chain / "concord.json"),
                            "--extract", str(chain / "other_extract.json"),
                            "--allow-unbound-extract", "--out", str(chain / "swapped.json")])
    assert r.exit_code == 0, r.output
    assert "UNBOUND" in r.output
    doc = json.loads((chain / "swapped.json").read_text(encoding="utf-8"))
    # the verdict really does move — which is the whole reason the stamp has to travel with it
    assert doc["verdicts"] == {"OPEN_TO_ADJUDICATION": 1}
    assert doc["cases"][0]["scaffold"]["causes"][1]["standing"] == SUPPORTED
    assert doc["inputs_bound"] is False
    assert doc["extract_binding"]["expected_digest"] != doc["extract_binding"]["actual_digest"]
    assert doc["cases"][0]["scaffold"]["validation_status"].startswith("UNBOUND")
