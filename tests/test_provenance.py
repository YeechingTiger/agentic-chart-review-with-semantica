"""Provenance: which lines of a spec came from a standard, and which a model made up.

Every spec in `specs/` was written by a language model in one commit (e5229b0), from the
STORE manual text that happened to be in its context, and committed under a human author's
name. No registrar has read any of it. The only provenance field that existed was
`source_authority` — free text, read by nothing, and the stage spec's own note admits its
item numbers may be wrong. A transcribed rule and an invented one were byte-identical to a
reader: both were just YAML.

So the marking is per element and it is enforced, because an unenforced marking decays into
`source_authority` — present, plausible, and false. Three properties are asserted here, and
each is the kind that only fails loudly:

  1. An enforced element with no provenance record makes the spec UNLOADABLE. Not a warning.
     A warning is a thing nobody reads, and the failure mode is silent adoption of an
     invented rule.
  2. A run reports the WEAKEST status among the elements it used, and a run that leaned on a
     model-authored draft is not reportable as validated however clean its coverage gate is.
     The gate proves the search was done. It cannot prove the search terms were the right
     ones — measured on the real corpus, STORE.400's five required keywords miss the stated
     diagnosis for 31.7% of patients, and every one of those runs could still pass the gate.
  3. A signature does not survive an edit to the thing it signed. A clinician who approves
     a keyword list on Monday has not approved the term someone adds to it on Tuesday.

The tests that mutate a spec mutate a COPY loaded from the shipped file, never a fixture
written from scratch: a hand-built spec would drift from the real ones and stop testing them.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from acr.corpus import Corpus
from acr.graph import Budget, ChartReviewAgent
from acr.llm import LLMClient, LLMConfig, LLMResponse
from acr.spec import (
    ExtractionSpec,
    ProvenanceError,
    StaleProvenanceError,
    UnprovenancedElementError,
    enforced_elements,
    load_spec,
    weakest_status,
)

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
SHB = SPECS / "STORE.400_522_523.site_histology_behavior.yaml"
STAGE = SPECS / "STORE.700_880.stage.yaml"
ALL_SPECS = sorted(SPECS.glob("*.yaml")) + sorted((SPECS / "ablation").glob("*.yaml"))

#: The element every test that needs "a real enforced element" reaches for: the keyword list
#: the corpus measurement falsified. If this path ever changes the tests must be re-pointed
#: deliberately, which is the point of naming it once.
KEYWORDS = "proof_obligation.for_negative.strata[may_mention].required_keywords"


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict, name: str = "s.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def _record(data: dict, element: str) -> dict:
    for r in data["provenance"]:
        if r["element"] == element:
            return r
    raise AssertionError(f"{element} carries no provenance record in the shipped spec")


# Every shipped spec carries a real `provenance:` list as of 2026-07-27, so the tests below
# run against the actual files. There was briefly an autouse fixture here that skipped the
# whole module while `specs/*.yaml` had no provenance data; it is deliberately gone rather
# than left dormant. A skip that survives the gap it was written for is a skip nobody will
# ever notice again, and this module is the one that is supposed to notice.


# ------------------------------------------------------------------ 1. it must not load
@pytest.mark.parametrize("path", ALL_SPECS, ids=lambda p: p.stem)
def test_every_shipped_spec_provenances_every_enforced_element(path):
    """The whole tree, including the ablation arm. A spec that is exempt is a spec that lies."""
    spec = load_spec(path)
    declared = {r.element for r in spec.provenance}
    enforced = {e.path for e in enforced_elements(spec)}
    assert enforced, f"{path.name} enumerates no enforced elements — the enumerator is broken"
    assert enforced == declared, (
        f"unprovenanced: {sorted(enforced - declared)}; stale: {sorted(declared - enforced)}")


def test_an_unprovenanced_enforced_element_refuses_to_load(tmp_path):
    """Delete one record from a shipped spec and the spec stops being loadable.

    This is the property that makes the marking worth having. Before it, an element with no
    record read exactly like an element transcribed from the manual.
    """
    data = _raw(SHB)
    data["provenance"] = [r for r in data["provenance"] if r["element"] != KEYWORDS]
    with pytest.raises(UnprovenancedElementError) as e:
        load_spec(_write(tmp_path, data))
    assert KEYWORDS in str(e.value)
    # The message must name what READS the element; "add provenance" is not actionable
    # without knowing what the line does.
    assert "coverage" in str(e.value).lower()


def test_a_record_naming_no_enforced_element_is_refused(tmp_path):
    """A stale record is a live lie: it says `clinician_reviewed` about a rule that is gone."""
    data = _raw(SHB)
    ghost = dict(_record(data, KEYWORDS))
    ghost["element"] = "proof_obligation.for_negative.strata[deleted_stratum].required_keywords"
    data["provenance"].append(ghost)
    with pytest.raises(StaleProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert "deleted_stratum" in str(e.value)


def test_one_element_may_not_carry_two_records(tmp_path):
    data = _raw(SHB)
    data["provenance"].append(dict(_record(data, KEYWORDS)))
    with pytest.raises(ProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert KEYWORDS in str(e.value)


# ------------------------------------------------------------------ record honesty rules
def test_a_signature_needs_a_signer(tmp_path):
    data = _raw(SHB)
    rec = _record(data, KEYWORDS)
    rec["status"] = "clinician_reviewed"
    with pytest.raises(ProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert "reviewed_by" in str(e.value)


def test_a_corpus_derived_claim_must_name_its_run_and_its_n(tmp_path):
    """`corpus_derived` without the run that produced it is model_authored with a number in it."""
    data = _raw(SHB)
    rec = _record(data, KEYWORDS)
    rec["origin"] = "corpus_derived"
    rec["measured"] = {"pct": 31.7}
    with pytest.raises(ProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert "n_patients" in str(e.value) or "run" in str(e.value)


def test_a_manual_origin_must_name_the_section(tmp_path):
    """If you cannot name the section, it did not come from the manual."""
    data = _raw(SHB)
    rec = _record(data, KEYWORDS)
    rec["origin"] = "store_manual"
    rec["basis"] = "this is what the manual says about searching for pathology"
    with pytest.raises(ProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert "section" in str(e.value) or "item" in str(e.value)


def test_a_model_authored_basis_has_to_say_so(tmp_path):
    """The one phrase that cannot be read two ways, required verbatim."""
    data = _raw(SHB)
    rec = _record(data, KEYWORDS)
    rec["origin"] = "model_authored"
    rec["basis"] = "derived from standard registry practice"
    with pytest.raises(ProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert "no external source" in str(e.value)


def test_a_falsified_measurement_may_not_be_worn_as_a_status(tmp_path):
    """`measured` outranks `draft`, so a refuted element must not be allowed to claim it.

    STORE.400's keyword list is the case: it HAS been measured on 1,788 real charts, and the
    measurement is the reason not to trust it. Letting that count as `measured` would make
    the run summary read better for the elements we know most about.
    """
    data = _raw(SHB)
    rec = _record(data, KEYWORDS)
    rec["status"] = "measured"
    rec["measured"] = dict(rec.get("measured") or {}, verdict="falsified")
    with pytest.raises(ProvenanceError) as e:
        load_spec(_write(tmp_path, data))
    assert "falsified" in str(e.value)


# ------------------------------------------------------------------ 3. sign-off decay
def _sign(data: dict, element: str) -> dict:
    """Sign an element the way a registrar would: hash of the element as it stood."""
    spec = ExtractionSpec.model_validate(data)
    hashes = {e.path: e.hash for e in enforced_elements(spec)}
    rec = _record(data, element)
    rec.update(status="clinician_reviewed", reviewed_by="R. Registrar, CTR",
               reviewed_on="2026-07-26", element_hash_at_review=hashes[element],
               spec_hash_at_review=spec.provenance_free_hash)
    return rec


def test_a_signature_on_an_untouched_element_survives(tmp_path):
    """The control. Without it, the decay test below passes for a loader that voids every
    signature it sees, which would be useless in the other direction."""
    data = _raw(SHB)
    _sign(data, KEYWORDS)
    spec = load_spec(_write(tmp_path, data))
    rec = spec.provenance_index[KEYWORDS]
    assert rec.status == "clinician_reviewed" and not rec.sign_off_voided_by_edit
    assert rec.reviewed_by == "R. Registrar, CTR"


def test_editing_a_reviewed_element_voids_the_signature(tmp_path):
    """Sign the keyword list, then add a keyword. The signature must not survive.

    `cancer` is the term the corpus measurement says is missing — so this is the exact edit
    somebody will make, and making it must cost the sign-off rather than inherit it.
    """
    data = _raw(SHB)
    _sign(data, KEYWORDS)
    for s in data["proof_obligation"]["for_negative"]["strata"]:
        if s["name"] == "may_mention":
            s["required_keywords"] = list(s["required_keywords"]) + ["cancer"]
    spec = load_spec(_write(tmp_path, data))

    rec = spec.provenance_index[KEYWORDS]
    assert rec.status == "draft", "an edited element must fall back to draft"
    assert rec.sign_off_voided_by_edit is True
    # The reviewer's name stays. Deleting it would hide that a signature was voided at all.
    assert rec.reviewed_by == "R. Registrar, CTR"
    assert rec.element_hash != rec.element_hash_at_review


def test_a_signature_survives_an_edit_somewhere_else(tmp_path):
    """Per element, not per spec. A typo fix in a `description` three hundred lines away must
    not void a registrar's approval of the keyword list — a rule that cried wolf on every
    commit would be turned off within a week."""
    data = _raw(SHB)
    _sign(data, KEYWORDS)
    data["question"] = data["question"] + " (typo fixed)"
    data["fields"][0]["description"] = "ICD-O-3 topography of the site of ORIGIN (no decimal)"
    spec = load_spec(_write(tmp_path, data))
    assert spec.provenance_index[KEYWORDS].status == "clinician_reviewed"


# ------------------------------------------------------------------ 2. weakest wins
def test_weakest_status_is_the_minimum_not_the_mode():
    assert weakest_status(["clinician_reviewed", "measured", "draft"]) == "draft"
    assert weakest_status(["clinician_reviewed", "measured"]) == "measured"
    assert weakest_status(["clinician_reviewed"]) == "clinician_reviewed"
    # No elements used is not "everything is fine". It is "nothing was checked".
    assert weakest_status([]) == "draft"


def test_the_run_summary_reports_the_weakest_element_it_used(tmp_path):
    """One reviewed element must not launder the draft one beside it."""
    data = _raw(SHB)
    for element in (e.path for e in enforced_elements(ExtractionSpec.model_validate(data))):
        _sign(data, element)
    spec = load_spec(_write(tmp_path, data))
    summary = spec.provenance_for_run({"histology": "8140"}, "FOUND")
    assert summary["weakest_status"] == "clinician_reviewed"
    assert summary["reportable_as_validated"] is True

    data2 = _raw(SHB)
    for element in (e.path for e in enforced_elements(ExtractionSpec.model_validate(data2))):
        _sign(data2, element)
    _record(data2, KEYWORDS).update(status="draft", reviewed_by=None, reviewed_on=None,
                                    element_hash_at_review=None, spec_hash_at_review=None)
    spec2 = load_spec(_write(tmp_path, data2, "s2.yaml"))
    summary2 = spec2.provenance_for_run({"histology": "8140"}, "EVIDENCE_INSUFFICIENT")
    assert summary2["weakest_status"] == "draft"
    assert summary2["reportable_as_validated"] is False
    assert KEYWORDS in summary2["weakest_elements"]


def test_a_field_the_run_did_not_answer_is_not_counted_as_used():
    """`used` has to mean used. check_field_formats skips an empty field, so its allowable
    values were never applied and the run cannot be blamed — or credited — for them."""
    spec = load_spec(SHB)
    used = set(spec.provenance_for_run({"histology": "8140"}, "FOUND")["elements_used"])
    assert "fields[histology].format" in used
    assert "fields[behavior].allowable_values" not in used


def test_the_shipped_specs_claim_no_clinician_review():
    """A fact about the world, asserted so it cannot quietly stop being true.

    Nobody has reviewed any of this. When a registrar signs an element, this test changes in
    the same commit as the signature — which is the only moment anybody should be editing it.
    """
    for path in ALL_SPECS:
        for rec in load_spec(path).provenance:
            assert rec.status != "clinician_reviewed", f"{path.name}: {rec.element}"
            assert rec.origin != "clinician", f"{path.name}: {rec.element}"


# ------------------------------------------------------------------ 2. in the manifest
class _ScriptedLLM(LLMClient):
    """Fixed tool script, no provider, no cost. It cites a span out of a real search result
    because a made-up note_id would be refused by the toolbox and the run would never reach
    the gate — and the gate is what this test is about."""

    def __init__(self, value: dict):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.value = value

    def _reply(self, obj, calls=None):
        self.calls += 1
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=10, completion_tokens=5)

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if tools is None:
            if "SUFFICIENT|CONTINUE|STUCK" in last:      # reflect: REPLAN is no longer a verdict a model may pick
                return self._reply({"verdict": "CONTINUE", "reason": "still gathering"})
            if "FOUND|EVIDENCE_INSUFFICIENT|SPEC_INSUFFICIENT" in last:
                return self._reply({"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                    "reasoning": "never finalised"})
            return self._reply({"plan": [{"id": "1", "goal": "find the pathology report",
                                          "rationale": "it establishes the histology"}]})
        hits = next((json.loads(m["content"]).get("hits") or []
                     for m in reversed(messages)
                     if m.get("role") == "tool" and m.get("name") == "search_notes"), [])
        if not hits:
            return self._reply({}, [{"id": "c0", "name": "list_documents", "arguments": {}},
                                    {"id": "c1", "name": "search_notes",
                                     "arguments": {"query": "carcinoma"}}])
        h = hits[0]
        return self._reply({}, [
            {"id": "c2", "name": "record_evidence",
             "arguments": {"note_id": h["note_id"], "start": h["start"], "end": h["end"],
                           "supports": "histology"}},
            {"id": "c3", "name": "submit_answer",
             "arguments": {"status": "FOUND", "value": self.value,
                           "reasoning": "coded from the cited span"}}])


def test_a_gate_validated_run_is_still_not_reportable_as_validated(tmp_path):
    """The end-to-end claim. A clean gate on a model-authored spec buys a proven SEARCH, not
    a proven QUESTION, and the manifest has to be able to say which one it has."""
    spec = load_spec(SHB)
    llm = _ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
    agent = ChartReviewAgent(spec, llm, budget=Budget(max_steps=8), out_dir=tmp_path,
                             sample_seed=7)
    chart = Corpus(ROOT / "corpus" / "patients").chart("SYN0001")
    result = agent.run(chart, run_id="prov-test")

    manifest = json.loads((tmp_path / "prov-test.manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate_validated"] is True, "the run has to pass the gate for this to bite"
    prov = manifest["provenance"]
    assert prov["weakest_status"] == "draft"
    assert prov["reportable_as_validated"] is False
    assert prov["elements_used"], "a manifest that used no elements is a manifest of nothing"
    assert result["provenance"] == prov


def test_the_manifest_names_the_elements_that_dragged_the_status_down(tmp_path):
    """A verdict with no subject is unactionable: the reader has to know which line to fix."""
    spec = load_spec(SHB)
    llm = _ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
    agent = ChartReviewAgent(spec, llm, budget=Budget(max_steps=8), out_dir=tmp_path,
                             sample_seed=7)
    agent.run(Corpus(ROOT / "corpus" / "patients").chart("SYN0001"), run_id="prov-test-2")
    prov = json.loads((tmp_path / "prov-test-2.manifest.json").read_text(
        encoding="utf-8"))["provenance"]
    assert set(prov["weakest_elements"]) <= set(prov["elements_used"])
    assert prov["counts_by_origin"]["model_authored"] >= 1
    assert prov["spec_id"] == spec.spec_id


# ------------------------------------------------------------------ hashing
def test_provenance_free_hash_ignores_the_provenance_block_and_spec_hash_does_not(tmp_path):
    """Two hashes, because they answer different questions.

    `spec_hash` identifies the artifact a run was conducted under, and a run conducted under
    a spec whose keyword list is signed is not the same run as one where it is not — the
    provenance changes what may be claimed, so it belongs in the run's identity.
    `provenance_free_hash` is what a signature is taken over: it has to be recomputable by a
    reviewer later, and a hash that includes the signature can never be recomputed at all.
    """
    data = _raw(SHB)
    before = load_spec(SHB)
    _record(data, KEYWORDS)["basis"] += " (clarified)"
    after = load_spec(_write(tmp_path, data))
    assert after.provenance_free_hash == before.provenance_free_hash
    assert after.spec_hash != before.spec_hash
