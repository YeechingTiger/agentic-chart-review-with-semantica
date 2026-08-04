"""A prior is only pluggable if it reaches the prompt AND the manifest says which one did.

Four conditions make an axis a usable ablation, and a prior has failed three of them until now:

  (1) an entry point                — `--prior` on the run commands
  (2) more than one implementation  — any number of assets; `build_prior` makes them
  (3) the choice is RECORDED        — `prompt_asset_manifest.retrieval_prior`, so two arms can be
                                      told apart afterwards. Without this, "accuracy rose" and
                                      "the prompt changed" are the same observation.
  (4) nothing downstream defeats it — the prompt is assembled in ONE place and `experience_block`
                                      is called there. Wiring only `run_patient`'s signature would
                                      repeat the Site Mapping defect, where the mapping reached the
                                      ledger and `plan_from_spec` rebuilt the stratification
                                      without it one line later.

And one property that is specific to a measured prior: `spec_hash` MUST NOT MOVE. That is the whole
reason to deliver experience as an asset rather than through `assets adopt`, which writes keywords
into the contract — after which `analyze_arms.py` correctly refuses to compare the arm with its
baseline, because a changed contract is a changed question.
"""

from __future__ import annotations

import inspect
import json

import pytest

from acr.contract.retrieval_prior import RetrievalPrior, to_experience_asset
from acr.improvement.prior import build_prior

FIELD = "date_of_initial_diagnosis"
SPEC = "STORE.390.date_of_initial_diagnosis"


def _labels(tmp_path):
    rows = [
        {"patient_id": "P1", "note_id": "n1", "doc_type": "Path", "spec_id": SPEC,
         "admissibility": {"verdicts": {FIELD: "can_establish"}},
         "retrieval_terms": [{"term": "carcinoma", "reason": "names_the_section"}],
         "model": "m", "prompt_hash": "h"},
        {"patient_id": "P2", "note_id": "n2", "doc_type": "Imaging", "spec_id": SPEC,
         "admissibility": {"verdicts": {FIELD: "neither"}},
         "retrieval_terms": [{"term": "mass", "reason": "names_the_section"}],
         "model": "m", "prompt_hash": "h"},
        {"patient_id": "P2", "note_id": "n3", "doc_type": "Path", "spec_id": SPEC,
         "admissibility": {"verdicts": {FIELD: "can_establish"}},
         "retrieval_terms": [{"term": "carcinoma", "reason": "names_the_section"}],
         "model": "m", "prompt_hash": "h"},
    ]
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


@pytest.fixture
def prior_file(tmp_path):
    prior = build_prior(_labels(tmp_path), fields=[FIELD], min_patients=2,
                        asset_id="STORE.390.scan-of-two")
    p = tmp_path / "prior.json"
    p.write_text(json.dumps(prior.to_dict(), indent=1), encoding="utf-8")
    return p


# ---------------------------------------------------------------- (1) the entry point

def test_run_patient_takes_a_prior():
    from acr.review.agent import run_patient
    assert "retrieval_prior" in inspect.signature(run_patient).parameters


def test_every_run_command_exposes_the_flag():
    from acr.commands.cli_chart import batch, consistency, run
    from acr.commands.cli_pipeline import extract
    for fn in (run, batch, consistency, extract):
        assert "prior" in inspect.signature(fn).parameters, fn.__name__


def test_a_missing_prior_file_refuses_by_name(tmp_path):
    import typer

    from acr.commands.cli_chart import _load_prior
    with pytest.raises(typer.BadParameter, match="prior"):
        _load_prior(str(tmp_path / "nope.json"))


def test_no_flag_means_no_prior_and_no_claim():
    """The baseline arm must be byte-identical to a run with no prior concept at all."""
    from acr.commands.cli_chart import _load_prior
    assert _load_prior("") is None


# ---------------------------------------------------------------- (3) recorded in the manifest

def test_the_manifest_identifies_the_prior(prior_file):
    from acr.contract.spec import load_spec
    from acr.core import site
    from acr.review.run_manifest import prompt_asset_manifest
    spec = load_spec(site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml")
    prior = RetrievalPrior.load(prior_file)

    entry = prompt_asset_manifest(spec, retrieval_prior=prior)["retrieval_prior"]

    assert entry["asset_id"] == "STORE.390.scan-of-two"
    assert entry["content_hash"] == prior.content_hash
    assert entry["status"] == "measured"
    assert entry["n_patients"] == 2


def test_no_prior_records_an_explicit_absence(prior_file):
    """`None`, not a missing key: an absent entry and an entry saying "no prior" are different
    claims, and only one of them survives a reader asking which arm this was."""
    from acr.contract.spec import load_spec
    from acr.core import site
    from acr.review.run_manifest import prompt_asset_manifest
    spec = load_spec(site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml")
    m = prompt_asset_manifest(spec)
    assert "retrieval_prior" in m
    assert m["retrieval_prior"] is None


def test_two_priors_are_distinguishable_in_the_manifest(tmp_path, prior_file):
    from acr.contract.spec import load_spec
    from acr.core import site
    from acr.review.run_manifest import prompt_asset_manifest
    spec = load_spec(site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml")
    a = RetrievalPrior.load(prior_file)
    b = build_prior(_labels(tmp_path), fields=[FIELD], min_patients=2,
                    asset_id="STORE.390.scan-of-two", version="2")
    ha = prompt_asset_manifest(spec, retrieval_prior=a)["retrieval_prior"]["content_hash"]
    hb = prompt_asset_manifest(spec, retrieval_prior=b)["retrieval_prior"]["content_hash"]
    assert ha != hb, "two versions of one asset must not share a content hash"


# ---------------------------------------------------------------- (4) it reaches the prompt

def test_the_prior_is_rendered_into_the_system_prompt(prior_file):
    """Signature-level wiring is not enough — the Site Mapping reached the ledger and a second call
    site rebuilt the plan without it. This asserts the text the model would read."""
    from acr.review.document_concepts import experience_block
    prior = RetrievalPrior.load(prior_file)
    block = experience_block(to_experience_asset(prior))
    assert "RETRIEVAL EXPERIENCE" in block
    assert "carcinoma" in block
    assert "Path" in block

    import acr.review.agent as agent_mod
    src = inspect.getsource(agent_mod.run_patient)
    assert "experience_block" in src, (
        "run_patient does not call experience_block, so a --prior would be accepted and discarded")


def test_the_prompt_says_the_prior_is_not_a_rule(prior_file):
    """The header is load-bearing: `experience_block`'s own docstring says a prior whose measurement
    is not shown beside it reads exactly like a rule, and the model cannot weigh a rule."""
    from acr.review.document_concepts import experience_block
    block = experience_block(to_experience_asset(RetrievalPrior.load(prior_file)))
    assert "not a rule" in block
    assert "2 patient(s)" in block, "the measurement must travel with the numbers"


# ---------------------------------------------------------------- the comparability property

def test_supplying_a_prior_does_not_move_the_spec_hash(prior_file):
    """THE REASON THIS FEATURE EXISTS. `assets adopt` writes keywords into the contract, which moves
    `spec_hash`; `analyze_arms.py:192` then refuses the comparison ("refusing to compare: these arms
    ran against N different spec versions"). A prior delivered as an asset leaves the question
    unchanged, so the two arms compare."""
    from acr.contract.spec import load_spec
    from acr.core import site
    path = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"
    before = load_spec(path).spec_hash
    prior = RetrievalPrior.load(prior_file)
    assert prior.field_prior(FIELD) is not None
    assert load_spec(path).spec_hash == before


def test_a_run_can_tell_it_is_scored_on_a_subject_the_prior_saw(prior_file):
    """The held-out check. `analyze_arms.py` already refuses to fold a chart that informed a method
    into a headline; a prior measured on the chart being scored is the same contamination."""
    prior = RetrievalPrior.load(prior_file)
    assert prior.informed_by("P1")
    assert not prior.informed_by("SYN0001")
