"""The control the agent has to beat, and the reason it has to be scored by the same scorer.

`tools/measure_agency.py` answers *"was the answer-bearing document reachable by a contract-word
query"* and produces no answer, so it cannot be scored against ground truth. Until now nothing in
this tree extracted a value without the agent loop, which means the central question of the whole
project — does the loop earn its cost — had no control arm at all.

## The property that makes the comparison mean anything

`evals.RunRecord` and `evals.score` must read a query-only manifest with **no special case**. If this
arm needed its own scorer, "the query-only arm lost" and "the query-only arm was scored differently"
would be the same observation, and no amount of care in the runner would separate them. Every
assertion below about scoring therefore goes through the real `evals.score`, not through my reading
of it.

## And the property that keeps the arms apart

`experiment_config_hash` is the read side's discriminator since `evals.BaselineKey` started carrying
it. A query-only arm and an agent arm on the same spec, model and ceiling must not hash alike — so
`runner` and the term list are both inputs to it.
"""

from __future__ import annotations

import json

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.review.query_only import RUNNER, hit_set, run_query_only

SPEC = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"
FIELD = "date_of_initial_diagnosis"
TERMS = ["carcinoma", "biopsy", "diagnos"]


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


@pytest.fixture
def corpus():
    from acr.chartstore.corpus import Corpus
    return Corpus(site.corpus_root())


def _model(value="20190312", status="FOUND"):
    """A one-call scripted provider. `ToolScript` returns `submit_answer` as soon as its script is
    exhausted, which for an empty script is the first call."""
    pytest.importorskip("deepagents")
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from hooks_harness import ToolScript
    m = ToolScript(script=[], submit={"status": status,
                                      "value": {FIELD: value} if value else {},
                                      "reasoning": "from the hits"})
    m.seen = []
    return m


# ------------------------------------------------------------------ the retrieval half

def test_the_hit_set_comes_from_the_corpus_own_matcher(corpus):
    """Not a substring test written here. `chart.search` folds separators, quote widths and date
    forms, so `adeno-carcinoma` and `adeno carcinoma` are one term to an agent run — and a control
    priced against a different matcher is a control for a search nobody performs."""
    chart = corpus.chart("SYN0001")
    ids = hit_set(chart, TERMS)
    assert ids and len(ids) < len(chart) / 10, "a keyword list selects a small slice, not the chart"
    assert ids == sorted(set(ids)), "deduplicated and ordered, so two runs read the same corpus"


def test_a_term_matching_nothing_contributes_nothing(corpus):
    assert hit_set(corpus.chart("SYN0001"), ["zzzznotawordzzz"]) == []


def test_no_terms_is_an_arm_and_not_an_error(corpus, spec, tmp_path):
    """The floor under the floor: an extractor with no terms retrieves nothing. It must run and
    abstain, because "we gave it nothing" is a measurement."""
    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                       model=_model(value=None, status="EVIDENCE_INSUFFICIENT"), terms=[])
    assert m["query"]["n_hits"] == 0
    assert m["answer"]["status"] == "EVIDENCE_INSUFFICIENT"


# ------------------------------------------------------------------ one call, and it says so

def test_exactly_one_model_call(corpus, spec, tmp_path):
    model = _model()
    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                       model=model, terms=TERMS)
    assert m["usage"]["llm_calls"] == 1 and m["n_model_calls"] == 1
    assert len(model.seen) == 1, "the provider was invoked more than once"


def test_the_model_is_shown_the_hits_and_told_there_is_nothing_else(corpus, spec, tmp_path):
    """The arm's claim is that it saw every hit and had no other move. Both halves are in the
    prompt, and a reader of the manifest can check the first against `query.note_ids`."""
    model = _model()
    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                       model=model, terms=TERMS)
    turn = model.seen[0]
    system = "\n".join(str(x.content) for x in turn if getattr(x, "type", None) == "system")
    user = "\n".join(str(x.content) for x in turn if getattr(x, "type", None) == "human")
    assert "no tool to call" in system
    assert f"DOCUMENTS RETRIEVED: {m['query']['n_hits']}" in system
    for nid in m["query"]["note_ids"]:
        assert nid in user, f"{nid} was counted as a hit and not shown to the model"


def test_only_submit_answer_is_bound(corpus, spec, tmp_path):
    """No retrieval tools. If one were bound the arm would be an agent with a call limit of 1."""
    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                       model=_model(), terms=TERMS)
    assert m["prompt_assets"]["tool_surface"]["names"] == ["submit_answer"]


# ------------------------------------------------------------------ the same scorer, no special case

def test_the_manifest_scores_through_the_real_scorer(corpus, spec, tmp_path):
    from acr.evaluation import evals as E
    run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                   model=_model("20190312"), terms=TERMS)
    (rec,) = [E.RunRecord.from_manifest(p) for p in sorted(tmp_path.glob("*.manifest.json"))]
    assert rec.patient_id == "SYN0001" and rec.spec_id == spec.spec_id
    assert rec.abstained is False and rec.value == {FIELD: "20190312"}
    assert rec.total_tokens is not None, "a scored arm must report tokens"

    key = {f"SYN0001__{spec.spec_id}": {"fields": {FIELD: "20190312"}, "spec_id": spec.spec_id}}
    rep = E.score([rec], key, fields=[FIELD], key=E.derive_baseline_key([rec]))
    assert rep.per_instance[0].outcomes[0].outcome == E.EXACT
    assert rep.totals["n_unkeyed"] == 0


def test_an_abstention_is_read_as_an_abstention(corpus, spec, tmp_path):
    """`RunRecord.abstained` is `status_kind != "value"`, so the runner has to resolve the kind from
    the CONTRACT. Getting this wrong scores a correct abstention as a wrong answer."""
    from acr.evaluation import evals as E
    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0002", out_dir=tmp_path,
                       model=_model(value=None, status="EVIDENCE_INSUFFICIENT"), terms=TERMS)
    assert m["answer"]["status_kind"] == "abstain_evidence"
    (rec,) = [E.RunRecord.from_manifest(p) for p in sorted(tmp_path.glob("*.manifest.json"))]
    assert rec.abstained is True

    key = {"SYN0002": {"fields": {FIELD: None}, "spec_id": spec.spec_id}}
    rep = E.score([rec], key, fields=[FIELD], key=E.derive_baseline_key([rec]))
    assert rep.per_instance[0].outcomes[0].outcome == E.ABSTAINED_CORRECT


def test_a_model_that_never_submits_is_a_recorded_outcome_not_a_crash(corpus, spec, tmp_path):
    """One call means one chance. Declining it is a real result and must land in the manifest."""

    class _Silent:
        model_name = "silent"

        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages):
            class R:
                tool_calls = ()
                usage_metadata = None
            return R()

    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                       model=_Silent(), terms=TERMS)
    assert m["termination_reason"] == "STOPPED_WITHOUT_ANSWER"
    assert m["degradation"]["model_call_limit_without_answer"] is True
    assert m["answer"]["status_kind"] == "undeclared", (
        "a status the contract does not declare must never be read as value-carrying")


# ------------------------------------------------------------------ it is its own arm

def test_the_arm_hash_cannot_collide_with_an_agent_arm(corpus, spec, tmp_path):
    """`runner` is an input to the hash. Without it, an agent arm sharing spec, model and ceiling
    could hash identically to this one and `eval compare` would call them one configuration."""
    a = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path / "a",
                       model=_model(), terms=TERMS)
    assert a["runtime"] == RUNNER
    src = json.dumps(a["experiment_config_hash"])
    assert src

    import inspect

    from acr.review import query_only
    body = inspect.getsource(query_only.run_query_only)
    assert '"runner": RUNNER' in body and '"terms":' in body


def test_two_term_lists_are_two_arms(corpus, spec, tmp_path):
    """The terms ARE the intervention, so they must move the hash and be recorded in full."""
    a = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path / "a",
                       model=_model(), terms=TERMS)
    b = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path / "b",
                       model=_model(), terms=[*TERMS, "cytology"])
    assert a["experiment_config_hash"] != b["experiment_config_hash"]
    assert a["query"]["terms"] != b["query"]["terms"]


def test_two_runs_of_one_query_arm_share_a_hash(corpus, spec, tmp_path):
    """The property everything rests on: same arm, same hash. A per-run id here would make
    `eval compare` refuse every real pairing as a mixture."""
    a = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path / "a",
                       model=_model(), terms=TERMS)
    b = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0002", out_dir=tmp_path / "b",
                       model=_model(), terms=TERMS)
    assert a["experiment_config_hash"] == b["experiment_config_hash"]


def test_the_arm_records_what_it_cost_to_look(corpus, spec, tmp_path):
    """The cost comparison is half the result, and the retrieval side of it is the hit fraction: an
    arm that read 1% of the chart and one that read 60% are not the same control."""
    m = run_query_only(spec=spec, corpus=corpus, patient_id="SYN0001", out_dir=tmp_path,
                       model=_model(), terms=TERMS)
    q = m["query"]
    assert 0 < q["hit_fraction"] < 1
    assert q["n_chars_read"] > 0 and q["n_documents_in_chart"] > q["n_hits"]
