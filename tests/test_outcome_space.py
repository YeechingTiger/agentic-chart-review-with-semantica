"""The outcome space belongs to the CONTRACT, not to the tool that submits into it.

Until 2026-08-02 the set of things a run was allowed to conclude was a literal list inside
`acr.review.tools.toolbox`:

    "status": {"type": "string", "enum": ["FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"]}

Two consequences, and both are mechanism faults rather than tuning:

  * A contract could not widen its own outcome space. STORE.390's `for_negative.statement`
    tells the model to answer with "no qualifying witness found, or the corpus itself
    insufficient" — a second, distinguishable abstention that the tool would not offer and no
    code path recognised. A contract instructing the model toward a status that does not
    exist is not a bad prompt; it is an unimplementable contract.

  * An UNDECLARED status was silently ACCEPTED. `gate_answer` branches on three literals and
    falls through to `{"accepted": True}`, so a run answering `TOTALLY_MADE_UP` cleared the
    gate having discharged no obligation at all — no evidence requirement, no coverage
    requirement, no spec-gap report. The most permissive outcome in the system was the one
    nobody had written down.

The fix is the same one this repo applies elsewhere: the contract declares, the code reads
the declaration. `result.status` names each outcome and its KIND, code branches on the kind,
and a status outside the declaration is refused. Adding an outcome is then an edit to a
contract with a provenance record attached, which is what it always was.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.contract import outcomes as O
from acr.contract.spec import ExtractionSpec, load_spec
from acr.core.state import EvidenceLedger
from acr.review.answer_gate import gate_answer
from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec
from acr.review.tools import Toolbox

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "patients"
SPEC_390 = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"


@pytest.fixture(scope="module")
def spec390():
    return load_spec(SPEC_390)


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0002")


def _ledgers(spec, chart):
    docs, _ = chart.list_documents(limit=100_000)
    return EvidenceLedger(), CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7))


def _mini(result_block: dict | None = None) -> ExtractionSpec:
    """A contract small enough that the outcome space is the only thing under test."""
    data: dict = {
        "spec_id": "TEST.outcomes",
        "question": "q?",
        "fields": [{"name": "v", "type": "string"}],
        "abstention": {"EVIDENCE_INSUFFICIENT": "nothing in the chart establishes it."},
    }
    if result_block is not None:
        data["result"] = result_block
    return ExtractionSpec.model_validate(data)


# --------------------------------------------------------------- the declaration itself

def test_a_contract_with_no_result_block_keeps_the_three_statuses_it_always_had():
    """Three of the four shipped contracts declare nothing. They must not change meaning.

    The default is in ONE place and is the thing a contract overrides, rather than a literal
    repeated at each site that used to hardcode it.
    """
    spec = _mini()
    assert O.submittable_statuses(spec) == ("FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT")
    assert O.status_kind(spec, "FOUND") == O.KIND_VALUE
    assert O.status_kind(spec, "EVIDENCE_INSUFFICIENT") == O.KIND_ABSTAIN_EVIDENCE
    assert O.status_kind(spec, "SPEC_INSUFFICIENT") == O.KIND_ABSTAIN_SPEC
    assert O.status_kind(spec, "TOTALLY_MADE_UP") is None


def test_a_declared_outcome_that_the_model_may_not_claim_is_still_part_of_the_space():
    """`TECHNICAL_FAILURE` is an outcome of a run and never an answer a model submits.

    Leaving it out of the contract would make the space look closed when it is not; putting
    it in the tool's enum would hand the model a give-up button. `submittable: false` says
    both things at once.
    """
    spec = _mini({"status": {
        "FOUND": {"kind": "value", "meaning": "a value"},
        "EVIDENCE_INSUFFICIENT": {"kind": "abstain_evidence", "meaning": "chart is silent"},
        "SPEC_INSUFFICIENT": {"kind": "abstain_spec", "meaning": "contract is silent"},
        "TECHNICAL_FAILURE": {"kind": "failure", "meaning": "the run died", "submittable": False},
    }})
    assert "TECHNICAL_FAILURE" in O.declared_statuses(spec)
    assert "TECHNICAL_FAILURE" not in O.submittable_statuses(spec)
    assert O.status_kind(spec, "TECHNICAL_FAILURE") == O.KIND_FAILURE


def test_a_kind_nothing_implements_is_refused_at_declaration_time():
    """A contract may not invent a kind: the branches are in code and there are four.

    The same rule as `ANSWER_CHECK_KINDS` — a declared behaviour nothing implements is
    indistinguishable from one that ran and found nothing.
    """
    spec = _mini({"status": {"FOUND": {"kind": "vlaue", "meaning": "typo"}}})
    with pytest.raises(ValueError, match="vlaue"):
        O.declared_statuses(spec)


# --------------------------------------------------------------- the tool surface

def test_the_status_enum_offered_to_the_model_is_built_from_the_contract(chart):
    spec = _mini({"status": {
        "FOUND": {"kind": "value", "meaning": "a value"},
        "EVIDENCE_INSUFFICIENT": {"kind": "abstain_evidence", "meaning": "chart is silent"},
        "CORPUS_INSUFFICIENT": {"kind": "abstain_evidence", "meaning": "the corpus is thin"},
        "SPEC_INSUFFICIENT": {"kind": "abstain_spec", "meaning": "contract is silent"},
    }})
    tb = Toolbox(chart, EvidenceLedger(), CoverageLedger(list(chart._docs.values()), []),
                 spec=spec)
    submit = next(s for s in tb.schemas() if s["function"]["name"] == "submit_answer")
    enum = submit["function"]["parameters"]["properties"]["status"]["enum"]
    assert enum == ["FOUND", "EVIDENCE_INSUFFICIENT", "CORPUS_INSUFFICIENT", "SPEC_INSUFFICIENT"]
    # And the prose the model reads has to agree with the enum, or the description becomes a
    # second, stale declaration of the same thing.
    assert "CORPUS_INSUFFICIENT" in submit["function"]["description"]


def test_every_runtime_binds_the_contract_to_the_toolbox_it_builds():
    """The omission this guards against is silent, which is the only reason it needs a test.

    `Toolbox(spec=None)` is legitimate — `mcp_server` builds a scratch toolbox before a spec
    is chosen — so forgetting `spec=` on the RUN path raises nothing. It falls back to the
    default three, and the symptom is a contract whose new status the model never uses, which
    reads as the model declining to use it.
    """
    import inspect

    import acr.review.agent as A
    import acr.review.mcp_server as M

    run = inspect.getsource(A)
    i = run.index("toolbox = Toolbox(")
    assert "spec=spec" in run[i:i + 300], "acr.review.agent builds its toolbox without the contract"
    mcp = inspect.getsource(M)
    j = mcp.index("toolbox = Toolbox(")
    assert "spec=spec" in mcp[j:j + 300], "mcp_server builds its run toolbox without the contract"


def test_the_answer_records_what_its_status_meant():
    """The eval plane reads manifests across contracts, months later, with no contract in hand.

    `CORPUS_INSUFFICIENT` tells such a reader nothing about whether a value was coded. The
    kind is resolved once, at emission, where the contract is present.
    """
    import inspect

    import acr.review.agent as A
    assert 'answer["status_kind"] = status_kind(' in inspect.getsource(A)

    from acr.evaluation.evals import RunRecord
    coded = RunRecord({"answer": {"status": "FOUND", "status_kind": "value"}})
    absent = RunRecord({"answer": {"status": "CORPUS_INSUFFICIENT",
                                   "status_kind": "abstain_evidence"}})
    assert coded.abstained is False
    assert absent.abstained is True
    # A manifest written before the kind was recorded still reads correctly.
    old = RunRecord({"answer": {"status": "EVIDENCE_INSUFFICIENT"}})
    assert old.abstained is True


def test_a_toolbox_built_without_a_spec_still_offers_the_default_three(chart):
    """`mcp_server` builds a scratch toolbox before any spec is chosen."""
    tb = Toolbox(chart, EvidenceLedger(), CoverageLedger(list(chart._docs.values()), []))
    submit = next(s for s in tb.schemas() if s["function"]["name"] == "submit_answer")
    assert submit["function"]["parameters"]["properties"]["status"]["enum"] == [
        "FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"]


# --------------------------------------------------------------- the gate

def test_an_undeclared_status_is_refused_rather_than_accepted_unchecked(spec390, chart):
    """The permissive fall-through, closed.

    Before this test the same submission was accepted with `missing: []`, having proved
    nothing: not the evidence a value owes, not the coverage an absence owes, not the section
    a spec complaint owes.
    """
    ev, cov = _ledgers(spec390, chart)
    v = gate_answer(spec390, {"status": "TOTALLY_MADE_UP", "value": {}, "reasoning": "because"},
                    evidence=ev, coverage=cov, chart=chart)
    assert v["accepted"] is False
    assert "TOTALLY_MADE_UP" in " ".join(v["missing"])
    # The refusal has to name the space, or the agent cannot recover from it inside its loop.
    assert "FOUND" in " ".join(v["missing"])


def test_a_second_evidence_abstention_inherits_the_first_ones_obligations(spec390, chart):
    """`CORPUS_INSUFFICIENT` is an abstention ABOUT THIS CHART, so it owes the same proof.

    This is the whole reason the contract declares a KIND rather than just a name. If the
    gate went on branching by literal, every status a contract added would arrive with zero
    obligations — a wider outcome space that is also a way around the gate.
    """
    assert O.status_kind(spec390, "CORPUS_INSUFFICIENT") == O.KIND_ABSTAIN_EVIDENCE
    ev, cov = _ledgers(spec390, chart)
    assert not cov.listed_documents, "precondition: this run never listed the chart"
    v = gate_answer(spec390, {"status": "CORPUS_INSUFFICIENT", "value": {},
                              "reasoning": "the record starts after the diagnosis"},
                    evidence=ev, coverage=cov, chart=chart, runtime_profile="guideline-only")
    assert v["accepted"] is False, v
    assert any("list" in m.lower() for m in v["missing"]), v["missing"]


def test_the_value_carrying_status_is_still_the_one_that_owes_evidence(spec390, chart):
    ev, cov = _ledgers(spec390, chart)
    v = gate_answer(spec390, {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20180101"},
                              "reasoning": "r"}, evidence=ev, coverage=cov, chart=chart)
    assert v["accepted"] is False
    assert "no evidence recorded" in v["why"]


# --------------------------------------------------------------- provenance and lint

def test_the_outcome_space_is_an_enforced_element(spec390):
    """It changes what the tool offers and what the gate accepts, so it needs a record.

    `enforced_elements` is the list of lines some code path changes behaviour on. An outcome
    space that could be widened without a provenance record would be the one invented rule in
    the file with no marking — and the marking is what tells a reader nobody has checked it.
    """
    from acr.contract.spec import enforced_elements
    paths = {e.path for e in enforced_elements(spec390)}
    assert "result.status" in paths
    assert "result.status" in spec390.provenance_index


def test_a_result_block_with_no_provenance_record_does_not_load(tmp_path):
    from acr.contract.spec import UnprovenancedElementError
    body = SPEC_390.read_text(encoding="utf-8")
    stripped = "\n".join(
        ln for ln in body.splitlines() if 'element: "result.status"' not in ln)
    p = tmp_path / "s.yaml"
    p.write_text(stripped, encoding="utf-8")
    with pytest.raises(UnprovenancedElementError):
        load_spec(p)


def test_speclint_refuses_an_abstention_the_outcome_space_does_not_declare():
    """Two blocks describing one thing must agree, and only a check makes them.

    `abstention` is the wording the model reads; `result.status` is what the tool offers and
    the gate accepts. A key in one and not the other is either an instruction the model
    cannot follow or an outcome nobody explained.
    """
    from acr.authoring.speclint import lint_spec
    spec = _mini({"status": {
        "FOUND": {"kind": "value", "meaning": "a value"},
        "SPEC_INSUFFICIENT": {"kind": "abstain_spec", "meaning": "contract is silent"},
    }})
    # `abstention` declares EVIDENCE_INSUFFICIENT; `result.status` does not.
    fails = [f for f in lint_spec(spec) if f.severity == "FAIL" and "EVIDENCE_INSUFFICIENT" in f.message]
    assert fails, [f"{f.check} {f.message}" for f in lint_spec(spec)]


def test_the_shipped_contract_lints_clean(spec390):
    from acr.authoring.speclint import lint_spec
    fails = [f"{f.check} {f.where}: {f.message}" for f in lint_spec(spec390) if f.severity == "FAIL"]
    assert fails == []
