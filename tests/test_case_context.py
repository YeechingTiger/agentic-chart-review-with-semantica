"""What a run is told about the case, as opposed to what the contract says or what it reads.

Three things were being carried implicitly and none of them had anywhere to live.

WHICH ENTITY THE QUESTION IS ABOUT. STORE.390 asks "on what date was THIS TUMOUR first
diagnosed" and nothing anywhere says which tumour. That is not a retrieval problem: no amount
of reading resolves an unresolved referent. Measured consequence, recorded in the
`submit_answer` schema: three runs answered about the wrong neoplasm — one coded a sigmoid
colon hyperplastic polyp in a lung-cancer chart, two picked the wrong lung lesion in a chart
documenting an upper-lobe and a middle-lobe mass — and in none of those cases could the trace
tell "did not notice the other lesion" from "noticed it and judged it not the primary" from
"the one-row-per-patient gold cannot express two tumours".

WHEN THE RECORD WAS CUT. A diagnosis date after the last document in the chart is impossible:
a document cannot report a diagnosis that has not happened. The `20999999` two E4 runs
submitted has year 2099 and is arithmetically fine — nothing at the field layer knows what
year it is, so nothing could refuse it. The bound is a fact about the CASE, not about the
contract, which is why it needed this layer before it could be checked.

WHETHER A TIME WINDOW MEANS ANYTHING HERE. A window anchored on the diagnosis date, for the
task of finding the diagnosis date, is circular — and would cut off exactly the earlier
clinical impression SYNX05 is built around. So the contract declares whether it is
time-anchorable, and supplying a window to one that is not RAISES rather than being quietly
dropped: a window nobody honours and nobody reports is indistinguishable from one that was
applied.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.contract import case_requirements as CR
from acr.contract.spec import ExtractionSpec, load_spec
from acr.core.case_context import CaseContext, WindowNotAnchorableError
from acr.core.state import EvidenceLedger
from acr.review.answer_gate import gate_answer
from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "patients"
SPEC_390 = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"


@pytest.fixture(scope="module")
def spec390():
    return load_spec(SPEC_390)


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0002")


def _needs_entity() -> ExtractionSpec:
    return ExtractionSpec.model_validate({
        "spec_id": "TEST.entity", "question": "what stage is THIS tumour?",
        "fields": [{"name": "v", "type": "string"}],
        "case_context": {"requires_target_entity": True, "time_anchorable": True},
    })


# --------------------------------------------------------------- the dataclass

def test_a_case_context_is_plain_data_and_knows_nothing_about_contracts():
    """It sits in `core`, which `tests/test_layering.py` forbids from importing a plane.

    Deliberate: the same three facts are needed by the runtime that answers, the evaluator
    that scores and the diagnosis that explains, and a shared type is the only way those three
    can talk about one case without importing each other.
    """
    import acr.core.case_context as M
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert "from ..contract" not in src and "from ..review" not in src


def test_a_window_on_a_contract_that_cannot_be_anchored_raises():
    c = CaseContext(patient_id="P1", anchor_date=date(2018, 1, 1), window_days=(180, 180))
    c.honour_window(time_anchorable=True)          # fine
    with pytest.raises(WindowNotAnchorableError, match="time_anchorable"):
        c.honour_window(time_anchorable=False)


def test_no_window_is_not_an_error_on_either_kind():
    c = CaseContext(patient_id="P1")
    assert c.honour_window(time_anchorable=False) is None
    assert c.honour_window(time_anchorable=True) is None


# --------------------------------------------------------------- what the contract declares

def test_a_contract_that_declares_nothing_requires_nothing(spec390):
    plain = ExtractionSpec.model_validate({"spec_id": "T", "question": "q?"})
    assert CR.requires_target_entity(plain) is False
    assert CR.is_time_anchorable(plain) is True, (
        "silence must not disable a window — that would make an undeclared contract quietly "
        "stricter than a declared one")


def test_the_date_contract_declares_itself_not_time_anchorable(spec390):
    """Anchoring the search window on the diagnosis date, to find the diagnosis date, is
    circular. SYNX05 is built on an earlier clinical impression that any such window cuts off."""
    assert CR.is_time_anchorable(spec390) is False


# --------------------------------------------------------------- the pre-read refusal

def test_a_missing_target_entity_ends_the_run_before_it_reads_anything():
    spec = _needs_entity()
    refusal = CR.refuse_before_reading(spec, CaseContext(patient_id="P1"))
    assert refusal is not None
    assert refusal["status"] == "TARGET_ENTITY_UNCLEAR"
    assert refusal["value"] == {}
    assert "target_entity" in refusal["reasoning"]


def test_supplying_the_entity_lets_the_run_proceed():
    spec = _needs_entity()
    assert CR.refuse_before_reading(
        spec, CaseContext(patient_id="P1", target_entity="right upper lobe mass")) is None


def test_the_shipped_date_contract_does_not_refuse_todays_runs(spec390, chart):
    """It SHOULD require one — its question says "this tumour" and nothing resolves it.

    It does not, and the reason is recorded in the contract rather than left to be inferred:
    nothing in this corpus supplies a target entity per patient, so turning the switch on would
    abstain every chart without making a single answer more correct. The weaker treatment
    already in place is `lesions_considered` / `reported_lesion` — enumerate rather than
    refuse — and that is what the switch is measured against when a source of entities exists.
    """
    assert CR.requires_target_entity(spec390) is False
    assert CR.refuse_before_reading(spec390, CaseContext(patient_id="SYN0002")) is None


def test_the_runtime_asks_before_it_builds_anything():
    """The refusal is worthless if it happens after the search it exists to skip."""
    import inspect

    import acr.review.agent as A
    src = inspect.getsource(A.run_patient)
    i_refuse = src.index("refuse_before_reading")
    i_tools = src.index("toolbox = Toolbox(")
    assert i_refuse < i_tools, "the case check runs after the run has already been assembled"


# --------------------------------------------------------------- the impossible-date bound

def _gate(spec, chart, submitted, **kw):
    """One real citation, because "no evidence recorded" refuses before anything else does."""
    from acr.core.state import Evidence
    docs, _ = chart.list_documents(limit=100_000)
    ev = EvidenceLedger()
    d = docs[0]
    ev.add(Evidence(d.note_id, d.doc_type, d.date.isoformat(), 0, 20,
                    chart.quote(d.note_id, 0, 20), "date_of_initial_diagnosis", "supports"))
    return gate_answer(spec, submitted, evidence=ev,
                       coverage=CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7)),
                       chart=chart, **kw)


def test_a_diagnosis_after_the_last_document_in_the_chart_is_refused(spec390, chart):
    """`20999999` is the value two E4 runs actually submitted, and it was format-valid.

    Refused rather than recorded, and the distinction is the one this repo paid for: the
    format check was demoted to advisory because it refused `C34.9` — the punctuated form
    ICD-O-3 itself writes — and destroyed correct values over notation. No correct diagnosis
    date can be after every document in the chart, so this check cannot destroy one.
    """
    v = _gate(spec390, chart, {"status": "FOUND", "reasoning": "r",
                               "value": {"date_of_initial_diagnosis": "20999999"}})
    assert v["accepted"] is False
    assert any("20999999" in m for m in v["missing"]), v["missing"]
    assert any("document" in m.lower() for m in v["missing"]), v["missing"]


def test_a_date_that_does_not_exist_is_refused_too(spec390, chart):
    v = _gate(spec390, chart, {"status": "FOUND", "reasoning": "r",
                               "value": {"date_of_initial_diagnosis": "20180229"}})
    assert v["accepted"] is False
    assert any("20180229" in m for m in v["missing"]), v["missing"]


def test_a_notation_miss_is_still_only_recorded(spec390, chart):
    """The line holds. A shape miss is an instruction-following failure and is measured; a
    date that cannot exist is refused. Collapsing them would re-create the round trips that
    cost twelve runs their correct answer."""
    v = _gate(spec390, chart, {"status": "FOUND", "reasoning": "r",
                               "value": {"date_of_initial_diagnosis": "2018-11-07"}})
    assert not any("2018-11-07" in m for m in v.get("missing", [])), v.get("missing")


def test_a_date_inside_the_chart_is_untouched(spec390, chart):
    v = _gate(spec390, chart, {"status": "FOUND", "reasoning": "r",
                               "value": {"date_of_initial_diagnosis": "20180107"}})
    assert not any("20180107" in m for m in v.get("missing", [])), v.get("missing")


def test_an_imputed_date_is_bounded_by_its_year_not_by_a_fabricated_day(spec390, chart):
    """`20189999` says the month and day are unknown. Reading `99` as a month would compare
    a date the run never claimed."""
    v = _gate(spec390, chart, {"status": "FOUND", "reasoning": "r",
                               "value": {"date_of_initial_diagnosis": "20189999"}})
    assert not any("20189999" in m for m in v.get("missing", [])), v.get("missing")
