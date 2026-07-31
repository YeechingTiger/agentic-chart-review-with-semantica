"""一次读要说明它为什么发生，否则归因只能靠相邻位置猜。

轨迹是平铺的事件序列，事件之间没有连线。归因 agent 被要求"分清记录证明的和推测的"，
但在没有成因字段的轨迹上，"第 9 步读 D12 是因为第 7 步搜索返回了它"永远是推测——
而报告不会说它是推测。这个字段把它变成记录。

不填不拒：成因是判断，判断可以缺席，缺席要能被数出来。
"""
from __future__ import annotations

import pytest

from acr.evaluation.evals import Finding, RunRecord, detect_uncaused_reads, run_detectors
from acr.tools.toolbox import CAUSE_PARAM, TOOL_SCHEMAS

#: The corpus filename convention is `<DocType>_<YYYY-MM-DD>[__<n>].txt` and the note_id IS the
#: stem — see `acr.corpus.FILENAME_RE`. A name that does not parse is skipped rather than
#: guessed at, so getting this wrong builds an EMPTY chart and every read below fails as
#: "unknown note_id" instead of as the thing under test.
NOTE_ID = "pathology_2024-01-01"


@pytest.fixture
def toolbox_with_one_doc(tmp_path):
    from acr.corpus import Corpus
    from acr.coverage import CoverageLedger, ForcedSampler
    from acr.state import EvidenceLedger
    from acr.tools.toolbox import Toolbox
    d = tmp_path / "patients" / "SYN01"
    d.mkdir(parents=True)
    (d / f"{NOTE_ID}.txt").write_text("final diagnosis: adenocarcinoma\n", encoding="utf-8")
    chart = Corpus(tmp_path / "patients").chart("SYN01")
    assert len(chart) == 1, "fixture built an empty chart; the filename does not parse"
    docs, _ = chart.list_documents(limit=100)
    return Toolbox(chart, EvidenceLedger(), CoverageLedger(docs, (), ForcedSampler(1)))


@pytest.fixture
def tracer(tmp_path):
    from acr.trace import Tracer
    return Tracer.create(tmp_path, "t1")


def _schema(name: str) -> dict:
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == name:
            return s["function"]["parameters"]["properties"]
    raise AssertionError(f"no tool {name!r}")


@pytest.mark.parametrize("tool", ["read_document", "read_documents_batch", "search_notes"])
def test_retrieval_tools_ask_for_a_cause(tool: str):
    assert CAUSE_PARAM in _schema(tool), f"{tool} never asks the model why"


@pytest.mark.parametrize("tool", ["read_document", "read_documents_batch", "search_notes"])
def test_cause_is_never_required(tool: str):
    """记录，不是闸门。必填会把判断变成仪式。"""
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == tool:
            assert CAUSE_PARAM not in s["function"]["parameters"].get("required", [])


def test_dispatch_accepts_and_strips_the_cause(toolbox_with_one_doc):
    """`_t_` 方法不必知道这个参数——它在 dispatch 就被摘掉了，一处而不是每个工具一处。"""
    tb = toolbox_with_one_doc
    out, _ms = tb.dispatch("read_document", {"note_id": NOTE_ID, CAUSE_PARAM: "search #7 hit"})
    assert "error" not in out
    assert tb.last_cause == "search #7 hit"


def test_dispatch_clears_the_cause_between_calls(toolbox_with_one_doc):
    """上一次的成因不许粘到下一次——那会造出一条没人写过的因果连线。"""
    tb = toolbox_with_one_doc
    tb.dispatch("read_document", {"note_id": NOTE_ID, CAUSE_PARAM: "thread T3"})
    tb.dispatch("read_document", {"note_id": NOTE_ID})
    assert tb.last_cause == ""


def test_tracer_promotes_the_cause_to_a_top_level_field(tracer):
    ev = tracer.tool("read_document", {"note_id": NOTE_ID}, {"ok": True}, because="thread T3")
    assert ev["because"] == "thread T3"


def test_detector_counts_reads_with_no_cause():
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document", "because": "search #2 hit"},
        {"kind": "tool", "tool": "read_document", "because": ""},
        {"kind": "tool", "tool": "read_document"},
        {"kind": "tool", "tool": "submit_answer"},          # 不是读，不进分母
    ])
    findings = detect_uncaused_reads(run)
    assert len(findings) == 1
    ev = findings[0].evidence
    assert (ev["n_reads"], ev["n_uncaused"]) == (3, 2)
    assert findings[0].detector == "uncaused_read"


def test_detector_is_silent_when_every_read_has_a_cause():
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document", "because": "search #2 hit"},
    ])
    assert detect_uncaused_reads(run) == []


def test_detector_is_silent_on_a_run_with_no_reads():
    """零阅读是 detect_zero_document_read 的案子，不是这个检测器的——两个检测器报同一件事，
    读的人会以为是两个问题。"""
    assert detect_uncaused_reads(RunRecord(manifest={}, trace=[])) == []


def test_detector_is_wired_into_run_detectors():
    from acr.evaluation.evals import DetectorConfig
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document"},
    ])
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    assert any(f.detector == "uncaused_read" for f in run_detectors(run, config=cfg))
