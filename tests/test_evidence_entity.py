"""一条引文说的是哪个标本，要能写下来。

"对的文档、错的片段"——真实的原文，说的却是另一个标本——是 eval-overconfidence 点名的
失败模式，而扁平的 span 列表在结构上无法表达它：没有地方写"这条说的是 A，那条说的是 B"。

原本这里还有一个 `entity_answer_mismatch`，拿锚点和 `reported_lesion` 精确比对。
2026-07-31 在十二次真实运行上测量：CRITICAL 报了十二次，十二次全错。两边根本不是同类
字符串——锚点是短标签，`reported_lesion` 是模型写的一整句解释——相等只可能不成立，
所以那个检查在正确的运行上也无法通过。一个永远为真的 CRITICAL 比没有检查更坏：它教人
连带跳过同一严重度里的 `patient_crossover`。

改成 `multiple_anchored_entities`：只数锚点有几个不同的标签，不判断它们是否一致。
这是数据答得了的问题；"两种说法是不是同一个病灶"是临床判断，写进 Python 就是
DETERMINISTIC_RULES_REMOVED.md 记录过的那个错误。

字段可选：没锚点不是缺陷，是没用起来。检查在没有锚点时沉默。
"""
from __future__ import annotations

from acr.evals import DetectorConfig, RunRecord, run_detectors
from acr.state import Evidence, EvidenceLedger


def test_evidence_carries_an_optional_entity():
    e = Evidence("N1", "pathology", "2024-01-01", 0, 10, "x", "primary_site",
                 entity="specimen A")
    assert e.to_dict()["entity"] == "specimen A"


def test_entity_defaults_to_empty_so_old_records_still_load():
    e = Evidence("N1", "pathology", "2024-01-01", 0, 10, "x", "primary_site")
    assert e.to_dict()["entity"] == ""


def test_same_span_different_entity_is_not_a_duplicate():
    """去重键必须带上实体，否则两个标本的同位置引文会被吞掉一条。"""
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen B"))
    assert len(led.items) == 2


def test_identical_entity_still_de_duplicates():
    led = EvidenceLedger()
    for _ in range(2):
        led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    assert len(led.items) == 1


def test_the_rendered_ledger_shows_the_anchor():
    """渲染给模型看的台账里要有实体，否则模型看不见自己刚记下的区分。"""
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    assert "specimen A" in led.render()


def test_the_rendered_ledger_says_nothing_when_no_anchor_was_recorded():
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology"))
    assert "entity" not in led.render()


def _run(evidence, reported_lesion="") -> RunRecord:
    return RunRecord(manifest={"patient_id": "SYN01", "evidence": evidence,
                               "answer": {"reported_lesion": reported_lesion}}, trace=[])


def test_no_detector_reads_the_anchor():
    """回归：`entity` 上不许再挂检测器，除非工具契约先要求标签稳定。

    两个都写过、都在同一批十二次运行上测过、都删了。`entity_answer_mismatch` 拿锚点和
    `reported_lesion`（一整句散文）精确比对，12/12 报 CRITICAL，12 次全错。
    `multiple_anchored_entities` 改数不同标签的个数，12/12 报，只有 1 次对——另外四次是
    同一个病灶换了说法（"肿块"→"癌"），而那正是病历的写法。

    它测的是措辞漂移，不是实体个数。要把 "sigmoid colon mass" 和 "sigmoid colon carcinoma"
    判成一个东西，需要临床判断；这棵树已经为"把临床判断写进 Python"付过一次代价。
    """
    ev = [{"note_id": "N1", "start": 0, "end": 9, "entity": "left upper lobe"},
          {"note_id": "N2", "start": 0, "end": 9, "entity": "right lower lobe"}]
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    names = {f.detector for f in run_detectors(_run(ev), config=cfg)}
    assert "entity_answer_mismatch" not in names
    assert "multiple_anchored_entities" not in names

    import acr.evals as E
    assert not hasattr(E, "detect_entity_answer_mismatch")
    assert not hasattr(E, "detect_multiple_anchored_entities")


def test_the_anchor_is_offered_and_never_required():
    from acr.tools.toolbox import TOOL_SCHEMAS
    schema = next(s for s in TOOL_SCHEMAS
                  if s["function"]["name"] == "record_evidence")["function"]["parameters"]
    assert "entity" in schema["properties"]
    assert "entity" not in schema["required"]


#: The corpus filename convention is `<DocType>_<YYYY-MM-DD>[__<n>].txt` and the note_id IS the
#: stem — see `acr.corpus.FILENAME_RE`, and the same note in tests/test_read_causality.py.
_NOTE_ID = "pathology_2024-01-01"


def _toolbox(tmp_path):
    from acr.corpus import Corpus
    from acr.coverage import CoverageLedger, ForcedSampler
    from acr.tools.toolbox import Toolbox
    d = tmp_path / "patients" / "SYN01"
    d.mkdir(parents=True)
    (d / f"{_NOTE_ID}.txt").write_text("final diagnosis: adenocarcinoma\n", encoding="utf-8")
    chart = Corpus(tmp_path / "patients").chart("SYN01")
    assert len(chart) == 1, "fixture built an empty chart; the filename does not parse"
    docs, _ = chart.list_documents(limit=100)
    return Toolbox(chart, EvidenceLedger(), CoverageLedger(docs, (), ForcedSampler(1)))


def test_the_tool_carries_the_anchor_through_to_the_ledger(tmp_path):
    """schema 上有这个参数还不够——`_t_record_evidence` 得真的把它传进 `Evidence`。"""
    tb = _toolbox(tmp_path)
    out, _ = tb.dispatch("record_evidence",
                         {"note_id": _NOTE_ID, "start": 0, "end": 16,
                          "supports": "histology", "entity": "specimen A"})
    assert out.get("recorded") is True, out
    assert tb.evidence.items[0].entity == "specimen A"


def test_the_tool_still_works_without_an_anchor(tmp_path):
    tb = _toolbox(tmp_path)
    out, _ = tb.dispatch("record_evidence",
                         {"note_id": _NOTE_ID, "start": 0, "end": 16, "supports": "histology"})
    assert out.get("recorded") is True, out
    assert tb.evidence.items[0].entity == ""
