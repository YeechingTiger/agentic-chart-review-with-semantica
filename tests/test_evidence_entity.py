"""一条引文说的是哪个标本，要能写下来。

"对的文档、错的片段"——真实的原文，说的却是另一个标本——是 eval-overconfidence 点名的
失败模式，而扁平的 span 列表在结构上无法表达它：没有地方写"这条说的是 A，那条说的是 B"。
submit_answer 早就有 reported_lesion 了；证据这边补上锚点，两边就能机器比对。

字段可选：没锚点不是缺陷，是没用起来。检查在两边都为空时沉默。
"""
from __future__ import annotations

from acr.evals import RunRecord, detect_entity_answer_mismatch
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


def test_evidence_about_another_lesion_than_the_one_reported_is_flagged():
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "left upper lobe"}]
    f = detect_entity_answer_mismatch(_run(ev, reported_lesion="right lower lobe"))
    assert len(f) == 1
    assert f[0].detector == "entity_answer_mismatch"
    assert f[0].evidence["reported_lesion"] == "right lower lobe"
    assert f[0].evidence["evidence_entities"] == ["left upper lobe"]


def test_matching_entity_is_clean():
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "right lower lobe"}]
    assert detect_entity_answer_mismatch(_run(ev, reported_lesion="right lower lobe")) == []


def test_silent_when_no_entity_was_recorded():
    """没用这个字段不是缺陷。一个对着空数据报警的检查，会教人把它关掉。"""
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports"}]
    assert detect_entity_answer_mismatch(_run(ev, reported_lesion="right lower lobe")) == []


def test_silent_when_the_answer_named_no_lesion():
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "left upper lobe"}]
    assert detect_entity_answer_mismatch(_run(ev, reported_lesion="")) == []


def test_the_detector_is_wired_into_run_detectors():
    from acr.evals import DetectorConfig, run_detectors
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "left upper lobe"}]
    run = _run(ev, reported_lesion="right lower lobe")
    assert any(f.detector == "entity_answer_mismatch"
               for f in run_detectors(run, config=cfg))


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
