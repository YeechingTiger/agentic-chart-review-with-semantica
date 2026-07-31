"""证据集合作为一个整体，也可以有毛病。

现有检查全是逐条的：这条引文回原文对得上吗、这条span非空吗。DeepEvidence 审计证据图时
报的是集合层面的数（重复率 0.6%、关系正确率 ≥99%），而这里连"同一个字段被同一段文字
支持了两次"都没人数过。三项，全部确定性，全部给比率——一个只说"有重复"的检查没法和
任何基准对话。
"""
from __future__ import annotations

from acr.evals import RunRecord, audit_evidence_set


def _run(evidence: list[dict]) -> RunRecord:
    return RunRecord(manifest={"patient_id": "SYN01", "evidence": evidence}, trace=[])


def _ev(note="N1", start=0, end=10, supports="primary_site", stance="supports") -> dict:
    return {"note_id": note, "start": start, "end": end, "supports": supports,
            "stance": stance, "quote": "x" * (end - start)}


def test_overlapping_spans_for_one_field_are_reported():
    """完全相同的 span 台账自己会去重；重叠的不会，而它是同一句话记了两遍。"""
    f = audit_evidence_set(_run([_ev(start=0, end=40), _ev(start=10, end=50)]))
    hit = [x for x in f if x.detector == "evidence_span_overlap"]
    assert len(hit) == 1
    assert hit[0].evidence["n_overlapping_pairs"] == 1
    assert hit[0].evidence["overlap_rate"] == 0.5      # 2 条里 1 条是多余的


def test_non_overlapping_spans_are_clean():
    f = audit_evidence_set(_run([_ev(start=0, end=10), _ev(start=20, end=30)]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_overlap_is_per_field_not_across_fields():
    """同一段文字同时支持部位和组织学是正常的，不是重复。"""
    f = audit_evidence_set(_run([_ev(supports="primary_site"),
                                 _ev(supports="histology")]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_a_contradiction_with_nothing_to_contradict_is_reported():
    f = audit_evidence_set(_run([_ev(supports="histology", stance="contradicts")]))
    hit = [x for x in f if x.detector == "orphan_contradiction"]
    assert len(hit) == 1
    assert hit[0].evidence["fields"] == ["histology"]


def test_a_contradiction_beside_a_support_is_a_conflict_not_an_orphan():
    """两边都有＝记录内部有矛盾，这是要如实报告的状态，不是缺陷。"""
    f = audit_evidence_set(_run([_ev(supports="histology"),
                                 _ev(supports="histology", start=99, end=120,
                                     stance="contradicts")]))
    assert not [x for x in f if x.detector == "orphan_contradiction"]


def test_a_field_resting_on_one_document_is_reported():
    f = audit_evidence_set(_run([_ev(note="N1", supports="primary_site")]))
    hit = [x for x in f if x.detector == "single_witness_field"]
    assert hit and hit[0].evidence["fields"] == ["primary_site"]


def test_two_documents_for_one_field_is_not_single_witness():
    f = audit_evidence_set(_run([_ev(note="N1"), _ev(note="N2", start=5, end=9)]))
    assert not [x for x in f if x.detector == "single_witness_field"]


def test_no_evidence_is_silent_here():
    """空台账是交卷检查的案子。这个审计只描述已经存在的证据集合。"""
    assert audit_evidence_set(_run([])) == []


def test_the_answer_copy_of_the_ledger_is_audited_too():
    """存档单两处都写了证据：顶层一份，`answer.evidence` 一份（agent.py:987 与 :1205）。

    顶层为空、`answer` 里有的存档单是真实存在的形状（`run_manifest.build_manifest` 只写
    后者），一个只看顶层的审计会对那些 run 报告"没有证据"——而"没有证据"是交卷检查的
    结论，不是这个审计能得出的。
    """
    ev = [_ev(start=0, end=40), _ev(start=10, end=50)]
    run = RunRecord(manifest={"patient_id": "SYN01", "answer": {"evidence": ev}}, trace=[])
    assert [x.detector for x in audit_evidence_set(run) if
            x.detector == "evidence_span_overlap"] == ["evidence_span_overlap"]


def test_audit_is_wired_into_run_detectors():
    from acr.evals import DetectorConfig, run_detectors
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    run = _run([_ev(start=0, end=40), _ev(start=10, end=50)])
    assert any(f.detector == "evidence_span_overlap" for f in run_detectors(run, config=cfg))
