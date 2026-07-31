"""证据集合作为一个整体，也可以有毛病。

现有检查全是逐条的：这条引文回原文对得上吗、这条span非空吗。DeepEvidence 审计证据图时
报的是集合层面的数（重复率 0.6%、关系正确率 ≥99%），而这里连"同一段文字被记了两遍"
都没人数过。

原本这里有三项。2026-07-31 在十二次真实运行上测量后删掉两项：`orphan_contradiction`
和 `single_witness_field` 都按 `supports` 分组，而 `supports` 不是字段名——
`record_evidence` 的说明书写的是 "which field **or assertion** this backs"，真实运行里
模型每次都写一整句话。按散文分组，每组必然只有一条，于是 single_witness 12/12 必报、
orphan 8/12 必报，全是构造性误报。一个不可能返回干净的检查什么都没测。

删掉而不是放宽：证据行和 spec 字段之间没有机器可读的连接，这两个问题在现有存档格式下
根本算不出来，而"猜一个"正是 DETERMINISTIC_RULES_REMOVED.md 已经付过代价的做法。

留下的一项按 note_id 分组，不碰 supports——同一份文档里两段字符范围重叠，就是同一段
文字记了两遍，与每行写了什么散文无关。
"""
from __future__ import annotations

from acr.evaluation.evals import RunRecord, audit_evidence_set


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


def test_overlap_is_per_document_and_ignores_supports():
    """按 note_id 分组，不按 supports。

    旧版本按 supports 分组，于是同一段文字用两句不同的话描述时**漏报**——而真实运行里
    supports 每行都不一样，所以它实际上从不触发。重叠是文档内的字符区间事实，与那行
    写了什么无关。
    """
    f = audit_evidence_set(_run([_ev(start=0, end=40, supports="一句话"),
                                 _ev(start=10, end=50, supports="另一句完全不同的话")]))
    hit = [x for x in f if x.detector == "evidence_span_overlap"]
    assert len(hit) == 1 and hit[0].evidence["n_overlapping_pairs"] == 1


def test_spans_in_different_documents_never_overlap():
    f = audit_evidence_set(_run([_ev(note="N1", start=0, end=40),
                                 _ev(note="N2", start=10, end=50)]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_the_two_prose_grouped_detectors_are_gone():
    """回归：按 supports 分组的两个检测器不得回来。

    它们在十二次真实运行上分别报了 12 次和 8 次，全部是误报，因为 supports 是自由文本。
    重新引入任何按 supports 分组的检查，都会重现同一个失效。
    """
    import acr.evaluation.evals as E
    assert not hasattr(E, "detect_orphan_contradiction")
    ev = [_ev(supports="独一无二的一句话"), _ev(note="N2", supports="另一句")]
    names = {f.detector for f in audit_evidence_set(_run(ev))}
    assert "orphan_contradiction" not in names
    assert "single_witness_field" not in names


def test_a_realistic_prose_supports_set_is_clean():
    """真实形状：每条 supports 都不同、每条各在一份文档里 —— 应当一条都不报。"""
    ev = [{"note_id": "Path-2023-04-27", "start": 100, "end": 180,
           "supports": "2023-04-27 biopsy definitively establishes adenocarcinoma",
           "stance": "supports"},
          {"note_id": "Cyto-2023-04-12", "start": 193, "end": 416,
           "supports": "2023-04-12 cytology was suspicious but recommended tissue",
           "stance": "supports"}]
    assert audit_evidence_set(_run(ev)) == []


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
    from acr.evaluation.evals import DetectorConfig, run_detectors
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    run = _run([_ev(start=0, end=40), _ev(start=10, end=50)])
    assert any(f.detector == "evidence_span_overlap" for f in run_detectors(run, config=cfg))
