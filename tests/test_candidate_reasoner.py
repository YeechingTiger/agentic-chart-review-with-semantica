"""Candidate Reasoner:独立调用,单一职责,越权是缺陷不是灵活性。

为什么它不是一张卡
------------------
先试过卡。`tactic-counterevidence` 已经用大白话说了目标那件事 —— "点名最可能的替代答案" ——
2026-08-03 在同样十二张图、同样种子上配对跑过:

    每次运行提交的不同值      1.00 -> 1.00   (24 次运行里 0 次提交过第二个)
    推理文本提到替代方案      4/12 -> 1/12   (而且多写了 11% 的字)

请模型去维护候选空间,不会让它维护候选空间;它只是把已有的答案多写几句。所以这里不邀请,
而是构造:一次只干这件事的调用,输出是 schema 不是段落,而且它做不了别的 —— 因为别的
根本没给它。

这一层不拒绝任何东西
--------------------
它的任何返回值都不能拒绝答案、阻止提交或改变一个值。五条确定性检查毁掉 58 个正确值的教训
是关于**闸门**的,不是关于**结构**的,而这两者一直被混在一起。所以 reasoner 崩了、超时了、
返回了垃圾,运行照跑,只是多花一次调用。
"""
from __future__ import annotations

import inspect
import re

from acr.core.state import CandidateLedger, Evidence, EvidenceLedger
from acr.review import candidate_reasoner as CR


def _evidence(n=2) -> EvidenceLedger:
    led = EvidenceLedger()
    for i in range(n):
        led.add(Evidence(f"D{i}", "Progress-Note", f"2020-0{i+1}-01", 0, 12,
                         f"quote {i}", "date_of_initial_diagnosis"))
    return led


def _reply(updates, discriminators=()):
    return {"tool_calls": [{"name": "update_candidates",
                            "args": {"candidate_updates": updates,
                                     "unresolved_discriminators": list(discriminators)}}]}


# --------------------------------------------------------------- 权限边界

def test_the_reasoner_is_given_exactly_one_tool_and_it_is_not_a_chart_tool():
    assert CR.UPDATE_TOOL["function"]["name"] == "update_candidates"
    name = CR.UPDATE_TOOL["function"]["name"]
    for forbidden in ("search", "read", "submit", "list_documents", "record_evidence"):
        assert forbidden not in name


def test_it_imports_nothing_that_could_touch_a_chart_a_gate_or_a_submission():
    """"不允许"写在 docstring 里是一个愿望。这里去数它有没有那个能力。

    查的是 IMPORT,不是文本 —— 系统提示里必须出现 "chart tools" 这些词才能禁止它们,所以
    按字符串搜会把禁令本身当成违规。一个 reasoner 只要能 import 工具箱或网关,下一个人就
    会用它。
    """
    import ast
    tree = ast.parse(inspect.getsource(CR))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
    for forbidden in ("Toolbox", "gate_answer", "answer_gate", "coverage", "corpus",
                      "chartstore", "tools", "toolbox"):
        assert not any(forbidden in i for i in imported), f"reasoner imports {forbidden}"
    # And no attribute access that would reach one anyway.
    body = re.sub(r'"""(.*?)"""', "", inspect.getsource(CR), flags=re.DOTALL)
    for forbidden in ("submit_answer(", "gate_answer(", "Toolbox(", ".dispatch("):
        assert forbidden not in body, f"reasoner 调了 {forbidden}"


def test_the_only_writer_takes_the_candidate_ledger_and_nothing_writable():
    """签名就是权限。给它 chart 或 gate,越权就只是下一个人手滑的距离。"""
    sig = inspect.signature(CR.apply_updates)
    assert list(sig.parameters) == ["ledger", "result", "step", "known_evidence_ids"]
    assert sig.parameters["ledger"].annotation == "CandidateLedger"
    # `reason` likewise: three inputs and a seam, no chart, no gate, no toolbox.
    assert list(inspect.signature(CR.reason).parameters) == \
        ["spec_block", "evidence", "ledger", "invoke"]


def test_the_prompt_forbids_the_four_things_in_words_the_model_reads():
    for phrase in ("NOT ANSWERING", "not deciding", "no chart tools", "Do not propose keywords"):
        assert phrase in CR.SYSTEM


def test_it_is_given_three_inputs_and_not_a_fourth():
    """给它文档清单只会请它去推理检索 —— 那是别的部件的活,而且会让它的输出不可归因。"""
    msgs = CR.build_messages("CONTRACT TEXT", _evidence(), CandidateLedger())
    text = "\n".join(m["content"] for m in msgs)
    assert "CONTRACT TEXT" in text and "EVIDENCE RECORDED" in text and "CANDIDATE SET" in text
    assert "document_type_summary" not in text and "inventory" not in text.lower()


# --------------------------------------------------------------- 三种情况

def test_case_one_a_single_clear_candidate():
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "create",
                                               "value": {"d": "20181107"},
                                               "supports": ["E1"]}]))
    assert CR.apply_updates(led, r, step=3, known_evidence_ids={"E1", "E2"}) == []
    assert len(led.candidates) == 1
    assert led.candidates[0].supporting_evidence_ids == ("E1",)


def test_case_two_two_competing_candidates_with_a_discriminator():
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led, invoke=lambda m, t: _reply(
        [{"action": "create", "value": {"d": "20100612"}, "supports": ["E1"], "contradicts": ["E2"]},
         {"action": "create", "value": {"d": "20100702"}, "supports": ["E2"], "contradicts": ["E1"]}],
        ["whether the earlier suspicious cytology still qualifies once the biopsy confirms"]))
    assert CR.apply_updates(led, r, step=4, known_evidence_ids={"E1", "E2"}) == []
    assert len(led.candidates) == 2
    assert led.evidence_view()["E1"] == {"supports_candidate_ids": ["C1"],
                                         "contradicts_candidate_ids": ["C2"]}
    assert "still qualifies" in led.open_discriminators[0]


def test_case_three_an_abstention_is_a_candidate():
    """"没有可支持的候选"是一个可辩护的读法,不是候选集为空。"""
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "create",
                                               "abstention": "CORPUS_INSUFFICIENT"}]))
    CR.apply_updates(led, r, step=2, known_evidence_ids={"E1", "E2"})
    assert len(led.candidates) == 1
    assert led.candidates[0].label == "CORPUS_INSUFFICIENT"


def test_two_different_abstentions_do_not_collapse_into_one():
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led, invoke=lambda m, t: _reply(
        [{"action": "create", "abstention": "EVIDENCE_INSUFFICIENT"},
         {"action": "create", "abstention": "CORPUS_INSUFFICIENT"}]))
    CR.apply_updates(led, r, step=2, known_evidence_ids={"E1", "E2"})
    assert len(led.candidates) == 2


# --------------------------------------------------------------- 拒绝而不是猜

def test_a_citation_to_a_span_that_was_never_recorded_is_refused():
    """默默丢掉一个链接,读起来和链接成功一模一样 —— 而这条链接正是 grounding 指标要数的。"""
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "create", "value": {"d": "1"},
                                               "supports": ["E1", "E99"]}]))
    bad = CR.apply_updates(led, r, step=1, known_evidence_ids={"E1", "E2"})
    assert any("E99" in x for x in bad)
    assert led.candidates[0].supporting_evidence_ids == ("E1",)


def test_an_update_to_a_candidate_that_does_not_exist_is_refused():
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "reject", "candidate_id": "C7",
                                               "reason": "x"}]))
    assert any("C7" in x for x in CR.apply_updates(led, r, step=1))
    assert led.candidates == []


def test_a_create_with_neither_value_nor_abstention_is_refused():
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "create", "label": "something"}]))
    assert CR.apply_updates(led, r, step=1) == ["create with neither value nor abstention"]


# --------------------------------------------------------------- 失败不能杀掉运行

def test_a_provider_exception_is_a_recorded_result_not_a_raise():
    def boom(m, t):
        raise TimeoutError("provider went away")
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=CandidateLedger(), invoke=boom)
    assert r.ok is False and "TimeoutError" in r.error and r.updates == []


def test_a_reply_with_no_tool_call_is_a_recorded_result():
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=CandidateLedger(),
                  invoke=lambda m, t: {"content": "I think it is 2018."})
    assert r.ok is False and "no update_candidates call" in r.error


def test_it_does_not_call_the_model_before_there_is_anything_to_reason_about():
    calls = []
    r = CR.reason(spec_block="c", evidence=EvidenceLedger(), ledger=CandidateLedger(),
                  invoke=lambda m, t: calls.append(1))
    assert calls == [] and r.ok is True and r.updates == []


def test_ok_false_and_a_genuinely_empty_set_are_different_facts():
    """否则"reasoner 说只有一个候选"和"reasoner 根本没跑成"在 manifest 里长得一样。"""
    failed = CR.reason(spec_block="c", evidence=_evidence(), ledger=CandidateLedger(),
                       invoke=lambda m, t: None)
    empty = CR.reason(spec_block="c", evidence=_evidence(), ledger=CandidateLedger(),
                      invoke=lambda m, t: _reply([]))
    assert failed.to_dict()["ok"] is False
    assert empty.to_dict()["ok"] is True and empty.to_dict()["n_updates"] == 0


# --------------------------------------------------------------- 增量,不是每次重写

def test_a_second_call_updates_the_ledger_rather_than_replacing_it():
    """状态稳定性是验收标准之一:新证据进来时,历史不能被抹掉。"""
    led = CandidateLedger()
    CR.apply_updates(led, CR.ReasonerResult([{"action": "create", "value": {"d": "A"},
                                              "supports": ["E1"]}]), step=1,
                     known_evidence_ids={"E1", "E2"})
    CR.apply_updates(led, CR.ReasonerResult([{"action": "create", "value": {"d": "B"}},
                                             {"action": "reject", "candidate_id": "C1",
                                              "reason": "E2 是更早的合格见证者"}]), step=5,
                     known_evidence_ids={"E1", "E2"})
    c1 = led.by_id("C1")
    assert c1.status == "REJECTED" and "更早" in c1.rejection_reason
    assert c1.supporting_evidence_ids == ("E1",), "先前的链接被第二次调用抹掉了"
    assert c1.created_at_step == 1 and c1.updated_at_step == 5
    assert len(led.candidates) == 2


def test_the_provider_envelope_shapes_all_resolve():
    """三个调用方,三种信封。为自己的管道抛异常的 reasoner 会带走一次运行。"""
    args = {"candidate_updates": [{"action": "create", "value": {"d": "1"}}]}
    import json as _j
    for reply in (args,
                  {"tool_calls": [{"name": "update_candidates", "args": args}]},
                  {"tool_calls": [{"function": {"arguments": _j.dumps(args)}}]}):
        assert CR._extract(reply) == args


def test_one_span_on_both_sides_of_one_candidate_in_one_update_is_refused():
    """真实运行里发生过。同一次更新里把 E4 同时列为支持和反对不是修订,是对一条证据的两个
    互斥主张 —— 两个都记下来,grounding 会把它数两次,而读的人看不出它到底是哪一边。"""
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "create", "value": {"d": "1"},
                                               "supports": ["E1", "E2"],
                                               "contradicts": ["E2"]}]))
    bad = CR.apply_updates(led, r, step=1, known_evidence_ids={"E1", "E2"})
    assert any("BOTH" in x for x in bad)
    got = led.candidates[0]
    assert got.supporting_evidence_ids == ("E1",)
    assert got.contradicting_evidence_ids == ()


def test_a_create_can_declare_itself_leading_without_inventing_an_id():
    """模型在真实运行里编了一个 `candidate_1` 出来,因为它刚创建的候选还没有 id 可引用。"""
    led = CandidateLedger()
    r = CR.reason(spec_block="c", evidence=_evidence(), ledger=led,
                  invoke=lambda m, t: _reply([{"action": "create", "value": {"d": "1"},
                                               "state": "LEADING", "supports": ["E1"]}]))
    assert CR.apply_updates(led, r, step=1, known_evidence_ids={"E1", "E2"}) == []
    assert led.leading().candidate_id == "C1"
