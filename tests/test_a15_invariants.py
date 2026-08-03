"""A1.5-v1:冻结的表面,和它必须一直成立的五条不变量。

为什么要冻结
------------
下一轮是**评价**,不是开发。如果每发现一个失败就改 seeder 的来源、候选身份、reasoner 提示、
冲突规则或 discriminator 的形状,最后那批指标就不再是对某个版本的测量,而是一份开发过程的
记录 —— 而这棵树上已经有一整份那样的文档(`MODULE_LADDER_EXPERIMENT.md` 里每一个数字都是
在被测对象自己的开发集上算的)。

所以这里把 v1 的表面钉死。改动它不是被禁止的,是**必须先改这个文件**,于是版本号会跟着动,
于是没有人能不小心地把两个版本的数字放进同一张表。

五条不变量
----------
每一条都来自一次真实运行里发生过的事,不是想象出来的失败模式。
"""
from __future__ import annotations

import inspect

import pytest

from acr.core.state import (
    ANSWERABILITY,
    CANDIDATE_STATES,
    DISCRIMINATOR_STATES,
    Candidate,
    CandidateLedger,
)
from acr.review import candidate_reasoner as CR

#: 冻结的版本名。改了下面任何一个被钉住的表面,就得改它。
#:
#: A1.5-v1 的机械 seeder 已经**删掉**了 —— 它只在日期上能跑(五份契约里三份静默 no-op)、
#: 过度包含使 SYN0002 三次运行全部丢掉 gold 答案、clear 层 40% 假竞争,而且"枚举再筛"这条
#: 原则本身是无界值域的产物:`value_domain: icdo3_lung` 那种目标,候选空间是一张声明好的表,
#: "把所有可能值 seed 进来"意味着几百个码。
#:
#: 留下来的是候选账本本身和那次独立调用。名字跟着动,于是两个版本的数字接不到一起。
A15_VERSION = "A1.5-v2-no-seeder"


# ====================================================================== 冻结的表面

def test_the_frozen_surface_is_what_v1_says_it_is():
    """五个词表。加一个值不是小事:它是 Controller 之后要分支的东西。"""
    assert CANDIDATE_STATES == ("ACTIVE", "LEADING", "REJECTED", "SELECTED")
    assert ANSWERABILITY == ("UNDETERMINED", "VALUE_AVAILABLE", "EVIDENCE_INSUFFICIENT",
                             "CORPUS_INSUFFICIENT")
    assert DISCRIMINATOR_STATES == ("UNRESOLVED", "ALREADY_RESOLVED",
                                    "UNRESOLVABLE_FROM_CORPUS", "SPEC_DEPENDENT")


def test_the_candidate_field_set_is_frozen():
    """一个字段少了,分析器会静默地少算一列;多了没人读,就是没人维护。"""
    assert {f for f in Candidate.__dataclass_fields__} == {
        "candidate_id", "value", "status", "abstention", "label",
        "supporting_evidence_ids", "contradicting_evidence_ids",
        "unresolved_discriminators", "confidence", "created_at_step", "updated_at_step",
        "rejection_reason", "not_a_target_value", "rejecting_rule", "state_history"}


def test_the_discriminator_shape_is_frozen():
    led = CandidateLedger()
    led.declare({"d": "A"}, step=1)
    led.declare({"d": "B"}, step=1)
    led.add_discriminator({"candidate_a": "C1", "candidate_b": "C2",
                           "unresolved_fact": "which is earlier"}, step=1)
    assert set(led.discriminators[0]) == {
        "candidate_a", "candidate_b", "status", "unresolved_fact", "evidence_needed",
        "likely_source", "can_be_resolved_from_current_corpus", "step"}


# ====================================================================== 五条不变量

def test_invariant_1_no_semantic_difference_means_no_new_candidate():
    """同一主张,带上 null / 空串 / False 之后仍是同一个候选。

    真实运行:seeder 建 `{date: ...}`,提交的答案带着三个 `False` 的 imputation flag,
    一个日期变成两个候选,第二个被盖上"从没声明过"的章,冲突集也被撑成 2。
    """
    led = CandidateLedger()
    a = led.declare({"d": "20200302"}, step=1)
    for extra in ({"y": False}, {"y": None}, {"y": ""}, {"y": "  "}):
        b = led.declare({"d": "20200302", **extra}, step=2)
        assert b.candidate_id == a.candidate_id, extra
    assert len(led.candidates) == 1


def test_invariant_2_a_flag_that_is_set_is_a_different_claim():
    """"1995,年份是估的"和"1995,年份读出来的"是两个不同的主张,而契约有三个字段来说它。"""
    led = CandidateLedger()
    led.declare({"d": "20159999"}, step=1)
    led.declare({"d": "20159999", "year_imputed": True}, step=1)
    assert len(led.candidates) == 2


def test_invariant_3_a_candidate_created_in_this_call_is_immediately_referenceable():
    """真实运行里 discriminator 的两端写成了 `"NEW"`,因为刚创建的候选还没有 id。

    模型永远知道它刚写下的**值**,所以值可以解析;占位符不能,而且解析不出来会被报出来,
    不会指到碰巧排第一的那个候选身上。
    """
    led = CandidateLedger()
    r = CR.ReasonerResult(
        [{"action": "create", "value": {"d": "20100517"}},
         {"action": "create", "value": {"d": "20100522"}}],
        [{"candidate_a": "20100517", "candidate_b": "20100522",
          "unresolved_fact": "which qualifies under the conflict rules"}])
    assert CR.apply_updates(led, r, step=1) == []
    d = led.discriminators[0]
    assert (d["candidate_a"], d["candidate_b"]) == ("C1", "C2")


def test_invariant_3_a_placeholder_reference_is_reported_not_resolved_to_anything():
    led = CandidateLedger()
    r = CR.ReasonerResult(
        [{"action": "create", "value": {"d": "20100517"}}],
        [{"candidate_a": "NEW", "candidate_b": "NEW", "unresolved_fact": "which is earlier"}])
    bad = CR.apply_updates(led, r, step=1)
    assert any("no resolvable candidate" in x for x in bad)


def test_invariant_4_an_answerability_status_never_becomes_a_value_candidate():
    """A1 的三次"多候选"全部是"值 vs 弃权"。放进同一个集合,conflict 就同时指两件事。"""
    led = CandidateLedger()
    led.set_answerability("CORPUS_INSUFFICIENT", step=1)
    led.declare({"d": "20200302"}, step=1)
    assert led.answerability == "CORPUS_INSUFFICIENT"
    assert [c.value for c in led.candidates] == [{"d": "20200302"}]
    assert led.conflict_sets == [], "弃权不该和一个值形成竞争集"
    for c in led.candidates:
        assert c.value, "一个 answerability 状态漏进了候选集"


def test_invariant_5_a_rejected_candidate_stays_with_everything_that_put_it_out():
    """物理删掉一个候选,就回到了"从没考虑过" —— 而那正是 A1 的问题。"""
    led = CandidateLedger()
    c = led.declare({"d": "20200401"}, step=1)
    led.link(c.candidate_id, "E3", "supports", step=1)
    led.set_state(c.candidate_id, "REJECTED", step=6,
                  reason="confirmatory; superseded by the earlier clinical impression")
    led.by_id(c.candidate_id).rejecting_rule = "conflict_rule.3"

    got = led.to_dict()["candidates"][0]
    assert got["status"] == "REJECTED"
    assert got["rejection_reason"] and got["rejecting_rule"] == "conflict_rule.3"
    assert got["supporting_evidence_ids"] == ["E3"]
    assert [(h["to"], h["step"]) for h in got["state_history"]] == [("REJECTED", 6)]


def test_invariant_5_nothing_in_the_ledger_can_delete_a_candidate():
    """签名就是保证。没有 remove,就没有人能顺手加一个。"""
    for name in ("remove", "delete", "drop", "prune", "clear", "pop"):
        assert not hasattr(CandidateLedger, name), f"CandidateLedger.{name} exists"


# ====================================================================== A2 的输入接口

def test_the_controller_input_is_defined_before_the_controller_is():
    """冻结 Controller 未来只能读什么,现在就冻。

    如果 Controller 允许回去读原始 evidence 自己重做候选推理,A1.5 就失去了架构意义 ——
    它会变成一个被绕过的中间层,而绕过它的那次推理没有任何记录。
    """
    led = CandidateLedger()
    a = led.declare({"d": "20100517"}, step=1)
    led.declare({"d": "20100522"}, step=1)
    led.link(a.candidate_id, "E1", "supports", step=1)
    led.add_discriminator({"candidate_a": "C1", "candidate_b": "C2",
                           "unresolved_fact": "which qualifies"}, step=1)
    ci = led.controller_input(coverage_facts={"n_read": 5}, budget={"usd_left": 1.0})

    assert set(ci) == {"active_candidates", "conflict_sets", "unresolved_discriminators",
                       "answerability", "coverage_facts", "remaining_budget"}
    assert set(ci["active_candidates"][0]) == {
        "candidate_id", "value", "status", "supporting_evidence_ids",
        "contradicting_evidence_ids"}
    # NOT in it, and each omission is the point: no evidence text, no chart, no document
    # inventory. A Controller that re-reads the spans is a Controller redoing the candidate
    # reasoning off the record.
    blob = repr(ci)
    for forbidden in ("quote", "note_id", "chart", "document_type"):
        assert forbidden not in blob


def test_only_live_candidates_reach_the_controller_but_the_rejected_stay_in_the_ledger():
    """Controller 要决定的是还没解决的选择。已经结案的东西留在账本里给读的人,
    不进它的输入 —— 否则它会对着一个已经解决的冲突继续搜。"""
    led = CandidateLedger()
    led.declare({"d": "A"}, step=1)
    led.declare({"d": "B"}, step=1)
    led.set_state("C2", "REJECTED", step=2, reason="x")
    ci = led.controller_input()
    assert [c["candidate_id"] for c in ci["active_candidates"]] == ["C1"]
    assert len(led.candidates) == 2


@pytest.mark.parametrize("state", ["ALREADY_RESOLVED", "UNRESOLVABLE_FROM_CORPUS"])
def test_a_settled_discriminator_does_not_reach_the_controller(state):
    """"已经解决"和"这个语料解决不了"都不是可以驱动下一步搜索的东西。"""
    led = CandidateLedger()
    led.declare({"d": "A"}, step=1)
    led.declare({"d": "B"}, step=1)
    led.add_discriminator({"candidate_a": "C1", "candidate_b": "C2",
                           "unresolved_fact": "which is earlier", "status": state}, step=1)
    assert led.controller_input()["unresolved_discriminators"] == []
