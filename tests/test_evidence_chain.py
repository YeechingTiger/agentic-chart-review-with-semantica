"""因果链必须是**链**，而不是一组注解。

现在 `because` 是散文，schema 里写着 "Recorded, never checked"。于是
`detect_uncaused_reads` 只能数有没有，不能验对不对 —— 一句编造的理由和一句真的理由，
在记录里长得一模一样。

一条链的构成要件只有一个：**每个标签是按 ID 指向另一个产物的可解析指针**。trace 事件本来
就有 `seq`，所以锚点是现成的；缺的是让 `because` 带上它，以及一个把指针走通的解析器。

走通之后能算三样现在算不出来的东西：

  * 解析失败率 —— 确定性的，不需要任何模型判断；
  * **前向引用** —— 一次读取声称由排在它后面的事件引起，这是可证明不可能的，而这正是
    模型事后为一个已做过的动作补理由时会产生的形状；
  * grounding ratio 与链深 —— 评测可以挂在上面。

向后兼容是硬要求：已记录的每一次运行里 `because` 都是字符串，它们必须继续可读、并且**不被
记成失败** —— 散文式的理由不是造假，它只是无法核对，而这两件事的区别就是这个文件的全部。
"""
from __future__ import annotations

from acr.evaluation import evals as E
from acr.evaluation.evidence_chain import (
    FORWARD_REF,
    GROUNDED,
    PROSE_ONLY,
    UNRESOLVED_REF,
    UNSOURCED,
    chain_report,
)


def _run(*calls: dict) -> E.RunRecord:
    """A RunRecord whose trace is the given tool calls, numbered from 1."""
    trace = [{"seq": i, "kind": "tool", "tool": c.pop("tool", "read_document"), **c}
             for i, c in enumerate(calls, start=1)]
    rec = E.RunRecord({"patient_id": "SYN0001", "spec_id": "S"}, source="synthetic")
    rec.trace = trace
    return rec


def _status(rep: dict) -> list[str]:
    return [link["status"] for link in rep["links"]]


# ------------------------------------------------------------------ the four link states
def test_a_pointer_at_an_earlier_event_resolves():
    rep = chain_report(_run(
        {"tool": "search_notes", "args": {"q": "adenocarc"}},
        {"args": {"doc": "path-1"}, "because": {"why": "the search surfaced it",
                                                "from": {"event": 1}}},
    ))
    assert _status(rep) == [UNSOURCED, GROUNDED]
    assert rep["links"][1]["why"] == "the search surfaced it"


def test_a_pointer_at_an_event_that_does_not_exist_is_unresolved():
    rep = chain_report(_run(
        {"args": {"doc": "path-1"}, "because": {"why": "x", "from": {"event": 99}}},
    ))
    assert _status(rep) == [UNRESOLVED_REF]


def test_a_pointer_at_a_LATER_event_is_impossible_and_named_as_such():
    """一次调用不可能由还没发生的事引起。

    这是这套改动带来的唯一一个**纯确定性**的新检测器，而且它抓的正是事后补理由的形状：
    动作已经做了，理由是回头写的，于是指向了当时还不存在的东西。分开成自己的状态而不是并进
    UNRESOLVED_REF，因为"指错了"和"指了个未来"是两种不同的失败，后者不可能是笔误。
    """
    rep = chain_report(_run(
        {"args": {"doc": "path-1"}, "because": {"why": "x", "from": {"event": 2}}},
        {"tool": "search_notes", "args": {"q": "later"}},
    ))
    assert _status(rep) == [FORWARD_REF, UNSOURCED]


def test_a_pointer_at_itself_is_also_forward():
    """自引用是环，不是链。按同一条规则处理：`seq` 不小于自己就不合法。"""
    rep = chain_report(_run(
        {"args": {"doc": "d"}, "because": {"why": "x", "from": {"event": 1}}},
    ))
    assert _status(rep) == [FORWARD_REF]


# ------------------------------------------------------------------ 向后兼容
def test_a_plain_string_because_is_prose_not_a_failure():
    """已记录的每一次运行都是这个形状。散文无法核对，但它不是造假。

    分成 PROSE_ONLY 而不是并进 GROUNDED 或 UNSOURCED：并进前者会把无法核对的说成已核对，
    并进后者会把历史上所有认真写了理由的运行记成没写。两种都会让这个数字失去意义。
    """
    rep = chain_report(_run(
        {"args": {"doc": "d"}, "because": "the search that surfaced this document"},
    ))
    assert _status(rep) == [PROSE_ONLY]
    assert rep["links"][0]["why"] == "the search that surfaced this document"


def test_a_because_object_with_no_pointer_is_prose_too():
    rep = chain_report(_run(
        {"args": {"doc": "d"}, "because": {"why": "no pointer here"}},
    ))
    assert _status(rep) == [PROSE_ONLY]


def test_a_malformed_pointer_does_not_crash_and_is_unresolved():
    """一个坏标签必须变成一条记录，不能变成一次异常 —— 评测跑的是别人已经产生的运行。"""
    for bad in ({"why": "x", "from": {"event": "not-a-number"}},
                {"why": "x", "from": "not-a-mapping"},
                {"why": "x", "from": {}}):
        rep = chain_report(_run({"args": {"doc": "d"}, "because": bad}))
        assert _status(rep) == [UNRESOLVED_REF], bad


# ------------------------------------------------------------------ 挂评测的那三个数
def test_the_grounding_ratio_counts_only_resolvable_links():
    """分母是**能带 because 的调用总数**，分子只有真的解析通的。

    散文不进分子。这条比率的用处正是把"写了理由"和"理由可核对"分开 —— 把散文算进去，
    这个数字就退回成 `detect_uncaused_reads` 已经在数的东西。
    """
    rep = chain_report(_run(
        {"tool": "search_notes", "args": {"q": "a"}},
        {"args": {"doc": "d1"}, "because": {"why": "x", "from": {"event": 1}}},
        {"args": {"doc": "d2"}, "because": "prose"},
        {"args": {"doc": "d3"}},
    ))
    assert rep["n_links"] == 4
    assert rep["n_grounded"] == 1
    assert rep["grounding_ratio"] == 0.25
    assert rep["n_prose_only"] == 1 and rep["n_unsourced"] == 2   # search 与 d3


def test_the_chain_is_walkable_and_its_depth_is_reported():
    """链之所以是链：走得下去。深度 1 说明每一步都只挂在根上，这和一条真正的推理线不同。"""
    rep = chain_report(_run(
        {"tool": "list_documents", "args": {}},
        {"tool": "search_notes", "args": {"q": "a"},
         "because": {"why": "the inventory named this type", "from": {"event": 1}}},
        {"args": {"doc": "d"}, "because": {"why": "the search surfaced it",
                                           "from": {"event": 2}}},
    ))
    assert rep["max_depth"] == 2                       # 3 <- 2 <- 1，两跳
    assert rep["links"][2]["chain"] == [3, 2, 1]


def test_a_cycle_cannot_hang_the_walk():
    """两个事件互指是不可能的（后向规则已经排除），但解析器不能依赖那条规则才不死循环。"""
    rec = _run({"args": {"doc": "a"}}, {"args": {"doc": "b"}})
    rec.trace[0]["because"] = {"why": "x", "from": {"event": 2}}
    rec.trace[1]["because"] = {"why": "y", "from": {"event": 1}}
    rep = chain_report(rec)                            # 不挂起即通过
    assert _status(rep) == [FORWARD_REF, GROUNDED]


def test_an_empty_run_reports_no_ratio_rather_than_zero():
    """0/0 报成 0.0，读起来是"完全没有接地"，而事实是"没有可判断的调用"。"""
    rep = chain_report(_run())
    assert rep["n_links"] == 0 and rep["grounding_ratio"] is None
