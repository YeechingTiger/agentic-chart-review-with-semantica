"""候选臂接进运行时之后,必须仍然是一个观察者。

Phase A 的整个设计取决于一件事:**开着这条臂的运行,搜索流程和基线一模一样,只多花调用。**
如果 ledger 被渲染回主循环的提示词,或者网关能因为它拒答,那么"这个状态可不可靠"这个问题
就没法在任何东西依赖它之前先回答 —— 而先回答它,正是不立刻上 Strategic Controller 的理由。

这里钉的是权限,不是行为:
  * 主循环的提示词里没有候选块;
  * 网关不接收候选账本;
  * reasoner 崩了,运行照跑;
  * 臂关着的时候,manifest 里是 null,不是空账本 —— 那是两个不同的事实。
"""
from __future__ import annotations

import inspect
import re

import acr.review.agent as A
from acr.core.state import CandidateLedger, Evidence, EvidenceLedger

# --------------------------------------------------------------- 观察者

def test_the_candidate_block_never_enters_the_main_loops_prompt():
    """`wrap_model_call` 组装的是 `parts`。候选账本不在里面。

    Phase A 的核心约束。一旦它进了提示词,这条臂就不再是"基线加一次调用",而是一个干预,
    accuracy 的任何差别都无法归因给"状态变得可观察"这件事。
    """
    src = inspect.getsource(A.AuditMiddleware.wrap_model_call)
    assert "_reason_about_candidates" in src, "reasoner 根本没被调用"
    body = re.sub(r'"""(.*?)"""', "", src, flags=re.DOTALL)
    body = re.sub(r"^\s*#.*$", "", body, flags=re.MULTILINE)
    assert "candidates.render" not in body and "candidates.to_dict" not in body
    for line in body.splitlines():
        if "parts.append" in line or "parts = [" in line:
            assert "candidate" not in line.lower(), line


def test_the_gate_is_not_handed_the_candidate_ledger():
    """结构性模块不能变成闸门。网关的签名就是这条界线。"""
    from acr.review.answer_gate import gate_answer
    assert "candidates" not in inspect.signature(gate_answer).parameters


def test_the_reasoner_runs_before_the_gate_rules_not_after():
    """否则记下来的候选集是提交前一轮的,而"它带着未解决的分歧提交了吗"问的正是提交那一刻。"""
    src = inspect.getsource(A.AuditMiddleware._gate_answer)
    assert src.index("_reason_about_candidates") < src.index("self.ctx.gate(submitted)")


# --------------------------------------------------------------- 开关

def test_the_arm_is_off_by_default_everywhere():
    assert inspect.signature(A.run_patient).parameters["candidates"].default is False
    assert inspect.signature(A.run_chart_review).parameters["candidates"].default is None


def test_off_and_empty_are_different_facts_in_the_manifest():
    """`null` 是"这条臂没开";`{"candidates": []}` 是"开了,一个都没声明"。

    合并这两者,"账本是空的"就同时吸收了"从没跑过",而空账本率正是验收标准之一。
    """
    src = inspect.getsource(A.run_chart_review)
    assert 'ctx.candidates.to_dict() if ctx.candidates is not None else None' in src


# --------------------------------------------------------------- 不能杀掉运行

def test_a_reasoner_that_raises_leaves_the_run_alone():
    from acr.review.candidate_reasoner import reason
    ev = EvidenceLedger()
    ev.add(Evidence("D1", "t", "2020-01-01", 0, 5, "q", "d"))

    def explode(messages, tools):
        raise RuntimeError("provider is down")

    r = reason(spec_block="c", evidence=ev, ledger=CandidateLedger(), invoke=explode)
    assert r.ok is False and r.updates == []


def test_the_call_is_skipped_when_no_new_evidence_arrived():
    """按证据条数计,不按轮次。连读六份文档的运行否则要为同一个状态付六次钱。"""
    src = inspect.getsource(A.AuditMiddleware._reason_about_candidates)
    assert "candidates_seen_evidence" in src
    assert "if n == 0 or n == ctx.candidates_seen_evidence" in src


# --------------------------------------------------------------- 提交进入账本

def test_a_submitted_value_the_reasoner_never_named_is_still_recorded():
    """"运行提交了一个它自己的候选推理从没考虑过的值"是这个账本能说的最有用的一句话,
    而它只有在提交值也进了账本、可以被比对时才看得见。"""
    src = inspect.getsource(A.AuditMiddleware._select_submitted)
    assert "SELECTED" in src and "never declared" in src


def test_selection_happens_only_on_an_accepted_answer():
    src = inspect.getsource(A.AuditMiddleware._gate_answer)
    i_acc = src.index('if verdict.get("accepted"):')
    assert src.index("_select_submitted") > i_acc


# --------------------------------------------------------------- 成本记在同一本账上

def test_the_reasoner_uses_the_runs_own_model():
    """另起一个 client 会让这条臂的 token 落在别处,而成本增量是要报告的指标之一。"""
    src = inspect.getsource(A.run_chart_review)
    assert "reasoner_model=model" in src


def test_every_call_is_recorded_whether_or_not_it_produced_anything():
    """`ok=False` 和"说了只有一个候选"在 manifest 里必须长得不一样。"""
    src = inspect.getsource(A.AuditMiddleware._reason_about_candidates)
    assert "candidate_calls.append" in src
    assert 'tracer.emit("candidate_reasoner"' in src


# --------------------------------------------------------------- 真的跑一遍这条路径

def test_the_reasoner_path_actually_executes_against_a_real_run_context():
    """上面十一条测试全是结构性的,没有一条执行过 `_reason_about_candidates` 的函数体。

    第一版在里面读 `ctx.evidence` —— 那个字段不存在,证据账本挂在 toolbox 上 —— 于是每一次
    运行都在第一轮 AttributeError。是一次真实的冒烟运行抓到的,不是测试。所以这里补一条会
    真的调用它的:桩掉模型,别的都用真的。
    """
    from pathlib import Path

    from acr.chartstore.corpus import Corpus
    from acr.contract.spec import load_spec
    from acr.contract.trace import Tracer
    from acr.core.state import CandidateLedger, EvidenceLedger
    from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec
    from acr.review.coverage_planner import OpenThreadLedger, load_marker_catalogue, plan_from_spec
    from acr.review.tools import Toolbox

    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml")
    chart = Corpus(root / "corpus" / "patients").chart("SYN0002")
    docs, _ = chart.list_documents(limit=100_000)
    ev = EvidenceLedger()
    tb = Toolbox(chart, ev, CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7)),
                 spec=spec)
    ctx = A.RunContext(spec=spec, chart=chart, plan=plan_from_spec(spec, chart),
                       coverage=tb.coverage, threads=OpenThreadLedger(),
                       catalogue=load_marker_catalogue(),
                       tracer=Tracer.create(Path("/tmp") / "acr-candidate-tests"),
                       gate=lambda s: {"accepted": False}, toolbox=tb,
                       candidates=CandidateLedger())

    class FakeModel:
        def bind_tools(self, tools, tool_choice=None):
            assert tool_choice == "update_candidates"
            return self
        def invoke(self, messages):
            return {"tool_calls": [{"name": "update_candidates", "args": {
                "candidate_updates": [{"action": "create", "value": {"d": "20200101"},
                                       "supports": ["E1"]}],
                "unresolved_discriminators": []}}]}

    ctx.reasoner_model = FakeModel()
    mw = A.AuditMiddleware(ctx)

    # No evidence yet: the call must be skipped, not attempted.
    mw._reason_about_candidates(why="new_evidence")
    assert ctx.candidate_calls == [] and ctx.candidates.candidates == []

    # Record one span through the real tool, so it gets a real evidence_id.
    d = docs[0]
    tb.dispatch("record_evidence", {"note_id": d.note_id, "start": 0, "end": 20,
                                    "supports": "date_of_initial_diagnosis"})
    assert ev.items and ev.items[0].evidence_id == "E1"

    mw._reason_about_candidates(why="new_evidence")
    assert len(ctx.candidates.candidates) == 1
    assert ctx.candidates.candidates[0].supporting_evidence_ids == ("E1",)
    assert ctx.candidate_calls[0]["ok"] is True and ctx.candidate_calls[0]["refused"] == []

    # Called again with nothing new: no second call.
    mw._reason_about_candidates(why="new_evidence")
    assert len(ctx.candidate_calls) == 1
