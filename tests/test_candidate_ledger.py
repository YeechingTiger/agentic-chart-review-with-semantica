"""竞争答案成为真实对象,以及证据能指向它。

为什么这层必须先于 Strategic Controller 存在
--------------------------------------------
STORE.390 有四条 conflict rule,每一条都在仲裁互相竞争的日期 —— 细胞学的日期对活检的日期,
早先的临床印象对后来的组织确认。而运行时里没有任何东西表示"一个还没被排除的候选日期"。
`RunContext.submitted` 是一个可被覆盖的 dict;`Evidence.supports` 是一段自由文本,不是指针。
所以那些规则只能在散文里被遵守,没被遵守时也说不出丢掉了哪个候选、为什么丢的。

同样的原因让"它是不是带着未解决的分歧就停了"这个问题今天无法回答 —— 而那正是
Strategic Controller 唯一要判断的事。**先有可被检验的候选状态,再有基于它的控制。**

这个账本不拒绝任何东西
----------------------
它只记录。这棵树有一段很重的历史:五条确定性内容检查毁掉 58 个正确值换 21 次帮助,
254 次拒绝里 60 次拒的正是登记处自己的答案 —— 所以覆盖门和线索拒绝都被降成了建议。
结构性模块(记录状态)和决策性模块(改变或阻止输出)是两回事,把它们混在一起是那段历史的根源。
这里是前者。
"""
from __future__ import annotations

import pytest

from acr.core.state import (
    CANDIDATE_STATES,
    Candidate,
    CandidateLedger,
    Evidence,
    EvidenceLedger,
)


def _ev(note="D1", start=0, end=10, supports="date") -> Evidence:
    return Evidence(note, "Progress-Note", "2020-01-01", start, end, "quote", supports)


# --------------------------------------------------------------- evidence gets an identity

def test_evidence_gains_a_stable_id_when_the_ledger_takes_it():
    """候选要指向证据,就得有东西可指。

    `E1`/`E2` 已经是 `EvidenceLedger.render()` 展示给模型的编号,所以这里用的是模型本来
    就看得见的那个标识,不是新造一个它没见过的。
    """
    led = EvidenceLedger()
    led.add(_ev("D1"))
    led.add(_ev("D2", 5, 20))
    assert [e.evidence_id for e in led.items] == ["E1", "E2"]
    assert led.render().startswith("[E1]")


def test_a_de_duplicated_span_does_not_consume_an_id():
    """否则两次记录同一句话会让编号跳号,而跳号读起来像丢了一条证据。"""
    led = EvidenceLedger()
    led.add(_ev("D1"))
    led.add(_ev("D1"))
    assert [e.evidence_id for e in led.items] == ["E1"]


def test_an_evidence_built_by_hand_still_loads():
    """每一条已记录的 trace 里都有 Evidence,新字段不能让它们反序列化失败。"""
    e = Evidence("D1", "t", "2020-01-01", 0, 5, "q")
    assert e.evidence_id == ""
    assert "evidence_id" in e.to_dict()


# --------------------------------------------------------------- the candidate object

def test_a_candidate_carries_what_it_needs_to_be_argued_with():
    c = Candidate(candidate_id="C1", value={"date_of_initial_diagnosis": "20100612"})
    assert c.status == "ACTIVE"
    assert c.supporting_evidence_ids == () and c.contradicting_evidence_ids == ()
    assert c.unresolved_discriminators == ()
    assert c.confidence is None
    assert c.rejection_reason == ""


def test_the_state_vocabulary_is_closed():
    """一个代码不分支的状态,和一个跑过但什么都没找到的状态,读起来一模一样。"""
    assert CANDIDATE_STATES == ("ACTIVE", "LEADING", "REJECTED", "SELECTED")
    led = CandidateLedger()
    with pytest.raises(ValueError, match="MAYBE"):
        led.declare({"d": "1"}, state="MAYBE", step=1)


# --------------------------------------------------------------- the ledger

def test_declaring_the_same_value_twice_updates_rather_than_forks():
    led = CandidateLedger()
    a = led.declare({"d": "20100612"}, step=1)
    b = led.declare({"d": "20100612"}, step=4, state="LEADING")
    assert a.candidate_id == b.candidate_id
    assert len(led.candidates) == 1
    assert led.by_id(a.candidate_id).status == "LEADING"


def test_two_notations_of_one_date_are_two_candidates():
    """按字面值去重,不做语义归一。

    `20100612` 和 `2010-06-12` 在这里是两个候选。把它们合并是记法归一化 —— 这棵树上一个
    独立的、已被钉住的缺陷(`C34.9` 对 `C341`),在没人找它的地方顺手修掉它,是把一个已知
    问题藏进一个新模块。
    """
    led = CandidateLedger()
    led.declare({"d": "20100612"}, step=1)
    led.declare({"d": "2010-06-12"}, step=1)
    assert len(led.candidates) == 2


def test_ids_are_minted_by_the_runtime_not_supplied():
    """模型只能引用它观察得到的标识。

    `after_event` 的第一版让模型自己编调用序号,结果每一个指针都解析不了 —— 同样的规则。
    """
    led = CandidateLedger()
    got = [led.declare({"d": str(i)}, step=1).candidate_id for i in range(3)]
    assert got == ["C1", "C2", "C3"]


def test_linking_evidence_is_incremental_and_deduped():
    led = CandidateLedger()
    c = led.declare({"d": "20100612"}, step=1)
    led.link(c.candidate_id, "E1", "supports", step=2)
    led.link(c.candidate_id, "E1", "supports", step=3)
    led.link(c.candidate_id, "E2", "contradicts", step=3)
    got = led.by_id(c.candidate_id)
    assert got.supporting_evidence_ids == ("E1",)
    assert got.contradicting_evidence_ids == ("E2",)


def test_the_same_quote_may_bear_on_two_candidates():
    """一句"细胞学 2010-06-12 可疑"同时是 A 的证人和 B 要打败的东西。

    多对多是这个关系的真实形状,所以链接住在账本里而不是 `Evidence` 上的一个标量列 ——
    标量列只能对其中一边说真话。
    """
    led = CandidateLedger()
    a = led.declare({"d": "20100612"}, step=1)
    b = led.declare({"d": "20100702"}, step=1)
    led.link(a.candidate_id, "E1", "supports", step=2)
    led.link(b.candidate_id, "E1", "contradicts", step=2)
    assert led.evidence_view()["E1"] == {"supports_candidate_ids": ["C1"],
                                         "contradicts_candidate_ids": ["C2"]}


def test_rejecting_records_a_reason_and_keeps_the_candidate():
    """删掉一个候选就是删掉"考虑过并排除了"和"从没想到"之间的区别。"""
    led = CandidateLedger()
    c = led.declare({"d": "20100702"}, step=1)
    led.set_state(c.candidate_id, "REJECTED", step=5, reason="conflict_rule.3: 更早的临床印象在先")
    got = led.by_id(c.candidate_id)
    assert got.status == "REJECTED"
    assert "conflict_rule.3" in got.rejection_reason
    assert got.updated_at_step == 5
    assert c.candidate_id in [x.candidate_id for x in led.candidates]


def test_state_history_survives_into_the_manifest():
    """manifest 比 trace 活得久。"哪个候选被丢掉了、为什么"必须只读 manifest 就能回答。"""
    led = CandidateLedger()
    c = led.declare({"d": "20100702"}, step=1)
    led.set_state(c.candidate_id, "LEADING", step=3, reason="唯一有见证者的")
    led.set_state(c.candidate_id, "REJECTED", step=7, reason="发现更早的")
    hist = led.to_dict()["candidates"][0]["state_history"]
    assert [(h["from"], h["to"], h["step"]) for h in hist] == \
        [("ACTIVE", "LEADING", 3), ("LEADING", "REJECTED", 7)]


def test_only_one_candidate_may_lead():
    """两个 LEADING 是"更多信息"的反面:它是一个没人写下来的第三个状态。"""
    led = CandidateLedger()
    a = led.declare({"d": "A"}, step=1)
    b = led.declare({"d": "B"}, step=1)
    led.set_state(a.candidate_id, "LEADING", step=2)
    led.set_state(b.candidate_id, "LEADING", step=3)
    assert led.by_id(a.candidate_id).status == "ACTIVE"
    assert led.leading().candidate_id == b.candidate_id


def test_an_unknown_candidate_id_is_refused_rather_than_ignored():
    """静默忽略一个指向不存在候选的链接,读起来和链接成功一模一样。"""
    led = CandidateLedger()
    with pytest.raises(KeyError, match="C9"):
        led.link("C9", "E1", "supports", step=1)


# --------------------------------------------------------------- what it says about itself

def test_an_empty_ledger_and_an_absent_one_are_different_facts():
    led = CandidateLedger()
    d = led.to_dict()
    assert d["candidates"] == [] and d["n_declared"] == 0
    assert led.render() == ""


def test_the_rendered_block_names_the_discriminator():
    led = CandidateLedger()
    a = led.declare({"d": "20100612"}, step=1, state="LEADING")
    b = led.declare({"d": "20100702"}, step=1)
    led.link(a.candidate_id, "E1", "supports", step=1)
    led.link(b.candidate_id, "E2", "supports", step=1)
    led.set_discriminators(["早先那份可疑细胞学在后来的确认之后是否仍然合格"], step=2)
    out = led.render()
    assert "C1" in out and "C2" in out and "LEADING" in out
    assert "仍然合格" in out


# --------------------------------------------------------------- 一条证据不能同时两边站

def test_a_span_cannot_support_and_contradict_the_same_candidate_at_once():
    """真实运行里出现过:E4 同时在 C1 的 for 和 against 里。

    这是一个没有意义的状态。一句话对同一个候选要么是证人要么是反证;两边都记,
    grounding 指标会把同一条证据数两次,而读的人看不出哪一边是它真正的意思。
    """
    led = CandidateLedger()
    c = led.declare({"d": "A"}, step=1)
    led.link(c.candidate_id, "E4", "supports", step=1)
    led.link(c.candidate_id, "E4", "contradicts", step=3)
    got = led.by_id(c.candidate_id)
    assert got.supporting_evidence_ids == ()
    assert got.contradicting_evidence_ids == ("E4",)


def test_changing_a_spans_role_is_recorded_as_a_change():
    """"我先读成支持,现在读成反对"是一次正当的修订,不是一次覆盖。"""
    led = CandidateLedger()
    c = led.declare({"d": "A"}, step=1)
    led.link(c.candidate_id, "E4", "supports", step=1)
    led.link(c.candidate_id, "E4", "contradicts", step=3)
    moves = [e for e in led.events if e["kind"] == "candidate_evidence_rerole"]
    assert moves and moves[0]["evidence_id"] == "E4" and moves[0]["to"] == "contradicts"


def test_a_create_may_declare_itself_leading_in_one_step():
    """否则模型只能引用一个还不存在的 id —— 真实运行里它就编了一个 `candidate_1` 出来。"""
    led = CandidateLedger()
    c = led.declare({"d": "A"}, step=1, state="LEADING")
    assert led.leading().candidate_id == c.candidate_id


# --------------------------------------------------------------- 弃权候选的身份

def test_an_abstention_candidate_is_identified_by_its_status_not_by_its_prose():
    """真实运行里的一个缺陷,而且它盖住了一个真发现。

    SYNX06 上 reasoner 声明了一个弃权候选,label 是 "EVIDENCE_INSUFFICIENT: No document..."。
    运行随后提交了 EVIDENCE_INSUFFICIENT,运行时用裸状态去找它 —— label 不同,于是又建了一个
    候选,标成"提交了但从没声明过"。**同一个弃权被记成两个候选,而且被记成了一次未声明提交。**

    "运行提交了一个它自己的候选推理从没考虑过的值"是这个账本能说的最有用的一句话
    (SYNY04 上就是真的),所以它不能被身份判断的口径差别污染。
    """
    led = CandidateLedger()
    a = led.declare({}, step=1, abstention="EVIDENCE_INSUFFICIENT: No document establishes one")
    b = led.declare({}, step=4, abstention="EVIDENCE_INSUFFICIENT")
    assert a.candidate_id == b.candidate_id
    assert len(led.candidates) == 1
    assert led.candidates[0].abstention == "EVIDENCE_INSUFFICIENT"


def test_the_prose_survives_in_the_label():
    """归一化的是身份,不是内容。模型写的那句话是它的理由,不能丢。"""
    led = CandidateLedger()
    c = led.declare({}, step=1, abstention="CORPUS_INSUFFICIENT: record starts after transfer")
    assert c.abstention == "CORPUS_INSUFFICIENT"
    assert "starts after transfer" in c.label


def test_two_different_abstentions_are_still_two_candidates():
    led = CandidateLedger()
    led.declare({}, step=1, abstention="EVIDENCE_INSUFFICIENT: nothing here")
    led.declare({}, step=1, abstention="CORPUS_INSUFFICIENT: nothing at all")
    assert len(led.candidates) == 2


def test_an_abstention_with_no_recognisable_status_keeps_the_whole_string():
    """没有可辨认的状态标记时不猜。整串就是身份,两个不同的串就是两个候选。"""
    led = CandidateLedger()
    led.declare({}, step=1, abstention="i am not sure")
    led.declare({}, step=1, abstention="i am also not sure")
    assert len(led.candidates) == 2


def test_a_value_is_identified_by_what_it_asserts_not_by_its_empty_fields():
    """真实运行里的重复:seeder 建的候选只有 `date_of_initial_diagnosis`,提交的答案还带着
    `year_imputed: False` 等三个默认假值,于是同一个日期变成两个候选 —— 而且第二个被
    盖上"从没声明过"的章,还把冲突集撑成 2。

    候选的身份是它**主张**了什么。`year_imputed: False` 什么都没主张。
    """
    led = CandidateLedger()
    a = led.declare({"date_of_initial_diagnosis": "20200302"}, step=1)
    b = led.declare({"date_of_initial_diagnosis": "20200302",
                     "year_imputed": False, "month_imputed": False, "day_imputed": False}, step=5)
    assert a.candidate_id == b.candidate_id
    assert len(led.candidates) == 1


def test_a_flag_that_is_set_is_part_of_what_a_candidate_asserts():
    """"1995,年份是估的"和"1995,年份读出来的"是两个不同的主张。"""
    led = CandidateLedger()
    led.declare({"date_of_initial_diagnosis": "20159999"}, step=1)
    led.declare({"date_of_initial_diagnosis": "20159999", "year_imputed": True}, step=1)
    assert len(led.candidates) == 2
