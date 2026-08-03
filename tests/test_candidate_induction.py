"""候选从哪里来:先机械地抽,再让模型判断。

A1 证明了两件事,而它们一起决定了这一层的形状:

  1. candidate ledger 这个机制成立 —— 10/10 有账本,0 次编造 id,0 次调用失败;
  2. 纯 LLM reasoner 不会稳定地主动枚举竞争值 —— 13 个候选里没有一个是"值 A 对值 B",
     三次出现第二候选每一次都是弃权。SYNY03 上证据里有三个合格日期,它声明了一个候选、
     零条反对证据。

所以问题不再是 schema 也不是调用时机,而是"候选从哪里来"。之前 reasoner 被压了两件事:
从证据里**发现**所有可能值,以及在这些值之间**推理**。第一件对日期这类目标基本是机械的。

这一层做第一件,而且**故意过度包含**:凡是证据里出现的、与目标类型相容的值都 seed 成候选。
筛选是 reasoner 的活,而且是有记录的活 —— "看到了这个日期、因为它是文档自己的日期而排除"
比"从来没考虑过"是完全不同的一件事,后者在账本里看不出来。

弃权不是候选。"日期 A 对 EVIDENCE_INSUFFICIENT"和"日期 A 对日期 B"是两种不同的分歧,
把它们放进同一个集合会让 conflict 这个概念在两种意义之间摇摆 —— A1 的账本正是这样,
三次"多候选"全部是前者。所以 answerability 单独放。
"""
from __future__ import annotations

import pytest

from acr.core.state import CandidateLedger, Evidence, EvidenceLedger
from acr.review.candidate_induction import (
    ANSWERABILITY,
    extract_values,
    normalise_date,
    seed_candidates,
)


class _Field:
    def __init__(self, name, calendar=None, fmt=None):
        self.name, self.calendar, self.format = name, calendar, fmt


class _Spec:
    spec_id = "TEST.dates"
    def __init__(self, fields):
        self.fields = fields


DATE_SPEC = _Spec([_Field("date_of_initial_diagnosis", calendar="partial_date_ccyymmdd"),
                   _Field("year_imputed"), _Field("month_imputed")])


def _ev(quote, note="D1", date="2020-01-01") -> Evidence:
    return Evidence(note, "Progress-Note", date, 0, len(quote), quote,
                    "date_of_initial_diagnosis")


def _ledger(*quotes, doc_date="2020-01-01") -> EvidenceLedger:
    """所有 span 共用一个文档日期,所以它在候选集里只多出一个值,而不是每条一个。

    文档日期本来就会被 seed(见 `test_the_documents_own_date_is_seeded_too`),这是设计;
    下面的断言里 `DOC` 就是它。
    """
    led = EvidenceLedger()
    for q in quotes:
        led.add(_ev(q, note=f"D{len(led.items)}", date=doc_date))
    return led


#: `_ledger` 默认文档日期归一化之后的样子。
DOC = "20200101"


# --------------------------------------------------------------- 归一化

@pytest.mark.parametrize("raw,want", [
    ("2010-05-17", "20100517"),
    ("05/17/2010", "20100517"),
    ("5/17/2010", "20100517"),
    ("May 17, 2010", "20100517"),
    ("17 May 2010", "20100517"),
    ("May 2010", "20100599"),
    ("in 2010", "20109999"),
])
def test_the_notations_this_corpus_actually_writes_all_normalise(raw, want):
    assert normalise_date(raw) == want


def test_a_two_digit_year_is_refused_rather_than_guessed():
    """`05/17/10` 是 2010 还是 1910?没有上下文就没有答案,猜一个会静默地造出一个候选。"""
    assert normalise_date("05/17/10") is None


def test_an_impossible_date_does_not_become_a_candidate():
    """`2018-02-29` 不是日期。让它进候选集会让 precision 去为一个不存在的读法背锅。"""
    assert normalise_date("2018-02-29") is None
    assert normalise_date("2010-13-01") is None


def test_a_bare_number_that_looks_like_a_year_is_not_a_date_on_its_own():
    """"cycle 2010"、"qty 2010" —— 裸数字太便宜了。要有 `in`/`of`/`since` 这类词带着。"""
    assert normalise_date("2010") is None


# --------------------------------------------------------------- 抽取

def test_every_date_in_the_evidence_is_seeded_not_only_the_likely_one():
    """故意过度包含。筛掉是 reasoner 的活,而且是有记录的活。

    "看到了这个日期、因为它是文档自己的日期而排除"和"从来没考虑过它",在账本里必须长得
    不一样 —— 而 A1 的结果正是后者:三个合格日期在证据里,一个候选在账本里。
    """
    ev = _ledger("Cytology 2010-05-17 suspicious for carcinoma.",
                 "Core biopsy 2010-05-22 positive.",
                 "Referred; the diagnosis was made on 2010-06-01 per the referring service.")
    got = extract_values(DATE_SPEC, ev)
    assert {v for vs in got.values() for v in vs} == \
        {"20100517", "20100522", "20100601", DOC}


def test_one_span_may_carry_two_dates():
    ev = _ledger("Cytology 2010-05-17 was suspicious; biopsy on 2010-05-22 confirmed.")
    assert set(extract_values(DATE_SPEC, ev)["E1"]) == {"20100517", "20100522", DOC}


def test_a_field_that_declares_no_extractable_type_yields_nothing():
    """`C\\d{3}` 是一个编码域,不是日期。对它跑日期抽取只会造出噪声候选。"""
    spec = _Spec([_Field("primary_site", fmt="C\\d{3}")])
    assert extract_values(spec, _ledger("Mass in the lung, coded C341 on 2010-05-17.")) == {}


def test_an_inadmissible_span_is_not_extracted_from():
    """契约说不算证据的东西,不该 seed 出候选来 —— 那会把 precision 的分母灌水。"""
    ev = EvidenceLedger()
    e = _ev("Imaging 2010-05-17 suspicious for malignancy.")
    e.admissibility = "INADMISSIBLE"
    ev.add(e)
    assert extract_values(DATE_SPEC, ev) == {}


def test_an_unjudged_span_is_extracted_from():
    """默认是 UNJUDGED,不是 INADMISSIBLE。还没人判过 ≠ 判过不合格。"""
    ev = _ledger("Cytology 2010-05-17 suspicious.")
    assert ev.items[0].admissibility == "UNJUDGED"
    assert extract_values(DATE_SPEC, ev)["E1"] == ["20100517", DOC]


# --------------------------------------------------------------- seeding

def test_seeding_creates_one_candidate_per_distinct_value_with_its_provenance():
    ev = _ledger("Cytology 2010-05-17 suspicious.", "Biopsy 2010-05-22 positive.")
    led = CandidateLedger()
    res = seed_candidates(led, DATE_SPEC, ev, step=1)
    assert {c.value["date_of_initial_diagnosis"] for c in led.candidates} == \
        {"20100517", "20100522", DOC}
    for c in led.candidates:
        assert c.seed_method == "evidence_value_extraction"
        assert c.seeded_from, "一个 seed 出来的候选必须说得出它是从哪条证据来的"
    assert res.n_seeded == 3


def test_the_same_value_from_two_spans_is_one_candidate_with_two_sources():
    ev = _ledger("Biopsy 2010-05-22 positive.", "Path report dated 2010-05-22.")
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, ev, step=1)
    c = next(x for x in led.candidates if x.value["date_of_initial_diagnosis"] == "20100522")
    assert set(c.seeded_from) == {"E1", "E2"}
    assert set(c.supporting_evidence_ids) == {"E1", "E2"}


def test_seeding_a_second_time_adds_without_disturbing_what_was_judged():
    """增量稳定性。第二批证据进来时,reasoner 已经做出的判断不能被 seeding 抹掉。"""
    ev = _ledger("Cytology 2010-05-17 suspicious.")
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, ev, step=1)
    led.set_state("C1", "REJECTED", step=2, reason="它是文档自己的日期")
    ev.add(_ev("Biopsy 2010-05-22 positive.", note="D9"))
    seed_candidates(led, DATE_SPEC, ev, step=3)
    assert {c.value["date_of_initial_diagnosis"] for c in led.candidates} == \
        {"20100517", "20100522", DOC}
    assert led.by_id("C1").status == "REJECTED", "seeding 复活了一个已被排除的候选"


# --------------------------------------------------------------- 冲突集

def test_two_or_more_seeded_values_form_an_explicit_conflict_set():
    """A1 里没有 conflict 这个对象,所以"存在未解决的分歧"无法被任何东西读到。"""
    ev = _ledger("Cytology 2010-05-17 suspicious.", "Biopsy 2010-05-22 positive.")
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, ev, step=1)
    assert len(led.conflict_sets) == 1
    cs = led.conflict_sets[0]
    assert cs["type"] == "competing_values"
    assert len(cs["candidate_ids"]) == 3            # 两个引文里的日期 + 文档日期


def test_one_value_forms_no_conflict_set():
    """文档日期和引文里的日期是同一天时,只有一个候选。"""
    ev = _ledger("Biopsy 2010-05-22 positive.", doc_date="2010-05-22")
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, ev, step=1)
    assert len(led.candidates) == 1
    assert led.conflict_sets == []


def test_a_rejected_candidate_leaves_the_conflict_set():
    """分歧被解决了就不再是分歧。否则 Controller 会对着一个已经结案的冲突继续搜。"""
    ev = _ledger("Cytology 2010-05-17 suspicious.", "Biopsy 2010-05-22 positive.")
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, ev, step=1)
    for c in list(led.candidates)[1:]:
        led.set_state(c.candidate_id, "REJECTED", step=2, reason="conflict_rule.2")
    assert led.conflict_sets == [], "被排除的候选还留在冲突集里"


# --------------------------------------------------------------- 弃权不是候选

def test_answerability_is_separate_from_the_value_candidates():
    """"日期 A 对弃权"和"日期 A 对日期 B"是两种分歧。放进同一个集合,conflict 这个概念
    就在两种意义之间摇摆 —— A1 的三次"多候选"全部是前者。"""
    led = CandidateLedger()
    assert led.answerability == "UNDETERMINED"
    assert ANSWERABILITY == ("UNDETERMINED", "VALUE_AVAILABLE", "EVIDENCE_INSUFFICIENT",
                             "CORPUS_INSUFFICIENT")
    led.set_answerability("CORPUS_INSUFFICIENT", step=2, reason="记录始于诊断之后")
    assert led.answerability == "CORPUS_INSUFFICIENT"
    assert led.candidates == [], "弃权不该产生一个候选对象"


def test_an_unknown_answerability_is_refused():
    led = CandidateLedger()
    with pytest.raises(ValueError, match="MAYBE"):
        led.set_answerability("MAYBE", step=1)


def test_seeding_a_value_says_the_question_is_answerable():
    """机械事实,不是判断:证据里出现了一个与目标相容的值。"""
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, _ledger("Biopsy 2010-05-22 positive."), step=1)
    assert led.answerability == "VALUE_AVAILABLE"


def test_seeding_nothing_leaves_answerability_undetermined():
    """抽不出值不等于弃权。区分这两者正是 EVIDENCE_INSUFFICIENT 和 CORPUS_INSUFFICIENT
    要分开报的原因,而那是判断,不是抽取的输出。"""
    led = CandidateLedger()
    ev = EvidenceLedger()
    ev.add(Evidence("D1", "Progress-Note", "", 0, 10, "Patient tolerating therapy well.", "d"))
    seed_candidates(led, DATE_SPEC, ev, step=1)
    assert led.answerability == "UNDETERMINED"
    assert led.candidates == []


# --------------------------------------------------------------- 文档自己的日期也要 seed

def test_the_documents_own_date_is_seeded_too():
    """真实运行发现的:一份病历里能定诊断日期的日期,通常在文档头,不在被引用的那句话里。

    SYNY03 上四条证据,只有一条的引文里含日期,而正确答案 20200302 是两份文档自己的日期。
    抽取器只读引文时 seed 出 1 个候选,模型自己补了另外两个 —— 也就是又回到了让模型去
    发现值。这一层的设计是故意过度包含,所以文档日期也进,由 reasoner 去排除。
    """
    ev = EvidenceLedger()
    ev.add(_ev("Atypical cells present, suspicious for carcinoma.", date="2020-03-02"))
    got = extract_values(DATE_SPEC, ev)
    assert got["E1"] == ["20200302"]


def test_a_candidate_records_which_kind_of_source_seeded_it():
    """"文档自己的日期"是 reasoner 最常要排除的一类,precision 必须能和它分开算。"""
    ev = EvidenceLedger()
    ev.add(_ev("Biopsy on 2020-04-10 per the referring service.", date="2020-04-01"))
    led = CandidateLedger()
    seed_candidates(led, DATE_SPEC, ev, step=1)
    kinds = {c.value["date_of_initial_diagnosis"]: set(c.seed_sources) for c in led.candidates}
    assert kinds == {"20200401": {"document_date"}, "20200410": {"quote"}}


def test_an_event_date_is_seeded_when_the_span_carries_one():
    """回溯性陈述:文档日期是 2021,而它说的事发生在 2019。decision_rule[2] 就是这件事。"""
    ev = EvidenceLedger()
    e = _ev("In retrospect the patient had cancer at the time of the earlier scan.",
            date="2021-06-08")
    e.event_date = "2019-03-12"
    ev.add(e)
    assert set(extract_values(DATE_SPEC, ev)["E1"]) == {"20210608", "20190312"}
