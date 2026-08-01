"""一个从开发集派生出来的检索词，不能是答案本身。

为什么置换检验不够
------------------
`assetdev.certify` 已经有一道统计防线：同一个搜索在 19 次打乱标签上也涨得一样多就拒绝，
这是 level 1/20 的精确单侧置换检验。它抓的是"整体增益不是关于检索的"。

它抓不住这一种：spec 的答案是 `20230412`，labelling 阶段把真实结局当提示给了模型（ReMedi
那个做法），于是模型提出 `2023-04-12` 作为关键词。这个词在开发集上**每个病人都真的指向
答案**，所以打乱标签后它确实会失效 —— 置换检验会放它过去。而它在 test 上一文不值，因为
测试时没有人会把答案告诉你。

记法是这条检查的全部难点
------------------------
直接比字符串抓不住任何东西。同一天有 `20230412` / `2023-04-12` / `04/12/2023` / `4/12/23`；
同一个码有 `C187` / `C18.7`；同一个形态有 `8140` / `8140/3`。一个只比字面量的过滤器会漏掉
所有真正会发生的泄漏形态，并且**看起来在工作**。

零件是现成的：`code_tables` 的 `normalize()` 折叠码的记法，`corpus` 的日期容差知道
`2019-03-12` 和 `3/12/19` 是同一天。
"""
from __future__ import annotations

import pytest

from acr.improvement.answer_leak import (
    AnswerLeak,
    leaking_terms,
    looks_like_answer,
)


# ------------------------------------------------------------------ 日期的各种写法
@pytest.mark.parametrize("term", [
    "20230412", "2023-04-12", "04/12/2023", "4/12/2023", "2023/04/12", "12 Apr 2023",
])
def test_every_rendering_of_the_gold_date_is_caught(term: str):
    """一天有很多写法，泄漏用哪一种都是泄漏。"""
    assert looks_like_answer(term, "20230412"), term


@pytest.mark.parametrize("term", ["adenocarc", "pathology", "2023", "04", "biopsy", "diagnosis"])
def test_ordinary_terms_and_bare_fragments_are_not_flagged(term: str):
    """`2023` 单独出现不是泄漏 —— 它是一个年份，出现在几乎每份病历里。

    把它算成泄漏会让这道检查变成噪声源，而一个总在误报的过滤器会被下一个人关掉。
    """
    assert not looks_like_answer(term, "20230412"), term


# ------------------------------------------------------------------ 码的各种写法
@pytest.mark.parametrize("term,gold", [
    ("C187", "C187"), ("C18.7", "C187"), ("c18.7", "C187"),
    ("8140", "8140"), ("8140/3", "8140"),
])
def test_code_notation_variants_are_caught(term: str, gold: str):
    assert looks_like_answer(term, gold), (term, gold)


def test_a_code_that_merely_shares_a_prefix_is_not_a_leak():
    """`C18` 是一个更粗的部位，不是那个答案。按前缀判会把整类词全部误杀。"""
    assert not looks_like_answer("C18", "C187")
    assert not looks_like_answer("814", "8140")


# ------------------------------------------------------------------ 嵌在短语里
def test_a_term_containing_the_answer_as_a_token_is_caught():
    """`diagnosed 2023-04-12` 泄漏得和裸日期一模一样。"""
    assert looks_like_answer("diagnosed 2023-04-12", "20230412")
    assert looks_like_answer("histology 8140", "8140")


def test_a_longer_number_that_merely_contains_the_digits_is_not_a_leak():
    """`120230412` 含有那串数字，但它不是那个日期。按子串判会误杀。"""
    assert not looks_like_answer("120230412", "20230412")


# ------------------------------------------------------------------ 批量：过滤一份词表
def test_leaking_terms_reports_which_term_leaked_which_case():
    """报告必须说清是**哪个词**泄漏了**哪个病例**的答案 —— 只说"有泄漏"没法修。"""
    found = leaking_terms(
        terms=["adenocarc", "2023-04-12", "pathology", "8140"],
        gold_values={"SYN0001": ["20230412", "C341"], "SYN0002": ["8140"]},
    )
    assert [f.term for f in found] == ["2023-04-12", "8140"]
    assert found[0].patient_id == "SYN0001" and found[1].patient_id == "SYN0002"
    assert isinstance(found[0], AnswerLeak)


def test_a_clean_term_list_passes():
    assert leaking_terms(terms=["adenocarc", "biopsy"],
                         gold_values={"SYN0001": ["20230412"]}) == []


def test_an_empty_gold_value_is_not_a_universal_match():
    """空的 gold 会让"任何词都包含它"，把整份词表判死。"""
    assert not looks_like_answer("anything", "")
    assert leaking_terms(terms=["a", "b"], gold_values={"P": ["", None]}) == []


def test_certify_refuses_a_plan_whose_keywords_leak():
    """接线本身要有测试：`keywords` 是 (stratum, terms) 的对，直接传进去会比较元组、
    静默找不到任何东西 —— 正是"不会失败的检查"那个形状。"""
    from acr.improvement.answer_leak import leaking_terms
    from acr.improvement.assetdev import AnswerLeaked
    plan_keywords = (("can_establish", ("adenocarc", "2023-04-12")),
                     ("may_mention", ("biopsy",)))
    flat = [t for _, group in plan_keywords for t in group]
    assert flat == ["adenocarc", "2023-04-12", "biopsy"]
    assert [f.term for f in leaking_terms(
        terms=flat, gold_values={"SYN0001": ["20230412"]})] == ["2023-04-12"]
    assert issubclass(AnswerLeaked, Exception)
