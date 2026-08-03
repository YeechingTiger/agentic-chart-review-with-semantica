"""检索一次接受多个关键词，而且每个词的命中要分开报。

为什么这是工具的事而不是策略的事
--------------------------------
E2 实测：`search-breadth-first` 在十八张图上发了 506 次检索，`biopsy` 一个词八十六次。当时
读成"这张卡教模型做了无用功"，但那只是一半 —— 另一半是**工具一次只收一个词**，所以"用五个
词覆盖这份病历"在记录里必然是五次调用，无论策略怎么写。

把词表放进一次调用，同样的覆盖成本从 N 次变成 1 次。这不改变任何策略，它把"扫得宽"从一个
预算问题变回一个检索问题 —— 而 E2 的另一半发现是 breadth-first 打开的文档反而比空槽更少
（2.6 对 3.3），因为预算烧在了发检索上。

每个词的命中必须分开
--------------------
合并成一个池子会丢掉唯一有用的那条信息：**哪个词把这份文档捞出来的**。E2 的链里
`read ←9` 和 `read ←10` 之所以能区分是哪次检索起的作用，靠的就是每次检索是独立事件。
一次多词调用如果把结果拍平，就把这条信息又丢回去了。
"""
from __future__ import annotations

import pytest

from acr.chartstore.corpus import Corpus
from acr.core import site

CORPUS = str(site.corpus_root())


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0001")


def test_a_single_string_still_works(chart):
    """向后兼容：已记录的每一次运行都传字符串。"""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, "adenocarc", max_hits=25)
    assert out["terms"] == ["adenocarc"]
    assert out["by_term"]["adenocarc"]["n_hits"] >= 1


def test_several_terms_in_one_call(chart):
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, ["adenocarc", "biopsy", "zzznotathing"], max_hits=25)
    assert out["terms"] == ["adenocarc", "biopsy", "zzznotathing"]
    assert set(out["by_term"]) == {"adenocarc", "biopsy", "zzznotathing"}
    assert out["by_term"]["zzznotathing"]["n_hits"] == 0


def test_hits_stay_attributed_to_the_term_that_found_them(chart):
    """合并成一个池子会丢掉"哪个词捞出这份文档"，而那正是因果链里唯一有用的区分。"""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, ["adenocarc", "biopsy"], max_hits=25)
    for term, block in out["by_term"].items():
        for h in block["hits"]:
            assert "note_id" in h and "start" in h, term


def test_the_cap_is_per_term_not_shared(chart):
    """共享上限会让第一个词吃掉预算，后面的词看起来像"这份病历里没有"。"""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, ["a", "e"], max_hits=3)
    for block in out["by_term"].values():
        assert block["n_hits"] <= 3


def test_an_empty_term_list_is_refused_not_silently_empty(chart):
    """空词表返回"零命中"读起来像"这份病历什么都没有"。"""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, [], max_hits=5)
    assert out.get("error")
