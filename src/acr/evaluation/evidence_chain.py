"""因果链：把 `because` 从一句读给人看的话，变成一条走得通的指针。

问题
----
`because` 的 schema 说得很清楚：*"Recorded, never checked; it is how a later reader tells your
reasoning from theirs."* 所以 `detect_uncaused_reads` 只能数**有没有**。一句编造的理由和一句
真的理由在记录里长得一模一样，而"这一步为什么发生"是归因报告全部结论所依赖的东西。

一组注解和一条链，差别只在一件事：**标签是不是按 ID 指向另一个产物的可解析指针**。trace 事件
本来就带 `seq`，锚点是现成的，所以这里不发明新的标识体系 —— 只是把 `because` 允许写成

    {"why": "<散文，保留>", "from": {"event": 14}}

散文留着，因为它才是告诉后来者推理过程的东西；指针是新加的，因为只有它能被核对。

它不是门
--------
解析失败**不拒绝任何东西**。这个仓库测量过把判断变成机械门的代价：五条临床检查，254 次
拒绝里 60 次（24%）拒掉的元组恰好是登记面自己的答案。所以这里的产物是一份报告和几个数字，
和 `detect_uncaused_reads` 的立场一致 —— 数出来，交给读的人。

五种状态，为什么不是三种
------------------------
`PROSE_ONLY` 必须和 `GROUNDED`、`UNSOURCED` 都分开：并进前者会把无法核对的说成已核对，
并进后者会把历史上所有认真写了理由的运行记成没写。已记录的每一次运行都是散文形状。

`FORWARD_REF` 必须和 `UNRESOLVED_REF` 分开：指错了可能是笔误，**指向一个当时还不存在的事件
不可能是笔误** —— 那是动作先做、理由后补才会出现的形状，而且它是这套改动带来的唯一一个纯
确定性的新检测器。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: 指针解析通了：指向一个真实存在、且排在自己**前面**的事件。
GROUNDED = "GROUNDED"
#: 写了理由但没给指针 —— 包括历史上所有字符串形态的 `because`。不是失败。
PROSE_ONLY = "PROSE_ONLY"
#: 没有 because。
UNSOURCED = "UNSOURCED"
#: 给了指针但解析不到：seq 不存在，或标签本身坏了。
UNRESOLVED_REF = "UNRESOLVED_REF"
#: 指向自己或更晚的事件。不可能，因为那时它还没发生。
FORWARD_REF = "FORWARD_REF"

#: 允许携带 `because` 的调用种类。`kind == "tool"` 就够了：哪些工具值得解释是策略问题，
#: 而这里只报告事实 —— 把范围收窄到读取会让"检索为什么发的"这件事无法被计入。
_TOOL_KIND = "tool"

#: 走链时的上限。后向规则本已排除环，但解析器不能依赖另一条规则才不死循环 —— 一个坏记录
#: 不该让评测挂起。
_MAX_WALK = 1000


def _pointer(because: Any) -> int | None | str:
    """从 `because` 里取出被指向的 seq。

    返回 int（有指针）、None（没有指针，是散文）、或 `UNRESOLVED_REF`（有指针但坏了）。
    坏标签必须变成一条记录而不是一次异常：评测跑的是别人已经产生的运行，而那些运行不会
    因为格式错误就重跑一遍。
    """
    if not isinstance(because, Mapping):
        return None                                    # 字符串 / 其它 —— 散文
    ref = because.get("from")
    if ref is None:
        return None
    if not isinstance(ref, Mapping):
        return UNRESOLVED_REF
    raw = ref.get("event")
    if raw is None:
        return UNRESOLVED_REF
    try:
        return int(raw)
    except (TypeError, ValueError):
        return UNRESOLVED_REF


def _as_seq(raw: Any) -> int | str:
    """扁平 `after_event` 的取值。坏值走和坏标签同一条路：变成记录，不变成异常。"""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return UNRESOLVED_REF


def _why(because: Any) -> str:
    if isinstance(because, Mapping):
        return str(because.get("why") or "")
    return str(because or "")


def chain_report(run) -> dict:
    """每一次工具调用的因果链状态，加上挂评测用的三个数。

    `run` 是一个 `RunRecord`；只读它的 trace。
    """
    events = [ev for ev in (run.trace or []) if ev.get("kind") == _TOOL_KIND]
    by_seq: dict[int, dict] = {}
    for ev in events:
        try:
            by_seq[int(ev.get("seq"))] = ev
        except (TypeError, ValueError):
            continue

    links: list[dict] = []
    for ev in events:
        try:
            seq = int(ev.get("seq"))
        except (TypeError, ValueError):
            seq = None
        because = ev.get("because")
        # 扁平字段优先。嵌套形态保留只为读懂用第一版 schema 跑出来的记录 —— 那一版实测
        # 执行率 0/18，所以它不会有多少数据，但读不懂旧记录是另一种损失。
        flat = ev.get("after_event")
        target = _pointer(because) if flat is None else _as_seq(flat)

        if because is None or (isinstance(because, str) and not because.strip()):
            status, ref = UNSOURCED, None
        elif target is None:
            status, ref = PROSE_ONLY, None
        elif target == UNRESOLVED_REF:
            status, ref = UNRESOLVED_REF, None
        elif target not in by_seq:
            # 存在性先判：一个指向不存在 seq 的指针，说它"在后面"是在假装它存在。
            status, ref = UNRESOLVED_REF, target
        elif seq is not None and target >= seq:
            status, ref = FORWARD_REF, target
        else:
            status, ref = GROUNDED, target

        links.append({"seq": seq, "tool": str(ev.get("tool") or ""), "status": status,
                      "why": _why(because), "ref": ref,
                      "chain": _walk(seq, by_seq) if status == GROUNDED else
                               ([seq] if seq is not None else [])})

    n = len(links)
    counts = {s: sum(1 for x in links if x["status"] == s)
              for s in (GROUNDED, PROSE_ONLY, UNSOURCED, UNRESOLVED_REF, FORWARD_REF)}
    return {
        "links": links,
        "n_links": n,
        "n_grounded": counts[GROUNDED],
        "n_prose_only": counts[PROSE_ONLY],
        "n_unsourced": counts[UNSOURCED],
        "n_unresolved": counts[UNRESOLVED_REF],
        "n_forward": counts[FORWARD_REF],
        # None, 不是 0.0 —— 一次没有工具调用的运行不是"完全没有接地"，是没有可判断的调用，
        # 而把两者渲染成同一个数字正是这份报告要消除的那种混淆。
        "grounding_ratio": (counts[GROUNDED] / n) if n else None,
        "max_depth": max((len(x["chain"]) - 1 for x in links), default=0),
    }


def _walk(seq: int | None, by_seq: dict[int, dict]) -> list[int]:
    """从 `seq` 沿指针回溯到根，返回经过的 seq 列表。

    带 `seen` 与硬上限：后向规则本已让环不可能出现，但一份坏记录不该让评测挂起。
    """
    out: list[int] = []
    seen: set[int] = set()
    cur = seq
    for _ in range(_MAX_WALK):
        if cur is None or cur in seen or cur not in by_seq:
            break
        out.append(cur)
        seen.add(cur)
        ev = by_seq[cur]
        flat = ev.get("after_event")
        nxt = _pointer(ev.get("because")) if flat is None else _as_seq(flat)
        if not isinstance(nxt, int) or nxt >= cur:
            break
        cur = nxt
    return out


def _claim_pointer(because: Any) -> tuple[str, int] | None | str:
    """从一条主张的 `because` 里取出锚点：`("event", n)` 或 `("evidence", i)`。

    两种锚点都要支持，因为归因主张有两种自然的依据：一步动作（"它从没搜过这个缩写"）和
    一条已引用的证据（"被引的那一段没提到 behaviour"）。只支持前者会逼调用方把证据主张
    硬塞进事件编号里。
    """
    if not isinstance(because, Mapping):
        return None
    ref = because.get("from")
    if ref is None:
        return None
    if not isinstance(ref, Mapping):
        return UNRESOLVED_REF
    for kind in ("event", "evidence"):
        if kind in ref:
            try:
                return (kind, int(ref[kind]))
            except (TypeError, ValueError):
                return UNRESOLVED_REF
    return UNRESOLVED_REF


def claim_report(claims: list[Mapping], run) -> dict:
    """散文产物里每条主张的接地状态。

    收一个**通用**的 claim 列表 —— 每项至少有 `text`，可选 `because` —— 而不是归因报告的
    类型。这个模块在 evaluation 平面；让它 import diagnosis 的 schema 就是把两个平面焊死，
    而 `tests/test_layering.py` 恰好禁止这件事。调用方负责适配。

    状态与调用层完全相同，理由也相同：`PROSE_ONLY` 既不算已核对也不算没写，而
    `grounding_ratio` 只把真的解析通的算进分子。
    """
    by_seq = {}
    for ev in (run.trace or []):
        if ev.get("kind") != _TOOL_KIND:
            continue
        try:
            by_seq[int(ev.get("seq"))] = ev
        except (TypeError, ValueError):
            continue
    n_evidence = len(run.manifest.get("evidence") or [])

    out: list[dict] = []
    for c in claims:
        because = c.get("because")
        target = _claim_pointer(because)
        if because is None or (isinstance(because, str) and not because.strip()):
            status, ref = UNSOURCED, None
        elif target is None:
            status, ref = PROSE_ONLY, None
        elif target == UNRESOLVED_REF:
            status, ref = UNRESOLVED_REF, None
        else:
            kind, idx = target
            ok = (idx in by_seq) if kind == "event" else (0 <= idx < n_evidence)
            status, ref = (GROUNDED if ok else UNRESOLVED_REF), f"{kind}:{idx}"
        out.append({"text": str(c.get("text") or "")[:200], "status": status,
                    "why": _why(because), "ref": ref})

    n = len(out)
    grounded = sum(1 for c in out if c["status"] == GROUNDED)
    return {"claims": out, "n_claims": n, "n_grounded": grounded,
            "grounding_ratio": (grounded / n) if n else None}
