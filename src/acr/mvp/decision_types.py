"""The taxonomy a finished run is read back through: what KIND of judgment each point was.

This vocabulary is applied **after** a run, by `acr.mvp.reconstruct`, never during one. The
reason is that it is the one part of this system still being grown from real data: asking a
model to classify against an unsettled vocabulary at the moment it acts would force every
judgment that fits nothing into `other` — or worse, into a type that merely looks close — and
the trace would keep only the forced choice. Applied afterwards, a changed taxonomy costs one
re-extraction instead of a re-run. What the runtime *does* collect is in
[`warrants`](warrants.py): what a decision used and where it knew it from, both facts about
the model's state that no reconstruction can recover.

  * `DECISION_TYPES` — thirteen kinds of judgment, in three groups. The list merges two
    sources that each cover half the ground. The perception half is the decision-precipitation
    design's ten families, a faithful decomposition of what a human reviewer decides; the
    synthesis half is THE_IDEAL_SYSTEM's judgment-layer operators (corroboration, arbitration,
    dedup/ordering, derivation), which the ten families deliberately EXCLUDE because that
    document assumes the retrieval operators are deterministic. Here they are not — they are
    judgments. Taking either source alone drops half the review.

  * `outcomes` — what a decision of each kind may conclude. Six kinds are naturally closed
    (`standing`'s three values are CONTEXT.md's own domain language; `is_it_absent`'s are the
    contract's two abstentions plus "found"), which is what makes divergence computable before
    any situation vocabulary has settled. The rest conclude from this run's candidate set.

The GRANULARITY RULE, because it is what keeps this list from growing: **two types split when
their divergences go to different people and change different things.** Routing is the only
reason a taxonomy exists. Fineness does not live here — the full identity of a decision point
is `{level}:{decision_type}:{situation_slug}`, and comparability lives in the slug, which is
open and evolves. This vocabulary is closed and meant to stay still.
"""
from __future__ import annotations

from dataclasses import dataclass

SMALL = "small"   # 小点: decidable within one action
BIG = "big"       # 大点: needs several small steps to reach


@dataclass(frozen=True, slots=True)
class DecisionType:
    group: str                        #: A / B / C — see the module docstring
    about: str                        #: one line: what a decision of this kind settles
    outcomes: tuple[str, ...] | None  #: closed vocabulary, or None for "this run's candidates"
    remedy: str                       #: where a divergence of this kind is fixed


#: A: about ONE document or span — decidable within one action ⇒ small point.
#: B: about the RELATION between two pieces of evidence — needs ≥2 ⇒ big point.
#: C: about what the case can now support — needs the whole picture ⇒ big point.
DECISION_TYPES: dict[str, DecisionType] = {
    # ---- A · one document or span -------------------------------------------------------
    "where_to_look": DecisionType(
        "A", "去哪找、用什么词、开哪一份、要不要扩", None,
        "retrieval prior / 检索卡片（工程，自动回归门）"),
    "is_this_it": DecisionType(
        "A", "这段说的是不是目标概念",
        ("is_target", "not_target", "unclear"),
        "合同的概念定义（专家）"),
    "what_it_asserts": DecisionType(
        "A", "断言了什么：否定 / 病史 / 假设 / 转述 / 计划 vs 已执行",
        ("asserted", "negated", "historical", "hypothetical", "planned", "reported_by_other"),
        "extractor / 先例库（能力）"),
    "when_it_happened": DecisionType(
        "A", "指向哪个时间：记录时间 vs 事件时间、copy-forward",
        ("event_time_stated", "recorded_time_only", "carried_forward", "undatable"),
        "extractor（检测）+ 合同（用哪个时间）"),
    "standing": DecisionType(
        "A", "这份文档对这个字段值多少",
        ("can_establish", "merely_mentions", "neither"),
        "合同的 evidence rules（专家）"),
    # ---- B · the relation between two pieces --------------------------------------------
    "same_or_ordered": DecisionType(
        "B", "两处提及是不是同一件事；两件的话谁先",
        ("same_event", "distinct_a_first", "distinct_b_first", "distinct_order_unknown"),
        "合同的事件同一性规则（专家）"),
    "corroborate": DecisionType(
        "B", "说的一致，强度能不能叠加",
        ("reinforces", "independent_but_same", "not_actually_about_the_same"),
        "合同的证据阶梯（专家）"),
    "which_wins": DecisionType(
        "B", "说的不一致，按哪条规则选一个", None,
        "合同的 Conflict Rule（专家）"),
    # ---- C · what the case can now support ----------------------------------------------
    "scope": DecisionType(
        "C", "案子 / 实体在不在范围内；时间锚用哪个", None,
        "合同的适用性（专家）"),
    "infer": DecisionType(
        "C", "无文档直接断言，从见证前提推出（欠前提逐条 witness + 排除竞争候选）", None,
        "合同的推断政策（专家）"),
    "is_it_absent": DecisionType(
        "C", "没找到意味着什么",
        ("absent_in_chart", "absent_from_corpus", "found"),
        "coverage 卡片（可强制那半归闸门）"),
    "enough": DecisionType(
        "C", "手上的证据够不够作答",
        ("enough", "not_enough"),
        "coverage 卡片（已测死，不可强制）"),
    "what_to_answer": DecisionType(
        "C", "满不满足定义、边界、值规范化、怎么拼", None,
        "合同的字段定义 / 格式 / 组合规则（专家）"),
    # ---- the escape valve -----------------------------------------------------------------
    "other": DecisionType(
        "?", "这张表还没命名的判断", None,
        "先看它堆积成什么样，再决定要不要收进词表"),
}

#: The one type that legitimately occurs at both levels: "which way should the next stretch of
#: looking go" is a big point reached by combining what earlier searches returned, while "what
#: term for this one call" is small. Every other type's level follows from its group.
BOTH_LEVELS = frozenset({"where_to_look"})

STANDING_VALUES = DECISION_TYPES["standing"].outcomes or ()


def group_of(decision_type: str) -> str:
    t = DECISION_TYPES.get(decision_type)
    return t.group if t else "?"


def level_of(decision_type: str, *, on_action: bool) -> str:
    """Derived, never asked of the model.

    `on_action` breaks the one tie: `where_to_look` is small when it rode in on a retrieval
    call and big when it was narrated on its own.
    """
    if decision_type in BOTH_LEVELS:
        return SMALL if on_action else BIG
    return SMALL if group_of(decision_type) == "A" else BIG


def types_for(level: str) -> tuple[str, ...]:
    """Which types may be claimed at this level. `other` is always allowed — refusing the
    escape valve would only push unnamed judgments into a wrong name."""
    out = [name for name, t in DECISION_TYPES.items()
           if name != "other"
           and (name in BOTH_LEVELS or level_of(name, on_action=(level == SMALL)) == level)]
    return (*out, "other")


def normalize_type(claimed: str | None) -> tuple[str, str | None]:
    """(canonical type, preserved claim if it was not canonical)."""
    t = (claimed or "").strip()
    if t in DECISION_TYPES:
        return t, None
    return "other", (t or None)


def _lines(pairs) -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in pairs)


def action_type_lines() -> str:
    return _lines((n, DECISION_TYPES[n].about) for n in types_for(SMALL) if n != "other")


def note_type_lines() -> str:
    return _lines((n, DECISION_TYPES[n].about) for n in types_for(BIG) if n != "other")
