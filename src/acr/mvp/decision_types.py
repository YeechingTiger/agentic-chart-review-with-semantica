"""Post-run chart-review decision taxonomy.

The runtime records taxonomy-neutral Decision Testimony and sealed receipts. Reconstruction applies
this vocabulary afterwards, so a taxonomy change requires only another read of the same Langtrace
trace rather than another chart review. Functions describe what semantic question was answered;
subjects describe what the atomic choice acted on.

The granularity rule is operational: split two functions when their divergences route to different
owners or require different remedies. The ``other`` escape valve preserves unnamed judgments until
repeated runs justify a stable new type.
"""
from __future__ import annotations

from dataclasses import dataclass

DECISION_TAXONOMY_SCHEMA = "acr.chart_review_decision_taxonomy.v1"

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
        "retrieval guideline / instrumentation / model behavior"),
    "is_this_it": DecisionType(
        "A", "这段说的是不是目标概念",
        ("is_target", "not_target", "unclear"),
        "合同的概念定义（专家）"),
    "what_it_asserts": DecisionType(
        "A", "断言了什么：否定 / 病史 / 假设 / 转述 / 计划 vs 已执行",
        ("asserted", "negated", "historical", "hypothetical", "planned", "reported_by_other"),
        "extraction prompt or model capability"),
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
        "coverage guideline or absence rule"),
    "enough": DecisionType(
        "C", "手上的证据够不够作答",
        ("enough", "not_enough"),
        "coverage guideline or stopping rule"),
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


#: The object of one atomic choice.  ``decision_function`` says what semantic question was
#: answered; this says what the choice acted on.  In particular, it keeps note-type selection,
#: a precommitted keyword batch, and selective note opening comparable without pretending they
#: are the same retrieval situation.
DECISION_SUBJECTS: dict[str, str] = {
    "retrieval_inventory": "whether/how to establish the available-note inventory",
    "retrieval_source": "which note type or source family to include",
    "retrieval_query_batch": "one keyword/query batch committed before observing its results",
    "retrieval_document_set": "which surfaced note or note set to open",
    "evidence_item": "one note or evidence span for one Field",
    "evidence_relationship": "the relation among two or more evidence candidates",
    "case_scope": "whether the case/entity/time anchor is in scope",
    "case_inference": "a case-level inference from witnessed premises",
    "case_absence": "what an unsuccessful search establishes",
    "case_sufficiency": "whether the current case state is sufficient to continue or answer",
    "answer_selection": "the answer value or abstention to return",
    "other": "a material choice whose subject is not yet named",
}

_SUBJECTS_BY_FUNCTION: dict[str, frozenset[str]] = {
    "where_to_look": frozenset({
        "retrieval_inventory", "retrieval_source", "retrieval_query_batch",
        "retrieval_document_set", "evidence_item",
    }),
    "is_this_it": frozenset({"evidence_item"}),
    "what_it_asserts": frozenset({"evidence_item"}),
    "when_it_happened": frozenset({"evidence_item"}),
    "standing": frozenset({"evidence_item"}),
    "same_or_ordered": frozenset({"evidence_relationship"}),
    "corroborate": frozenset({"evidence_relationship"}),
    "which_wins": frozenset({"evidence_relationship"}),
    "scope": frozenset({"case_scope"}),
    "infer": frozenset({"case_inference"}),
    "is_it_absent": frozenset({"case_absence"}),
    "enough": frozenset({"case_sufficiency"}),
    "what_to_answer": frozenset({"answer_selection"}),
    "other": frozenset({"other"}),
}


def subjects_for(decision_function: str) -> frozenset[str]:
    """Return coherent subjects while preserving both taxonomy escape valves.

    A reader may know that a choice acts on one evidence item without yet having a settled
    function name, or know the function while the subject vocabulary is incomplete.  Forcing
    either known half to ``other`` would throw away useful similarity information.
    """
    if decision_function == "other":
        return frozenset(DECISION_SUBJECTS)
    allowed = _SUBJECTS_BY_FUNCTION.get(decision_function, frozenset())
    return frozenset({*allowed, "other"}) if allowed else frozenset()


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
