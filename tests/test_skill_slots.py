"""每张卡必须说清自己装在哪个槽。

装错槽不是小事：`search` 槽的卡是做对照试验时唯一被替换的变量，一张 `task` 卡混进去，
两次 run 的差别就不再只来自检索策略，而结论会照样被写进报告。所以槽位是声明的、
校验的，不是靠目录名猜的。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from acr.skills import (
    SLOTS,
    SkillError,
    SkillStack,
    load_skill_body,
    skill_slot,
    skills_block,
    skills_manifest,
)

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# `skills/guideline-to-rules/SKILL.md` 从未写过——见 tests/test_skills_load.py:49-55。
_NO_SKILL_MD = {"guideline-to-rules"}


def _skill_names() -> list[str]:
    return sorted(p.name for p in SKILLS_DIR.iterdir()
                  if p.is_dir() and p.name not in _NO_SKILL_MD)


@pytest.mark.parametrize("name", _skill_names())
def test_every_skill_declares_a_known_slot(name: str):
    slot = skill_slot(name)
    assert slot in SLOTS, f"{name}: slot {slot!r} not one of {SLOTS}"


@pytest.mark.parametrize("name", _skill_names())
def test_declared_slot_matches_the_file(name: str):
    """skill_slot 读的就是文件里那一行，不是别处推断的。"""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(_FM.match(text).group(1))
    assert fm["slot"] == skill_slot(name)


def test_missing_slot_raises(tmp_path: Path):
    d = tmp_path / "no-slot"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: no-slot\ndescription: x\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError, match="declares no `slot`"):
        skill_slot("no-slot", tmp_path)


def test_unknown_slot_raises(tmp_path: Path):
    d = tmp_path / "bad-slot"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: bad-slot\ndescription: x\nslot: wherever\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError, match="unknown slot 'wherever'"):
        skill_slot("bad-slot", tmp_path)


def test_stack_renders_task_then_search_then_general():
    stack = SkillStack(task="store-icdo-coding", search="keyword-strategy",
                       general=("chart-triage", "coverage-judgement"))
    assert stack.names() == ("store-icdo-coding", "keyword-strategy",
                             "chart-triage", "coverage-judgement")


def test_stack_rejects_a_skill_in_the_wrong_slot():
    """把一张 general 卡塞进 search 槽——这正是会让对照试验失去意义的装配错误。"""
    with pytest.raises(SkillError, match="chart-triage.*declares slot 'general'.*'search'"):
        SkillStack(search="chart-triage").validate()


def test_stack_rejects_an_eval_skill_in_the_chart_agent(tmp_path: Path):
    """`slot: eval` 的卡属于评测那边的 agent，装进跑病历的任何一个槽都是错。

    计划里这条测试直接点名 `eval-contrast-traces`，但那几张卡要到 Task 6 才建出来，现在点名
    它只会撞上"卡不存在"，测不到槽位校验。这里临时造一张 `slot: eval` 的卡，测的是同一件事。
    """
    d = tmp_path / "eval-contrast-traces"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eval-contrast-traces\ndescription: x\nslot: eval\n---\n\nbody\n",
        encoding="utf-8")
    with pytest.raises(SkillError, match="slot 'eval'"):
        SkillStack(general=("eval-contrast-traces",)).validate(tmp_path)


def test_manifest_carries_the_slot():
    entries = skills_manifest(SkillStack(general=("coverage-judgement",)))
    assert [e["slot"] for e in entries] == ["general"]
    assert entries[0]["skill"] == "coverage-judgement"
    assert entries[0]["content_hash"]


def test_default_profile_renders_exactly_what_it_rendered_before():
    """这次改动不许改变默认 profile 送给模型的字节。

    历史上每一次 run 都是在 `coverage-judgement` 这一张卡下跑的。如果重构顺手多塞了两张，
    过去的 run 和以后的 run 就不可比，而存档单不会说这件事——它只记得当时渲染了什么。
    """
    from acr.runtime_profiles import DEFAULT_RUNTIME_PROFILE, runtime_policy_skills
    stack = runtime_policy_skills(DEFAULT_RUNTIME_PROFILE)
    assert stack.names() == ("coverage-judgement",)
    assert skills_block(stack).endswith(load_skill_body("coverage-judgement"))
    # `endswith` only pins the tail. The header and the separators are prompt bytes too, and a
    # reworded header would move every run onto a different prompt while the manifest — which
    # hashes the skill BODY, not the block — went on reporting the same content hash. So the
    # whole block is spelled out here, character for character, as it rendered before slots
    # existed.
    assert skills_block(stack) == "\n".join([
        "METHOD GUIDANCE — JUDGEMENT YOU APPLY, NOT CONDITIONS THE RUNTIME ENFORCES",
        "",
        "Nothing below is checked mechanically. It is how a careful reviewer approaches these "
        "questions, and where it does not fit this chart you should depart from it and say so in "
        "your reasoning. Your departure is recorded, not refused.",
        "",
        "--- skill: coverage-judgement ---",
        "",
        load_skill_body("coverage-judgement"),
    ])


def test_unknown_profile_still_falls_back_to_coverage_judgement():
    from acr.runtime_profiles import runtime_policy_skills
    assert runtime_policy_skills("not-a-profile").names() == ("coverage-judgement",)
