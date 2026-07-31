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

from acr.skills import SLOTS, SkillError, skill_slot

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
