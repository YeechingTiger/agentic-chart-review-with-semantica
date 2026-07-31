"""评测卡只能教怎么找原因，不能教怎么判分。

判分是 `evals.py` 的确定性函数的事：correctness 就是 `==`。让 AI 判分会洗白弃答——
病历里确实没写、机器人正确地答了 EVIDENCE_INSUFFICIENT，AI judge 却当成"没完成任务"扣分，
优化这种分数等于教模型在最高风险的子群上猜。围栏写在提示词里一次改写就没了；写在这里，
改写就红。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from acr.contract.skills import (
    EVAL_FORBIDDEN_VERBS,
    SkillError,
    eval_skill_judges,
    load_skill_body,
    skill_slot,
)

SKILLS_DIR = Path(__file__).resolve().parents[1] / "assets" / "skills"


def _eval_skills() -> list[str]:
    out = []
    for p in sorted(SKILLS_DIR.iterdir()):
        if not (p / "SKILL.md").is_file():
            continue
        if skill_slot(p.name) == "eval":
            out.append(p.name)
    return out


def test_there_is_at_least_one_eval_skill():
    assert _eval_skills(), "no skill declares slot: eval"


@pytest.mark.parametrize("name", _eval_skills())
def test_eval_skill_declares_what_it_may_judge(name: str):
    judges = eval_skill_judges(name)
    assert judges, f"{name}: `judges` is required and non-empty"
    assert all(isinstance(j, str) and j.strip() for j in judges)


@pytest.mark.parametrize("name", _eval_skills())
def test_eval_skill_does_not_instruct_scoring(name: str):
    """一张教 AI 宣布对错的卡，就是把判分从程序挪回了模型。"""
    body = load_skill_body(name).lower()
    hits = [v for v in EVAL_FORBIDDEN_VERBS if re.search(rf"\b{re.escape(v)}\b", body)]
    assert not hits, (
        f"{name}: eval skills diagnose, they do not score. Found {hits}. "
        f"Ask the deterministic scorer instead — it is exposed as a read-only tool.")


def test_a_scoring_instruction_is_caught(tmp_path: Path):
    d = tmp_path / "eval-bad"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eval-bad\ndescription: x\nslot: eval\njudges: [search_behaviour]\n---\n\n"
        "Decide whether the answer is correct and mark it as such.\n", encoding="utf-8")
    body = load_skill_body("eval-bad", tmp_path).lower()
    assert any(re.search(rf"\b{re.escape(v)}\b", body) for v in EVAL_FORBIDDEN_VERBS)


def test_missing_judges_raises(tmp_path: Path):
    d = tmp_path / "eval-nojudge"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eval-nojudge\ndescription: x\nslot: eval\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError, match="declares no `judges`"):
        eval_skill_judges("eval-nojudge", tmp_path)


def test_judges_on_a_non_eval_skill_raises(tmp_path: Path):
    d = tmp_path / "not-eval"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: not-eval\ndescription: x\nslot: general\njudges: [x]\n---\n\nbody\n",
        encoding="utf-8")
    with pytest.raises(SkillError, match="slot 'general'.*judges"):
        eval_skill_judges("not-eval", tmp_path)
