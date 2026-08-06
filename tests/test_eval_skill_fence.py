"""An eval skill may only teach how to find a cause. It may not teach how to score.

Scoring is the business of the deterministic functions in `evals.py`: correctness is `==`. Letting
an AI score whitewashes abstention — the chart genuinely does not say so, the agent correctly
answered EVIDENCE_INSUFFICIENT, and the AI judge marks it down as "did not finish the task".
Optimising against a score like that is teaching the model to guess on exactly the subpopulation
where guessing is most dangerous. A fence written into the prompt is gone after one rewording;
written here, a rewording turns red.
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
    """A card that teaches the AI to declare right or wrong has moved scoring back out of the
    program and into the model."""
    body = load_skill_body(name).lower()
    hits = [v for v in EVAL_FORBIDDEN_VERBS if re.search(rf"\b{re.escape(v)}\b", body)]
    assert not hits, (
        f"{name}: eval skills diagnose, they do not score. Found {hits}. "
        f"Ask the deterministic scorer instead — it is exposed as a read-only tool.")


def test_a_scoring_instruction_is_caught(tmp_path: Path):
    d = tmp_path / "eval-bad"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eval-bad\ndescription: x\nmetadata:\n  slot: eval\n  judges: [search_behaviour]\n---\n\n"
        "Decide whether the answer is correct and mark it as such.\n", encoding="utf-8")
    body = load_skill_body("eval-bad", tmp_path).lower()
    assert any(re.search(rf"\b{re.escape(v)}\b", body) for v in EVAL_FORBIDDEN_VERBS)


def test_missing_judges_raises(tmp_path: Path):
    d = tmp_path / "eval-nojudge"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eval-nojudge\ndescription: x\nmetadata:\n  slot: eval\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError, match="declares no `judges`"):
        eval_skill_judges("eval-nojudge", tmp_path)


def test_judges_on_a_non_eval_skill_raises(tmp_path: Path):
    d = tmp_path / "not-eval"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: not-eval\ndescription: x\nmetadata:\n  slot: general\n  judges: [x]\n---\n\nbody\n",
        encoding="utf-8")
    with pytest.raises(SkillError, match="slot 'general'.*judges"):
        eval_skill_judges("not-eval", tmp_path)
