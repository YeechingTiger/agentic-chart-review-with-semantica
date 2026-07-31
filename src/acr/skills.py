"""Load a skill's prose into the system prompt. The missing half of "it lives in a skill".

WHY THIS EXISTS
---------------
`skills/` held six skills and NOTHING READ ANY OF THEM INTO A PROMPT. The only path from that
directory into the runtime was `coverage_planner.load_marker_catalogue`, which parses the
thread-chasing skill's Markdown TABLES into a marker set — it extracts data, not instruction.
`refine.py` treats `skills/*/SKILL.md` as a tunable develop-plane file. `cli_plan` checks a
skills directory only so that a route to a missing skill is reported rather than printed as
advice.

So on 2026-07-30, when the coverage obligation was moved out of `evaluate_gate` and into
`skills/coverage-judgement/SKILL.md`, it was not moved. It was deleted, and the commit message
said moved. That is the mirror image of the failure this project already names — a check that
cannot fail — and it is worse, because "the guidance is in a skill" reads like a design decision
while the model receives nothing.

WHAT REACHES THE MODEL, AND WHAT DOES NOT
-----------------------------------------
The frontmatter `description` is what a skill-aware harness uses to decide whether to open a
skill. Here there is no such harness: the runtime chooses, so the description is dropped and the
BODY is rendered. `name` and `license` are dropped too — neither is an instruction.

Which skills load is the RUNTIME PROFILE's decision and not this module's, because a skill is
retrieval/judgement guidance and swapping it is exactly the kind of change an arm is supposed to
isolate. `agent` passes the names; nothing here has a default list.

SIZE IS A REAL COST
-------------------
Every byte here is re-sent on every model call. `thread-chasing/SKILL.md` carries measurement
tables that exist for a human reader and would be dead weight in a prompt — `load_marker_catalogue`
already extracts the part the runtime needs. So loading is opt-in per skill, `MAX_SKILL_BYTES`
refuses a skill that has grown past what a prompt should carry rather than truncating it (a
truncated instruction is an instruction that ends mid-sentence), and the rendered size is
returned so a caller can record it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"

#: A skill bigger than this is refused rather than truncated. Roughly 3k tokens: enough for real
#: guidance, small enough that three of them do not dominate a prompt.
MAX_SKILL_BYTES = 12_000

#: 一张卡装在哪个槽。槽不是分类标签，是装配位置：`search` 槽恰好装一张，因为它是对照试验里
#: 唯一被替换的变量；`task` 槽最多一张，跟着 spec 走；`general` 槽不限张数；`eval` 槽属于
#: 评测那边的 agent，永远不进跑病历的提示词。
SLOTS: tuple[str, ...] = ("task", "search", "general", "eval")

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


class SkillError(ValueError):
    """A skill was requested that cannot be rendered into a prompt."""


def load_skill_body(name: str, skills_dir: Path | str | None = None) -> str:
    """The instruction half of one skill: its Markdown body, frontmatter stripped.

    Raises rather than returning "" for a missing or oversized skill. A silently empty skill is
    the bug this module was written to fix — the runtime would report that guidance was supplied
    and the model would receive nothing.
    """
    root = Path(skills_dir) if skills_dir else SKILLS_DIR
    path = root / name / "SKILL.md"
    if not path.is_file():
        raise SkillError(
            f"no skill {name!r} at {path}. A profile that offers a skill the tree does not have "
            f"would silently supply no guidance; available: "
            f"{sorted(p.name for p in root.iterdir() if (p / 'SKILL.md').is_file())}")
    raw = path.read_bytes()
    if len(raw) > MAX_SKILL_BYTES:
        raise SkillError(
            f"skill {name!r} is {len(raw)} bytes, over the {MAX_SKILL_BYTES}-byte prompt cap. "
            f"Split the reference material into `references/` — a truncated instruction is an "
            f"instruction that stops mid-sentence, so this refuses instead of trimming.")
    text = raw.decode("utf-8")
    body = _FRONTMATTER.sub("", text, count=1).strip()
    if not body:
        raise SkillError(f"skill {name!r} has frontmatter and no body")
    return body


def _frontmatter(name: str, skills_dir: Path | str | None = None) -> dict:
    """One skill's frontmatter as a mapping. Raises for anything a loader would drop silently."""
    root = Path(skills_dir) if skills_dir else SKILLS_DIR
    path = root / name / "SKILL.md"
    if not path.is_file():
        raise SkillError(f"no skill {name!r} at {path}")
    m = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not m:
        raise SkillError(f"skill {name!r} has no frontmatter block at byte 0")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise SkillError(f"skill {name!r} frontmatter is not a mapping")
    return data


def skill_slot(name: str, skills_dir: Path | str | None = None) -> str:
    """Which slot this skill declares it belongs in.

    Refuses a skill that declares nothing rather than defaulting it. A default here would put
    every unlabelled skill in one slot, and the first time somebody added a second search
    policy the two would render together — which reads, in the manifest, exactly like one
    policy that happens to be long.
    """
    fm = _frontmatter(name, skills_dir)
    slot = fm.get("slot")
    if not slot:
        raise SkillError(
            f"skill {name!r} declares no `slot`. Add one of {list(SLOTS)} to its frontmatter; "
            f"an undeclared slot cannot be assembled without guessing.")
    if slot not in SLOTS:
        raise SkillError(
            f"skill {name!r} declares unknown slot {slot!r}; expected one of {list(SLOTS)}")
    return str(slot)


def skills_block(names: Sequence[str], skills_dir: Path | str | None = None) -> str:
    """Render the named skills for the system prompt, in the order given.

    The header says what they are, because the distinction is the whole point of this session's
    work: these are judgement the model applies, not conditions the runtime enforces. A model that
    departs from a skill is not violating anything — it owes an account of why, and the account is
    what gets recorded.
    """
    if not names:
        return ""
    parts = [
        "METHOD GUIDANCE — JUDGEMENT YOU APPLY, NOT CONDITIONS THE RUNTIME ENFORCES",
        "",
        "Nothing below is checked mechanically. It is how a careful reviewer approaches these "
        "questions, and where it does not fit this chart you should depart from it and say so in "
        "your reasoning. Your departure is recorded, not refused.",
    ]
    for n in names:
        parts += ["", f"--- skill: {n} ---", "", load_skill_body(n, skills_dir)]
    return "\n".join(parts)


def skills_manifest(names: Sequence[str], skills_dir: Path | str | None = None) -> list[dict]:
    """What was actually rendered, per skill, for the run manifest.

    Content-hashed rather than named. A skill is prose the model acts on, so editing a sentence
    in it changes the run without changing its name or its version — and `refine` treats
    `skills/*/SKILL.md` as a tunable file, so editing them is a supported operation. A manifest
    that recorded only `["coverage-judgement"]` would say two runs were comparable when the text
    between them had moved.
    """
    import hashlib
    out = []
    for n in names:
        body = load_skill_body(n, skills_dir)
        out.append({"skill": n, "bytes": len(body.encode("utf-8")),
                    "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]})
    return out
