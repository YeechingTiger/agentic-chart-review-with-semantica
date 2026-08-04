"""Load a skill's prose into the system prompt. The missing half of "it lives in a skill".

WHY THIS EXISTS
---------------
`assets/skills/` held six skills and NOTHING READ ANY OF THEM INTO A PROMPT. The only path from that
directory into the runtime was `coverage_planner.load_marker_catalogue`, which parses the
thread-chasing skill's Markdown TABLES into a marker set — it extracts data, not instruction.
`refine.py` treats `assets/skills/*/SKILL.md` as a tunable develop-plane file. `cli_plan` checks a
skills directory only so that a route to a missing skill is reported rather than printed as
advice.

So on 2026-07-30, when the coverage obligation was moved out of `evaluate_gate` and into
`assets/skills/coverage-judgement/SKILL.md`, it was not moved. It was deleted, and the commit message
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
isolate. `agent` passes the assembled `SkillStack`; nothing here has a default list.

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
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..core import site
from ..core.repo_paths import asset_dir

SKILLS_DIR = asset_dir(str(site.skills_root()))

#: A skill bigger than this is refused rather than truncated. Roughly 3k tokens: enough for real
#: guidance, small enough that three of them do not dominate a prompt.
MAX_SKILL_BYTES = 12_000

#: 一张卡装在哪个槽。槽不是分类标签，是装配位置。
#:
#: `policy` 恰好装一张，并且总是生效 —— 它是一次运行的检索方针，是对照试验里唯一被替换的
#: 变量。`tactic` 不限张数且各带前提，因为一个战术是前提成立时才可以调用的动作，不是整个方
#: 针；2026-08-02 之前八张卡挤在同一个槽里八选一，量到的是不同种类干预之间的差别，不是策略
#: 之间的差别。
#:
#: 这个槽 2026-08-03 之前叫 `controller`，那个名字是从架构文档里搬来的，而文档里的
#: Strategic Controller 是一次独立的模型调用，输出 CONTINUE / STOP / ABSTAIN / ESCALATE 的
#: 封闭枚举且不碰检索。这里从来没有那个东西：没有任何代码读这个槽做决定，它唯一的去处是
#: `names()` 的渲染顺序。一个承诺了不存在机制的名字，在这棵树上已经付过一次代价（见
#: `tools/run_ladder.py` 的开头）。`experience` 只在经验
#: 臂打开。`task` 最多一张，跟着 spec 走。`general` 不限张数，是每个条件都有的东西。`eval`
#: 属于评测那边的 agent，永远不进跑病历的提示词。
SLOTS: tuple[str, ...] = ("task", "policy", "tactic", "experience", "general", "eval")

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class SkillError(ValueError):
    """A skill was requested that cannot be rendered into a prompt."""


def load_skill_body(name: str, skills_dir: Path | str | None = None) -> str:
    """The instruction half of one skill: its Markdown body, frontmatter stripped.

    Raises rather than returning "" for a missing or oversized skill. A silently empty skill is
    the bug this module was written to fix — the runtime would report that guidance was supplied
    and the model would receive nothing.
    """
    roots = [Path(skills_dir)] if skills_dir else list(site.skill_roots())
    path = next((r / name / "SKILL.md" for r in roots if (r / name / "SKILL.md").is_file()),
                roots[0] / name / "SKILL.md")
    root = path.parent.parent
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
    roots = [Path(skills_dir)] if skills_dir else list(site.skill_roots())
    path = next((r / name / "SKILL.md" for r in roots if (r / name / "SKILL.md").is_file()),
                roots[0] / name / "SKILL.md")
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
        # A slot is WHERE prose goes in a prompt, so only prose needs one. A script, an llm call or
        # a subagent is invoked through `contract/skill_invoke.py` and never rendered, and demanding
        # a slot for it would mean picking an assembly position for something that is not assembled.
        if str(fm.get("kind") or "prose") != "prose":
            return ""
        raise SkillError(
            f"skill {name!r} declares no `slot`. Add one of {list(SLOTS)} to its frontmatter; "
            f"an undeclared slot cannot be assembled without guessing. (Only `kind: prose` skills "
            f"need one — a script or subagent skill is invoked, not rendered.)")
    if slot not in SLOTS:
        raise SkillError(
            f"skill {name!r} declares unknown slot {slot!r}; expected one of {list(SLOTS)}")
    return str(slot)


@dataclass(frozen=True)
class SkillStack:
    """How one run's method guidance is assembled: which skill sits in which slot.

    A POLICY IS A SEARCH POLICY: how this run chooses what to look at next and how it judges
    that it is done. It holds AT MOST ONE because it is the variable an arm replaces, and
    because two policies rendered together are not "more guidance" — they are an unlabelled
    third policy, and the manifest would record two names where the model received one merged
    instruction.

    `tactics` is a TUPLE, and the difference is the whole point of the 2026-08-02 split. A
    policy always applies; a tactic is a move available when its precondition holds. Following a deferral needs a deferral to exist, entering at a summary needs a summary
    to exist, working a prior needs a prior — those are moves, and an arm that drew one of them
    as its entire policy on a record that did not meet its precondition was handed nothing.
    Eight cards competing for one slot measured the difference between kinds of intervention,
    not between policies.
    """

    task: str | None = None
    policy: str | None = None
    tactics: tuple[str, ...] = ()
    #: The develop-set prior, when a run is given one. A TUPLE and a separate slot from
    #: `tactics` because it is the third factor of the three-factor experiment and has to be
    #: switchable on its own: a prior is knowledge somebody measured elsewhere, a tactic is a
    #: move this run may make. Declared in `SLOTS` since the 2026-08-02 split and unreachable
    #: until 2026-08-03 -- the field was missing here, so the one card that declares
    #: `slot: experience` could not be stacked and the factor could not be turned on at all.
    experience: tuple[str, ...] = ()
    general: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        """Render order: the task, how to decide, the moves available, what somebody already
        measured, the standing habits."""
        out: list[str] = []
        if self.task:
            out.append(self.task)
        if self.policy:
            out.append(self.policy)
        out.extend(self.tactics)
        out.extend(self.experience)
        out.extend(self.general)
        return tuple(out)

    def validate(self, skills_dir: Path | str | None = None) -> None:
        """Every named skill exists and declares the slot it was placed in."""
        placed = [(self.task, "task"), (self.policy, "policy")]
        placed += [(n, "tactic") for n in self.tactics]
        placed += [(n, "experience") for n in self.experience]
        placed += [(n, "general") for n in self.general]
        seen: set[str] = set()
        for name, slot in placed:
            if not name:
                continue
            if name in seen:
                raise SkillError(f"skill {name!r} appears twice in one stack")
            seen.add(name)
            declared = skill_slot(name, skills_dir)
            if declared != slot:
                raise SkillError(
                    f"skill {name!r} declares slot {declared!r} but was placed in the {slot!r} "
                    f"slot. Placement is not a preference: the policy slot holds the one "
                    f"variable an arm replaces, and a tactic placed there becomes a whole policy "
                    f"on records where its precondition never fires.")


def parse_skill_stack(spec: str, base: SkillStack,
                      skills_dir: Path | str | None = None) -> SkillStack:
    """Apply a `slot=value` override string to a profile's stack.

    Exists so that swapping one search policy does not require authoring a whole new profile.
    A profile is a certified, content-hashed asset; a one-off arm in a pilot is not, and
    forcing the second to masquerade as the first is how uncertified assets get adopted.
    The result is validated, so a typo fails before a single model call is paid for.

    Clauses are comma-separated, which is why a multi-card REPLACEMENT list is joined by `|`
    instead: `general=a|b` is one clause naming two cards, where `general=a,general=b` would
    be two clauses of which only the last survives. `tactics` takes the same form, and `+name`
    appends to either instead of replacing.
    """
    if not spec.strip():
        return base
    task, policy = base.task, base.policy
    lists = {"tactics": list(base.tactics), "experience": list(base.experience),
             "general": list(base.general)}
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if "=" not in clause:
            raise SkillError(f"skill override {clause!r}: expected slot=value")
        slot, _, value = clause.partition("=")
        slot, value = slot.strip(), value.strip()
        if slot not in ("task", "policy", "tactics", "experience", "general"):
            raise SkillError(
                f"skill override: unknown slot {slot!r}; expected task, policy, tactics, "
                f"experience or general (the eval slot belongs to the evaluation agent, not a "
                f"chart run). "
                f"This slot was `search` until 2026-08-02, when the traversal tactics moved "
                f"out of it and it became `controller`; it became `policy` on 2026-08-03, "
                f"because nothing reads it to make a decision and `controller` named a "
                f"component this system does not have.")
        if slot == "task":
            task = value or None
        elif slot == "policy":
            policy = value or None
        elif value.startswith("+"):
            lists[slot].append(value[1:])
        else:
            lists[slot] = [v for v in value.split("|") if v]
    out = SkillStack(task=task, policy=policy,
                     tactics=tuple(lists["tactics"]),
                     experience=tuple(lists["experience"]),
                     general=tuple(lists["general"]))
    out.validate(skills_dir)
    return out


#: Phrases that turn a diagnostic skill into a scoring instruction. The fence this enforces is
#: recorded in README §2.6 and it is not stylistic: an AI judge scores a CORRECT
#: `EVIDENCE_INSUFFICIENT` as a task failure, and optimising against that teaches the agent to
#: guess on exactly the subpopulation where guessing is most dangerous. Scoring is `==` in
#: `evals.py`; a skill may ask the scorer, never replace it.
EVAL_FORBIDDEN_VERBS: tuple[str, ...] = (
    "score the", "grade the", "mark it correct", "mark it incorrect", "mark as correct",
    "decide whether the answer is correct", "judge whether the answer is correct",
    "rate the answer", "assign a score", "declare the answer wrong", "declare the answer right",
)


def eval_skill_judges(name: str, skills_dir: Path | str | None = None) -> tuple[str, ...]:
    """The sub-questions this eval skill is permitted to form a judgement about.

    The fence is PER SUB-QUESTION, not per dimension, so a skill that may diagnose search
    behaviour is not thereby licensed to opine on correctness. Declaring the list is what makes
    an overstep checkable; without it, scope is whatever the prose happens to imply.
    """
    fm = _frontmatter(name, skills_dir)
    slot = fm.get("slot")
    if slot != "eval":
        raise SkillError(
            f"skill {name!r} has slot {slot!r}, so it has no `judges` scope to read; that key "
            f"belongs to eval skills only")
    judges = fm.get("judges")
    if not judges:
        raise SkillError(
            f"eval skill {name!r} declares no `judges`. List the sub-questions it may form a "
            f"judgement about; an undeclared scope cannot be checked for overreach.")
    if isinstance(judges, str):
        judges = [judges]
    return tuple(str(j) for j in judges)


def eval_skills_block(names: Sequence[str], skills_dir: Path | str | None = None) -> str:
    """Render eval skills for the evaluation agent's prompt.

    A different header from `skills_block` because the standing instruction is different: the
    chart agent may depart from a skill, whereas the evaluation agent may not depart from the
    fence. What it may judge is declared; what it may not, it asks the deterministic scorer.
    """
    if not names:
        return ""
    parts = [
        "DIAGNOSTIC METHOD — HOW TO FIND A CAUSE. YOU DO NOT SCORE.",
        "",
        "Whether an answer was correct, whether a quote re-reads at its offsets, what a run "
        "cost: these are settled by the deterministic scorer, which is available to you as a "
        "read-only tool. Ask it. You have no channel for asserting a verdict yourself, and a "
        "diagnosis that assumes one is unusable.",
    ]
    for n in names:
        if skill_slot(n, skills_dir) != "eval":
            raise SkillError(f"skill {n!r} is not an eval skill")
        judges = ", ".join(eval_skill_judges(n, skills_dir))
        parts += ["", f"--- eval skill: {n} (may judge: {judges}) ---", "",
                  load_skill_body(n, skills_dir)]
    return "\n".join(parts)


def eval_skills_identity(block: str, names: Sequence[str]) -> dict:
    """The identity of a rendered eval-skills block: which cards, and a hash of what was sent.

    WHAT WAS RECORDED BEFORE was `eval_skills_bytes` — a byte count. Two entirely different card
    sets of equal length are indistinguishable by it, and a card edited in place very often does not
    change its length at all. That is the same defect `prompt_assets` was added to fix, in the
    diagnosis plane: content that reaches a model with no record of which content it was.

    It matters here specifically because `attribute meta-certify` scores these diagnoses against
    human adjudications. "Which method produced this causal judgement" is the first thing that
    comparison needs, and a length cannot answer it.

    TAKES THE RENDERED BLOCK, never the names alone. Rendering a second time here would put two
    renders between the prompt and the record, and one `skills_dir` disagreement would then make the
    manifest describe text no model ever read.

    An empty hash for no cards, not the hash of an empty string: "no method was loaded" and "a
    method was loaded and it rendered to nothing" are different facts about a diagnosis.
    """
    import hashlib
    return {"names": list(names), "n_cards": len(names), "n_chars": len(block),
            "content_hash": (hashlib.sha256(block.encode("utf-8")).hexdigest()[:16]
                             if block else "")}


def skills_block(stack: SkillStack, skills_dir: Path | str | None = None) -> str:
    """Render the stack for the system prompt, in slot order.

    The header says what they are, because the distinction is the whole point: these are
    judgement the model applies, not conditions the runtime enforces. A model that departs from
    a skill is not violating anything — it owes an account of why, and the account is recorded.
    """
    stack.validate(skills_dir)
    names = stack.names()
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
        head = f"--- skill: {n} ---"
        # A tactic's PRECONDITION has to reach the model or it is frontmatter nobody reads —
        # and the whole reason tactics were split out of the policy slot is that a move
        # drawn on a record that does not meet its precondition is a move that gives no
        # guidance. The policy needs to know when each one applies in order not to call it.
        pre = _frontmatter(n, skills_dir).get("precondition")
        if pre:
            head += f"\n    CALL THIS WHEN: {pre}"
        parts += ["", head, "", load_skill_body(n, skills_dir)]
    return "\n".join(parts)


def skills_manifest(stack: SkillStack, skills_dir: Path | str | None = None) -> list[dict]:
    """What was actually rendered, per skill and per slot, for the run manifest.

    Content-hashed rather than named. A skill is prose the model acts on, so editing a sentence
    changes the run without changing its name or version — and `refine` treats `assets/skills/*/SKILL.md`
    as a tunable file. The slot is recorded beside the hash because "which search policy ran" is
    the question a paired ablation asks, and a flat list cannot answer it.
    """
    import hashlib
    stack.validate(skills_dir)
    slot_of = {}
    if stack.task:
        slot_of[stack.task] = "task"
    if stack.policy:
        slot_of[stack.policy] = "policy"
    for n in stack.tactics:
        slot_of[n] = "tactic"
    for n in stack.experience:
        slot_of[n] = "experience"
    for n in stack.general:
        slot_of[n] = "general"
    out = []
    for n in stack.names():
        body = load_skill_body(n, skills_dir)
        out.append({"skill": n, "slot": slot_of[n], "bytes": len(body.encode("utf-8")),
                    "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]})
    return out
