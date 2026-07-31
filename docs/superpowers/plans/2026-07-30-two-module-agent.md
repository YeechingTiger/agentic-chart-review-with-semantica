# 两个模块（跑病历 / 评测）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把经验卡（skill）分成有名字的槽，让"病历怎么翻"这一张能被单独换掉做对照；再给评测那边一个统一入口，程序算的和 AI 看的都能单个跑也能批量跑，AI 那边每种复盘思路是一张卡。

**Architecture:** 跑病历这边，`skills.py` 学会读每张卡声明的槽位并校验装配，`runtime_profiles._PROFILE_SKILLS`（今天是空字典）填成按槽组织的结构，命令行加 `--skills` 临时换卡。评测这边，新建 `acr signal` 组作为薄壳，`--kind rule` 转发到现有的无模型 `evals`，`--kind agent` 转发到现有的 `attribution`，两边都吐 `SignalEnvelope`；AI 复盘的每种思路是一张 `slot: eval` 的卡，注入 `attribution.py` 的提示词。

**Tech Stack:** Python 3.11+、typer（CLI）、pyyaml（frontmatter）、pytest。不引入新依赖。

## Global Constraints

- **代码里的名字一个都不改。** `spec`、`trace`、`gate`、`manifest`、`coverage`、`plane` 照旧；`spec_hash`、`SPEC_INSUFFICIENT`、`gate_validated`、`EVIDENCE_INSUFFICIENT`、`coverage_attested` 这五个字段名绝对不动——它们在历史存档里，改了过去和未来的 run 对不上账。
- **默认行为必须逐字节不变。** 这次改动之后，`current-stratified-coverage`（默认 profile）渲染给模型的 skills 文本必须和改动前完全一样：只有 `coverage-judgement` 一张。Task 2 有专门的测试钉死这一点。新卡只能通过 `--skills` 或新 profile 到达模型。
- **`evals.py` 的无模型闭包不许破。** `tests/test_evals.py::test_no_model_is_reachable_from_this_module` 用 AST 走 `src/acr/evals.py` 的 import，禁止出现 `llm graph deep_runner cli openai anthropic litellm langchain langgraph deepagents requests httpx urllib socket http`。新代码一律不往 `evals.py` 里加 import。
- **`acr eval` 组保持无模型。** 它的模块 docstring 承诺"NOTHING IN THIS GROUP CALLS A MODEL"。AI 复盘不进这个组，进新的 `acr signal` 组。
- **AI 没有判分权。** 评测卡里不许出现判定对错的指令；Task 5 用测试检查。判分只走 `evals.py` / `audit_loop.py` 的确定性函数。
- **`assets/skills/*/SKILL.md` 单文件上限 12000 字节**（`skills.MAX_SKILL_BYTES`），超了报错不截断。
- **卡的 frontmatter 现有必填字段**：`name`（小写连字符、必须等于目录名、≤64 字符）、`description`（≤1024 字符）。本计划新增 `slot`，评测卡再加 `judges`。
- 提交信息用英文，结尾附 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`。
- **必须在装了 `.venv` 的机器上执行。** 项目要 Python 3.11+（`kernel.py` 用 `datetime.UTC`），
  开发机 macOS 自带的是 3.9 且没装 typer——那里连 `import acr` 都会失败。所以计划里所有验证
  命令一律写成 `.venv/bin/...`。`deepagents` 在第二个环境 `.venv-deep`，本计划不碰它。
- 每个任务结束前跑完整套件：`.venv/bin/pytest -q`，失败/skip 数必须与改动前一致（树里有已知
  缺口，见 `tests/test_skills_load.py:49-61`）。**开工前先跑一次记下基线。**

---

## 文件结构

**新建：**

| 文件 | 职责 |
|---|---|
| `assets/skills/search-native/SKILL.md` | "病历怎么翻"卡：模型自己判断查够了没有 |
| `assets/skills/search-preplanned/SKILL.md` | "病历怎么翻"卡：先按经验统计排序再查 |
| `assets/skills/eval-contrast-traces/SKILL.md` | 复盘卡：对照答对与答错的工作记录 |
| `assets/skills/eval-cluster-failures/SKILL.md` | 复盘卡：把多个失败归类 |
| `assets/skills/eval-missed-evidence/SKILL.md` | 复盘卡：答案在某份报告里却没翻到 |
| `assets/skills/eval-overconfidence/SKILL.md` | 复盘卡：斩钉截铁但答错 |
| `src/acr/cli_signal.py` | `acr signal run` / `acr signal batch` 薄壳，三种做法一个出口（Task 7-8 先接 rule/agent 两种，Task 9 接入 judge） |
| `tests/test_skill_slots.py` | 槽位声明与装配校验 |
| `tests/test_eval_skill_fence.py` | 评测卡不得含判分指令、必须声明 `judges` |
| `tests/test_cli_signal.py` | 新入口的路由与 signal 形状 |

**修改：**

| 文件 | 改什么 |
|---|---|
| `src/acr/skills.py` | 加 `SLOTS`、`skill_slot()`、`SkillStack`、`skills_block(stack)`、`skills_manifest` 带槽 |
| `src/acr/runtime_profiles.py:468` | `_PROFILE_SKILLS` 从空字典填成 `dict[str, SkillStack]` |
| `src/acr/agent.py:1368-1372, 1514` | `run_patient` 收 `skill_stack` 参数；渲染改走 stack |
| `src/acr/run_manifest.py:412-419` | 存档单里每张卡带 `slot` |
| `src/acr/cli_chart.py:61, 138` | `acr run` / `acr batch` 加 `--skills` |
| `src/acr/cli.py:97` 附近 | 挂上 `signal_app` |
| `src/acr/attribution.py:1790-1816` | 提示词里注入评测卡 |
| `assets/skills/*/SKILL.md`（现有 10 张） | frontmatter 加 `slot:` |

---

## Task 1: 每张卡声明自己属于哪个槽

**Files:**
- Modify: `src/acr/skills.py`
- Modify: `assets/skills/chart-triage/SKILL.md`, `assets/skills/thread-chasing/SKILL.md`, `assets/skills/coverage-judgement/SKILL.md`, `assets/skills/keyword-strategy/SKILL.md`, `assets/skills/store-icdo-coding/SKILL.md`, `assets/skills/store-staging/SKILL.md`, `assets/skills/store-to-spec/SKILL.md`, `assets/skills/crc-guideline-registry-authoring/SKILL.md`, `assets/skills/non-concordance-triage/SKILL.md`, `assets/skills/guideline-to-rules/SKILL.md`
- Test: `tests/test_skill_slots.py`（新建）

**Interfaces:**
- Consumes: 无（第一个任务）
- Produces: `acr.skills.SLOTS: tuple[str, ...]`、`acr.skills.skill_slot(name: str, skills_dir=None) -> str`、`acr.skills.SkillError`（已存在，复用）

**背景**：`assets/skills/guideline-to-rules/` 只有 `references/`，`SKILL.md` 从未写过（build agent 被 spend limit 杀了），`tests/test_skills_load.py:49-55` 已经为它挂了 skip。本任务同样跳过它，不要补写。

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_skill_slots.py`：

```python
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

# `assets/skills/guideline-to-rules/SKILL.md` 从未写过——见 tests/test_skills_load.py:49-55。
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_skill_slots.py -q`
Expected: FAIL — `ImportError: cannot import name 'SLOTS' from 'acr.skills'`

- [ ] **Step 3: 在 `skills.py` 里实现**

在 `src/acr/skills.py` 顶部 import 区加 `import yaml`（`from __future__ import annotations` 之后、`import re` 附近，按字母序放在 `re` 之后）。

在 `MAX_SKILL_BYTES` 常量之后插入：

```python
#: 一张卡装在哪个槽。槽不是分类标签，是装配位置：`search` 槽恰好装一张，因为它是对照试验里
#: 唯一被替换的变量；`task` 槽最多一张，跟着 spec 走；`general` 槽不限张数；`eval` 槽属于
#: 评测那边的 agent，永远不进跑病历的提示词。
SLOTS: tuple[str, ...] = ("task", "search", "general", "eval")
```

在 `load_skill_body` 之后插入：

```python
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
        raise SkillError(f"skill {name!r} declares unknown slot {slot!r}; expected one of {list(SLOTS)}")
    return str(slot)
```

- [ ] **Step 4: 给现有 9 张卡加 `slot:`**

在每个 `SKILL.md` 的 frontmatter 里，`description` 之后、`license` 之前加一行。逐张对应：

```
assets/skills/chart-triage/SKILL.md                      slot: general
assets/skills/thread-chasing/SKILL.md                    slot: general
assets/skills/coverage-judgement/SKILL.md                slot: general
assets/skills/keyword-strategy/SKILL.md                  slot: search
assets/skills/store-icdo-coding/SKILL.md                 slot: task
assets/skills/store-staging/SKILL.md                     slot: task
assets/skills/store-to-spec/SKILL.md                     slot: task
assets/skills/crc-guideline-registry-authoring/SKILL.md  slot: task
assets/skills/non-concordance-triage/SKILL.md            slot: general
```

`assets/skills/guideline-to-rules/` 跳过——没有 `SKILL.md`。

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_skill_slots.py tests/test_skills_load.py -q`
Expected: PASS（`guideline-to-rules` 显示 skip）

- [ ] **Step 6: 跑全套确认没打破别的**

Run: `.venv/bin/pytest -q`
Expected: PASS，失败数与改动前一致

- [ ] **Step 7: 提交**

```bash
git add src/acr/skills.py assets/skills/ tests/test_skill_slots.py
git commit -m "$(cat <<'EOF'
Skills declare their slot; an undeclared slot is refused, not defaulted

A search policy is the one variable a retrieval arm replaces. With every skill in one
undifferentiated list, a second search skill renders alongside the first and the manifest
records that as one long policy. Slots are declared in frontmatter and validated on load.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: profile 按槽组织装配，存档单记下槽位

**Files:**
- Modify: `src/acr/skills.py`
- Modify: `src/acr/runtime_profiles.py:459-473`
- Modify: `src/acr/agent.py:1397, 1514`
- Modify: `src/acr/run_manifest.py:412-419`
- Test: `tests/test_skill_slots.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `SLOTS`、`skill_slot()`、`SkillError`
- Produces:
  - `acr.skills.SkillStack`（frozen dataclass，字段 `task: str | None = None`、`search: str | None = None`、`general: tuple[str, ...] = ()`），方法 `names() -> tuple[str, ...]`（渲染序 task→search→general）、`validate(skills_dir=None) -> None`
  - `acr.skills.skills_block(stack: SkillStack, skills_dir=None) -> str`
  - `acr.skills.skills_manifest(stack: SkillStack, skills_dir=None) -> list[dict]`，每项含 `skill` / `slot` / `bytes` / `content_hash`
  - `acr.runtime_profiles.runtime_policy_skills(module_id: str) -> SkillStack`

**注意**：`skills_block` 和 `skills_manifest` 的参数类型从 `Sequence[str]` 变成 `SkillStack`，两个调用点（`agent.py:1514`、`run_manifest.py:417`）都要跟着改。全树只有这两处调用它们。

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_skill_slots.py` 末尾：

```python
from acr.skills import SkillStack, skills_block, skills_manifest


def test_stack_renders_task_then_search_then_general():
    stack = SkillStack(task="store-icdo-coding", search="keyword-strategy",
                       general=("chart-triage", "coverage-judgement"))
    assert stack.names() == ("store-icdo-coding", "keyword-strategy",
                             "chart-triage", "coverage-judgement")


def test_stack_rejects_a_skill_in_the_wrong_slot():
    """把一张 general 卡塞进 search 槽——这正是会让对照试验失去意义的装配错误。"""
    with pytest.raises(SkillError, match="chart-triage.*declares slot 'general'.*'search'"):
        SkillStack(search="chart-triage").validate()


def test_stack_rejects_an_eval_skill_in_the_chart_agent():
    with pytest.raises(SkillError, match="slot 'eval'"):
        SkillStack(general=("eval-contrast-traces",)).validate()


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
    from acr.skills import load_skill_body
    assert skills_block(stack).endswith(load_skill_body("coverage-judgement"))


def test_unknown_profile_still_falls_back_to_coverage_judgement():
    from acr.runtime_profiles import runtime_policy_skills
    assert runtime_policy_skills("not-a-profile").names() == ("coverage-judgement",)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_skill_slots.py -q`
Expected: FAIL — `ImportError: cannot import name 'SkillStack'`

- [ ] **Step 3: 在 `skills.py` 里加 `SkillStack`**

在 `skill_slot` 之后、`skills_block` 之前插入。同时在文件顶部 import 区加 `from dataclasses import dataclass, field`：

```python
@dataclass(frozen=True)
class SkillStack:
    """How one run's method guidance is assembled: which skill sits in which slot.

    `search` holds AT MOST ONE because it is the variable a retrieval arm replaces. Two search
    policies rendered together are not "more guidance" — they are an unlabelled third policy,
    and the manifest would record two names where the model received one merged instruction.
    """
    task: str | None = None
    search: str | None = None
    general: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        """Render order: what the task is, then how to look, then the standing habits."""
        out: list[str] = []
        if self.task:
            out.append(self.task)
        if self.search:
            out.append(self.search)
        out.extend(self.general)
        return tuple(out)

    def validate(self, skills_dir: Path | str | None = None) -> None:
        """Every named skill exists and declares the slot it was placed in."""
        placed = [(self.task, "task"), (self.search, "search")]
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
                    f"slot. Placement is not a preference: the search slot is the one variable "
                    f"a retrieval arm replaces.")
```

- [ ] **Step 4: `skills_block` 和 `skills_manifest` 改吃 `SkillStack`**

把 `skills_block` 的签名和头两行改成：

```python
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
        parts += ["", f"--- skill: {n} ---", "", load_skill_body(n, skills_dir)]
    return "\n".join(parts)
```

`skills_manifest` 改成：

```python
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
    if stack.search:
        slot_of[stack.search] = "search"
    for n in stack.general:
        slot_of[n] = "general"
    out = []
    for n in stack.names():
        body = load_skill_body(n, skills_dir)
        out.append({"skill": n, "slot": slot_of[n], "bytes": len(body.encode("utf-8")),
                    "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]})
    return out
```

`from collections.abc import Sequence` 如果不再被用到就删掉。

- [ ] **Step 5: 填 `_PROFILE_SKILLS`**

`src/acr/runtime_profiles.py` 顶部 import 区加 `from .skills import SkillStack`。把 468-473 行整段替换成：

```python
#: WHICH METHOD SKILLS EACH PROFILE OFFERS THE MODEL, BY SLOT. A skill is judgement guidance,
#: so swapping one is exactly the kind of change an arm has to isolate — which is why this is a
#: property of the profile and not a default buried in the prompt builder.
#:
#: EVERY PROFILE BELOW RENDERS EXACTLY WHAT IT RENDERED BEFORE SLOTS EXISTED: `coverage-judgement`
#: and nothing else. That is deliberate. Every run ever recorded was made under that one skill,
#: and quietly adding a second here would make past and future runs incomparable while the
#: manifest went on looking the same. New search policies reach a run through `--skills` or
#: through a NEW profile, both of which are recorded as the change they are.
#:
#: `coverage-judgement` is in `general` and not `search`: it supplies no keywords, no note-type
#: prior and no strata, and it activates only when the answer is about to claim something is
#: absent. It is not the retrieval asset the arms compare.
_PROFILE_SKILLS: dict[str, SkillStack] = {
    GUIDELINE_ONLY_PROFILE: SkillStack(general=("coverage-judgement",)),
    CONDITIONAL_COVERAGE_PROFILE: SkillStack(general=("coverage-judgement",)),
    ALWAYS_COVERAGE_PROFILE: SkillStack(general=("coverage-judgement",)),
    WITNESS_FIRST_PROFILE: SkillStack(general=("coverage-judgement",)),
    STRATIFIED_COVERAGE_PROFILE: SkillStack(general=("coverage-judgement",)),
}

_FALLBACK_SKILLS = SkillStack(general=("coverage-judgement",))


def runtime_policy_skills(module_id: str) -> SkillStack:
    """The method skills this profile renders into the system prompt, by slot."""
    return _PROFILE_SKILLS.get(module_id, _FALLBACK_SKILLS)
```

- [ ] **Step 6: 改两个调用点**

`src/acr/agent.py:1514` 附近——`skills_block(runtime_policy_skills(runtime_profile_asset.module_id))` 的参数现在是 `SkillStack`，调用形式不变，不用改。确认一下即可。

`src/acr/run_manifest.py:417` 同理，`skills_manifest(runtime_policy_skills(module_id))` 形式不变。确认即可。

（两处都是把 `runtime_policy_skills(...)` 的返回值直接传进去，返回类型变了但调用写法没变。这就是为什么这两个函数一起改签名是安全的。）

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_skill_slots.py -q`
Expected: PASS

- [ ] **Step 8: 跑全套**

Run: `.venv/bin/pytest -q`
Expected: PASS，失败数与改动前一致

- [ ] **Step 9: 提交**

```bash
git add src/acr/skills.py src/acr/runtime_profiles.py src/acr/run_manifest.py tests/test_skill_slots.py
git commit -m "$(cat <<'EOF'
Profiles assemble skills by slot; the manifest records which slot each came from

_PROFILE_SKILLS was an empty dict, so every profile fell through to the same one skill and
swapping a profile never swapped guidance. It now holds a SkillStack per profile — rendering
exactly what each rendered before, so no past run becomes incomparable — and the manifest
carries the slot beside each content hash, which is what a paired ablation has to read.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 命令行 `--skills` 临时换卡

**Files:**
- Modify: `src/acr/agent.py:1368-1372`（`run_patient` 签名）、`:1514`（渲染处）
- Modify: `src/acr/cli_chart.py:61-136`（`run`）、`:138-187`（`batch`）
- Test: `tests/test_skill_slots.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `SkillStack`、`runtime_policy_skills()`
- Produces:
  - `acr.skills.parse_skill_stack(spec: str, base: SkillStack, skills_dir=None) -> SkillStack`——把 `"search=search-native,general=+chart-triage"` 这样的字符串套在 base 上
  - `run_patient(..., skill_stack: SkillStack | None = None)`——`None` 表示用 profile 的

**语法**（写进 `--skills` 的帮助文本）：逗号分隔的 `slot=value`。`task=` / `search=` 直接替换；`general=+name` 追加一张，`general=name` 整个替换；`search=` 后面留空表示清空这个槽。

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_skill_slots.py`：

```python
from acr.skills import parse_skill_stack


def test_parse_replaces_the_search_slot():
    base = SkillStack(general=("coverage-judgement",))
    got = parse_skill_stack("search=keyword-strategy", base)
    assert got == SkillStack(search="keyword-strategy", general=("coverage-judgement",))


def test_parse_appends_to_general():
    base = SkillStack(general=("coverage-judgement",))
    got = parse_skill_stack("general=+chart-triage", base)
    assert got.general == ("coverage-judgement", "chart-triage")


def test_parse_replaces_general_without_plus():
    base = SkillStack(general=("coverage-judgement", "chart-triage"))
    assert parse_skill_stack("general=thread-chasing", base).general == ("thread-chasing",)


def test_parse_empty_value_clears_the_slot():
    base = SkillStack(search="keyword-strategy", general=("coverage-judgement",))
    assert parse_skill_stack("search=", base).search is None


def test_parse_rejects_an_unknown_slot():
    with pytest.raises(SkillError, match="unknown slot 'polish'"):
        parse_skill_stack("polish=x", SkillStack())


def test_parse_rejects_a_missing_equals():
    with pytest.raises(SkillError, match="expected slot=value"):
        parse_skill_stack("search-native", SkillStack())


def test_parse_validates_placement():
    with pytest.raises(SkillError, match="declares slot 'general'"):
        parse_skill_stack("search=chart-triage", SkillStack())


def test_parse_empty_string_is_the_base():
    base = SkillStack(general=("coverage-judgement",))
    assert parse_skill_stack("", base) == base
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_skill_slots.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_skill_stack'`

- [ ] **Step 3: 实现 `parse_skill_stack`**

加在 `skills.py` 的 `SkillStack` 之后：

```python
def parse_skill_stack(spec: str, base: SkillStack,
                      skills_dir: Path | str | None = None) -> SkillStack:
    """Apply a `slot=value` override string to a profile's stack.

    Exists so that swapping one search policy does not require authoring a whole new profile.
    A profile is a certified, content-hashed asset; a one-off arm in a pilot is not, and
    forcing the second to masquerade as the first is how uncertified assets get adopted.
    The result is validated, so a typo fails before a single model call is paid for.
    """
    if not spec.strip():
        return base
    task, search, general = base.task, base.search, list(base.general)
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if "=" not in clause:
            raise SkillError(f"skill override {clause!r}: expected slot=value")
        slot, _, value = clause.partition("=")
        slot, value = slot.strip(), value.strip()
        if slot not in ("task", "search", "general"):
            raise SkillError(
                f"skill override: unknown slot {slot!r}; expected task, search or general "
                f"(the eval slot belongs to the evaluation agent, not a chart run)")
        if slot == "task":
            task = value or None
        elif slot == "search":
            search = value or None
        elif value.startswith("+"):
            general.append(value[1:])
        else:
            general = [v for v in value.split("|") if v]
    out = SkillStack(task=task, search=search, general=tuple(general))
    out.validate(skills_dir)
    return out
```

- [ ] **Step 4: `run_patient` 收 `skill_stack`**

`src/acr/agent.py:1368-1372` 签名末尾加一个参数：

```python
def run_patient(*, spec, corpus, patient_id: str, out_dir, model, max_model_calls: int,
                seed: int = 1234, expansion_budget=None, run_id: str | None = None,
                ctx_out: list | None = None, max_usd: float = 5.0,
                additional_task_context: str = "",
                runtime_profile: str = "current-stratified-coverage",
                skill_stack=None) -> dict:
```

在 docstring 之后、`import time` 之前加一句注释说明：

```python
    # `skill_stack` is an explicit override of the profile's assembly, for a pilot arm that
    # swaps one policy without minting a certified profile. None means "whatever the profile
    # says", which is the only path any recorded run has taken.
```

把 1514 行附近的渲染改成用解析后的 stack：

```python
                       + (f"\n\n{sk}" if (sk := skills_block(
                           skill_stack if skill_stack is not None
                           else runtime_policy_skills(runtime_profile_asset.module_id))) else "")
```

- [ ] **Step 5: 两个 CLI 命令加 `--skills`**

`src/acr/cli_chart.py` 的 `run`（61 行起），在 `runtime_profile` 参数之后、`conflict_refine` 之前插入：

```python
    skills: str = typer.Option(
        "", "--skills",
        help="override the profile's skill assembly: comma-separated slot=value. "
             "`search=search-native` replaces the search policy, `general=+chart-triage` "
             "appends one, `search=` clears the slot. Validated before any model call."),
```

在函数体里 `sp = load_spec(spec)` 之后加：

```python
    from .runtime_profiles import resolve_runtime_policy
    from .skills import parse_skill_stack
    from .runtime_profiles import runtime_policy_skills
    profile_asset, _ = resolve_runtime_policy(runtime_profile)
    stack = parse_skill_stack(skills, runtime_policy_skills(profile_asset.module_id))
```

把两处 `run_patient(...)` 调用都加上 `skill_stack=stack`。

`batch`（138 行起）同样加 `--skills` 参数、同样在 `sp = load_spec(spec)` 后解析、循环里的 `run_patient(...)` 加 `skill_stack=stack`。

- [ ] **Step 6: 跑测试**

Run: `.venv/bin/pytest tests/test_skill_slots.py tests/test_cli_composition.py -q`
Expected: PASS

- [ ] **Step 7: 确认帮助文本能出来**

Run: `.venv/bin/acr run --help 2>&1 | grep -A3 -- --skills`
Expected: 打印出 `--skills` 及其说明

- [ ] **Step 8: 跑全套并提交**

Run: `.venv/bin/pytest -q`

```bash
git add src/acr/skills.py src/acr/agent.py src/acr/cli_chart.py tests/test_skill_slots.py
git commit -m "$(cat <<'EOF'
--skills swaps one policy without minting a profile

A profile is a certified, content-hashed asset. A one-off pilot arm is not, and making the
second masquerade as the first is how an uncertified asset gets adopted. The override string
is parsed and validated before the first model call, so a typo costs nothing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 两张"病历怎么翻"的卡

**Files:**
- Create: `assets/skills/search-native/SKILL.md`
- Create: `assets/skills/search-preplanned/SKILL.md`

**Interfaces:**
- Consumes: Task 1 的 `slot:` frontmatter 约定
- Produces: 两个可以放进 `search` 槽的名字，供 `--skills search=...` 使用

两张卡的内容依据 `docs/SEARCH_PLANNING_PILOT.md` 的两个试验臂写，不是新发明。

- [ ] **Step 1: 写 `assets/skills/search-native/SKILL.md`**

```markdown
---
name: search-native
description: Use when deciding how to search a chart and when to stop searching, and no keyword list or note-type prior has been supplied. Tells you how to choose terms from the task contract and the patient's own document inventory, how to widen after an empty result, and how to judge that further searching would not change your answer. Not a rule about how much searching is enough - that judgement stays yours and is recorded, not enforced.
slot: search
license: MIT
---

# Choosing your own search, and deciding when it is done

Nobody has handed you a term list. That is the point of this arm: the task contract says what
the answer must mean, the document inventory says what this chart contains, and the searching
in between is yours.

## Where terms come from

Read the contract's field definitions first, then `document_type_summary`. Terms come from
three places, in this order of reliability:

1. **The contract's own words** — the names of the values it asks you to distinguish. These
   are the words a report writer would have used for the same concept.
2. **The chart's vocabulary** — a term that appears in this patient's document type names
   costs nothing to try and is written in the local dialect.
3. **Synonyms you supply** — the least reliable, because you are guessing at a local
   convention you have not seen. Try these after the first two return nothing.

## Widening after an empty result

An empty search is information: either the term is wrong or the concept is absent. You cannot
tell which from the miss alone, so widen along ONE axis at a time and note which:

- **Shorter stem** — `adenocarc` before `adenocarcinoma`; abbreviations and truncations are how
  dictation actually reads.
- **A different word for the same thing** — the concept's other name, not a related concept.
- **A different document type** — same term, somewhere else in the chart.

Widening along two axes at once means a hit tells you nothing about which change earned it.

## Deciding you are done

You are done when you can say what a further search would have to find in order to change your
answer, and you have looked where that thing would be. That sentence is the test. If you cannot
write it, you are not done; if you can write it and you have looked there, more searching is
spending without a hypothesis.

Two things are NOT reasons to stop: running out of ideas for terms, and having read a lot of
documents. Neither is a statement about the chart.

If your answer claims something is absent, the coverage-judgement skill applies on top of this
one — an absence claim owes more than a positive one.
```

- [ ] **Step 2: 写 `assets/skills/search-preplanned/SKILL.md`**

```markdown
---
name: search-preplanned
description: Use when a keyword list and document-type prior have been supplied with the task and you must decide how far to trust them. Tells you the order to work a supplied plan, what a miss against a high-yield type actually means, when to leave the plan and search on your own words, and what to record when the plan and the chart disagree. The plan is a prior measured on other patients, not a description of this one.
slot: search
license: MIT
---

# Working a supplied plan without being trapped by it

You have been given a term list and a document-type prior. Both were measured on OTHER
patients. They are a good place to start and a bad place to stop.

## The order

1. **High-yield types first, with the supplied terms.** This is the cheapest path to a witness
   and it is why the plan exists. If it produces an admissible witness, you are close to done.
2. **The same terms across the rest of the chart.** A type prior is a rate, not a rule; the
   answer lands in an unexpected type often enough that skipping this step is how a documented
   value gets reported as absent.
3. **Your own terms, if the first two came back thin.** At this point the plan has been tried
   and has not settled the question, and continuing to work it is repetition rather than
   escalation.

## What a miss means

A supplied term missing in a high-yield type is the informative case, and it has three
explanations you must tell apart:

- the concept is genuinely absent from this chart;
- this site writes it differently from the sites the list was measured on;
- the term is right but the type filter is wrong.

Try the second before concluding the first: search the term with no type filter. That one call
separates "not in this chart" from "not where the prior said".

## When the plan and the chart disagree

The chart wins. The plan is evidence about a population; the chart in front of you is the
case. When you depart from the plan — searching a type it calls low-yield, using a term it does
not list — say so in your reasoning and say what prompted it. That sentence is the whole value
of this arm to whoever reads the run afterwards: it is the record of where a measured prior
failed to fit, and it is the only way the prior ever gets better.

Do not let the plan's coverage stand in for your own. Having worked every step of a supplied
plan is not a statement that the chart was adequately searched — if your answer claims absence,
the coverage-judgement skill applies on top of this one.
```

- [ ] **Step 3: 跑卡片加载测试**

Run: `.venv/bin/pytest tests/test_skills_load.py tests/test_skill_slots.py -q`
Expected: PASS，两张新卡都被参数化进去并通过

- [ ] **Step 4: 确认能装进 search 槽**

Run:
```bash
.venv/bin/python -c "
from acr.skills import SkillStack, parse_skill_stack
base = SkillStack(general=('coverage-judgement',))
for s in ('search-native', 'search-preplanned'):
    st = parse_skill_stack(f'search={s}', base)
    print(s, '->', st.names())
"
```
Expected: 两行，分别是 `search-native -> ('search-native', 'coverage-judgement')` 和 `search-preplanned -> ('search-preplanned', 'coverage-judgement')`

- [ ] **Step 5: 提交**

```bash
git add assets/skills/search-native assets/skills/search-preplanned
git commit -m "$(cat <<'EOF'
Two search policies as swappable skills, from the pilot's two arms

SEARCH_PLANNING_PILOT compared native planning against a spec-derived prior and found the
prior did not improve accuracy. Both arms now exist as skills a run can be pointed at, so the
comparison is one --skills flag rather than a branch in the prompt builder.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 评测卡的槽位、判断范围声明，和"不许判分"的检查

**Files:**
- Modify: `src/acr/skills.py`
- Test: `tests/test_eval_skill_fence.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `skill_slot()`、`_frontmatter()`；Task 2 的 `SkillError`
- Produces:
  - `acr.skills.eval_skill_judges(name: str, skills_dir=None) -> tuple[str, ...]`——这张卡声明它能判断哪些子问题
  - `acr.skills.EVAL_FORBIDDEN_VERBS: tuple[str, ...]`——评测卡正文里不许出现的判分措辞
  - `acr.skills.eval_skills_block(names, skills_dir=None) -> str`

**为什么围栏要用测试而不是靠提示词自觉**：`README §2.6` 记着一条已经吃过的亏——用 AI 判"任务完成没有"，会把正确的 `EVIDENCE_INSUFFICIENT` 判成失败，长期优化等于教模型在最不该猜的地方猜。围栏写在提示词里，一次改写就没了；写成测试，改写就红。

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_eval_skill_fence.py`：

```python
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

from acr.skills import (
    EVAL_FORBIDDEN_VERBS,
    SkillError,
    eval_skill_judges,
    load_skill_body,
    skill_slot,
)

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_eval_skill_fence.py -q`
Expected: FAIL — `ImportError: cannot import name 'EVAL_FORBIDDEN_VERBS'`

- [ ] **Step 3: 实现**

加到 `skills.py` 的 `parse_skill_stack` 之后：

```python
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
            f"skill {name!r} has slot {slot!r} but carries `judges`; that key belongs to eval "
            f"skills only")
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
```

若 Task 2 删掉了 `from collections.abc import Sequence`，这里要加回来。

- [ ] **Step 4: 跑测试**

Run: `.venv/bin/pytest tests/test_eval_skill_fence.py -q`
Expected: 除 `test_there_is_at_least_one_eval_skill` 外全 PASS；那一条 FAIL 是对的——卡还没写，Task 6 补上

- [ ] **Step 5: 提交**

```bash
git add src/acr/skills.py tests/test_eval_skill_fence.py
git commit -m "$(cat <<'EOF'
Eval skills declare what they may judge, and cannot instruct scoring

An AI judge scores a correct EVIDENCE_INSUFFICIENT as a task failure; optimising against that
teaches the agent to guess where guessing is most dangerous (README 2.6). A fence written into
a prompt survives until the next rewrite. This one is a test.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 四张复盘卡

**Files:**
- Create: `assets/skills/eval-contrast-traces/SKILL.md`
- Create: `assets/skills/eval-cluster-failures/SKILL.md`
- Create: `assets/skills/eval-missed-evidence/SKILL.md`
- Create: `assets/skills/eval-overconfidence/SKILL.md`

**Interfaces:**
- Consumes: Task 5 的 `slot: eval` + `judges:` 约定、`EVAL_FORBIDDEN_VERBS` 禁用词
- Produces: 四个可传给 `eval_skills_block()` 的名字

四张卡都不得出现 `EVAL_FORBIDDEN_VERBS` 里的措辞。每张卡都要显式写"对错问评分工具"。

- [ ] **Step 1: 写 `assets/skills/eval-contrast-traces/SKILL.md`**

```markdown
---
name: eval-contrast-traces
description: Use when two runs of the same spec reached different answers, or one run matched the answer key and another did not, and you need to locate where their paths diverged. Tells you how to align two traces step by step, which differences are causal and which are noise, and how to state a divergence point as a claim someone could check. Does not settle which answer was right - ask the deterministic scorer for that.
slot: eval
judges: [search_behaviour, divergence_point, plan_adherence]
license: MIT
---

# Putting two traces side by side

You have two work logs for the same question. One of them may have matched the answer key —
**ask the scorer, do not infer it from how confident the reasoning sounds.**

## Align before you compare

Traces are not comparable turn by turn: one run may spend three calls where the other spends
one. Align on EVENTS, not on turn numbers:

1. first search issued
2. first admissible witness read
3. first widening after an empty result
4. the read the final answer cites
5. submission

A run that never reached one of these has a divergence point at that event, and that is
usually the whole finding.

## Which differences matter

Most differences between two traces are noise: different phrasing, different order among
equally-productive searches, one extra confirmatory read. A difference is worth reporting when
it changes what text the model ever saw. Three that do:

- **A term one run tried and the other did not**, where the term hit.
- **A document one run opened and the other did not**, where the document carries the field.
- **A stopping decision** taken at different evidence states.

A difference in wording of reasoning, with identical reads, is not a divergence — it is the
same run described twice.

## Stating the finding

Name the earliest event where the paths differ, quote both sides from the trace, and say what
the later run would have had to do differently to arrive where the other one did. If you
cannot point at a step, you have found a correlation, and say that instead — a divergence you
cannot locate is a real observation and a false explanation.
```

- [ ] **Step 2: 写 `assets/skills/eval-cluster-failures/SKILL.md`**

```markdown
---
name: eval-cluster-failures
description: Use when several runs across a cohort came out wrong and you must tell whether they share one cause or have several. Tells you which trace features to cluster on, why clustering on the wrong answer's value misleads, how many cases a cluster needs before it is worth reporting, and how to name a cluster so that a fix can be aimed at it. Does not decide which runs were wrong - ask the deterministic scorer for the list.
slot: eval
judges: [failure_grouping, shared_mechanism, cluster_support]
license: MIT
---

# Telling one problem from six

Six wrong answers can be one bug or six. The difference decides whether anything is worth
fixing, and it is not visible in the answers themselves.

**Get the list of wrong runs from the scorer.** Which ones missed is not yours to decide.

## Cluster on behaviour, not on the answer

The tempting axis is the value that came out wrong — all the runs that said `C349`. That axis
groups by SYMPTOM, and a symptom shared by two different mechanisms produces a cluster nobody
can fix. Cluster on what the run DID:

- **the last search before submission** — what it was looking for when it gave up
- **whether an admissible witness was ever read** — never-read and read-but-misjudged are
  different failures with different owners
- **where the answer's citation came from** — which document type, which section
- **the shape of the stop** — budget exhausted, no more ideas, or an affirmative decision

## Support

A cluster of one is an anecdote. Say the size every time you name a cluster, and when a
cluster has one or two members say explicitly that it is under-supported rather than reporting
it beside a cluster of nine as though they were comparable. The cohort here is ten cases;
almost everything will be under-supported, and saying so is the finding.

## Naming

Name a cluster by its mechanism, in a sentence that says what would have to change:
"the witness was in an imaging report and imaging was never searched" is a name a fix can be
aimed at. "Primary site errors" is a bucket, not a cause.
```

- [ ] **Step 3: 写 `assets/skills/eval-missed-evidence/SKILL.md`**

```markdown
---
name: eval-missed-evidence
description: Use when the answer key says a value is documented but the run reported it absent or wrong, and you must find why the text was never reached. Tells you the four places a retrieval failure can sit, how to distinguish never-searched from searched-and-missed from read-and-misjudged, and what evidence from the trace each conclusion requires. Does not establish that the value is in the chart - that comes from the answer key and the scorer.
slot: eval
judges: [retrieval_failure_locus, term_coverage, type_filter_effect]
license: MIT
---

# The answer was in the chart and the run did not use it

**Confirm from the scorer and the answer key that the value is genuinely documented before you
start.** A run that reported absence on a chart that truly lacks the value is correct
behaviour, and diagnosing it as a miss is how correct abstention gets trained away.

## Four places the failure can sit

Work them in order; each is ruled out by different evidence in the trace.

1. **Never searched.** No term the run issued could have matched the text. Evidence: the list
   of terms in the trace, and the text that carries the value. If no term is a substring of
   the surrounding line, stop here — the rest is moot.
2. **Searched, filtered out.** A term that would have hit was issued with a document-type
   filter that excluded the document holding it. Evidence: the search call's filter argument
   and the document's type.
3. **Hit, not read.** The search returned the document and the run did not open it. Evidence:
   the search result list and the absence of a matching read call. Look at what it opened
   instead and in what order — this is usually a ranking problem.
4. **Read, not used.** The document was read in full and the value was not extracted, or was
   extracted and then discarded. Evidence: the read call, and the reasoning that follows it.
   This is not a retrieval failure and must not be reported as one; it belongs to whoever owns
   the task contract's evidence standing.

## What each conclusion owes

Every locus you name requires the specific trace evidence listed beside it. A conclusion of
"never searched" with no term list quoted is a guess. If the trace does not contain what you
need — for instance a truncated read whose extent is unrecorded — say the locus is
undetermined and say what would settle it.
```

- [ ] **Step 4: 写 `assets/skills/eval-overconfidence/SKILL.md`**

```markdown
---
name: eval-overconfidence
description: Use when a run submitted a definite answer that did not match the answer key, and you need to find what it treated as sufficient. Tells you how to read the evidence a run actually rested on, which evidence patterns precede confident errors, and how to separate a reasoning failure from a contract gap. Does not determine that the answer was wrong - ask the deterministic scorer.
slot: eval
judges: [evidence_sufficiency_reasoning, witness_standing, contract_gap]
license: MIT
---

# A confident answer that did not hold

**The scorer tells you it did not match.** Your question is narrower and more useful: what did
the run treat as enough?

## Read what it rested on, not what it said

Go to the cited evidence first, before the reasoning. The reasoning is a story told after the
reads; the citation is what the answer is actually made of. Four patterns recur:

- **A single witness of the wrong standing.** The cited document mentions the value but is not
  a type the contract lets establish it. The run treated a mention as an establishment.
- **An interim line.** The citation is from a preliminary or pending report whose final version
  says something else, and the run never chased the thread.
- **An inference across two documents.** Neither cited document states the value; the answer is
  a join the run performed. Sometimes correct, never admissible on its own.
- **The right document, the wrong span.** The value cited is real text from a real report about
  a different specimen, date, or entity.

## Reasoning failure or contract gap

These need different owners, and confusing them wastes a fix.

- If the contract clearly covers this case and the run misapplied it: instruction-following
  failure. Say which sentence of the contract the run departed from.
- If the contract does not settle the case — two documents of equal standing disagree and no
  precedence rule applies — then this is a **contract gap** and the correct behaviour was
  `SPEC_INSUFFICIENT`. Report it as a gap and quote the ambiguity. A run punished for a gap it
  correctly walked into learns to guess instead.

Distinguishing these two is the main value of this skill. When you cannot, say which evidence
would settle it.
```

- [ ] **Step 5: 跑围栏测试**

Run: `.venv/bin/pytest tests/test_eval_skill_fence.py tests/test_skills_load.py tests/test_skill_slots.py -q`
Expected: 全 PASS，包括之前红的 `test_there_is_at_least_one_eval_skill`

- [ ] **Step 6: 确认渲染出来是对的**

Run:
```bash
.venv/bin/python -c "
from acr.skills import eval_skills_block
b = eval_skills_block(['eval-contrast-traces', 'eval-missed-evidence'])
print(b[:400])
print('...')
print('bytes:', len(b.encode()))
"
```
Expected: 开头是 `DIAGNOSTIC METHOD — HOW TO FIND A CAUSE. YOU DO NOT SCORE.`，两张卡的标题行都带 `may judge:`

- [ ] **Step 7: 提交**

```bash
git add assets/skills/eval-contrast-traces assets/skills/eval-cluster-failures assets/skills/eval-missed-evidence assets/skills/eval-overconfidence
git commit -m "$(cat <<'EOF'
Four diagnostic skills for the evaluation agent

One skill per way of asking "why did this run go wrong": contrast two traces, cluster a
cohort's failures, locate a retrieval miss, read what a confident wrong answer rested on.
Adding a fifth angle is adding a directory. None of them scores; each names the sub-questions
it may judge and sends the rest to the deterministic scorer.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `acr signal run` —— 单个跑，两种做法一个出口

**Files:**
- Create: `src/acr/cli_signal.py`
- Modify: `src/acr/cli.py`（挂 typer）
- Modify: `src/acr/attribution.py:1790-1816`（注入评测卡）
- Test: `tests/test_cli_signal.py`（新建）

**Interfaces:**
- Consumes: Task 5 的 `eval_skills_block()`、`eval_skill_judges()`
- Produces:
  - `acr.cli_signal.signal_app: typer.Typer`
  - `acr.cli_signal.KINDS = ("rule", "agent")`
  - CLI：`acr signal run --kind rule|agent --run <manifest> ...`

**为什么新建一个组而不是塞进 `acr eval`**：`cli_eval.py` 的模块 docstring 承诺这个组不调模型，`evals.py` 有 AST 闭包测试钉着。AI 复盘要调模型，进去就破了承诺。新组是薄壳，`--kind rule` 转发给现有 `evals`，`--kind agent` 转发给现有 `attribution`，两边都已经测过。

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_cli_signal.py`：

```python
"""`acr signal` 是问信号的唯一入口，程序算的和 AI 看的都从这里进。

它必须是薄的：转发给已经测过的 `evals` 和 `attribution`，自己不含判分逻辑。特别是
`--kind rule` 这条路必须一个模型都不碰——否则"无模型评测面"就只是文档里的一句话。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acr.cli_signal import KINDS, signal_app

SRC = Path(__file__).resolve().parents[1] / "src"
runner = CliRunner()


def test_both_kinds_are_offered():
    assert KINDS == ("rule", "agent")


def test_run_help_names_both_kinds():
    res = runner.invoke(signal_app, ["run", "--help"])
    assert res.exit_code == 0
    assert "rule" in res.stdout and "agent" in res.stdout


def test_unknown_kind_is_refused():
    res = runner.invoke(signal_app, ["run", "--kind", "vibes", "--run", "x.manifest.json"])
    assert res.exit_code != 0
    assert "vibes" in res.stdout


def test_module_imports_no_provider_at_module_scope():
    """薄壳的代价必须是零：模型侧的 import 只在 --kind agent 的分支里发生。

    `acr eval` 组承诺不调模型。如果这个新组在模块层面就 import 了 litellm，任何人 import
    cli 都会把 provider 拖进来，那条承诺在实践中就没了。
    """
    tree = ast.parse((SRC / "acr" / "cli_signal.py").read_text(encoding="utf-8"))
    top: set[str] = set()
    for node in tree.body:                      # 只看模块层，函数体内的延迟 import 不算
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    forbidden = {"litellm", "langchain", "langgraph", "deepagents", "openai", "anthropic"}
    assert not top & forbidden, f"module-scope provider import: {top & forbidden}"


def test_signal_envelope_shape_is_the_contract():
    from acr.kernel import SIGNAL_TYPES
    from acr.cli_signal import SIGNAL_TYPE_FOR_KIND
    assert set(SIGNAL_TYPE_FOR_KIND.values()) <= SIGNAL_TYPES
    assert set(SIGNAL_TYPE_FOR_KIND) == set(KINDS)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_cli_signal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'acr.cli_signal'`

- [ ] **Step 3: 写 `src/acr/cli_signal.py`**

```python
"""One place to ask a completed run for signals, whichever way the signal is produced.

TWO WAYS, ONE OUTPUT
--------------------
A signal about a run comes from one of two places and they could not be less alike:

  RULE   deterministic checks over the trace and the manifest. Same input, same output,
         forever. `acr eval` and `acr audit` already do this and no model is reachable from
         either — `tests/test_evals.py::test_no_model_is_reachable_from_this_module` walks the
         import graph of `evals.py` and fails if one appears.
  AGENT  a model reads the work log and says why something happened. `acr attribute` already
         does this, under a tool surface that gives it no way to assert a verdict.

Both emit a `SignalEnvelope`, so whatever consumes signals consumes one shape.

WHY THIS IS A NEW GROUP AND NOT A FLAG ON `acr eval`
----------------------------------------------------
`cli_eval` opens by promising that nothing in it calls a model. Adding `--kind agent` there
would make that sentence false while leaving it on the page. So this group is a thin dispatcher
over the two existing surfaces, and the provider-side imports happen inside the agent branch —
`tests/test_cli_signal.py::test_module_imports_no_provider_at_module_scope` keeps them there.

WHAT THIS MODULE MUST NEVER GROW
--------------------------------
Scoring. If a question can be settled by comparing two values, it belongs in `evals.py` where
it is deterministic and testable. A dispatcher that starts deciding correctness is a second
answer to a question that already has one.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from .cli_common import con

signal_app = typer.Typer(add_completion=False, help=(
    "Ask a completed run for signals. --kind rule runs the deterministic checks and calls no "
    "model; --kind agent runs the diagnostic agent, whose method comes from eval skills."))

#: The two ways a signal is produced. Not an open set: a third would need its own guarantee
#: about what it may and may not decide.
KINDS: tuple[str, ...] = ("rule", "agent")

#: Which `SignalEnvelope.signal_type` each kind emits. Both are already in `kernel.SIGNAL_TYPES`.
SIGNAL_TYPE_FOR_KIND: dict[str, str] = {
    "rule": "EVALUATION_RESULT",
    "agent": "ATTRIBUTION_REPORT",
}

#: The eval skills the diagnostic agent is offered by default. Every one is `slot: eval` and
#: declares what it may judge; see `acr.skills.eval_skills_block`.
DEFAULT_EVAL_SKILLS: tuple[str, ...] = (
    "eval-contrast-traces",
    "eval-cluster-failures",
    "eval-missed-evidence",
    "eval-overconfidence",
)


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise typer.BadParameter(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    return kind


def _eval_skill_names(raw: str) -> tuple[str, ...]:
    """Which diagnostic skills to offer. Empty string means the default set."""
    if not raw.strip():
        return DEFAULT_EVAL_SKILLS
    return tuple(s.strip() for s in raw.split(",") if s.strip())


@signal_app.command("run")
def signal_run(
    kind: str = typer.Option(..., "--kind", help=f"one of {list(KINDS)}"),
    run: str = typer.Option(..., "--run", help="one *.manifest.json from a completed chart run"),
    spec: str = typer.Option(..., "--spec", "-s", help="the spec that run was made under"),
    gold: str = typer.Option("", "--gold", help="answer key; agent kind only, enables contrast"),
    case_id: str = typer.Option("", "--case-id", help="pseudonymous case id; agent kind only"),
    eval_skills: str = typer.Option(
        "", "--eval-skills",
        help="comma list of eval skills to offer the agent; default is all four"),
    out: str = typer.Option("", "--out", help="write the signal JSON here instead of stdout"),
):
    """Produce signals for ONE completed run.

    The deterministic kind reads the trace and manifest and calls no model. The agent kind
    reads the same files, plus the answer key when one is supplied, and returns a diagnosis —
    never a verdict, because it has no tool that emits one.
    """
    _check_kind(kind)
    if kind == "rule":
        payload = _rule_signal(run=run, spec=spec)
    else:
        payload = _agent_signal(run=run, spec=spec, gold=gold, case_id=case_id,
                                eval_skills=_eval_skill_names(eval_skills))
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        con.print(f"→ {out}")
    else:
        con.print_json(text)


def _rule_signal(*, run: str, spec: str, subject_id: str = "",
                 provider_boundary: str = "UNKNOWN", local_root: str | None = None) -> dict:
    """Deterministic checks over one run. No model is imported on this path.

    Delegates to `cli_audit.audit_run_payload`, which is the body of `acr audit run` extracted
    for reuse. Rebuilding the AuditContext here would be a second place where a trajectory is
    assembled from a manifest, and the two would drift.
    """
    from . import evals
    from .cli_audit import audit_run_payload

    report = audit_run_payload(manifest=run, subject_id=subject_id,
                               provider_boundary=provider_boundary, local_root=local_root)
    return {
        "schema": "acr.signal/1",
        "signal_type": SIGNAL_TYPE_FOR_KIND["rule"],
        "kind": "rule",
        "run": run,
        "spec": spec,
        "deterministic": True,
        # `rule_compliance` is deterministic in the registry but can never fire:
        # `answer_checks.ANSWER_CHECK_KINDS` has been empty since 2026-07-30, when all five
        # clinical checks were measured and removed (58 firings destroyed a correct value).
        # Advertising it here would claim a check that no run receives.
        "dimensions": [d.name for d in evals.REGISTRY.values()
                       if d.deterministic and d.name != "rule_compliance"],
        "report": report,
    }


def _agent_signal(*, run: str, spec: str, gold: str, case_id: str,
                  eval_skills: tuple[str, ...]) -> dict:
    """The diagnostic agent over one run. Provider imports live here, not at module scope."""
    from .skills import eval_skills_block

    block = eval_skills_block(list(eval_skills))     # validates slot and `judges` before spending
    con.print(f"[dim]{len(eval_skills)} eval skills, {len(block.encode('utf-8'))} bytes[/]")
    from .cli_attribute import attribute_case_payload
    return attribute_case_payload(
        run=run, spec=spec, gold=gold, case_id=case_id, eval_skills_prompt=block,
        signal_type=SIGNAL_TYPE_FOR_KIND["agent"])
```

- [ ] **Step 4: 从 `cli_audit.py` 里抽出 `audit_run_payload`**

`acr audit run`（`src/acr/cli_audit.py:63-121`）的函数体现在直接 `con.print_json` 并在有
incident 时 `raise typer.Exit(2)`。把中间那段搬进一个可复用函数，命令改为调用它：

```python
def audit_run_payload(*, manifest: str, subject_id: str = "",
                      provider_boundary: str = "UNKNOWN",
                      declared_tool: tuple[str, ...] = (),
                      rule: tuple[str, ...] = (),
                      local_root: str | None = None) -> dict:
    """Run the truth-blind audit over one manifest and return the report as a dict.

    Split out of the `run` command so `acr signal run --kind rule` reaches the same
    AuditContext construction rather than assembling a second one. Two places that build a
    trajectory from a manifest is two places that can disagree about what the run did.

    `subject_id` defaults to the manifest's own `patient_id`; the command still requires it
    explicitly because an operator naming the wrong subject is a boundary error, whereas a
    dispatcher reading it from the file it was handed is not.
    """
```

函数体沿用 63-118 行现有逻辑（`_store` → `require_input` → `_rows` → `TrajectoryAdapter.
from_run_artifacts` → `AuditRunner(...).run(...)`），末尾 `return report.to_dict()`。
`patient_scope` 那行改成 `str(raw.get("patient_id") or subject_id)`，保持原有优先级。

`run` 命令体替换为：

```python
    report = audit_run_payload(
        manifest=manifest, subject_id=subject_id, provider_boundary=provider_boundary,
        declared_tool=tuple(declared_tool), rule=tuple(rule), local_root=local_root)
    con.print_json(json.dumps(report, ensure_ascii=False))
    if report.get("incidents"):
        raise typer.Exit(2)
```

跑 `.venv/bin/pytest tests/ -q -k audit` 确认审计相关测试仍通过。

- [ ] **Step 5: 在 `cli_attribute.py` 里加 `attribute_case_payload`**

`_run_one` 已经封装了单例执行。在 `attribute_case` 命令函数之后加一个可复用的函数（把命令体里那段搬出来，命令改为调用它），签名：

```python
def attribute_case_payload(*, run: str, spec: str, gold: str, case_id: str,
                           eval_skills_prompt: str = "",
                           signal_type: str = "ATTRIBUTION_REPORT",
                           corpus: str = "corpus/patients", model: str = "",
                           api_base: str = "", max_model_calls: int = 12,
                           max_usd: float = 1.0, max_chart_reads: int = 12,
                           local_root: str | None = None) -> dict:
    """Attribute one run and return the report as a signal-shaped dict.

    Split out of `attribute_case` so `acr signal run --kind agent` reaches the same code path
    rather than a parallel one. A second path to a diagnosis is a second thing to keep honest.
    """
```

函数体沿用 `attribute_case` 现有逻辑（`_store` → `_case_map` → `_packet` → `_run_one`），把 `eval_skills_prompt` 透传给 `_run_one`，返回：

```python
    return {"schema": "acr.signal/1", "signal_type": signal_type, "kind": "agent",
            "run": run, "spec": spec, "deterministic": False,
            "eval_skills_bytes": len(eval_skills_prompt.encode("utf-8")),
            "report": report.to_dict()}
```

`_run_one` 加一个 `eval_skills_prompt: str = ""` 参数并透传给 `attribution` 的运行入口。

- [ ] **Step 6: 在 `attribution.py` 里注入评测卡**

`src/acr/attribution.py:1790-1816`，`module_instructions` 拼装之后、提示词模板之前，加：

```python
    # Eval skills are METHOD, and they sit beside the stage instructions rather than replacing
    # them: a stage says what this run of the evaluator must produce, a skill says how a
    # careful reviewer goes about finding it. Neither may score — the scorer is a tool.
    eval_block = eval_skills_prompt.strip()
```

在模板里 `{module_instructions}` 之后加一行 `{eval_block}`，并给运行入口加 `eval_skills_prompt: str = ""` 参数一路透传。

- [ ] **Step 7: 挂上 typer**

`src/acr/cli.py`，在 `app.add_typer(eval_app, name="eval")`（97 行）之后加：

```python
from .cli_signal import signal_app
app.add_typer(signal_app, name="signal")
```

（import 放到文件顶部的 import 区，与其它 `cli_*` import 同处。）

- [ ] **Step 8: 跑测试**

Run: `.venv/bin/pytest tests/test_cli_signal.py tests/test_cli_composition.py tests/test_cli_eval_plane.py tests/test_attribution.py -q`
Expected: PASS

- [ ] **Step 9: 确认 CLI 挂上了**

Run: `.venv/bin/acr signal run --help`
Expected: 帮助文本里有 `--kind`、`--run`、`--eval-skills`

- [ ] **Step 10: 跑全套并提交**

Run: `.venv/bin/pytest -q`

```bash
git add src/acr/cli_signal.py src/acr/cli.py src/acr/cli_attribute.py src/acr/attribution.py tests/test_cli_signal.py
git commit -m "$(cat <<'EOF'
acr signal run: one entry for both kinds of signal, single case

Rule-based checks and the diagnostic agent answer different questions about the same run and
had separate front doors. This is a thin dispatcher over both existing surfaces. It is a new
group rather than a flag on `acr eval` because that group promises it calls no model, and a
test keeps provider imports out of this module's scope so the promise stays true in practice.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `acr signal batch` —— 批量跑

**Files:**
- Modify: `src/acr/cli_signal.py`
- Test: `tests/test_cli_signal.py`（追加）

**Interfaces:**
- Consumes: Task 7 的 `_rule_signal()`、`_agent_signal()`、`KINDS`、`SIGNAL_TYPE_FOR_KIND`
- Produces: CLI `acr signal batch --kind rule|agent --runs <dir|file> ...`；输出一个 JSON 数组，一个 run 一项

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_cli_signal.py`：

```python
def test_batch_help_names_both_kinds():
    res = runner.invoke(signal_app, ["batch", "--help"])
    assert res.exit_code == 0
    assert "--runs" in res.stdout


def test_batch_collects_manifests_from_a_directory(tmp_path: Path):
    from acr.cli_signal import _manifest_paths
    (tmp_path / "a.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    got = _manifest_paths(str(tmp_path))
    assert [p.name for p in got] == ["a.manifest.json", "b.manifest.json"]


def test_batch_accepts_a_single_file(tmp_path: Path):
    from acr.cli_signal import _manifest_paths
    f = tmp_path / "only.manifest.json"
    f.write_text("{}", encoding="utf-8")
    assert _manifest_paths(str(f)) == [f]


def test_batch_refuses_an_empty_directory(tmp_path: Path):
    from acr.cli_signal import _manifest_paths
    with pytest.raises(typer.BadParameter, match="no \\*.manifest.json"):
        _manifest_paths(str(tmp_path))


def test_one_failure_does_not_abort_the_batch(tmp_path: Path, monkeypatch):
    """一个 run 炸了不能让整批丢掉——已经花掉的钱和已经算出的信号都还在。"""
    import acr.cli_signal as cs
    ok = tmp_path / "ok.manifest.json"; ok.write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.manifest.json"; bad.write_text("{}", encoding="utf-8")

    def fake(*, run, spec):
        if "bad" in run:
            raise RuntimeError("boom")
        return {"kind": "rule", "run": run}

    monkeypatch.setattr(cs, "_rule_signal", fake)
    out = cs._batch_signals(kind="rule", paths=[ok, bad], spec="s.yaml",
                            gold="", case_map={}, eval_skills=())
    assert len(out) == 2
    assert out[0]["run"].endswith("ok.manifest.json")
    assert out[1]["error"] == "RuntimeError: boom"
```

测试文件顶部需要 `import typer`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_cli_signal.py -q`
Expected: FAIL — `cannot import name '_manifest_paths'`

- [ ] **Step 3: 实现**

追加到 `src/acr/cli_signal.py`：

```python
def _manifest_paths(runs: str) -> list[Path]:
    """One manifest, or every manifest in a directory, sorted so two batches line up."""
    p = Path(runs)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise typer.BadParameter(f"{runs}: not a file or directory")
    found = sorted(p.rglob("*.manifest.json"))
    if not found:
        raise typer.BadParameter(f"{runs}: no *.manifest.json below it")
    return found


def _batch_signals(*, kind: str, paths: list[Path], spec: str, gold: str,
                   case_map: dict[str, str], eval_skills: tuple[str, ...]) -> list[dict]:
    """Signals for every run, in path order.

    A failure on one run is recorded and the batch continues. Aborting would discard the
    signals already produced — and on the agent kind, the money already spent producing them.
    """
    out: list[dict] = []
    for path in paths:
        try:
            if kind == "rule":
                out.append(_rule_signal(run=str(path), spec=spec))
            else:
                out.append(_agent_signal(
                    run=str(path), spec=spec, gold=gold,
                    case_id=case_map.get(path.stem, path.stem), eval_skills=eval_skills))
        except Exception as exc:                # noqa: BLE001 - one bad run is not the batch
            con.print(f"[red]{path.name}: {type(exc).__name__}: {exc}[/]")
            out.append({"schema": "acr.signal/1", "run": str(path), "kind": kind,
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


@signal_app.command("batch")
def signal_batch(
    kind: str = typer.Option(..., "--kind", help=f"one of {list(KINDS)}"),
    runs: str = typer.Option(..., "--runs",
                             help="a *.manifest.json, or a directory searched recursively"),
    spec: str = typer.Option(..., "--spec", "-s"),
    gold: str = typer.Option("", "--gold", help="answer key; agent kind only"),
    case_map: str = typer.Option("", "--case-map",
                                 help="JSON file mapping manifest stem -> case id; agent only"),
    eval_skills: str = typer.Option("", "--eval-skills",
                                    help="comma list; default is all four"),
    out: str = typer.Option("", "--out", help="write the JSON array here instead of stdout"),
):
    """Produce signals for MANY completed runs. One bad run is recorded, not fatal."""
    _check_kind(kind)
    paths = _manifest_paths(runs)
    mapping = json.loads(Path(case_map).read_text(encoding="utf-8")) if case_map else {}
    con.print(f"[dim]{len(paths)} runs, kind={kind}[/]")
    signals = _batch_signals(kind=kind, paths=paths, spec=spec, gold=gold,
                             case_map=mapping, eval_skills=_eval_skill_names(eval_skills))
    text = json.dumps(signals, indent=2, ensure_ascii=False, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        con.print(f"→ {out}  ({sum('error' in s for s in signals)} failed)")
    else:
        con.print_json(text)
```

- [ ] **Step 4: 跑测试**

Run: `.venv/bin/pytest tests/test_cli_signal.py -q`
Expected: PASS

- [ ] **Step 5: 确认两个子命令都在**

Run: `.venv/bin/acr signal --help`
Expected: 列出 `run` 和 `batch`

- [ ] **Step 6: 跑全套**

Run: `.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 7: 更新 README**

在 `README.md` §3 "How to run each component" 里，`acr extract` 那段之后，加一节（照抄本计划末尾"怎么用"的四个例子），并在 §2.6 eval 平面的描述里加一句：`acr signal` 是信号的统一入口，`--kind rule` 无模型、`--kind agent` 走 eval skills（Task 9 再加第三种 `--kind judge`）。

- [ ] **Step 8: 提交**

```bash
git add src/acr/cli_signal.py tests/test_cli_signal.py README.md
git commit -m "$(cat <<'EOF'
acr signal batch: signals over a cohort, one bad run recorded not fatal

Aborting a batch on one failure discards the signals already produced, and on the agent kind
the money already spent producing them. Failures land in the output array beside the successes
so the count of each is visible without re-running.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `--kind judge` —— AI 评卷进统一入口

**Files:**
- Modify: `src/acr/cli_judge.py:76-90`（`_JsonModel` 改名公开）
- Modify: `src/acr/cli_signal.py`（`KINDS` 加第三种、`_judge_signal`、`run`/`batch` 加评卷参数）
- Test: `tests/test_cli_signal.py`（追加 + 改一处断言）

**Interfaces:**
- Consumes: `acr.judge` 现成的机器——`judge()`、`blind_packet()`、`keyed_packet()`、
  `JUDGEABLE_DIMENSIONS`、`KEY_PERMITTED_DIMENSIONS`、`apply_verdict()`；
  `evals.precedence_gate()`；`cli_common.llm_client()`；Task 7 的 `_check_kind` 分发结构
- Produces:
  - `acr.cli_judge.JsonJudgeModel`（原 `_JsonModel` 改名，保留 `_JsonModel = JsonJudgeModel` 别名）
  - `acr.cli_signal._judge_signal(...) -> dict`
  - CLI：`acr signal run --kind judge --dimension trajectory_quality --run ... --usd-per-call ...`

**这条线评什么、归谁管**：`judge.py` 是现成的"评卷员"（agent-as-a-judge，评 trajectory
本身，不归因）。五个可评维度（翻病历的路子、证据支撑的判断半、步骤效率的判断半、解释
质量、坏案例排序），每维度三个内置镜头。围栏、蒙答案（`BlindPacket`）、"观感分不当闸门"
（`apply_verdict` 只认 SCREEN/RANK/FLAG）全部已经在 `judge.py` 里钉死——**本任务一个字
都不改它**，只做两件事：把 run manifest+trace 自动组装成 packet（现有 `acr judge panel`
要求操作者手拼 JSON，这是真实的使用门槛），和把这条线纳入 `acr signal` 统一入口。

**扩展方式随之明确**：加一个评卷角度 = 加一个 `assets/evaluators/*.yaml`（`judge.py` 自己的
规矩：内置 LENSES 字典不许再长，新评测走声明式 YAML，加载时对着判分注册表核查）。
诊断角度加 `assets/skills/eval-*`，评卷角度加 `assets/evaluators/*.yaml`，都不改代码。

- [ ] **Step 1: 写失败的测试**

`tests/test_cli_signal.py` 里，把 `test_both_kinds_are_offered` 改成：

```python
def test_all_three_kinds_are_offered():
    assert KINDS == ("rule", "judge", "agent")
```

并追加：

```python
def test_judge_kind_requires_a_dimension():
    res = runner.invoke(signal_app, ["run", "--kind", "judge",
                                     "--run", "x.manifest.json", "--spec", "s.yaml"])
    assert res.exit_code != 0
    assert "--dimension" in res.stdout


def test_judge_signal_builds_a_blind_packet_for_blinded_dimensions(tmp_path: Path):
    """蒙着答案不是嘱咐，是包上没有放答案的口袋。给了 --gold 也不能漏进去。"""
    import acr.cli_signal as cs
    from acr import judge as J
    m = tmp_path / "r.manifest.json"
    m.write_text(json.dumps({"patient_id": "SYN01"}), encoding="utf-8")
    (tmp_path / "r.jsonl").write_text('{"event": "run_start"}\n', encoding="utf-8")
    g = tmp_path / "gold.json"
    g.write_text(json.dumps({"SYN01": {"primary_site": "C341"}}), encoding="utf-8")
    packet = cs._packet_from_run(run=str(m), gold=str(g), dimension="trajectory_quality")
    assert isinstance(packet, J.BlindPacket)          # keyed 包根本没被构造


def test_judge_signal_allows_the_key_only_for_triage(tmp_path: Path):
    import acr.cli_signal as cs
    from acr import judge as J
    m = tmp_path / "r.manifest.json"
    m.write_text(json.dumps({"patient_id": "SYN01"}), encoding="utf-8")
    g = tmp_path / "gold.json"
    g.write_text(json.dumps({"SYN01": {"primary_site": "C341"}}), encoding="utf-8")
    packet = cs._packet_from_run(run=str(m), gold=str(g), dimension="bad_case_triage")
    assert isinstance(packet, J.KeyedPacket)
```

测试文件顶部需要 `import json`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_cli_signal.py -q`
Expected: FAIL — `KINDS` 还是两元组，`_packet_from_run` 不存在

- [ ] **Step 3: `_JsonModel` 改名公开**

`src/acr/cli_judge.py:76` 附近，类改名 `JsonJudgeModel`，类定义之后加一行
`_JsonModel = JsonJudgeModel`（模块内两处使用可以不动）。改名的理由写在类 docstring 尾部：

```python
    Public because `cli_signal` builds the same adapter for --kind judge; a second JSON-mode
    adapter would be a second place for the parsing rules to drift.
```

- [ ] **Step 4: 实现 `_judge_signal` 和参数**

`src/acr/cli_signal.py`：`KINDS` 改为 `("rule", "judge", "agent")`，
`SIGNAL_TYPE_FOR_KIND` 加 `"judge": "EVALUATION_RESULT"`（评卷也是 EVALUATION_RESULT，
和 rule 的区别由 `deterministic: false` 和 `evidence_class: JUDGED` 字段表达——这正是
`judge.py` "两种分数不许平均"的约定在信号层的样子）。

追加：

```python
def _packet_from_run(*, run: str, gold: str, dimension: str):
    """Assemble a judge packet from a run's manifest and trace.

    The blind/keyed decision is NOT made here by policy — it falls out of `judge.py`'s
    constants. For a blinded dimension the gold argument is ignored ENTIRELY: the packet type
    has no field to carry it, which is the isolation working as designed.
    """
    from . import judge as J

    manifest_path = Path(run)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_path = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".jsonl"))
    trace = ([json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if trace_path.is_file() else [])
    subject = str(manifest.get("patient_id") or manifest_path.stem)
    if dimension in J.KEY_PERMITTED_DIMENSIONS and gold:
        key = json.loads(Path(gold).read_text(encoding="utf-8"))
        return J.keyed_packet(trace=trace,
                              artifacts={"manifest": manifest, "answer_key": key},
                              subject_id=subject)
    return J.blind_packet(trace=trace, artifacts={"manifest": manifest}, subject_id=subject)


def _judge_signal(*, run: str, spec: str, dimension: str, gold: str,
                  usd_per_call: float, max_usd: float, model: str, api_base: str) -> dict:
    """One judged verdict as a signal. Refusals are judge()'s own, reported verbatim."""
    from . import evals
    from . import judge as J
    from .cli_common import llm_client
    from .cli_judge import JsonJudgeModel

    if not dimension:
        raise typer.BadParameter("--kind judge requires --dimension "
                                 f"(one of {list(J.JUDGEABLE_DIMENSIONS)})")
    n_lenses = len(J.LENSES[dimension]) if dimension in J.LENSES else 3
    planned = round(n_lenses * usd_per_call, 6)
    if planned > max_usd:
        raise typer.BadParameter(f"{n_lenses} lenses x ${usd_per_call} = ${planned} exceeds "
                                 f"--max-usd {max_usd}; nothing was called")
    packet = _packet_from_run(run=run, gold=gold, dimension=dimension)
    verdict = J.judge(dimension, packet, registry=evals.precedence_gate(),
                      model=JsonJudgeModel(llm_client(model, api_base), model))
    return {
        "schema": "acr.signal/1",
        "signal_type": SIGNAL_TYPE_FOR_KIND["judge"],
        "kind": "judge",
        "run": run,
        "spec": spec,
        "deterministic": False,
        "evidence_class": "JUDGED",   # judge.py's rule, restated where consumers read it:
                                      # this number screens and ranks; it never gates and it
                                      # never averages with a deterministic score
        "dimension": dimension,
        "verdict": verdict.to_dict() if hasattr(verdict, "to_dict") else vars(verdict),
    }
```

`signal_run` 加参数（放在 `eval_skills` 之后）：

```python
    dimension: str = typer.Option("", "--dimension",
                                  help="judge kind only: which judged dimension to ask"),
    usd_per_call: float = typer.Option(0.05, "--usd-per-call",
                                       help="judge kind only: price of one lens call"),
    max_usd: float = typer.Option(1.0, "--max-usd", help="judge/agent kinds: cost ceiling"),
    model: str = typer.Option("", "--model", help="judge kind only: judge model"),
    api_base: str = typer.Option("", "--api-base"),
```

分发处加一支：

```python
    elif kind == "judge":
        payload = _judge_signal(run=run, spec=spec, dimension=dimension, gold=gold,
                                usd_per_call=usd_per_call, max_usd=max_usd,
                                model=model, api_base=api_base)
```

`signal_batch` 同样加这几个参数，`_batch_signals` 的分发处同样加 judge 一支（签名加
`dimension/usd_per_call/max_usd/model/api_base` 透传）。

- [ ] **Step 5: 跑测试**

Run: `.venv/bin/pytest tests/test_cli_signal.py tests/test_judge.py -q`
Expected: PASS（`test_judge.py` 是回归确认：本任务没碰 `judge.py`）

- [ ] **Step 6: 确认帮助与拒绝路径**

Run: `.venv/bin/acr signal run --kind judge --run x.manifest.json --spec s.yaml 2>&1 | tail -3`
Expected: 报 `--dimension` 缺失，并列出五个可评维度

- [ ] **Step 7: 跑全套并提交**

Run: `.venv/bin/pytest -q`

```bash
git add src/acr/cli_signal.py src/acr/cli_judge.py tests/test_cli_signal.py
git commit -m "$(cat <<'EOF'
acr signal --kind judge: the trajectory judge joins the one entry point

judge.py already had the fenced agent-as-a-judge — five judgeable dimensions, three lenses
each, key-blinding in the packet type. What it lacked was ergonomics: panel takes a hand-built
JSON packet. The signal entry assembles the packet from a run's manifest and trace, and the
blind/keyed decision falls out of judge.py's own constants rather than being re-decided here.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: 每一次读都记下"为什么读它"

**Files:**
- Modify: `src/acr/tools/toolbox.py`（`TOOL_SCHEMAS` 三处、`dispatch`）
- Modify: `src/acr/trace.py:441`（`Tracer.tool` 把成因提到顶层字段）
- Modify: `src/acr/evals.py`（加一个检测器 + 挂进 `run_detectors`）
- Test: `tests/test_read_causality.py`（新建）

**Interfaces:**
- Consumes: 无（独立于 Task 1-9，可以先做）
- Produces:
  - `acr.tools.toolbox.CAUSE_PARAM = "because"`——工具参数名，一处定义
  - `Toolbox.dispatch` 返回值不变；`Toolbox.last_cause: str` 供发射端读取
  - `Tracer.tool(name, args, result, ok, ms, because="")`——`because` 升为顶层字段
  - `acr.evals.detect_uncaused_reads(run) -> list[Finding]`，detector 名 `uncaused_read`

**为什么做这个、为什么排第一**：轨迹现在是一串平铺事件，信封只有
`run_id / seq / ts / elapsed_s / kind`，**事件之间没有连线**。归因 agent 第一步是"重建
工作记录，分清记录证明的和推测的"——它现在只能靠**相邻位置猜**因果：第 7 步搜索返回了
D12，第 9 步读了 D12，所以大概是因为那次搜索。中间穿插别的事件、或者模型其实是照着未决
线索去读的，这个推断就错，而**报告不会说它是猜的**。加一个字段，归因第一步从推断变成查表。

改动之所以便宜，是因为 `Tracer.tool` 已经把整个 `args` 写进事件了——模型填了就自动留痕。
本任务做三件事：把参数**加进工具说明书**（不问模型就不会填）、**提到顶层**（免得归因和检测器
去 `args` 里挖）、**数出来**（有成因的读占多大比例，本身就是一个信号）。

**这不是硬控制。** 不填 `because` 不会被拒——它是记录，不是闸门。理由和 `skills` 一样：
成因是模型的判断，判断可以缺席，缺席要能被看见。检测器报的是比例，不是违规。

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_read_causality.py`：

```python
"""一次读要说明它为什么发生，否则归因只能靠相邻位置猜。

轨迹是平铺的事件序列，事件之间没有连线。归因 agent 被要求"分清记录证明的和推测的"，
但在没有成因字段的轨迹上，"第 9 步读 D12 是因为第 7 步搜索返回了它"永远是推测——
而报告不会说它是推测。这个字段把它变成记录。

不填不拒：成因是判断，判断可以缺席，缺席要能被数出来。
"""
from __future__ import annotations

import pytest

from acr.evals import Finding, RunRecord, detect_uncaused_reads, run_detectors
from acr.tools.toolbox import CAUSE_PARAM, TOOL_SCHEMAS


def _schema(name: str) -> dict:
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == name:
            return s["function"]["parameters"]["properties"]
    raise AssertionError(f"no tool {name!r}")


@pytest.mark.parametrize("tool", ["read_document", "read_documents_batch", "search_notes"])
def test_retrieval_tools_ask_for_a_cause(tool: str):
    assert CAUSE_PARAM in _schema(tool), f"{tool} never asks the model why"


@pytest.mark.parametrize("tool", ["read_document", "read_documents_batch", "search_notes"])
def test_cause_is_never_required(tool: str):
    """记录，不是闸门。必填会把判断变成仪式。"""
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == tool:
            assert CAUSE_PARAM not in s["function"]["parameters"].get("required", [])


def test_dispatch_accepts_and_strips_the_cause(toolbox_with_one_doc):
    """`_t_` 方法不必知道这个参数——它在 dispatch 就被摘掉了，一处而不是每个工具一处。"""
    tb = toolbox_with_one_doc
    out, _ms = tb.dispatch("read_document", {"note_id": "N1", CAUSE_PARAM: "search #7 hit"})
    assert "error" not in out
    assert tb.last_cause == "search #7 hit"


def test_dispatch_clears_the_cause_between_calls(toolbox_with_one_doc):
    """上一次的成因不许粘到下一次——那会造出一条没人写过的因果连线。"""
    tb = toolbox_with_one_doc
    tb.dispatch("read_document", {"note_id": "N1", CAUSE_PARAM: "thread T3"})
    tb.dispatch("read_document", {"note_id": "N1"})
    assert tb.last_cause == ""


def test_tracer_promotes_the_cause_to_a_top_level_field(tracer):
    ev = tracer.tool("read_document", {"note_id": "N1"}, {"ok": True}, because="thread T3")
    assert ev["because"] == "thread T3"


def test_detector_counts_reads_with_no_cause():
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document", "because": "search #2 hit"},
        {"kind": "tool", "tool": "read_document", "because": ""},
        {"kind": "tool", "tool": "read_document"},
        {"kind": "tool", "tool": "submit_answer"},          # 不是读，不进分母
    ])
    findings = detect_uncaused_reads(run)
    assert len(findings) == 1
    ev = findings[0].evidence
    assert (ev["n_reads"], ev["n_uncaused"]) == (3, 2)
    assert findings[0].detector == "uncaused_read"


def test_detector_is_silent_when_every_read_has_a_cause():
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document", "because": "search #2 hit"},
    ])
    assert detect_uncaused_reads(run) == []


def test_detector_is_silent_on_a_run_with_no_reads():
    """零阅读是 detect_zero_document_read 的案子，不是这个检测器的——两个检测器报同一件事，
    读的人会以为是两个问题。"""
    assert detect_uncaused_reads(RunRecord(manifest={}, trace=[])) == []


def test_detector_is_wired_into_run_detectors():
    from acr.evals import DetectorConfig
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document"},
    ])
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    assert any(f.detector == "uncaused_read" for f in run_detectors(run, config=cfg))
```

`toolbox_with_one_doc` 和 `tracer` 两个 fixture 加在同文件顶部（不要放进 `conftest.py`
——只有这个文件用）：

```python
@pytest.fixture
def toolbox_with_one_doc(tmp_path):
    from acr.corpus import Corpus
    from acr.coverage import CoverageLedger, ForcedSampler
    from acr.state import EvidenceLedger
    from acr.tools.toolbox import Toolbox
    d = tmp_path / "patients" / "SYN01"
    d.mkdir(parents=True)
    (d / "2024-01-01__pathology__N1.txt").write_text("final diagnosis: adenocarcinoma\n",
                                                     encoding="utf-8")
    chart = Corpus(tmp_path / "patients").chart("SYN01")
    docs, _ = chart.list_documents(limit=100)
    return Toolbox(chart, EvidenceLedger(), CoverageLedger(docs, (), ForcedSampler(1)))


@pytest.fixture
def tracer(tmp_path):
    from acr.trace import Tracer
    return Tracer.create(tmp_path, "t1")
```

**注意**：corpus 的文件名约定要和 `acr.corpus` 实际解析的一致。开工第一件事是读
`src/acr/corpus.py` 确认命名格式，对不上就照实际的改——fixture 造不出 chart，后面每个
测试都会以看不懂的方式失败。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_read_causality.py -q`
Expected: FAIL — `cannot import name 'CAUSE_PARAM'`

- [ ] **Step 3: 工具说明书加参数**

`src/acr/tools/toolbox.py`，`_tool` 定义之后加常量：

```python
#: The parameter every retrieval tool offers so an action can name what prompted it.
#: NEVER in `required`: a cause is a judgement, and a required judgement becomes a ritual —
#: the model writes "to find the answer" on every call and the field measures nothing. What
#: it is for is the OTHER end: `attribution` reconstructing a run currently has to infer
#: "step 9 read D12 because step 7's search returned it" from adjacency, and reports that
#: inference without marking it as one.
CAUSE_PARAM = "because"

_CAUSE_PROPERTY = {
    "type": "string",
    "description": ("optional: what prompted this call — the search that surfaced the "
                    "document, the open thread it settles, the stratum it samples. Recorded, "
                    "never checked; it is how a later reader tells your reasoning from theirs."),
}
```

给 `read_document`、`read_documents_batch`、`search_notes` 三个 schema 的 properties
各加一行 `CAUSE_PARAM: _CAUSE_PROPERTY`（`required` 一律不动）。

**开工前先确认搜索工具的真名**：`grep -n '"search' src/acr/tools/toolbox.py`。上面按
`search_notes` 写，注释里出现过这个名字；实际叫别的就照实际的改，三个测试的参数化列表
也要同步改。

- [ ] **Step 4: `dispatch` 摘掉并记住成因**

`Toolbox.__init__` 末尾加一行：

```python
        #: What the model said prompted the most recent dispatch. Read by the tracing hook
        #: immediately after the call; cleared each dispatch so a stale cause cannot attach
        #: itself to a later action as a causal link nobody wrote.
        self.last_cause: str = ""
```

`dispatch`（187 行）改成：

```python
    def dispatch(self, name: str, args: dict) -> tuple[dict, float]:
        t0 = time.time()
        # Stripped centrally, not accepted per tool: one place to change, and no `_t_` method
        # has to grow a parameter it does not use. Cleared every call — see `last_cause`.
        args = dict(args)
        self.last_cause = str(args.pop(CAUSE_PARAM, "") or "")
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}", "available": [s["function"]["name"] for s in TOOL_SCHEMAS]}, 0.0
        try:
            out = fn(**args)
        except TypeError as e:
            out = {"error": f"bad arguments for {name}: {e}"}
        except KeyError as e:
            out = {"error": f"unknown note_id {e}"}
        except Exception as e:  # noqa: BLE001 - surface tool errors to the model, don't crash the run
            out = {"error": f"{type(e).__name__}: {e}"}
        return out, (time.time() - t0) * 1000
```

- [ ] **Step 5: 轨迹把成因提到顶层**

`src/acr/trace.py:441`：

```python
    def tool(self, name, args, result, ok=True, ms=0.0, because=""):
        # Promoted out of `args` and into the envelope on purpose. It is already inside `args`,
        # but a consumer that has to reach into a free-form argument bag to find the causal
        # link will not do it, and a field nobody reads is a field nobody maintains.
        return self.emit("tool", tool=name, args=args, result=result, ok=ok, ms=ms,
                         because=str(because or ""))
```

在 `agent.py` 调用 `tracer.tool(...)` 的地方（`wrap_tool_call` 钩子里，紧跟
`toolbox.dispatch` 之后）补上 `because=toolbox.last_cause`。开工时用
`grep -n "tracer.tool(" src/acr/agent.py` 定位；有多处就每处都补。

- [ ] **Step 6: 加检测器**

`src/acr/evals.py`，`detect_resource_band` 之后：

```python
#: Tools whose call is a retrieval action and therefore has a reason worth recording.
_READ_TOOLS = ("read_document", "read_documents_batch")


def detect_uncaused_reads(run: RunRecord) -> list[Finding]:
    """How much of this run's reading is causally unexplained in its own record.

    Not a violation — `because` is optional and a model that omits it has broken no rule.
    This counts, because the number is what tells a reader how much of an attribution report
    over this run rests on adjacency rather than on the record. A run at 0% caused reads can
    still be diagnosed; the diagnosis is just weaker, and it should say so.

    Silent on a run with no reads at all: `detect_zero_document_read` owns that case, and two
    detectors reporting one fact reads as two problems.
    """
    reads = [e for e in run.trace
             if e.get("kind") == "tool" and e.get("tool") in _READ_TOOLS]
    if not reads:
        return []
    uncaused = [e for e in reads if not str(e.get("because") or "").strip()]
    if not uncaused:
        return []
    # WARN, and it is the mildest tier this module has — there is no informational level, and
    # inventing one here would put a fourth string into `_SEVERITY_ORDER`'s blind spot, where
    # unknown severities sort to 9 by accident rather than by decision.
    return [Finding(
        "uncaused_read", WARN,
        f"{len(uncaused)} of {len(reads)} reads record no cause; attribution over this run "
        f"must infer their motivation from adjacency",
        {"n_reads": len(reads), "n_uncaused": len(uncaused),
         "caused_fraction": round(1 - len(uncaused) / len(reads), 3)},
    )]
```

`run_detectors` 里加一行 `out += detect_uncaused_reads(run)`。

**严重度只有三档**（`evals.py:386`：`IRB, CRITICAL, WARN = "IRB", "CRITICAL", "WARN"`），
本仓库的检测器一律用**位置参数**构造 `Finding("name", SEVERITY, "message", {...})`——
上面照这个风格写。别引入第四档：`_SEVERITY_ORDER` 只认这三个，未知值回落到 9，排序
就成了偶然而不是决定。

- [ ] **Step 7: 跑测试**

Run: `.venv/bin/pytest tests/test_read_causality.py tests/test_evals.py tests/test_agent_tool_surface.py -q`
Expected: PASS

- [ ] **Step 8: 跑全套并提交**

Run: `.venv/bin/pytest -q`

```bash
git add src/acr/tools/toolbox.py src/acr/trace.py src/acr/evals.py tests/test_read_causality.py
git commit -m "$(cat <<'EOF'
Reads may name what prompted them, and the unexplained ones are counted

The trace is a flat event sequence with no edges. Attribution is asked to separate what the
record proves from what it infers, but "step 9 read D12 because step 7's search returned it"
is only ever adjacency — and the report does not mark it as inferred. `because` is optional
by design: a required judgement becomes a ritual. What is enforced is visibility — the
detector reports the caused fraction, so a diagnosis resting on adjacency says so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: 广度优先 / 深度优先 / 组合——三张检索卡和四臂对照

**Files:**
- Create: `assets/skills/search-breadth-first/SKILL.md`
- Create: `assets/skills/search-depth-first/SKILL.md`
- Create: `assets/skills/search-breadth-then-depth/SKILL.md`
- Create: `docs/BFS_DFS_SEARCH_PILOT.md`（试验协议，结果留空待填）
- Test: 沿用 `tests/test_skills_load.py` / `tests/test_skill_slots.py` 的参数化，无需新测试文件

**Interfaces:**
- Consumes: Task 1 的 `slot:` 约定、Task 3 的 `--skills search=...`
- Produces: 三个可放进 `search` 槽的名字

**依据**：DeepEvidence 的消融结论是**单用广度或单用深度都不够，只有组合最优**。你们的
pilot（`docs/SEARCH_PLANNING_PILOT.md`）比的是另一根轴——计划从哪来（模型自己想 vs
dev set 统计），结论是统计先验没赢。**走法的形状这根轴从没测过。**

**深度优先你们其实已经有了，只是没当成检索策略**：`OpenThreadLedger` + `thread-chasing`
就是顺藤摸瓜，P05 那个 8046 错误（"特殊染色待出"的结论就在同一份文件往后 353 字符处）
是典型的深度优先失败。本任务不改那套机制，只是把"走法"提出来变成可替换的一张卡。

- [ ] **Step 1: 写 `assets/skills/search-breadth-first/SKILL.md`**

```markdown
---
name: search-breadth-first
description: Use when deciding how to traverse a chart and you want coverage before depth - building the full candidate pool of documents that could carry the field before reading any of them closely. Tells you how to sweep the document inventory by type, what to record about each candidate, when a sweep is complete, and what a sweep cannot tell you. Pairs with but does not replace chasing an individual lead.
slot: search
license: MIT
---

# Sweeping wide before reading deep

Build the pool first. One `document_type_summary` call gives you every type, its count and
its date span; a type-filtered search over each candidate type gives you the documents that
could carry the field. Only then start reading.

## The sweep

1. **Inventory by type.** Which types exist in THIS chart, and how many of each. Types the
   contract can never accept still get listed — you need them to say what you excluded.
2. **One search per candidate type.** Same terms, type filter varied. This is what makes the
   result a comparison rather than a walk: a term that hits in pathology and misses in
   imaging tells you something about the term; a term tried only in pathology does not.
3. **Record the pool before reading.** Which documents matched, which type, which date. The
   pool is your denominator, and once you begin reading you will stop being able to say what
   you started with.

## When the sweep is done

A sweep is complete when every type in the inventory has been searched or explicitly
excluded, with a reason for each exclusion. It is not complete because you found a hit —
a hit ends the sweep only if the contract lets that document type establish the field
outright, and even then the rest of the pool is what a later absence claim rests on.

## What a sweep cannot do

It cannot follow a lead. A pathology report saying "stains pending", a note saying "see
outside records", an addendum referenced but not returned — a sweep records these as pool
members and walks on. Each of them is a question the sweep has now RAISED and not answered.

So a sweep alone under-reads exactly the documents that were about to become decisive. When
your pool contains a deferred conclusion, the sweep has done its job and the next move is not
another sweep.
```

- [ ] **Step 2: 写 `assets/skills/search-depth-first/SKILL.md`**

```markdown
---
name: search-depth-first
description: Use when a document points somewhere else and you want to follow the lead to where the question was settled, rather than sampling more of the chart. Tells you which pointers are worth following and in what order, how far to follow one before abandoning it, how to avoid walking in a circle, and what following a lead cannot tell you about the rest of the chart. Pairs with but does not replace a broad sweep.
slot: search
license: MIT
---

# Following one lead to where it ends

A chart is written forward in time by people who did not yet know the answer. Any document
may defer its own conclusion, and the resolving text is usually reachable from the deferral.
This traversal chases that, one thread at a time, to its end.

## Which pointers to follow, in order

1. **Deferral in a document that COULD establish the field.** A pathology report saying
   stains are pending is the highest-value pointer in a chart: the thing it defers is exactly
   the thing you need, and the resolution is usually in the same file or its addendum.
2. **An explicit cross-reference.** "See addendum", "per outside records", "correlate with
   the 3/14 biopsy" — a named destination. Follow the name.
3. **A hedge that a later document would have resolved.** "Favor squamous" invites a later
   definite statement. Search forward in time from that date.

## How far, and when to stop following

Follow one thread until it resolves, or until the next hop would be a guess rather than a
named destination. A resolved thread is worth more than three half-followed ones: the value
is in reaching the settled statement, and a chain abandoned in the middle has cost the reads
and produced nothing citable.

Never revisit a document you have already read in full during this chase — that is the circle
this traversal is prone to. Keep the thread's hops in mind and, if a hop returns you to a
document already read, the thread is exhausted, not continuing.

## What following a lead cannot do

It tells you nothing about the documents no lead pointed at. A chase that ends in a confident,
well-cited answer has still read a narrow slice of the chart, and if your answer claims that
something is ABSENT, the slice is not the basis for that claim — the rest of the chart is,
and you have not looked at it.
```

- [ ] **Step 3: 写 `assets/skills/search-breadth-then-depth/SKILL.md`**

```markdown
---
name: search-breadth-then-depth
description: Use as the default traversal when neither coverage nor lead-chasing alone fits - sweep the document inventory to build a candidate pool, then chase the leads the sweep raised. Tells you the handover point between the two modes, which leads earn a chase and which do not, when to return to the sweep, and how to record which mode produced the decisive read. Combines the breadth-first and depth-first methods rather than choosing between them.
slot: search
license: MIT
---

# Sweep, then chase what the sweep raised

Two traversals, in one order, with a stated handover. Each covers the other's blind spot: a
sweep records deferred conclusions and walks past them; a chase reads a narrow slice and
cannot support a claim of absence.

## Phase 1 — sweep

Inventory the chart by document type. One search per candidate type, same terms, type filter
varied. Record the pool — which documents matched, which type, which date — BEFORE reading
closely. The pool is the denominator any later absence claim rests on.

## The handover

Leave the sweep when it has done what only it can do: every type searched or explicitly
excluded. At that moment you hold two things — a pool of candidates, and a list of the
questions the sweep RAISED without answering. The second list is the input to phase 2.

Do not hand over early because you found a hit. A hit that the contract lets establish the
field outright can end the whole run, but it does not license skipping the rest of the sweep
if your answer will also assert that nothing else was documented.

## Phase 2 — chase

Rank the raised questions by whether the deferral sits in a document type that COULD establish
the field, and chase them in that order. Follow one thread until it resolves or until the next
hop would be a guess. Do not start a second thread while a higher-ranked one is unresolved.

## Returning to the sweep

Go back when a chase turns up a document type absent from your inventory — an outside report,
a scanned addendum filed under an unexpected type. That is new territory, and the sweep is the
tool for territory. Sweep only the new type, then resume the chase where you left it.

## Recording which mode found it

For each read, say whether it came from the sweep or from a chase, and for a chase, which
thread. The whole reason for running two traversals is to learn which one earns its cost, and
a run that cannot say which mode produced the decisive read cannot contribute to that.
```

- [ ] **Step 4: 写试验协议 `docs/BFS_DFS_SEARCH_PILOT.md`**

```markdown
# Breadth / Depth / Combined Search Pilot

## Question

Does the SHAPE of the chart traversal change accuracy — sweeping wide, chasing leads, or
both — holding the clinical contract, the model, the seed and everything else fixed?

`SEARCH_PLANNING_PILOT.md` answered a different question: where the plan COMES FROM (the
model's own vs a dev-set-derived prior). It found the prior did not improve accuracy. Traversal
shape was never varied, and it is the axis DeepEvidence reports an ablation on — that neither
breadth nor depth alone was sufficient and only the combination was best. Whether that holds
for a chart, which is one patient's record rather than a federation of knowledge bases, is
an open question and this pilot is how it gets answered here.

## Arms

Four, differing in exactly one skill:

| Arm | `--skills` |
|---|---|
| native (control) | `search=search-native` |
| breadth-first | `search=search-breadth-first` |
| depth-first | `search=search-depth-first` |
| combined | `search=search-breadth-then-depth` |

Everything else is held: same spec, same corpus, same model, same `--seed 1234`, same
`--runtime-profile`, same cost and call ceilings, same `task` and `general` skill slots.

## Protocol

```bash
for arm in native breadth-first depth-first breadth-then-depth; do
  acr batch --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
            --skills "search=search-$arm" --seed 1234 --out "runs/bfs-$arm"
done

for arm in native breadth-first depth-first breadth-then-depth; do
  acr signal batch --kind rule --runs "runs/bfs-$arm" \
                   --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
                   --out "signals/bfs-$arm.json"
done
```

Note `search-native` is written `search=search-native` while the others follow the loop
variable; the loop above produces `search=search-native` for the first arm because the skill
is named for the arm. Verify the four skill names resolve before spending:
`acr run --help` and one `--dry-run` per arm.

## What is measured

Primary, from the deterministic scorer only — a judged opinion decides nothing here:

| Metric | Why it is in the table |
|---|---|
| three-field exact match | the headline, and the only one that is the task |
| per-field match (site / histology / behavior) | a traversal can help one field and hurt another |
| correct abstentions retained | the failure mode this repo guards hardest: an arm that trades a correct EVIDENCE_INSUFFICIENT for a guess has made things worse at any accuracy |
| documents read per patient | the bill, and breadth's expected cost |
| search calls / read calls | which traversal actually ran, independent of what the skill said |
| open threads left unresolved | depth's expected advantage, measured directly |
| priced model cost | the other bill |
| caused-read fraction (Task 10) | how much of each arm is causally legible afterwards |

## Powering, stated before the run

Ten patients. One chart moves every rate by ten points, so this pilot can detect a large
effect and nothing else. A difference of one or two cases is NOT a result and must be reported
as underpowered — `MIN_PATIENTS_FOR_SUPPORT` is 20 in `assetdev.py` for this reason. What ten
cases CAN do is expose a traversal that is grossly worse, and rule an arm out.

## Results

Not yet run.
```

- [ ] **Step 5: 跑卡片测试**

Run: `.venv/bin/pytest tests/test_skills_load.py tests/test_skill_slots.py -q`
Expected: PASS，三张新卡自动进参数化

- [ ] **Step 6: 确认三张卡都能进 search 槽**

Run:
```bash
.venv/bin/python -c "
from acr.skills import SkillStack, parse_skill_stack
base = SkillStack(general=('coverage-judgement',))
for s in ('search-breadth-first', 'search-depth-first', 'search-breadth-then-depth'):
    print(s, '->', parse_skill_stack(f'search={s}', base).names())
"
```
Expected: 三行，每行第一个名字是对应的卡

- [ ] **Step 7: 提交**

```bash
git add assets/skills/search-breadth-first assets/skills/search-depth-first assets/skills/search-breadth-then-depth docs/BFS_DFS_SEARCH_PILOT.md
git commit -m "$(cat <<'EOF'
Traversal shape as a swappable skill: breadth, depth, and the combination

The earlier pilot varied where the plan came from and found a measured prior did not beat the
model's own planning. It never varied the SHAPE of the traversal, which is the axis
DeepEvidence reports an ablation on — neither breadth nor depth alone sufficed there.

Depth-first already existed in this tree as an obligation that blocks submission
(OpenThreadLedger, thread-chasing); this makes it a traversal a run can be pointed at, so the
three shapes are one --skills flag apart and the comparison is clean.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: 证据集合的结构审计

**Files:**
- Modify: `src/acr/evals.py`
- Test: `tests/test_evidence_set_audit.py`（新建）

**Interfaces:**
- Consumes: Task 10 的 `Finding` 用法（已存在）
- Produces: `acr.evals.audit_evidence_set(run) -> list[Finding]`，三个 detector 名：
  `evidence_span_overlap` / `orphan_contradiction` / `single_witness_field`；
  挂进 `run_detectors`

**依据与它补的洞**：DeepEvidence 对 10 张证据图、316 个条目做人工审计，报了五个数：
出处有效性 100%、规范化 99.7%、重复率 0.6%、关系正确率 ≥99%、**断言一致性 93.3%**。
前四项接近满分、第五项明显掉下来——**落差本身就是结论**：把东西挂对位置几乎不出错，
出错的是"我写的观察忠不忠于我引的来源"。

对照你们的现状：

| | 现状 |
|---|---|
| 出处有效性 | **比论文强**——引文按位置从原文切出来，在工具边界强制，不是事后审计 |
| 断言一致性 | 有对应镜头：`quote_states_the_value`、`hedge_read_as_fact`（AI 评卷那条线） |
| **证据集合作为整体的结构** | **空白** |

现在查的是**每一条证据**合不合格，从没查过**这一整套证据作为集合**合不合格。本任务补
三项可确定性计算的：跨度重复、孤儿反证、单一见证。**每一项都要能给出一个可以和论文
0.6% 重复率对话的数字**，所以证据里带的是比率而不只是布尔。

**不做的**：跨知识库规范标识符去重（你们只有一个病历库，没有这个问题）；实体归属正确性
（要等 Task 13 的实体锚点，那时再加一个 detector）。

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_evidence_set_audit.py`：

```python
"""证据集合作为一个整体，也可以有毛病。

现有检查全是逐条的：这条引文回原文对得上吗、这条span非空吗。DeepEvidence 审计证据图时
报的是集合层面的数（重复率 0.6%、关系正确率 ≥99%），而这里连"同一个字段被同一段文字
支持了两次"都没人数过。三项，全部确定性，全部给比率——一个只说"有重复"的检查没法和
任何基准对话。
"""
from __future__ import annotations

from acr.evals import RunRecord, audit_evidence_set


def _run(evidence: list[dict]) -> RunRecord:
    return RunRecord(manifest={"patient_id": "SYN01", "evidence": evidence}, trace=[])


def _ev(note="N1", start=0, end=10, supports="primary_site", stance="supports") -> dict:
    return {"note_id": note, "start": start, "end": end, "supports": supports,
            "stance": stance, "quote": "x" * (end - start)}


def test_overlapping_spans_for_one_field_are_reported():
    """完全相同的 span 台账自己会去重；重叠的不会，而它是同一句话记了两遍。"""
    f = audit_evidence_set(_run([_ev(start=0, end=40), _ev(start=10, end=50)]))
    hit = [x for x in f if x.detector == "evidence_span_overlap"]
    assert len(hit) == 1
    assert hit[0].evidence["n_overlapping_pairs"] == 1
    assert hit[0].evidence["overlap_rate"] == 0.5      # 2 条里 1 条是多余的


def test_non_overlapping_spans_are_clean():
    f = audit_evidence_set(_run([_ev(start=0, end=10), _ev(start=20, end=30)]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_overlap_is_per_field_not_across_fields():
    """同一段文字同时支持部位和组织学是正常的，不是重复。"""
    f = audit_evidence_set(_run([_ev(supports="primary_site"),
                                 _ev(supports="histology")]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_a_contradiction_with_nothing_to_contradict_is_reported():
    f = audit_evidence_set(_run([_ev(supports="histology", stance="contradicts")]))
    hit = [x for x in f if x.detector == "orphan_contradiction"]
    assert len(hit) == 1
    assert hit[0].evidence["fields"] == ["histology"]


def test_a_contradiction_beside_a_support_is_a_conflict_not_an_orphan():
    """两边都有＝记录内部有矛盾，这是要如实报告的状态，不是缺陷。"""
    f = audit_evidence_set(_run([_ev(supports="histology"),
                                 _ev(supports="histology", start=99, end=120,
                                     stance="contradicts")]))
    assert not [x for x in f if x.detector == "orphan_contradiction"]


def test_a_field_resting_on_one_document_is_reported():
    f = audit_evidence_set(_run([_ev(note="N1", supports="primary_site")]))
    hit = [x for x in f if x.detector == "single_witness_field"]
    assert hit and hit[0].evidence["fields"] == ["primary_site"]


def test_two_documents_for_one_field_is_not_single_witness():
    f = audit_evidence_set(_run([_ev(note="N1"), _ev(note="N2", start=5, end=9)]))
    assert not [x for x in f if x.detector == "single_witness_field"]


def test_no_evidence_is_silent_here():
    """空台账是交卷检查的案子。这个审计只描述已经存在的证据集合。"""
    assert audit_evidence_set(_run([])) == []


def test_audit_is_wired_into_run_detectors():
    from acr.evals import DetectorConfig, run_detectors
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    run = _run([_ev(start=0, end=40), _ev(start=10, end=50)])
    assert any(f.detector == "evidence_span_overlap" for f in run_detectors(run, config=cfg))
```

**开工前先确认存档单里证据放在哪个键下**：
`.venv/bin/python -c "import json;print(list(json.load(open('<某个manifest>')).keys()))"`。
上面按 `manifest["evidence"]` 写；实际是嵌套的（比如 `answer.evidence`）就改
`_evidence_of()` 一处，测试的 `_run()` 也跟着改。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_evidence_set_audit.py -q`
Expected: FAIL — `cannot import name 'audit_evidence_set'`

- [ ] **Step 3: 实现**

`src/acr/evals.py`，`detect_uncaused_reads` 之后：

```python
def _evidence_of(run: RunRecord) -> list[dict]:
    """The recorded evidence set. One accessor, so three checks cannot disagree about it."""
    ev = run.manifest.get("evidence")
    if isinstance(ev, list):
        return [e for e in ev if isinstance(e, dict)]
    answer = run.manifest.get("answer")
    if isinstance(answer, dict) and isinstance(answer.get("evidence"), list):
        return [e for e in answer["evidence"] if isinstance(e, dict)]
    return []


def audit_evidence_set(run: RunRecord) -> list[Finding]:
    """Structural defects in the evidence set AS A SET, not in any one item.

    Every existing check is per-item: does this quote re-read at its offsets, is this span
    non-empty. DeepEvidence's evidence-graph audit reports set-level numbers instead — a 0.6%
    duplication rate, ≥99% relation correctness — and nothing here had ever counted the
    equivalent. Each finding carries a RATE, not a flag, because a check that can only say
    "there is duplication" cannot be compared against a baseline or tracked across arms.

    Silent on an empty ledger: that is the gate's case, and it already refuses the answer.
    """
    items = _evidence_of(run)
    if not items:
        return []
    out: list[Finding] = []

    # 1. Overlapping spans supporting the SAME field. The ledger de-duplicates identical
    #    (note, start, end, supports) tuples; it cannot see that chars 0-40 and 10-50 of one
    #    note are largely the same sentence recorded twice. Per field, because one sentence
    #    legitimately supports both a site and a histology.
    by_field: dict[str, list[dict]] = {}
    for e in items:
        by_field.setdefault(str(e.get("supports") or ""), []).append(e)
    pairs = 0
    for field_items in by_field.values():
        for i, a in enumerate(field_items):
            for b in field_items[i + 1:]:
                if a.get("note_id") != b.get("note_id"):
                    continue
                if int(a.get("start", 0)) < int(b.get("end", 0)) and \
                   int(b.get("start", 0)) < int(a.get("end", 0)):
                    pairs += 1
    if pairs:
        out.append(Finding(
            "evidence_span_overlap", WARN,
            f"{pairs} overlapping span pair(s) support the same field; the same sentence is "
            f"recorded more than once",
            {"n_evidence": len(items), "n_overlapping_pairs": pairs,
             "overlap_rate": round(pairs / len(items), 3)}))

    # 2. A contradiction with nothing to contradict. `stance=contradicts` beside a supporting
    #    span is a conflict in the record — a real state, reported honestly. Alone, it means
    #    the run recorded what argues AGAINST a field and never recorded what argues for it,
    #    and the answer is resting on something outside the ledger.
    supported = {str(e.get("supports") or "") for e in items
                 if e.get("stance") != "contradicts"}
    orphans = sorted({str(e.get("supports") or "") for e in items
                      if e.get("stance") == "contradicts"} - supported)
    if orphans:
        out.append(Finding(
            "orphan_contradiction", WARN,
            f"{len(orphans)} field(s) carry contradicting evidence and no supporting "
            f"evidence: {', '.join(orphans)}",
            {"fields": orphans, "n_evidence": len(items)}))

    # 3. A field resting on a single document. Not a defect — a chart may document a value
    #    once — but it is the shape in which "the right document, the wrong specimen" survives
    #    to submission, because there is no second witness to disagree with.
    docs_per_field: dict[str, set] = {}
    for e in items:
        if e.get("stance") == "contradicts":
            continue
        docs_per_field.setdefault(str(e.get("supports") or ""), set()).add(e.get("note_id"))
    singles = sorted(f for f, docs in docs_per_field.items() if len(docs) == 1 and f)
    if singles:
        out.append(Finding(
            "single_witness_field", WARN,
            f"{len(singles)} field(s) rest on one document: {', '.join(singles)}",
            {"fields": singles,
             "single_witness_rate": round(len(singles) / max(1, len(docs_per_field)), 3)}))
    return out
```

`run_detectors` 里加 `out += audit_evidence_set(run)`。

- [ ] **Step 4: 跑测试**

Run: `.venv/bin/pytest tests/test_evidence_set_audit.py tests/test_evals.py -q`
Expected: PASS

- [ ] **Step 5: 跑全套并提交**

Run: `.venv/bin/pytest -q`

```bash
git add src/acr/evals.py tests/test_evidence_set_audit.py
git commit -m "$(cat <<'EOF'
Audit the evidence set as a set: overlap, orphan contradictions, single witnesses

Every check here was per-item — does this quote re-read at its offsets. DeepEvidence's
evidence-graph audit reports set-level rates instead, and the equivalent had never been
counted in this tree. Each finding carries a rate rather than a flag, because a check that
can only say "there is duplication" cannot be compared against a baseline or across arms.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: 证据带实体锚点（哪条引文说的是哪个标本）

**Files:**
- Modify: `src/acr/state.py`（`Evidence` 加字段）
- Modify: `src/acr/tools/toolbox.py`（`record_evidence` 加参数）
- Modify: `src/acr/evals.py`（加一个 detector）
- Test: `tests/test_evidence_entity.py`（新建）

**Interfaces:**
- Consumes: Task 12 的 `_evidence_of()`
- Produces:
  - `acr.state.Evidence.entity: str = ""`（可选字段，默认空）
  - `record_evidence` 多一个可选参数 `entity`
  - `acr.evals.detect_entity_answer_mismatch(run) -> list[Finding]`，detector 名
    `entity_answer_mismatch`

**它补的洞**：证据台账是**一维 span 列表**，`eval-overconfidence` 卡里点名的一种失败——
"对的文档、错的片段：真实的原文，但说的是**另一个标本**"——在这个数据结构里**根本无法
表达**。只能靠人读引文自己判断。

**为什么这个洞能便宜地补上**：`submit_answer` 已经有 `lesions_considered` 和
`reported_lesion` 两个字段（记录模型选了哪个病灶作为答案的锚）。证据这边加上锚点，
两边就能**机器比对**：证据说的是标本 A，答案报的是病灶 B，这是确定性可查的不一致。

**排最后的原因**：动的是核心数据结构和存档格式。字段可选、默认空字符串，所以历史存档
照常读，`Evidence.to_dict()` 多一个键；`detect_entity_answer_mismatch` 在两边都为空时
沉默——没锚点不是缺陷，是这个字段还没被用起来。

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_evidence_entity.py`：

```python
"""一条引文说的是哪个标本，要能写下来。

"对的文档、错的片段"——真实的原文，说的却是另一个标本——是 eval-overconfidence 点名的
失败模式，而扁平的 span 列表在结构上无法表达它：没有地方写"这条说的是 A，那条说的是 B"。
submit_answer 早就有 reported_lesion 了；证据这边补上锚点，两边就能机器比对。

字段可选：没锚点不是缺陷，是没用起来。检查在两边都为空时沉默。
"""
from __future__ import annotations

from acr.evals import RunRecord, detect_entity_answer_mismatch
from acr.state import Evidence, EvidenceLedger


def test_evidence_carries_an_optional_entity():
    e = Evidence("N1", "pathology", "2024-01-01", 0, 10, "x", "primary_site",
                 entity="specimen A")
    assert e.to_dict()["entity"] == "specimen A"


def test_entity_defaults_to_empty_so_old_records_still_load():
    e = Evidence("N1", "pathology", "2024-01-01", 0, 10, "x", "primary_site")
    assert e.to_dict()["entity"] == ""


def test_same_span_different_entity_is_not_a_duplicate():
    """去重键必须带上实体，否则两个标本的同位置引文会被吞掉一条。"""
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen B"))
    assert len(led.items) == 2


def test_identical_entity_still_de_duplicates():
    led = EvidenceLedger()
    for _ in range(2):
        led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    assert len(led.items) == 1


def _run(evidence, reported_lesion="") -> RunRecord:
    return RunRecord(manifest={"patient_id": "SYN01", "evidence": evidence,
                               "answer": {"reported_lesion": reported_lesion}}, trace=[])


def test_evidence_about_another_lesion_than_the_one_reported_is_flagged():
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "left upper lobe"}]
    f = detect_entity_answer_mismatch(_run(ev, reported_lesion="right lower lobe"))
    assert len(f) == 1
    assert f[0].detector == "entity_answer_mismatch"
    assert f[0].evidence["reported_lesion"] == "right lower lobe"
    assert f[0].evidence["evidence_entities"] == ["left upper lobe"]


def test_matching_entity_is_clean():
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "right lower lobe"}]
    assert detect_entity_answer_mismatch(_run(ev, reported_lesion="right lower lobe")) == []


def test_silent_when_no_entity_was_recorded():
    """没用这个字段不是缺陷。一个对着空数据报警的检查，会教人把它关掉。"""
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports"}]
    assert detect_entity_answer_mismatch(_run(ev, reported_lesion="right lower lobe")) == []


def test_silent_when_the_answer_named_no_lesion():
    ev = [{"note_id": "N1", "start": 0, "end": 9, "supports": "histology",
           "stance": "supports", "entity": "left upper lobe"}]
    assert detect_entity_answer_mismatch(_run(ev, reported_lesion="")) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_evidence_entity.py -q`
Expected: FAIL — `Evidence.__init__() got an unexpected keyword argument 'entity'`

- [ ] **Step 3: `Evidence` 加字段，去重键带上它**

`src/acr/state.py`，`Evidence` 的 `stance` 之后加：

```python
    #: WHICH THING IN THE CHART this span is about — the specimen, the lesion, the procedure.
    #: Optional and empty by default, so every recorded run still loads. It exists because a
    #: flat span list cannot express "this quote is about specimen A and that one about
    #: specimen B", and "the right document, the wrong specimen" is a failure mode this repo
    #: has already named. `submit_answer` records `reported_lesion`; with both, the agreement
    #: between them is machine-checkable instead of something a reader has to notice.
    entity: str = ""
```

`EvidenceLedger.add` 的去重键加上实体：

```python
    def add(self, e: Evidence) -> None:
        for x in self.items:  # de-duplicate identical spans
            # `entity` is IN the key: without it, one sentence quoted about two specimens
            # collapses to one item, and the collapse is silent — exactly the confusion the
            # field was added to make visible.
            if (x.note_id, x.start, x.end, x.supports, x.entity) == \
               (e.note_id, e.start, e.end, e.supports, e.entity):
                return
        self.items.append(e)
```

`render()` 里，实体非空时加进去（跟在 `supports` 那行后面）：

```python
            ent = f"\n      entity:   {e.entity}" if e.entity else ""
```
并把 `ent` 拼进该条的输出。

- [ ] **Step 4: 工具加参数**

`src/acr/tools/toolbox.py`，`record_evidence` 的 schema properties 加：

```python
            "entity": {"type": "string",
                       "description": ("optional: which specimen, lesion or procedure this "
                                       "quote is ABOUT. Record it when the chart describes "
                                       "more than one; it is how a reader tells a quote about "
                                       "the reported lesion from a quote about another.")},
```

（`required` 不动。）`_t_record_evidence` 签名加 `entity: str = ""`，构造 `Evidence` 时
传进去：

```python
    def _t_record_evidence(self, note_id: str, start: int, end: int, supports: str,
                           stance: str = "supports", entity: str = "") -> dict:
```
```python
        self.evidence.add(Evidence(note_id, meta.doc_type, meta.date.isoformat(), start, end,
                                   quote, supports,
                                   "contradicts" if stance == "contradicts" else "supports",
                                   entity=str(entity or "")))
```

- [ ] **Step 5: 加 detector**

`src/acr/evals.py`，`audit_evidence_set` 之后：

```python
def detect_entity_answer_mismatch(run: RunRecord) -> list[Finding]:
    """Evidence anchored to one lesion, an answer reported about another.

    Silent unless BOTH sides recorded an anchor. An entity-less ledger is not a defect — the
    field is optional and most runs will not use it at first — and a detector that fires on
    absent data teaches people to switch it off, which costs the cases where it was right.
    """
    items = [e for e in _evidence_of(run) if str(e.get("entity") or "").strip()]
    answer = run.manifest.get("answer")
    reported = str((answer or {}).get("reported_lesion") or "").strip() \
        if isinstance(answer, dict) else ""
    if not items or not reported:
        return []
    entities = sorted({str(e["entity"]).strip() for e in items})
    if reported in entities:
        return []
    return [Finding(
        "entity_answer_mismatch", CRITICAL,
        f"the answer reports lesion {reported!r} but every anchored span is about "
        f"{', '.join(repr(x) for x in entities)}",
        {"reported_lesion": reported, "evidence_entities": entities,
         "n_anchored": len(items)})]
```

`run_detectors` 里加 `out += detect_entity_answer_mismatch(run)`。

**`CRITICAL` 而不是 `WARN`，但仍然只是一个检测器**——不进交卷检查、不拒答案。理由和
已删掉的那五条临床规则一样：判断答案**内容**的确定性规则，这棵树测量过并全部移除了
（254 次拒绝里 60 次refused 的正是登记处自己的答案）。这个检测器判断的是**记录自身
前后一致吗**——属于"关于这次运行的事实"，可以报；升级成闸门要先有测量，不在本任务范围。

**字符串比较是有意的下限**：`reported in entities` 是精确匹配，"right lower lobe" 和
"RLL" 会被判成不一致。宁可这样，也不要在这里放一个近似匹配——同义词判断是临床知识，
放进 Python 就是这棵树已经犯过五次的那个错误。误报的代价是有人多看一眼；漏报的代价是
一条引文说的是另一个标本，而没人知道。

- [ ] **Step 6: 跑测试**

Run: `.venv/bin/pytest tests/test_evidence_entity.py tests/test_evals.py tests/test_agent_tool_surface.py tests/test_single_ledger.py -q`
Expected: PASS

- [ ] **Step 7: 跑全套并提交**

Run: `.venv/bin/pytest -q`

```bash
git add src/acr/state.py src/acr/tools/toolbox.py src/acr/evals.py tests/test_evidence_entity.py
git commit -m "$(cat <<'EOF'
Evidence may name which lesion it is about, and disagreement with the answer is detectable

"The right document, the wrong specimen" is a failure this repo already named in prose and
could not express in data: a flat span list has nowhere to say that one quote is about
specimen A and another about specimen B. submit_answer already recorded reported_lesion; with
an anchor on the evidence too, the agreement between them stops being something a reader has
to notice. Optional and empty by default, so every recorded run still loads, and the detector
stays silent unless both sides recorded an anchor.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 怎么用（十三个任务做完之后）

### 跑病历——单个

```bash
set -a && . ./.env && set +a          # OpenRouter gpt-5.6-luna

acr run SYN0001 --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml
```

### 跑病历——批量

```bash
acr batch --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
          --patients SYN0001,SYN0002,SYN0003

acr batch --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml   # 整个 corpus
```

### 跑病历——换一张"病历怎么翻"的卡做对照

```bash
acr batch --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
          --skills search=search-native     --seed 1234 --out runs/arm-native

acr batch --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
          --skills search=search-preplanned --seed 1234 --out runs/arm-preplanned
```

两次唯一的差别就是那张卡；`--seed` 一样，抽样一致。存档单里 `skills[].slot == "search"` 那一项的 `content_hash` 不同，其余相同。

### 评测——单个，程序算的（不调模型）

```bash
acr signal run --kind rule \
               --run runs/arm-native/SYN0001.manifest.json \
               --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml
```

### 评测——单个，AI 评卷（trajectory 观感分，蒙着答案）

评的是**过程质量**，不是对错——对错程序已经算了。五个可评维度：`trajectory_quality`
（翻病历的路子）、`evidence_support.judged`（证据到底撑不撑）、`step_efficiency.judged`
（每一步值不值）、`l5_explanation_quality`（解释质量）、`bad_case_triage`（坏案例排序，
唯一允许看答案的一个）。每个维度三个镜头、三次模型调用，所以要报单价：

```bash
acr signal run --kind judge --dimension trajectory_quality \
               --run runs/arm-native/SYN0001.manifest.json \
               --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
               --usd-per-call 0.05 --max-usd 0.5 --model openrouter/openai/gpt-5.6-luna
```

给不给 `--gold` 都一样——蒙着答案的维度**包上就没有放答案的口袋**。观感分只用来
筛查和排序，不拦答案、不和程序分平均。

### 评测——单个，AI 看的（归因 agent）

**你不需要预先知道错因是哪一类。** 顺序是：先用 `--kind rule` 判出**哪些** case 错了，
再把错的 case 交给 `--kind agent`——它跑的是现有的归因 agent（`attribution.py` 的八段
流程：重建工作记录 → 锁定要解释的那个错误 → 提出候选原因 → 带着候选原因去翻病历 →
反事实检验 → 唱反调复核 → 引用校验 → 交结构化报告）。四张复盘卡默认**全部**给它，
哪张适用是它读完记录自己判断的。

```bash
acr signal run --kind agent \
               --run runs/arm-native/SYN0001.manifest.json \
               --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
               --gold gold/store400.csv \
               --case-id CASE-001
```

`--eval-skills` 是可选的收窄：只在人已经有怀疑方向、想省 prompt 的时候用，不是必答题：

```bash
acr signal run --kind agent --run runs/arm-native/SYN0001.manifest.json \
               --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
               --gold gold/store400.csv --case-id CASE-001 \
               --eval-skills eval-missed-evidence,eval-overconfidence
```

### 评测——批量

```bash
acr signal batch --kind rule  --runs runs/arm-native \
                 --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
                 --out signals/native-rule.json

acr signal batch --kind judge --dimension step_efficiency.judged --runs runs/arm-native \
                 --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
                 --usd-per-call 0.05 --max-usd 5 --model openrouter/openai/gpt-5.6-luna \
                 --out signals/native-judge.json

acr signal batch --kind agent --runs runs/arm-native \
                 --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \
                 --gold gold/store400.csv --case-map runs/case-map.json \
                 --out signals/native-agent.json
```

### 加一种评测角度——两条扩展路，都不改 Python

**加诊断角度**（归因 agent 的新思路）——一张卡：

```bash
mkdir -p assets/skills/eval-timeline-drift
# 写 SKILL.md，frontmatter 里 slot: eval，judges: [...]
.venv/bin/pytest tests/test_eval_skill_fence.py -q      # 围栏会检查它不含判分措辞
acr signal run --kind agent ... --eval-skills eval-timeline-drift
```

**加评卷角度**（trajectory 观感的新问题）——一个 YAML：

```bash
# 写 assets/evaluators/tool-selection.yaml：声明评哪个判断维度、问什么问题、需要什么材料
acr judge evaluators                       # 列出并围栏核查——声称判"正确性"的会被拒绝加载
acr judge run --evaluator tool-selection --context ... --subject-id CASE-001 ...
```

诊断角度是 Markdown 卡（给会用工具的归因 agent 读的方法），评卷角度是 YAML（机器要
核查它评的维度不被程序管辖、材料按声明注入）。格式不同，思想相同：加文件，不改代码。

---

## 不在这一期的

评测发现的问题自动变成新卡、自动改 spec、自动更新经验库——全部推迟，等两个模块各自跑起来、
看到真实产出再设计。
