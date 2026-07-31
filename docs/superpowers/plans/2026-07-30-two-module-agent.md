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
- **`skills/*/SKILL.md` 单文件上限 12000 字节**（`skills.MAX_SKILL_BYTES`），超了报错不截断。
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
| `skills/search-native/SKILL.md` | "病历怎么翻"卡：模型自己判断查够了没有 |
| `skills/search-preplanned/SKILL.md` | "病历怎么翻"卡：先按经验统计排序再查 |
| `skills/eval-contrast-traces/SKILL.md` | 复盘卡：对照答对与答错的工作记录 |
| `skills/eval-cluster-failures/SKILL.md` | 复盘卡：把多个失败归类 |
| `skills/eval-missed-evidence/SKILL.md` | 复盘卡：答案在某份报告里却没翻到 |
| `skills/eval-overconfidence/SKILL.md` | 复盘卡：斩钉截铁但答错 |
| `src/acr/cli_signal.py` | `acr signal run` / `acr signal batch` 薄壳，两种做法一个出口 |
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
| `skills/*/SKILL.md`（现有 10 张） | frontmatter 加 `slot:` |

---

## Task 1: 每张卡声明自己属于哪个槽

**Files:**
- Modify: `src/acr/skills.py`
- Modify: `skills/chart-triage/SKILL.md`, `skills/thread-chasing/SKILL.md`, `skills/coverage-judgement/SKILL.md`, `skills/keyword-strategy/SKILL.md`, `skills/store-icdo-coding/SKILL.md`, `skills/store-staging/SKILL.md`, `skills/store-to-spec/SKILL.md`, `skills/crc-guideline-registry-authoring/SKILL.md`, `skills/non-concordance-triage/SKILL.md`, `skills/guideline-to-rules/SKILL.md`
- Test: `tests/test_skill_slots.py`（新建）

**Interfaces:**
- Consumes: 无（第一个任务）
- Produces: `acr.skills.SLOTS: tuple[str, ...]`、`acr.skills.skill_slot(name: str, skills_dir=None) -> str`、`acr.skills.SkillError`（已存在，复用）

**背景**：`skills/guideline-to-rules/` 只有 `references/`，`SKILL.md` 从未写过（build agent 被 spend limit 杀了），`tests/test_skills_load.py:49-55` 已经为它挂了 skip。本任务同样跳过它，不要补写。

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
skills/chart-triage/SKILL.md                      slot: general
skills/thread-chasing/SKILL.md                    slot: general
skills/coverage-judgement/SKILL.md                slot: general
skills/keyword-strategy/SKILL.md                  slot: search
skills/store-icdo-coding/SKILL.md                 slot: task
skills/store-staging/SKILL.md                     slot: task
skills/store-to-spec/SKILL.md                     slot: task
skills/crc-guideline-registry-authoring/SKILL.md  slot: task
skills/non-concordance-triage/SKILL.md            slot: general
```

`skills/guideline-to-rules/` 跳过——没有 `SKILL.md`。

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_skill_slots.py tests/test_skills_load.py -q`
Expected: PASS（`guideline-to-rules` 显示 skip）

- [ ] **Step 6: 跑全套确认没打破别的**

Run: `.venv/bin/pytest -q`
Expected: PASS，失败数与改动前一致

- [ ] **Step 7: 提交**

```bash
git add src/acr/skills.py skills/ tests/test_skill_slots.py
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
    changes the run without changing its name or version — and `refine` treats `skills/*/SKILL.md`
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
- Create: `skills/search-native/SKILL.md`
- Create: `skills/search-preplanned/SKILL.md`

**Interfaces:**
- Consumes: Task 1 的 `slot:` frontmatter 约定
- Produces: 两个可以放进 `search` 槽的名字，供 `--skills search=...` 使用

两张卡的内容依据 `docs/SEARCH_PLANNING_PILOT.md` 的两个试验臂写，不是新发明。

- [ ] **Step 1: 写 `skills/search-native/SKILL.md`**

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

- [ ] **Step 2: 写 `skills/search-preplanned/SKILL.md`**

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
git add skills/search-native skills/search-preplanned
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
- Create: `skills/eval-contrast-traces/SKILL.md`
- Create: `skills/eval-cluster-failures/SKILL.md`
- Create: `skills/eval-missed-evidence/SKILL.md`
- Create: `skills/eval-overconfidence/SKILL.md`

**Interfaces:**
- Consumes: Task 5 的 `slot: eval` + `judges:` 约定、`EVAL_FORBIDDEN_VERBS` 禁用词
- Produces: 四个可传给 `eval_skills_block()` 的名字

四张卡都不得出现 `EVAL_FORBIDDEN_VERBS` 里的措辞。每张卡都要显式写"对错问评分工具"。

- [ ] **Step 1: 写 `skills/eval-contrast-traces/SKILL.md`**

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

- [ ] **Step 2: 写 `skills/eval-cluster-failures/SKILL.md`**

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

- [ ] **Step 3: 写 `skills/eval-missed-evidence/SKILL.md`**

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

- [ ] **Step 4: 写 `skills/eval-overconfidence/SKILL.md`**

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
git add skills/eval-contrast-traces skills/eval-cluster-failures skills/eval-missed-evidence skills/eval-overconfidence
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
        "dimensions": [d.name for d in evals.REGISTRY.values() if d.deterministic],
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

在 `README.md` §3 "How to run each component" 里，`acr extract` 那段之后，加一节（照抄本计划末尾"怎么用"的四个例子），并在 §2.6 eval 平面的描述里加一句：`acr signal` 是两种信号的统一入口，`--kind rule` 无模型、`--kind agent` 走 eval skills。

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

## 怎么用（八个任务做完之后）

### 跑病历——单个

```bash
set -a && . ./.env && set +a          # OpenRouter gpt-5.6-luna

acr run SYN0001 --spec specs/STORE.400_522_523.site_histology_behavior.yaml
```

### 跑病历——批量

```bash
acr batch --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
          --patients SYN0001,SYN0002,SYN0003

acr batch --spec specs/STORE.400_522_523.site_histology_behavior.yaml   # 整个 corpus
```

### 跑病历——换一张"病历怎么翻"的卡做对照

```bash
acr batch --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
          --skills search=search-native     --seed 1234 --out runs/arm-native

acr batch --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
          --skills search=search-preplanned --seed 1234 --out runs/arm-preplanned
```

两次唯一的差别就是那张卡；`--seed` 一样，抽样一致。存档单里 `skills[].slot == "search"` 那一项的 `content_hash` 不同，其余相同。

### 评测——单个，程序算的（不调模型）

```bash
acr signal run --kind rule \
               --run runs/arm-native/SYN0001.manifest.json \
               --spec specs/STORE.400_522_523.site_histology_behavior.yaml
```

### 评测——单个，AI 看的

```bash
acr signal run --kind agent \
               --run runs/arm-native/SYN0001.manifest.json \
               --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
               --gold gold/store400.csv \
               --case-id CASE-001
```

只用其中两张复盘卡：

```bash
acr signal run --kind agent --run runs/arm-native/SYN0001.manifest.json \
               --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
               --gold gold/store400.csv --case-id CASE-001 \
               --eval-skills eval-missed-evidence,eval-overconfidence
```

### 评测——批量

```bash
acr signal batch --kind rule  --runs runs/arm-native \
                 --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
                 --out signals/native-rule.json

acr signal batch --kind agent --runs runs/arm-native \
                 --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
                 --gold gold/store400.csv --case-map runs/case-map.json \
                 --out signals/native-agent.json
```

### 加一种复盘角度

```bash
mkdir -p skills/eval-timeline-drift
# 写 SKILL.md，frontmatter 里 slot: eval，judges: [...]
.venv/bin/pytest tests/test_eval_skill_fence.py -q      # 围栏会检查它不含判分措辞
acr signal run --kind agent ... --eval-skills eval-timeline-drift
```

不用改一行 Python。

---

## 不在这一期的

评测发现的问题自动变成新卡、自动改 spec、自动更新经验库——全部推迟，等两个模块各自跑起来、
看到真实产出再设计。
