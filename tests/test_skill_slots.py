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
    parse_skill_stack,
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


# --- Task 3: `--skills` swaps one card without minting a profile -----------------------------

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
CORPUS = ROOT / "corpus" / "patients"


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


def test_parse_replaces_general_with_a_pipe_separated_list():
    """`|` is the only way to put more than one card in `general` in a single clause.

    Comma already means "next clause", so a replacement list cannot use it. The separator is
    undiscoverable unless the help text says so, which is why the help text is tested too.
    """
    base = SkillStack(general=("coverage-judgement",))
    got = parse_skill_stack("general=chart-triage|thread-chasing", base)
    assert got.general == ("chart-triage", "thread-chasing")


def test_parse_rejects_an_unknown_slot():
    with pytest.raises(SkillError, match="unknown slot 'polish'"):
        parse_skill_stack("polish=x", SkillStack())


def test_parse_rejects_a_missing_equals():
    with pytest.raises(SkillError, match="expected slot=value"):
        parse_skill_stack("search-native", SkillStack())


def test_parse_validates_placement():
    with pytest.raises(SkillError, match="declares slot 'general'"):
        parse_skill_stack("search=chart-triage", SkillStack())


def test_parse_empty_value_clears_the_slot():
    base = SkillStack(search="keyword-strategy", general=("coverage-judgement",))
    assert parse_skill_stack("search=", base).search is None


def test_parse_empty_string_is_the_base():
    base = SkillStack(general=("coverage-judgement",))
    assert parse_skill_stack("", base) == base


def _scripted_system_prompt(tmp_path: Path, **run_kwargs) -> str:
    """The system message a real run was actually given, with a scripted provider.

    Asserting on the prompt the runtime built, not on a second assembly of it. `skill_stack`
    is read at exactly one place inside `run_patient`, and the only way to see whether that
    place honoured the argument is to read the bytes that reached the model.
    """
    pytest.importorskip("deepagents")
    from hooks_harness import ToolScript

    from acr.agent import run_patient
    from acr.corpus import Corpus
    from acr.spec import load_spec

    model = ToolScript(script=[], submit={"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                          "reasoning": "the script submits at once"})
    model.seen = []
    run_patient(spec=load_spec(SPEC), corpus=Corpus(CORPUS), patient_id="SYN0001",
                out_dir=tmp_path, model=model, max_model_calls=1, seed=7,
                run_id="skill-stack", **run_kwargs)
    systems = [m.content for turn in model.seen for m in turn
               if getattr(m, "type", None) == "system"]
    assert systems, "the scripted provider never saw a system message"
    return "\n".join(str(c) for c in systems)


def _was_rendered(prompt: str, name: str) -> bool:
    """Whether this card was assembled into the system prompt the model was given.

    Checked by its `skills_block` separator AND its own opening line, rather than by its whole
    body, because the runtime hands the provider a system message in which part of the text has
    been through `repr` — newlines and apostrophes arrive escaped, so a multi-line body never
    matches verbatim no matter what was rendered. Both probes here are single escape-free lines
    and both are READ FROM THE SKILL, so a reworded card does not need this test edited.
    `skills_block`'s exact bytes are pinned separately by the default-profile test above.
    """
    return (f"--- skill: {name} ---" in prompt
            and load_skill_body(name).splitlines()[0] in prompt)


def test_run_patient_without_a_stack_renders_the_profiles_own(tmp_path: Path):
    """`skill_stack=None` is the path every recorded run has taken; it must not move."""
    prompt = _scripted_system_prompt(tmp_path)
    assert _was_rendered(prompt, "coverage-judgement")
    assert not _was_rendered(prompt, "keyword-strategy")


def test_run_patient_renders_the_stack_it_was_given(tmp_path: Path):
    """An explicit stack replaces the profile's, and it is the override that reaches the model."""
    prompt = _scripted_system_prompt(
        tmp_path, skill_stack=SkillStack(search="keyword-strategy"))
    assert _was_rendered(prompt, "keyword-strategy")
    assert not _was_rendered(prompt, "coverage-judgement")


def _invoke(monkeypatch, tmp_path: Path, *args):
    """Run a CLI command with the model client wired to explode if it is ever built."""
    from typer.testing import CliRunner

    from acr import cli_common
    from acr.cli import app

    def refuse(*a, **kw):
        raise AssertionError("a model client was built before --skills was validated")

    monkeypatch.setattr(cli_common, "chat_model", refuse)
    return CliRunner().invoke(app, [*args, "--spec", str(SPEC), "--corpus", str(CORPUS),
                                    "--out", str(tmp_path)])


def test_a_bad_skills_string_costs_nothing_on_run(monkeypatch, tmp_path: Path):
    """A typo in `--skills` must be refused before the first token is paid for.

    Validation after the client is built is validation that has already spent the run's
    startup, and validation after the first call is a bill for a misspelling.
    """
    r = _invoke(monkeypatch, tmp_path, "run", "SYN0001", "--skills", "polish=x")
    assert r.exit_code != 0
    assert isinstance(r.exception, SkillError), r.output
    assert "unknown slot 'polish'" in str(r.exception)


def test_a_bad_skills_string_costs_nothing_on_batch(monkeypatch, tmp_path: Path):
    """`batch` swallows a per-patient exception, so an unparsed override would be retried.

    The parse therefore has to happen OUTSIDE the loop: inside it, a misspelt slot would be
    printed once per patient as if the chart had failed, and the summary would record ten
    failures whose cause was one typo.
    """
    r = _invoke(monkeypatch, tmp_path, "batch", "--patients", "SYN0001",
                "--skills", "search=chart-triage")
    assert r.exit_code != 0
    assert isinstance(r.exception, SkillError), r.output
    assert "declares slot 'general'" in str(r.exception)


@pytest.mark.parametrize("command", ["run", "batch"])
def test_the_skills_help_documents_every_form_of_the_syntax(command: str):
    """A syntax nobody can discover is a syntax nobody can use — including `|`.

    `parse_skill_stack` accepts four forms and three of them are invisible from the flag name.
    The pipe is the worst of them: comma is already the clause separator, so the obvious guess
    for "two general cards" silently parses as two clauses instead of one list.
    """
    from typer.testing import CliRunner

    from acr.cli import app

    out = CliRunner().invoke(app, [command, "--help"]).output
    flat = " ".join(out.split())
    assert "--skills" in flat
    for form in ("search=", "general=+", "|"):
        assert form in flat, f"{command} --help does not document {form!r}"
