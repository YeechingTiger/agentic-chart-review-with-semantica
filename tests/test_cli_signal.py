"""`acr signal` 是问信号的唯一入口，程序算的和 AI 看的都从这里进。

它必须是薄的：转发给已经测过的 `evals` 和 `attribution`，自己不含判分逻辑。特别是
`--kind rule` 这条路必须一个模型都不碰——否则"无模型评测面"就只是文档里的一句话。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from acr.cli_signal import KINDS, signal_app

SRC = Path(__file__).resolve().parents[1] / "src"
runner = CliRunner()


def test_all_three_kinds_are_offered():
    assert KINDS == ("rule", "judge", "agent")


def test_run_help_names_every_kind():
    res = runner.invoke(signal_app, ["run", "--help"])
    assert res.exit_code == 0
    flat = _flat(res)
    assert all(k in flat for k in KINDS)


def _flat(res) -> str:
    """One line, boxes and wrapping removed.

    Click 8.4 sends usage errors to stderr, so the plan's `res.stdout` is empty for every
    refusal here; and rich draws them in a width-80 box, so the sentence arrives folded. What
    is being asserted is that the message names the offending value, and that survives both.
    """
    return " ".join(res.output.replace("│", " ").split())


def test_unknown_kind_is_refused():
    res = runner.invoke(signal_app, ["run", "--kind", "vibes", "--run", "x.manifest.json"])
    assert res.exit_code != 0
    assert "vibes" in _flat(res)


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
    from acr.cli_signal import SIGNAL_TYPE_FOR_KIND
    from acr.kernel import SIGNAL_TYPES
    assert set(SIGNAL_TYPE_FOR_KIND.values()) <= SIGNAL_TYPES
    assert set(SIGNAL_TYPE_FOR_KIND) == set(KINDS)


# ------------------------------------------------------------- THE RULE PATH, END TO END
# Not in the plan, and the only thing here that would notice if the extraction of
# `audit_run_payload` out of `acr audit run` changed what an audit does: that command has no
# test of its own, so without this the refactor's only witness would be a reviewer's reading.
def _manifest(root: Path, name: str = "run", **over) -> Path:
    """A run manifest in the shape the audit's TrajectoryAdapter ingests."""
    doc = {"patient_id": "SYN0001", "spec_id": "SPEC.A", "spec_hash": "hash1",
           "gate_validated": True, "steps": 12, "cost_usd": 0.10,
           "usage": {"total_tokens": 50_000, "llm_calls": 12},
           "declared_tools": ["search_notes", "read_note", "submit_answer"],
           "answer": {"status": "FOUND", "value": {"primary_site": "C341"}, "evidence": []}}
    doc.update(over)
    path = root / f"{name}.manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_rule_kind_produces_a_signal_envelope_without_a_model(tmp_path, monkeypatch):
    """The deterministic path, driven through the CLI, against a real manifest.

    `pytest`'s tmp_path is outside the worktree, which is what `LocalArtifactStore` demands of
    a root that may hold patient-derived artifacts.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    manifest = _manifest(tmp_path)
    res = runner.invoke(signal_app, ["run", "--kind", "rule", "--run", str(manifest),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["signal_type"] == "EVALUATION_RESULT"
    assert payload["kind"] == "rule" and payload["deterministic"] is True
    assert payload["report"]["schema"] == "acr.audit_report/1"


def test_the_premise_of_the_rule_compliance_exclusion_still_holds():
    """`rule_compliance` is deterministic in the registry and unfirable since 2026-07-30, when
    `answer_checks.ANSWER_CHECK_KINDS` was emptied. The dispatcher drops it from the advertised
    dimensions for that reason and no other — if the kinds ever refill, the drop is wrong."""
    from acr import evals
    from acr.answer_checks import ANSWER_CHECK_KINDS
    assert not ANSWER_CHECK_KINDS
    assert evals.REGISTRY["rule_compliance"].deterministic


def test_dimensions_exclude_the_unfirable_check(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    manifest = _manifest(tmp_path, "dims")
    res = runner.invoke(signal_app, ["run", "--kind", "rule", "--run", str(manifest),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    dims = json.loads(res.stdout)["dimensions"]
    assert "rule_compliance" not in dims
    assert "correctness" in dims and "hallucination" in dims


def test_out_writes_the_signal_instead_of_printing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    manifest = _manifest(tmp_path, "outed")
    out = tmp_path / "signal.json"
    res = runner.invoke(signal_app, ["run", "--kind", "rule", "--run", str(manifest),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "rule"


# ==================================== TWO POSTURES TOWARD THE KEY, NEVER IN ONE PROMPT
# A failed run has two readings and they license opposite mistakes. Believe the key and the
# cause must be in the run — a term never searched, a type filter that masked the document, a
# passage read and misjudged. Doubt the key and the question is whether it was ever derivable
# from THIS chart. Offer both at once and the agent picks whichever fits what it happened to
# find: every hard failure can exit through "the key may be wrong", and every unreachable key
# can be booked as an agent error. The modes exist so that the posture is an input.


def test_the_posture_vocabulary_is_attributions_own_and_not_a_third_spelling():
    """立场就是 truth mode，而 truth mode 已经存在于两处。

    第一版在这里发明了 `run-fault` / `key-suspect`，那是同一个概念的**第三种**拼法，而且是
    唯一一种绕过资产层的：`attribution.ATTRIBUTION_MODES` 是 (GOLD, REGISTRY_REFERENCE,
    BLIND)，`EvaluationTask.truth_mode` 校验的是同一个集合，`module_catalog/**/*.yaml` 里
    每个模块声明 `truth_modes:`。方法论文档 §4.1 的标题就是"Truth mode 决定结论上限"。

    这条断言读 attribution 自己的常量而不是抄一份 —— 抄一份的那天起两份就可以各自漂移。
    """
    from acr.attribution import ATTRIBUTION_MODES
    from acr.cli_signal import EVAL_MODES
    assert tuple(EVAL_MODES) == ATTRIBUTION_MODES


def test_no_truth_mode_puts_both_postures_in_one_prompt():
    """原来的缺陷，用正确的词表重述一遍。

    `eval-key-challenge` 开篇是"the key is also a suspect"，`eval-missed-evidence` 开篇是
    "confirm the value is genuinely documented before you start"。两句话进同一个系统提示词
    不是"更多方法"，是一个没有立场的提示词 —— 每个难解的失败都能从"key 可能有问题"退出，
    每个不可达的 key 都能记成 agent 的错，而它在两者之间的选择不被记录在任何地方。
    """
    from acr.cli_signal import EVAL_MODES, KEY_IS_RIGHT_SKILLS, KEY_IS_SUSPECT_SKILLS
    for mode, cards in EVAL_MODES.items():
        believes = set(cards) & set(KEY_IS_RIGHT_SKILLS)
        doubts = set(cards) & set(KEY_IS_SUSPECT_SKILLS)
        assert not (believes and doubts), (
            f"truth mode {mode!r} carries both postures: believes={sorted(believes)} "
            f"doubts={sorted(doubts)}")


def test_each_truth_mode_gets_the_posture_its_boundary_licenses():
    """卡组跟着 truth mode 的结论上限走，而不是反过来。

    GOLD 的边界说 packet 里的 gold 是**人工裁定过**的，所以怀疑 key 不在选项里，因在 run。
    REGISTRY_REFERENCE 的边界说登记值是"an UNRESOLVED reference, not truth"，分歧只能是
    NEEDS_ADJUDICATION —— 那正是 eval-key-challenge 要问的。
    BLIND 根本没有 truth，所以两种立场都不适用：只给与 key 无关的卡。
    """
    from acr.cli_signal import (
        EVAL_MODES,
        KEY_AGNOSTIC_SKILLS,
        KEY_IS_RIGHT_SKILLS,
        KEY_IS_SUSPECT_SKILLS,
    )
    assert set(EVAL_MODES["GOLD"]) == set(KEY_AGNOSTIC_SKILLS) | set(KEY_IS_RIGHT_SKILLS)
    assert set(EVAL_MODES["REGISTRY_REFERENCE"]) == (
        set(KEY_AGNOSTIC_SKILLS) | set(KEY_IS_SUSPECT_SKILLS))
    assert set(EVAL_MODES["BLIND"]) == set(KEY_AGNOSTIC_SKILLS)


def test_every_eval_card_in_the_tree_belongs_to_exactly_one_posture():
    """分不到立场的卡就是没有任何 truth mode 会加载的卡，也就是没人会收到的卡。

    这正是 `acr.skills` 存在要防的那个失败，高一层：运行时报告提供了方法而模型什么也没拿到。
    """
    from pathlib import Path

    from acr.cli_signal import (
        KEY_AGNOSTIC_SKILLS,
        KEY_IS_RIGHT_SKILLS,
        KEY_IS_SUSPECT_SKILLS,
    )
    from acr.skills import skill_slot

    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    in_tree = {p.name for p in skills_dir.iterdir()
               if (p / "SKILL.md").is_file() and skill_slot(p.name) == "eval"}
    postures = [set(KEY_AGNOSTIC_SKILLS), set(KEY_IS_RIGHT_SKILLS), set(KEY_IS_SUSPECT_SKILLS)]
    assigned = set().union(*postures)
    assert assigned == in_tree, f"unassigned: {sorted(in_tree - assigned)}"
    for i, a in enumerate(postures):
        for b in postures[i + 1:]:
            assert not a & b, f"card in two postures: {sorted(a & b)}"


def test_blind_is_the_default_because_a_key_must_be_asked_for():
    """默认 BLIND，和 `acr attribute case` 显式的默认值一致。

    这修掉了一处真实隐患：`cli_attribute` 的 `resolved_mode = mode or (GOLD if gold else
    BLIND)` 让**光是传了 --gold** 就把归因升到 GOLD，而 GOLD 的边界宣称那份 key 人工裁定过。
    按 §4.1 那是 HUMAN 权限才能赋予的。现在必须有人把 --truth-mode GOLD 打出来。
    """
    from acr.cli_signal import DEFAULT_TRUTH_MODE, _eval_skill_names
    assert DEFAULT_TRUTH_MODE == "BLIND"
    assert "eval-key-challenge" not in _eval_skill_names("")
    assert "eval-missed-evidence" not in _eval_skill_names("")


def test_an_explicit_eval_skills_list_still_overrides_the_truth_mode():
    """逃生口保留：truth mode 是一对命名默认，不是白名单。"""
    from acr.cli_signal import _eval_skill_names
    assert _eval_skill_names("eval-overconfidence, eval-missed-evidence") == (
        "eval-overconfidence", "eval-missed-evidence")


def test_an_unknown_truth_mode_is_refused_and_names_the_real_ones(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    m = _manifest(tmp_path, "moded")
    res = runner.invoke(signal_app, ["run", "--kind", "agent", "--run", str(m),
                                     "--spec", "s.yaml", "--case-id", "C1",
                                     "--truth-mode", "TRUST_NOBODY",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code != 0
    flat = _flat(res)
    assert "TRUST_NOBODY" in flat and "REGISTRY_REFERENCE" in flat


def test_an_unknown_truth_mode_is_refused_on_a_kind_that_would_have_ignored_it(
        tmp_path, monkeypatch):
    """`--kind rule` 从不读 truth mode，而这正是它仍然必须被检查的理由。

    `_check_kind` 立的规矩是：在选项**解析时**拒绝，不在命令体里。只在消费它的分支上校验，
    意味着确定性那一趟上打错的 `--truth-mode GOLDD` 被静默接受，而操作员要等到排在后面的
    agent 运行才知道。
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    m = _manifest(tmp_path, "ruled")
    res = runner.invoke(signal_app, ["run", "--kind", "rule", "--run", str(m),
                                     "--spec", "s.yaml", "--truth-mode", "GOLDD",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code != 0
    assert "GOLDD" in _flat(res)


@pytest.mark.parametrize("cmd", ["run", "batch"])
def test_both_commands_name_the_truth_modes_in_their_help(cmd: str):
    """断言在 mode 的**名字**上，不在字符串 `--mode` 上。

    `--model` 含有子串 `--mode`，所以这条测试最显然的写法会在 flag 还不存在时就通过 —— 第一版
    就是这样，而 CLI 当时回的是 `No such option: --mode`。
    """
    from acr.cli_signal import EVAL_MODES
    res = runner.invoke(signal_app, [cmd, "--help"])
    assert res.exit_code == 0
    flat = _flat(res)
    for mode in EVAL_MODES:
        assert mode in flat, f"{cmd} --help never names the {mode!r} truth mode"


def test_the_truth_mode_decides_the_cards_and_reaches_attribution(monkeypatch, tmp_path):
    """一个被解析而没接线的 flag 等于没有 flag。

    两件事一起断言：渲染出的卡片块（模型唯一看得见的东西），以及 `mode` 真的到了
    `attribute_case_payload` —— 否则归因会自己按 `--gold` 推导，卡片和边界指令又会矛盾。
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    import acr.cli_attribute as CA
    seen: dict = {}

    def fake(**kw):
        seen.update(kw)
        return {"schema": "acr.signal/1", "kind": "agent", "report": {}}

    monkeypatch.setattr(CA, "attribute_case_payload", fake)
    m = _manifest(tmp_path, "wired")
    res = runner.invoke(signal_app, ["run", "--kind", "agent", "--run", str(m),
                                     "--spec", "s.yaml", "--case-id", "C1",
                                     "--truth-mode", "REGISTRY_REFERENCE",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert seen["mode"] == "REGISTRY_REFERENCE"          # 边界指令跟着走
    block = seen["eval_skills_prompt"]
    assert "eval skill: eval-key-challenge" in block     # 卡片也跟着走
    assert "eval skill: eval-missed-evidence" not in block


@pytest.mark.parametrize("mode", ["GOLD", "REGISTRY_REFERENCE", "BLIND"])
def test_every_mode_renders_and_names_only_cards_that_exist(mode: str):
    """A mode that names a card nobody wrote fails at spend time, not at read time.

    Per mode, not over one merged list: a card that only the unused mode names would otherwise
    be validated by whichever mode happened to include it.
    """
    from acr.cli_signal import EVAL_MODES
    from acr.skills import eval_skills_block
    cards = EVAL_MODES[mode]
    block = eval_skills_block(list(cards))
    for name in cards:
        assert f"eval skill: {name}" in block


def test_the_agent_path_refuses_a_non_eval_skill_before_spending(tmp_path, monkeypatch):
    """Validation is ahead of the provider import on purpose: a typo in --eval-skills must
    cost nothing."""
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    manifest = _manifest(tmp_path, "agentic")
    res = runner.invoke(signal_app, ["run", "--kind", "agent", "--run", str(manifest),
                                     "--spec", "specs/whatever.yaml", "--case-id", "CASE1",
                                     "--eval-skills", "coverage-judgement",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code != 0
    flat = _flat(res)
    assert "coverage-judgement" in flat and "not an eval skill" in flat


# ============================================================ BATCH — A COHORT OF RUNS
# The load-bearing property is at the bottom of this block: one bad run does not abort the
# batch. Everything above it is the plumbing that has to be right for that property to be
# reachable at all.
def test_the_group_still_lists_both_commands():
    """`acr signal run` and `acr signal batch`, both spelled as subcommands.

    Typer collapses a single-command app into a bare command. Two commands means the collapse
    cannot happen today, but the group shape is the thing runbooks depend on, and pinning it
    here means deleting `batch` later fails a test instead of silently renaming `signal run`.
    """
    res = runner.invoke(signal_app, ["--help"])
    assert res.exit_code == 0
    flat = _flat(res)
    assert "run" in flat and "batch" in flat


def test_batch_help_names_the_runs_option():
    res = runner.invoke(signal_app, ["batch", "--help"])
    assert res.exit_code == 0
    assert "--runs" in _flat(res)


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
    with pytest.raises(typer.BadParameter, match=r"no \*\.manifest\.json"):
        _manifest_paths(str(tmp_path))


def test_batch_refuses_a_path_that_is_neither(tmp_path: Path):
    from acr.cli_signal import _manifest_paths
    with pytest.raises(typer.BadParameter, match="not a file or directory"):
        _manifest_paths(str(tmp_path / "nowhere"))


def test_one_failure_does_not_abort_the_batch(tmp_path: Path, monkeypatch):
    """One bad run must not discard the rest.

    Aborting throws away the signals already produced and, on the agent and judge kinds, the
    money already spent producing them. The failure belongs in the output array beside the
    successes, where a reader counts both without re-running anything.
    """
    import acr.cli_signal as cs
    ok = tmp_path / "ok.manifest.json"
    ok.write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.manifest.json"
    bad.write_text("{}", encoding="utf-8")

    def fake(*, run, spec, local_root=None):
        if "bad" in run:
            raise RuntimeError("boom")
        return {"kind": "rule", "run": run}

    monkeypatch.setattr(cs, "_rule_signal", fake)
    out = cs._batch_signals(kind="rule", paths=[ok, bad], spec="s.yaml", gold="",
                            patient_to_case={}, eval_skills=())
    assert len(out) == 2
    assert out[0]["run"].endswith("ok.manifest.json")
    assert out[1]["error"] == "RuntimeError: boom"
    assert out[1]["kind"] == "rule" and out[1]["run"].endswith("bad.manifest.json")


def test_the_rule_batch_records_a_broken_manifest_and_keeps_the_good_one(tmp_path, monkeypatch):
    """The same property end to end, with no monkeypatch on the thing under test.

    A manifest that is not JSON is the cheapest real failure to construct, and it is the one an
    operator actually hits — a run killed mid-write. The good run's signal must still arrive.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    _manifest(tmp_path, "aa")
    (tmp_path / "zz.manifest.json").write_text("{ truncated", encoding="utf-8")
    res = runner.invoke(signal_app, ["batch", "--kind", "rule", "--runs", str(tmp_path),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    signals = json.loads(res.stdout)
    assert len(signals) == 2
    assert signals[0]["kind"] == "rule" and signals[0]["deterministic"] is True
    assert "error" not in signals[0]
    assert "JSONDecodeError" in signals[1]["error"]


def test_batch_stdout_carries_only_the_json_array(tmp_path, monkeypatch):
    """Progress and failures go to stderr. stdout is one document a pipe can parse.

    The count of runs and the name of each failure are exactly what an operator wants to watch
    scroll past, and exactly what makes `acr signal batch ... | jq` stop working if it lands on
    stdout beside the array.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    _manifest(tmp_path, "aa")
    (tmp_path / "zz.manifest.json").write_text("{ truncated", encoding="utf-8")
    res = runner.invoke(signal_app, ["batch", "--kind", "rule", "--runs", str(tmp_path),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert isinstance(json.loads(res.stdout), list)      # nothing else got in
    assert "zz.manifest.json" in " ".join(res.stderr.split())


def test_a_batch_where_every_run_failed_is_not_reported_as_success(tmp_path, monkeypatch):
    """One bad run is data; nothing but bad runs is a broken invocation.

    Exit 0 with an array of nothing but errors tells a shell script the cohort was evaluated.
    It was not.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    (tmp_path / "zz.manifest.json").write_text("{ truncated", encoding="utf-8")
    res = runner.invoke(signal_app, ["batch", "--kind", "rule", "--runs", str(tmp_path),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 2
    assert json.loads(res.stdout)[0]["error"]            # the array is still emitted


def test_batch_out_writes_the_array_instead_of_printing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    _manifest(tmp_path, "aa")
    out = tmp_path / "signals.json"
    res = runner.invoke(signal_app, ["batch", "--kind", "rule", "--runs", str(tmp_path),
                                     "--spec", "specs/whatever.yaml",
                                     "--local-root", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text(encoding="utf-8"))[0]["kind"] == "rule"


def test_the_batch_case_id_comes_from_the_same_case_map_acr_attribute_takes(tmp_path):
    """`--case-map` is `{case_id: patient_id}` here because that is what it is everywhere else.

    The plan specified `{manifest stem: case id}` for this command alone. Two shapes behind one
    flag name in one CLI is a trap, and the stem of `SYN0001.manifest.json` is
    `SYN0001.manifest`, so the wrong shape also produces case ids with a file extension in them.
    """
    from acr.cli_signal import _case_id_for
    manifest = _manifest(tmp_path, "SYN0001")
    assert _case_id_for(manifest, {"SYN0001": "CASE-001"}) == "CASE-001"
    # No map: the manifest's own patient id, which `attribution.safe_case_id` refuses
    # downstream if it looks like a real person rather than a synthetic subject.
    assert _case_id_for(manifest, {}) == "SYN0001"


# =================================================== JUDGE — THE TRAJECTORY JUDGE, FENCED
# Everything the fence does is already `judge.py`'s. What is tested here is that the
# dispatcher reaches it rather than reproducing it, and that the ergonomics it adds — a packet
# assembled from a run instead of hand-built JSON — do not smuggle the answer key in with them.
class _StubReply:
    def __init__(self, content: str):
        self.content = content


class _StubClient:
    """An `acr.llm` client that answers every judge prompt with the same usable JSON."""

    def __init__(self, content: str = '{"score": 0.7, "observation": "saw it", "concerns": []}'):
        self.content, self.prompts = content, []

    def chat(self, messages, tools=None):
        self.prompts.append(messages[0]["content"])
        return _StubReply(self.content)


def _traced(root: Path, name: str = "r", **over) -> Path:
    """A manifest with the sibling `.jsonl` trace beside it, which is how runs land on disk."""
    path = _manifest(root, name, **over)
    (root / f"{name}.jsonl").write_text(
        '{"seq": 1, "kind": "tool_call", "tool": "search_notes", "args": {"q": "adenoca"}}\n'
        '{"seq": 2, "kind": "tool_call", "tool": "read_document", "args": {"doc": "path-1"}}\n',
        encoding="utf-8")
    return path


def test_judge_kind_requires_a_dimension():
    res = runner.invoke(signal_app, ["run", "--kind", "judge",
                                     "--run", "x.manifest.json", "--spec", "s.yaml"])
    assert res.exit_code != 0
    assert "--dimension" in _flat(res)


def test_judge_kind_will_not_run_on_a_price_nobody_typed(tmp_path, monkeypatch):
    """`acr judge panel` requires --usd-per-call and --max-usd with no default, because an
    unpriced call reads as free. Arriving at the same judge through a different front door
    must not be how somebody gets a default."""
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    m = _traced(tmp_path)
    res = runner.invoke(signal_app, ["run", "--kind", "judge", "--dimension",
                                     "trajectory_quality", "--run", str(m), "--spec", "s.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code != 0
    flat = _flat(res)
    assert "--usd-per-call" in flat and "--max-usd" in flat


def test_judge_signal_builds_a_blind_packet_for_blinded_dimensions(tmp_path: Path):
    """Blinding is not an instruction, it is a packet with nowhere to put the key.

    `--gold` is supplied here and must be ignored ENTIRELY — not read and filtered, not read
    and dropped. The type that comes back has no field it could have gone into.

    The key says C509 and the run answered C341, deliberately: with the two equal, "the key
    leaked" and "the judge can see what the run itself concluded" would look identical, and the
    second is not a leak — a trajectory judge is supposed to see the run's own output.
    """
    import acr.cli_signal as cs
    from acr import judge as J
    m = _traced(tmp_path)
    g = tmp_path / "gold.json"
    g.write_text(json.dumps({"SYN0001": {"primary_site": "C509"}}), encoding="utf-8")
    packet = cs._packet_from_run(run=str(m), gold=str(g), dimension="trajectory_quality",
                                 local_root=str(tmp_path))
    assert isinstance(packet, J.BlindPacket)
    assert not hasattr(packet, "answer_key")
    assert "C509" not in J._render(packet)          # nothing of the key reaches the prompt
    assert packet.subject_id == "SYN0001"


def test_judge_signal_allows_the_key_only_for_triage(tmp_path: Path):
    import acr.cli_signal as cs
    from acr import judge as J
    m = _traced(tmp_path, "t")
    g = tmp_path / "gold.json"
    g.write_text(json.dumps({"SYN0001": {"primary_site": "C341"}}), encoding="utf-8")
    packet = cs._packet_from_run(run=str(m), gold=str(g), dimension="bad_case_triage",
                                 local_root=str(tmp_path))
    assert isinstance(packet, J.KeyedPacket)
    assert packet.answer_key["SYN0001"]["primary_site"] == "C341"


def test_only_triage_is_key_permitted_and_the_dispatcher_reads_that_from_judge():
    """The blind/keyed split is judge.py's constant, not a list retyped over here."""
    from acr import judge as J
    assert J.KEY_PERMITTED_DIMENSIONS == ("bad_case_triage",)


def test_the_manifest_does_not_crowd_the_trace_out_of_the_packet(tmp_path: Path):
    """A whole manifest pasted into the packet evicts the trajectory, silently.

    `judge._render` serialises artifacts BEFORE the trace and truncates the pair at
    PACKET_CHAR_BUDGET. Run manifests run to tens of kilobytes — `develop_plane_candidates`
    alone can — so the naive packet shows a trajectory judge no trajectory, and it still
    returns three confident scores.
    """
    import acr.cli_signal as cs
    from acr import judge as J
    m = _traced(tmp_path, "big", develop_plane_candidates={"terms": ["x" * 40] * 400})
    packet = cs._packet_from_run(run=str(m), gold="", dimension="trajectory_quality",
                                 local_root=str(tmp_path))
    assert set(packet.artifacts["manifest"]) <= set(cs.MANIFEST_KEYS_SHOWN)
    assert "develop_plane_candidates" not in packet.artifacts["manifest"]
    assert "search_notes" in J._render(packet)          # the trace survived the budget


def test_the_fence_is_judges_own_and_not_a_copy(tmp_path, monkeypatch):
    """`correctness` is `==`. Asking the judge for it must fail with judge()'s own sentence.

    The dispatcher never inspects the precedence registry itself: a second copy of the
    judgement is free to drift the first time somebody adds a row.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    m = _traced(tmp_path, "fenced")
    res = runner.invoke(signal_app, ["run", "--kind", "judge", "--dimension", "correctness",
                                     "--run", str(m), "--spec", "s.yaml",
                                     "--usd-per-call", "0.05", "--max-usd", "1",
                                     "--model", "stub/model", "--local-root", str(tmp_path)])
    assert res.exit_code == 2
    flat = _flat(res)
    assert "DeterministicEvaluatorExists" in flat and "acr.evals.score" in flat


def test_a_dimension_the_registry_never_heard_of_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    m = _traced(tmp_path, "unknown")
    res = runner.invoke(signal_app, ["run", "--kind", "judge", "--dimension", "vibes",
                                     "--run", str(m), "--spec", "s.yaml",
                                     "--usd-per-call", "0.05", "--max-usd", "1",
                                     "--model", "stub/model", "--local-root", str(tmp_path)])
    assert res.exit_code == 2
    assert "RegistryUnavailable" in _flat(res)


def test_the_panel_is_priced_before_the_first_call(tmp_path, monkeypatch):
    """A ceiling enforced after the spend is a report. Three lenses at $1 exceed $2, and the
    stub client must never have been reached."""
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    client = _StubClient()
    monkeypatch.setattr("acr.cli_common.llm_client", lambda *a, **k: client)
    m = _traced(tmp_path, "pricey")
    res = runner.invoke(signal_app, ["run", "--kind", "judge", "--dimension",
                                     "trajectory_quality", "--run", str(m), "--spec", "s.yaml",
                                     "--usd-per-call", "1", "--max-usd", "2",
                                     "--model", "stub/model", "--local-root", str(tmp_path)])
    assert res.exit_code != 0
    assert "exceeds" in _flat(res)
    assert client.prompts == []


def test_the_judge_envelope_is_stamped_judged(tmp_path, monkeypatch):
    """A judged number screens and ranks. It never gates and never averages with a
    deterministic score, and the envelope has to say so where a consumer reads it."""
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    client = _StubClient()
    monkeypatch.setattr("acr.cli_common.llm_client", lambda *a, **k: client)
    m = _traced(tmp_path, "judged")
    res = runner.invoke(signal_app, ["run", "--kind", "judge", "--dimension",
                                     "trajectory_quality", "--run", str(m), "--spec", "s.yaml",
                                     "--usd-per-call", "0.05", "--max-usd", "1",
                                     "--model", "stub/model", "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["kind"] == "judge" and payload["deterministic"] is False
    assert payload["signal_type"] == "EVALUATION_RESULT"
    assert payload["evidence_class"] == "JUDGED"
    assert payload["verdict"]["evidence_class"] == "JUDGED"
    assert payload["verdict"]["validation_status"] == "NOT_VALIDATED"
    assert payload["verdict"]["score"] == pytest.approx(0.7)   # the mean of three 0.7s
    assert len(client.prompts) == 3                       # one call per lens, not one per run
    # A Verdict has no `passed`; the envelope must not grow one on the way out either.
    from acr import judge as J
    assert not set(payload) & set(J.DECISION_FIELD_NAMES)
    assert not set(payload["verdict"]) & set(J.DECISION_FIELD_NAMES)


def test_the_judge_kind_batches_and_one_refusal_is_not_the_cohort(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    monkeypatch.setattr("acr.cli_common.llm_client", lambda *a, **k: _StubClient())
    _traced(tmp_path, "aa")
    (tmp_path / "zz.manifest.json").write_text("{ truncated", encoding="utf-8")
    res = runner.invoke(signal_app, ["batch", "--kind", "judge", "--dimension",
                                     "trajectory_quality", "--runs", str(tmp_path),
                                     "--spec", "s.yaml", "--usd-per-call", "0.05",
                                     "--max-usd", "1", "--model", "stub/model",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    signals = json.loads(res.stdout)
    assert len(signals) == 2
    assert signals[0]["evidence_class"] == "JUDGED"
    assert "JSONDecodeError" in signals[1]["error"]


def test_the_json_judge_model_is_public_and_the_old_name_still_resolves():
    """`cli_signal` needs the same JSON-mode adapter `acr judge panel` uses. A second one
    would be a second place for the parsing rules to drift."""
    from acr import cli_judge
    assert cli_judge._JsonModel is cli_judge.JsonJudgeModel


def test_the_agent_kind_gets_more_turns_than_acr_attribute_case_defaults_to():
    """回归：12 次调用装不下八段流程加四张复盘卡。

    第一次真实归因停在 11/12，`cause: UNRESOLVED`，理由是"model-call limit reached without a
    gate-valid attribution"——反事实检验和唱反调复核都没做，报告门控于是（正确地）拒绝把它
    标为已解决。一个产不出交付物的默认值不是预算，是墙。

    提到 24 之后同一个案例跑完了八段，`gate_rejections` 为空，结论仍是 UNRESOLVED——但换成了
    "独立的唱反调模型不接受这个因果链"，也就是对抗性检查在起作用，而不是预算撞墙。
    """
    from acr.cli_signal import DEFAULT_AGENT_CHART_READS, DEFAULT_AGENT_MODEL_CALLS
    assert DEFAULT_AGENT_MODEL_CALLS > 12
    assert DEFAULT_AGENT_CHART_READS >= 12


def test_both_agent_budgets_reach_the_attribution_payload(monkeypatch):
    """参数必须真的穿到底 —— 加了 flag 却没接线，和没加一样。"""
    import acr.cli_attribute as CA
    import acr.cli_signal as cs
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"schema": "acr.signal/1", "kind": "agent", "report": {}}

    monkeypatch.setattr(CA, "attribute_case_payload", fake)
    cs._agent_signal(run="r.manifest.json", spec="s.yaml", gold="", case_id="C1",
                     eval_skills=(), max_model_calls=31, max_chart_reads=7)
    assert seen["max_model_calls"] == 31
    assert seen["max_chart_reads"] == 7


@pytest.mark.parametrize("cmd", ["run", "batch"])
def test_both_budget_flags_are_offered_on_both_commands(cmd: str):
    res = runner.invoke(signal_app, [cmd, "--help"])
    assert res.exit_code == 0
    flat = _flat(res)
    assert "--max-model-calls" in flat and "--max-chart-reads" in flat


def test_the_agent_batch_reaches_the_diagnosis_at_all(tmp_path, monkeypatch):
    """回归：`_batch_signals` 的函数体用了两个签名里没有的名字。

    `acr signal batch --kind agent` 于是对每一个 run 抛 `NameError`，而这条路径上的
    `except Exception` 正是为"一个坏 run 不算整批"写的——它把 NameError 一视同仁地记成
    per-run error，整批以 exit 2 结束，看起来像"这批 run 都有问题"，而不是"这个命令从来没跑过"。
    上面那个测试只检查 flag 出现在 help 里，`--kind agent` 的批量路径没有任何测试穿过。

    这是两个模式跑一个 cohort 的必经之路，所以在这里补上。
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    import acr.cli_attribute as CA
    seen: list[dict] = []

    def fake(**kw):
        seen.append(kw)
        return {"schema": "acr.signal/1", "kind": "agent", "report": {}}

    monkeypatch.setattr(CA, "attribute_case_payload", fake)
    _manifest(tmp_path, "SYN0001")
    res = runner.invoke(signal_app, ["batch", "--kind", "agent", "--runs", str(tmp_path),
                                     "--spec", "s.yaml", "--truth-mode", "REGISTRY_REFERENCE",
                                     "--max-model-calls", "31", "--max-chart-reads", "7",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    signals = json.loads(res.stdout)
    assert len(signals) == 1 and "error" not in signals[0], signals
    assert len(seen) == 1
    assert seen[0]["max_model_calls"] == 31      # the flags reach the payload, not just the help
    assert seen[0]["max_chart_reads"] == 7
    assert "eval-key-challenge" in seen[0]["eval_skills_prompt"]     # and so does the mode
