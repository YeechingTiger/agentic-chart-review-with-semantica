"""`acr signal` 是问信号的唯一入口，程序算的和 AI 看的都从这里进。

它必须是薄的：转发给已经测过的 `evals` 和 `attribution`，自己不含判分逻辑。特别是
`--kind rule` 这条路必须一个模型都不碰——否则"无模型评测面"就只是文档里的一句话。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

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


def test_eval_skill_names_default_to_all_four():
    from acr.cli_signal import DEFAULT_EVAL_SKILLS, _eval_skill_names
    assert _eval_skill_names("") == DEFAULT_EVAL_SKILLS
    assert _eval_skill_names("  ") == DEFAULT_EVAL_SKILLS
    assert _eval_skill_names("eval-overconfidence, eval-missed-evidence") == (
        "eval-overconfidence", "eval-missed-evidence")


def test_the_default_eval_skills_all_exist_and_are_eval_slot():
    """A default that names a card nobody wrote fails at spend time, not at read time."""
    from acr.cli_signal import DEFAULT_EVAL_SKILLS
    from acr.skills import eval_skills_block
    block = eval_skills_block(list(DEFAULT_EVAL_SKILLS))
    for name in DEFAULT_EVAL_SKILLS:
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
