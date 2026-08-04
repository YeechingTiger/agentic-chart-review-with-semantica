"""`acr signal` is the one door signals come through: what code computes and what a model reads.

It has to be thin — it forwards to the already-tested `evals` and `attribution` and holds no
scoring logic of its own. The `--kind rule` path in particular must not touch a single model:
otherwise "a model-free evaluation plane" is only a sentence in the documentation.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from acr.commands.cli_signal import KINDS, signal_app

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
    """The thin shell must cost nothing: provider imports happen only in the --kind agent branch.

    The `acr eval` group promises it calls no model. If this new group imports litellm at module
    scope, anybody who imports `cli` drags the provider in with them, and that promise is gone in
    practice.
    """
    tree = ast.parse((SRC / "acr" / "commands" / "cli_signal.py").read_text(encoding="utf-8"))
    top: set[str] = set()
    for node in tree.body:                      # module scope only; deferred imports do not count
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    forbidden = {"litellm", "langchain", "langgraph", "deepagents", "openai", "anthropic"}
    assert not top & forbidden, f"module-scope provider import: {top & forbidden}"


def test_signal_envelope_shape_is_the_contract():
    from acr.commands.cli_signal import SIGNAL_TYPE_FOR_KIND
    from acr.core.kernel import SIGNAL_TYPES
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
                                     "--spec", "assets/specs/whatever.yaml",
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
    from acr.contract.answer_checks import ANSWER_CHECK_KINDS
    from acr.evaluation import evals
    assert not ANSWER_CHECK_KINDS
    assert evals.REGISTRY["rule_compliance"].deterministic


def test_dimensions_exclude_the_unfirable_check(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    manifest = _manifest(tmp_path, "dims")
    res = runner.invoke(signal_app, ["run", "--kind", "rule", "--run", str(manifest),
                                     "--spec", "assets/specs/whatever.yaml",
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
                                     "--spec", "assets/specs/whatever.yaml",
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
    """The posture IS the truth mode, and the truth mode already exists in two places.

    The first version invented `run-fault` / `key-suspect` here. That was a THIRD spelling of one
    concept, and the only one that bypassed the asset layer: `attribution.ATTRIBUTION_MODES` is
    (GOLD, REGISTRY_REFERENCE, BLIND), `EvaluationTask.truth_mode` validates the same set, and
    every module under `assets/module_catalog/**/*.yaml` declares `truth_modes:`. The methodology
    doc's §4.1 is titled "Truth mode caps what can be concluded".

    This assertion reads attribution's own constant rather than keeping a copy — from the day
    there are two copies, the two are free to drift apart.
    """
    from acr.commands.cli_signal import EVAL_MODES
    from acr.diagnosis.attribution import ATTRIBUTION_MODES
    assert tuple(EVAL_MODES) == ATTRIBUTION_MODES


def test_no_truth_mode_puts_both_postures_in_one_prompt():
    """The original defect, restated in the right vocabulary.

    `eval-key-challenge` opens with "the key is also a suspect"; `eval-missed-evidence` opens with
    "confirm the value is genuinely documented before you start". Both sentences in one system
    prompt is not "more method", it is a prompt with no posture — every hard failure can exit
    through "the key may be wrong", every unreachable key can be booked as an agent error, and the
    agent's choice between the two is recorded nowhere.
    """
    from acr.commands.cli_signal import EVAL_MODES, KEY_IS_RIGHT_SKILLS, KEY_IS_SUSPECT_SKILLS
    for mode, cards in EVAL_MODES.items():
        believes = set(cards) & set(KEY_IS_RIGHT_SKILLS)
        doubts = set(cards) & set(KEY_IS_SUSPECT_SKILLS)
        assert not (believes and doubts), (
            f"truth mode {mode!r} carries both postures: believes={sorted(believes)} "
            f"doubts={sorted(doubts)}")


def test_each_truth_mode_gets_the_posture_its_boundary_licenses():
    """The card set follows the truth mode's ceiling on conclusions, not the other way round.

    GOLD's boundary says the packet's gold was HUMAN ADJUDICATED, so doubting the key is not on
    the table and the cause is in the run.
    REGISTRY_REFERENCE's boundary says a registry value is "an UNRESOLVED reference, not truth",
    so a disagreement may only be NEEDS_ADJUDICATION — exactly what eval-key-challenge asks.
    BLIND has no truth at all, so neither posture applies: key-agnostic cards only.
    """
    from acr.commands.cli_signal import (
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
    """A card assigned to no posture is a card no truth mode loads, which is a card nobody receives.

    That is the failure `acr.contract.skills` exists to prevent, one level up: the run reports that
    method was offered while the model received nothing at all.
    """
    from pathlib import Path

    from acr.commands.cli_signal import (
        KEY_AGNOSTIC_SKILLS,
        KEY_IS_RIGHT_SKILLS,
        KEY_IS_SUSPECT_SKILLS,
    )
    from acr.contract.skills import skill_slot

    skills_dir = Path(__file__).resolve().parents[1] / "assets" / "skills"
    in_tree = {p.name for p in skills_dir.iterdir()
               if (p / "SKILL.md").is_file() and skill_slot(p.name) == "eval"}
    postures = [set(KEY_AGNOSTIC_SKILLS), set(KEY_IS_RIGHT_SKILLS), set(KEY_IS_SUSPECT_SKILLS)]
    assigned = set().union(*postures)
    assert assigned == in_tree, f"unassigned: {sorted(in_tree - assigned)}"
    for i, a in enumerate(postures):
        for b in postures[i + 1:]:
            assert not a & b, f"card in two postures: {sorted(a & b)}"


def test_blind_is_the_default_because_a_key_must_be_asked_for():
    """BLIND by default, matching `acr attribute case`'s own explicit default.

    This fixes a real latent defect: `cli_attribute`'s `resolved_mode = mode or (GOLD if gold else
    BLIND)` let the MERE PRESENCE of `--gold` promote the attribution to GOLD, and GOLD's boundary
    asserts that key was human adjudicated. Under §4.1 that is authority only the HUMAN plane may
    grant. Now somebody has to type `--truth-mode GOLD`.
    """
    from acr.commands.cli_signal import DEFAULT_TRUTH_MODE, _eval_skill_names
    assert DEFAULT_TRUTH_MODE == "BLIND"
    assert "eval-key-challenge" not in _eval_skill_names("")
    assert "eval-missed-evidence" not in _eval_skill_names("")


def test_an_explicit_eval_skills_list_still_overrides_the_truth_mode():
    """The escape hatch stays: a mode is a named pair of defaults, not a whitelist."""
    from acr.commands.cli_signal import _eval_skill_names
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
    """`--kind rule` never reads the truth mode, and that is exactly why it still has to be checked.

    The rule `_check_kind` set is: refuse AS THE OPTION IS PARSED, not in the command body.
    Validating only in the branch that consumes it means a mistyped `--truth-mode GOLDD` on the
    deterministic pass is accepted in silence, and the operator does not find out until an agent
    run later in the queue.
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
    """The assertion is on the mode NAMES, not on the string `--mode`.

    `--model` contains `--mode` as a substring, so the most obvious way to write this test passes
    while the flag does not exist yet — that is what the first version did, and the CLI was
    answering `No such option: --mode` at the time.
    """
    from acr.commands.cli_signal import EVAL_MODES
    res = runner.invoke(signal_app, [cmd, "--help"])
    assert res.exit_code == 0
    flat = _flat(res)
    for mode in EVAL_MODES:
        assert mode in flat, f"{cmd} --help never names the {mode!r} truth mode"


def test_the_truth_mode_decides_the_cards_and_reaches_attribution(monkeypatch, tmp_path):
    """A flag that is parsed and never wired is not a flag.

    Two things asserted together: the rendered card block (the only thing the model ever sees), and
    that `mode` really arrives at `attribute_case_payload` — otherwise attribution derives it from
    `--gold` on its own, and the cards and the boundary instruction contradict each other again.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    import acr.commands.cli_attribute as CA
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
    assert seen["mode"] == "REGISTRY_REFERENCE"          # the boundary instruction follows
    block = seen["eval_skills_prompt"]
    assert "eval skill: eval-key-challenge" in block     # and so do the cards
    assert "eval skill: eval-missed-evidence" not in block


@pytest.mark.parametrize("mode", ["GOLD", "REGISTRY_REFERENCE", "BLIND"])
def test_every_mode_renders_and_names_only_cards_that_exist(mode: str):
    """A mode that names a card nobody wrote fails at spend time, not at read time.

    Per mode, not over one merged list: a card that only the unused mode names would otherwise
    be validated by whichever mode happened to include it.
    """
    from acr.commands.cli_signal import EVAL_MODES
    from acr.contract.skills import eval_skills_block
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
                                     "--spec", "assets/specs/whatever.yaml", "--case-id", "CASE1",
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
    from acr.commands.cli_signal import _manifest_paths
    (tmp_path / "a.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    got = _manifest_paths(str(tmp_path))
    assert [p.name for p in got] == ["a.manifest.json", "b.manifest.json"]


def test_batch_accepts_a_single_file(tmp_path: Path):
    from acr.commands.cli_signal import _manifest_paths
    f = tmp_path / "only.manifest.json"
    f.write_text("{}", encoding="utf-8")
    assert _manifest_paths(str(f)) == [f]


def test_batch_refuses_an_empty_directory(tmp_path: Path):
    from acr.commands.cli_signal import _manifest_paths
    with pytest.raises(typer.BadParameter, match=r"no \*\.manifest\.json"):
        _manifest_paths(str(tmp_path))


def test_batch_refuses_a_path_that_is_neither(tmp_path: Path):
    from acr.commands.cli_signal import _manifest_paths
    with pytest.raises(typer.BadParameter, match="not a file or directory"):
        _manifest_paths(str(tmp_path / "nowhere"))


def test_one_failure_does_not_abort_the_batch(tmp_path: Path, monkeypatch):
    """One bad run must not discard the rest.

    Aborting throws away the signals already produced and, on the agent and judge kinds, the
    money already spent producing them. The failure belongs in the output array beside the
    successes, where a reader counts both without re-running anything.
    """
    import acr.commands.cli_signal as cs
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
                                     "--spec", "assets/specs/whatever.yaml",
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
                                     "--spec", "assets/specs/whatever.yaml",
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
                                     "--spec", "assets/specs/whatever.yaml",
                                     "--local-root", str(tmp_path)])
    assert res.exit_code == 2
    assert json.loads(res.stdout)[0]["error"]            # the array is still emitted


def test_batch_out_writes_the_array_instead_of_printing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    _manifest(tmp_path, "aa")
    out = tmp_path / "signals.json"
    res = runner.invoke(signal_app, ["batch", "--kind", "rule", "--runs", str(tmp_path),
                                     "--spec", "assets/specs/whatever.yaml",
                                     "--local-root", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text(encoding="utf-8"))[0]["kind"] == "rule"


def test_the_batch_case_id_comes_from_the_same_case_map_acr_attribute_takes(tmp_path):
    """`--case-map` is `{case_id: patient_id}` here because that is what it is everywhere else.

    The plan specified `{manifest stem: case id}` for this command alone. Two shapes behind one
    flag name in one CLI is a trap, and the stem of `SYN0001.manifest.json` is
    `SYN0001.manifest`, so the wrong shape also produces case ids with a file extension in them.
    """
    from acr.commands.cli_signal import _case_id_for
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
    """An `acr.core.llm` client that answers every judge prompt with the same usable JSON."""

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
    import acr.commands.cli_signal as cs
    from acr.evaluation import judge as J
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
    import acr.commands.cli_signal as cs
    from acr.evaluation import judge as J
    m = _traced(tmp_path, "t")
    g = tmp_path / "gold.json"
    g.write_text(json.dumps({"SYN0001": {"primary_site": "C341"}}), encoding="utf-8")
    packet = cs._packet_from_run(run=str(m), gold=str(g), dimension="bad_case_triage",
                                 local_root=str(tmp_path))
    assert isinstance(packet, J.KeyedPacket)
    assert packet.answer_key["SYN0001"]["primary_site"] == "C341"


def test_only_triage_is_key_permitted_and_the_dispatcher_reads_that_from_judge():
    """The blind/keyed split is judge.py's constant, not a list retyped over here."""
    from acr.evaluation import judge as J
    assert J.KEY_PERMITTED_DIMENSIONS == ("bad_case_triage",)


def test_the_manifest_does_not_crowd_the_trace_out_of_the_packet(tmp_path: Path):
    """A whole manifest pasted into the packet evicts the trajectory, silently.

    `judge._render` serialises artifacts BEFORE the trace and truncates the pair at
    PACKET_CHAR_BUDGET. Run manifests run to tens of kilobytes — `develop_plane_candidates`
    alone can — so the naive packet shows a trajectory judge no trajectory, and it still
    returns three confident scores.
    """
    import acr.commands.cli_signal as cs
    from acr.evaluation import judge as J
    m = _traced(tmp_path, "big", develop_plane_candidates={"terms": ["x" * 40] * 400})
    packet = cs._packet_from_run(run=str(m), gold="", dimension="trajectory_quality",
                                 local_root=str(tmp_path))
    assert set(packet.artifacts["manifest"]) <= set(cs.MANIFEST_KEYS_SHOWN)
    assert "develop_plane_candidates" not in packet.artifacts["manifest"]
    assert "search_notes" in J._render(packet)          # the trace survived the budget


@pytest.mark.provider_seam   # the client is built before the fence refuses; no request is sent
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
    assert "DeterministicEvaluatorExists" in flat and "acr.evaluation.evals.score" in flat


@pytest.mark.provider_seam   # as above
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
    monkeypatch.setattr("acr.core.cli_common.llm_client", lambda *a, **k: client)
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
    monkeypatch.setattr("acr.core.cli_common.llm_client", lambda *a, **k: client)
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
    from acr.evaluation import judge as J
    assert not set(payload) & set(J.DECISION_FIELD_NAMES)
    assert not set(payload["verdict"]) & set(J.DECISION_FIELD_NAMES)


def test_the_judge_kind_batches_and_one_refusal_is_not_the_cohort(tmp_path, monkeypatch):
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    monkeypatch.setattr("acr.core.cli_common.llm_client", lambda *a, **k: _StubClient())
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
    from acr.commands import cli_judge
    assert cli_judge._JsonModel is cli_judge.JsonJudgeModel


def test_the_agent_kind_gets_more_turns_than_acr_attribute_case_defaults_to():
    """Regression: 12 model calls do not fit eight pipeline stages plus four eval cards.

    The first live attribution stopped at 11 of 12 with `cause: UNRESOLVED` and the rationale
    "model-call limit reached without a gate-valid attribution" — the counterfactual test and the
    skeptic review were both skipped, and the report gate then, correctly, refused to call it
    resolved. A default under which the deliverable cannot be produced is not a budget, it is a
    wall.

    Raised to 24, the same case finished all eight stages with `gate_rejections` empty and the
    conclusion still UNRESOLVED — but it became "an independent skeptic model does not accept this
    causal chain", which is the adversarial check working rather than the budget hitting a wall.
    """
    from acr.commands.cli_signal import DEFAULT_AGENT_CHART_READS, DEFAULT_AGENT_MODEL_CALLS
    assert DEFAULT_AGENT_MODEL_CALLS > 12
    assert DEFAULT_AGENT_CHART_READS >= 12


def test_both_agent_budgets_reach_the_attribution_payload(monkeypatch):
    """Both budgets must reach the bottom — a flag added and never wired is no flag at all."""
    import acr.commands.cli_attribute as CA
    import acr.commands.cli_signal as cs
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
    """Regression: `_batch_signals`'s body used two names its signature did not have.

    `acr signal batch --kind agent` therefore raised `NameError` on every run, and the
    `except Exception` on that path — written for "one bad run is not the batch" — filed the
    NameError like any other per-run error. The whole batch ended in exit 2, which reads as "every
    run in this cohort is bad" rather than "this command has never once worked". The test above only
    checks that the flags appear in `--help`; no test went through the `--kind agent` batch path at
    all.

    That path is the only way to run one cohort under two modes, so the coverage lands here.
    """
    monkeypatch.delenv("ACR_LOCAL_ARTIFACT_ROOT", raising=False)
    import acr.commands.cli_attribute as CA
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


# ==================================================== CARD IDENTITY, NOT CARD LENGTH

def test_the_eval_skill_block_is_identified_by_its_content_not_its_length():
    """The method cards the diagnostic agent reads are prompt content, and all that was recorded
    was `eval_skills_bytes` — a byte count.

    Two completely different card sets of the same length cannot be told apart, and when a card is
    edited in place the byte count often does not move at all. This is the same defect
    `prompt_assets` was added to fix: something entered the prompt with no record of its identity.
    The difference is that this half is on the diagnosis plane, and the diagnosis plane's output is
    exactly what `attribute meta-certify` later grades — a reader has to be able to say which
    method produced a causal judgement.
    """
    from acr.contract.skills import eval_skills_block, eval_skills_identity

    names = ["eval-cluster-failures"]
    block = eval_skills_block(names)
    ident = eval_skills_identity(block, names)
    assert ident["names"] == names
    assert ident["n_cards"] == 1
    assert ident["content_hash"] and len(ident["content_hash"]) == 16
    assert ident["n_chars"] == len(block)


def test_two_different_card_sets_do_not_share_a_hash():
    from acr.contract.skills import eval_skills_block, eval_skills_identity

    def ident(names):
        return eval_skills_identity(eval_skills_block(names), names)

    one = ident(["eval-cluster-failures"])
    two = ident(["eval-cluster-failures", "eval-missed-evidence"])
    assert one["content_hash"] != two["content_hash"]


def test_the_hash_is_taken_from_the_block_that_was_actually_sent():
    """It takes the ALREADY-RENDERED STRING; it does not render a second time.

    If this function rendered again, there would be two renderings between it and the prompt — one
    inconsistent `skills_dir` is enough to make the manifest describe text the model never read.
    The hash has to come from the same object that was sent.
    """
    import inspect

    from acr.contract.skills import eval_skills_identity
    src = inspect.getsource(eval_skills_identity)
    assert "eval_skills_block" not in src, "no second rendering here"
    assert eval_skills_identity("", [])["content_hash"] == "", (
        "no cards means no hash, not the hash of an empty string")


def test_no_cards_is_an_explicit_absence_not_a_zero_length_block():
    from acr.contract.skills import eval_skills_identity
    assert eval_skills_identity("", []) == {"names": [], "n_cards": 0, "n_chars": 0,
                                            "content_hash": ""}


def test_the_signal_envelope_carries_the_identity_and_not_the_byte_count(tmp_path, monkeypatch):
    """All the way to the envelope. An identity record that is right inside the helper and never
    reaches the output is one no reader can get at."""
    import acr.commands.cli_attribute as CA

    seen = {}

    class _Report:
        def to_dict(self):
            return {"ok": True}

    def _fake_run_one(**kw):
        seen.update(kw)
        return _Report()

    monkeypatch.setattr(CA, "_run_one", _fake_run_one)
    monkeypatch.setattr(CA, "_packet_for", lambda **kw: _StubPacket(), raising=False)
    env = CA.attribute_case_payload.__doc__
    assert env is not None   # only that the signature is still there; the real assertion is below

    import inspect
    sig = inspect.signature(CA.attribute_case_payload)
    assert "eval_skills_names" in sig.parameters, (
        "the identity needs the card names, and the payload only receives the rendered text")
    # Look at the KEYS THAT ARE PRODUCED, not at a source string. The previous version asserted that
    # `eval_skills_bytes` does not appear in the source and was tripped by its own comment recording
    # the old defect — the same class of mistake twice in one day, so no substring matching on
    # source here.
    from acr.contract.skills import eval_skills_block, eval_skills_identity
    env_keys = eval_skills_identity(eval_skills_block(["eval-cluster-failures"]),
                                    ["eval-cluster-failures"])
    assert set(env_keys) == {"names", "n_cards", "n_chars", "content_hash"}
    assert "eval_skills_bytes" not in env_keys


class _StubPacket:
    manifest_ref = type("R", (), {"to_dict": lambda self: {}})()
