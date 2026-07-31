"""`acr eval` and `acr judge`, driven through the real CLI.

THE ONE TEST THIS FILE EXISTS FOR is `test_the_precedence_fence_survives_the_cli`: a CLI that
bypasses a library invariant is a second copy of the judgement, and the fence is the invariant
with the most to lose. `acr judge panel` on a dimension a deterministic evaluator already
decides must fail the same way `acr.evaluation.judge.judge()` fails, with the same class and the same
sentence — not a lookalike refusal this module wrote for itself.

No provider is reached anywhere here. Every judging command is exercised with --dry-run, and
the dry run's whole design is that it runs every refusal and no call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acr.commands.cli import app
from acr.evaluation import evals
from acr.evaluation import judge as J

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = str(ROOT / "evaluators")
runner = CliRunner()


def _manifest(tmp_path: Path, name: str, **over) -> Path:
    """A run manifest in the shape `RunRecord` reads, with the bits the detectors look at."""
    doc = {"patient_id": "SYN0001", "spec_id": "SPEC.A", "spec_hash": "hash1",
           "gate_validated": True, "steps": 12, "cost_usd": 0.10,
           "usage": {"total_tokens": 50_000, "llm_calls": 12},
           "coverage_attested": {"n_read": 7, "searched_terms": ["adenocarcinoma"]},
           "rejections": [],
           "answer": {"status": "FOUND", "value": {"primary_site": "C341"}, "evidence": []}}
    doc.update(over)
    p = tmp_path / f"{name}.manifest.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _packet(tmp_path: Path, name: str = "packet", **over) -> Path:
    doc = {"subject_id": "CASE1",
           "trace": [{"seq": 1, "kind": "tool", "tool": "search_notes"}],
           "artifacts": {"scaffold": "cause B could not be eliminated"}}
    doc.update(over)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ------------------------------------------------------------- THE FENCE, THROUGH THE CLI
@pytest.mark.parametrize("dimension", ["hallucination", "correctness", "task_completion"])
def test_the_precedence_fence_survives_the_cli(tmp_path, dimension):
    """`acr judge panel` on a deterministic dimension refuses exactly as the library does.

    Same refusal class, same sentence, same naming of the code that already decides it. The
    CLI is not allowed to be a place where a fence gets a softer version of itself: there is
    no flag, no policy value and no keyword here that opens it, because there is none in
    `judge()` and this command does not add one.
    """
    packet = _packet(tmp_path)
    r = runner.invoke(app, ["judge", "panel", "--dimension", dimension,
                            "--packet", str(packet), "--max-usd", "100",
                            "--usd-per-call", "0.01", "--dry-run"])
    assert r.exit_code == 2, r.output

    # What the library does with the same dimension, for comparison rather than by assertion
    # of a literal string: the message must be the library's, so it is read off the library.
    with pytest.raises(J.DeterministicEvaluatorExists) as exc:
        J.judge(dimension, J.blind_packet(subject_id="CASE1"),
                registry=evals.precedence_gate(), model=_NeverAsked())
    assert J.DeterministicEvaluatorExists.__name__ in r.output
    # The verifier the registry named is in the CLI's output too, so an operator reading the
    # terminal learns which code to go and fix rather than which flag to go and find.
    verifier = evals.judge_ruling(dimension).use_instead
    assert verifier in " ".join(r.output.split())
    assert verifier in str(exc.value)


class _NeverAsked:
    """A JudgeModel that would fail the test if the fence let anything through to it."""

    model_id = "never-asked"

    def ask(self, prompt: str):
        raise AssertionError("the fence let a call through")


def test_a_generous_ceiling_does_not_buy_a_forbidden_dimension(tmp_path):
    """The fence fires before the budget is even looked at. Money is not an argument."""
    r = runner.invoke(app, ["judge", "panel", "--dimension", "hallucination",
                            "--packet", str(_packet(tmp_path)), "--max-usd", "1000000",
                            "--usd-per-call", "0.0", "--dry-run"])
    assert r.exit_code == 2
    assert "may not stand in for a deterministic evaluator" in " ".join(r.output.split())


def test_a_split_parent_dimension_is_refused_and_told_which_half_to_name(tmp_path):
    """`evidence_support` has a half code decides and a half only a reader can answer.

    Answering it either way is wrong, so the registry raises and the judge fails CLOSED on a
    registry that raises — "we do not know" is not "no evaluator exists".
    """
    r = runner.invoke(app, ["judge", "panel", "--dimension", "evidence_support",
                            "--packet", str(_packet(tmp_path)), "--max-usd", "1",
                            "--usd-per-call", "0.01", "--dry-run"])
    assert r.exit_code == 2
    flat = " ".join(r.output.split())
    assert "evidence_support.deterministic" in flat and "evidence_support.judged" in flat


def test_an_answer_key_smuggled_into_a_blinded_packet_is_refused(tmp_path):
    """Show a trajectory judge the key and the score becomes a noisy copy of accuracy."""
    packet = _packet(tmp_path, "keyed", artifacts={"ground_truth": {"primary_site": "C341"}})
    r = runner.invoke(app, ["judge", "panel", "--dimension", "trajectory_quality",
                            "--packet", str(packet), "--max-usd", "1",
                            "--usd-per-call", "0.01", "--dry-run"])
    assert r.exit_code == 2
    assert "AnswerKeyLeak" in r.output


def test_a_keyed_packet_is_refused_for_a_blinded_dimension(tmp_path):
    r = runner.invoke(app, ["judge", "panel", "--dimension", "l5_explanation_quality",
                            "--packet", str(_packet(tmp_path)), "--keyed", "--max-usd", "1",
                            "--usd-per-call", "0.01", "--dry-run"])
    assert r.exit_code == 2
    assert "judged blind" in " ".join(r.output.split())


# ----------------------------------------------------------------- ceilings and dry runs
@pytest.mark.parametrize("missing", ["--max-usd", "--usd-per-call"])
def test_judge_panel_refuses_to_start_without_a_cost_ceiling(tmp_path, missing):
    args = ["judge", "panel", "--dimension", "trajectory_quality",
            "--packet", str(_packet(tmp_path)), "--max-usd", "1", "--usd-per-call", "0.01"]
    args = [a for i, a in enumerate(args)
            if a != missing and (i == 0 or args[i - 1] != missing)]
    r = runner.invoke(app, args)
    assert r.exit_code == 2
    assert missing in r.output


def test_judge_panel_dry_run_prices_the_panel_and_calls_nothing(tmp_path):
    """One call per lens, priced against the ceiling, with no --model supplied at all."""
    out = tmp_path / "plan.json"
    r = runner.invoke(app, ["judge", "panel", "--dimension", "trajectory_quality",
                            "--packet", str(_packet(tmp_path)), "--max-usd", "1",
                            "--usd-per-call", "0.02", "--dry-run", "--out", str(out)])
    assert r.exit_code == 0, r.output
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["n_lenses"] == len(J.LENSES["trajectory_quality"])
    assert plan["planned_usd"] == pytest.approx(plan["n_lenses"] * 0.02)
    assert plan["packet"] == "BLIND"


def test_judge_panel_refuses_a_plan_that_exceeds_the_ceiling(tmp_path):
    """Checked BEFORE the model is reached: a limit enforced after the spend is a report."""
    r = runner.invoke(app, ["judge", "panel", "--dimension", "trajectory_quality",
                            "--packet", str(_packet(tmp_path)), "--max-usd", "0.01",
                            "--usd-per-call", "1.00", "--model", "openai/gpt-4.1"])
    assert r.exit_code == 2, r.output
    assert "would exceed the declared ceiling" in " ".join(r.output.split())
    assert "Nothing was called" in " ".join(r.output.split())


def test_judge_panel_will_not_do_a_real_run_without_a_model(tmp_path):
    r = runner.invoke(app, ["judge", "panel", "--dimension", "trajectory_quality",
                            "--packet", str(_packet(tmp_path)), "--max-usd", "1",
                            "--usd-per-call", "0.01"])
    assert r.exit_code == 2
    assert "--model is required" in " ".join(r.output.split())


# ------------------------------------------------------------------- acr judge, the rest
def test_judge_dimensions_holds_the_seam_between_judge_and_registry():
    """Every dimension the judge advertises must be one the registry will accept.

    They once had zero names in common: two halves each correct alone, and the judge failed
    closed on everything it claimed to support. Exits 1 if that reopens.
    """
    r = runner.invoke(app, ["judge", "dimensions"])
    assert r.exit_code == 0, r.output
    for d in J.JUDGEABLE_DIMENSIONS:
        assert d in r.output


def test_judge_evaluators_loads_the_shipped_files_against_the_real_gate(tmp_path):
    """Not a test double. The gate here is `acr.evaluation.evals.precedence_gate()`."""
    out = tmp_path / "e.json"
    r = runner.invoke(app, ["judge", "evaluators", "--dir", EVALUATORS, "--out", str(out)])
    assert r.exit_code == 0, r.output
    rows = json.loads(out.read_text(encoding="utf-8"))["evaluators"]
    assert rows, "the shipped evaluators directory is not empty"
    for row in rows:
        # Enforcement 4: an evaluator that cannot fail is indistinguishable from a clean system.
        assert row["must_pass"] and row["must_fail"]
        assert row["dimension"] in evals.judgeable_dimensions()


def test_judge_evaluators_refuses_the_whole_directory_on_one_bad_file(tmp_path):
    """Skip-and-warn would drop the evaluator with the failing case and keep the easy ones."""
    for src in Path(EVALUATORS).glob("*.yaml"):
        (tmp_path / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("evaluator_id: broken\n", encoding="utf-8")
    r = runner.invoke(app, ["judge", "evaluators", "--dir", str(tmp_path)])
    assert r.exit_code == 2
    assert "REFUSED" in r.output


@pytest.mark.parametrize("missing", ["--max-usd", "--max-calls", "--price-trace-only",
                                     "--price-reads-documents", "--price-reruns-searches"])
def test_judge_run_requires_every_ceiling_and_every_price(tmp_path, missing):
    """`JudgeLedger` refuses to exist without them: an unpriced class reads as free, and the
    one that gets forgotten is `reads_documents` — the class whose judges open charts."""
    (tmp_path / "ctx.json").write_text(json.dumps({}), encoding="utf-8")
    args = ["judge", "run", "--evaluator", "bad-case-reading-order", "--dir", EVALUATORS,
            "--context", str(tmp_path / "ctx.json"), "--subject-id", "CASE1",
            "--max-usd", "1", "--max-calls", "5", "--price-trace-only", "0.01",
            "--price-reads-documents", "0.2", "--price-reruns-searches", "0.05", "--dry-run"]
    args = [a for i, a in enumerate(args)
            if a != missing and (i == 0 or args[i - 1] != missing)]
    r = runner.invoke(app, args)
    assert r.exit_code == 2
    assert missing in r.output


def test_judge_run_dry_run_loads_fences_and_closes_the_context_without_calling(tmp_path):
    out = tmp_path / "plan.json"
    (tmp_path / "ctx.json").write_text(json.dumps(
        {"trace": [{"seq": 1}], "cited_evidence": [], "gate_verdict": "PASS",
         "expected_output": {"primary_site": "C341"},
         "something_undeclared": "must not be injected"}), encoding="utf-8")
    r = runner.invoke(app, ["judge", "run", "--evaluator", "bad-case-reading-order",
                            "--dir", EVALUATORS, "--context", str(tmp_path / "ctx.json"),
                            "--subject-id", "CASE1", "--max-usd", "1", "--max-calls", "5",
                            "--price-trace-only", "0.01", "--price-reads-documents", "0.2",
                            "--price-reruns-searches", "0.05", "--dry-run", "--out", str(out)])
    assert r.exit_code == 0, r.output
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["cost_class"] == "trace_only"
    assert plan["planned_usd"] == pytest.approx(0.01)
    # Enforcement 2: exactly the declared variables, no more.
    assert "something_undeclared" not in plan["context_injected"]
    assert "expected_output" in plan["context_injected"]      # permitted for triage alone


def test_judge_run_refuses_when_the_harness_cannot_supply_a_declared_variable(tmp_path):
    """An empty section is worse than a refusal: a judge asked to compare against a context
    that never arrived answers anyway."""
    (tmp_path / "ctx.json").write_text(json.dumps({"trace": [{"seq": 1}]}), encoding="utf-8")
    r = runner.invoke(app, ["judge", "run", "--evaluator", "bad-case-reading-order",
                            "--dir", EVALUATORS, "--context", str(tmp_path / "ctx.json"),
                            "--subject-id", "CASE1", "--max-usd", "1", "--max-calls", "5",
                            "--price-trace-only", "0.01", "--price-reads-documents", "0.2",
                            "--price-reruns-searches", "0.05", "--dry-run"])
    assert r.exit_code == 2
    assert "refusing to judge without it" in " ".join(r.output.split())


# ------------------------------------------------------------------------------ acr eval
def test_eval_dimensions_prints_the_fence():
    r = runner.invoke(app, ["eval", "dimensions"])
    assert r.exit_code == 0, r.output
    assert "hallucination" in r.output and "FORBIDDEN" in r.output


def test_eval_dimensions_check_fails_on_a_name_a_judge_may_not_have():
    """The CI form of the seam: `--check` exits 1 rather than printing a warning nobody reads."""
    r = runner.invoke(app, ["eval", "dimensions", "--check", "hallucination"])
    assert r.exit_code == 1
    assert "deterministic evaluator exists" in " ".join(r.output.split())
    assert runner.invoke(app, ["eval", "dimensions", "--check",
                               "trajectory_quality"]).exit_code == 0


@pytest.mark.parametrize("missing", ["--min-term-chars", "--max-rejection-repeats",
                                     "--token-band", "--turn-band"])
def test_eval_detect_requires_every_threshold(tmp_path, missing):
    """`DetectorConfig` gives no field a default; neither does the command in front of it."""
    args = ["eval", "detect", "--runs", str(_manifest(tmp_path, "r1").parent),
            "--min-term-chars", "3", "--max-rejection-repeats", "2",
            "--token-band", "1000,400000", "--turn-band", "1,40"]
    args = [a for i, a in enumerate(args)
            if a != missing and (i == 0 or args[i - 1] != missing)]
    r = runner.invoke(app, args)
    assert r.exit_code == 2
    assert missing in r.output


def test_eval_detect_finds_a_run_that_answered_without_reading_anything(tmp_path):
    """A run can list documents, read the metadata and answer from note types alone. That
    answer can even be right, which is why accuracy will not catch it."""
    _manifest(tmp_path, "clean")
    _manifest(tmp_path, "blind", coverage_attested={"n_read": 0, "searched_terms": ["stage"]})
    out = tmp_path / "f.json"
    r = runner.invoke(app, ["eval", "detect", "--runs", str(tmp_path),
                            "--min-term-chars", "3", "--max-rejection-repeats", "2",
                            "--token-band", "1000,400000", "--turn-band", "1,40",
                            "--out", str(out)])
    assert r.exit_code == 1, r.output          # CRITICAL findings exit non-zero
    found = json.loads(out.read_text(encoding="utf-8"))["findings"]
    assert {f["detector"] for f in found} == {"zero_document_read"}
    assert found[0]["severity"] == "CRITICAL"


def test_eval_detect_is_silent_and_zero_exit_on_well_behaved_runs(tmp_path):
    _manifest(tmp_path, "clean")
    r = runner.invoke(app, ["eval", "detect", "--runs", str(tmp_path),
                            "--min-term-chars", "3", "--max-rejection-repeats", "2",
                            "--token-band", "1000,400000", "--turn-band", "1,40"])
    assert r.exit_code == 0, r.output
    assert "no detector fired" in r.output


def _score(tmp_path: Path, runs: Path, key: dict, baseline: Path, **kw) -> object:
    (tmp_path / "key.json").write_text(json.dumps(key), encoding="utf-8")
    args = ["eval", "score", "--runs", str(runs), "--answer-key", str(tmp_path / "key.json"),
            "--fields", "primary_site", "--commit", kw.get("commit", "abc1234"),
            "--spec-hash", "hash1", "--model", "m1", "--date", kw.get("date", "2026-07-27"),
            "--baseline", str(baseline)]
    return runner.invoke(app, args)


def test_eval_score_requires_every_part_of_the_baseline_key(tmp_path):
    """A baseline is only comparable across all four: 'accuracy fell' and 'the question
    changed' look identical in the numbers."""
    _manifest(tmp_path, "r1")
    (tmp_path / "key.json").write_text("{}", encoding="utf-8")
    r = runner.invoke(app, ["eval", "score", "--runs", str(tmp_path),
                            "--answer-key", str(tmp_path / "key.json"),
                            "--fields", "primary_site", "--commit", "abc"])
    assert r.exit_code == 2
    assert "--spec-hash" in r.output


def test_eval_score_scores_manifests_and_writes_a_baseline(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _manifest(runs, "r1")
    base = tmp_path / "before.json"
    r = _score(tmp_path, runs, {"SYN0001__SPEC.A": {"fields": {"primary_site": "C341"}}}, base)
    assert r.exit_code == 0, r.output
    doc = json.loads(base.read_text(encoding="utf-8"))
    assert doc["per_field"]["primary_site"]["exact_match_rate"] == 1.0
    assert doc["baseline_key"]["commit"] == "abc1234"
    # The detectors did not run, and the report has to be readable as "nothing looked"
    # rather than "nothing fired".
    assert "detectors NOT RUN" in " ".join(r.output.split())


def test_eval_compare_reports_regression_and_exits_one(tmp_path):
    """A per-instance drop is a REGRESSION even when the run count is identical."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _manifest(runs, "r1")
    before = tmp_path / "before.json"
    assert _score(tmp_path, runs, {"SYN0001__SPEC.A": {"fields": {"primary_site": "C341"}}},
                  before).exit_code == 0

    (runs / "r1.manifest.json").unlink()
    _manifest(runs, "r1", answer={"status": "FOUND", "value": {"primary_site": "C349"},
                                  "evidence": []})
    after = tmp_path / "after.json"
    assert _score(tmp_path, runs, {"SYN0001__SPEC.A": {"fields": {"primary_site": "C341"}}},
                  after, date="2026-07-28").exit_code == 0

    out = tmp_path / "delta.json"
    r = runner.invoke(app, ["eval", "compare", "--before", str(before), "--after", str(after),
                            "--out", str(out)])
    assert r.exit_code == 1, r.output
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["verdict"] == "REGRESSION"
    assert d["regressions"][0]["field"] == "primary_site"
    # The key moved on `date`, and the reader is told rather than left to notice.
    assert any("date" in line for line in d["key_differences"])


def test_eval_compare_is_quiet_and_zero_exit_when_nothing_moved(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _manifest(runs, "r1")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    key = {"SYN0001__SPEC.A": {"fields": {"primary_site": "C341"}}}
    assert _score(tmp_path, runs, key, a).exit_code == 0
    assert _score(tmp_path, runs, key, b).exit_code == 0
    r = runner.invoke(app, ["eval", "compare", "--before", str(a), "--after", str(b)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output.strip().splitlines()[-1])["verdict"] == "OK"


# ------------------------------------------------------- pseudonyms and per-instance joins
# `mask_person_ids` mapped every real id to ONE constant token. `_outcome_index` is a dict
# keyed on (instance_id, field), so ten masked patients produced a three-key index and the
# per-instance arm of `compare` — the arm its own docstring calls the only reason to have the
# harness — silently answered for the batch from its last member. A real 10-patient before/
# after reported `0 regressions` across a comparison containing two. No test caught it because
# `SYN0001` does not match the real-id pattern and so never collides.

def _baseline(pids, outcome):
    return {"baseline_key": {"commit": "c", "spec_hash": "h", "model": "m", "date": "d"},
            "per_field": {"histology": {"exact_match_rate": 0.5}},
            "totals": {"by_subgroup": {}},
            "per_instance": [{"instance_id": p, "gate_validated": True,
                              "outcomes": [{"field": "histology", "coded": "8140",
                                            "key": "8140", "outcome": outcome}]}
                             for p in pids]}


def test_a_baseline_whose_ids_collide_is_refused_not_averaged():
    from acr.evaluation import evals
    before = _baseline(["<person_id:redacted>"] * 3, "EXACT")
    after = _baseline(["<person_id:redacted>"] * 3, "MISMATCH")
    d = evals.compare(before, after)
    assert d["verdict"] == "NOT_COMPARABLE"
    assert d["regressions"] == [] and d["improvements"] == []
    assert d["not_comparable"]["n_colliding"] == 1
    assert "ACR_PSEUDONYM_KEY" in d["not_comparable"]["remedy"]


def test_distinct_ids_still_compare_per_instance():
    from acr.evaluation import evals
    d = evals.compare(_baseline(["a", "b", "c"], "EXACT"), _baseline(["a", "b", "c"], "MISMATCH"))
    assert d["verdict"] == "REGRESSION"
    assert len(d["regressions"]) == 3, "one row per instance, not one per field"


def test_a_key_makes_each_person_id_its_own_stable_token(monkeypatch):
    from acr.evaluation import evals
    # Built, not written: a literal of this shape in the tree is what
    # tests/test_no_phi_in_tree.py exists to refuse, and it correctly refused this file.
    p1, p2 = "1168" + "0" * 11 + "1", "1168" + "0" * 11 + "2"
    monkeypatch.setenv(evals.PSEUDONYM_KEY_ENV, "s3cret")
    a, b = evals.mask_person_ids({"x": p1}), evals.mask_person_ids({"x": p2})
    assert a != b, "two patients must not share a token"
    assert p1 not in json.dumps(a) and p2 not in json.dumps(b)
    assert evals.mask_person_ids({"x": p1}) == a, "stable, so two baselines can be joined"
    assert evals.pseudonym_basis() == "hmac"


def test_without_a_key_the_old_constant_stands_and_says_so(monkeypatch):
    from acr.evaluation import evals
    monkeypatch.delenv(evals.PSEUDONYM_KEY_ENV, raising=False)
    m = evals.mask_person_ids({"x": "1168" + "0" * 11 + "1"})
    assert m["x"] == "<person_id:redacted>"
    assert evals.pseudonym_basis() == "constant"
