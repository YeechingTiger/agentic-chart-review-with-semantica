"""§7 — the evaluation plane, and the three things it must refuse to do.

The property under test is never "the harness computes a number". It is that the number
cannot be produced by the three routes that would make it meaningless:

  * a model judge standing in for a check the repo can compute exactly (`assert_judge_allowed`
    must RAISE, and a dimension nobody registered must raise rather than default to allowed);
  * a correct abstention scored as a failure — the way a task-completion judge launders the
    charts with no admissible evidence, which cluster on outside-hospital and declined
    biopsies and are therefore never missing at random;
  * an aggregate that rose while an instance or a subgroup fell.

The last one has its own test with the mean held CONSTANT while a chart starts guessing:
`test_guessing_over_an_abstention_is_a_regression_the_mean_cannot_see`.

Every fixture below is fabricated. No corpus is read, no manifest under `runs/` is opened,
and no model is called. The real-corpus person_id shape (a 1168 prefix and twelve digits) is
CONSTRUCTED AT RUNTIME rather than typed, so this file can exercise the redaction path
without becoming the thing tests/test_no_phi_in_tree.py exists to forbid.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from acr import evals as E

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Shaped like a real person_id, assembled at import so the literal never appears in source.
OTHER_PATIENT = "1168" + "0" * 11 + "7"

CFG = E.DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                       token_band=(1_000, 200_000), turn_band=(3, 60))

SPEC = "SYN.400.site_histology"


def manifest(patient="SYN0001", *, status="FOUND", value=None, gate=True, n_read=7,
             searched=("adenocarcinoma",), tokens=50_000, turns=12, cost=0.42,
             rejections=(), evidence=()) -> dict:
    return {"patient_id": patient, "spec_id": SPEC, "spec_hash": "abc123",
            "answer": {"status": status, "value": value or {}},
            "gate_validated": gate, "rejections": list(rejections),
            "usage": {"total_tokens": tokens, "llm_calls": turns}, "cost_usd": cost,
            "coverage_attested": {"n_read": n_read, "searched_terms": list(searched)},
            "evidence": list(evidence)}


def run(**kw) -> E.RunRecord:
    return E.RunRecord(manifest(**kw), source="synthetic")


def names(findings) -> list[str]:
    return [f.detector for f in findings]


# ============================================================ PART 1: precedence registry
def test_the_three_llm_judge_metrics_are_registered_as_deterministic():
    """hallucination, correctness and task completion all have exact equivalents here.

    If any of these three ever flips to `deterministic=False`, the repo has given up the one
    advantage it has over the usual agent-eval stack: verifiable rewards.
    """
    for dim in ("hallucination", "correctness", "task_completion"):
        d = E.REGISTRY[dim]
        assert d.deterministic and d.verifier, dim
        assert d.replaces_judge_metric, f"{dim} must name the judge metric it replaces"


@pytest.mark.parametrize("dim", [n for n, d in E.REGISTRY.items() if d.deterministic])
def test_a_judge_is_forbidden_wherever_an_exact_check_exists(dim):
    with pytest.raises(E.JudgeForbidden) as exc:
        E.assert_judge_allowed(dim)
    # The refusal has to carry the alternative, or the operator just turns the judge back on.
    assert E.REGISTRY[dim].verifier in str(exc.value)
    assert E.judge_ruling(dim).allowed is False


# Split parents are neither judged nor deterministic and are excluded here BY THE REGISTRY's
# own accessor, not by a hand-maintained skip list; each is covered instead by
# test_a_split_parent_refuses_to_be_ruled_on_as_a_whole below.
@pytest.mark.parametrize("dim", E.judgeable_dimensions())
def test_a_judge_is_permitted_only_where_no_exact_check_exists(dim):
    r = E.assert_judge_allowed(dim)
    assert r.allowed and r.use_instead is None and r.reason


def test_an_unregistered_dimension_raises_instead_of_defaulting_to_allowed():
    """Fail closed. 'We have not decided' must never render as 'a judge may proceed'."""
    with pytest.raises(E.UnknownDimension):
        E.judge_ruling("vibes")
    with pytest.raises(E.UnknownDimension):
        E.assert_judge_allowed("vibes")


# ------------------------------------------------ the fence is PER SUB-QUESTION
# A per-dimension fence has to give one answer for a dimension with a deterministic half and
# a judged half, and is wrong whichever it gives: it either forbids "does this quote actually
# support this value", which nothing can compute, or it licenses a model to re-decide
# admissibility the gate already computed from the spec.
@pytest.mark.parametrize("parent", ["evidence_support", "step_efficiency"])
def test_a_split_parent_refuses_to_be_ruled_on_as_a_whole(parent):
    with pytest.raises(E.DimensionIsSplit) as exc:
        E.judge_ruling(parent)
    halves = E.REGISTRY[parent].sub_questions
    assert len(halves) == 2
    # The refusal names both halves, or the reader has to go and find them.
    assert all(h in str(exc.value) for h in halves)
    assert isinstance(exc.value, E.UnknownDimension)      # every fail-closed caller stays closed


@pytest.mark.parametrize("parent,det,judged", [
    ("evidence_support", "evidence_support.deterministic", "evidence_support.judged"),
    ("step_efficiency", "step_efficiency.deterministic", "step_efficiency.judged")])
def test_one_half_is_forbidden_and_the_other_is_permitted(parent, det, judged):
    with pytest.raises(E.JudgeForbidden) as exc:
        E.assert_judge_allowed(det)
    assert E.REGISTRY[det].verifier in str(exc.value)
    assert E.assert_judge_allowed(judged).allowed
    assert E.REGISTRY[parent].sub_questions == (det, judged)


def test_the_registry_refuses_a_parent_that_names_a_half_nobody_registered(monkeypatch):
    """A refusal that points at an unregistered dimension is a fence with a hole in it."""
    bad = dict(E.REGISTRY)
    bad["ghost"] = E.Dimension("ghost", False, "SPLIT", None, None, "y",
                               sub_questions=("ghost.judged",))
    monkeypatch.setattr(E, "REGISTRY", bad)
    with pytest.raises(ValueError, match="ghost.judged"):
        E._validate_registry()


def test_a_sub_question_row_with_no_parent_cannot_load(monkeypatch):
    """Otherwise `x.judged` exists while `x` still answers for the whole dimension — the
    per-dimension fence coming back in through the side door."""
    bad = dict(E.REGISTRY)
    bad["orphan.judged"] = E.Dimension("orphan.judged", False, "judge permitted", None, None,
                                       "y")
    monkeypatch.setattr(E, "REGISTRY", bad)
    with pytest.raises(ValueError, match="no parent row declares it"):
        E._validate_registry()


# ------------------------------------------------------------ one namespace, one spelling
def test_every_superseded_name_resolves_to_a_registered_row():
    """Two names for one question is how the two halves of this plane drifted apart. A name
    that once appeared in a report has to keep resolving or last month's numbers stop being
    readable, so superseded names alias rather than disappear."""
    for old, new in E.ALIASES.items():
        assert old not in E.REGISTRY, f"{old} is both a row and an alias"
        assert new in E.REGISTRY and not E.REGISTRY[new].sub_questions
        assert E.judge_ruling(old).dimension == new


@pytest.mark.parametrize("spelling", ["Evidence_Relevance", "  evidence_relevance  ",
                                      "EVIDENCE_SUPPORT.JUDGED"])
def test_case_and_padding_are_folded_before_the_lookup(spelling):
    """A precedence check bypassed by a typo is the same as no precedence check."""
    assert E.judge_ruling(spelling).dimension == "evidence_support.judged"


def test_the_seam_check_reports_a_deterministic_dimension_as_not_permitted():
    """`unknown_dimensions` is what a module advertising dimensions runs against the
    registry. It has to catch both 'never registered' and 'registered but fenced'."""
    problems = E.unknown_dimensions(["task_completion", "vibes", "evidence_support",
                                     "trajectory_quality"])
    assert set(problems) == {"task_completion", "vibes", "evidence_support"}
    assert "acr.answer_gate.check_gate" in problems["task_completion"]
    assert "split" in problems["evidence_support"]


def test_the_precedence_gate_answers_the_query_shape_the_judge_makes():
    """The one object that makes this registry answerable to `acr.judge`'s protocol. Before
    it existed the only thing satisfying that protocol was a test double, which is exactly
    how the seam could be wrong in production with every test on both sides passing."""
    gate = E.precedence_gate()
    assert gate.deterministic_evaluator_for("correctness") == "acr.evals.score"
    assert gate.deterministic_evaluator_for("cost_and_turns") == "acr.evals.detect_resource_band"
    assert gate.deterministic_evaluator_for("trajectory_quality") is None
    for unknown in ("vibes", "evidence_support"):
        with pytest.raises(E.UnknownDimension):
            gate.deterministic_evaluator_for(unknown)


def test_a_row_claiming_determinism_without_a_verifier_cannot_load(monkeypatch):
    bad = dict(E.REGISTRY)
    bad["bogus"] = E.Dimension("bogus", True, "trust me", None, None, "no reason")
    monkeypatch.setattr(E, "REGISTRY", bad)
    with pytest.raises(ValueError, match="bogus"):
        E._validate_registry()


def _defines(path: Path, dotted: str) -> bool:
    """Is `dotted` (e.g. 'Toolbox._t_record_evidence') defined in this source file?"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for part in dotted.split("."):
        match = next((n for n in ast.iter_child_nodes(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and n.name == part), None)
        if match is None:
            return False
        tree = match
    return True


@pytest.mark.parametrize("dim", [n for n, d in E.REGISTRY.items() if d.deterministic])
def test_every_deterministic_verifier_names_code_that_actually_exists(dim):
    """The registry is only worth anything if its claims resolve.

    Checked by AST rather than by import: resolving `acr.answer_gate.check_gate` through the
    import system would drag a provider SDK into this test process for no benefit.
    """
    parts = E.REGISTRY[dim].verifier.split(".")
    for cut in range(len(parts), 1, -1):
        path = SRC.joinpath(*parts[:cut]).with_suffix(".py")
        if path.is_file():
            assert _defines(path, ".".join(parts[cut:])) or cut == len(parts), \
                f"{E.REGISTRY[dim].verifier} does not exist"
            return
    pytest.fail(f"{E.REGISTRY[dim].verifier} resolves to no file under src/")


def test_abstention_correctness_is_its_own_dimension():
    """The structural guard against laundering: completion and abstention are separate rows,
    so a correct abstention can never be folded into 'did the agent finish'."""
    assert E.REGISTRY["abstention_correctness"].deterministic
    assert E.REGISTRY["task_completion"].name != E.REGISTRY["abstention_correctness"].name


FORBIDDEN_MODULES = {
    "llm", "graph", "deep_runner", "cli",                      # first-party paths to a model
    "openai", "anthropic", "litellm", "langchain", "langgraph", "deepagents",
    "requests", "httpx", "urllib", "socket", "http",           # and to a network
}


def test_no_model_is_reachable_from_this_module():
    """An eval plane that can call a model is an eval plane that will."""
    tree = ast.parse((SRC / "acr" / "evals.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & FORBIDDEN_MODULES, f"reachable: {imported & FORBIDDEN_MODULES}"


# ================================================== PART 2: abnormal-behaviour detectors
def test_submitting_with_zero_documents_read_fires_with_its_evidence():
    f, = E.detect_zero_document_read(run(status="FOUND", value={"primary_site": "C341"},
                                         n_read=0))
    assert (f.detector, f.severity) == ("zero_document_read", E.CRITICAL)
    assert f.evidence["n_documents_read"] == 0 and f.evidence["gate_validated"] is True


def test_zero_read_does_not_fire_when_documents_were_read_or_never_counted():
    assert E.detect_zero_document_read(run(n_read=7)) == []
    # n_read absent is "nobody counted", which is not a claim that nothing was read.
    m = manifest(n_read=0)
    m["coverage_attested"].pop("n_read")
    assert E.detect_zero_document_read(E.RunRecord(m)) == []


def test_a_read_recorded_only_in_the_trace_still_counts():
    m = manifest(n_read=0)
    tr = [{"seq": 0, "kind": "tool", "tool": "chart.read_documents_batch",
           "args": {"note_ids": ["Path-Report_2020-01-01", "Endoscopy_2020-02-02"]}}]
    assert E.RunRecord(m, tr).n_documents_read == 2
    assert E.detect_zero_document_read(E.RunRecord(m, tr)) == []


@pytest.mark.parametrize("term,reason", [
    ("t", "shorter"), ("", "empty"), ("  ", "empty"), (".*", "matches every document"),
    ("...", "no alphanumeric"), ("!!", "no alphanumeric")])
def test_degenerate_search_terms_fire(term, reason):
    f, = E.detect_degenerate_search(run(searched=(term,)), min_term_chars=3)
    assert f.detector == "degenerate_search" and f.severity == E.CRITICAL
    assert reason.split()[0] in f.evidence["reason"]
    assert f.evidence["min_term_chars"] == 3


def test_a_real_search_term_does_not_fire():
    assert E.detect_degenerate_search(run(searched=("squamous", "right upper lobe")),
                                      min_term_chars=3) == []


def test_the_degeneracy_threshold_is_required_and_sane():
    with pytest.raises(TypeError):
        E.detect_degenerate_search(run())                     # no default exists
    with pytest.raises(ValueError, match="min_term_chars"):
        E.detect_degenerate_search(run(), min_term_chars=None)
    with pytest.raises(ValueError, match="min_term_chars"):
        E.detect_degenerate_search(run(), min_term_chars=0)


def test_touching_another_patient_is_an_irb_finding_with_the_id_redacted():
    tr = [{"seq": 4, "kind": "tool", "tool": "chart.read_document",
           "args": {"patient": OTHER_PATIENT, "note_id": "Path-Report_2020-01-01"}}]
    f, = E.detect_patient_crossover(E.RunRecord(manifest(), tr),
                                    expected_patient="SYN0001")
    assert f.severity == E.IRB and f.evidence["other_patients"] == [OTHER_PATIENT]
    # Serialised for a report, the identifier must not survive.
    assert OTHER_PATIENT not in json.dumps(f.to_dict())
    assert "redacted" in json.dumps(f.to_dict())


def test_no_crossover_when_every_mention_is_the_expected_patient():
    tr = [{"seq": 1, "kind": "tool", "tool": "chart.list_documents",
           "args": {"patient": "SYN0001"}}]
    assert E.detect_patient_crossover(E.RunRecord(manifest(), tr),
                                      expected_patient="SYN0001") == []
    with pytest.raises(ValueError, match="expected_patient"):
        E.detect_patient_crossover(run(), expected_patient="")


def rej(seq, why, attempted):
    return {"seq": seq, "kind": "answer_rejected", "why": why, "attempted": attempted}


def test_the_same_rejection_against_an_unrevised_answer_is_a_loop():
    same = {"primary_site": "C349"}
    tr = [rej(1, "NOS code with no search", same), rej(2, "NOS  code with no  search", same)]
    f, = E.detect_rejection_loop(E.RunRecord(manifest(), tr), max_repeats=2)
    assert f.severity == E.CRITICAL and f.evidence["repeats"] == 2
    assert f.evidence["revision_observable"] is True
    assert f.evidence["tokens_after_loop"] == 50_000


def test_a_rejection_the_agent_answered_is_not_a_loop():
    """The gate rejecting twice and the agent revising twice is the design working."""
    tr = [rej(1, "NOS code with no search", {"primary_site": "C349"}),
          rej(2, "NOS code with no search", {"primary_site": "C341"})]
    assert E.detect_rejection_loop(E.RunRecord(manifest(), tr), max_repeats=2) == []


def test_a_manifest_only_repeat_is_reported_as_unverifiable_not_as_proven():
    reps = [{"why": "sampling not done"}, {"why": "sampling not done"}]
    f, = E.detect_rejection_loop(run(rejections=reps), max_repeats=2)
    assert f.severity == E.WARN and f.evidence["revision_observable"] is False


def test_the_loop_threshold_is_required_and_at_least_two():
    with pytest.raises(TypeError):
        E.detect_rejection_loop(run())
    for bad in (None, 1):
        with pytest.raises(ValueError, match="max_repeats"):
            E.detect_rejection_loop(run(), max_repeats=bad)


@pytest.mark.parametrize("tokens,turns,side", [(900_000, 12, "above"), (400, 12, "below"),
                                               (50_000, 200, "above"), (50_000, 1, "below")])
def test_resources_outside_the_declared_band_fire_on_both_sides(tokens, turns, side):
    f, = [x for x in E.detect_resource_band(run(tokens=tokens, turns=turns),
                                            token_band=(1_000, 200_000), turn_band=(3, 60))
          if x.detector == "resource_out_of_band"]
    assert f.evidence["side"] == side and f.severity == E.WARN


def test_an_unmeasured_counter_is_reported_rather_than_read_as_a_cheap_run():
    m = manifest()
    m.pop("usage")
    fs = E.detect_resource_band(E.RunRecord(m), token_band=(1_000, 200_000), turn_band=(3, 60))
    assert {f.detector for f in fs} == {"resource_unmeasured"}
    assert {f.evidence["metric"] for f in fs} == {"tokens", "turns"}


def test_bands_are_required_and_ordered():
    with pytest.raises(TypeError):
        E.detect_resource_band(run())
    with pytest.raises(ValueError, match="token_band"):
        E.detect_resource_band(run(), token_band=None, turn_band=(3, 60))
    with pytest.raises(ValueError, match="turn_band"):
        E.detect_resource_band(run(), token_band=(1, 2), turn_band=(60, 3))


def test_every_detector_threshold_must_be_supplied_by_the_caller():
    with pytest.raises(TypeError):
        E.DetectorConfig(min_term_chars=3)                     # no defaults anywhere
    with pytest.raises(ValueError, match="token_band"):
        E.DetectorConfig(min_term_chars=3, max_rejection_repeats=2, token_band=None,
                         turn_band=(3, 60))


def test_run_detectors_reports_the_irb_finding_first():
    tr = [{"seq": 1, "kind": "tool", "tool": "chart.read_document",
           "args": {"patient": OTHER_PATIENT}}]
    m = manifest(n_read=0, searched=("t",), tokens=900_000)
    fs = E.run_detectors(E.RunRecord(m, tr), config=CFG, expected_patient="SYN0001")
    assert fs[0].detector == "patient_crossover"
    assert set(names(fs)) >= {"patient_crossover", "degenerate_search",
                              "resource_out_of_band"}


# ==================================================== PART 3: the regression harness
def test_cost_is_read_from_the_key_a_real_manifest_actually_writes():
    """`spend.usd`, not `cost_usd`. Every baseline this repo produced reported cost unknown.

    `RunRecord.cost_usd` read `manifest["cost_usd"]`, a key nothing in this repo writes -- the
    priced ceiling in `spend.py` writes `spend: {usd, priced, cache_hit_rate, ...}`. So the
    ten-patient real batch of 2026-07-28 scored as `cost None / n_cost_unknown 10` while its
    manifests summed to $3.5247, each carrying its own price.

    Note that `manifest()` above builds the fictional `cost_usd` key, which is why the whole
    eval suite was green on a property no real run had: the fixture agreed with the bug. These
    two use the real shape.
    """
    priced = E.RunRecord({**manifest(), "spend": {"usd": 0.6752, "priced": True,
                                                  "model": "gpt-5.6-luna", "max_usd": 5.0}},
                         source="real-shape")
    assert priced.cost_usd == pytest.approx(0.6752)

    # And the legacy key still works, for a manifest written by something that does report it.
    legacy = E.RunRecord({**manifest(cost=0.11)}, source="legacy-shape")
    assert legacy.cost_usd == pytest.approx(0.11)


def test_an_unpriced_model_reads_as_unknown_and_never_as_zero():
    """`spend.usd` is None when the model is not in prices.json, and None must survive.

    A cost of 0.0 for an unpriced model is the worst available answer: it sums into a total
    that reads as measured, and `n_cost_unknown` -- the field whose whole job is to say how
    much of the total is missing -- goes to zero at the same time.
    """
    unpriced = E.RunRecord({**{k: v for k, v in manifest().items() if k != "cost_usd"},
                            "spend": {"usd": None, "priced": False, "model": "some-local-gguf"}},
                           source="unpriced")
    assert unpriced.cost_usd is None, "an unpriced model is unknown, not free"



KEY = {
    f"SYN0001__{SPEC}": {"fields": {"primary_site": "C341", "histology": "8070"},
                         "subgroups": ["squamous"]},
    # No admissible evidence on this chart: abstention IS the answer key.
    f"SYN0002__{SPEC}": {"fields": {"primary_site": None, "histology": None},
                         "subgroups": ["outside_hospital"]},
}
FIELDS = ["primary_site", "histology"]
KEY_A = E.BaselineKey(commit="c1b5914", spec_hash="abc123", model="syn-model",
                      date="2026-07-27")


def score(runs, key=KEY_A, **kw):
    return E.score(runs, KEY, fields=FIELDS, key=key, **kw)


def outcomes(report, instance_id) -> dict[str, str]:
    r, = [x for x in report.per_instance if x.instance_id == instance_id]
    return {o.field: o.outcome for o in r.outcomes}


CORRECT = [run(patient="SYN0001", value={"primary_site": "C341", "histology": "8070"}),
           run(patient="SYN0002", status="EVIDENCE_INSUFFICIENT", gate=True)]


def test_exact_match_and_abstention_are_scored_separately():
    rep = score(CORRECT)
    assert outcomes(rep, f"SYN0001__{SPEC}") == {"primary_site": E.EXACT,
                                                 "histology": E.EXACT}
    assert outcomes(rep, f"SYN0002__{SPEC}") == {"primary_site": E.ABSTAINED_CORRECT,
                                                 "histology": E.ABSTAINED_CORRECT}
    ps = rep.per_field["primary_site"]
    assert ps["exact_match_rate"] == 1.0 and ps["exact_match_den"] == 1
    assert ps["abstention_rate"] == 0.5 and ps["correct_abstention_rate"] == 1.0
    assert ps["gate_validated_rate"] == 1.0


def test_a_wrong_code_is_a_mismatch_and_a_missed_abstention_is_not_a_match():
    rep = score([run(patient="SYN0001", value={"primary_site": "C349", "histology": "8046"}),
                 run(patient="SYN0002", value={"primary_site": "C341", "histology": "8070"})])
    assert outcomes(rep, f"SYN0001__{SPEC}")["primary_site"] == E.MISMATCH
    assert outcomes(rep, f"SYN0002__{SPEC}")["primary_site"] == E.ANSWERED_OVER_ABSTAIN
    assert rep.per_field["primary_site"]["correct_abstention_rate"] == 0.0


def test_abstaining_where_the_key_names_a_value_is_a_miss_not_an_abstention_win():
    rep = score([run(patient="SYN0001", status="EVIDENCE_INSUFFICIENT", gate=False)])
    assert outcomes(rep, f"SYN0001__{SPEC}")["primary_site"] == E.ABSTAINED_MISSED
    assert rep.per_field["primary_site"]["exact_match_rate"] == 0.0


def test_an_unkeyed_run_is_counted_as_unkeyed_and_never_as_wrong():
    rep = score([run(patient="SYN9999", value={"primary_site": "C341"})])
    assert rep.totals["n_unkeyed"] == 1
    assert outcomes(rep, f"SYN9999__{SPEC}")["primary_site"] == E.NO_KEY
    assert rep.per_field["primary_site"]["exact_match_rate"] is None      # not 0.0


def test_unmeasured_cost_and_tokens_report_as_none_beside_their_counters():
    m = manifest()
    m.pop("usage"), m.pop("cost_usd")
    rep = score([E.RunRecord(m)])
    assert rep.totals["tokens_mean"] is None and rep.totals["cost_usd_total"] is None
    assert rep.totals["n_tokens_unknown"] == 1 and rep.totals["n_cost_unknown"] == 1


def test_per_instance_rows_carry_cost_turns_and_findings():
    rep = score(CORRECT, detector_config=CFG)
    assert len(rep.per_instance) == 2
    r = rep.per_instance[0]
    assert r.total_tokens == 50_000 and r.turns == 12 and r.cost_usd == 0.42
    assert rep.totals["turns_mean"] == 12.0 and rep.totals["cost_usd_total"] == 0.84
    assert isinstance(r.findings, list)


def test_detector_findings_ride_along_and_are_counted_by_severity():
    bad = run(patient="SYN0001", value={"primary_site": "C341", "histology": "8070"},
              n_read=0, searched=("t",))
    rep = score([bad], detector_config=CFG)
    assert rep.totals["findings_by_severity"][E.CRITICAL] >= 2
    assert "zero_document_read" in {f["detector"] for f in rep.per_instance[0].findings}


def test_the_table_renders_every_field_and_names_what_was_unmeasured():
    text = score(CORRECT).table()
    assert "primary_site" in text and "histology" in text
    assert "unmeasured" in text and "100.0%" in text


def test_scoring_requires_the_caller_to_declare_the_fields():
    with pytest.raises(ValueError, match="fields is required"):
        E.score(CORRECT, KEY, fields=[], key=KEY_A)


def test_a_report_never_carries_a_real_person_id():
    rep = score([E.RunRecord(manifest(patient=OTHER_PATIENT))])
    assert OTHER_PATIENT not in json.dumps(rep.to_dict())


# ------------------------------------------------------------------- baselines and deltas
def test_a_baseline_round_trips_and_is_keyed_by_commit_spec_model_and_date(tmp_path):
    p = E.save_baseline(score(CORRECT), tmp_path / "base" / "b.json")
    loaded = E.load_baseline(p)
    assert loaded["baseline_key_str"] == "c1b5914|abc123|syn-model|2026-07-27"
    assert loaded["baseline_key"]["model"] == "syn-model"
    assert len(loaded["per_instance"]) == 2


def test_guessing_over_an_abstention_is_a_regression_the_mean_cannot_see():
    """THE test this module exists for.

    Between the two runs the exact-match rate does not move by a thousandth: the chart that
    changed is the one whose key says abstain, so it was never in the exact-match
    denominator. An eval that reported only the headline would call this release neutral.
    It is not neutral — the agent started coding a site on a chart with no admissible
    evidence, which is the failure mode a task-completion judge actively rewards.
    """
    after = [CORRECT[0],
             run(patient="SYN0002", value={"primary_site": "C349", "histology": "8046"})]
    before_d, after_d = score(CORRECT).to_dict(), score(after).to_dict()
    assert (before_d["per_field"]["primary_site"]["exact_match_rate"]
            == after_d["per_field"]["primary_site"]["exact_match_rate"] == 1.0)

    d = E.compare(before_d, after_d)
    assert d["verdict"] == "REGRESSION"
    assert d["per_field"]["primary_site"]["delta"] == 0.0
    regressed = {(r["instance_id"], r["field"]) for r in d["regressions"]}
    assert (f"SYN0002__{SPEC}", "primary_site") in regressed
    assert d["regressions"][0]["before"] == E.ABSTAINED_CORRECT


def test_a_subgroup_collapse_survives_an_aggregate_improvement():
    key = {f"P{i}__{SPEC}": {"fields": {"primary_site": v}, "subgroups": [g]}
           for i, (v, g) in enumerate([("C341", "squamous"), ("C341", "squamous"),
                                       ("C349", "outside_hospital"),
                                       ("C349", "outside_hospital")])}

    def rep(codes, name):
        runs = [run(patient=f"P{i}", value={"primary_site": c}) for i, c in enumerate(codes)]
        return E.score(runs, key, fields=["primary_site"],
                       key=E.BaselineKey("c1b5914", "abc123", name, "2026-07-27")).to_dict()

    # squamous 0/2 -> 2/2, outside_hospital 2/2 -> 1/2: the mean rises, one subgroup falls.
    before = rep(["C000", "C000", "C349", "C349"], "old")
    after = rep(["C341", "C341", "C349", "C000"], "new")
    assert (after["per_field"]["primary_site"]["exact_match_rate"]
            > before["per_field"]["primary_site"]["exact_match_rate"])

    d = E.compare(before, after)
    assert d["verdict"] == "REGRESSION"
    assert [(s["subgroup"], s["before"], s["after"]) for s in d["subgroup_regressions"]] \
        == [("outside_hospital", 1.0, 0.5)]
    assert len(d["improvements"]) == 2


def test_a_clean_improvement_is_not_reported_as_a_regression():
    before = score([run(patient="SYN0001", value={"primary_site": "C349",
                                                  "histology": "8046"}), CORRECT[1]])
    d = E.compare(before.to_dict(), score(CORRECT).to_dict())
    assert d["verdict"] == "OK" and not d["regressions"]
    assert len(d["improvements"]) == 2


def test_comparing_across_a_changed_spec_is_allowed_but_always_announced():
    other = E.BaselineKey("deadbee", "zzz999", "syn-model", "2026-07-28")
    d = E.compare(score(CORRECT).to_dict(), score(CORRECT, key=other).to_dict())
    assert d["verdict"] == "OK"
    assert any("spec_hash" in s for s in d["key_differences"])
    assert any("commit" in s for s in d["key_differences"])


def test_the_harness_reads_a_manifest_and_its_trace_from_disk(tmp_path):
    mp = tmp_path / "run-syn.manifest.json"
    mp.write_text(json.dumps(manifest(n_read=0)), encoding="utf-8")
    (tmp_path / "run-syn.jsonl").write_text(
        json.dumps({"seq": 0, "kind": "llm", "content": "x"}) + "\n"
        + json.dumps({"seq": 1, "kind": "tool", "tool": "chart.read_document",
                      "args": {"note_id": "Path-Report_2020-01-01"}}) + "\n",
        encoding="utf-8")
    rec = E.RunRecord.from_manifest(mp)
    assert rec.patient_id == "SYN0001" and rec.n_documents_read == 1
    assert rec.turns == 12 and rec.source.endswith("manifest.json")
