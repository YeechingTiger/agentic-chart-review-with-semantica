"""The extract -> concord -> explain chain, driven through the real CLI.

No provider is called anywhere in this file, and nothing is mocked past the model boundary.
`extract` runs the real `ChartReviewAgent` over the real synthetic corpus against a scripted
`LLMClient`, so the graph, the toolbox, the coverage ledger and the gate are all genuinely
exercised — only the completions are fixed. `concord` and `explain` then run twice: once on
an artifact that agent produced, and once on a hand-built one covering the outcomes the
synthetic corpus cannot reach. That every stage after L2 is reproducible from a file, with
no agent, is the property the artifacts exist for.

What is under test is the plumbing between layers, because that is where the meaning gets
lost. A variable that quietly fails to arrive, a coverage ledger that quietly fails to
travel, or an outcome quietly folded into a denominator all produce a number that looks
exactly like a real one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from acr.commands.cli import (
    CONCORD_SCHEMA,
    EXPLAIN_SCHEMA,
    EXTRACT_SCHEMA,
    _variable_records,
    app,
    read_cohort,
)
from acr.contract.concordance import variables_from_answer
from acr.core import site
from acr.core.llm import LLMClient, LLMConfig, LLMResponse

ROOT = Path(__file__).resolve().parents[1]
GUIDELINE = ROOT / "assets" / "guidelines" / "nccn_nsclc_subset.yaml"
runner = CliRunner()

SHB = "STORE.400_522_523.site_histology_behavior"
STG = "STORE.700_880.stage"
SHB_F = ["primary_site", "histology", "behavior"]
STG_F = ["clinical_t", "clinical_n", "clinical_m", "clinical_stage_group", "pathologic_t",
         "pathologic_n", "pathologic_m", "pathologic_stage_group", "summary_stage"]

#: A gate-validated stratified ledger, in the shape `CoverageLedger.to_dict()` emits. The
#: elusion bound is the one `tests/test_stage_spec.py` measured on SYN0001 (25 drawn, zero
#: hits -> 0.113), which is under the 0.12 cap `explain.py` defaults to.
LEDGER = {
    "mode": "stratified_exclusion", "sample_seed": 7, "listed_documents": True,
    "universe": {"n_documents": 321, "n_types": 44}, "n_read": 40,
    "searched_terms": ["stage", "pathologic stage", "pleural"],
    "strata": [
        {"name": "can_establish", "N": 23, "reviewed": 23, "complete": True, "elusion_upper": 0.0},
        {"name": "cannot_establish", "N": 173, "sampled": 25, "sample_hits": 0,
         "elusion_upper": 0.113},
        {"name": "may_mention", "N": 125, "misses_sampled": 25, "miss_sample_hits": 0,
         "elusion_upper": 0.113},
    ],
    "suspected_recognition_failures": [],
}


def _found(value: dict) -> dict:
    return {"status": "FOUND", "value": value, "proof_basis": "WITNESS", "witness_count": 1,
            "reasoning": "cited", "evidence": [
                {"note_id": "Surgical-Pathology-Document_2019-11-07",
                 "doc_type": "Surgical-Pathology-Document", "date": "2019-11-07",
                 "start": 10, "end": 40, "quote": "q", "supports": "x", "stance": "supports"}]}


def _gated_negative(fields: list[str]) -> dict:
    return {"status": "EVIDENCE_INSUFFICIENT", "value": {f: None for f in fields},
            "reasoning": "proved absent", "negative_basis": "GATE_VALIDATED",
            "proof_obligation": {"verdict": "PASS", "checks": {}, "missing": []},
            "coverage_attested": LEDGER, "evidence": []}


def _gave_up(fields: list[str]) -> dict:
    return {"status": "EVIDENCE_INSUFFICIENT", "value": {f: None for f in fields},
            "reasoning": "ran out", "negative_basis": "BUDGET_EXHAUSTED", "route_to_human": True,
            "coverage_note": "no coverage claim is made", "evidence": []}


def _stage(clinical=None, pathologic=None) -> dict:
    v = {f: None for f in STG_F}
    v["clinical_stage_group"] = clinical
    v["pathologic_stage_group"] = pathologic
    return _found(v)


def _rows(answer: dict, spec_id: str, fields: list[str]) -> dict:
    return {n: {"status": vv.status, "value": vv.value, "negative_basis": vv.negative_basis,
                "source": vv.source, "spec_id": spec_id, "output_field": n,
                "gate_validated": True, "proof_basis": answer.get("proof_basis")}
            for n, vv in variables_from_answer(answer, fields, source=spec_id).items()}


def _patient(pid: str, shb: dict, stg: dict) -> dict:
    return {"patient_id": pid, "runs": [], "errors": [],
            "answers": {SHB: shb, STG: stg},
            "variables": _rows(shb, SHB, SHB_F) | _rows(stg, STG, STG_F)}


NSCLC = {"primary_site": "C341", "histology": "8140", "behavior": "3"}
_REG = {"status": "FOUND", "source": "registry_feed"}


def _reg(**kw) -> dict:
    return {k: _REG | {"value": v} for k, v in kw.items()}


@pytest.fixture
def pipeline(tmp_path: Path) -> Path:
    """Six patients chosen so every L4 outcome and both L5 verdict families appear."""
    doc = {
        "schema": EXTRACT_SCHEMA, "created_utc": "2026-07-26T21:00:00+00:00",
        "code_sha": "test", "corpus": str(site.corpus_root()), "specs_dir": str(site.specs_root()),
        "cohort": "cohort.csv", "model": "test", "sample_seed": 7, "n_failed_runs": 0,
        "resolution": {"requested": ["primary_site", "histology", "behavior", "stage"],
                       "variables": [], "spec_ids": [SHB, STG]},
        "specs": {SHB: {"spec_id": SHB}, STG: {"spec_id": STG}},
        "patients": [
            _patient("P1", _found(NSCLC), _stage(pathologic="IIB")),    # treated on time
            _patient("P2", _found(NSCLC), _stage(pathologic="IIIA")),   # therapy absent
            _patient("P3", _found(NSCLC), _gated_negative(STG_F)),      # stage proved absent
            _patient("P4", _found(NSCLC), _stage(pathologic="99")),     # registry sentinel
            _patient("P5", _found(NSCLC), _stage(clinical="IA2")),      # a different rec
            _patient("P6", _found(NSCLC), _gave_up(STG_F)),             # the agent gave up
        ],
    }
    false = _REG | {"value": "false"}
    none = _REG | {"value": None}
    extras = {
        "P1": _reg(class_of_case="10", surgical_resection_extent="lobectomy",
                   surgical_margins="negative",
                   adjuvant_systemic_therapy_class="platinum_doublet_chemotherapy",
                   date_of_definitive_surgery="20200310",
                   date_of_first_adjuvant_systemic_therapy="20200502"),
        "P2": _reg(class_of_case="10", surgical_resection_extent="lobectomy",
                   surgical_margins="negative",
                   ecog_performance_status_after_surgery="1",
                   date_of_definitive_surgery="20200310") | {
            "adjuvant_systemic_therapy_class": none,
            "date_of_first_adjuvant_systemic_therapy": none, "date_of_death": none,
            "patient_refused_adjuvant_systemic_therapy": false,
            "contraindication_to_systemic_therapy": false, "clinical_trial_enrollment": false},
        **{p: _reg(class_of_case="10") for p in ("P3", "P4", "P5", "P6")},
    }
    (tmp_path / "extract.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "extra.json").write_text(json.dumps(extras), encoding="utf-8")
    return tmp_path


def _concord(d: Path, *extra_args: str):
    return runner.invoke(app, ["concord", "--guideline", str(GUIDELINE),
                               "--input", str(d / "extract.json"),
                               "--extra-variables", str(d / "extra.json"), *extra_args])


def _outcomes(d: Path) -> dict[tuple[str, str], str]:
    doc = json.loads((d / "concord.json").read_text(encoding="utf-8"))
    return {(p["patient_id"], r["recommendation_id"].split("-", 1)[1]): r["outcome"]
            for p in doc["patients"] for r in p["results"]}


# ------------------------------------------------------------------ cohort input
def test_cohort_csv_with_a_header(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("patient_id,dx_year\nSYN0001,2019\nSYN0002,2020\n", encoding="utf-8")
    assert read_cohort(p) == ["SYN0001", "SYN0002"]


def test_cohort_bare_list_and_comments(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# cohort\nSYN0001\nSYN0002\n\n", encoding="utf-8")
    assert read_cohort(p) == ["SYN0001", "SYN0002"]


def test_cohort_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"patients": [{"patient_id": "SYN0001"}, "SYN0002"]}),
                 encoding="utf-8")
    assert read_cohort(p) == ["SYN0001", "SYN0002"]


def test_duplicate_ids_are_collapsed_and_reported(tmp_path, capsys):
    """A repeated id moves every denominator computed from the cohort."""
    p = tmp_path / "c.csv"
    p.write_text("patient_id\nSYN0001\nSYN0002\nSYN0001\n", encoding="utf-8")
    assert read_cohort(p) == ["SYN0001", "SYN0002"]
    assert "duplicate" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["", "# only a comment\n"])
def test_an_empty_cohort_is_an_error(tmp_path, content):
    p = tmp_path / "c.csv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        read_cohort(p)


# ------------------------------------------------------------------ flattening
def test_a_populated_field_survives_an_abstaining_answer():
    """Measured on `aprime_SYN0002`: the answer said EVIDENCE_INSUFFICIENT overall and still
    coded `primary_site`, because the spec permits reporting the site when it is documented.
    The status is per answer and the value is per field, and the two need not agree.

    `extract` delegates this to `variables_from_answer` rather than re-deriving it, so that
    L2's flattening and L4's are the same rule and cannot drift apart.
    """
    answer = {"status": "EVIDENCE_INSUFFICIENT", "negative_basis": "GATE_VALIDATED",
              "value": {"primary_site": "C186", "histology": None}}
    got = _variable_records(answer, SHB, SHB_F, gate_validated=True)

    assert got["primary_site"]["status"] == "FOUND"          # an assertion, kept
    assert got["histology"]["status"] == "EVIDENCE_INSUFFICIENT"   # explicit null
    assert got["behavior"]["status"] == "EVIDENCE_INSUFFICIENT"    # a silence, never FOUND
    assert got["behavior"]["value"] is None
    assert {r["source"] for r in got.values()} == {SHB}


def test_flattened_rows_are_what_the_rule_engine_accepts():
    """A row without a status is rejected by `_coerce` — extract must not emit one."""
    from acr.contract.concordance import Guideline, _bind
    rows = _variable_records(_found(NSCLC), SHB, SHB_F, gate_validated=True)
    bound = _bind(rows, Guideline("g"))
    assert bound["primary_site"].value == "C341"
    assert bound["primary_site"].resolution == "KNOWN"


# ------------------------------------------------------------------ extract (L0-L3)
# ---------------------------------------------------------------------------------------
# EVERY `extract` INVOCATION BELOW NAMES `--runtime langgraph`, and that is the point of the
# flag rather than an accident of it. These tests pin plumbing that belongs to the
# hand-written loop — `cli_common.budget`, the `run_budget` manifest block, the single plan
# block in the transcript — and they were written when it was the only runtime, so the
# default carried the choice implicitly. When the default moved to `hooks` (measured better
# on ten real charts) all eight failed, which read as a regression and was really a test
# saying "whatever runs by default" about something runtime-specific.
#
# The hooks runtime's own end-to-end coverage is tests/test_hooks_runtime_cli.py.
# ---------------------------------------------------------------------------------------

def test_extract_dry_run_plans_the_work_without_a_model(tmp_path):
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\nSYN0002\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "primary_site,histology,stage", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "2 patient(s) x 2 spec(s) = 4 agent run(s)" in r.output
    assert "no model was called" in r.output


def test_many_variables_of_one_spec_are_one_run(tmp_path):
    """The unit of work is the spec. Three fields of one spec is one pass over the chart."""
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "primary_site,histology,behavior", "--dry-run"])
    assert "1 patient(s) x 1 spec(s) = 1 agent run(s)" in r.output


def test_an_unknown_variable_stops_the_command(tmp_path):
    """Never a shorter extract than was asked for — and the vocabulary comes back with it."""
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "histology,tx1_date", "--dry-run"])
    assert r.exit_code == 2
    assert "tx1_date" in r.output and "known variables" in r.output


def test_an_outside_notes_variable_is_flagged_before_the_cohort_is_spent(tmp_path):
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "class_of_case", "--dry-run"])
    assert r.exit_code == 0
    assert "WRONG_DATA_SOURCE" in r.output


def test_a_missing_cohort_file_is_a_usage_error(tmp_path):
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "nope.csv"),
                            "--variables", "histology", "--dry-run"])
    assert r.exit_code != 0


# ------------------------------------- extract, driven all the way through the agent
class ScriptedLLM(LLMClient):
    """A model that follows a fixed tool script, so `extract` can be run without a provider.

    It reads the transcript rather than a step counter, the way the real one does: the
    evidence span it cites is taken out of the `search_notes` result that came back two
    messages earlier. A stub that invented a note_id would be rejected by the toolbox and the
    run would never reach the gate — which is the behaviour under test, so the stub has to
    earn its citation the same way.
    """

    def __init__(self, value: dict):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.value = value
        #: Every message list this stub was handed, in order. What is IN the prompt is as
        #: much a property under test as what comes back out of it.
        self.seen: list[list[dict]] = []

    def _reply(self, obj: dict, calls: list[dict] | None = None):
        self.calls += 1
        self.prompt_tokens += 100
        self.completion_tokens += 20
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=100, completion_tokens=20)

    @staticmethod
    def _last_tool(messages: list[dict], name: str) -> dict:
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("name") == name:
                return json.loads(m["content"])
        return {}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.seen.append([dict(m) for m in messages])
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if tools is None:
            if "SUFFICIENT|CONTINUE|STUCK" in last:      # reflect: REPLAN is no longer a verdict a model may pick
                return self._reply({"verdict": "CONTINUE", "reason": "still gathering"})
            if "FOUND|EVIDENCE_INSUFFICIENT|SPEC_INSUFFICIENT" in last:
                return self._reply({"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                    "reasoning": "the scripted run never finalised"})
            return self._reply({"plan": [{"id": "1", "goal": "find the pathology report",
                                          "rationale": "it is the establishing document"}]})

        def call(n, name, args):
            return {"id": f"c{n}", "name": name, "arguments": args}

        hits = self._last_tool(messages, "search_notes").get("hits") or []
        if not hits:
            return self._reply({}, [call(0, "list_documents", {}),
                                    call(1, "search_notes", {"query": "carcinoma"})])
        h = hits[0]
        return self._reply({}, [
            call(2, "record_evidence", {"note_id": h["note_id"], "start": h["start"],
                                        "end": h["end"], "supports": "histology"}),
            call(3, "submit_answer", {"status": "FOUND", "value": self.value,
                                      "reasoning": "coded from the cited span"})])


@pytest.fixture
def scripted(monkeypatch):
    """The same script, delivered through the seam the runtime actually uses.

    It patched `cli_common.llm_client`, which is the litellm seam; `extract` runs the library
    graph and reaches the provider through `cli_common.chat_model`. The script itself is
    unchanged — `LitellmScriptAdapter` translates it — because what these tests assert is the
    L0-L3 plumbing, not which client shape the provider happens to want.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import LitellmScriptAdapter

    llm = ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
    monkeypatch.setattr("acr.core.cli_common.chat_model",
                        lambda *a, **k: LitellmScriptAdapter(inner=llm))
    return llm


def test_extract_runs_the_agent_and_writes_a_usable_artifact(tmp_path, scripted):
    """The whole L0-L3 leg: resolve, run the gated agent per patient x spec, flatten, write."""
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\nSYN0002\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "primary_site,histology,behavior",
                            "--out", str(tmp_path / "runs"), "--seed", "7"])
    assert r.exit_code == 0, r.output
    (path,) = list((tmp_path / "runs").glob("extract__*/extract.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))

    assert doc["schema"] == EXTRACT_SCHEMA and doc["n_failed_runs"] == 0
    assert [p["patient_id"] for p in doc["patients"]] == ["SYN0001", "SYN0002"]
    p1 = doc["patients"][0]
    assert p1["variables"]["primary_site"]["value"] == "C341"
    assert p1["variables"]["primary_site"]["gate_validated"] is True
    assert p1["runs"][0]["proof_basis"] == "WITNESS"
    # Degradation first, before any other number: a non-zero counter means a node silently
    # fell back and the behaviour this test claims to exercise was not exercised.
    # This runtime's own counters, not the old loop's plan/reflect fallbacks. Non-zero means a
    # node did less than it claims and no conclusion may be drawn from the run above it.
    assert set(p1["runs"][0]["degradation"].values()) == {0}
    assert p1["answers"][SHB]["evidence"], "the citation must survive into the artifact"


def test_the_run_is_traceable_back_to_the_code_and_spec_that_made_it(tmp_path, scripted):
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                        "--variables", "histology", "--out", str(tmp_path / "runs")])
    (path,) = list((tmp_path / "runs").glob("extract__*/extract.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["specs"][SHB]["spec_hash"]) == 16
    assert doc["code_sha"] and doc["sample_seed"] is None
    assert (path.parent / f"SYN0001__{SHB}.jsonl").exists(), "the trace lands beside the artifact"


def test_a_real_extract_feeds_concord_and_explain(tmp_path, scripted):
    """End to end on an artifact the agent actually produced, not a hand-built one."""
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                        "--variables", "primary_site,histology,behavior",
                        "--out", str(tmp_path / "runs")])
    (ex,) = list((tmp_path / "runs").glob("extract__*/extract.json"))
    r = runner.invoke(app, ["concord", "--guideline", str(GUIDELINE), "--input", str(ex)])
    assert r.exit_code == 0, r.output
    doc = json.loads((ex.parent / "concord.json").read_text(encoding="utf-8"))
    # Site, histology and behaviour alone cannot settle any of the three recommendations —
    # stage, class of case and the treatment variables are all missing. NOT_ASSESSABLE is
    # the correct answer, and it names what is missing.
    assert {r["outcome"] for p in doc["patients"] for r in p["results"]} == {"NOT_ASSESSABLE"}
    assert "class_of_case" in doc["summary"]["blocking_inputs"]
    assert doc["summary"]["denominator"] == 0 and doc["summary"]["concordance_rate"] is None

    r = runner.invoke(app, ["explain", "--input", str(ex.parent / "concord.json")])
    assert r.exit_code == 0, r.output
    assert json.loads((ex.parent / "explain.json").read_text(encoding="utf-8"))["cases"] == []


def test_one_failing_patient_does_not_shrink_the_cohort_silently(tmp_path, scripted):
    """A patient missing from the corpus is a cohort/corpus mismatch. The row survives
    carrying the error and the command exits non-zero; dropping it would move a denominator
    with nothing recording that it moved."""
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\nNO_SUCH_PATIENT\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "histology", "--out", str(tmp_path / "runs")])
    assert r.exit_code == 1
    (path,) = list((tmp_path / "runs").glob("extract__*/extract.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert [p["patient_id"] for p in doc["patients"]] == ["SYN0001", "NO_SUCH_PATIENT"]
    assert doc["patients"][1]["errors"] and doc["n_failed_runs"] == 1
    assert doc["patients"][1]["variables"] == {}


# ------------------------------------------------------------------ concord (L4)
def test_concord_scores_the_cohort(pipeline):
    r = _concord(pipeline)
    assert r.exit_code == 0, r.output
    got = _outcomes(pipeline)
    assert got[("P1", "ADJ-SYSTEMIC-II-IIIA")] == "CONCORDANT"
    assert got[("P2", "ADJ-SYSTEMIC-II-IIIA")] == "NON_CONCORDANT"
    assert got[("P5", "STAGE-I-DEFINITIVE-LOCAL-THERAPY")] == "NOT_ASSESSABLE"


def test_an_unproved_variable_is_not_assessable_not_a_care_gap(pipeline):
    """P3's stage came back EVIDENCE_INSUFFICIENT and P6's agent gave up. Neither patient
    can be scored, and folding either into the denominator is how a rate gets inflated."""
    _concord(pipeline)
    got = _outcomes(pipeline)
    for pid in ("P3", "P6"):
        assert got[(pid, "ADJ-SYSTEMIC-II-IIIA")] == "NOT_ASSESSABLE"
    doc = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    blocked = {p["patient_id"]: p["results"][0]["blocking_inputs"] for p in doc["patients"]}
    assert "pathologic_stage_group" in blocked["P3"]


def test_a_registry_sentinel_is_unknown_not_out_of_population(pipeline):
    """P4's stage is `99` — a present, well-formed value meaning nobody established it.
    Failing set membership like an ordinary non-member would report the patient as
    determinately outside the population."""
    _concord(pipeline)
    assert _outcomes(pipeline)[("P4", "ADJ-SYSTEMIC-II-IIIA")] == "NOT_ASSESSABLE"
    doc = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    p4 = next(p for p in doc["patients"] if p["patient_id"] == "P4")
    use = next(u for u in p4["results"][0]["inputs_used"]
               if u["variable"] == "pathologic_stage_group")
    assert use["status"] == "FOUND" and use["unknown_sentinel"] is True
    assert use["resolution"] == "UNKNOWN"


def test_the_denominator_states_its_exclusions(pipeline):
    _concord(pipeline)
    s = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))["summary"]
    assert s["denominator"] == s["counts"]["CONCORDANT"] + s["counts"]["NON_CONCORDANT"]
    assert sum(s["denominator_excludes"].values()) + s["denominator"] == s["n_recommendations"]
    assert s["concordance_rate"] == pytest.approx(0.5)


def test_no_scorable_case_reports_no_rate_rather_than_zero(pipeline):
    r = _concord(pipeline, "--recommendations", "NSCLC-BIOMARKER-BEFORE-FIRST-LINE")
    assert r.exit_code == 0
    s = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))["summary"]
    assert s["denominator"] == 0 and s["concordance_rate"] is None
    assert "nothing scorable" in r.output


def test_a_variable_in_both_sources_is_a_conflict_not_a_merge(pipeline):
    """Two sources for one variable is the two-ledger failure; picking one silently would
    make the rate depend on dict order."""
    extras = json.loads((pipeline / "extra.json").read_text(encoding="utf-8"))
    extras["P1"]["histology"] = {"status": "FOUND", "value": "8070", "source": "registry_feed"}
    (pipeline / "extra.json").write_text(json.dumps(extras), encoding="utf-8")
    r = _concord(pipeline)
    assert r.exit_code == 2
    assert "histology" in r.output and "--prefer" in r.output

    r = _concord(pipeline, "--prefer", "extract")
    assert r.exit_code == 0
    doc = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    p1 = next(p for p in doc["patients"] if p["patient_id"] == "P1")
    use = next(u for u in p1["results"][0]["inputs_used"] if u["variable"] == "histology")
    assert use["value"] == "8140" and use["source"] == SHB


def test_concord_refuses_an_artifact_of_the_wrong_schema(pipeline):
    r = runner.invoke(app, ["concord", "--guideline", str(GUIDELINE),
                            "--input", str(pipeline / "extra.json")])
    assert r.exit_code != 0
    assert EXTRACT_SCHEMA in r.output


def test_concord_records_the_identity_of_everything_it_used(pipeline):
    _concord(pipeline)
    doc = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    assert doc["schema"] == CONCORD_SCHEMA
    assert doc["engine"] == "acr.contract.concordance/deterministic"
    assert len(doc["guideline"]["guideline_hash"]) == 16
    assert doc["extract_input"].endswith("extract.json")
    assert doc["guideline_binding_warnings"] == []


def test_the_rule_and_exception_halves_are_recorded_separately(pipeline):
    """L5 holds an exception variable to the exception standard and a driving variable to
    the coverage standard, so the split has to survive into the artifact — including the
    overlap, since `date_of_definitive_surgery` is read by both halves."""
    _concord(pipeline)
    doc = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    rec = doc["recommendations"]["NSCLC-ADJ-SYSTEMIC-II-IIIA"]
    assert "adjuvant_systemic_therapy_class" in rec["rule_inputs"]
    assert "patient_refused_adjuvant_systemic_therapy" in rec["exception_inputs"]
    assert "date_of_definitive_surgery" in rec["rule_inputs"]
    assert "date_of_definitive_surgery" in rec["exception_inputs"]


# ------------------------------------------------------------------ explain (L5)
def test_explain_cannot_distinguish_a_care_gap_from_a_records_gap(pipeline):
    """P2's treatment variables came from an external feed with no coverage proof. Nothing
    separates 'not done' from 'not documented', and the command must say so."""
    _concord(pipeline)
    r = runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json")])
    assert r.exit_code == 0, r.output
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    assert doc["schema"] == EXPLAIN_SCHEMA
    (case,) = doc["cases"]
    assert case["case_id"] == "P2"
    assert case["scaffold"]["verdict"] == "CANNOT_DISTINGUISH"
    forbidden = " ".join(case["scaffold"]["packet"]["forbidden"])
    assert "Do NOT choose between A" in forbidden


def test_explain_only_runs_on_non_concordant_by_default(pipeline):
    _concord(pipeline)
    runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json")])
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    assert {c["outcome"] for c in doc["cases"]} == {"NON_CONCORDANT"}


def test_another_outcome_is_reported_as_unexplainable_not_scaffolded(pipeline):
    """NOT_ASSESSABLE is an outcome in its own right. Scaffolding one would fold a case
    into a rate it cannot be scored in."""
    _concord(pipeline)
    r = runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json"),
                            "--only", "NOT_ASSESSABLE"])
    assert r.exit_code == 0
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    assert doc["cases"] and all(c["scaffold"] is None for c in doc["cases"])
    assert all("only on NON_CONCORDANT" in c["not_explainable"] for c in doc["cases"])


def test_a_missing_extract_stops_explain_rather_than_inventing_a_verdict(pipeline, tmp_path):
    """Without the ledgers every case would report CANNOT_DISTINGUISH — a fabricated
    finding indistinguishable from a real one."""
    _concord(pipeline)
    moved = tmp_path / "elsewhere" / "concord.json"
    moved.parent.mkdir()
    doc = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    doc["extract_input"] = str(tmp_path / "gone" / "extract.json")
    moved.write_text(json.dumps(doc), encoding="utf-8")
    r = runner.invoke(app, ["explain", "--input", str(moved)])
    assert r.exit_code == 2
    assert "falsely report CANNOT_DISTINGUISH" in r.output


def test_a_gate_validated_absence_reaches_the_scaffold_as_a_proof(pipeline):
    """The one case where B can be settled: swap P2's absent treatment variable for a
    gate-validated extraction. The ledger has to survive extract.json -> concord.json ->
    explain.json for the standing to change at all."""
    doc = json.loads((pipeline / "extract.json").read_text(encoding="utf-8"))
    p2 = next(p for p in doc["patients"] if p["patient_id"] == "P2")
    p2["answers"]["FAKE.tx"] = _gated_negative(["adjuvant_systemic_therapy_class"])
    p2["variables"]["adjuvant_systemic_therapy_class"] = {
        "status": "EVIDENCE_INSUFFICIENT", "value": None, "negative_basis": "GATE_VALIDATED",
        "source": "FAKE.tx", "spec_id": "FAKE.tx",
        "output_field": "adjuvant_systemic_therapy_class", "gate_validated": True,
        "proof_basis": None}
    (pipeline / "extract.json").write_text(json.dumps(doc), encoding="utf-8")
    extras = json.loads((pipeline / "extra.json").read_text(encoding="utf-8"))
    extras["P2"].pop("adjuvant_systemic_therapy_class")
    (pipeline / "extra.json").write_text(json.dumps(extras), encoding="utf-8")

    _concord(pipeline)
    runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json")])
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    case = next(c for c in doc["cases"] if c["case_id"] == "P2")
    proof = next(p for p in case["scaffold"]["coverage_proofs"]
                 if p["variable"] == "adjuvant_systemic_therapy_class")
    assert proof["adequate"] is True
    assert proof["mode"] == "stratified_exclusion"
    assert proof["worst_elusion_upper"] == pytest.approx(0.113)


def test_registry_truth_is_opt_in_and_recorded(pipeline, tmp_path):
    """C stays OPEN with no truth file — the registry covers 20% of patients, and
    eliminating C on the rest would turn a coverage limit into a clean bill of health."""
    _concord(pipeline)
    runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json")])
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    case = doc["cases"][0]
    assert doc["registry_truth_supplied"] is False
    c = next(x for x in case["scaffold"]["causes"] if x["cause"] == "C_EXTRACTION_ERROR")
    assert c["standing"] == "OPEN"

    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"P2": {"histology": "8070"}}), encoding="utf-8")
    runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json"),
                        "--truth", str(truth)])
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    c = next(x for x in doc["cases"][0]["scaffold"]["causes"]
             if x["cause"] == "C_EXTRACTION_ERROR")
    assert c["standing"] == "SUPPORTED"
    assert any("8140" in b and "8070" in b for b in c["because"])
    assert any("does NOT prove the extraction wrong" in b for b in c["because"])
    assert json.loads((pipeline / "explain.json").read_text())["registry_truth_supplied"]


def test_every_scaffold_carries_the_no_ground_truth_notice(pipeline):
    """L5 has no ground truth and the disclaimer must not be shippable separately."""
    _concord(pipeline)
    runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json")])
    doc = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    for c in doc["cases"]:
        assert "human adjudication" in c["scaffold"]["validation_status"]
        assert "human adjudication" in c["scaffold"]["packet"]["validation_status"]


def test_the_three_artifacts_chain_by_recorded_path(pipeline):
    """Each stage names its input, so a number can be walked back to the run that made it."""
    _concord(pipeline)
    runner.invoke(app, ["explain", "--input", str(pipeline / "concord.json")])
    con = json.loads((pipeline / "concord.json").read_text(encoding="utf-8"))
    exp = json.loads((pipeline / "explain.json").read_text(encoding="utf-8"))
    assert Path(con["extract_input"]) == (pipeline / "extract.json").resolve()
    assert Path(exp["concord_input"]) == (pipeline / "concord.json").resolve()
    assert exp["extract_input"] == con["extract_input"]
    assert con["extract_created_utc"] == "2026-07-26T21:00:00+00:00"


def test_the_existing_commands_still_work():
    """The pipeline was added beside them, not over them.

    Read off the COMPOSED click group rather than `app.registered_commands`. Since the split
    into one module per command group, the parent Typer registers no commands of its own —
    every name below arrives through a nameless sub-app — and `registered_commands` would be
    empty while `acr patients` still worked perfectly. This asks the question the user asks:
    what can I actually type.
    """
    names = set(typer.main.get_command(app).commands)
    assert {"patients", "chart", "specs", "run", "batch", "consistency", "trace"} <= names
    assert {"extract", "concord", "explain"} <= names
    assert runner.invoke(app, ["specs"]).exit_code == 0


# --------------------------------------------------------------- the run budget on the CLI
# `Budget` has carried max_tokens and max_seconds since it was written and every construction
# site passed max_steps alone, so the two limits that actually bind on a real chart were
# unreachable defaults. That is not a hypothetical: on a 10-patient batch of real charts, 7 of
# 10 abstained with `max_tokens (400000) reached` at 8-16 steps against a 24-step cap. The
# manifest recorded no budget, so the abstention read as a fact about the charts rather than
# about a number nobody could set. Both halves are pinned here — the value reaches the run,
# and the run says what it was.

# ------------------------------------------------ the budget and the plan, on THIS runtime
# Four tests stood here and they pinned the hand-written loop's plumbing: `cli_common.budget`
# reaching a `Budget` dataclass, the `run_budget` manifest block, and exactly one
# `install_plan_block` copy in the transcript. All three are gone with that loop, and the
# replacements are not renames — they are different mechanisms:
#
#     max_steps + Budget(max_tokens, max_seconds) -> max_model_calls + ModelCallLimitMiddleware
#     run_budget block                            -> max_model_calls / recursion_limit
#     install_plan_block dedupe                   -> the plan lives in the system message, so
#                                                    there is nowhere for a copy to accumulate
#
# The last one is the point worth keeping: the old defect was ELEVEN plan copies in one real
# transcript, 41% of its prompt spend. It is now impossible by construction rather than deduped
# after the fact, and the test that proves that is the one below — it asserts on the mechanism
# (`ModelRequest.override`, never `messages`), because a count of copies cannot distinguish
# "deduped" from "cannot accumulate".


def test_the_plan_cannot_accumulate_in_the_transcript(tmp_path, scripted):
    import inspect

    import acr.review.agent as A
    hook = inspect.getsource(A.AuditMiddleware.wrap_model_call)
    assert "override(system_message=" in hook, (
        "the plan must ride the system message, which is replaced wholesale each call")
    assert "messages" not in hook.split("return handler")[0].replace("request.messages", ""), (
        "nothing may append the plan to the message list; that is what accumulated eleven times")


def test_the_call_budget_reaches_the_run_and_lands_in_the_manifest(tmp_path, scripted):
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    r = runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                            "--variables", "histology", "--out", str(tmp_path / "runs"),
                            "--max-steps", "9"])
    assert r.exit_code == 0, r.output
    (m,) = list((tmp_path / "runs").glob("extract__*/*.manifest.json"))
    d = json.loads(m.read_text(encoding="utf-8"))
    assert d["max_model_calls"] == 9, "the number the operator set must be the number that binds"
    # Derived from the graph, so a middleware added later moves it. `ModelCallLimitMiddleware`
    # must always bind first: it stops with a reason, the recursion limit stops with a trace.
    assert d["recursion_limit"] > d["max_model_calls"]








# -------------------------------------------------- the plan is state, not history
# `plan.render()` was APPENDED on every plan-node entry and again on every applied revision.
# Measured on a real 293-document chart: 6,310 chars, eleven copies, each re-sent on all
# forty-nine later calls — ~425,000 of that run's 1,030,179 prompt tokens, 41%, spent
# re-reading ten stale copies of a plan whose current version sat at the bottom of the same
# prompt. Uniqueness AND position are both load-bearing, so both are pinned.

def test_only_one_plan_block_survives_and_it_is_the_last_message():
    from acr.review.plan_expansion import install_plan_block, is_plan_block
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "PLAN:\nrevision zero"},
            {"role": "assistant", "content": "read something"},
            {"role": "user", "content": "PLAN (revision 1):\nrevision one"},
            {"role": "tool", "content": "a result"}]
    out = install_plan_block(msgs, "PLAN (revision 2):\nrevision two")
    plans = [m for m in out if is_plan_block(m)]
    assert len(plans) == 1, "a stale plan is not history, it is a second answer to one question"
    assert out[-1] is plans[0], "the plan governs the next call; buried, it is recalled not read"
    assert "revision two" in out[-1]["content"]


def test_nothing_but_a_plan_block_is_dropped():
    """The transcript either keeps the work or the run has amnesia.

    The bare prefix "PLAN" matches "PLANNING the next read", so a marker chosen carelessly
    deletes the agent's own words and presents as the model forgetting.
    """
    from acr.review.plan_expansion import install_plan_block
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "PLANNING my next move"},   # prose, not a block
            {"role": "assistant", "content": "PLAN: I will read the path report"},
            {"role": "tool", "content": "PLAN-shaped tool output"}]
    out = install_plan_block(msgs, "PLAN:\nthe real one")
    assert len(out) == len(msgs) + 1, "nothing here was a plan block; nothing may be dropped"
    assert any(m["content"] == "PLANNING my next move" for m in out)
    assert any(m["role"] == "assistant" and m["content"].startswith("PLAN:") for m in out)
    assert any(m["role"] == "tool" for m in out)


