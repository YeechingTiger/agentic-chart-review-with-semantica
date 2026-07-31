"""`acr label` and `acr refine`, driven through the real CLI.

These two groups are the develop plane, and until they were mounted the 2,084 lines behind
them had tests and no entry point — which is the same condition as not existing. So the
question every test here asks is the one nobody could ask before: what happens when a person
actually types this.

NO PROVIDER IS REACHED ANYWHERE IN THIS FILE. `label scan` is only ever exercised with
--dry-run, and the assertion that matters is that it stays that way: `azure_client` is
monkeypatched to a bomb in the dry-run tests, so a dry run that quietly built a client would
fail here rather than on somebody's invoice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acr.commands.cli import app
from acr.improvement import labelling as lab

ROOT = Path(__file__).resolve().parents[1]
SPEC = str(ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml")
#: A spec with no `evidence_rules`: it does not say what would establish its answer, so the
#: standing question has no definition and `Requirement.from_spec` refuses it.
UNLABELLABLE = str(ROOT / "specs" / "STORE.610.class_of_case.yaml")
CORPUS = str(ROOT / "corpus" / "patients")

runner = CliRunner()


@pytest.fixture
def no_client(monkeypatch):
    """Make building a labelling client an error, so a 'dry' run that isn't is a test failure."""
    def boom(*a, **k):
        raise AssertionError("azure_client() was built during a --dry-run")
    monkeypatch.setattr(lab, "azure_client", boom)
    return boom


# ------------------------------------------------------------------------------ acr label
def test_label_requirement_prints_the_spec_and_nothing_of_its_own():
    """Every line the reading model sees comes from the spec, and this is where you read it."""
    r = runner.invoke(app, ["label", "requirement", "--spec", SPEC])
    assert r.exit_code == 0, r.output
    assert "primary_site" in r.output and "histology" in r.output
    assert "WHAT THIS QUESTION COUNTS AS EVIDENCE" in r.output


def test_label_requirement_refuses_a_spec_that_defines_no_evidence():
    """The refusal is `Requirement.from_spec`'s and it has to reach the shell as an exit code.

    A spec with no evidence_rules cannot be labelled at all, because the three standing
    classes are DEFINED from that clause. Falling back on a default would be this CLI
    inventing a clinical rule.
    """
    r = runner.invoke(app, ["label", "requirement", "--spec", UNLABELLABLE])
    assert r.exit_code == 2, r.output
    assert "evidence_rules" in r.output


def test_label_scan_refuses_to_start_without_a_cost_ceiling():
    """--max-usd has no default, exactly as `ScanConfig.max_usd` has none."""
    r = runner.invoke(app, ["label", "scan", "--spec", SPEC,
                            "--max-terms-per-note", "8", "--min-term-chars", "4"])
    assert r.exit_code == 2
    assert "--max-usd" in r.output


@pytest.mark.parametrize("missing", ["--max-terms-per-note", "--min-term-chars"])
def test_label_scan_refuses_to_start_without_question_twos_bounds(missing, tmp_path):
    """`TermConfig` gives neither bound a default and neither does the flag.

    Both are part of the prompt hash and therefore part of the store key: a scan run under
    different bounds must not append to a file whose manifest names the first pair.
    """
    args = ["label", "scan", "--spec", SPEC, "--max-usd", "1",
            "--max-terms-per-note", "8", "--min-term-chars", "4"]
    args = [a for i, a in enumerate(args)
            if a != missing and (i == 0 or args[i - 1] != missing)]
    r = runner.invoke(app, args)
    assert r.exit_code == 2
    assert missing in r.output


def test_label_scan_dry_run_plans_and_prices_without_calling(tmp_path, no_client):
    """The plan comes from the same `scope`/`pending` the real run uses, not a second estimate."""
    out = tmp_path / "plan.json"
    r = runner.invoke(app, ["label", "scan", "--spec", SPEC, "--patients", "SYN0001",
                            "--corpus", CORPUS, "--max-usd", "5",
                            "--max-terms-per-note", "8", "--min-term-chars", "4",
                            "--labels-root", str(tmp_path / "labels"),
                            "--dry-run", "--out", str(out)])
    assert r.exit_code == 0, r.output
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["n_notes_in_scope"] > 0
    assert plan["n_pending"] == plan["n_notes_in_scope"]      # nothing labelled yet
    assert plan["spend_so_far_usd"] == 0
    assert plan["max_usd"] == 5.0
    # The input side is a FLOOR and the output side is named as unpriced. An estimate that
    # reads as the price is how a ceiling gets set below the true cost.
    assert plan["input_cost_floor_usd"] > 0
    assert "UNPRICED" in plan["output_cost"]
    assert plan["measured_mean_usd_per_label"] is None       # no priced label exists yet
    # Nothing was written into the label store beyond its manifest, and no note was read.
    assert not (Path(plan["store_dir"]) / "labels.jsonl").exists()


def test_label_scan_dry_run_and_progress_agree_on_the_store_key(tmp_path, no_client):
    """The dry run must plan against the file the real run would append to.

    The store key folds in the model, the prompt version, the spec and the term bounds. If the
    two commands derived it differently, the plan would be counting a different labelling.
    """
    common = ["--spec", SPEC, "--max-terms-per-note", "8", "--min-term-chars", "4",
              "--labels-root", str(tmp_path / "labels")]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    assert runner.invoke(app, ["label", "scan", *common, "--patients", "SYN0001",
                               "--corpus", CORPUS, "--max-usd", "1", "--dry-run",
                               "--out", str(a)]).exit_code == 0
    assert runner.invoke(app, ["label", "progress", *common, "--out", str(b)]).exit_code == 0
    plan, prog = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    assert plan["run_key"] == prog["run_key"]
    assert plan["requirement_hash"] == prog["requirement_hash"]


def test_label_progress_counts_a_store_and_leaks_no_note_text(tmp_path):
    """Counts, never rows. A label carries a person_id, a note date and a verbatim quote."""
    terms = lab.TermConfig(max_terms_per_note=8, min_term_chars=4)
    req = lab.Requirement.from_spec(__import__("acr.contract.spec", fromlist=["x"]).load_spec(SPEC))
    store = lab.LabelStore(str(tmp_path / "labels"), model=f"openai/{lab.DEPLOYMENT}",
                           requirement=req, terms=terms)
    store.append(lab.NoteLabel(
        patient_id="SYN0001", note_id="n1", doc_type="Pathology", date="2020-01-01",
        spec_id=req.spec_id, cost_usd=0.002,
        admissibility=lab.Admissibility(
            verdicts={"primary_site": "can_establish", "histology": "neither",
                      "behavior": "neither"},
            quote="left upper lobe adenocarcinoma", quote_verified=True),
        retrieval_terms=(lab.RetrievalTerm("adenocarcinoma", "names_the_answer"),),
        n_terms_proposed=3, n_terms_hallucinated=1))
    store.append(lab.NoteLabel(patient_id="SYN0001", note_id="n2", spec_id=req.spec_id,
                               error="TimeoutError: provider timed out"))

    out = tmp_path / "prog.json"
    r = runner.invoke(app, ["label", "progress", "--spec", SPEC,
                            "--max-terms-per-note", "8", "--min-term-chars", "4",
                            "--labels-root", str(tmp_path / "labels"), "--out", str(out)])
    assert r.exit_code == 0, r.output
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["n_labels"] == 2 and doc["n_ok"] == 1 and doc["n_errors"] == 1
    assert doc["spend_usd"] == pytest.approx(0.002)
    assert doc["standing_per_field"]["primary_site"]["can_establish"] == 1
    assert doc["standing_per_field"]["histology"]["neither"] == 1
    assert doc["n_terms_hallucinated"] == 1
    # The quote and the note text must not be anywhere in what this command emits.
    blob = json.dumps(doc) + r.output
    assert "left upper lobe adenocarcinoma" not in blob


def test_label_refuses_a_store_inside_the_repository():
    """A label carries PHI; `tests/test_no_phi_in_tree.py` exists because it got in once."""
    r = runner.invoke(app, ["label", "progress", "--spec", SPEC,
                            "--max-terms-per-note", "8", "--min-term-chars", "4",
                            "--labels-root", str(ROOT / "runs" / "labels")])
    assert r.exit_code == 2
    # Flattened first. Rich wraps the refusal to the terminal width and the wrap landed between
    # "inside the" and "repository", so the literal substring was absent while the message was
    # exactly right — a failure that comes and goes with the width of whoever ran it.
    # `tests/test_cli_signal.py::_flat` already carries this note for the same reason.
    flat = " ".join(r.output.split())
    assert "inside the repository" in flat


# ----------------------------------------------------------------------------- acr refine
def test_refine_parameters_prints_the_registry_and_marks_the_objective():
    """Exactly one row is inside the objective, and the table has to say which."""
    r = runner.invoke(app, ["refine", "parameters"])
    assert r.exit_code == 0, r.output
    for p in ("keyword_list", "document_type_policy", "skill", "spec_rules",
              "agent_system_prompt", "answer_check_rejection_messages"):
        assert p in r.output


def test_refine_parameters_json_carries_the_one_in_objective_row(tmp_path):
    out = tmp_path / "p.json"
    assert runner.invoke(app, ["refine", "parameters", "--out", str(out)]).exit_code == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["parameters"]
    in_obj = [p["id"] for p in rows if p["in_objective"]]
    assert in_obj == ["spec_rules"]


@pytest.fixture
def routing_inputs(tmp_path):
    """One FORM case, one CONTENT case and one retrieval failure, plus the spec they cite."""
    sentence = "Code the primary site from the pathology report when one exists."
    (tmp_path / "spec.txt").write_text(sentence, encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps([
        {"case_id": "C1", "spec_id": "S1", "field": "primary_site", "coded_value": "C349",
         "key_value": "C341", "establishing_evidence_surfaced": True,
         "answer_key_adjudication": "ADJUDICATED_KEY_CORRECT"},
        {"case_id": "C2", "spec_id": "S1", "field": "histology", "coded_value": "8046",
         "key_value": "8070", "establishing_evidence_surfaced": False,
         "answer_key_adjudication": "NOT_ADJUDICATED"},
        {"case_id": "C3", "spec_id": "S1", "field": "behavior", "coded_value": "2",
         "key_value": "3", "establishing_evidence_surfaced": True,
         "answer_key_adjudication": "ADJUDICATED_KEY_CORRECT"},
    ]), encoding="utf-8")
    (tmp_path / "verdicts.json").write_text(json.dumps({
        "C1": {"verdict": "SPEC_AMBIGUITY", "parameter_id": "spec_rules",
               "rationale": "the sentence reads two ways", "quoted_passage": sentence,
               "readings": ["laterality is part of the site", "laterality is separate"],
               "proposed_text": "Code the primary site, with laterality, from the report."},
        "C3": {"verdict": "SPEC_ERROR", "parameter_id": "spec_rules",
               "rationale": "the rule is substantively wrong", "quoted_passage": sentence},
    }), encoding="utf-8")
    return tmp_path


def test_refine_route_sends_form_to_a_proposal_and_content_to_a_question(routing_inputs):
    """THE ASYMMETRY, surviving the CLI. A CONTENT gradient cannot become an edit.

    There is no flag on this command that widens that, because there is none in `Proposal`:
    editing a rule because the data disagreed with it is moving the target to where the
    arrows landed.
    """
    out = routing_inputs / "routed.json"
    r = runner.invoke(app, ["refine", "route",
                            "--cases", str(routing_inputs / "cases.json"),
                            "--verdicts", str(routing_inputs / "verdicts.json"),
                            "--spec-text", f"S1={routing_inputs / 'spec.txt'}",
                            "--out", str(out)])
    assert r.exit_code == 0, r.output
    doc = json.loads(out.read_text(encoding="utf-8"))
    by_case = {x["case_id"]: x for x in doc["routings"]}
    assert by_case["C1"]["destination"] == "PROPOSAL" and by_case["C1"]["change_class"] == "FORM"
    assert by_case["C3"]["destination"] == "CLINICIAN_QUESTION"
    assert by_case["C3"]["change_class"] == "CONTENT"
    # Cut 1: the establishing evidence never surfaced, so the spec is irrelevant and the
    # reflector is not even consulted — C2 has no verdict in the file and still routes.
    assert by_case["C2"]["verdict"] == "RETRIEVAL_FAILURE"
    assert len(doc["batches"]) == 1 and len(doc["questions"]) == 1
    assert doc["questions"][0]["kind"] == "QUESTION"
    assert "proposed_text" not in doc["questions"][0]


def test_refine_route_reports_a_blast_radius_it_could_not_compute(routing_inputs):
    """An ABSENT number reads as zero. Where prose makes the count impossible it says so."""
    out = routing_inputs / "routed.json"
    runner.invoke(app, ["refine", "route", "--cases", str(routing_inputs / "cases.json"),
                        "--verdicts", str(routing_inputs / "verdicts.json"),
                        "--spec-text", f"S1={routing_inputs / 'spec.txt'}", "--out", str(out)])
    element = json.loads(out.read_text(encoding="utf-8"))["batches"][0]["elements"][0]
    assert element["blast_radius"]["computable"] is False
    assert "re-running" in json.dumps(element["blast_radius"])


def test_refine_route_needs_the_spec_text_the_mask_checks_against(routing_inputs):
    """Without the text the mask can check that a quote is present but not that it is true."""
    r = runner.invoke(app, ["refine", "route", "--cases", str(routing_inputs / "cases.json"),
                            "--verdicts", str(routing_inputs / "verdicts.json")])
    assert r.exit_code == 2
    assert "--spec-text" in r.output


def test_refine_route_refuses_a_case_id_shaped_like_a_real_person_id(routing_inputs):
    """`FailureCase` refuses it, and the refusal has to be an exit code and not a traceback."""
    shaped_like_a_real_id = "1168" + "0" * 12   # the shape, built at runtime, never written
    (routing_inputs / "phi.json").write_text(json.dumps([
        {"case_id": shaped_like_a_real_id, "spec_id": "S1", "field": "primary_site",
         "establishing_evidence_surfaced": True,
         "answer_key_adjudication": "ADJUDICATED_KEY_CORRECT"}]), encoding="utf-8")
    r = runner.invoke(app, ["refine", "route", "--cases", str(routing_inputs / "phi.json"),
                            "--verdicts", str(routing_inputs / "verdicts.json"),
                            "--spec-text", f"S1={routing_inputs / 'spec.txt'}"])
    assert r.exit_code == 2
    assert "Pseudonymise" in r.output


def test_refine_sample_size_requires_every_constant_including_the_zs():
    r = runner.invoke(app, ["refine", "sample-size", "--baseline-accuracy", "0.8",
                            "--detectable-regression-pp", "5"])
    assert r.exit_code == 2
    assert "--z-alpha" in r.output


def test_refine_sample_size_prices_both_arms():
    r = runner.invoke(app, ["refine", "sample-size", "--baseline-accuracy", "0.8",
                            "--detectable-regression-pp", "5", "--z-alpha", "1.96",
                            "--z-power", "0.84", "--cost-per-case-usd", "0.5"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.output.strip().splitlines()[-1])
    assert doc["per_arm_n"] > 100
    assert doc["estimated_cost_usd"] == pytest.approx(2 * doc["per_arm_n"] * 0.5)


def test_refine_read_results_refuses_a_batch_that_regressed_one_subgroup(tmp_path):
    """A mean of zero over a +100/-100 split is exactly what an average is built to hide."""
    (tmp_path / "res.json").write_text(json.dumps([
        {"case_id": "C1", "subgroup": "a", "control_correct": False, "candidate_correct": True},
        {"case_id": "C2", "subgroup": "b", "control_correct": True, "candidate_correct": False},
    ]), encoding="utf-8")
    r = runner.invoke(app, ["refine", "read-results", "--results", str(tmp_path / "res.json"),
                            "--max-tolerated-subgroup-drop-pp", "5"])
    assert r.exit_code == 1, r.output
    doc = json.loads(r.output.strip().splitlines()[-1])
    assert doc["accept"] is False
    assert doc["mean_delta_pp"] == pytest.approx(0.0)


def test_refine_read_results_requires_the_tolerated_drop(tmp_path):
    (tmp_path / "res.json").write_text("[]", encoding="utf-8")
    r = runner.invoke(app, ["refine", "read-results", "--results", str(tmp_path / "res.json")])
    assert r.exit_code == 2
    assert "--max-tolerated-subgroup-drop-pp" in r.output
