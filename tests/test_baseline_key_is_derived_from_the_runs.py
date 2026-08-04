"""The read side must take its identity from the runs, not from four strings an operator typed.

`BaselineKey(commit, spec_hash, model, date)` was assembled entirely from command-line options
([cli_eval.py](../src/acr/commands/cli_eval.py)) and reconciled against nothing. Every manifest
records `code_sha`, `spec_hash`, `model` and `experiment_config_hash` — the runtime's own statement
of which arm it was — and the scoring plane read none of them. Three consequences, all measured on
this tree's `runs/` (509 manifests spanning 7 spec hashes, 2 models, 11 code shas and 13 experiment
config hashes, 294 of them recording no config hash at all):

  1. `--model TOTALLY-WRONG-MODEL` was accepted in silence and written into the baseline.
  2. A run tree spanning two spec hashes scored as ONE clean baseline and reported +5.6 points.
     `tools/analyze_arms.py` refuses exactly this ("Refusing to compare: these arms ran on
     N different spec versions"); `eval score` averaged it.
  3. Two arms that genuinely differed — a whole skill card in the prompt — produced
     `key_differences: []` and `verdict: OK`, because the four key parts were identical and the
     part that moved (`prompt_assets`) was in no key.

`experiment_config_hash` exists precisely to answer (3) and had ZERO consumers: the runtime computed
it for every run and nothing ever read it. This file is the consumer.

## The refusal can only fire on a wrong claim

Reconciliation refuses when a DECLARED value contradicts what the runs recorded. It cannot refuse a
correct claim, and it says nothing about a field no manifest recorded — an absent value contradicts
nothing, and treating silence as disagreement would make every manifest written before 2026-08-02
unscoreable. That property is what separates this from the five deterministic content checks this
repo removed after they destroyed 58 correct values.

Heterogeneity itself is DATA, not a refusal: scoring a mixed tree is the only way to run the
detectors over everything in `runs/`, and that is a legitimate thing to do. The refusal lands one
step later, in `compare`, where a mixture cannot be one endpoint of a delta.
"""

from __future__ import annotations

import inspect

from acr.evaluation import evals as E

SPEC = "SYN.400.site_histology"
FIELDS = ["primary_site"]
KEY = {f"P{i}__{SPEC}": {"fields": {"primary_site": "C341"}} for i in range(4)}


def rec(patient="P0", *, spec_hash="abc123", model="syn-model", code_sha="c1b5914",
        ech="e0f5772713f1438f", run_id="run-20260727-081500-aaaaaa", value="C341") -> E.RunRecord:
    """A manifest carrying the identity block the runtime actually writes.

    Any of the four may be dropped with `None`, which is what a manifest written before that field
    existed looks like — 294 of this tree's 509 carry no `experiment_config_hash`.
    """
    m = {"patient_id": patient, "spec_id": SPEC,
         "answer": {"status": "FOUND", "value": {"primary_site": value}},
         "gate_validated": True}
    for k, v in (("spec_hash", spec_hash), ("model", model), ("code_sha", code_sha),
                 ("experiment_config_hash", ech), ("run_id", run_id)):
        if v is not None:
            m[k] = v
    return E.RunRecord(m, source=f"{patient}.manifest.json")


def score(runs, key=None, **kw):
    return E.score(runs, KEY, fields=FIELDS, key=key or E.derive_baseline_key(runs), **kw)


# ------------------------------------------------------- the manifest states its own identity

def test_a_run_exposes_the_identity_the_runtime_wrote():
    """Four accessors that did not exist. `evals` could not read `experiment_config_hash` at all."""
    r = rec()
    assert (r.code_sha, r.model, r.experiment_config_hash) == (
        "c1b5914", "syn-model", "e0f5772713f1438f")
    assert r.run_date == "2026-07-27", "the date is in the run id; nobody should have to type it"


def test_an_unrecorded_identity_field_reads_as_absent_never_as_a_value():
    r = rec(ech=None, code_sha=None, run_id=None)
    assert r.experiment_config_hash == "" and r.code_sha == "" and r.run_date == ""


def test_the_date_falls_back_to_the_batch_directory_when_the_run_id_carries_none():
    """`run_id` IS NOT RELIABLY A RUN ID. Measured over this tree's 509 manifests: 493 record the
    PATIENT id there (`SYN0007`) and only 16 carry `run-YYYYMMDD-HHMMSS-hex`. The batch launcher's
    directory name is where the timestamp actually is for the other 493, so it is the fallback.

    First-wins, never combined. Two sources summed for one fact is the `searched_terms` scar: the
    manifest half and the trace half were concatenated, every term counted twice, and
    `detect_degenerate_search` reported 10 findings where 5 occurrences existed.
    """
    r = E.RunRecord({"patient_id": "SYN0007", "run_id": "SYN0007"},
                    source="runs/floor/floor__20260802T112511Z__3a8f768-dirty/SYN0007.manifest.json")
    assert r.run_date == "2026-08-02"

    stamped = E.RunRecord({"run_id": "run-20260727-081500-aaaaaa"},
                          source="runs/x__20260802T112511Z__abc/run-20260727-081500-aaaaaa.manifest.json")
    assert stamped.run_date == "2026-07-27", "the runtime's own stamp outranks the directory name"


def test_a_path_with_no_timestamp_yields_no_date_rather_than_today():
    """"" not the clock. Stamping a two-week-old baseline with the day somebody read it is a
    fabricated provenance record, and every reader downstream would take it as measured."""
    assert E.RunRecord({"run_id": "SYN0007"}, source="runs/a15eval/SYN0007.manifest.json").run_date \
        == ""


# ------------------------------------------------------- derivation

def test_the_key_is_derived_from_the_runs_with_nothing_declared():
    key = E.derive_baseline_key([rec("P0"), rec("P1")])
    assert (key.commit, key.spec_hash, key.model, key.date) == (
        "c1b5914", "abc123", "syn-model", "2026-07-27")
    assert key.experiment_config_hash == "e0f5772713f1438f"
    assert key.basis == "derived"


def test_a_derived_key_over_a_mixed_tree_says_MIXED_rather_than_picking_one():
    """Picking the first, the most common, or the alphabetically-least would each produce a key
    that NAMES ONE ARM for a baseline that is a mixture — the shape that reported +5.6 points."""
    key = E.derive_baseline_key([rec("P0", spec_hash="abc123"), rec("P1", spec_hash="zzz999")])
    assert key.spec_hash == E.MIXED
    assert key.model == "syn-model", "only the field that actually moved is MIXED"


def test_a_date_range_is_a_range_not_a_day():
    key = E.derive_baseline_key([rec("P0", run_id="run-20260727-081500-aaaaaa"),
                                 rec("P1", run_id="run-20260803-115748-bbbbbb")])
    assert key.date == "2026-07-27..2026-08-03"


def test_deriving_from_no_runs_refuses_rather_than_returning_an_empty_key():
    """An empty key would score as a baseline whose identity is four empty strings, and two of
    those compare equal to each other."""
    try:
        E.derive_baseline_key([])
    except ValueError as e:
        assert "no runs" in str(e).lower()
    else:
        raise AssertionError("an empty run set must not yield a key")


# ------------------------------------------------------- reconciliation

def test_a_declared_model_the_runs_contradict_is_named():
    runs = [rec("P0"), rec("P1")]
    declared = E.BaselineKey(commit="c1b5914", spec_hash="abc123",
                             model="TOTALLY-WRONG-MODEL", date="2026-07-27")
    (complaint,) = E.reconcile_baseline_key(declared, runs)
    assert "model" in complaint
    assert "TOTALLY-WRONG-MODEL" in complaint and "syn-model" in complaint


def test_a_declared_key_that_matches_is_accepted_in_silence():
    runs = [rec("P0"), rec("P1")]
    assert E.reconcile_baseline_key(E.derive_baseline_key(runs), runs) == []


def test_a_field_no_manifest_recorded_contradicts_nothing():
    """THE PROPERTY THAT KEEPS THIS FROM BEING THE NEXT DELETED CHECK. Every fixture in
    `tests/test_evals.py` and every manifest written before `code_sha` reached the identity block
    records nothing here, and a check that read silence as disagreement would refuse them all."""
    runs = [rec("P0", code_sha=None, ech=None)]
    declared = E.BaselineKey(commit="whatever-the-operator-typed", spec_hash="abc123",
                             model="syn-model", date="2026-07-27")
    assert E.reconcile_baseline_key(declared, runs) == []


def test_a_declaration_is_not_contradicted_by_a_mixed_field_it_could_not_have_stated():
    """MIXED is reported by `key_basis`, and reported once. Also calling it a contradiction of the
    operator's string would file one defect as two, and the remedy for the two is different."""
    runs = [rec("P0", spec_hash="abc123"), rec("P1", spec_hash="zzz999")]
    declared = E.BaselineKey("c1b5914", "abc123", "syn-model", "2026-07-27")
    assert [c for c in E.reconcile_baseline_key(declared, runs) if "spec_hash" in c] == []


# ------------------------------------------------------- it reaches the recorded artifact

def test_the_report_records_what_the_runs_said_about_their_own_arm():
    d = score([rec("P0"), rec("P1")]).to_dict()
    basis = d["key_basis"]
    assert basis["n_runs"] == 2
    assert basis["fields"]["experiment_config_hash"]["value"] == "e0f5772713f1438f"
    assert basis["fields"]["experiment_config_hash"]["mixed"] is False
    assert basis["fields"]["experiment_config_hash"]["n_unrecorded"] == 0


def test_the_report_counts_the_runs_that_recorded_no_arm_hash():
    """294 of this tree's 509 manifests. A reader who cannot see that number cannot tell a baseline
    whose arm is established from one whose arm is a guess."""
    d = score([rec("P0"), rec("P1", ech=None)]).to_dict()
    f = d["key_basis"]["fields"]["experiment_config_hash"]
    assert f["n_unrecorded"] == 1 and f["value"] == "e0f5772713f1438f"


def test_a_contradiction_survives_into_the_baseline_file_even_if_the_cli_was_bypassed():
    """`analyze_arms.py`, `measure_controller_value.py` and any future reader call `score` directly.
    A guard that lived only in the command would be absent on every one of those paths."""
    runs = [rec("P0")]
    bad = E.BaselineKey("c1b5914", "abc123", "NOT-THE-MODEL", "2026-07-27")
    d = E.score(runs, KEY, fields=FIELDS, key=bad).to_dict()
    assert any("model" in c for c in d["key_contradictions"])


# ------------------------------------------------------- compare: the +5.6 points

def test_two_arms_differing_only_in_their_prompt_are_no_longer_identical_keys():
    """THE ZERO-CONSUMER FIX. Both arms ran the same commit, spec and model; one carried an extra
    skill card, so only `experiment_config_hash` moved. Before this, `key_differences` was empty and
    the reader was told the two runs were the same configuration."""
    a = score([rec("P0", ech="aaaaaaaaaaaaaaaa")]).to_dict()
    b = score([rec("P0", ech="bbbbbbbbbbbbbbbb")]).to_dict()
    d = E.compare(a, b)
    assert any("experiment_config_hash" in s for s in d["key_differences"])


def test_a_baseline_that_is_a_mixture_of_arms_cannot_be_an_endpoint_of_a_delta():
    """The measured failure, in its own test: a two-spec-hash tree scored as a clean +5.6 points.
    `tools/analyze_arms.py:191` already refuses this shape; the evaluation plane averaged it."""
    mixed = score([rec("P0", spec_hash="abc123"), rec("P1", spec_hash="zzz999")]).to_dict()
    clean = score([rec("P0"), rec("P1")]).to_dict()
    d = E.compare(clean, mixed)
    assert d["verdict"] == "NOT_COMPARABLE"
    nc = d["not_comparable"]
    assert "spec_hash" in nc["mixed_fields"]
    assert nc["remedy"]


def test_a_contradicted_baseline_cannot_be_an_endpoint_either():
    runs = [rec("P0")]
    bad = E.score(runs, KEY, fields=FIELDS,
                  key=E.BaselineKey("c1b5914", "abc123", "NOT-THE-MODEL", "2026-07-27")).to_dict()
    d = E.compare(score(runs).to_dict(), bad)
    assert d["verdict"] == "NOT_COMPARABLE"
    assert d["not_comparable"]["contradictions"]


def test_an_older_baseline_with_no_arm_hash_is_not_reported_as_a_changed_arm():
    """"The arm changed" and "one of these files cannot say" are different claims, and only one of
    them is a reason to re-run. Every baseline written before today is the second."""
    old = score([rec("P0", ech=None)]).to_dict()
    new = score([rec("P0", ech="bbbbbbbbbbbbbbbb")]).to_dict()
    d = E.compare(old, new)
    assert d["verdict"] != "NOT_COMPARABLE"
    line = [s for s in d["key_differences"] if "experiment_config_hash" in s]
    assert line and "not recorded" in line[0]


def test_a_clean_pair_still_compares_cleanly():
    """The regression guard: none of the above may turn an ordinary comparison into a refusal."""
    d = E.compare(score([rec("P0")]).to_dict(), score([rec("P0")]).to_dict())
    assert d["verdict"] == "OK" and d["key_differences"] == []


# ------------------------------------------------------- the command no longer demands the strings

def test_the_four_strings_are_optional_on_the_command_line():
    from acr.commands.cli_eval import score as cmd
    for name in ("commit", "spec_hash", "model", "date"):
        p = inspect.signature(cmd).parameters[name]
        assert p.default is not inspect.Parameter.empty, name
        assert getattr(p.default, "default", None) == "", (
            f"--{name.replace('_', '-')} must default to empty so the key comes from the runs")


def test_the_baseline_string_carries_the_arm_when_there_is_one():
    """`baseline_key_str` is what a human pastes into a note. It has to differ between two arms."""
    assert E.BaselineKey("c", "h", "m", "d").as_str() == "c|h|m|d", "an old key gains no empty field"
    assert E.BaselineKey("c", "h", "m", "d", experiment_config_hash="a1").as_str() == "c|h|m|d|a1"
