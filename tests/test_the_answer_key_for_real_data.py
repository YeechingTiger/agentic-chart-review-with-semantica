"""An empty registry cell is not a claim that abstention is correct, and mapping it to one inverts
the metric on exactly the cases the study is about.

[evals.py](../src/acr/evaluation/evals.py) `_outcome_for`: a key value of `None` **asserts** that
abstaining is the right answer. So on a real registry export, where a missing value is simply an
empty cell:

  * the run that finds the right value scores `ANSWERED_OVER_ABSTAIN` — **a failure for being right**
  * the run that gives up scores `ABSTAINED_CORRECT` — **a success for giving up**

Both land in the aggregate looking like ordinary outcomes. That is the failure mode: it is not a
crash, it is a number.

## The vocabulary for this already existed

`contract/behaviour.py` carries it and nothing was consuming it for evaluation:

    chart_derivability ∈ DERIVABLE | PARTIALLY_DERIVABLE | NOT_DERIVABLE | UNRESOLVED
    adjudication       ∈ key_correct | key_wrong | outside_chart | unresolved
    GoldField.status   ∈ FOUND | EVIDENCE_INSUFFICIENT | SPEC_INSUFFICIENT | NOT_APPLICABLE

`ChartObservableGold.__post_init__` already enforces that `outside_chart` requires `NOT_DERIVABLE` —
which is precisely "a human read the chart and the value is not in it", the only thing that licenses
`None`. And `GoldField`'s own docstring says it covers "a correct abstention".

So the converter invents no semantics. It maps a vocabulary that exists onto the one `eval score`
reads, and **refuses to guess** where the vocabulary says nobody has decided yet.

| gold row | eval key | why |
|---|---|---|
| `chart_answer[f].status == FOUND` | `fields[f] = value` | scored for exact match |
| `chart_answer[f].status` is an abstention | `fields[f] = None` | scored for correct abstention |
| `f` absent from `chart_answer` | field omitted | `NO_KEY`; counted in `n_unkeyed`, in nothing else |
| `adjudication == key_wrong` | case omitted | scoring against a key a human found wrong |
| `adjudication == unresolved` | case omitted | nobody has looked |

The omissions are not silent: every one becomes a row in the worklist, which is the human's queue and
the honest denominator of the study.
"""

from __future__ import annotations

import json

import pytest

from acr.contract import behaviour as B
from acr.contract.behaviour import answer_key_from_gold

SPEC = "STORE.390.date_of_initial_diagnosis"
FIELD = "date_of_initial_diagnosis"


def _row(case, *, status=B.FOUND, value="20190312", derivability=B.DERIVABLE,
         adjudication=B.KEY_CORRECT, omit_field=False, subgroups=()):
    answer = {} if omit_field else {FIELD: {"status": status, "value": value}}
    return {"case_id": case, "spec_id": SPEC,
            "registry_value": {FIELD: value or ""},
            "registry_source_version": "v1",
            "chart_derivability": derivability,
            "chart_answer": answer,
            "gold_evidence": [], "adjudication": adjudication,
            "subgroups": list(subgroups)}


def _gold(*rows):
    return {"schema": B.GOLD_SCHEMA, "cases": list(rows)}


def convert(*rows, **kw):
    return answer_key_from_gold(_gold(*rows), fields=[FIELD], **kw)


# ------------------------------------------------------------------ the three populations

def test_a_found_field_is_scored_for_exact_match():
    key, work = convert(_row("C1"))
    assert key["C1"]["fields"] == {FIELD: "20190312"}
    assert work == []


def test_an_adjudicated_abstention_becomes_null():
    """`outside_chart` requires `NOT_DERIVABLE`, which `ChartObservableGold` already enforces: a
    human read the chart and the value is not in it. That is the only thing that licenses `None`."""
    key, work = convert(_row("C1", status=B.EVIDENCE_INSUFFICIENT, value=None,
                             derivability=B.NOT_DERIVABLE, adjudication=B.OUTSIDE_CHART))
    assert key["C1"]["fields"] == {FIELD: None}
    assert work == []


def test_an_unadjudicated_empty_cell_is_omitted_and_queued():
    """THE DEFECT THIS FILE EXISTS FOR. Omitted, so `_outcome_for` returns `NO_KEY` and the case is
    counted in `n_unkeyed` and in no rate — instead of asserting that giving up was correct."""
    key, work = convert(_row("C1", omit_field=True, derivability=B.UNRESOLVED,
                             adjudication=B.ADJUDICATION_UNRESOLVED))
    assert key["C1"]["fields"] == {}, "no field may be asserted for this case"
    assert [(w["case_id"], w["field"]) for w in work] == [("C1", FIELD)]
    assert work[0]["why"]


def test_a_key_a_human_found_wrong_is_not_scored_against():
    """`key_wrong` means the registry value is wrong. Scoring a run against it measures agreement
    with a known error, and the run that gets it right is marked wrong."""
    key, work = convert(_row("C1", adjudication=B.KEY_WRONG))
    assert "C1" not in key
    assert work and work[0]["adjudication"] == B.KEY_WRONG


def test_the_three_populations_are_counted_separately():
    key, work = convert(_row("C1"),
                        _row("C2", status=B.EVIDENCE_INSUFFICIENT, value=None,
                             derivability=B.NOT_DERIVABLE, adjudication=B.OUTSIDE_CHART),
                        _row("C3", omit_field=True, derivability=B.UNRESOLVED,
                             adjudication=B.ADJUDICATION_UNRESOLVED))
    counts = key["_summary"]
    assert counts == {"n_cases": 3, "n_with_value": 1, "n_correct_abstention": 1,
                      "n_unadjudicated": 1, "n_key_wrong": 0, "n_worklist": 1}
    assert len(work) == 1


# ------------------------------------------------------------------ what eval score needs

def test_every_row_carries_the_spec_it_answers():
    """`evals._key_row` compares `run.spec_id` against the row's declared ids. Without one, a run of
    a DIFFERENT contract scores against this key as a wrong answer — measured on this tree: 8
    cross-spec runs dragged a published 75.8% to 74.6% with `n_unkeyed: 0`."""
    key, _ = convert(_row("C1"))
    assert key["C1"]["spec_id"] == SPEC


def test_extra_spec_ids_can_be_declared_for_an_ablation_arm():
    """An arm like `<base>.UNSTRATIFIED` has the same correct answers and a different policy."""
    key, _ = convert(_row("C1"), also_scores=[f"{SPEC}.UNSTRATIFIED"])
    assert set(key["C1"]["spec_ids"]) == {SPEC, f"{SPEC}.UNSTRATIFIED"}


def test_subgroups_travel_so_the_subgroup_arm_is_computable():
    """`eval compare` calls a subgroup regression the only reason to have the harness. It reads
    `totals.by_subgroup`, which is empty unless the KEY carries subgroups."""
    key, _ = convert(_row("C1", subgroups=("outside_hospital",)))
    assert key["C1"]["subgroups"] == ["outside_hospital"]


def test_the_metadata_keys_cannot_collide_with_a_case():
    """`eval score` reads the key as `{case: row}`, so the metadata sits under reserved `_` keys and
    a case id that starts with `_` is refused rather than shadowing them."""
    with pytest.raises(B.SpecRepairError, match="_"):
        convert(_row("_summary"))


def test_the_key_scores_through_the_real_scorer_unchanged():
    """PRODUCER TO CONSUMER. The reserved keys must not become instances, the omitted field must
    become `n_unkeyed`, and the null must become a correct abstention — asserted on the real
    `evals.score`, not on my reading of it."""
    from acr.evaluation import evals as E
    key, _ = convert(_row("C1"),
                     _row("C2", status=B.EVIDENCE_INSUFFICIENT, value=None,
                          derivability=B.NOT_DERIVABLE, adjudication=B.OUTSIDE_CHART),
                     _row("C3", omit_field=True, derivability=B.UNRESOLVED,
                          adjudication=B.ADJUDICATION_UNRESOLVED))

    def run(pid, value):
        answer = ({"status": "FOUND", "status_kind": "value", "value": {FIELD: value}}
                  if value else {"status": "EVIDENCE_INSUFFICIENT",
                                 "status_kind": "abstain_evidence", "value": {}})
        return E.RunRecord({"patient_id": pid, "spec_id": SPEC, "spec_hash": "h",
                            "answer": answer, "gate_validated": True}, source=pid)

    runs = [run("C1", "20190312"), run("C2", None), run("C3", "20200101")]
    rep = E.score(runs, key, fields=[FIELD], key=E.derive_baseline_key(runs))
    got = {r.instance_id.split("__")[0]: r.outcomes[0].outcome for r in rep.per_instance}
    assert got["C1"] == E.EXACT
    assert got["C2"] == E.ABSTAINED_CORRECT
    assert got["C3"] == E.NO_KEY, "an unadjudicated cell must score as nothing, not as a failure"
    assert rep.totals["n_instances"] == 3, "the reserved `_` keys must not become instances"
    # And the sharp one: C3 answering must NOT be `ANSWERED_OVER_ABSTAIN`. Those two outcomes look
    # alike in an aggregate and lead to opposite conclusions about the run.
    assert got["C3"] != E.ANSWERED_OVER_ABSTAIN


# ------------------------------------------------------------------ the PHI boundary

def test_the_converter_writes_only_through_the_local_store(tmp_path, monkeypatch):
    """A real answer key is patient-derived. `eval score` reads its key with plain `read_json` and
    enforces nothing, so nothing stopped one sitting inside the git worktree."""
    import typer
    from typer.testing import CliRunner

    from acr.commands.cli import app
    # The GOLD has to live in the store too, and `require_input` refuses it otherwise — correct, and
    # worth stating: gold carries registry values, so it is the same class of file as the key.
    local = tmp_path / "local"
    local.mkdir(mode=0o700)
    (local / "gold.json").write_text(json.dumps(_gold(_row("C1"))), encoding="utf-8")
    monkeypatch.setenv("ACR_LOCAL_ARTIFACT_ROOT", str(local))
    r = CliRunner().invoke(app, ["gold", "to-answer-key", "--gold", "gold.json",
                                 "--fields", FIELD, "--out", "key.json",
                                 "--worklist", "work.json"])
    assert r.exit_code == 0, r.output
    written = json.loads((tmp_path / "local" / "key.json").read_text(encoding="utf-8"))
    assert written["_contains_phi"] is True
    assert written["C1"]["fields"] == {FIELD: "20190312"}
    assert (tmp_path / "local" / "work.json").is_file()
    assert typer  # the import is the point: a BadParameter here is a refusal, not a crash

