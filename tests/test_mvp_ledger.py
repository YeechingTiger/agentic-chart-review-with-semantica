"""The judgment ledger: ingestion distills a Layer-1 trace, and the audit verb reads it back.

The write-behind claim is tested by construction: ingestion runs on a finished run directory,
so nothing here can affect a run. The NullLedger case pins the seam — the same ingestion drives
any ReviewLedger — and the semantica case (skipped when the pinned dependency is absent) proves
the chain: result <- gate <- submission, with evidence hanging off the submission by `uses` edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.mvp.ledger import NullLedger, ingest_run


def _write_run(run_dir: Path) -> None:
    """A minimal but honest run: one evidence span, one refused submission, one accepted."""
    run_dir.mkdir(parents=True)
    events = [
        {"seq": 1, "ts": "t", "kind": "run_meta", "spec_id": "STORE.390.date_of_initial_diagnosis",
         "spec_hash": "h", "patient_id": "SYN0001", "submittable": ["FOUND"]},
        {"seq": 2, "ts": "t", "kind": "tool_call", "tool": "submit_answer",
         "args": {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"}},
         "result": {"accepted": False, "why": "a value answer owes recorded evidence"}, "ok": True},
        {"seq": 3, "ts": "t", "kind": "tool_call", "tool": "record_evidence",
         "args": {"note_id": "Surgical-Pathology-Document_2023-04-12", "start": 310, "end": 324,
                  "supports": "histology named"},
         "result": {"recorded": True, "n_evidence": 1, "quote": "adenocarcinoma"}, "ok": True},
        {"seq": 4, "ts": "t", "kind": "tool_call", "tool": "submit_answer",
         "args": {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
                  "reasoning": "FNA pathology"},
         "result": {"accepted": True, "why": "obligations for this status are discharged"},
         "ok": True},
        {"seq": 5, "ts": "t", "kind": "answer_accepted", "status": "FOUND"},
    ]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({
        "status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
        "reasoning": "FNA pathology", "patient_id": "SYN0001",
    }), encoding="utf-8")


def test_ingestion_summary_and_the_null_ledger(tmp_path: Path):
    run_dir = tmp_path / "20260825T000000Z_SYN0001_STORE_390"
    _write_run(run_dir)
    ledger = NullLedger()
    summary = ingest_run(run_dir, ledger)
    assert summary == {"run_id": run_dir.name, "case_id": "SYN0001", "n_evidence": 1,
                       "n_submissions": 2, "result": "FOUND"}
    # 1 evidence; 2 submissions + 2 gate verdicts + 1 result = 5 judgments;
    # uses(submission1->ev0 does not exist: evidence came after) — the refused submission had
    # no evidence yet, the accepted one links 1; plus 2 CAUSED (sub->gate) + 1 (gate->result).
    assert ledger.counts["evidence"] == 1
    assert ledger.counts["judgments"] == 5
    assert ledger.counts["edges"] == 4


def test_a_run_that_never_answered_still_ingests(tmp_path: Path):
    run_dir = tmp_path / "20260825T000001Z_SYN0002_STORE_390"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text(json.dumps(
        {"seq": 1, "ts": "t", "kind": "run_meta", "spec_id": "s", "patient_id": "SYN0002"}) + "\n",
        encoding="utf-8")
    summary = ingest_run(run_dir, NullLedger())   # no result.json: the honest NO_ANSWER
    assert summary["result"] == "NO_ANSWER"
    assert summary["n_submissions"] == 0


def test_semantica_chain_walks_result_gate_submission(tmp_path: Path):
    pytest.importorskip("semantica")
    from acr.mvp.ledger import SemanticaLedger

    run_dir = tmp_path / "20260825T000000Z_SYN0001_STORE_390"
    _write_run(run_dir)
    ledger_path = tmp_path / "ledger.json"
    ledger = SemanticaLedger(ledger_path)
    summary = ingest_run(run_dir, ledger)
    assert summary["result"] == "FOUND"

    chain = ledger.chain("SYN0001")
    assert chain, "the audit verb returned nothing for an ingested case"
    makers = {row.get("decision_maker") for row in chain}
    assert "deterministic_runtime" in makers          # the result node itself
    assert {"rule_engine", "model"} & makers          # at least one upstream judgment

    # Persistence: a fresh ledger over the same files answers the same audit.
    assert ledger_path.exists() and ledger_path.with_suffix(".index.json").exists()
    reloaded = SemanticaLedger(ledger_path)
    assert reloaded.chain("SYN0001"), "the chain did not survive save/load"
    assert reloaded.stats()["cases"] == 1
