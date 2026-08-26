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
    """A minimal but honest run: two decision points, one evidence span, one refused
    submission, one accepted."""
    run_dir.mkdir(parents=True)
    events = [
        {"seq": 1, "ts": "t", "kind": "run_meta", "spec_id": "STORE.390.date_of_initial_diagnosis",
         "spec_hash": "h", "patient_id": "SYN0001", "submittable": ["FOUND"]},
        {"seq": 2, "ts": "t", "kind": "tool_call", "tool": "note_decision",
         "args": {"decision_type": "stopping", "facing": "no evidence gathered yet",
                  "decision": "submit from memory", "because": "overconfidence"},
         "result": {"noted": True, "n_decisions": 1, "decision_type": "stopping",
                    "context": {"n_searches": 0, "n_evidence": 0}}, "ok": True},
        {"seq": 3, "ts": "t", "kind": "tool_call", "tool": "submit_answer",
         "args": {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"}},
         "result": {"accepted": False, "why": "a value answer owes recorded evidence"}, "ok": True},
        {"seq": 4, "ts": "t", "kind": "tool_call", "tool": "note_decision",
         "args": {"decision_type": "sufficiency",
                  "facing": "refused: a value answer owes recorded evidence",
                  "decision": "record the pathology span, then resubmit",
                  "because": "the gate names the missing obligation",
                  "options": ["abstain instead"]},
         "result": {"noted": True, "n_decisions": 2, "decision_type": "sufficiency",
                    "context": {"n_searches": 0, "n_evidence": 0}}, "ok": True},
        {"seq": 5, "ts": "t", "kind": "tool_call", "tool": "record_evidence",
         "args": {"note_id": "Surgical-Pathology-Document_2023-04-12", "start": 310, "end": 324,
                  "supports": "histology named"},
         "result": {"recorded": True, "n_evidence": 1, "quote": "adenocarcinoma"}, "ok": True},
        {"seq": 6, "ts": "t", "kind": "tool_call", "tool": "submit_answer",
         "args": {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
                  "reasoning": "FNA pathology"},
         "result": {"accepted": True, "why": "obligations for this status are discharged"},
         "ok": True},
        {"seq": 7, "ts": "t", "kind": "answer_accepted", "status": "FOUND"},
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
                       "n_steps": 2, "n_submissions": 2, "result": "FOUND"}
    # 1 evidence; 2 steps + 2 submissions + 2 gate verdicts + 1 result = 7 judgments.
    # Edges: INFLUENCED step1->step2, step1->submission1, step2->submission2 (3);
    # uses only on the accepted submission — the refused one preceded the evidence (1);
    # CAUSED submission->gate twice and last-gate->result once (3). Total 7.
    assert ledger.counts["evidence"] == 1
    assert ledger.counts["judgments"] == 7
    assert ledger.counts["edges"] == 7


def test_a_run_that_never_answered_still_ingests(tmp_path: Path):
    run_dir = tmp_path / "20260825T000001Z_SYN0002_STORE_390"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text(json.dumps(
        {"seq": 1, "ts": "t", "kind": "run_meta", "spec_id": "s", "patient_id": "SYN0002"}) + "\n",
        encoding="utf-8")
    summary = ingest_run(run_dir, NullLedger())   # no result.json: the honest NO_ANSWER
    assert summary["result"] == "NO_ANSWER"
    assert summary["n_submissions"] == 0
    assert summary["n_steps"] == 0


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

    # The model's decision points are ON the chain, ordered: the walk reaches back through
    # the accepted submission to step 2 and from there to step 1.
    cats = [row["category"].split(":")[0] for row in chain]
    assert cats[0] == "result"
    steps = [row for row in chain if row["category"].startswith("step:")]
    assert [s["outcome"] for s in steps] == [
        "record the pathology span, then resubmit", "submit from memory"]  # nearest first
    assert [s["category"] for s in steps] == ["step:sufficiency", "step:stopping"]
    assert all(row.get("distance") is not None for row in chain)

    # The compare verb: decision points selected by TYPE, across everything ingested,
    # each carrying its server-recorded context snapshot.
    by_type = ledger.decisions(category_prefix="step:sufficiency")
    assert len(by_type) == 1
    assert by_type[0]["outcome"] == "record the pathology span, then resubmit"
    assert by_type[0]["case_id"] == "SYN0001"
    assert by_type[0]["context"] == {"n_searches": 0, "n_evidence": 0}
    assert ledger.decisions(category_prefix="step:coverage") == []

    # Persistence: a fresh ledger over the same files answers the same audit.
    assert ledger_path.exists() and ledger_path.with_suffix(".index.json").exists()
    reloaded = SemanticaLedger(ledger_path)
    assert reloaded.chain("SYN0001"), "the chain did not survive save/load"
    assert reloaded.stats()["cases"] == 1


def test_live_recording_equals_replaying_the_trace(tmp_path: Path):
    """Semantica's own usage pattern — record at decision time — and our rebuild path must
    produce the same books. Drive the real toolserver with a live ledger, then replay its
    trace into a fresh one, and compare what each recorded (ids are uuids; content isn't)."""
    pytest.importorskip("semantica")
    from acr.mvp.ledger import SemanticaLedger
    from acr.mvp.toolserver import ChartToolServer

    root = Path(__file__).resolve().parents[1]
    live_path = tmp_path / "live.json"
    server = ChartToolServer(root / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml",
                             root / "corpus" / "patients" / "SYN0001",
                             tmp_path / "run", ledger_path=live_path)
    # Lazy by design: the ledger must not exist yet — constructing it here would delay the
    # MCP handshake past the model's first calls.
    assert server._recorder is None and not live_path.exists()
    server.call("note_decision", {"decision_type": "search_strategy", "facing": "f",
                                  "decision": "search pathology terms", "because": "b"})
    hit = json.loads(json.dumps(server.call("search", {"query": "adenocarcinoma"})[0]))["hits"][0]
    server.call("record_evidence", {"note_id": hit["note_id"], "start": hit["start"],
                                    "end": hit["end"], "supports": "s"})
    server.call("submit_answer", {"status": "FOUND",
                                  "value": {"date_of_initial_diagnosis": "20230412"},
                                  "reasoning": "r"})

    replayed = SemanticaLedger(tmp_path / "replayed.json")
    summary = ingest_run(tmp_path / "run", replayed)
    assert summary["result"] == "FOUND" and summary["n_steps"] == 1

    live = SemanticaLedger(live_path)   # reload from disk: what the sync path persisted

    def content(ledger):
        return [(r["category"], r["outcome"], r["decision_maker"], r["seq"], r["context"])
                for r in ledger.decisions()]

    assert content(live) == content(replayed)
    assert [r["outcome"] for r in live.chain("SYN0001")] == \
           [r["outcome"] for r in replayed.chain("SYN0001")]
    live_stats, replay_stats = live.stats(), replayed.stats()
    assert live_stats["node_count"] == replay_stats["node_count"]
    assert live_stats["edge_count"] == replay_stats["edge_count"]
