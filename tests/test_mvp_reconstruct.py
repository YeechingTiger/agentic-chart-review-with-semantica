"""Reading a finished run back as a decision tree, and refusing to believe the reader.

The extractor is stubbed everywhere here. A test that called a real model would measure the
model, and the thing under test is the opposite: what this module does when the extractor is
wrong — a span that points nowhere, a quote nobody said, a citation to a document the run
never opened. Those are the cases the whole design exists for, and they cannot be provoked
reliably from a live model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from acr.mvp.ledger import NullLedger
from acr.mvp.reconstruct import (RECONSTRUCTED, SELF_REPORTED, build_prompt, reconstruct_run,
                                 render, run_sheet, verify)
from acr.mvp.warrants import RunFacts


def _write_run(run_dir: Path) -> None:
    """One honest little run: a search, a narrated decision, a read, evidence, a submission."""
    run_dir.mkdir(parents=True)
    events = [
        {"seq": 1, "kind": "run_meta", "spec_id": "STORE.390.date_of_initial_diagnosis",
         "patient_id": "SYN0001"},
        {"seq": 2, "kind": "tool_call", "tool": "search", "ok": True,
         "args": {"query": "adenocarcinoma", "objective": "find the diagnosing pathology"},
         "result": {"hits": [{"note_id": "SPD_2023-04-12"}], "n": 1,
                    "objective": "find the diagnosing pathology"}},
        {"seq": 3, "kind": "tool_call", "tool": "note_decision", "ok": True,
         "args": {"facing": "two documents could date the diagnosis",
                  "decision": "date the case by the cytology",
                  "because": "the earlier document governs when it is unambiguous",
                  "used": ["search:adenocarcinoma"], "grounding": ["own_knowledge"]},
         "result": {"noted": True, "grounding": ["own_knowledge"],
                    "used": [{"ref": "search:adenocarcinoma", "kind": "search",
                              "verified": True}],
                    "context": {"n_searches": 1, "n_evidence": 0}}},
        {"seq": 4, "kind": "tool_call", "tool": "read", "ok": True,
         "args": {"note_id": "SPD_2023-04-12"},
         "result": {"note_id": "SPD_2023-04-12", "total_chars": 900, "returned_chars": 900}},
        {"seq": 5, "kind": "tool_call", "tool": "record_evidence", "ok": True,
         "args": {"note_id": "SPD_2023-04-12", "start": 0, "end": 14, "supports": "histology"},
         "result": {"recorded": True, "quote": "adenocarcinoma", "n_evidence": 1}},
        {"seq": 6, "kind": "tool_call", "tool": "submit_answer", "ok": True,
         "args": {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"}},
         "result": {"accepted": True, "why": "obligations discharged"}},
        {"seq": 7, "kind": "answer_accepted", "status": "FOUND"},
    ]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _facts(run_dir: Path) -> RunFacts:
    return RunFacts.from_trace(
        json.loads(ln) for ln in
        (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip())


class StubLLM:
    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies, self.prompts = list(replies), []

    def generate_structured(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


def _good(**over: Any) -> dict[str, Any]:
    point = {"span": [2, 3], "decision_type": "which_wins",
             "scenario": "two sources could date the case and they disagree",
             "reasoning": "the earlier document was unambiguous",
             "outcome": "date the case by the cytology",
             "grounding": ["own_knowledge"], "used": ["search:adenocarcinoma"],
             "quote": "the earlier document governs when it is unambiguous",
             "small_points": [{"span": [2, 2], "decision_type": "where_to_look",
                               "scenario": "nothing examined yet", "reasoning": "pathology first",
                               "outcome": "search adenocarcinoma", "grounding": [],
                               "used": [], "quote": None}]}
    return {"big_points": [{**point, **over}]}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "20260826T000000Z_SYN0001_STORE_390"
    _write_run(d)
    return d


def test_the_sheet_anchors_every_line_and_separates_the_two_channels(run_dir: Path):
    sheet = run_sheet(run_dir)
    assert sheet["seqs"] == {2, 3, 4, 5, 6, 7}
    body = "\n".join(sheet["lines"])
    assert "[seq 2] action (SERVER FACT): search" in body
    assert 'objective it stated: "find the diagnosing pathology"' in body
    assert "[seq 3] decision (SELF-REPORTED content" in body
    assert "grounding it claimed: own_knowledge" in body
    assert "[seq 6] gate verdict (SERVER FACT): ACCEPTED" in body
    # The extractor is shown the taxonomy and told which half belongs to which level.
    prompt = build_prompt(sheet)
    assert "which_wins" in prompt and "where_to_look" in prompt
    assert "is_this_it" in prompt.split("Small point types:")[1]


def test_a_quote_the_model_really_said_is_self_reported(run_dir: Path):
    v = verify(_good(), run_sheet(run_dir), _facts(run_dir))
    big = v["points"][0]
    assert big["provenance"] == SELF_REPORTED
    assert big["grounding"] == ["own_knowledge"]
    assert big["grounding_provenance"] == SELF_REPORTED
    assert v["own_knowledge"] == {"total": 1, "self_reported": 1}
    assert big["small_points"][0]["provenance"] == RECONSTRUCTED   # honest null


def test_a_quote_nobody_said_is_rejected_and_the_point_drops_to_reconstructed(run_dir: Path):
    """The load-bearing check. An extractor that invents a plausible sentence must not be
    able to launder it into 'the model said so'."""
    v = verify(_good(quote="the contract's evidence ladder settles this"),
               run_sheet(run_dir), _facts(run_dir))
    big = v["points"][0]
    assert big["quote"] is None
    assert big["provenance"] == RECONSTRUCTED
    assert big["quote_rejected"] is True
    assert v["quotes_rejected"] == 1
    # The grounding claim survives as a claim, but stops being a question for an expert.
    assert big["grounding"] == ["own_knowledge"]
    assert big["grounding_provenance"] == RECONSTRUCTED
    assert v["own_knowledge"] == {"total": 1, "self_reported": 0}


def test_a_server_fact_cannot_be_quoted_as_the_models_thinking(run_dir: Path):
    """Quoting the trace back at us proves the server observed something, never that the
    model thought it."""
    v = verify(_good(quote="search {\"query\": \"adenocarcinoma\"}"),
               run_sheet(run_dir), _facts(run_dir))
    assert v["points"][0]["provenance"] == RECONSTRUCTED


def test_a_point_anchored_to_no_real_seq_is_dropped_not_stored(run_dir: Path):
    v = verify(_good(span=[80, 90]), run_sheet(run_dir), _facts(run_dir))
    assert v["points"] == []
    assert v["dropped"] == {"big": 1, "small": 1}   # its children go with it
    assert v["n_big"] == 0


def test_a_citation_the_run_never_made_is_marked_false_not_believed(run_dir: Path):
    """The one finding a post-hoc reader CAN still make, because the run said it out loud:
    the model cited a document, and the record says it never opened one."""
    v = verify(_good(used=["note:Never-Opened-Note_2019-01-01", "search:adenocarcinoma"]),
               run_sheet(run_dir), _facts(run_dir))
    big = v["points"][0]
    assert big["used_unverified"] == ["note:Never-Opened-Note_2019-01-01"]
    assert v["unverified_warrants"] == [
        {"seq": 2, "decided": "date the case by the cytology",
         "refs": ["note:Never-Opened-Note_2019-01-01"]}]
    assert "FALSE WARRANT" in render({**v, "run_id": "r", "case_id": "c"})


def test_a_type_at_the_wrong_level_keeps_its_name_and_moves_level(run_dir: Path):
    """A big-point name arriving in the small list is a segmentation error, not a naming one.
    Discarding either half would lose information we have."""
    raw = _good()
    raw["big_points"][0]["small_points"][0]["decision_type"] = "which_wins"
    small = verify(raw, run_sheet(run_dir), _facts(run_dir))["points"][0]["small_points"][0]
    assert small["decision_type"] == "which_wins"
    assert small["level"] == "big"


def test_an_unknown_type_becomes_other_and_keeps_what_was_claimed(run_dir: Path):
    big = verify(_good(decision_type="vibes"),
                 run_sheet(run_dir), _facts(run_dir))["points"][0]
    assert big["decision_type"] == "other" and big["claimed_type"] == "vibes"


def test_seqs_no_point_covers_are_reported_as_unexplained(run_dir: Path):
    v = verify(_good(), run_sheet(run_dir), _facts(run_dir))
    assert v["seqs_unaccounted"] == [4, 5, 6, 7]
    assert "UNACCOUNTED" in render({**v, "run_id": "r", "case_id": "c"})


def test_the_whole_verb_writes_a_tree_and_reports_its_own_stability(run_dir: Path):
    ledger = NullLedger()
    other = _good()
    other["big_points"][0]["small_points"] = []          # a second, disagreeing reading
    llm = StubLLM(_good(), other)
    summary = reconstruct_run(run_dir, ledger, llm, passes=2)

    assert summary["case_id"] == "SYN0001"
    assert summary["n_big"] == 1 and summary["n_small"] == 1
    assert summary["types"] == {"big:which_wins": 1, "small:where_to_look": 1}
    # 2 judgments; INFLUENCED needs two big points so there is none, PART_OF ties the small
    # one to its parent.
    assert ledger.counts["judgments"] == 2 and ledger.counts["edges"] == 1
    # Only the first reading was stored; the second exists to disagree with it.
    assert summary["stability"] == {"passes": 2, "n_big": [1, 1], "n_small": [1, 0],
                                    "types_agree": False}
    assert len(llm.prompts) == 2


def test_semantica_gets_a_tree_whose_chain_shows_only_the_big_points(tmp_path: Path,
                                                                    run_dir: Path):
    """PART_OF is composition, not causation — so the audit chain reads as conclusions, and
    the steps behind one are a query away rather than in the way."""
    pytest.importorskip("semantica")
    from acr.mvp.ledger import SemanticaLedger, ingest_run

    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ingest_run(run_dir, ledger)          # the run's own result, so the case has a chain
    two = _good()
    two["big_points"].append({**two["big_points"][0], "span": [5, 6],
                              "decision_type": "enough", "scenario": "evidence is in hand",
                              "outcome": "submit FOUND", "small_points": []})
    reconstruct_run(run_dir, ledger, StubLLM(two))

    cats = [r["category"] for r in ledger.chain("SYN0001")]
    assert "small:where_to_look" not in cats
    assert {"big:which_wins", "big:enough"} <= set(cats)

    big = next(r for r in ledger.decisions(category_prefix="big:which_wins"))
    parts = ledger.parts_of(big["decision_id"])
    assert [p["category"] for p in parts] == ["small:where_to_look"]

    reloaded = SemanticaLedger(tmp_path / "ledger.json")
    assert reloaded.precedents("two sources could date the case and they disagree",
                               category="big:which_wins"), "the tree did not survive a reload"
