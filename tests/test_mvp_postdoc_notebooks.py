"""Regression checks for the teaching-first notebook surface."""
from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _read(name: str):
    return nbformat.read(NOTEBOOKS / name, as_version=4)


def _visible_source(notebook) -> str:
    cells = []
    for cell in notebook.cells:
        hidden = (cell.get("metadata") or {}).get("jupyter", {}).get("source_hidden")
        if not hidden:
            cells.append(cell.source)
    return "\n".join(cells)


def test_each_walkthrough_has_one_small_teaching_job():
    run = _read("01_run_chart_review_experiments.ipynb")
    audit = _read("02_trace_to_decision_chain.ipynb")
    intelligence = _read("03_semantica_decision_intelligence.ipynb")

    assert len(run.cells) <= 7
    assert len(audit.cells) <= 11
    assert len(intelligence.cells) <= 11

    run_text = _visible_source(run)
    assert "The agent in one picture" in run_text
    assert "What came back?" in run_text
    assert "causal" not in run_text.lower()

    audit_text = _visible_source(audit)
    assert "long receipt" in audit_text
    assert "eight questions" in audit_text
    assert "Where should the human spend time?" in audit_text

    intelligence_text = _visible_source(intelligence)
    assert "decision card box with an index" in intelligence_text
    assert "Find similar decisions" in intelligence_text
    assert "Semantica—not ACR—calculates the similarity" in intelligence_text
    assert "graph.find_similar_decisions" in intelligence_text
    assert "same question with different answers" in intelligence_text
    assert "policy changes" in intelligence_text

    intelligence_source = "\n".join(cell.source for cell in intelligence.cells)
    native_call = intelligence_source.index("ledger.graph.find_similar_decisions(")
    acr_guard = intelligence_source.index("ledger.similar_candidates(")
    assert native_call < acr_guard
    assert "Presentation-only thinning" in intelligence_source


def test_default_notebook_surface_hides_machine_index_and_noncore_analytics():
    text = "\n".join(
        _visible_source(_read(name))
        for name in (
            "01_run_chart_review_experiments.ipynb",
            "02_trace_to_decision_chain.ipynb",
            "03_semantica_decision_intelligence.ipynb",
        )
    ).lower()
    forbidden = (
        "predecision decision_function=",
        "centrality",
        "community detection",
        "w3c prov turtle",
        "graphrag",
        "generic impact candidates",
        "node types",
        "relationship types",
    )
    assert not [phrase for phrase in forbidden if phrase in text]


def test_executed_companions_are_code_free_reading_copies_of_current_sources():
    for name in (
        "01_run_chart_review_experiments",
        "02_trace_to_decision_chain",
        "03_semantica_decision_intelligence",
    ):
        source = _read(f"{name}.ipynb")
        executed = _read(f"{name}.executed.ipynb")
        assert executed.metadata["acr"]["reading_copy"] is True
        executed_cells = [
            cell for cell in executed.cells
            if not (cell.get("metadata") or {}).get("acr_reading_copy_note")
        ]
        assert len(source.cells) == len(executed_cells)
        saw_output = False
        for source_cell, executed_cell in zip(source.cells, executed_cells):
            if source_cell.cell_type != "code":
                continue
            expected = hashlib.sha256(source_cell.source.encode("utf-8")).hexdigest()
            assert executed_cell.source == ""
            assert executed_cell.metadata["acr_source_cell_sha256"] == expected
            assert not [row for row in executed_cell.get("outputs") or []
                        if row.get("output_type") == "error"]
            saw_output |= bool(executed_cell.get("outputs"))
        assert saw_output
