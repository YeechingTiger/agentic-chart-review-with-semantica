"""Acceptance tests for the observable-period / empty-window logic.

The four SYN0009-0012 controls exist as a set, not individually. SYN0011 alone would pass
under an implementation that simply waved through every truncated record; SYN0012 is what
stops that. SYN0009 and SYN0010 alone would pass under an implementation that guessed; the
point is that they are indistinguishable, so the pair is the test.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from acr.corpus import Corpus
from acr.coverage import (StratumResult, clip_and_judge, clopper_pearson_upper,
                          enumerate_windows, evaluate_gate, summarise_windows)

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "patients"
QUALIFYING = ["Onc-", "CT", "PET", "Pathology", "Imaging"]


def _run(pid: str):
    gt = json.loads((CORPUS / pid / "_ground_truth.json").read_text())
    chart = Corpus(CORPUS).chart(pid)
    docs, _ = chart.list_documents(limit=10_000)
    dx = gt["ground_truth"]["STORE.390.date_of_initial_diagnosis"]["value"]
    anchor = date(int(dx[:4]), int(dx[4:6]), int(dx[6:8]))
    obs_end = max(d.date for d in docs)

    windows = clip_and_judge(
        enumerate_windows(anchor, obs_end), docs,
        anchor=anchor, observable_start=None, observable_end=obs_end,
        qualifying_doc_types=QUALIFYING,
    )
    summary = summarise_windows(windows, snapshot=date(2026, 7, 25))
    gate = evaluate_gate(
        {"require_can_establish_nonempty": True, "max_elusion_upper": 0.12},
        [StratumResult("can_establish", 5, 5, True, elusion_upper=0.0),
         StratumResult("cannot_establish", 280, sampled=25, sample_hits=0,
                       elusion_upper=clopper_pearson_upper(0, 25))],
        windows,
    )
    return gt, summary, gate, obs_end


# ---------------------------------------------------------------- statistics
def test_clopper_pearson_zero_hits_closed_form():
    assert clopper_pearson_upper(0, 25) == pytest.approx(0.1129, abs=1e-4)
    assert clopper_pearson_upper(0, 50) == pytest.approx(0.0582, abs=1e-4)
    assert clopper_pearson_upper(0, 0) == 1.0          # nothing sampled proves nothing


def test_more_sampling_tightens_the_bound():
    bounds = [clopper_pearson_upper(0, m) for m in (10, 25, 50, 100)]
    assert bounds == sorted(bounds, reverse=True)


# ---------------------------------------------------------------- interior gaps
@pytest.mark.parametrize("pid", ["SYN0009", "SYN0010"])
def test_interior_gap_blocks_the_gate(pid):
    _, summary, gate, _ = _run(pid)
    assert summary["interior_gaps"], "a two-year hole in surveillance must be detected"
    assert gate.verdict == "FAIL"
    assert any("interior follow-up gap" in m for m in gate.missing)


def test_the_interior_gap_pair_is_indistinguishable():
    """SYN0010 recurred during the hole, at another hospital. Nothing here can show that.

    If these two ever diverge, the implementation is reading something it cannot have.
    """
    _, a, ga, _ = _run("SYN0009")
    _, b, gb, _ = _run("SYN0010")
    assert a["interior_gaps"] == b["interior_gaps"]
    assert a["through_date"] == b["through_date"]
    assert a["finality"] == b["finality"]
    assert ga.verdict == gb.verdict == "FAIL"


# ---------------------------------------------------------------- truncation
@pytest.mark.parametrize("pid", ["SYN0011", "SYN0012"])
def test_truncation_is_not_a_gap(pid):
    """The acceptance point. Loss to follow-up narrows the scope; it does not break it."""
    _, summary, gate, obs_end = _run(pid)
    assert summary["interior_gaps"] == [], "trailing empty windows are truncation, not gaps"
    assert gate.verdict == "PASS"
    assert summary["windows_clipped"] >= 1


@pytest.mark.parametrize("pid", ["SYN0011", "SYN0012"])
def test_truncated_claims_report_scope_and_provisional_finality(pid):
    _, summary, _, obs_end = _run(pid)
    assert summary["through_date"] is not None, "a negative claim without a through_date is unscopeable"
    assert date.fromisoformat(summary["through_date"]) <= obs_end
    assert summary["finality"]["value"] == "Provisional"
    assert "OBSERVATION_TRUNCATED" in summary["finality"]["reason"], (
        "PENDING_FILING and OBSERVATION_TRUNCATED need different remedies, so the reason "
        "has to survive into the output"
    )


def test_clipping_changes_scope_not_verdict():
    """SYN0011 and SYN0012 truncate identically; only SYN0012 has a visible recurrence.

    Coverage machinery must treat them the same, because deciding *what happened* is the
    extraction step's job, not the coverage gate's.
    """
    _, a, ga, _ = _run("SYN0011")
    _, b, gb, _ = _run("SYN0012")
    assert ga.verdict == gb.verdict == "PASS"
    assert a["interior_gaps"] == b["interior_gaps"] == []
    assert (b["through_date"] is not None) and (a["through_date"] is not None)


def test_syn0012_recurrence_is_actually_visible():
    """Guards the negative control itself: if the corpus stopped showing the recurrence,
    test_clipping_changes_scope_not_verdict would pass for the wrong reason."""
    chart = Corpus(CORPUS).chart("SYN0012")
    assert chart.search("recurren", max_hits=50), "SYN0012 must contain visible recurrence text"
    assert not Corpus(CORPUS).chart("SYN0010").search("recurren", max_hits=50), (
        "SYN0010's recurrence happened elsewhere and must leave no trace"
    )
