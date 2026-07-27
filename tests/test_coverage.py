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
from acr.coverage import (CoverageLedger, ForcedSampler, StratumResult, clip_and_judge,
                          clopper_pearson_upper, enumerate_windows, evaluate_gate,
                          strata_from_spec, summarise_windows)
from acr.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "patients"
SHB = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
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


# ================================================================= frame revalidation
# A SAMPLE IS TIED TO THE FRAME IT WAS DRAWN FROM.
#
# The audit that produced these numbers, on SYN0001 under STORE.400_522_523:
#
#   baseline    112 misses, 25 drawn, none relevant  -> elusion 0.1129, gate PASS
#   +1 term      92 misses; 20 of the 25 draws are now HITS, not misses
#                draws still inside the frame: 5     -> earned 0.4507
#                reported, unchanged:                          0.1129, gate PASS
#
# The spec's cap is 0.12: the reported bound cleared it and the earned bound missed it
# fourfold. Expansion strengthens the EVIDENCE and it can weaken the BOUND, and reaching a
# gate PASS on an inherited bound is anti-conservative — the worst direction for a mechanism
# whose entire job is to make an absence claim trustworthy.
SPEC = load_spec(SHB)
GATE = dict(SPEC.proof_obligation.for_negative["gate"])
N_DRAWN = 25
N_LEAVING_THE_FRAME = 20


def _honest_run(pid: str = "SYN0001") -> CoverageLedger:
    """A run that has genuinely done the work the proof obligation asks for.

    Every assertion below is measured against this, so if it ever stops reaching PASS the
    tests that follow prove nothing — `test_the_honest_run_earns_the_bound_it_reports`
    guards exactly that.
    """
    chart = Corpus(CORPUS).chart(pid)
    docs, _ = chart.list_documents(limit=100_000)
    specs = strata_from_spec(SPEC)
    cov = CoverageLedger(docs, specs, ForcedSampler(20260727))
    cov.listed_documents = True
    for d in cov.by_stratum["can_establish"]:                  # policy: exhaustive
        cov.note_read(d.note_id, d.doc_type)
    for kw in {k for s in specs for k in s.required_keywords}:  # ran, and turned up nothing
        cov.note_search(kw, [])
    for _stratum, drawn in cov.pending_samples().items():       # drawn by the runtime
        for d in drawn:
            cov.note_read(d.note_id, d.doc_type)
    cov.resolve_sample_verdicts(cited=set())
    return cov


def _stratum(cov: CoverageLedger, name: str) -> StratumResult:
    return next(r for r in cov.stratum_results() if r.name == name)


def test_the_honest_run_earns_the_bound_it_reports():
    cov = _honest_run()
    r = _stratum(cov, "may_mention")
    assert (r.misses, r.misses_sampled, r.miss_sample_hits) == (112, N_DRAWN, 0)
    assert r.elusion_upper == pytest.approx(0.1129, abs=1e-4)
    assert evaluate_gate(GATE, cov.stratum_results()).verdict == "PASS"
    assert GATE["max_elusion_upper"] == 0.12, "the cap this whole scenario turns on"


def test_one_added_term_invalidates_the_draws_that_left_the_frame():
    """THE DEFECT. One monotone term addition, and the bound must not be inherited.

    Nothing here is a hit: no drawn document turned out to be relevant, and the prior was
    never falsified. What changed is the POPULATION the sample was a sample of. Twenty of the
    twenty-five draws are search hits now, so they are no longer members of the miss frame
    the bound is a bound over, and crediting their verdicts to it is crediting evidence about
    one population to a claim about another.
    """
    cov = _honest_run()
    drawn = list(cov.drawn["may_mention"])
    assert len(drawn) == N_DRAWN, "fixture assumption"

    cov.note_search("a-newly-added-term", drawn[:N_LEAVING_THE_FRAME])

    r = _stratum(cov, "may_mention")
    assert r.misses == 112 - N_LEAVING_THE_FRAME
    assert r.miss_sample_hits == 0, "no verdict was overturned; only the frame moved"
    assert r.misses_sampled == N_DRAWN - N_LEAVING_THE_FRAME, (
        "the bound may only be credited to draws that are still inside the frame it is a "
        "bound over — `stratum_results` is still counting all 25 verdicts"
    )
    assert len(r.draws_invalidated) == N_LEAVING_THE_FRAME
    assert r.elusion_upper == pytest.approx(0.4507, abs=1e-3), (
        "the bound was inherited across a frame revision instead of being recomputed"
    )

    g = evaluate_gate(GATE, cov.stratum_results())
    assert g.verdict == "FAIL", (
        "a gate PASS on an inherited bound is anti-conservative: the reported bound clears "
        "the 0.12 cap and the earned bound misses it fourfold"
    )
    assert any("frame" in m for m in g.missing), g.missing


def test_the_revision_forces_replacement_draws_until_n_is_restored():
    """`pending_samples` refused replacements because `n_s >= need` counted departed draws."""
    cov = _honest_run()
    assert cov.pending_samples() == {}, "the honest run owes nothing before the revision"
    drawn = list(cov.drawn["may_mention"])
    cov.note_search("a-newly-added-term", drawn[:N_LEAVING_THE_FRAME])

    pending = cov.pending_samples()
    replacements = pending.get("may_mention") or []
    assert len(replacements) == N_LEAVING_THE_FRAME, (
        "the frame lost 20 of its 25 draws and the runtime demanded no replacement"
    )
    assert not {d.note_id for d in replacements} & set(drawn), (
        "a replacement must be a fresh draw, not a redraw of one already inspected")

    for d in replacements:
        cov.note_read(d.note_id, d.doc_type)
    cov.resolve_sample_verdicts(cited=set())

    r = _stratum(cov, "may_mention")
    assert r.misses_sampled == N_DRAWN and r.elusion_upper == pytest.approx(0.1129, abs=1e-4)
    assert evaluate_gate(GATE, cov.stratum_results()).verdict == "PASS", (
        "restoring n must be a way OUT — an obligation no work discharges is not a gate"
    )


def test_a_run_that_cannot_restore_n_does_not_pass():
    """Expand until the miss frame is too small to earn the cap, and the gate must refuse.

    The frame here has been CENSUSED: five misses remain and all five were inspected, so
    `keyword_list_validated` is satisfied and there is nothing left to draw. That is not the
    same as having earned the bound. Five clean observations bound the relevance rate at
    0.4507 and the spec asked for 0.12, so the run is EVIDENCE_INSUFFICIENT and must say so
    rather than inherit the 25-draw number it can no longer support.
    """
    cov = _honest_run()
    pool = cov.by_stratum["may_mention"]
    survivors = set(cov.drawn["may_mention"][:5])
    cov.note_search("a-term-that-hits-nearly-everything",
                    [d.note_id for d in pool if d.note_id not in survivors])
    for d in pool:                                   # every hit read: no unread-hit excuse
        cov.note_read(d.note_id, d.doc_type)
    cov.resolve_sample_verdicts(cited=set())

    r = _stratum(cov, "may_mention")
    assert (r.misses, r.misses_sampled, r.hits_unread) == (5, 5, [])
    assert r.keyword_list_validated is True, "the surviving frame really was censused"
    assert cov.pending_samples().get("may_mention") is None, "there is nothing left to draw"
    assert r.elusion_upper == pytest.approx(0.4507, abs=1e-3)

    g = evaluate_gate(GATE, cov.stratum_results())
    assert g.verdict == "FAIL"
    assert any("elusion upper bound" in m for m in g.missing), g.missing


def test_an_unsearched_stratum_keeps_its_whole_frame():
    """The negative control. `validate_by_sampling` draws from the stratum, not from the
    misses, so a search cannot move a document out of its frame and no draw is invalidated —
    otherwise this fix would be charging every run for expansions that changed nothing."""
    cov = _honest_run()
    drawn = list(cov.drawn["cannot_establish"])
    cov.note_search("a-newly-added-term", drawn[:N_LEAVING_THE_FRAME])
    r = _stratum(cov, "cannot_establish")
    assert r.draws_invalidated == [] and r.sampled == N_DRAWN
    assert r.elusion_upper == pytest.approx(0.1129, abs=1e-4)
