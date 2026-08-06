"""What the model left open must survive the run that left it open.

`write_todos` REPLACES the whole list on every call. That is what keeps the prompt honest — there
is no stale copy for the model to read — and it is exactly what makes the ledger disappear
afterwards, because no earlier version survives anywhere. A run that opens a gap at step three and
closes it at step nine finishes with an empty list, and an empty list is also what a run that never
looked produces. Same bytes, opposite meanings: the `not_considered` / `not_applicable` confusion
this tree keeps meeting, one plane over.

So the ledger is captured as it moves, and this file is the check that it still is. Two consumers,
two shapes:

    the trace     one `open_gaps` event per write, in order — which gaps opened, which closed, when
    the manifest  `open_gaps.final` and `open_gaps.n_writes` — the state at submission, and whether
                  the ledger was maintained at all

`n_writes` is the field that separates the two silences. `{"n_writes": 0, "final": {}}` is a run
that never used the ledger; `{"n_writes": 4, "final": {}}` is a run that worked through its gaps
and closed them. Keeping only the final state would report both as clean.

WHY THIS NEEDS A TEST AT ALL. `write_todos` is a library tool. It does not go through
`Toolbox.dispatch`, so none of this repo's tool tracing sees it; the capture lives in
`AuditMiddleware.wrap_tool_call`, which is the one hook that sees every call. Delete that branch
and nothing else fails — the run still answers, the trace still looks full, and the ledger is
simply gone. That is this repository's named defect: a check that cannot fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_trigger_pipeline import (  # noqa: E402
    ScriptedLLM,
    _assignments,
    _run,
)

GAPS = [
    {"content": "whether an oncology note records an impression at the cytology date",
     "status": "pending", "activeForm": "checking the cytology date"},
    {"content": "the biopsy date is established", "status": "completed",
     "activeForm": "establishing the biopsy date"},
]


@pytest.fixture(scope="module")
def spec():
    from acr.contract.spec import load_spec
    from acr.core import site
    return load_spec(site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml")


@pytest.fixture(scope="module")
def chart():
    from acr.chartstore.corpus import Corpus
    from acr.core import site
    return Corpus(site.corpus_root()).chart("SYN0001")


def _gap_events(events):
    return [e for e in events if e.get("kind") == "open_gaps"]


# ------------------------------------------------------------------ the trace


def test_a_write_reaches_the_trace(spec, chart, tmp_path):
    """The library tool bypasses `Toolbox.dispatch`, so without the hook this is silence."""
    llm = ScriptedLLM(acts=[("write_todos", {"todos": GAPS})],
                      assignments=_assignments(chart))
    _, _, events = _run(spec, chart, llm, tmp_path, "gap-trace", max_steps=3)

    (ev,) = _gap_events(events)
    assert ev["n_open"] == 1, "one pending entry is one open gap"
    assert ev["by_status"]["pending"] == [GAPS[0]["content"]]
    assert ev["by_status"]["completed"] == [GAPS[1]["content"]]
    assert ev["malformed"] is False


def test_every_write_is_its_own_event_so_the_ledger_can_be_watched_moving(spec, chart, tmp_path):
    """THE POINT OF THE WHOLE FILE. One event per write, in order, is what lets a reader see a
    gap open and then close — which the final state alone cannot show."""
    closed = [dict(GAPS[0], status="completed"), GAPS[1]]
    llm = ScriptedLLM(acts=[("write_todos", {"todos": GAPS}),
                            ("write_todos", {"todos": closed})],
                      assignments=_assignments(chart))
    _, manifest, events = _run(spec, chart, llm, tmp_path, "gap-moves", max_steps=4)

    evs = _gap_events(events)
    assert len(evs) == 2, "two writes, two events"
    assert [e["n_open"] for e in evs] == [1, 0], "the gap was open, then it was closed"
    assert manifest["open_gaps"]["n_writes"] == 2


# ------------------------------------------------------------------ the manifest


def test_the_manifest_carries_what_was_still_open_at_submission(spec, chart, tmp_path):
    llm = ScriptedLLM(acts=[("write_todos", {"todos": GAPS})],
                      assignments=_assignments(chart))
    _, manifest, _ = _run(spec, chart, llm, tmp_path, "gap-manifest", max_steps=3)

    gaps = manifest["open_gaps"]
    assert gaps["n_writes"] == 1
    assert gaps["final"]["pending"] == [GAPS[0]["content"]]


def test_a_run_that_never_used_the_ledger_is_distinguishable_from_one_that_emptied_it(
        spec, chart, tmp_path):
    """The two silences. Both end with nothing open; only `n_writes` says which happened, and
    conflating them would report a run that never looked as a run that found nothing."""
    never = ScriptedLLM(acts=[], assignments=_assignments(chart))
    _, m_never, _ = _run(spec, chart, never, tmp_path, "gap-never", max_steps=3)

    emptied = ScriptedLLM(acts=[("write_todos", {"todos": [dict(GAPS[0], status="completed")]})],
                          assignments=_assignments(chart))
    _, m_emptied, _ = _run(spec, chart, emptied, tmp_path, "gap-emptied", max_steps=3)

    assert m_never["open_gaps"] == {"n_writes": 0, "final": {}}
    assert m_emptied["open_gaps"]["n_writes"] == 1
    assert not m_emptied["open_gaps"]["final"].get("pending")


# ------------------------------------------------------------------ what it tolerates


def test_a_malformed_write_is_recorded_rather_than_dropped(spec, chart, tmp_path):
    """A write that does not match the expected shape is itself a finding. Dropping it would
    make a confused run look like a quiet one, which is the failure this file exists to prevent."""
    llm = ScriptedLLM(acts=[("write_todos", {"todos": ["just a string", {"status": "pending"}]})],
                      assignments=_assignments(chart))
    _, manifest, events = _run(spec, chart, llm, tmp_path, "gap-malformed", max_steps=3)

    (ev,) = _gap_events(events)
    assert ev["malformed"] is True, "one of the two entries was not an object"
    assert manifest["open_gaps"]["n_writes"] == 1, "the write still counts as a write"
