"""Detect, mechanically and without asking a model anything, the observations a reflection
must answer for.

WHY DETECTION IS SEPARATE FROM THE LOOP THAT REACTS TO IT
---------------------------------------------------------
The old reflect node posed an open question — "does something learned change what should be
done next?" — whose default answer is no, and which was answered no 291 times running. The
fix was to stop asking and start deciding: these are facts read off a tool result or off the
gate's own list of misses, and the supervisor is shown them as things that HAPPENED rather
than as a question. Keeping the detectors here rather than inside `graph` is what stops the
loop from quietly acquiring a fifth "trigger" that is really a judgement call.

Nothing in this module decides validity. `detect_gate_obligations` READS `answer_gate`'s
misses to notice a structural deadlock; it never rules on them, and there is still exactly
one gate.

Both functions are pure with respect to the run's bookkeeping: they return triggers and
leave the counting, queuing and tracing to the caller, so the queue that feeds the next
reflection has one owner.
"""
from __future__ import annotations

from .answer_gate import check_gate
from .corpus import PatientChart
from .coverage import CoverageLedger
from .coverage_planner import (TRIGGER_UNSETTLED_THREAD, CoveragePlan, MarkerCatalogue,
                               OpenThreadLedger, Trigger, gate_obligation_triggers,
                               triggers_from_tool_result)
from .spec import ExtractionSpec


def detect_from_tool_result(name: str, args: dict, out: dict, *, step: int,
                            plan: CoveragePlan, markers: MarkerCatalogue,
                            threads: OpenThreadLedger) -> list[Trigger]:
    """Mechanical conditions, read off one tool result. Opens any thread it finds.

    Opening the thread is part of DETECTING it — a marker noticed and not written into the
    ledger is the 8046 error again, where the machinery to catch it existed as advice. The
    ledger's own de-duplication is what stops a re-read from multiplying the debt, and a
    trigger is only returned for a thread that was genuinely new.
    """
    found = triggers_from_tool_result(
        name, args, out if isinstance(out, dict) else {}, plan=plan,
        catalogue=markers, step=step,
        quote=str((out or {}).get("quote", "")) if name == "record_evidence" else "")
    detected: list[Trigger] = []
    for t in found:
        if t.kind == TRIGGER_UNSETTLED_THREAD:
            m = markers.by_text().get(t.marker)
            th = threads.open_thread(
                note_id=t.note_id, doc_type=t.doc_type, marker=t.marker,
                obligation=(m.obligation if m else "unsettled"), excerpt=t.observation,
                step=step)
            if th is None:
                continue        # already outstanding; re-reading must not multiply debt
        detected.append(t)
    return detected


def detect_gate_obligations(*, spec: ExtractionSpec, coverage: CoverageLedger,
                            chart: PatientChart, plan: CoveragePlan,
                            step: int) -> list[Trigger]:
    """The fourth trigger: an obligation the CURRENT plan structurally cannot discharge.

    A gate that says "read these search hits" while the plan says "you may not open that
    type" is a deadlock, not a rejection. The old loop would spend the rest of its budget
    in it, which is what a 400k-token run defending an interim answer looks like.
    """
    try:
        g = check_gate(spec, coverage, plan)
    except Exception:      # noqa: BLE001 - trigger detection may never break a run
        return []
    unread_types: list[str] = []
    for r in coverage.stratum_results():
        for nid in r.hits_unread:
            meta = chart._docs.get(nid)
            if meta:
                unread_types.append(meta.doc_type)
    return list(gate_obligation_triggers(g.missing, plan=plan,
                                         unread_hit_types=unread_types, step=step))
