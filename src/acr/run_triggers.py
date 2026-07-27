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
from .coverage_planner import (OPEN_REQUEST_OPENED, TRIGGER_UNSETTLED_THREAD, CoveragePlan,
                               MarkerCatalogue, OpenThreadLedger, Trigger,
                               gate_obligation_triggers, triggers_from_tool_result)
from .spec import ExtractionSpec
from .trace import Tracer


def detect_from_tool_result(name: str, args: dict, out: dict, *, step: int,
                            plan: CoveragePlan, markers: MarkerCatalogue,
                            threads: OpenThreadLedger) -> list[Trigger]:
    """Mechanical conditions, read off one tool result. Opens any thread it finds.

    Opening the thread is part of DETECTING it — a marker noticed and not written into the
    ledger is the 8046 error again, where the machinery to catch it existed as advice. The
    ledger's own de-duplication is what stops a re-read from multiplying the debt, and a
    trigger is only returned for a thread that was genuinely new.

    A TRIGGER IS OWED FOR `opened` AND FOR NOTHING ELSE, and the guard says so in those
    words. It used to say `if th is None: continue`, with the meaning of the None in a
    trailing comment; when `open_thread` started returning the existing thread for
    `already_open` this file kept compiling, kept passing, and turned
    `triggers_fired.UNSETTLED_THREAD` from a count of threads into a count of short reads —
    while announcing already-RESOLVED threads to the supervisor as live obligations. The
    status is now branched on by name, so a fifth outcome cannot be absorbed silently.
    """
    found = triggers_from_tool_result(
        name, args, out if isinstance(out, dict) else {}, plan=plan,
        catalogue=markers, step=step,
        quote=str((out or {}).get("quote", "")) if name == "record_evidence" else "")
    detected: list[Trigger] = []
    for t in found:
        if t.kind == TRIGGER_UNSETTLED_THREAD:
            m = markers.by_text().get(t.marker)
            request = threads.open_thread(
                note_id=t.note_id, doc_type=t.doc_type, marker=t.marker,
                obligation=(m.obligation if m else "unsettled"), excerpt=t.observation,
                step=step)
            if request.status != OPEN_REQUEST_OPENED:
                # The other three statuses are three different ways of owing nothing new:
                #   already_open        the obligation is already outstanding and already in
                #                       front of the supervisor; re-reading must not multiply
                #                       the debt, and must not re-announce it either;
                #   already_settled     the debt is paid. Announcing it again tells the run
                #                       it is blocked by something that is not blocking it;
                #   discharged_on_read  the runtime's own character count already settled it,
                #                       so no thread exists and none is owed.
                continue
        detected.append(t)
    return detected


def detect_gate_obligations(*, spec: ExtractionSpec, coverage: CoverageLedger,
                            chart: PatientChart, plan: CoveragePlan,
                            step: int, tracer: Tracer) -> list[Trigger]:
    """The fourth trigger: an obligation the CURRENT plan structurally cannot discharge.

    A gate that says "read these search hits" while the plan says "you may not open that
    type" is a deadlock, not a rejection. The old loop would spend the rest of its budget
    in it, which is what a 400k-token run defending an interim answer looks like.

    `tracer` IS REQUIRED, and that is the same lesson as the status above. This function
    swallows every exception and returns `[]`, which is the identical value it returns when
    the plan has no deadlock — so a detector that has silently stopped firing and a run with
    nothing to report are indistinguishable in the trace, and the trigger that is silent here
    is the one whose whole job is breaking deadlocks. An optional reporting channel is one a
    new call site forgets, so there is no default.
    """
    try:
        g = check_gate(spec, coverage, plan)
    except Exception as exc:      # noqa: BLE001 - trigger detection may never break a run
        # Returning [] is still right; returning it QUIETLY is not. The empty list keeps its
        # one meaning ("no obligation is unreachable") because the other meaning is written
        # somewhere a reader can find it.
        tracer.emit("gate_obligation_detection_failed", severity="error", step=step,
                    error=f"{type(exc).__name__}: {exc}",
                    message=("the fourth trigger could not be evaluated at this step; no "
                             "GATE_OBLIGATION_UNREACHABLE can fire from it, and an empty "
                             "result here does NOT mean the plan can discharge the gate"))
        return []
    unread_types: list[str] = []
    for r in coverage.stratum_results():
        for nid in r.hits_unread:
            meta = chart._docs.get(nid)
            if meta:
                unread_types.append(meta.doc_type)
    return list(gate_obligation_triggers(g.missing, plan=plan,
                                         unread_hit_types=unread_types, step=step))
