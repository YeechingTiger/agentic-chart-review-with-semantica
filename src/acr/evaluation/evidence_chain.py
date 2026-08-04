"""The causal chain: turning `because` from a sentence a human reads into a pointer that resolves.

THE PROBLEM
-----------
`because`'s schema says it plainly: *"Recorded, never checked; it is how a later reader tells your
reasoning from theirs."* So `detect_uncaused_reads` can only count **whether there is one**. A
fabricated reason and a true one look exactly alike in the record, and "why this step happened" is
what every conclusion in an attribution report rests on.

A set of annotations and a chain differ in exactly one thing: **whether the label is a resolvable
pointer at another artefact, by ID**. Trace events already carry `seq`, so the anchor already
exists and no new identifier scheme is invented here — `because` is merely allowed to be written as

    {"why": "<prose, kept>", "from": {"event": 14}}

The prose stays, because it is what tells a later reader the reasoning; the pointer is what is new,
because it is the only part that can be checked.

IT IS NOT A GATE
----------------
A pointer that fails to resolve **refuses nothing**. This repo has measured what it costs to turn a
judgement into a mechanical gate: five clinical checks, and 60 of 254 recorded rejections (24%)
refused a tuple that was exactly the registry's own answer. So what comes out of here is a report
and a few numbers, the same stance as `detect_uncaused_reads` — count it, hand it to the reader.

FIVE STATES, AND WHY NOT THREE
------------------------------
`PROSE_ONLY` has to be kept apart from both `GROUNDED` and `UNSOURCED`: folding it into the first
calls something checked that cannot be checked, and folding it into the second records every run
that ever wrote a careful reason as having written none. Every run recorded so far is prose-shaped.

`FORWARD_REF` has to be kept apart from `UNRESOLVED_REF`: pointing at the wrong step can be a typo,
but **pointing at an event that did not exist at the time cannot be a typo** — that is the shape
that only appears when the action came first and the reason was written afterwards, and it is the
one purely deterministic new detector this change buys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The pointer resolved: it names an event that really exists and that sits **before** this one.
GROUNDED = "GROUNDED"
#: A reason was written but no pointer given — including every string-shaped `because` on record.
#: Not a failure.
PROSE_ONLY = "PROSE_ONLY"
#: No `because` at all.
UNSOURCED = "UNSOURCED"
#: A pointer was given but does not resolve: the seq does not exist, or the label itself is broken.
UNRESOLVED_REF = "UNRESOLVED_REF"
#: Points at itself or at a later event. Impossible, because that event had not happened yet.
FORWARD_REF = "FORWARD_REF"

#: The kind of call allowed to carry a `because`. `kind == "tool"` is enough: which tools deserve an
#: explanation is a policy question, and this module only reports facts — narrowing the scope to
#: reads would make "why this search was issued" impossible to count.
_TOOL_KIND = "tool"

#: The ceiling on a chain walk. The backwards-only rule already rules cycles out, but the resolver
#: must not depend on another rule to avoid looping forever — one bad record should not hang the
#: evaluation.
_MAX_WALK = 1000


def _pointer(because: Any) -> int | None | str:
    """Pull the pointed-at seq out of a `because`.

    Returns an int (there is a pointer), None (no pointer, it is prose), or `UNRESOLVED_REF` (there
    is a pointer but it is broken). A broken label has to become a record rather than an exception:
    the evaluation runs over runs somebody else already produced, and those runs are not going to be
    run again because of a formatting error.
    """
    if not isinstance(because, Mapping):
        return None                                    # a string, or anything else — prose
    ref = because.get("from")
    if ref is None:
        return None
    if not isinstance(ref, Mapping):
        return UNRESOLVED_REF
    raw = ref.get("event")
    if raw is None:
        return UNRESOLVED_REF
    try:
        return int(raw)
    except (TypeError, ValueError):
        return UNRESOLVED_REF


def _as_seq(raw: Any) -> int | str:
    """The value of the flat `after_event`. A bad value takes the same route as a bad label: it
    becomes a record, not an exception."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return UNRESOLVED_REF


def _why(because: Any) -> str:
    if isinstance(because, Mapping):
        return str(because.get("why") or "")
    return str(because or "")


def chain_report(run) -> dict:
    """The causal-chain status of every tool call, plus the three numbers an evaluation can hang on.

    `run` is a `RunRecord`; only its trace is read.
    """
    events = [ev for ev in (run.trace or []) if ev.get("kind") == _TOOL_KIND]
    by_seq: dict[int, dict] = {}
    for ev in events:
        try:
            by_seq[int(ev.get("seq"))] = ev
        except (TypeError, ValueError):
            continue

    links: list[dict] = []
    for ev in events:
        try:
            seq = int(ev.get("seq"))
        except (TypeError, ValueError):
            seq = None
        because = ev.get("because")
        # The flat field wins. The nested shape is kept only so records produced by the first
        # version of the schema stay readable — that version measured 0 of 18 calls emitting one,
        # so there will not be much data in it, but being unable to read an old record is its own
        # kind of loss.
        flat = ev.get("after_event")
        target = _pointer(because) if flat is None else _as_seq(flat)

        if because is None or (isinstance(because, str) and not because.strip()):
            status, ref = UNSOURCED, None
        elif target is None:
            status, ref = PROSE_ONLY, None
        elif target == UNRESOLVED_REF:
            status, ref = UNRESOLVED_REF, None
        elif target not in by_seq:
            # Existence is judged first: calling a pointer at a seq that does not exist "later" is
            # pretending that seq exists.
            status, ref = UNRESOLVED_REF, target
        elif seq is not None and target >= seq:
            status, ref = FORWARD_REF, target
        else:
            status, ref = GROUNDED, target

        links.append({"seq": seq, "tool": str(ev.get("tool") or ""), "status": status,
                      "why": _why(because), "ref": ref,
                      "chain": _walk(seq, by_seq) if status == GROUNDED else
                               ([seq] if seq is not None else [])})

    n = len(links)
    counts = {s: sum(1 for x in links if x["status"] == s)
              for s in (GROUNDED, PROSE_ONLY, UNSOURCED, UNRESOLVED_REF, FORWARD_REF)}
    return {
        "links": links,
        "n_links": n,
        "n_grounded": counts[GROUNDED],
        "n_prose_only": counts[PROSE_ONLY],
        "n_unsourced": counts[UNSOURCED],
        "n_unresolved": counts[UNRESOLVED_REF],
        "n_forward": counts[FORWARD_REF],
        # None, not 0.0 — a run with no tool calls is not "grounded in nothing", it is a run with
        # no call to judge, and rendering the two as the same number is exactly the confusion this
        # report exists to remove.
        "grounding_ratio": (counts[GROUNDED] / n) if n else None,
        "max_depth": max((len(x["chain"]) - 1 for x in links), default=0),
    }


def _walk(seq: int | None, by_seq: dict[int, dict]) -> list[int]:
    """Walk back from `seq` along the pointers to the root, returning the seqs it passed through.

    Carries both a `seen` set and a hard ceiling: the backwards-only rule already makes a cycle
    impossible, but a bad record should not hang the evaluation.
    """
    out: list[int] = []
    seen: set[int] = set()
    cur = seq
    for _ in range(_MAX_WALK):
        if cur is None or cur in seen or cur not in by_seq:
            break
        out.append(cur)
        seen.add(cur)
        ev = by_seq[cur]
        flat = ev.get("after_event")
        nxt = _pointer(ev.get("because")) if flat is None else _as_seq(flat)
        if not isinstance(nxt, int) or nxt >= cur:
            break
        cur = nxt
    return out


def _claim_pointer(because: Any) -> tuple[str, int] | None | str:
    """Pull the anchor out of a claim's `because`: `("event", n)` or `("evidence", i)`.

    Both anchors have to be supported, because an attribution claim has two natural grounds: a step
    that was taken ("it never searched for the abbreviation") and a piece of evidence already cited
    ("the cited span does not mention behaviour"). Supporting only the first would force callers to
    cram an evidence claim into an event number.
    """
    if not isinstance(because, Mapping):
        return None
    ref = because.get("from")
    if ref is None:
        return None
    if not isinstance(ref, Mapping):
        return UNRESOLVED_REF
    for kind in ("event", "evidence"):
        if kind in ref:
            try:
                return (kind, int(ref[kind]))
            except (TypeError, ValueError):
                return UNRESOLVED_REF
    return UNRESOLVED_REF


def claim_report(claims: list[Mapping], run) -> dict:
    """The grounding status of every claim in a prose artefact.

    Takes a **generic** list of claims — each with at least `text`, optionally `because` — rather
    than the attribution report's own types. This module sits on the evaluation plane; making it
    import diagnosis's schema would weld the two planes together, and `tests/test_layering.py`
    forbids exactly that. Adapting is the caller's job.

    The states are identical to the call layer's, for identical reasons: `PROSE_ONLY` counts neither
    as checked nor as unwritten, and `grounding_ratio` puts only the pointers that actually resolved
    into the numerator.
    """
    by_seq = {}
    for ev in (run.trace or []):
        if ev.get("kind") != _TOOL_KIND:
            continue
        try:
            by_seq[int(ev.get("seq"))] = ev
        except (TypeError, ValueError):
            continue
    n_evidence = len(run.manifest.get("evidence") or [])

    out: list[dict] = []
    for c in claims:
        because = c.get("because")
        target = _claim_pointer(because)
        if because is None or (isinstance(because, str) and not because.strip()):
            status, ref = UNSOURCED, None
        elif target is None:
            status, ref = PROSE_ONLY, None
        elif target == UNRESOLVED_REF:
            status, ref = UNRESOLVED_REF, None
        else:
            kind, idx = target
            ok = (idx in by_seq) if kind == "event" else (0 <= idx < n_evidence)
            status, ref = (GROUNDED if ok else UNRESOLVED_REF), f"{kind}:{idx}"
        out.append({"text": str(c.get("text") or "")[:200], "status": status,
                    "why": _why(because), "ref": ref})

    n = len(out)
    grounded = sum(1 for c in out if c["status"] == GROUNDED)
    return {"claims": out, "n_claims": n, "n_grounded": grounded,
            "grounding_ratio": (grounded / n) if n else None}
