"""The Candidate Reasoner: one job, one call, no authority over anything else.

WHY THIS IS A SEPARATE CALL AND NOT A SKILL CARD
------------------------------------------------
It was tried as a card first, and the measurement is the reason this module exists.
`tactic-counterevidence` already says the target thing in plain words — "name the most
plausible alternative value explicitly" — and it was run paired against the same twelve charts
with the same seed on 2026-08-03:

    distinct values submitted per run     1.00 -> 1.00   (0 of 24 runs ever submitted two)
    reasoning text mentions an alternative  4/12 -> 1/12   (in 11% more words)

Asking the model to maintain a candidate space does not make it maintain one. It writes more
prose about the answer it already has. So the space is not invited here, it is CONSTRUCTED: a
call whose only job is to return candidate updates, whose output is a schema rather than a
paragraph, and which cannot do anything else because it is not given anything else to do.

WHAT IT MAY NOT DO, and each is a real temptation
--------------------------------------------------
  * call a chart tool — it is not given any
  * decide whether to continue or stop — that is the Strategic Controller, and it is not built
    yet ON PURPOSE: a controller reading an unreliable candidate state would make more elaborate
    decisions on worse information
  * generate search terms — that is the action proposer's job and a keyword here would be a
    retrieval policy hiding inside an answer module
  * submit the answer
  * change what a piece of evidence says

`apply_updates` is the only writer, and it touches the candidate ledger and nothing else. There
is a test that greps this module for the forbidden capabilities, because "must not" in a
docstring is a wish.

IT REFUSES NOTHING
------------------
No return value of this module can reject an answer, block a submission, or change a value. It
records a state that was previously implicit. This tree removed five deterministic content
checks after they destroyed 58 correct values against 21 helps; the lesson is about GATES, not
about structure, and the two were being conflated. A reasoner that fails, times out, or returns
nonsense costs the run nothing but the call — `reason()` returns an empty result and the failure
is recorded rather than raised.

PHASE A IS AN OBSERVER
----------------------
The ledger it maintains is NOT rendered back into the main loop's prompt. That keeps the search
flow of the candidate arm byte-identical to the baseline's apart from cost, so the first
question — is this state reliable? — is answerable before anything is allowed to act on it.
`render_to_loop` exists for the phase where the Strategic Controller consumes it.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..core.state import CANDIDATE_STATES, CandidateLedger, EvidenceLedger

#: The one tool the call is given. Forcing a tool call rather than parsing prose is the whole
#: mechanism: a schema the provider validates cannot degrade into a paragraph, which is what
#: the card version produced.
UPDATE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "update_candidates",
        "description": (
            "Record the current defensible candidate set. Call this exactly once. You are not "
            "answering the question and you are not deciding whether the review is finished."),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string",
                                       "enum": ["create", "update", "reject", "select_leading"]},
                            "candidate_id": {"type": "string", "description":
                                             "required for update / reject / select_leading; "
                                             "omit for create — the runtime assigns it"},
                            "value": {"type": "object", "description":
                                      "create only: object keyed by the contract's output field "
                                      "names, the same shape submit_answer takes"},
                            "abstention": {"type": "string", "description":
                                           "create only, INSTEAD of value: the abstention this "
                                           "candidate is, when the defensible reading is that no "
                                           "value can be given"},
                            "state": {"type": "string", "enum": ["ACTIVE", "LEADING"],
                                      "description":
                                      "create only: LEADING if this is the reading you currently "
                                      "favour. Say it here rather than in a second update — a "
                                      "candidate you have just created has no id yet, and one "
                                      "you invent for it will not resolve"},
                            "label": {"type": "string", "description": "a short name for this reading"},
                            "supports": {"type": "array", "items": {"type": "string"},
                                         "description": "evidence ids (E1, E2, ...) that back it"},
                            "contradicts": {"type": "array", "items": {"type": "string"},
                                            "description": "evidence ids that cut against it"},
                            "discriminators": {"type": "array", "items": {"type": "string"},
                                               "description":
                                               "what would settle THIS one against the others. "
                                               "'more information' is not a discriminator"},
                            "reason": {"type": "string", "description":
                                       "required for reject: why this reading is out, citing a "
                                       "rule id or an evidence id"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["action"],
                    },
                },
                "unresolved_discriminators": {
                    "type": "array", "items": {"type": "string"},
                    "description": ("what the record would have to contain for the choice BETWEEN "
                                    "candidates to be made. Empty when one candidate stands "
                                    "unopposed."),
                },
            },
            "required": ["candidate_updates"],
        },
    },
}

SYSTEM = """You maintain the CANDIDATE SET for a chart-review task. That is your only job.

You are given the task contract, the evidence recorded so far, and the candidate set as it
stands. Return the candidate set as it should now stand.

YOU ARE NOT ANSWERING THE QUESTION. Another component submits the answer. You are not deciding
whether enough has been read; another component decides that. You have no chart tools and you
may not ask for a search. Do not propose keywords. Do not say the review is complete.

WHAT A CANDIDATE IS: a value the recorded evidence could defensibly support, under this
contract's own rules. Three shapes, and only three:

  1. ONE candidate. The evidence points one way and nothing substantive cuts against it.
     Declare it, link its evidence, and stop. This is the common case.
  2. TWO OR MORE candidates. The contract's rules could be read to give different answers on
     the evidence in hand — a different date, a different source ordering, a different reading
     of what qualifies. Declare each, link what supports and what contradicts each, and say what
     would DISCRIMINATE them.
  3. NO candidate with support. Declare one candidate carrying the abstention that fits.

DO NOT MANUFACTURE ALTERNATIVES. A second candidate invented so the set looks considered is
worse than one candidate, because it makes a clear case read like a contested one. If the
evidence points one way, say so with one candidate.

A DISCRIMINATOR IS A SPECIFIC MISSING FACT. "Whether the 2010-06-12 cytology still qualifies
once the biopsy confirms" is a discriminator. "More information is needed" is not; leave it out
rather than write it.

EVERY CANDIDATE SHOULD CITE EVIDENCE by its id. If you are declaring a reading that nothing
recorded supports yet, say so in its label rather than attaching evidence that does not bear
on it."""


class ReasonerResult:
    """What one call produced, including the ways it can produce nothing."""

    __slots__ = ("discriminators", "error", "ok", "raw", "updates")

    def __init__(self, updates=None, discriminators=None, ok=True, error="", raw=None):
        self.updates = list(updates or [])
        self.discriminators = list(discriminators or [])
        self.ok = ok
        self.error = error
        self.raw = raw

    def to_dict(self) -> dict:
        return {"ok": self.ok, "error": self.error, "n_updates": len(self.updates),
                "discriminators": self.discriminators}


def build_messages(spec_block: str, evidence: EvidenceLedger,
                   ledger: CandidateLedger) -> list[dict]:
    """Exactly three inputs, and deliberately not a fourth.

    No chart text beyond the recorded spans, no document inventory, no search history. The
    reasoner cannot reach for a document, so giving it the inventory would only invite it to
    reason about retrieval — which is another component's job and would make its output
    unattributable to this one.
    """
    current = ledger.render() or "(no candidates yet)"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"# TASK CONTRACT\n\n{spec_block}\n\n"
            f"# EVIDENCE RECORDED SO FAR\n\n{evidence.render()}\n\n"
            f"# CANDIDATE SET AS IT STANDS\n\n{current}\n\n"
            "Return the candidate set as it should now stand.")},
    ]


def _extract(reply: Any) -> dict | None:
    """Pull the forced tool call out of whatever the provider seam handed back.

    Three shapes tolerated because three callers exist (LangChain message, litellm response,
    a plain dict from a test). A reasoner that raises on an unfamiliar envelope would take down
    a run over its own plumbing.
    """
    if reply is None:
        return None
    if isinstance(reply, dict) and ("candidate_updates" in reply or "arguments" in reply):
        return reply.get("arguments", reply) if "arguments" in reply else reply
    calls = getattr(reply, "tool_calls", None)
    if calls is None and isinstance(reply, dict):
        calls = reply.get("tool_calls")
    for c in (calls or []):
        args = c.get("args") if isinstance(c, dict) else getattr(c, "args", None)
        if args is None and isinstance(c, dict):
            args = (c.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(args, dict):
            return args
    return None


def reason(*, spec_block: str, evidence: EvidenceLedger, ledger: CandidateLedger,
           invoke: Callable[[list[dict], list[dict]], Any]) -> ReasonerResult:
    """One call. Never raises; a failure is a recorded result, not an exception.

    The run's answer does not depend on this succeeding, so a provider hiccup here must not be
    able to end a review. `ok=False` in the manifest is how a reader tells "the reasoner said
    there is one candidate" from "the reasoner never ran".
    """
    if not evidence.items:
        return ReasonerResult(ok=True, error="no evidence recorded yet")
    try:
        reply = invoke(build_messages(spec_block, evidence, ledger), [UPDATE_TOOL])
    except Exception as e:                        # noqa: BLE001 - see the docstring
        return ReasonerResult(ok=False, error=f"{type(e).__name__}: {e}")
    args = _extract(reply)
    if args is None:
        return ReasonerResult(ok=False, error="no update_candidates call in the reply", raw=reply)
    return ReasonerResult(updates=args.get("candidate_updates"),
                          discriminators=args.get("unresolved_discriminators"),
                          raw=args)


def apply_updates(ledger: CandidateLedger, result: ReasonerResult, *, step: int,
                  known_evidence_ids: set[str] | None = None) -> list[str]:
    """THE ONLY WRITER. Touches the candidate ledger and nothing else.

    Returns the list of things it refused to apply — an id that does not exist, an action it
    does not implement, a create with neither a value nor an abstention. Refused rather than
    guessed: a link silently dropped reads exactly like a link that was made, and a candidate
    invented from a malformed update is a fabricated candidate the metrics would then count.
    """
    rejected: list[str] = []
    for u in result.updates:
        if not isinstance(u, dict):
            rejected.append(f"not an object: {u!r}"); continue
        action = str(u.get("action") or "").strip()
        cid = str(u.get("candidate_id") or "").strip()
        try:
            if action == "create":
                value = u.get("value") if isinstance(u.get("value"), dict) else {}
                label = str(u.get("label") or "")
                abst = str(u.get("abstention") or "").strip()
                if not value and not abst:
                    rejected.append("create with neither value nor abstention"); continue
                state = str(u.get("state") or "ACTIVE").strip().upper()
                if state not in ("ACTIVE", "LEADING"):
                    state = "ACTIVE"
                c = ledger.declare(value, step=step, label=(abst or label), state=state,
                                   confidence=_num(u.get("confidence")))
                cid = c.candidate_id
            elif action in ("update", "reject", "select_leading"):
                if not cid:
                    rejected.append(f"{action} with no candidate_id"); continue
                ledger.by_id(cid)                      # raises KeyError if unknown
                if action == "reject":
                    ledger.set_state(cid, "REJECTED", step=step,
                                     reason=str(u.get("reason") or ""))
                elif action == "select_leading":
                    ledger.set_state(cid, "LEADING", step=step,
                                     reason=str(u.get("reason") or ""))
                if u.get("confidence") is not None:
                    ledger.by_id(cid).confidence = _num(u.get("confidence"))
                if u.get("label"):
                    ledger.by_id(cid).label = str(u["label"])
            else:
                rejected.append(f"unknown action {action!r}"); continue

            both = {str(x) for x in (u.get("supports") or [])} & \
                {str(x) for x in (u.get("contradicts") or [])}
            if both:
                rejected.append(f"{cid} lists {sorted(both)} as BOTH supporting and "
                                f"contradicting in one update; that is not a revision, it is "
                                f"two claims about one span")
            for role, key in (("supports", "supports"), ("contradicts", "contradicts")):
                for eid in (u.get(key) or []):
                    eid = str(eid).strip()
                    if eid in both:
                        continue
                    if known_evidence_ids is not None and eid not in known_evidence_ids:
                        rejected.append(f"{cid} cites {eid}, which is not a recorded span")
                        continue
                    ledger.link(cid, eid, role, step=step)
            if u.get("discriminators"):
                ledger.set_discriminators(u["discriminators"], step=step, cid=cid)
        except (KeyError, ValueError) as e:
            rejected.append(f"{action} {cid}: {e}")
    if result.discriminators:
        ledger.set_discriminators(result.discriminators, step=step)
    return rejected


def _num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


#: Re-exported so a caller does not have to reach into `core.state` for the vocabulary.
__all__ = [
    "CANDIDATE_STATES",
    "SYSTEM",
    "UPDATE_TOOL",
    "ReasonerResult",
    "apply_updates",
    "build_messages",
    "reason",
]
