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
                            "not_a_target_value": {"type": "boolean", "description":
                                                   "reject only: this value is in the evidence "
                                                   "but is not a candidate for THIS question — "
                                                   "the note's own date, a treatment date, a "
                                                   "date about another entity"},
                            "reason": {"type": "string", "description":
                                       "required for reject: why this reading is out, citing a "
                                       "rule id or an evidence id"},
                            "rejecting_rule": {"type": "string", "description":
                                               "reject only: the contract rule id that puts it "
                                               "out, e.g. conflict_rule.3. Omit if no declared "
                                               "rule decides it — an invented id is discarded"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["action"],
                    },
                },
                "unresolved_discriminators": {
                    "type": "array",
                    "description": ("REQUIRED when two or more candidates remain active. What "
                                    "would settle the choice between them — one entry per "
                                    "competing pair. Empty only when one candidate stands "
                                    "unopposed."),
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_a": {"type": "string", "description": "candidate id"},
                            "candidate_b": {"type": "string", "description": "candidate id"},
                            "unresolved_fact": {"type": "string", "description":
                                                "the specific thing that is not settled. NOT "
                                                "'more information is needed' — a fact someone "
                                                "could go and check"},
                            "evidence_needed": {"type": "string", "description":
                                                "what document or statement would settle it"},
                            "likely_source": {"type": "array", "items": {"type": "string"},
                                              "description":
                                              "where such a document would be, in general terms "
                                              "— a contemporaneous physician note, an outside "
                                              "record, a rule in the contract"},
                            "can_be_resolved_from_current_corpus": {"type": "boolean"},
                            "status": {"type": "string",
                                       "enum": ["UNRESOLVED", "ALREADY_RESOLVED",
                                                "UNRESOLVABLE_FROM_CORPUS", "SPEC_DEPENDENT"],
                                       "description":
                                       "default UNRESOLVED. Say ALREADY_RESOLVED if the "
                                       "recorded evidence settles it, UNRESOLVABLE_FROM_CORPUS "
                                       "if no document in this record could, SPEC_DEPENDENT if "
                                       "the contract itself does not say which reading wins"},
                        },
                        "required": ["unresolved_fact"],
                    },
                },
                "answerability": {
                    "type": "string",
                    "enum": ["VALUE_AVAILABLE", "EVIDENCE_INSUFFICIENT", "CORPUS_INSUFFICIENT"],
                    "description": ("whether the question is answerable AT ALL, which is a "
                                    "separate axis from which value wins. Say "
                                    "EVIDENCE_INSUFFICIENT when the chart holds documents but "
                                    "none establishes the answer, and CORPUS_INSUFFICIENT when "
                                    "the documents that would are not in this record."),
                },
            },
            "required": ["candidate_updates"],
        },
    },
}

SYSTEM = """You COMPARE candidates for a chart-review task. That is your only job.

The candidate set has already been seeded MECHANICALLY: every value in the recorded evidence
that is type-compatible with the target is already in the set, whether or not it is a plausible
answer. Your job is not to find values. Your job is to decide which of them are candidates for
THIS question, which are not, how they stand against each other, and what would settle it.

YOU ARE NOT ANSWERING THE QUESTION. Another component submits the answer. You are not deciding
whether enough has been read; another component decides that. You have no chart tools and you
may not ask for a search. Do not propose keywords. Do not say the review is complete.

WHAT TO DO WITH THE SEEDED SET, in order:

  1. REJECT what is not a target value. The set is deliberately over-inclusive. A note's own
     service date, a treatment date, a follow-up date, a date about a different entity — reject
     each with `not_a_target_value: true` and say which it is. REJECTING IS THE MAIN WORK.
     A rejection with a reason is a far better record than a value that was never listed.
  2. MERGE nothing silently. Two notations of one date are two entries; if they are the same
     reading, reject one and say it duplicates the other.
  3. For what remains, LINK the evidence: what supports each, and what cuts against it.
  4. If ONE candidate remains, select it as leading and stop.
  5. If TWO OR MORE remain, they are in genuine competition. You must give a DISCRIMINATOR for
     each competing pair.

A DISCRIMINATOR IS A FACT SOMEBODY COULD GO AND CHECK. "Whether a physician recorded a clinical
impression of cancer on or before 2010-05-17" is one. "More information is needed" is not — it
names nothing, and a component downstream has to act on this. Say which two candidates, what
exact fact is unsettled, what document would settle it, where such a document would be, and
whether this record could contain it.

ANSWERABILITY IS A SEPARATE AXIS from which value wins. If the evidence supports no value at
all, do not invent a candidate for the abstention — set `answerability` and leave the candidate
set as it is."""


class ReasonerResult:
    """What one call produced, including the ways it can produce nothing."""

    __slots__ = ("answerability", "discriminators", "error", "ok", "raw", "updates")

    def __init__(self, updates=None, discriminators=None, ok=True, error="", raw=None,
                 answerability=""):
        self.updates = list(updates or [])
        self.discriminators = list(discriminators or [])
        self.answerability = str(answerability or "")
        self.ok = ok
        self.error = error
        self.raw = raw

    def to_dict(self) -> dict:
        return {"ok": self.ok, "error": self.error, "n_updates": len(self.updates),
                "answerability": self.answerability,
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
    n_live = len([c for c in ledger.candidates if c.status in ("ACTIVE", "LEADING", "SELECTED")])
    ask = ("Reject what is not a target value for this question, link the evidence for what "
           "remains, and select the leading candidate."
           if n_live > 1 else
           "Confirm or reject what is there, and link its evidence.")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"# TASK CONTRACT\n\n{spec_block}\n\n"
            f"# EVIDENCE RECORDED SO FAR\n\n{evidence.render()}\n\n"
            f"# CANDIDATE SET, SEEDED FROM THE EVIDENCE\n\n{current}\n\n"
            f"{ask}")},
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
                          answerability=args.get("answerability"),
                          raw=args)


def apply_updates(ledger: CandidateLedger, result: ReasonerResult, *, step: int,
                  known_evidence_ids: set[str] | None = None, spec=None) -> list[str]:
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
                c = ledger.declare(value, step=step, label=label, abstention=abst,
                                   state=state, confidence=_num(u.get("confidence")))
                cid = c.candidate_id
            elif action in ("update", "reject", "select_leading"):
                if not cid:
                    rejected.append(f"{action} with no candidate_id"); continue
                ledger.by_id(cid)                      # raises KeyError if unknown
                if action == "reject":
                    ledger.set_state(cid, "REJECTED", step=step,
                                     reason=str(u.get("reason") or ""))
                    if u.get("not_a_target_value"):
                        ledger.by_id(cid).not_a_target_value = True
                    if u.get("rejecting_rule"):
                        # Recognised ids only. An invented rule id recorded as a rule is worse
                        # than none, because a reader checking the rejection would go looking
                        # for it.
                        #
                        # `parse_rule_citations(source, known)` takes an ITERABLE OF RULE IDS as
                        # its second argument and returns `(recognised, unknown)`. The first
                        # version of this block passed the SPEC there and indexed the result,
                        # so `set(known)` iterated a pydantic model and raised
                        # `TypeError: unhashable type: 'dict'` — killing 35 of 42 runs in a
                        # frozen evaluation whose numbers had already been reported. The tests
                        # did not catch it because every one of them called `apply_updates`
                        # with the default `spec=None`, which takes the other branch.
                        from ..contract.trace import parse_rule_citations, rule_catalog
                        if spec is not None:
                            ids = [r.rule_id for r in rule_catalog(spec)]
                            recognised, _unknown = parse_rule_citations(
                                str(u["rejecting_rule"]), ids)
                        else:
                            recognised = [str(u["rejecting_rule"])]
                        if recognised:
                            ledger.by_id(cid).rejecting_rule = str(recognised[0])
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
    for d in result.discriminators:
        if isinstance(d, dict):
            try:
                ledger.add_discriminator(d, step=step)
            except ValueError as e:
                rejected.append(f"discriminator: {e}")
            else:
                got = ledger.discriminators[-1]
                if not got["candidate_a"] and not got["candidate_b"]:
                    # INVARIANT 3, reported rather than silently kept. A discriminator that
                    # points at no candidate cannot drive an action, and a live run produced
                    # "NEW" as a reference on every one it wrote.
                    rejected.append(f"discriminator names no resolvable candidate: "
                                    f"{got['unresolved_fact'][:60]}")
        elif str(d).strip():
            # The prose form, still accepted so a provider that ignores the object schema does
            # not lose the content — but it lands in `open_discriminators`, not in the
            # structured list, so the two cannot be counted as one thing.
            ledger.set_discriminators([*ledger.open_discriminators, str(d)], step=step)
    if result.answerability:
        try:
            ledger.set_answerability(result.answerability, step=step, reason="reasoner")
        except ValueError as e:
            rejected.append(f"answerability: {e}")
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
