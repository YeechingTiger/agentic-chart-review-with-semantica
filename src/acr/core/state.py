"""Run state: the plan, the evidence ledger, and the coverage ledger.

The two ledgers are the point of this module.

  EvidenceLedger  every claim the agent will finally make must be backed by a recorded
                  span in a specific note. The finalize step sees ONLY this ledger, not
                  the scratchpad, so the model cannot "remember" an uncited detail.

  CoverageLedger  an append-only record of what the agent actually did — which documents
                  it listed, which keywords it searched, which documents it read in full.
                  The spec's proof_obligation is checked against this, in code, before a
                  negative answer is allowed. Attestation is computed, never self-reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict


@dataclass
class Evidence:
    note_id: str
    doc_type: str
    date: str
    start: int
    end: int
    quote: str
    supports: str = ""          # which field / assertion this backs
    stance: Literal["supports", "contradicts"] = "supports"
    #: WHICH THING IN THE CHART this span is about — the specimen, the lesion, the procedure.
    #: Optional and empty by default, so every recorded run still loads. It exists because a
    #: flat span list cannot express "this quote is about specimen A and that one about
    #: specimen B", and "the right document, the wrong specimen" is a failure mode this repo
    #: has already named. `submit_answer` records `reported_lesion`; with both, the agreement
    #: between them is machine-checkable instead of something a reader has to notice.
    entity: str = ""
    #: A STABLE IDENTITY, assigned by the ledger, so a candidate has something to point at.
    #: `E1` / `E2` is already the numbering `render()` shows the model, so this is the
    #: identifier the model has been reading all along rather than a new one it has never seen
    #: — the `after_event` rule: a pointer to something unobservable gets invented, not resolved.
    #:
    #: Empty on an Evidence built by hand, which is every one in every recorded trace. Defaulted
    #: and last so those still load.
    evidence_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceLedger:
    items: list[Evidence] = field(default_factory=list)

    def add(self, e: Evidence) -> None:
        for x in self.items:  # de-duplicate identical spans
            # `entity` is IN the key: without it, one sentence quoted about two specimens
            # collapses to one item, and the collapse is silent — exactly the confusion the
            # field was added to make visible.
            if (x.note_id, x.start, x.end, x.supports, x.entity) == \
               (e.note_id, e.start, e.end, e.supports, e.entity):
                return
        # Numbered from the length AFTER the de-dup check, so a repeated span does not consume
        # an id. A gap in the numbering reads like a lost piece of evidence.
        e.evidence_id = f"E{len(self.items) + 1}"
        self.items.append(e)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.items]

    def cited_notes(self) -> set[str]:
        return {e.note_id for e in self.items}

    def render(self) -> str:
        if not self.items:
            return "(no evidence recorded)"
        out = []
        for i, e in enumerate(self.items, 1):
            mark = "" if e.stance == "supports" else "  [CONTRADICTS]"
            # Omitted entirely when empty rather than rendered as "-": an absent anchor is the
            # field going unused, and a placeholder on every line would train the model to read
            # past the one line where an anchor was actually recorded.
            ent = f"\n      entity:   {e.entity}" if e.entity else ""
            out.append(
                f"[E{i}]{mark} {e.note_id} ({e.doc_type}, {e.date}) chars {e.start}-{e.end}\n"
                f'      supports: {e.supports or "-"}{ent}\n'
                f'      "{e.quote.strip()[:400]}"'
            )
        return "\n".join(out)


# ======================================================================================
# THE CANDIDATE LEDGER
#
# WHAT WAS MISSING. `RunContext.submitted` is one overwritable dict, and `Evidence.supports` is
# a free-text string naming a field, not a pointer to anything. So a run that weighed two dates
# and dropped one left no record that there had been two — and STORE.390 has four conflict
# rules whose entire job is arbitrating between competing dates. The rules could only ever be
# followed in prose, and when they were not followed nothing could say which candidate went or
# why. It is also why "did it stop with an unresolved disagreement?" has no answer today, which
# is the one question a stopping policy exists to get right.
#
# THIS LEDGER REFUSES NOTHING. It records. The distinction is load-bearing in this tree: five
# deterministic content checks were removed after destroying 58 correct values against 21
# helps, 60 of 254 rejections refusing a tuple that was exactly the registry's, and both the
# coverage gate and the thread refusal were demoted for the same reason. Structural modules
# (make state observable) and decision modules (change or block the output) are different
# things, and the second must be built on top of a first that has been shown to be reliable.
# Nothing here touches `answer_gate`.
#
# WHY THE LINKS LIVE HERE and not as a column on `Evidence`: the relation is many-to-many. One
# sentence — "cytology 2010-06-12, suspicious" — is simultaneously the witness for candidate A
# and the thing candidate B has to beat. A scalar column can only tell the truth about one of
# them. `evidence_view()` renders the reverse index for a reader that wants it, derived rather
# than stored, so there is exactly one owner of the fact.
# ======================================================================================

#: Closed. A state nothing branches on reads, in a manifest, exactly like a state that was
#: evaluated and found not to apply.
#:
#: SELECTED is the one that was submitted. It is distinct from LEADING because "what the
#: reasoner currently favours" and "what the run actually sent" are the same thing right up
#: until they are not, and the gap between them is the interesting case.
CANDIDATE_STATES: tuple[str, ...] = ("ACTIVE", "LEADING", "REJECTED", "SELECTED")


@dataclass
class Candidate:
    """One value the record could defensibly support, and what stands for and against it."""

    candidate_id: str
    #: Same shape as `submit_answer`'s `value`: keyed by the contract's own field names. The one
    #: nested shape every recorded run fills reliably, and identical shape is what lets
    #: "was the submitted answer ever declared as a candidate" be an exact comparison.
    value: dict = field(default_factory=dict)
    status: str = "ACTIVE"
    #: THE ABSTENTION THIS CANDIDATE IS, when it is one. Normalised to the bare status token —
    #: `EVIDENCE_INSUFFICIENT`, not "EVIDENCE_INSUFFICIENT: no document establishes one" —
    #: because it is the IDENTITY of an abstention candidate and prose cannot be one.
    #:
    #: A live run made the case: the reasoner declared an abstention labelled with a whole
    #: sentence, the runtime then looked for the submitted status as a bare token, the two did
    #: not match, and one abstention became two candidates — the second stamped "submitted;
    #: never declared as a candidate". That stamp is the single most useful thing this ledger
    #: says (on SYNY04 it was TRUE and pointed at a real disagreement between the reasoner and
    #: the run), so it must not be manufactured by a difference in spelling.
    abstention: str = ""
    #: The model's short name for this reading, prose included. Matched against nothing —
    #: normalising identity does not mean discarding the reason.
    label: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    #: WHAT WOULD SETTLE THIS ONE against the others. The field that separates a real competing
    #: reading from a second value written down for form's sake — "we need more information" is
    #: not a discriminator, "whether the 2010-06-12 cytology qualifies once the biopsy confirms"
    #: is.
    unresolved_discriminators: tuple[str, ...] = ()
    confidence: float | None = None
    created_at_step: int = 0
    updated_at_step: int = 0
    rejection_reason: str = ""
    #: Every transition, with the step and the reason. On the candidate rather than only in the
    #: trace because manifests outlive traces, and "which candidate was dropped and why" has to
    #: be answerable from the manifest alone.
    state_history: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("supporting_evidence_ids", "contradicting_evidence_ids",
                  "unresolved_discriminators", "state_history"):
            d[k] = list(d[k])
        return d


_STATUS_TOKEN = None


def normalise_abstention(text: str) -> str:
    """The leading status token of an abstention, or the whole string when there is none.

    `"EVIDENCE_INSUFFICIENT: no document establishes one"` -> `EVIDENCE_INSUFFICIENT`. A
    SCREAMING_SNAKE run at the start is a status; anything else is left exactly as written,
    because guessing at an identity is how two different readings quietly become one.
    """
    global _STATUS_TOKEN
    if _STATUS_TOKEN is None:
        import re
        _STATUS_TOKEN = re.compile(r"^\s*([A-Z][A-Z0-9_]{3,})\b")
    m = _STATUS_TOKEN.match(str(text or ""))
    return m.group(1) if m else str(text or "").strip()


def _value_key(value: dict, label: str = "") -> str:
    """Identity by the LITERAL value, with no semantic folding.

    `20100612` and `2010-06-12` are two candidates here. Collapsing them is notation
    normalisation — a separate, already-pinned defect in this tree (`C34.9` versus `C341`) — and
    fixing it inside a new module is how a known problem gets hidden somewhere nobody looks
    for it.

    THE LABEL IS PART OF THE KEY WHEN THERE IS NO VALUE, because an abstention is a candidate
    with nothing in its value dict. Without this, "no qualifying witness found" and "the record
    does not go back far enough" — two different reports with two different remedies — merge
    into one candidate, which is the same collapse this function exists to refuse one level up.
    """
    import json
    if not value:
        return f"abstain:{normalise_abstention(label).lower()}"
    return json.dumps({str(k): str(v).strip() for k, v in sorted((value or {}).items())},
                      sort_keys=True, ensure_ascii=False)


@dataclass
class CandidateLedger:
    """The competing readings a run is holding, and the evidence for and against each."""

    candidates: list[Candidate] = field(default_factory=list)
    #: Discriminators that belong to the SET rather than to one candidate: what the run would
    #: have to find to choose at all.
    open_discriminators: tuple[str, ...] = ()
    #: Every mutation, in order. The Strategic Controller and the evidence graph both need the
    #: sequence, not only the end state.
    events: list[dict] = field(default_factory=list)

    # -- reads ---------------------------------------------------------------------
    def by_id(self, cid: str) -> Candidate:
        for c in self.candidates:
            if c.candidate_id == cid:
                return c
        raise KeyError(f"no candidate {cid!r}; declared: {[c.candidate_id for c in self.candidates]}")

    def leading(self) -> Candidate | None:
        return next((c for c in self.candidates if c.status == "LEADING"), None)

    def active(self) -> list[Candidate]:
        return [c for c in self.candidates if c.status in ("ACTIVE", "LEADING", "SELECTED")]

    def evidence_view(self) -> dict[str, dict]:
        """evidence_id -> which candidates it bears on. DERIVED, never stored.

        One owner for the relation. Two mutable copies of a link is the two-ledger failure this
        module's own docstring already records one layer up.
        """
        out: dict[str, dict] = {}
        for c in self.candidates:
            for e in c.supporting_evidence_ids:
                out.setdefault(e, {"supports_candidate_ids": [], "contradicts_candidate_ids": []})
                out[e]["supports_candidate_ids"].append(c.candidate_id)
            for e in c.contradicting_evidence_ids:
                out.setdefault(e, {"supports_candidate_ids": [], "contradicts_candidate_ids": []})
                out[e]["contradicts_candidate_ids"].append(c.candidate_id)
        return out

    # -- writes --------------------------------------------------------------------
    def declare(self, value: dict, *, step: int, state: str = "ACTIVE", label: str = "",
                abstention: str = "", confidence: float | None = None) -> Candidate:
        """Create, or update the existing candidate with this literal value.

        An abstention candidate carries no value, so its identity is `abstention` normalised to
        its status token — see `normalise_abstention`. Its prose goes to `label` and is kept.
        """
        if state not in CANDIDATE_STATES:
            raise ValueError(f"candidate state {state!r} is not one of {CANDIDATE_STATES}")
        abst = normalise_abstention(abstention) if abstention else ""
        label = label or (abstention if abstention else "")
        key = _value_key(value, abstention or label)
        for c in self.candidates:
            if _value_key(c.value, c.abstention or c.label) == key:
                if label and len(label) > len(c.label):
                    c.label = label                # keep the fuller statement of the reason
                if abst:
                    c.abstention = abst
                if confidence is not None:
                    c.confidence = confidence
                if state != c.status:
                    self.set_state(c.candidate_id, state, step=step)
                else:
                    c.updated_at_step = step
                return c
        # Minted by the runtime, never supplied. A model-invented id is re-invented differently
        # next turn and every pointer to it stops resolving.
        c = Candidate(candidate_id=f"C{len(self.candidates) + 1}", value=dict(value or {}),
                      status=state, label=label, abstention=abst, confidence=confidence,
                      created_at_step=step, updated_at_step=step)
        self.candidates.append(c)
        self._emit("candidate_declared", candidate_id=c.candidate_id, value=c.value,
                   state=state, step=step)
        if state == "LEADING":
            self._demote_other_leaders(c.candidate_id, step)
        return c

    def set_state(self, cid: str, state: str, *, step: int, reason: str = "") -> Candidate:
        if state not in CANDIDATE_STATES:
            raise ValueError(f"candidate state {state!r} is not one of {CANDIDATE_STATES}")
        c = self.by_id(cid)
        if c.status == state:
            c.updated_at_step = step
            return c
        c.state_history = (*c.state_history,
                           {"step": step, "from": c.status, "to": state, "reason": reason})
        c.status = state
        c.updated_at_step = step
        if state == "REJECTED" and reason:
            c.rejection_reason = reason
        self._emit("candidate_state_changed", candidate_id=cid, to=state, reason=reason, step=step)
        if state == "LEADING":
            self._demote_other_leaders(cid, step)
        return c

    def link(self, cid: str, evidence_id: str, role: str, *, step: int) -> None:
        """One span, one role, per candidate. THE ROLES ARE MUTUALLY EXCLUSIVE.

        A live run on 2026-08-03 put E4 in both `for` and `against` on the same candidate. That
        state means nothing: a sentence is either a witness for a reading or a reason against
        it, and holding both makes the grounding count double the same span while a reader
        cannot tell which way it was meant.

        A LATER ROLE WINS AND THE MOVE IS RECORDED. "I first read this as support and now read
        it as contradiction" is a legitimate revision, and dropping the earlier link silently
        would lose the fact that the reading changed — which is exactly the kind of thing the
        Strategic Controller will need to see.
        """
        if role not in ("supports", "contradicts"):
            raise ValueError(f"evidence role {role!r} is not 'supports' or 'contradicts'")
        c = self.by_id(cid)
        attr = "supporting_evidence_ids" if role == "supports" else "contradicting_evidence_ids"
        other = "contradicting_evidence_ids" if role == "supports" else "supporting_evidence_ids"
        cur = getattr(c, attr)
        if evidence_id in cur:
            return
        if evidence_id in getattr(c, other):
            setattr(c, other, tuple(x for x in getattr(c, other) if x != evidence_id))
            self._emit("candidate_evidence_rerole", candidate_id=cid, evidence_id=evidence_id,
                       to=role, step=step)
        setattr(c, attr, (*cur, evidence_id))
        c.updated_at_step = step
        self._emit("candidate_evidence_linked", candidate_id=cid, evidence_id=evidence_id,
                   role=role, step=step)

    def set_discriminators(self, items, *, step: int, cid: str | None = None) -> None:
        vals = tuple(str(x).strip() for x in (items or []) if str(x).strip())
        if cid is None:
            self.open_discriminators = vals
        else:
            self.by_id(cid).unresolved_discriminators = vals
        self._emit("candidate_discriminators", candidate_id=cid, items=list(vals), step=step)

    def _demote_other_leaders(self, keep: str, step: int) -> None:
        """Exactly one LEADING. Two of them is an unwritten third state, not more information."""
        for c in self.candidates:
            if c.candidate_id != keep and c.status == "LEADING":
                c.state_history = (*c.state_history,
                                   {"step": step, "from": "LEADING", "to": "ACTIVE",
                                    "reason": f"{keep} took the lead"})
                c.status = "ACTIVE"
                c.updated_at_step = step

    def _emit(self, kind: str, **payload) -> None:
        self.events.append({"kind": kind, **payload})

    # -- reporting -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": "acr.core.state/candidates/1",
            "candidates": [c.to_dict() for c in self.candidates],
            "open_discriminators": list(self.open_discriminators),
            "evidence_view": self.evidence_view(),
            "n_declared": len(self.candidates),
            "n_active": len(self.active()),
            "n_rejected": sum(1 for c in self.candidates if c.status == "REJECTED"),
            "leading": (self.leading().candidate_id if self.leading() else None),
            "events": list(self.events),
        }

    def render(self) -> str:
        """For a reader, and later for the Strategic Controller. "" when empty.

        Empty rather than a placeholder: an empty block in the prompt on every run of every arm
        trains the model to read past the one run where it is populated.
        """
        if not self.candidates:
            return ""
        out = []
        for c in self.candidates:
            sup = ", ".join(c.supporting_evidence_ids) or "-"
            con = ", ".join(c.contradicting_evidence_ids) or "-"
            line = (f"[{c.candidate_id}] {c.status:<8} {c.value}"
                    f"{('  — ' + c.label) if c.label else ''}\n"
                    f"      for: {sup}   against: {con}")
            if c.unresolved_discriminators:
                line += "\n      needs: " + "; ".join(c.unresolved_discriminators)
            if c.rejection_reason:
                line += f"\n      rejected: {c.rejection_reason}"
            out.append(line)
        if self.open_discriminators:
            out.append("UNRESOLVED: " + "; ".join(self.open_discriminators))
        return "\n".join(out)


# CoverageLedger and the proof-obligation gate used to live here as a flat record
# (listed_documents / searched_terms / read_notes). They now live in `coverage.py`, where
# the stratified accounting, forced sampling and Clopper-Pearson bounds are — and there is
# deliberately only ONE of them. Two independent accounts of "how much was covered" can
# disagree, nothing raises when they do, and you are left with two numbers and no way to
# choose. Import from `acr.review.coverage`.


class PlanStep(TypedDict, total=False):
    id: str
    goal: str
    rationale: str
    status: Literal["pending", "active", "done", "dropped"]


class RunState(TypedDict, total=False):
    """LangGraph channel dict."""
    patient_id: str
    spec_id: str
    plan: list[PlanStep]
    plan_revisions: int
    messages: list[dict]
    step: int
    max_steps: int
    evidence: list[dict]
    coverage: dict
    reflection: dict
    answer: dict
    done: bool
    # Declared here or LangGraph drops it. A channel set by a node but absent from this
    # TypedDict is silently discarded, the downstream read returns the falsy default, and
    # nothing errors — which is how an accepted answer came out labelled UNGATED.
    gate_validated: bool
    rejections: list[dict]
    usage: dict


@dataclass
class Budget:
    max_steps: int = 24
    max_plan_revisions: int = 6
    max_tokens: int = 400_000
    max_seconds: int = 1200

    @property
    def report(self) -> dict:
        """The limits, for the manifest.

        `negative_basis: BUDGET_EXHAUSTED` without the numbers is not a finding a reader can
        act on: it does not say which of the three limits bound, and it does not say whether
        the limit was the default or something the operator chose. Seven real runs abstained
        on `max_tokens (400000)` while their step counts sat at 8-16 against a cap of 24, and
        the manifest recorded no budget at all — so the abstention read as a fact about the
        charts. It was a fact about this dataclass.
        """
        return {"max_steps": self.max_steps, "max_tokens": self.max_tokens,
                "max_seconds": self.max_seconds, "max_plan_revisions": self.max_plan_revisions,
                "is_library_default": self == Budget()}

    def exceeded(self, *, step: int, tokens: int, elapsed: float) -> str | None:
        if step >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if tokens >= self.max_tokens:
            return f"max_tokens ({self.max_tokens}) reached"
        if elapsed >= self.max_seconds:
            return f"max_seconds ({self.max_seconds}) reached"
        return None
