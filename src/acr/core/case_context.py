"""What a run is TOLD about the case, as distinct from the contract and from the chart.

Three layers were being conflated, and the middle one had nowhere to live:

    the CONTRACT   what the answer must mean. One per variable, reused across every patient.
    the CASE       which patient, which entity within that patient, over what stretch of
                   record, cut on what date. One per run, and none of it is derivable from
                   the contract.
    the CHART      the documents themselves.

Everything here is a fact about the second. It is `core` rather than `contract` because all
three working planes need it and none of them may import another: the runtime needs to know
which entity it is answering about, the evaluator needs to know what the run was told before
it can call an answer wrong, and the diagnosis needs both. A shared type is how they talk
about one case without importing each other's functions.

Nothing here reads a spec. The contract DECLARES what it requires of a case
(`acr.contract.case_requirements`); this holds what was supplied.

WHY THE WINDOW RAISES INSTEAD OF BEING IGNORED
----------------------------------------------
A time window narrows the record a run considers. For a target that is itself a date, the
only anchor available is the answer, and a window around the answer is circular — worse, on
this corpus it would cut off exactly the earlier clinical impression that decides SYNX05.
So a contract declares whether it is time-anchorable, and handing a window to one that is not
raises. Silently dropping it would leave the caller believing the run was scoped when it was
not, and a scope nobody applied is indistinguishable in the manifest from one that was.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class WindowNotAnchorableError(ValueError):
    """A time window was supplied for a contract whose target cannot anchor one."""


@dataclass(frozen=True)
class CaseContext:
    """One run's case. Frozen: what a run was told must not change while it runs."""

    patient_id: str

    #: WHICH entity the question is about, when the corpus holds more than one and the
    #: contract's question presupposes a choice ("this tumour", "the index admission"). None
    #: means none was supplied, which is a different thing from "there is only one" — see
    #: `acr.contract.case_requirements.refuse_before_reading`.
    target_entity: str | None = None

    #: WHAT BODY OF RECORD was in scope, in the caller's own words. Free text and read by
    #: nobody, on purpose: it is what a later reader needs in order to know whether "the chart
    #: does not say" was a statement about this institution's record or about all of them.
    corpus_scope: str = ""

    #: WHEN THE RECORD WAS CUT. Nothing after this date can be in the chart, so nothing after
    #: it can have been documented, so no answer may claim it. Optional — a caller that does
    #: not know must not be made to invent one.
    extract_date: date | None = None

    #: THE LAST DOCUMENT IN THIS PATIENT'S CHART. A tighter bound than `extract_date` and
    #: derivable without being told: a document cannot report something that has not happened
    #: yet, so a date after every document in the chart is unwitnessable by construction.
    latest_document_date: date | None = None

    #: Carried, and honoured only where the contract says a window means something.
    anchor_date: date | None = None
    window_days: tuple[int, int] | None = None

    def honour_window(self, *, time_anchorable: bool) -> tuple[date, date] | None:
        """The (from, to) this window resolves to, or None when there is no window.

        Raises when a window was supplied and the contract cannot anchor one. See the module
        docstring: the alternative is a scope the caller thinks was applied and was not.
        """
        if self.window_days is None:
            return None
        if not time_anchorable:
            raise WindowNotAnchorableError(
                f"a window of {self.window_days} days was supplied for a contract that "
                "declares `time_anchorable: false`. Its target is itself a point in time, so "
                "the only anchor available is the answer and the window would be circular. "
                "Drop the window, or answer a different question.")
        if self.anchor_date is None:
            raise WindowNotAnchorableError(
                "a window was supplied with no anchor_date to hang it on.")
        from datetime import timedelta
        before, after = self.window_days
        return (self.anchor_date - timedelta(days=int(before)),
                self.anchor_date + timedelta(days=int(after)))

    def unwitnessable_after(self) -> date | None:
        """The latest date any answer about this case could truthfully carry.

        The tighter of the two bounds. Both are honest and neither is always present, so
        `min` over what there is, and None when there is nothing — never a fabricated
        "today", which would make the bound a fact about when the check ran.
        """
        bounds = [d for d in (self.extract_date, self.latest_document_date) if d is not None]
        return min(bounds) if bounds else None

    def to_dict(self) -> dict:
        """For the manifest. A run that was told nothing must SAY it was told nothing."""
        return {
            "patient_id": self.patient_id,
            "target_entity": self.target_entity,
            "corpus_scope": self.corpus_scope,
            "extract_date": self.extract_date.isoformat() if self.extract_date else None,
            "latest_document_date": (self.latest_document_date.isoformat()
                                     if self.latest_document_date else None),
            "anchor_date": self.anchor_date.isoformat() if self.anchor_date else None,
            "window_days": list(self.window_days) if self.window_days else None,
        }
