"""L5: why a case is non-concordant. Four causes, and they must not become one number.

    A. Care gap            the guideline applied and was not followed   -> clinician / QI
    B. Documentation gap   it happened, the chart does not show it      -> health info mgmt
    C. Extraction error    we got the variable wrong                    -> us
    D. Justified exception the guideline did not apply                  -> nobody, correct care

Reporting a single "non-concordance rate" fuses four findings with four different owners and
four different remedies. B is filed against records management, C is filed against this
codebase, D is not a defect at all — and a rate that mixes them tells a clinician to change
behaviour that may already be correct.

A VERSUS B IS THE CRUX, AND ONLY THE COVERAGE LEDGER DECIDES IT
---------------------------------------------------------------
"Not documented" and "not done" are indistinguishable to any system that reasons over a
retrieved subset: top-k retrieval can never say *I looked everywhere that could bear on this
and it is absent*. A stratified coverage proof with an elusion bound can, and that is the
whole justification for the gate machinery in `coverage.py`. So B is assertable only when the
can_establish stratum was exhaustively reviewed, the elusion bound on everything unread is
satisfied, and the evidence is still absent. Without that proof the honest output is
CANNOT_DISTINGUISH, and `assert_cause_is_earned` refuses any other answer. Guessing is not
available: the guess would be right about half the time and wrong in a direction that blames
a clinician for a records problem.

Note what a proof still does not buy. It establishes absence from *this chart*, which is
exactly the documentation finding. It does not establish that the care never happened, so
this module never marks A eliminated — see `_standing_a`.

WHAT IS DETERMINISTIC HERE AND WHAT IS NOT
------------------------------------------
Elimination is code. Which causes the ledger rules out is a mechanical reading of counters
that the agent cannot address and must not be asked to weigh; delegating it reproduces the
circularity forced sampling exists to prevent — the thing under audit choosing its own
scope. What remains open goes to an agent plus `skills/non-concordance-triage`, over the
packet `prepare_case_packet` builds. Same split as everywhere else in this project:
enforced in code, advisory in skills, declarative in the spec.

NO GROUND TRUTH EXISTS FOR THIS LAYER
-------------------------------------
There is no label for "why". The registry limited dataset carries variable values, not
causes, so nothing here can be scored the way L2 extraction is scored against site and
histology. Validating L5 requires human adjudication on a stratified sample designed up
front, and that sample does not exist in this project. **No output of this module may be
reported as validated, and every serialised scaffold carries `validation_status` saying so.**
An unvalidated four-way split is still strictly better than a validated-looking single
number, because it names its own open questions instead of hiding them.

This module reads serialised L2 answers — `ChartReviewAgent.run()["answer"]` — so it needs no
chart, no LLM and no network, and it is testable from manifests alone.

AND SO THE INPUTS HAVE TO BE PINNED DOWN
----------------------------------------
Every standing above is a function of the extract it was computed from, which makes an
unpinned input a verdict anybody can choose. Measured on synthetic fixtures: one concord.json,
two extract.json files, `acr explain --extract <the other one>` — CANNOT_DISTINGUISH became
B_DOCUMENTATION_GAP SUPPORTED, exit 0, nothing printed, and explain.json recorded only a path
pointing at the swapped file. `resolve_bound_extract` is the answer: a concord.json names the
extract it scored, so the extract handed to explain must BE that one by content digest, and
an override that is not gets refused or branded. See its docstring for why content and not
filename.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The four causes. Kept as module constants because they are written into manifests and
# compared downstream; a typo'd string literal is a silently different cause.
A_CARE_GAP = "A_CARE_GAP"
B_DOCUMENTATION_GAP = "B_DOCUMENTATION_GAP"
C_EXTRACTION_ERROR = "C_EXTRACTION_ERROR"
D_JUSTIFIED_EXCEPTION = "D_JUSTIFIED_EXCEPTION"
CAUSES = (A_CARE_GAP, B_DOCUMENTATION_GAP, C_EXTRACTION_ERROR, D_JUSTIFIED_EXCEPTION)

CAUSE_OWNER = {
    A_CARE_GAP: "clinician / quality improvement",
    B_DOCUMENTATION_GAP: "health information management",
    C_EXTRACTION_ERROR: "us — the extraction layer",
    D_JUSTIFIED_EXCEPTION: "nobody — this is correct care",
}

SUPPORTED, ELIMINATED, OPEN = "SUPPORTED", "ELIMINATED", "OPEN"

CANNOT_DISTINGUISH = "CANNOT_DISTINGUISH"
SINGLE_CAUSE_REMAINS = "SINGLE_CAUSE_REMAINS"
OPEN_TO_ADJUDICATION = "OPEN_TO_ADJUDICATION"

# The three shipped specs that declare a gate all use 0.12; matching them keeps an L5 claim
# no stronger than the L2 gate that produced its input.
DEFAULT_MAX_ELUSION_UPPER = 0.12

NO_GROUND_TRUTH_NOTICE = (
    "L5 has NO ground truth in this project. There is no label for why a case is "
    "non-concordant, so this scaffold cannot be scored and must never be reported as "
    "validated. It requires human adjudication on a stratified sample designed up front. "
    "The deterministic eliminations are auditable; the cause finally chosen is not."
)


class ExplanationClaimError(AssertionError):
    """A cause was asserted that the deterministic scaffold did not leave available."""


# ---------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class VariableResult:
    """One L2 answer, as it appears in a run manifest.

    `output_field` names the single spec field the concordance rule actually read, for specs
    that emit several — STORE.400_522_523 returns primary_site, histology and behavior in one
    answer, and a rule about histology must not be told the variable is present because
    primary_site came back populated. Leave it None when the whole answer is the variable.
    """

    name: str
    status: str = ""
    value: dict = field(default_factory=dict)
    negative_basis: str | None = None
    proof_basis: str | None = None
    coverage_attested: dict | None = None
    evidence: list[dict] = field(default_factory=list)
    output_field: str | None = None

    @classmethod
    def from_answer(cls, name: str, answer: Mapping[str, Any],
                    output_field: str | None = None) -> "VariableResult":
        return cls(
            name=name,
            status=str(answer.get("status") or ""),
            value=dict(answer.get("value") or {}),
            negative_basis=answer.get("negative_basis"),
            proof_basis=answer.get("proof_basis"),
            coverage_attested=answer.get("coverage_attested"),
            evidence=list(answer.get("evidence") or []),
            output_field=output_field,
        )

    @property
    def is_absent(self) -> bool:
        """True when the chart yielded nothing for the thing the rule needed."""
        if self.output_field is not None:
            v = self.value.get(self.output_field)
            return v is None or str(v).strip() == ""
        return self.status != "FOUND"

    def coded(self) -> dict:
        if self.output_field:
            return {self.output_field: self.value.get(self.output_field)}
        return dict(self.value)


# ---------------------------------------------------------------------------- the proof
@dataclass(frozen=True)
class CoverageProof:
    """Whether one variable's absence is proven absence or merely unfound."""

    variable: str
    adequate: bool = False
    mode: str = "none"
    can_establish_complete: bool = False
    worst_elusion_upper: float = 1.0
    max_elusion_upper: float = DEFAULT_MAX_ELUSION_UPPER
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def assess_coverage_proof(vr: VariableResult, *,
                          max_elusion_upper: float = DEFAULT_MAX_ELUSION_UPPER) -> CoverageProof:
    """Read the attested ledger and decide whether it proves absence.

    Deliberately stricter than `evaluate_gate`, in two places, because a gate PASS is one
    control among several inside a run whereas this is the sole basis for a causal claim
    made to a clinician.

      * `mode` must be `stratified_exclusion`. `evaluate_gate` skips every stratum check when
        the stratum is absent (`s is None -> ok`), so a spec that declares no strata passes
        it vacuously. Measured: `runs/aprime_SYN0002__20260726T035724Z__183f3f3` carries
        `negative_basis: GATE_VALIDATED` with `mode: unstratified` and `strata: []`. That
        answer is a legitimate L2 abstention; it is not a documentation-gap proof, and
        trusting `negative_basis` alone would file it as one.
      * a stratum other than can_establish must exist to carry the bound. `evaluate_gate`
        maxes with `default=0.0`, which reads an empty set of samples as perfect coverage.
        Here nothing sampled means nothing bounded, so the bound stays 1.0 and fails.
    """
    missing: list[str] = []
    led = vr.coverage_attested or {}

    if not vr.is_absent:
        missing.append(f"{vr.name} is documented in the chart — there is no absence to prove")
    if vr.status != "EVIDENCE_INSUFFICIENT":
        missing.append(f"status is {vr.status or 'unset'!r}, not EVIDENCE_INSUFFICIENT")
    if vr.negative_basis != "GATE_VALIDATED":
        # AGENT_GAVE_UP and BUDGET_EXHAUSTED are the two other bases the finalizer emits.
        # Both mean the run stopped looking, which is the opposite of proving absence.
        missing.append(f"negative_basis is {vr.negative_basis!r}, not GATE_VALIDATED")
    if not led:
        missing.append("no coverage ledger travelled with the answer")
        return CoverageProof(vr.name, False, "none", False, 1.0, max_elusion_upper, missing)

    mode = str(led.get("mode") or "none")
    if mode != "stratified_exclusion":
        missing.append(f"coverage mode is {mode!r}; only a stratified exclusion proves absence")
    if not led.get("listed_documents"):
        missing.append("the run never listed the patient's documents")

    strata = list(led.get("strata") or [])
    ce = next((s for s in strata if s.get("name") == "can_establish"), None)
    complete = bool(ce and ce.get("complete"))
    if ce is None:
        missing.append("no can_establish stratum — nothing was declared able to settle this")
    elif not complete:
        missing.append(f"can_establish not exhaustively reviewed "
                       f"({ce.get('reviewed', 0)}/{ce.get('N', 0)})")

    others = [s for s in strata if s.get("name") != "can_establish"]
    if not others:
        worst = 1.0
        missing.append("no sampled stratum bounds the unread remainder")
    else:
        worst = max(_as_float(s.get("elusion_upper"), 1.0) for s in others)
        if worst > max_elusion_upper:
            missing.append(f"elusion upper bound {worst:.3f} exceeds {max_elusion_upper}")

    return CoverageProof(vr.name, not missing, mode, complete, worst, max_elusion_upper, missing)


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------- standings
@dataclass(frozen=True)
class CauseStanding:
    cause: str
    standing: str
    owner: str
    because: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _standing_b(absent: Sequence[VariableResult],
                proofs: Sequence[CoverageProof]) -> CauseStanding:
    """B is exactly as strong as the coverage proof, and never stronger."""
    if not absent:
        return CauseStanding(
            B_DOCUMENTATION_GAP, ELIMINATED, CAUSE_OWNER[B_DOCUMENTATION_GAP],
            [("every variable the rule read is documented in the chart; there is no "
              "absence to attribute to documentation")])
    unproven = [p for p in proofs if not p.adequate]
    if unproven:
        return CauseStanding(
            B_DOCUMENTATION_GAP, OPEN, CAUSE_OWNER[B_DOCUMENTATION_GAP],
            [f"{p.variable}: absence not proven — " + "; ".join(p.missing) for p in unproven])
    return CauseStanding(
        B_DOCUMENTATION_GAP, SUPPORTED, CAUSE_OWNER[B_DOCUMENTATION_GAP],
        [f"{p.variable}: can_establish exhaustively reviewed, elusion <= "
         f"{p.worst_elusion_upper:.3f} (cap {p.max_elusion_upper}), evidence still absent"
         for p in proofs])


def _standing_a(absent: Sequence[VariableResult], present: Sequence[VariableResult],
                b: CauseStanding, d: CauseStanding) -> CauseStanding:
    """A is never ELIMINATED here, and that is not an oversight.

    Nothing in a coverage ledger can prove care was delivered. The only thing that could is
    positive evidence of the recommended care — and that evidence would have made the case
    CONCORDANT at L4, so it cannot be present in a case this module ever sees.
    """
    because: list[str] = []
    if not absent:
        coded = ", ".join(f"{k}={val!r}" for v in present for k, val in v.coded().items())
        because.append("the chart documents every variable the rule read, and those "
                       f"documented values are what violate it: {coded}")
        standing = SUPPORTED
    elif b.standing == SUPPORTED:
        because.append("the absence is proven absent from THIS chart, which is the "
                       "documentation finding. It does not establish that the care never "
                       "happened — care delivered outside the reporting facility looks "
                       "identical. Separating A from a proven B needs data this project "
                       "does not have: claims, an HIE feed, or the patient.")
        standing = OPEN
    else:
        because.append("the driving variable is absent and the absence is not proven, so "
                       "A and B are indistinguishable — do not choose between them")
        standing = OPEN
    if d.standing == SUPPORTED:
        # Deliberately does not eliminate A. Whether a documented refusal or performance
        # status actually covers THIS recommendation and THIS episode is a scope judgement,
        # and scope judgements are the skill's job. Recording the tension is code's job.
        because.append("a documented exception is on the record (see D); if an adjudicator "
                       "accepts its scope, this is not a care gap")
    return CauseStanding(A_CARE_GAP, standing, CAUSE_OWNER[A_CARE_GAP], because)


def _standing_c(driving: Sequence[VariableResult],
                truth: Mapping[str, Mapping[str, Any] | None] | None) -> CauseStanding:
    """Checkable only on the subset where registry truth exists. Elsewhere it stays live.

    Silently eliminating C off the 80% of patients with no registry record would turn a
    coverage limitation into a clean bill of health for the extraction layer.
    """
    own = CAUSE_OWNER[C_EXTRACTION_ERROR]
    if truth is None:
        return CauseStanding(C_EXTRACTION_ERROR, OPEN, own,
                             [("no registry ground truth was supplied for this case; "
                               "extraction error cannot be ruled out")])
    disagreements: list[str] = []
    untruthed: list[str] = []
    for v in driving:
        t = truth.get(v.name)
        if not t:
            untruthed.append(v.name)
            continue
        ours = v.coded()
        for k, theirs in t.items():
            if k not in ours:
                continue
            if _norm_code(ours.get(k)) != _norm_code(theirs):
                disagreements.append(f"{v.name}.{k}: we coded {ours.get(k)!r}, "
                                     f"registry has {theirs!r}")
    if disagreements:
        return CauseStanding(
            C_EXTRACTION_ERROR, SUPPORTED, own,
            disagreements + [
                ("registry disagreement routes this to us and does NOT prove the extraction "
                 "wrong — measured 2026-07-26, the pathologist wrote 'best classified as "
                 "non-small cell carcinoma, NOS' and the registrar coded squamous from a "
                 "hedged 'favor'. Adjudicate the chart, not the registry.")])
    if untruthed:
        return CauseStanding(C_EXTRACTION_ERROR, OPEN, own,
                             [(f"no registry ground truth for {untruthed}; extraction error "
                               "remains a live possibility for those variables")])
    return CauseStanding(C_EXTRACTION_ERROR, ELIMINATED, own,
                         [("registry ground truth exists for every driving variable and "
                           "agrees with what we extracted")])


def _standing_d(exceptions: Sequence[VariableResult],
                proofs: Sequence[CoverageProof]) -> CauseStanding:
    """A justified exception needs its own evidence, held to the primary variable's standard.

    Counting a patient who declined chemotherapy as a care gap is the classic botch, and the
    fix is not to accept any mention of refusal — it is to demand a witness-proved finding,
    and to demand a coverage proof before declaring the exception catalogue empty.
    """
    own = CAUSE_OWNER[D_JUSTIFIED_EXCEPTION]
    witnessed = [e for e in exceptions
                 if e.status == "FOUND" and e.evidence and e.proof_basis == "WITNESS"]
    if witnessed:
        return CauseStanding(D_JUSTIFIED_EXCEPTION, SUPPORTED, own,
                             [f"{e.name}: witness-proved with {len(e.evidence)} cited span(s)"
                              for e in witnessed])
    if not exceptions:
        return CauseStanding(D_JUSTIFIED_EXCEPTION, OPEN, own,
                             [("the exception catalogue was never queried; absence of a "
                               "documented exception has not been established")])
    ungated = [e.name for e in exceptions if e.status == "FOUND"]
    unproven = [p for p in proofs if not p.adequate]
    if unproven or ungated:
        because = [f"{p.variable}: exception absence not proven — " + "; ".join(p.missing)
                   for p in unproven]
        because += [f"{n}: claimed FOUND but never passed the witness gate, so it cannot "
                    "carry an exception" for n in ungated]
        return CauseStanding(D_JUSTIFIED_EXCEPTION, OPEN, own, because)
    return CauseStanding(D_JUSTIFIED_EXCEPTION, ELIMINATED, own,
                         [f"{p.variable}: proven absent under the same coverage standard as "
                          f"the primary variable (elusion <= {p.worst_elusion_upper:.3f})"
                          for p in proofs])


def _norm_code(v: Any) -> str:
    return str(v if v is not None else "").strip().upper()


# ---------------------------------------------------------------------------- scaffold
@dataclass(frozen=True)
class ExplanationScaffold:
    case_id: str
    recommendation_id: str
    verdict: str
    causes: list[CauseStanding]
    proofs: list[CoverageProof]
    packet: dict = field(default_factory=dict)

    def standing(self, cause: str) -> str:
        return next(c.standing for c in self.causes if c.cause == cause)

    @property
    def open_causes(self) -> list[str]:
        return [c.cause for c in self.causes if c.standing == OPEN]

    @property
    def eliminated_causes(self) -> list[str]:
        return [c.cause for c in self.causes if c.standing == ELIMINATED]

    @property
    def available_causes(self) -> list[str]:
        """What an adjudicator is still allowed to choose from."""
        return [c.cause for c in self.causes if c.standing != ELIMINATED]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "recommendation_id": self.recommendation_id,
            "verdict": self.verdict,
            "causes": [c.to_dict() for c in self.causes],
            "coverage_proofs": [p.to_dict() for p in self.proofs],
            "available_causes": self.available_causes,
            "packet": self.packet,
            "validation_status": NO_GROUND_TRUTH_NOTICE,
        }

    def render(self) -> str:
        lines = [f"{self.case_id} / {self.recommendation_id}: {self.verdict}"]
        for c in self.causes:
            lines.append(f"  [{c.standing:<10}] {c.cause}  -> {c.owner}")
            lines.extend(f"       {b}" for b in c.because)
        return "\n".join(lines)


def scaffold_explanation(
    *,
    case_id: str,
    recommendation_id: str,
    concordance: str,
    driving_variables: Sequence[VariableResult],
    registry_truth: Mapping[str, Mapping[str, Any] | None] | None = None,
    exception_results: Sequence[VariableResult] = (),
    max_elusion_upper: float = DEFAULT_MAX_ELUSION_UPPER,
) -> ExplanationScaffold:
    """Eliminate what the ledger eliminates; hand the rest to an agent with the triage skill.

    `driving_variables` are the variables the L4 rule actually read to reach
    NON_CONCORDANT, and the caller must include the ones whose ABSENCE drove it, not only
    the ones that came back populated. A rule that reports only its present inputs makes B
    look eliminated when the whole question was what is missing.
    """
    if concordance != "NON_CONCORDANT":
        # NOT_ASSESSABLE is a first-class outcome, not a quiet non-concordance. Explaining it
        # would fold cases whose inputs are EVIDENCE_INSUFFICIENT into a rate they cannot be
        # scored in, which is how concordance rates get inflated or deflated.
        raise ValueError(f"explain runs only on NON_CONCORDANT cases, not {concordance!r}")
    if not driving_variables:
        raise ValueError("no driving variables supplied — without them nothing is decidable "
                         "and an empty scaffold would read as 'no cause found'")

    absent = [v for v in driving_variables if v.is_absent]
    present = [v for v in driving_variables if not v.is_absent]
    proofs = [assess_coverage_proof(v, max_elusion_upper=max_elusion_upper) for v in absent]
    exc_absent = [e for e in exception_results if e.is_absent]
    exc_proofs = [assess_coverage_proof(e, max_elusion_upper=max_elusion_upper)
                  for e in exc_absent]

    b = _standing_b(absent, proofs)
    d = _standing_d(exception_results, exc_proofs)
    a = _standing_a(absent, present, b, d)
    c = _standing_c(driving_variables, registry_truth)
    causes = [a, b, c, d]

    # The hard rule. A and B both open means the ledger did not separate them, and no amount
    # of reasoning over the same chart can — the missing information is not in the chart.
    if a.standing == OPEN and b.standing == OPEN:
        verdict = CANNOT_DISTINGUISH
    elif len([x for x in causes if x.standing != ELIMINATED]) == 1:
        verdict = SINGLE_CAUSE_REMAINS
    else:
        verdict = OPEN_TO_ADJUDICATION

    packet = prepare_case_packet(
        case_id=case_id, recommendation_id=recommendation_id, verdict=verdict,
        driving_variables=driving_variables, exception_results=exception_results,
        causes=causes, proofs=proofs + exc_proofs)
    return ExplanationScaffold(case_id, recommendation_id, verdict, causes,
                               proofs + exc_proofs, packet)


# ---------------------------------------------------------------------------- the packet
def prepare_case_packet(*, case_id: str, recommendation_id: str, verdict: str,
                        driving_variables: Sequence[VariableResult],
                        exception_results: Sequence[VariableResult],
                        causes: Sequence[CauseStanding],
                        proofs: Sequence[CoverageProof]) -> dict:
    """Everything the agent+skill needs, and the rules it may not overturn.

    `forbidden` travels inside the packet rather than sitting only in the SKILL.md, because a
    skill is advisory by construction — the model chooses whether to load it. The same
    sentences are then re-checked by `assert_cause_is_earned` on the way out, so a model that
    ignores them produces an error rather than a confident wrong owner.
    """
    forbidden = [
        "Do not report a single non-concordance rate. Report the four causes separately.",
        ("Do not overturn an ELIMINATED standing. Those were decided from the coverage "
         "ledger, which you cannot address and did not produce."),
        "Do not treat registry disagreement as proof that the extraction is wrong.",
        ("An exception (D) needs its own cited evidence, to the same standard as the "
         "primary variable. A passing mention is not documentation of a refusal."),
    ]
    if verdict == CANNOT_DISTINGUISH:
        forbidden.insert(0,
                         "Do NOT choose between A (care gap) and B (documentation gap). The "
                         "coverage proof that separates them is absent, so the answer is "
                         "CANNOT_DISTINGUISH plus a statement of what would settle it.")

    questions: list[str] = []
    by = {c.cause: c for c in causes}
    if verdict == CANNOT_DISTINGUISH:
        questions.append("What specifically is missing from the coverage proof, and is it "
                         "obtainable by rerunning extraction or not at all?")
    if by[A_CARE_GAP].standing == SUPPORTED:
        questions.append("The chart documents the departure. Is there an undocumented-but-"
                         "plausible reason a reviewer should look for before filing this "
                         "against a clinician?")
    if by[C_EXTRACTION_ERROR].standing != ELIMINATED:
        questions.append("Re-read the cited spans. Does the extracted value actually follow "
                         "from them, independent of what the registry says?")
    if by[D_JUSTIFIED_EXCEPTION].standing == OPEN:
        questions.append("Search the exception catalogue — performance status, comorbidity, "
                         "patient refusal, clinical trial enrolment, hospice, death before "
                         "the recommended interval — and cite evidence for each verdict.")

    return {
        "case_id": case_id,
        "recommendation_id": recommendation_id,
        "verdict": verdict,
        "causes": [c.to_dict() for c in causes],
        "variables": [_variable_packet(v) for v in driving_variables],
        "exceptions": [_variable_packet(v) for v in exception_results],
        "coverage_proofs": [p.to_dict() for p in proofs],
        "forbidden": forbidden,
        "questions": questions,
        "validation_status": NO_GROUND_TRUTH_NOTICE,
    }


def _variable_packet(v: VariableResult) -> dict:
    return {
        "name": v.name,
        "output_field": v.output_field,
        "status": v.status,
        "absent": v.is_absent,
        "value": v.coded(),
        "negative_basis": v.negative_basis,
        "proof_basis": v.proof_basis,
        "evidence": [{"note_id": e.get("note_id"), "doc_type": e.get("doc_type"),
                      "date": e.get("date"), "quote": str(e.get("quote") or "")[:400],
                      "supports": e.get("supports"), "stance": e.get("stance")}
                     for e in v.evidence],
        "coverage": _ledger_summary(v.coverage_attested),
    }


def _ledger_summary(led: Mapping[str, Any] | None) -> dict:
    """Counters only. The agent must not be handed the ledger as something to argue with."""
    if not led:
        return {"mode": "none"}
    return {
        "mode": led.get("mode"),
        "n_documents": (led.get("universe") or {}).get("n_documents"),
        "n_read": led.get("n_read"),
        "n_searches": len(led.get("searched_terms") or []),
        "strata": [{"name": s.get("name"), "N": s.get("N"), "reviewed": s.get("reviewed"),
                    "complete": s.get("complete"), "sampled": s.get("sampled"),
                    "misses_sampled": s.get("misses_sampled"),
                    "elusion_upper": s.get("elusion_upper")}
                   for s in (led.get("strata") or [])],
        "suspected_recognition_failures": len(led.get("suspected_recognition_failures") or []),
    }


# ---------------------------------------------------------------------------- enforcement
def assert_cause_is_earned(scaffold: ExplanationScaffold, chosen: str) -> None:
    """Refuse a cause the deterministic layer did not leave available.

    Sits where `assert_coverage_claim_is_earned` sits: at the point of emission, on the way
    out. The failure family it guards against is a model that read the packet, found the
    reasoning finely balanced, and picked the cause that made the case tidy. A tidy wrong
    owner costs a clinician a chart review they did not need, or lets a records gap go
    unfiled — and downstream it is indistinguishable from an adjudicated one.
    """
    if chosen == CANNOT_DISTINGUISH:
        return
    if chosen not in CAUSES:
        raise ExplanationClaimError(
            f"{chosen!r} is not one of the four causes {list(CAUSES)} nor CANNOT_DISTINGUISH")
    if scaffold.verdict == CANNOT_DISTINGUISH and chosen in (A_CARE_GAP, B_DOCUMENTATION_GAP):
        raise ExplanationClaimError(
            f"cannot choose {chosen} — without a gate-validated coverage proof a care gap "
            "and a documentation gap are the same observation. Report CANNOT_DISTINGUISH.")
    if scaffold.standing(chosen) == ELIMINATED:
        why = next(c.because for c in scaffold.causes if c.cause == chosen)
        raise ExplanationClaimError(
            f"cannot choose {chosen} — eliminated by the coverage ledger: {'; '.join(why)}")


# ------------------------------------------------- the artifacts a verdict was computed from
#: 64 bits of sha256 — the truncation `spec.py` already uses for a spec hash. Long enough that
#: two extracts in one project will not collide, short enough to print inside a refusal.
ARTIFACT_DIGEST_CHARS = 16

BOUND_BY_DIGEST = "bound_by_recorded_digest"
BOUND_BY_REFERENCE = "bound_by_reference_artifact"
UNBOUND = "unbound"

UNBOUND_NOTICE = (
    "UNBOUND: this explanation was NOT computed from the extract its concord.json names. "
    "Every standing here is a function of the inputs it was handed, so none of it may be "
    "counted, reported, or filed against an owner until it is rerun on the bound extract. ")

UNBOUND_FORBIDDEN = (
    "UNBOUND INPUTS — do not choose a cause at all. This packet was built from an extract "
    "that is not the one the concordance verdict was computed from, so the eliminations you "
    "are being asked to respect were derived from something else.")

#: The first three sentences are word for word what the command printed before binding existed,
#: and they stay that way: it is the sharpest reason in this layer and a test pins the wording.
#: With no ledgers every case reports CANNOT_DISTINGUISH, and a fabricated CANNOT_DISTINGUISH is
#: indistinguishable from an earned one. Only the routes out of it are new.
MISSING_EXTRACT = (
    "the extract this concord.json was computed from is missing: {src!r}. "
    "Pass --extract. Without the coverage ledgers no absence can be proved and "
    "every case would falsely report CANNOT_DISTINGUISH. A relocated copy of that same "
    "extract binds by content; anything else additionally needs --allow-unbound-extract "
    "and comes out stamped UNBOUND.")


class ArtifactBindingError(ValueError):
    """The extract handed to explain is not the one the concordance verdict was computed from."""


def artifact_digest(doc: Mapping[str, Any]) -> str:
    """Content identity of one artifact — insensitive to its filename and its formatting.

    Canonical JSON rather than the file's bytes, for a reason that decides which cases the
    check catches. Every stage here re-serialises what it reads, so a copy that went through
    a different indent or a `default=str` is the same evidence; a byte hash would call that
    tampering and reject the one override this command exists to serve. Sorted keys for the
    same reason — no stage promises to preserve key order. What it must and does catch is a
    single counter moving anywhere inside a coverage ledger, because that is what decides
    whether B is provable.
    """
    blob = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:ARTIFACT_DIGEST_CHARS]


@dataclass(frozen=True)
class ExtractBinding:
    """Which extract produced this explanation, and whether that was proved or asserted."""

    basis: str
    bound: bool
    recorded_path: str = ""
    used_path: str = ""
    expected_digest: str = ""
    actual_digest: str = ""
    overridden: bool = False
    relocated: bool = False
    absent_patients: list[str] = field(default_factory=list)
    because: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_bound_extract(
    concord: Mapping[str, Any], *,
    load: Callable[[str], Mapping[str, Any]],
    override_path: str = "",
    allow_unbound: bool = False,
) -> tuple[dict, ExtractBinding]:
    """Return the extract this concord.json was scored from, or refuse to guess.

    The check is on CONTENT, and both halves of that matter. A path proves nothing about what
    is at the end of it: `acr extract` writes extract.json into a run directory, a rerun
    overwrites it in place, and the path recorded in concord.json then names an artifact L4
    never saw — with no override involved at all. Conversely the legitimate use of --extract
    is relocation, where the content is right and only the absolute path died when the
    artifacts left the cluster; a filename check would reject exactly that and wave through
    the dangerous one.

    `load` is injected so schema validation stays in the CLI, where `_load_artifact` already
    refuses a document of the wrong schema, and this module stays free of typer.
    """
    recorded = str(concord.get("extract_input") or "")
    used = str(override_path or recorded)
    if not used or not Path(used).exists():
        raise ArtifactBindingError(MISSING_EXTRACT.format(src=used))

    extract = dict(load(used))
    actual = artifact_digest(extract)
    expected, basis, because = _expected_extract_digest(concord, recorded, load)
    overridden = bool(override_path)
    absent = _absent_patients(concord, extract)
    common = {"recorded_path": _resolved(recorded), "used_path": _resolved(used),
              "expected_digest": expected, "actual_digest": actual, "overridden": overridden,
              "absent_patients": absent}

    if expected and expected == actual:
        return extract, ExtractBinding(
            basis=basis, bound=True,
            # Same content under another name: the artifact moved, the evidence did not.
            relocated=overridden and _resolved(used) != _resolved(recorded),
            because=because, **common)

    if expected:
        because = [(f"the extract handed in is not the one concord scored: expected digest "
                    f"{expected}, got {actual}")] + because
    else:
        because = [f"nothing binds this extract (digest {actual}) to that concord.json"] + because
    if absent:
        # The quiet half of the same defect: explain looks patients up by id and falls back to
        # the concord-recorded value when the id is absent, which strips the coverage ledger
        # and turns the case into a CANNOT_DISTINGUISH nobody chose.
        because.append(f"{len(absent)} patient(s) scored in concord.json are not in this "
                       f"extract at all ({', '.join(absent[:5])}): their ledgers would go "
                       f"missing and every one of them would report CANNOT_DISTINGUISH")

    binding = ExtractBinding(basis=UNBOUND, bound=False, relocated=False, because=because,
                             **common)
    if not allow_unbound:
        raise ArtifactBindingError(_refusal(binding))
    return extract, binding


def _expected_extract_digest(concord: Mapping[str, Any], recorded: str,
                             load: Callable[[str], Mapping[str, Any]]) -> tuple[str, str, list]:
    """What the extract SHOULD digest to, and how strongly we know it.

    Two sources, in order of strength. If concord.json carries the digest of the extract it
    read, that is the real chain of custody and it survives the original file being deleted.
    It does not carry one today — `concord` records a path and a timestamp — so the fallback
    re-reads the artifact concord names and hashes it, with `extract_created_utc` closing the
    gap that the named file may have been overwritten since. That fallback is weaker by one
    link, and the fix belongs upstream in the concord command, which is not this file.
    """
    rec = str(concord.get("extract_digest") or "")
    if rec:
        return rec, BOUND_BY_DIGEST, ["concord.json records the digest of the extract it read"]
    if not recorded:
        return "", UNBOUND, [("that concord.json does not name the extract it was computed "
                              "from, so there is nothing to check this one against")]
    if not Path(recorded).exists():
        return "", UNBOUND, [(f"the extract concord names is gone ({recorded}), so its content "
                              f"cannot be recomputed and only the path is left to trust")]
    ref = load(recorded)
    stamp = str(concord.get("extract_created_utc") or "")
    got = str((ref or {}).get("created_utc") or "")
    if stamp and got != stamp:
        return "", UNBOUND, [(f"{recorded} is no longer the extract concord read: it says "
                              f"created_utc {got!r} and concord recorded {stamp!r} — an "
                              f"`acr extract` rerun overwrites that file in place")]
    return artifact_digest(ref), BOUND_BY_REFERENCE, [
        f"checked against the artifact concord names ({recorded})"]


def _absent_patients(concord: Mapping[str, Any], extract: Mapping[str, Any]) -> list[str]:
    have = {str(p.get("patient_id")) for p in (extract.get("patients") or [])}
    return [str(p.get("patient_id")) for p in (concord.get("patients") or [])
            if str(p.get("patient_id")) not in have]


def _resolved(p: str) -> str:
    return str(Path(p).resolve()) if p else ""


def _refusal(b: ExtractBinding) -> str:
    return ("refusing to explain from an unbound extract. " + ". ".join(b.because) + ". "
            f"concord.json names {b.recorded_path!r}; you handed {b.used_path!r}. L5 decides "
            "whether a non-concordance is a care gap, a documentation gap, an extraction "
            "error or a justified exception, and that verdict is a function of the coverage "
            "ledgers inside the extract — swap the extract and the case is filed against a "
            "different owner. Hand it the artifact concord scored, or pass "
            "--allow-unbound-extract to run anyway and have every scaffold stamped UNBOUND.")


def side_input_record(path: str, doc: Any) -> dict:
    """Identity of an artifact explain reads that no upstream stage names.

    The registry truth file is the other input that moves a cause standing: hand `--truth` a
    different file and C goes from OPEN to SUPPORTED or ELIMINATED. It cannot be BOUND the way
    the extract can, because nothing upstream records which truth snapshot was meant — so the
    honest thing available here is provenance, not verification. Recording the digest at least
    makes two runs that disagree about C attributable to the files that produced them, instead
    of leaving `registry_truth_supplied: true` as the entire audit trail.
    """
    if not path:
        return {"supplied": False, "path": "", "digest": ""}
    return {"supplied": True, "path": _resolved(path), "digest": artifact_digest(doc),
            "bound": False,
            "note": ("nothing upstream names the registry snapshot this should be, so this is "
                     "provenance only — it records which file produced the C standings, it "
                     "does not verify the file was the right one")}


def mark_binding(outdoc: dict, binding: ExtractBinding) -> dict:
    """Stamp the run, every scaffold, and every packet with the inputs they came from.

    All three, because they travel separately. A case row gets lifted into a spreadsheet and a
    packet gets handed to an adjudicating agent, and neither reader ever sees the top of the
    file — a stamp only there is a stamp only for the reader who least needs it. The packet
    gets it as a `forbidden` line as well, since that is the channel this layer already uses
    for instructions an agent may not talk itself out of.
    """
    rec = binding.to_dict()
    outdoc["extract_binding"] = rec
    outdoc["inputs_bound"] = binding.bound
    for case in outdoc.get("cases") or []:
        scaffold = case.get("scaffold")
        if not scaffold:
            continue        # an outcome reported as unexplainable has nothing to brand
        scaffold["extract_binding"] = rec
        if binding.bound:
            continue
        scaffold["validation_status"] = UNBOUND_NOTICE + str(scaffold.get("validation_status")
                                                             or "")
        packet = scaffold.get("packet")
        if isinstance(packet, dict):
            packet["validation_status"] = UNBOUND_NOTICE + str(packet.get("validation_status")
                                                               or "")
            packet["forbidden"] = [UNBOUND_FORBIDDEN] + list(packet.get("forbidden") or [])
    return outdoc
