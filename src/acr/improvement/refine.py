"""§6b — ONE reflective optimizer over every text parameter the agent reads.

The spec and the skills are not two improvement projects. They are both text in the model's
context, so both are PARAMETERS of one optimizer, exactly as GEPA and TextGrad treat any text
node in a forward pass as something a textual gradient can reach. What differs between them
is not the machinery but WHO MAY UPDATE THEM — which is why the update policy is a field on
a registry row here, and not an `if` buried in whatever code happens to touch the file.

Four things live in this module and nothing else:

  1. `PARAMETER_REGISTRY` — the six text parameters, declared as DATA in one auditable place.
  2. `GradientRouter` — the §6b decision tree, whose first two cuts are not about text at all,
     plus the CITATION MASK that stops gradients pooling in the spec.
  3. The SPEC ASYMMETRY — FORM proposals are allowed, CONTENT verdicts escalate as questions
     and cannot be turned into an edit by any argument, flag or policy value.
  4. Batching and an acceptance plan with a PER-INSTANCE result shape.

TWO THINGS THIS MODULE REFUSES TO DO, and both refusals are the point.

  * It never proposes an edit to a spec RULE on the strength of the data. The spec defines
    what a correct answer is, so editing it edits the loss; loosening an evidence rule raises
    agreement with the answer key and teaches us nothing. Data can demonstrate that a passage
    is AMBIGUOUS or SILENT — properties of the text, decidable without reference to
    correctness. "This rule is wrong" is a question for a clinician, carrying the evidence.
  * It never accepts a verdict without its citation. A reflection model blames whichever text
    is easiest to rewrite, and prose is the most plastic thing in the system, so gradients
    pool in the spec by default — not because the spec is usually at fault but because it is
    usually the most editable. An uncited verdict returns UNRESOLVED rather than a guess.

NOTHING HERE CALLS A MODEL OR READS A CHART. The reflection call is a seam (`Reflector`);
the shipped implementations are a stub and a `NotImplementedError`. `FailureCase` refuses to
hold a real person_id, so a case assembled from the real corpus cannot be constructed at all.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..contract.answer_checks import check_answer


# ------------------------------------------------------------------------------- errors
class RefineError(ValueError):
    """Base for every refusal in this module."""


class UnknownParameterError(RefineError):
    """A gradient was routed at something that is not a registered parameter."""

    def __init__(self, pid: str, known: Sequence[str]):
        super().__init__(f"unknown text parameter {pid!r}. registered: {', '.join(known)}. "
                         f"A parameter nobody registered is a parameter nobody can audit.")


class ContentEscalationRequired(RefineError):
    """Someone tried to express a CONTENT gradient as an edit. There is no such path."""


class MissingThresholdError(RefineError):
    """A threshold arrived as None. Every threshold here is a required argument."""


class OverlappingSetsError(RefineError):
    """Diagnosis and validation (or test) share a case. Proposing and scoring on the same
    case is fitting, and it reports as a gain."""


class PhiInFailureCaseError(RefineError):
    """A real person_id reached a routing input. Pseudonymise upstream, not here."""


# --------------------------------------------------------------- 1. parameter registry
#: Who may update a parameter. The whole table is these four values; anything else means a
#: policy was invented at a call site instead of declared here.
AUTO_ON_CERTIFICATION = "AUTO_ON_CERTIFICATION"
AUTO_ON_HELDOUT_GAIN = "AUTO_ON_HELDOUT_GAIN"
CLINICIAN_SIGNS = "CLINICIAN_SIGNS"
ENGINEER = "ENGINEER"
UPDATE_POLICIES = (AUTO_ON_CERTIFICATION, AUTO_ON_HELDOUT_GAIN, CLINICIAN_SIGNS, ENGINEER)

#: Kinds whose blast radius a machine can count offline. Everything else is prose, and the
#: honest answer for prose is "not computable" written out, never an omitted field.
MECHANICAL_KINDS = ("answer_check_rule", "value_domain", "keyword_list")


@dataclass(frozen=True)
class TextParameter:
    """One text node the optimizer can reach: where it lives, and who may change it."""

    id: str
    file: str            # the file, as a repo-relative path or glob
    path_within: str     # the address inside that file — a YAML path, a heading, a symbol
    kind: str
    update_policy: str
    in_objective: bool   # True only where editing this edits the definition of "correct"
    why: str

    @property
    def mechanical(self) -> bool:
        return self.kind in MECHANICAL_KINDS


#: The six parameters of §6b, DATA in one place a reader can audit. Order follows the design
#: table. `in_objective` is the asymmetry: exactly one row carries it.
PARAMETER_REGISTRY: tuple[TextParameter, ...] = (
    TextParameter(
        id="keyword_list",
        file="assets/specs/*.yaml",
        path_within="proof_obligation.for_negative.strata[*].required_keywords",
        kind="keyword_list",
        update_policy=AUTO_ON_CERTIFICATION,
        in_objective=False,
        why="Retrieval only. It changes what the agent SEES, never what counts as correct, "
            "and a keyword's effect is priced by grep — so certification is affordable and "
            "the update can be automatic.",
    ),
    TextParameter(
        id="document_type_policy",
        file="assets/specs/*.yaml",
        path_within="proof_obligation.for_negative.strata[*].match.doc_type_matches",
        kind="doc_type_policy",
        update_policy=CLINICIAN_SIGNS,
        in_objective=False,
        why="It encodes ADMISSIBILITY — which document types may establish a field. Adding "
            "a type is a clinical claim about evidence, not a retrieval tweak.",
    ),
    TextParameter(
        id="skill",
        file="assets/skills/*/SKILL.md",
        path_within="(whole document)",
        kind="prose_procedure",
        update_policy=AUTO_ON_HELDOUT_GAIN,
        in_objective=False,
        why="A skill says how to do the job; the spec says what a right answer is. The "
            "target does not move when a skill changes, so held-out gain is a real test.",
    ),
    TextParameter(
        id="spec_rules",
        file="assets/specs/*.yaml",
        path_within="decision_rule | evidence_rules | conflict_rules | abstention",
        kind="prose_rule",
        update_policy=CLINICIAN_SIGNS,
        in_objective=True,
        why="THE ASYMMETRY. These sentences define what a correct answer is, so editing them "
            "edits the loss. Loosening an evidence rule raises agreement with the answer key "
            "and teaches us nothing. FORM gradients (ambiguous, silent) may become proposals; "
            "CONTENT gradients ('this rule is wrong') escalate as questions and never edits.",
    ),
    TextParameter(
        id="agent_system_prompt",
        file="src/acr/deep_runner.py, src/acr/tools/",
        path_within="system prompt and tool descriptions",
        kind="prose_procedure",
        update_policy=ENGINEER,
        in_objective=False,
        why="It is code, versioned and reviewed as code. A model rewriting the harness that "
            "runs it is a loop with no fixed point.",
    ),
    TextParameter(
        id="answer_check_rejection_messages",
        file="assets/specs/*.yaml",
        path_within="answer_checks[*].message",
        kind="rejection_message",
        update_policy=ENGINEER,
        in_objective=False,
        why="NOBODY HAS EVER LOOKED AT THESE. What a rejection says to the agent decides "
            "whether it can recover: one real run was rejected twice for coding 8046 over "
            "'favor squamous' and then burned a 400k-token budget without revising. That is "
            "at least as likely to be a bad rejection message as a bad spec. The message is "
            "not the check — the check's clinical content is a spec rule — so tuning the "
            "wording does not move the target, and this row is an engineer's.",
    ),
    TextParameter(
        # NOT one of the six in the design table. It is registered because the decision tree's
        # last leaf routes "mechanically decidable" here, and a destination the registry does
        # not know is a gradient with nowhere to land.
        id="answer_check_rule",
        file="assets/specs/*.yaml",
        path_within="answer_checks[*] (kind, nos_values, contradicted_by, required_searches)",
        kind="answer_check_rule",
        update_policy=CLINICIAN_SIGNS,
        in_objective=False,
        why="A check MECHANISES a rule the spec already states — the spec's own note says so — "
            "so adding one enforces the target rather than moving it. The moment a candidate "
            "check asserts something no spec sentence states, it is a SPEC_GAP and must be "
            "routed as one. Clinician signs because the phrase lists are clinical content; "
            "the blast radius, uniquely here, is computable offline.",
    ),
)

_BY_ID = {p.id: p for p in PARAMETER_REGISTRY}

#: The six rows of the §6b design table, as distinct from the one added above for the tree's
#: last leaf. Kept explicit so a reader can check the table against the code.
DESIGN_TABLE_PARAMETER_IDS = ("keyword_list", "document_type_policy", "skill", "spec_rules",
                              "agent_system_prompt", "answer_check_rejection_messages")


def get_parameter(pid: str) -> TextParameter:
    if pid not in _BY_ID:
        raise UnknownParameterError(pid, sorted(_BY_ID))
    return _BY_ID[pid]


def registry_invariants() -> None:
    """Assertions the registry must satisfy to be worth trusting. Called by the tests, and
    cheap enough to call at import time in a caller that wants the guarantee."""
    ids = [p.id for p in PARAMETER_REGISTRY]
    if len(set(ids)) != len(ids):
        raise RefineError(f"duplicate parameter ids: {ids}")
    for p in PARAMETER_REGISTRY:
        if p.update_policy not in UPDATE_POLICIES:
            raise RefineError(f"{p.id}: update_policy {p.update_policy!r} is not one of "
                              f"{UPDATE_POLICIES}")
        # An automatic policy on a parameter inside the objective is the failure this whole
        # module exists to prevent: the loop would optimise the definition of correctness.
        if p.in_objective and p.update_policy.startswith("AUTO"):
            raise RefineError(f"{p.id} is in the objective and may never update automatically")


# ------------------------------------------------------------------- 2. routing inputs
#: The real corpus's person_id shape. A pattern, deliberately not an example.
_PERSON_ID = re.compile(r"1168\d{12}")

RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
ANSWER_KEY_WRONG = "ANSWER_KEY_WRONG"
SPEC_GAP = "SPEC_GAP"
SPEC_AMBIGUITY = "SPEC_AMBIGUITY"
SPEC_ERROR = "SPEC_ERROR"
NOT_A_SPEC_FIX = "NOT_A_SPEC_FIX"
UNRESOLVED = "UNRESOLVED"

ADJUDICATED_KEY_CORRECT = "ADJUDICATED_KEY_CORRECT"
ADJUDICATED_KEY_WRONG = "ADJUDICATED_KEY_WRONG"
NOT_ADJUDICATED = "NOT_ADJUDICATED"

FORM = "FORM"
CONTENT = "CONTENT"

# Where a routed failure goes. Not a verdict — a verdict can have only one destination, and
# naming the destination separately is what lets a caller fan them out without re-deciding.
TO_RETRIEVAL_6C = "SEC_6C_RETRIEVAL"
TO_ADJUDICATED_OUT = "ADJUDICATED_OUT_OF_DENOMINATOR"
TO_PROPOSAL = "PROPOSAL"
TO_CLINICIAN_QUESTION = "CLINICIAN_QUESTION"
TO_UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class FailureCase:
    """One disagreement with the answer key, with the two facts the first two cuts need.

    `establishing_evidence_surfaced` and `answer_key_adjudication` are FACTS carried in, not
    model judgements: the first comes from the trace (was the establishing note ever put in
    front of the agent), the second from a human adjudication. Letting the reflection model
    decide either one would let it route its own gradient.
    """

    case_id: str
    spec_id: str
    field: str
    coded_value: str
    key_value: str
    establishing_evidence_surfaced: bool
    answer_key_adjudication: str
    invoked_rules: tuple[str, ...] = ()       # §6b prerequisite 1: which rule the agent used
    rejection_messages_seen: tuple[str, ...] = ()
    subgroup: str = "unassigned"

    def __post_init__(self) -> None:
        if _PERSON_ID.search(self.case_id):
            raise PhiInFailureCaseError(
                "case_id looks like a real person_id. Pseudonymise before routing; the map "
                "lives outside this tree.")
        if self.answer_key_adjudication not in (
                ADJUDICATED_KEY_CORRECT, ADJUDICATED_KEY_WRONG, NOT_ADJUDICATED):
            raise RefineError(f"unknown adjudication {self.answer_key_adjudication!r}")


@dataclass(frozen=True)
class ReflectionVerdict:
    """What the reflection model returns. UNTRUSTED until the citation mask has run.

    The citation slots are separate fields rather than free prose because the mask has to be
    able to check them mechanically. A verdict that reads well and cites nothing is exactly
    the output this design predicts and rejects.
    """

    verdict: str
    parameter_id: str | None
    rationale: str
    missing_sentence: str | None = None    # SPEC_GAP: the sentence that should exist
    quoted_passage: str | None = None      # SPEC_AMBIGUITY / SPEC_ERROR / NOT_A_SPEC_FIX
    readings: tuple[str, ...] = ()         # SPEC_AMBIGUITY: both readings, written out
    proposed_text: str | None = None


class Reflector(Protocol):
    """The one model call in this loop, kept behind a seam so nothing here can reach a chart."""

    def __call__(self, case: FailureCase, spec_text: str) -> ReflectionVerdict: ...


@dataclass(frozen=True)
class StubReflector:
    """Canned verdicts by case_id. The only reflector the tests are permitted to use."""

    canned: Mapping[str, ReflectionVerdict]

    def __call__(self, case: FailureCase, spec_text: str) -> ReflectionVerdict:
        if case.case_id not in self.canned:
            raise KeyError(f"no canned verdict for {case.case_id}")
        return self.canned[case.case_id]


def llm_reflector(*_a, **_kw) -> ReflectionVerdict:  # pragma: no cover - deliberately unbuilt
    """TODO(§6b): the real reflection call. NOT BUILT. Building it is not the same decision
    as pointing it at a corpus, and this module is only allowed to make the first one."""
    raise NotImplementedError(
        "the reflection call is a seam. Wire a client here, and note that doing so spends "
        "money on chart text — a decision this module does not make.")


# ---------------------------------------------------------------- the citation mask
def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def citation_defect(v: ReflectionVerdict, spec_text: str) -> str | None:
    """Return why this verdict's citation fails, or None if it holds.

    THIS IS THE GRADIENT MASK. Every message below is written AT THE REFLECTOR, because a
    rejection the recipient cannot act on is the failure mode the rejection-message parameter
    was registered for — and rejecting a verdict is the same act as rejecting an answer.
    """
    text = _norm(spec_text)
    if v.verdict == SPEC_GAP:
        if not _norm(v.missing_sentence):
            return ("SPEC_GAP without `missing_sentence`. Name the sentence that should exist "
                    "and does not. If you cannot write it, you have not found a gap.")
        if _norm(v.missing_sentence) in text:
            return ("SPEC_GAP claims a sentence is missing that is already present in the "
                    "spec. That makes this NOT_A_SPEC_FIX or SPEC_ERROR, not a gap.")
    elif v.verdict == SPEC_AMBIGUITY:
        if not _norm(v.quoted_passage):
            return "SPEC_AMBIGUITY without `quoted_passage`. Quote the passage you mean."
        if _norm(v.quoted_passage) not in text:
            return ("SPEC_AMBIGUITY quotes a passage that does not appear in the spec. Quote "
                    "the text verbatim; a paraphrase cannot be checked.")
        distinct = {_norm(r) for r in v.readings if _norm(r)}
        if len(distinct) < 2:
            return ("SPEC_AMBIGUITY needs BOTH readings written out. One reading is a "
                    "complaint; two readings is a demonstration that the text underdetermines "
                    "behaviour.")
    elif v.verdict in (SPEC_ERROR, NOT_A_SPEC_FIX):
        if not _norm(v.quoted_passage):
            return (f"{v.verdict} without `quoted_passage`. Quote the spec sentence that "
                    f"covers this case. Without it there is no way to tell 'the rule was "
                    f"there and ignored' from 'the rule was never written'.")
        if _norm(v.quoted_passage) not in text:
            return (f"{v.verdict} quotes a passage that does not appear in the spec. If the "
                    f"sentence you want to quote is not there, the verdict is SPEC_GAP.")
    return None


# ---------------------------------------------------------------------- 3. the routing
@dataclass(frozen=True)
class Routing:
    """The classification of one failure. `in_denominator=False` only for an adjudicated key."""

    case: FailureCase
    verdict: str
    destination: str
    parameter_id: str | None = None
    change_class: str | None = None      # FORM/CONTENT, and only for a parameter in the objective
    citation: Mapping[str, object] = field(default_factory=dict)
    rejected_reason: str | None = None
    in_denominator: bool = True
    reflection: ReflectionVerdict | None = None


def spec_change_class(verdict: str) -> str:
    """FORM or CONTENT, as a pure function of the verdict — never an argument.

    SPEC_GAP and SPEC_AMBIGUITY are properties of the TEXT: silent here, two readings there,
    both decidable without asking whether the rule is right. SPEC_ERROR is a claim about the
    world, and the data cannot make it. Making this derivable rather than settable is the
    enforcement: classifying CONTENT as FORM to get an edit through is the single most
    damaging thing this loop could do, because the result looks like an accuracy improvement.
    """
    if verdict in (SPEC_GAP, SPEC_AMBIGUITY):
        return FORM
    if verdict == SPEC_ERROR:
        return CONTENT
    raise RefineError(f"{verdict} is not a gradient at the spec; it has no FORM/CONTENT class")


class GradientRouter:
    """The §6b decision tree. One failure in, one `Routing` out."""

    def __init__(self, spec_texts: Mapping[str, str], reflector: Reflector):
        # Required, and not optional-with-a-fallback: without the spec text the mask can check
        # that a quote is PRESENT but not that it is TRUE, and a mask that cannot be false is
        # not a mask.
        self.spec_texts = dict(spec_texts)
        self.reflector = reflector

    def route(self, case: FailureCase) -> Routing:
        # Cut 1 — not about text at all. If the establishing evidence never reached the agent,
        # no sentence anywhere would have helped, and the spec is irrelevant. The reflector is
        # not even consulted: asking it would invite a spec verdict on a retrieval failure.
        if not case.establishing_evidence_surfaced:
            return Routing(case, RETRIEVAL_FAILURE, TO_RETRIEVAL_6C,
                           citation={"note": "establishing evidence never surfaced; §6c owns this"})

        # Cut 2 — is the answer key even right?
        if case.answer_key_adjudication == ADJUDICATED_KEY_WRONG:
            return Routing(case, ANSWER_KEY_WRONG, TO_ADJUDICATED_OUT, in_denominator=False,
                           citation={"note": "answer key adjudicated wrong; recorded, out of "
                                             "the denominator"})
        if case.answer_key_adjudication == NOT_ADJUDICATED:
            return Routing(case, UNRESOLVED, TO_UNRESOLVED,
                           rejected_reason="the answer key has not been adjudicated. Routing "
                                           "now would attribute a registry error to the text.")

        if case.spec_id not in self.spec_texts:
            raise RefineError(f"no spec text for {case.spec_id}; the citation mask cannot run")
        v = self.reflector(case, self.spec_texts[case.spec_id])

        defect = citation_defect(v, self.spec_texts[case.spec_id])
        if defect is not None:
            # Unresolved, NOT a guess. An uncited verdict is the default output of a reflection
            # model under pressure, and accepting it is how the spec fills with restatements.
            return Routing(case, UNRESOLVED, TO_UNRESOLVED, parameter_id=v.parameter_id,
                           rejected_reason=defect, reflection=v)

        citation = {"missing_sentence": v.missing_sentence,
                    "quoted_passage": v.quoted_passage,
                    "readings": list(v.readings),
                    "invoked_rules": list(case.invoked_rules)}

        if v.verdict in (SPEC_GAP, SPEC_AMBIGUITY, SPEC_ERROR):
            klass = spec_change_class(v.verdict)
            dest = TO_PROPOSAL if klass == FORM else TO_CLINICIAN_QUESTION
            return Routing(case, v.verdict, dest, parameter_id="spec_rules",
                           change_class=klass, citation=citation, reflection=v)

        if v.verdict == NOT_A_SPEC_FIX:
            # The leaf that keeps this honest. The rule was there and was broken anyway, so the
            # gradient goes to the skill or to a mechanical check — never to more spec prose.
            pid = v.parameter_id or "skill"
            get_parameter(pid)
            return Routing(case, NOT_A_SPEC_FIX, TO_PROPOSAL, parameter_id=pid,
                           citation=citation, reflection=v)

        return Routing(case, UNRESOLVED, TO_UNRESOLVED, reflection=v,
                       rejected_reason=f"unroutable verdict {v.verdict!r}")


# ------------------------------------------------------------------- 4. blast radius
@dataclass(frozen=True)
class BlastRadius:
    """How many OTHER cases a proposed addition would change.

    `computable` is always serialised. A rule added for one patient that changes two hundred
    others is a new policy in disguise, and an ABSENT number reads as zero — so where prose
    makes the count impossible, the field says so in words instead of disappearing.
    """

    computable: bool
    n_cases_changed: int | None
    n_cases_examined: int | None
    basis: str

    @classmethod
    def not_computable(cls, reason: str) -> BlastRadius:
        return cls(False, None, None, reason)

    def to_dict(self) -> dict:
        return {"computable": self.computable, "n_cases_changed": self.n_cases_changed,
                "n_cases_examined": self.n_cases_examined, "basis": self.basis}


@dataclass(frozen=True)
class CodedCase:
    """An already-coded answer, replayable offline: no model, no chart, just the ledger."""

    case_id: str
    value: Mapping[str, str]
    evidence: tuple[Mapping[str, str], ...] = ()
    searched: tuple[str, ...] = ()


def blast_radius_for_answer_check(candidate: Mapping, existing: Sequence[Mapping],
                                  cases: Iterable[CodedCase]) -> BlastRadius:
    """Replay a candidate `answer_checks` rule over already-coded answers.

    Mechanical rules are the one place the number is real, and it is real because it needs no
    model: `check_answer` is the same code the run uses. A case counts as CHANGED when the
    candidate rejects an answer the existing checks accepted.
    """
    changed, examined = 0, 0
    for c in cases:
        examined += 1
        before = check_answer(list(existing), dict(c.value), [dict(e) for e in c.evidence],
                              c.searched)
        after = check_answer([*existing, dict(candidate)], dict(c.value),
                             [dict(e) for e in c.evidence], c.searched)
        if len(after) > len(before):
            changed += 1
    return BlastRadius(True, changed, examined,
                       f"replayed candidate answer_check over {examined} already-coded answers "
                       f"with acr.contract.answer_checks.check_answer; no model call")


def blast_radius_for_keyword(term: str, note_texts: Iterable[str]) -> BlastRadius:
    """A keyword is priced by grep, and that is the whole reason keyword updates can be
    automatic while prose updates cannot."""
    texts = list(note_texts)
    hits = sum(1 for t in texts if _norm(term) in _norm(t))
    return BlastRadius(True, hits, len(texts),
                       f"substring count of {term!r} over {len(texts)} note texts; no model call")


def blast_radius_for(parameter_id: str, *, mechanism: str | None = None, **kw) -> BlastRadius:
    """Dispatch on how the number would be obtained, and refuse to invent one for prose."""
    p = get_parameter(parameter_id)
    if mechanism == "answer_check":
        return blast_radius_for_answer_check(kw["candidate"], kw["existing"], kw["cases"])
    if mechanism == "keyword":
        return blast_radius_for_keyword(kw["term"], kw["note_texts"])
    if p.mechanical:
        return BlastRadius.not_computable(
            f"{p.id} is mechanical, but no mechanism was named for this proposal, so nothing "
            f"was replayed. TODO(§6b): pass mechanism=; until then this must not read as zero.")
    return BlastRadius.not_computable(
        f"{p.id} is prose ({p.kind}). Its effect on other cases can only be measured by "
        f"re-running them, which is the validation run this proposal is asking for. That is a "
        f"real limit of the method, not a measurement someone forgot to take.")


# ------------------------------------------------------- proposals and escalations
@dataclass(frozen=True)
class Proposal:
    """A candidate edit to ONE parameter, carrying its evidence and its blast radius.

    Never applied here. For a CLINICIAN_SIGNS parameter it is a row in the §7 review document;
    for an AUTO_* parameter it is still only a candidate until its acceptance test passes.
    """

    parameter_id: str
    case_id: str
    verdict: str
    citation: Mapping[str, object]
    proposed_text: str
    blast_radius: BlastRadius
    change_class: str | None = None

    def __post_init__(self) -> None:
        p = get_parameter(self.parameter_id)
        if p.in_objective:
            # The only door into the objective, and it opens for FORM alone. There is no flag,
            # no policy value and no argument anywhere in this module that widens it.
            if self.change_class != FORM:
                raise ContentEscalationRequired(
                    f"{p.id} is inside the objective. A {self.change_class!r} gradient cannot "
                    f"become a proposed edit — it escalates as a question for a human. "
                    f"Editing a rule because the data disagreed with it is moving the target "
                    f"to where the arrows landed.")
        elif self.change_class is not None:
            raise RefineError(f"{p.id} is outside the objective; FORM/CONTENT does not apply")
        if not self.proposed_text.strip():
            raise RefineError("a proposal with no text is not a proposal")

    @property
    def may_apply_automatically(self) -> bool:
        return get_parameter(self.parameter_id).update_policy in (
            AUTO_ON_CERTIFICATION, AUTO_ON_HELDOUT_GAIN)


@dataclass(frozen=True)
class ClinicianQuestion:
    """A CONTENT gradient, escalated with its evidence. Deliberately has no `proposed_text`
    field: there is nothing to accept, only something to answer."""

    parameter_id: str
    case_id: str
    question: str
    quoted_passage: str
    evidence: Mapping[str, object]

    def to_dict(self) -> dict:
        return {"kind": "QUESTION", "parameter_id": self.parameter_id, "case_id": self.case_id,
                "question": self.question, "quoted_passage": self.quoted_passage,
                "evidence": dict(self.evidence),
                "note": "the data cannot answer this. Only the standard or a clinician can."}


def escalate(r: Routing) -> ClinicianQuestion:
    if r.change_class != CONTENT:
        raise RefineError("escalate() is for CONTENT gradients only")
    v = r.reflection
    return ClinicianQuestion(
        parameter_id=r.parameter_id or "spec_rules", case_id=r.case.case_id,
        question=(f"This rule may be substantively wrong for {r.case.field}. "
                  f"{v.rationale if v else ''}").strip(),
        quoted_passage=str(r.citation.get("quoted_passage") or ""),
        evidence={"coded_value": r.case.coded_value, "answer_key": r.case.key_value,
                  "invoked_rules": list(r.case.invoked_rules),
                  "rejection_messages_seen": list(r.case.rejection_messages_seen)})


# ----------------------------------------------------- 5. batching and acceptance
@dataclass
class RevisionBatch:
    """One coherent revision to one parameter, addressing SEVERAL failures.

    Batched because the acceptance test is expensive: a keyword is priced by grep, but a text
    parameter can only be validated by re-running, and it moves every patient rather than the
    failing one. Testing one change at a time is unaffordable, so changes arrive together.
    """

    parameter_id: str
    proposals: list[Proposal] = field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        return [p.case_id for p in self.proposals]

    def to_dict(self) -> dict:
        return {"parameter_id": self.parameter_id,
                "update_policy": get_parameter(self.parameter_id).update_policy,
                "n_elements": len(self.proposals),
                "elements": [{"case_id": p.case_id, "verdict": p.verdict,
                              "citation": dict(p.citation), "proposed_text": p.proposed_text,
                              "change_class": p.change_class,
                              "blast_radius": p.blast_radius.to_dict()}
                             for p in self.proposals]}


def assemble(routings: Iterable[Routing],
             blast_radius_of: Callable[[Routing], BlastRadius],
             ) -> tuple[list[RevisionBatch], list[ClinicianQuestion], list[Routing]]:
    """Accumulate classified failures into batches, questions and leftovers. Acts on none of
    them. `blast_radius_of` is required so that no proposal can be built without one."""
    batches: dict[str, RevisionBatch] = {}
    questions: list[ClinicianQuestion] = []
    leftover: list[Routing] = []
    for r in routings:
        if r.destination == TO_CLINICIAN_QUESTION:
            questions.append(escalate(r))
        elif r.destination == TO_PROPOSAL:
            text = (r.reflection.proposed_text if r.reflection else None) or \
                   (r.reflection.missing_sentence if r.reflection else None) or ""
            if not text.strip():
                # A verdict that survived the mask but proposes no replacement text is a
                # diagnosis, not a revision. It stays unresolved rather than becoming an
                # empty element that a clinician has to reconstruct from the citation.
                leftover.append(replace(r, rejected_reason="verdict carries no proposed text"))
                continue
            prop = Proposal(parameter_id=r.parameter_id or "", case_id=r.case.case_id,
                            verdict=r.verdict, citation=r.citation, proposed_text=text,
                            blast_radius=blast_radius_of(r), change_class=r.change_class)
            batches.setdefault(prop.parameter_id, RevisionBatch(prop.parameter_id))
            batches[prop.parameter_id].proposals.append(prop)
        else:
            leftover.append(r)
    return list(batches.values()), questions, leftover


def required_per_arm_n(*, baseline_accuracy: float, detectable_regression_pp: float,
                       z_alpha: float, z_power: float) -> int:
    """Two-proportion sample size per arm. Every constant is an argument, including the z's:
    hard-coding 1.96 is how a power calculation stops being reviewable."""
    for name, val in (("baseline_accuracy", baseline_accuracy),
                      ("detectable_regression_pp", detectable_regression_pp),
                      ("z_alpha", z_alpha), ("z_power", z_power)):
        if val is None:
            raise MissingThresholdError(f"{name} is required and has no default")
    p1 = baseline_accuracy
    p2 = baseline_accuracy - detectable_regression_pp / 100.0
    if not 0 < p2 < p1 < 1:
        raise RefineError("baseline_accuracy and detectable_regression_pp must leave two "
                          "proportions strictly inside (0,1)")
    var = p1 * (1 - p1) + p2 * (1 - p2)
    return math.ceil((z_alpha + z_power) ** 2 * var / ((p1 - p2) ** 2))


@dataclass(frozen=True)
class ValidationPlan:
    """What it would take to accept a batch. A PLAN — nothing here executes anything."""

    batch_id: str
    diagnosis_case_ids: frozenset[str]
    validation_case_ids: frozenset[str]
    per_arm_n: int
    cost_per_case_usd: float
    estimated_cost_usd: float
    arms: tuple[str, str] = ("control", "candidate")

    def to_dict(self) -> dict:
        return {"batch_id": self.batch_id, "arms": list(self.arms),
                "per_arm_n": self.per_arm_n, "cost_per_case_usd": self.cost_per_case_usd,
                "estimated_cost_usd": round(self.estimated_cost_usd, 2),
                "n_diagnosis": len(self.diagnosis_case_ids),
                "n_validation": len(self.validation_case_ids),
                "read_as": "PER INSTANCE. See per_instance_result_shape().",
                "status": "NOT RUN. TODO(§6b): execution is out of scope for this module."}


def plan_validation(batch: RevisionBatch, *, diagnosis_case_ids: Iterable[str],
                    validation_case_ids: Iterable[str], test_case_ids: Iterable[str],
                    baseline_accuracy: float, detectable_regression_pp: float,
                    z_alpha: float, z_power: float, cost_per_case_usd: float) -> ValidationPlan:
    """Two disjoint sets, a powered n, and a price. Refuses on any overlap.

    Proposing from failures on a set and scoring on the same set is fitting, and it reports as
    a gain. The test set is not one of these two and is checked against both.
    """
    if cost_per_case_usd is None:
        raise MissingThresholdError("cost_per_case_usd is required and has no default")
    diag, val, test = frozenset(diagnosis_case_ids), frozenset(validation_case_ids), \
        frozenset(test_case_ids)
    for a, b, an, bn in ((diag, val, "diagnosis", "validation"),
                         (diag, test, "diagnosis", "test"), (val, test, "validation", "test")):
        if a & b:
            raise OverlappingSetsError(f"{an} and {bn} share {len(a & b)} case(s): "
                                       f"{sorted(a & b)[:5]}")
    # The proposals were written from the diagnosis set, so a batch element sitting anywhere
    # else is the same fitting error wearing a different name.
    stray = {p.case_id for p in batch.proposals} - diag
    if stray:
        raise OverlappingSetsError(f"batch elements not in the diagnosis set: {sorted(stray)[:5]}")
    n = required_per_arm_n(baseline_accuracy=baseline_accuracy,
                           detectable_regression_pp=detectable_regression_pp,
                           z_alpha=z_alpha, z_power=z_power)
    if len(val) < n:
        raise RefineError(f"validation set has {len(val)} cases; detecting a "
                          f"{detectable_regression_pp}pp regression at {baseline_accuracy} "
                          f"accuracy needs {n} per arm. An underpowered run that shows no "
                          f"regression has not shown there is none.")
    return ValidationPlan(batch_id=batch.parameter_id, diagnosis_case_ids=diag,
                          validation_case_ids=val, per_arm_n=n,
                          cost_per_case_usd=cost_per_case_usd,
                          estimated_cost_usd=2 * n * cost_per_case_usd)


@dataclass(frozen=True)
class PerInstanceResult:
    """One patient, both arms. The unit the validation run must record."""

    case_id: str
    subgroup: str
    control_correct: bool
    candidate_correct: bool


def per_instance_result_shape() -> dict:
    return {"case_id": "str, pseudonymous", "subgroup": "str, the stratum to read separately",
            "control_correct": "bool", "candidate_correct": "bool",
            "why": "a mean over these hides a revision that lifts the average while "
                   "destroying one subgroup, which is the failure an average is built to hide"}


@dataclass(frozen=True)
class ValidationReading:
    mean_delta_pp: float
    per_subgroup_delta_pp: Mapping[str, float]
    regressed_subgroups: tuple[str, ...]
    accept: bool


def read_per_instance(results: Sequence[PerInstanceResult], *,
                      max_tolerated_subgroup_drop_pp: float) -> ValidationReading:
    """Read the validation run per instance. A positive mean does not carry a regressed
    subgroup: `accept` is False whenever any subgroup drops past the tolerated amount."""
    if max_tolerated_subgroup_drop_pp is None:
        raise MissingThresholdError("max_tolerated_subgroup_drop_pp is required, no default")
    if not results:
        raise RefineError("no per-instance results; there is nothing to read")
    by: dict[str, list[PerInstanceResult]] = {}
    for r in results:
        by.setdefault(r.subgroup, []).append(r)
    deltas = {g: 100.0 * (sum(r.candidate_correct for r in rs) - sum(r.control_correct for r in rs))
              / len(rs) for g, rs in by.items()}
    mean = 100.0 * (sum(r.candidate_correct for r in results)
                    - sum(r.control_correct for r in results)) / len(results)
    regressed = tuple(sorted(g for g, d in deltas.items() if d < -max_tolerated_subgroup_drop_pp))
    return ValidationReading(mean, deltas, regressed, mean > 0 and not regressed)
