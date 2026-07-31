"""Extraction specifications: load, validate, freeze.

A spec is the agent's *contract*. It states the decision boundary and the evidentiary
rules, but deliberately NOT the navigation path — how the agent finds the evidence is
its own business; what counts as evidence, and what must be true before it may assert a
negative, are not.

Two fields carry most of the design weight:

  proof_obligation  what must be demonstrably done before a negative/absent answer is allowed
  abstention        two distinct "I can't answer" states, which mean different things:
                      SPEC_INSUFFICIENT      the specification does not cover this case
                      EVIDENCE_INSUFFICIENT  the specification is clear, the chart is not

Every spec is content-hashed. A label is only comparable to another label produced under
the same spec_hash — that is the first of the three ground-truth layers.

The third field carrying design weight is `provenance`, and it is here because of what the
other two could not say. Every spec in this tree was written by a language model in one
commit (e5229b0), from whatever STORE manual text was in its context, and committed under a
human author's name; no registrar has read any of it. The only marking was
`source_authority` — free text, read by nothing, verified by nothing, and by the stage spec's
own admission possibly carrying wrong item numbers. Transcribed lines, inferred lines and
invented lines are all present, mixed, and to a reader they look identical: they are all
just YAML. `provenance` marks them one enforced element at a time and `load_spec` refuses a
spec that leaves one unmarked, because a marking that can be skipped becomes a marking that
is skipped exactly where it matters.

TWO CHANNELS, and the split is deliberate.

  provenance            ENFORCED. Narrow: only the elements `enforced_elements()` enumerates,
                        the ones some code path changes behaviour on. A missing record is an
                        ERROR and the spec does not load. This channel gates whether a run may
                        be reported as validated, so a hole in it is a hole in a claim.

  editorial_provenance  EDITORIAL. Broad: any statement a reviewer could argue with —
                        decision_rule items, evidence_rules, conflict_rules, boundary_cases,
                        the abstention wording. Those are rendered into the prompt and read by
                        a human, not applied by code, so they are outside the enforced set by
                        construction. A missing record here is NOT an error; it is a FINDING,
                        which `specview` prints as "origin not recorded" in WHAT WE MADE UP.
                        Nothing in this channel may raise: an author half-way through
                        annotating a file must still be able to load and lint it, and a
                        channel that punished partial work would be left empty.

Same record shape in both — `ProvenanceRecord`, same origins, same honesty rules. The whole
asymmetry is what a missing record does. Conflating them was a real bug: the elements a
clinician most needs to see in WHAT WE MADE UP are exactly the editorial ones, and routing
them through the enforced channel got them rejected as `StaleProvenanceError` for naming
something the runtime does not read.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"]


class ProofObligation(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Two grammars, both legal.
    #
    #   for_positive: "One pathology report establishing the histology."      <- prose only
    #
    #   for_positive:                                                          <- binding
    #     statement: "One pathology report establishing the histology."
    #     witness:
    #       histology: [can_establish]
    #
    # The prose form is a wish: nothing reads it. STORE.400_522_523 declared "histology must
    # not be inferred from imaging or a clinical assertion" and on 2026-07-26 SYN0002 — a
    # patient with zero pathology documents — had a FOUND histology accepted, cited from a
    # progress note, stamped gate_validated. `witness` is the same sentence written where the
    # gate can read it: per field, the strata a citation may come from.
    for_positive: str | dict[str, Any] = "A single sufficient piece of evidence is enough."
    for_negative: dict[str, Any] = Field(default_factory=dict)

    @property
    def positive_statement(self) -> str:
        """The prose half of for_positive, whichever grammar was used."""
        fp = self.for_positive
        if isinstance(fp, dict):
            return str(fp.get("statement") or "").strip()
        return str(fp)

    @property
    def witness_strata(self) -> dict[str, list[str]]:
        """field -> stratum names admitted as a witness for it. Empty when undeclared.

        Deliberately raw: which of these names are real strata, and whether each one's
        `establishes` actually claims the field, is resolved in `coverage.witness_policy`
        where the stratum definitions live. A spec loader that silently dropped an
        unresolvable name would hide exactly the authoring fault worth reporting.
        """
        fp = self.for_positive
        if not isinstance(fp, dict):
            return {}
        raw = fp.get("witness") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list[str]] = {}
        for fname, names in raw.items():
            seq = names if isinstance(names, (list, tuple)) else [names]
            out[str(fname)] = [str(n) for n in seq]
        return out

    @property
    def required_coverage(self) -> list[str]:
        rc = self.for_negative.get("required_coverage", []) if self.for_negative else []
        return list(rc) if isinstance(rc, list) else []

    @property
    def required_keywords(self) -> list[str]:
        kw = self.for_negative.get("required_keywords", []) if self.for_negative else []
        return list(kw) if isinstance(kw, list) else []

    @property
    def required_doc_types(self) -> list[str]:
        dt = self.for_negative.get("required_doc_types_read", []) if self.for_negative else []
        return list(dt) if isinstance(dt, list) else []


class OutputField(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: str = "string"
    format: str | None = None
    allowable_values: list[Any] | None = None
    nullable: bool = True
    description: str = ""


# ---------------------------------------------------------------------------- provenance
#: Where the CONTENT of an element came from. Not how much anyone trusts it — that is
#: `status`. The two are separate because they fail apart: a faithful transcription nobody
#: has checked and a confabulation nobody has checked are both `draft`, and the difference
#: between them is exactly what a reader needs.
Origin = Literal["store_manual", "ajcc_manual", "corpus_derived", "model_authored", "clinician"]

#: How far the element has been verified. Ordered, weakest first: a run reports the minimum
#: over what it used, so adding a status here means deciding where it sits in that order.
ProvenanceStatus = Literal["draft", "measured", "clinician_reviewed"]
STATUS_ORDER: tuple[str, ...] = ("draft", "measured", "clinician_reviewed")

#: What a measurement concluded. `measured` as a STATUS requires `supports`: STORE.400's
#: keyword list has been measured on 1,788 real charts and the measurement is the reason to
#: distrust it, so letting any measurement raise the status would rank the elements we know
#: are broken above the ones nobody has looked at.
MeasuredVerdict = Literal["supports", "underpowered", "falsified"]

#: A manual citation has to be findable by someone holding the manual. This does not check
#: that the locator is CORRECT — nothing here can — it only stops "it reads like the manual"
#: from passing as a citation, which is the specific laundering this field exists to prevent.
_MANUAL_LOCATOR = re.compile(
    r"\[\d+\]|§|\bitem\b|\bsection\b|\brule\b|\btable\b|\bchapter\b|\bappendix\b|\bp\.\s*\d",
    re.IGNORECASE)

#: Required verbatim in a model_authored basis. An author who has to type this sentence has
#: to notice they are typing it; "derived from standard registry practice" is the same claim
#: with the admission filed off.
MODEL_AUTHORED_ADMISSION = "no external source"


class ProvenanceError(ValueError):
    """A spec's provenance block does not hold up. Always fatal at load."""


class UnprovenancedElementError(ProvenanceError):
    """An enforced element carries no provenance record.

    Fatal rather than a warning on purpose. The failure this prevents is not that someone
    reads a warning and ignores it — it is that an unmarked invented rule is indistinguishable
    from a transcribed one, so nothing about the run ever looks wrong.
    """

    def __init__(self, spec_id: str, elements: "list[EnforcedElement]"):
        self.spec_id = spec_id
        self.elements = list(elements)
        lines = [f"{spec_id}: {len(self.elements)} enforced element(s) carry no provenance "
                 f"record. Each is read by the runtime and must declare origin/basis/status:"]
        lines += [f"  - {e.path}   [{e.kind}, read by {e.read_by}]" for e in self.elements]
        super().__init__("\n".join(lines))


class StaleProvenanceError(ProvenanceError):
    """A record names an element the spec no longer has.

    The dangerous direction of drift: rename a stratum and its clinician sign-off keeps
    sitting in the file, attached to nothing, still counting as a signature to any reader.
    """


@dataclass(frozen=True)
class EnforcedElement:
    """One thing the runtime actually reads out of a spec and acts on.

    `read_by` is part of the record, not decoration. When the loader refuses a spec it names
    the consumer, because "add provenance for strata[may_mention].required_keywords" is not
    actionable until you know that the coverage ledger decides a gate verdict with it.
    """
    path: str
    kind: str
    value: Any
    read_by: str
    field: str | None = None

    @property
    def hash(self) -> str:
        return _content_hash(self.value)


def _content_hash(value: Any) -> str:
    """Hash of an element's content, canonicalised so YAML formatting cannot change it.

    Taken over the parsed value, never the source text: re-indenting a list or re-quoting a
    string must not read as an edit, or every reformat would void every signature and the
    signatures would be turned off within a week.
    """
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ProvenanceRecord(BaseModel):
    """Per-element provenance. One record, one enforced element, no exemptions.

    `spec_hash_at_review` and `element_hash_at_review` are both recorded and they do
    different jobs. The element hash is what invalidates a sign-off, because it is the only
    one that changes when — and only when — the reviewed rule changes. The spec hash is the
    version of the document that was on the reviewer's screen, which is what you need to go
    and read what they actually approved; it is the PROVENANCE-FREE hash, since a hash that
    covered the signature could never be recomputed by the person checking it.
    """
    model_config = ConfigDict(extra="forbid")

    element: str
    origin: Origin
    #: For store_manual/ajcc_manual: the section, item or rule. For corpus_derived: the run
    #: and how many patients. For model_authored: the admission, verbatim.
    basis: str
    status: ProvenanceStatus = "draft"
    #: Numbers from the development plane. Carries `verdict` so that measuring an element and
    #: finding it broken is recordable as what it is, rather than as an upgrade.
    measured: dict[str, Any] = Field(default_factory=dict)
    reviewed_by: str | None = None
    reviewed_on: str | None = None
    spec_hash_at_review: str | None = None
    element_hash_at_review: str | None = None

    # -- written by the loader, never by an author -------------------------------------
    element_hash: str | None = None
    element_kind: str | None = None
    #: True when a signed element has since been edited. The signature is voided (status
    #: falls back to draft) but `reviewed_by` is kept: erasing the name would hide that
    #: somebody's approval was overridden, which is the fact worth surfacing.
    sign_off_voided_by_edit: bool = False

    @property
    def rank(self) -> int:
        return STATUS_ORDER.index(self.status)


def _answer_check_key(chk: dict) -> str:
    """Identity of an answer_check, from its content rather than its position.

    Indices would be shorter and wrong: inserting a check at the top would silently re-point
    every record below it at a different rule, sign-offs included. field+kind+first NOS value
    is what actually distinguishes the checks that exist (STORE.700_880 declares
    `clinical_t/not_less_specific` twice, for cT1 and for cT2).
    """
    # `nos_value` singular is `conflict_requires_nos`'s spelling — it names the one code to
    # fall back to rather than a set of codes to guard. Without it in the discriminator, two
    # conflict checks on one field would share a rule id, and a sign-off on either would read
    # as a sign-off on both.
    nos = chk.get("nos_values") or ([chk["nos_value"]] if chk.get("nos_value") else [])
    return (f"{chk.get('field', '?')}.{chk.get('kind', 'not_less_specific')}"
            + (f".{nos[0]}" if nos else ""))


def enforced_elements(spec: "ExtractionSpec") -> list[EnforcedElement]:
    """Everything in this spec that the runtime reads and acts on, in declaration order.

    The list is deliberately narrower than "everything in the file". `decision_rule` and
    `boundary_cases` are rendered into the prompt and are the model's problem; the elements
    below are applied by code, to accept or reject an answer, with no model in the loop. An
    element enters this list when some code path CHANGES BEHAVIOUR on its value:

      fields[].format / .allowable_values     answer_checks.check_field_formats
      answer_checks[]                         answer_checks.check_answer
      for_negative.required_keywords          graph.check_gate  (+ rendered in every prompt)
      for_negative.required_coverage          graph.check_gate  (+ rendered in every prompt)
      for_negative.gate.*                     coverage.evaluate_gate
      strata[].match / .partition_by          coverage.assign_strata
      strata[].establishes                    coverage.witness_policy
      strata[].required_keywords              coverage.CoverageLedger.stratum_results
      strata[].min_sample*                    coverage.CoverageLedger.pending_samples

    Three near-misses are deliberately absent, and each is a finding rather than an oversight:
    `max_tolerated_hits` is parsed into StratumSpec and then read by nothing (the tolerance is
    hard-coded to zero, which happens to agree with every spec here); a claim-level
    `required_keywords` is read by nothing, since `strata_from_spec` descends into
    `claims[].strata` only; and `qualifying_doc_types` / `empty_window_policy` /
    `surveillance_schedule` are read only by `clip_and_judge`, which no runtime path calls.
    Requiring provenance for a line the runtime ignores would teach authors that these records
    are paperwork. When any of them is wired up, it belongs in this list in the same commit.

    Gate keys absent from the YAML are also absent here: `evaluate_gate` defaults them to
    True, so an unwritten key is the CODE's decision and carries the code's provenance, not
    the spec's.
    """
    out: list[EnforcedElement] = []

    def add(path: str, kind: str, value: Any, read_by: str, field: str | None = None) -> None:
        out.append(EnforcedElement(path, kind, value, read_by, field))

    for f in spec.fields:
        if f.format:
            add(f"fields[{f.name}].format", "field_format", f.format,
                "answer_checks.check_field_formats", f.name)
        if f.allowable_values:
            add(f"fields[{f.name}].allowable_values", "field_allowable_values",
                list(f.allowable_values), "answer_checks.check_field_formats", f.name)

    for chk in (spec.answer_checks or []):
        if isinstance(chk, dict):
            add(f"answer_checks[{_answer_check_key(chk)}]", "answer_check", chk,
                "answer_checks.check_answer", chk.get("field"))

    fn = (spec.proof_obligation.for_negative or {}) if spec.proof_obligation else {}
    if fn.get("required_coverage"):
        add("proof_obligation.for_negative.required_coverage", "required_coverage",
            list(fn["required_coverage"]), "graph.check_gate")
    if fn.get("required_keywords"):
        add("proof_obligation.for_negative.required_keywords", "required_keywords",
            list(fn["required_keywords"]), "graph.check_gate")
    for key, val in (fn.get("gate") or {}).items():
        add(f"proof_obligation.for_negative.gate.{key}", "gate_threshold", val,
            "coverage.evaluate_gate")

    scopes: list[tuple[str, dict]] = [("proof_obligation.for_negative", fn)]
    for claim in (fn.get("claims") or []):
        scopes.append((f"proof_obligation.for_negative.claims[{claim.get('id')}]", claim))
    for prefix, holder in scopes:
        for s in (holder.get("strata") or []):
            base = f"{prefix}.strata[{s.get('name')}]"
            if "match" in s:
                add(f"{base}.match", "stratum_match", s["match"], "coverage.assign_strata")
            elif "partition_by" in s:
                add(f"{base}.partition_by", "stratum_match", s["partition_by"],
                    "coverage.assign_strata")
            if "establishes" in s:
                add(f"{base}.establishes", "stratum_establishes", list(s["establishes"] or []),
                    "coverage.witness_policy")
            if s.get("required_keywords"):
                add(f"{base}.required_keywords", "required_keywords",
                    list(s["required_keywords"]), "coverage.CoverageLedger.stratum_results")
            for k in ("min_sample", "min_sample_of_misses"):
                if k in s:
                    add(f"{base}.{k}", "sample_threshold", s[k],
                        "coverage.CoverageLedger.pending_samples")

    seen: dict[str, EnforcedElement] = {}
    for e in out:
        if e.path in seen:
            raise ProvenanceError(
                f"{spec.spec_id}: two enforced elements resolve to the same path {e.path!r}. "
                "One provenance record cannot speak for two rules — give them distinguishable "
                "identities (for an answer_check, a distinct field/kind/first nos_value).")
        seen[e.path] = e
    return out


def weakest_status(statuses: Iterable[str]) -> str:
    """The minimum, and `draft` for nothing at all.

    An empty set is the interesting case: "this run used no provenanced element" is not a
    clean bill of health, it is an absence of evidence, and it must not read as one.
    """
    ranks = [STATUS_ORDER.index(s) for s in statuses if s in STATUS_ORDER]
    return STATUS_ORDER[min(ranks)] if ranks else STATUS_ORDER[0]


def _validate_record(spec_id: str, rec: ProvenanceRecord) -> None:
    """The honesty rules. Each one exists to close a way of writing a record that says less
    than it appears to."""
    where = f"{spec_id} / {rec.element}"
    if not (rec.basis or "").strip():
        raise ProvenanceError(f"{where}: basis is empty. An origin with no basis is a label.")

    if rec.origin in ("store_manual", "ajcc_manual") and not _MANUAL_LOCATOR.search(rec.basis):
        raise ProvenanceError(
            f"{where}: origin {rec.origin!r} but the basis names no section, item, rule or "
            f"table. If you cannot name the section, it did not come from the manual — say "
            f"model_authored instead.\n  basis: {rec.basis.strip()[:160]}")
    if rec.origin == "model_authored" and MODEL_AUTHORED_ADMISSION not in rec.basis.lower():
        raise ProvenanceError(
            f"{where}: a model_authored basis must contain the phrase "
            f"{MODEL_AUTHORED_ADMISSION!r} verbatim. Everything else — 'derived from standard "
            f"practice', 'consistent with the manual' — is the same claim with the admission "
            f"filed off.\n  basis: {rec.basis.strip()[:160]}")
    if rec.origin == "corpus_derived":
        missing = [k for k in ("run", "n_patients") if k not in rec.measured]
        if missing:
            raise ProvenanceError(
                f"{where}: origin corpus_derived but measured is missing {missing}. A "
                "corpus-derived claim without the run that produced it and the n it was "
                "measured over is a model-authored claim with a number in it.")
    if rec.origin == "clinician" and not rec.reviewed_by:
        raise ProvenanceError(f"{where}: origin clinician but nobody is named in reviewed_by.")

    verdict = rec.measured.get("verdict")
    if rec.measured and verdict not in ("supports", "underpowered", "falsified"):
        raise ProvenanceError(
            f"{where}: measured block must carry verdict: supports | underpowered | "
            f"falsified, got {verdict!r}. Numbers without a verdict get read as support.")
    if rec.status == "measured":
        if not rec.measured:
            raise ProvenanceError(f"{where}: status measured with no measured block.")
        if verdict != "supports":
            raise ProvenanceError(
                f"{where}: status measured, but the measurement's verdict is {verdict!r}. A "
                "falsified or underpowered element stays draft — the measurement is the "
                "reason to distrust it, and it must not rank above an element nobody has "
                "looked at. Keep the numbers; leave the status at draft.")
    if rec.status == "clinician_reviewed":
        missing = [k for k in ("reviewed_by", "reviewed_on", "element_hash_at_review")
                   if not getattr(rec, k)]
        if missing:
            raise ProvenanceError(
                f"{where}: status clinician_reviewed but {missing} not set. An unsigned "
                "sign-off is the thing this whole block exists to make impossible; "
                "element_hash_at_review is what a later edit is compared against.")


def bind_provenance(spec: "ExtractionSpec") -> None:
    """Attach each record to its element, void signatures on edited elements, then refuse
    the spec if anything enforced is left unmarked. Mutates `spec.provenance` in place."""
    # A DECLARED CHECK MUST BE A CHECK THAT EXISTS. `check_answer_detail` dispatches on
    # `chk["kind"]` through an if/elif chain with no final else, so a misspelled kind matched
    # nothing and raised nothing: the rule appeared in the YAML and in the manifest's
    # `rule_catalog`, and produced zero rejections forever. Refused here rather than warned
    # about, because a check that cannot fire is worse than an absent one -- it is an absent one
    # that a reader counts as present.
    from .answer_checks import ANSWER_CHECK_KINDS
    for chk in (spec.answer_checks or []):
        c = chk if isinstance(chk, dict) else (chk.model_dump() if hasattr(chk, "model_dump")
                                               else dict(chk))
        kind = str(c.get("kind") or "not_less_specific")
        if kind not in ANSWER_CHECK_KINDS:
            raise ProvenanceError(
                f"{spec.spec_id}: answer_checks[{c.get('field')}] declares kind {kind!r}, which "
                f"nothing implements. Known kinds: {sorted(ANSWER_CHECK_KINDS)}. A kind no code "
                "dispatches on is a rule that can never fire, and it reads in the manifest "
                "exactly like a rule that fired and found nothing.")

    by_path = {e.path: e for e in enforced_elements(spec)}
    seen: set[str] = set()
    for rec in spec.provenance:
        _validate_record(spec.spec_id, rec)
        if rec.element in seen:
            raise ProvenanceError(
                f"{spec.spec_id}: two provenance records for {rec.element!r}. Which one is "
                "the provenance? Merge them.")
        seen.add(rec.element)
        element = by_path.get(rec.element)
        if element is None:
            # Two different mistakes land here and the repair for one is wrong for the other:
            # a stale record is deleted, an editorial one is MOVED. Say which, or the natural
            # fix looks like widening `enforced_elements()` to admit `decision_rule` — the
            # conflated-channels bug, declaring the runtime to read a line it does not read.
            raise StaleProvenanceError(
                f"{spec.spec_id}: provenance record for {rec.element!r}, which is not an "
                f"enforced element of this spec (renamed? deleted?). A record left behind by "
                f"a rename is a signature attached to nothing that still reads as a signature."
                f"\n  If it names a statement a REVIEWER reads rather than one the runtime "
                f"applies — a decision_rule, an evidence rule, a conflict rule, a boundary "
                f"case — it is editorial and belongs in `editorial_provenance`, where a "
                f"missing or malformed record is a finding rather than an unloadable file."
                f"\n  enforced elements: {sorted(by_path)}")
        rec.element_hash = element.hash
        rec.element_kind = element.kind
        if rec.element_hash_at_review and rec.element_hash_at_review != element.hash:
            # The sign-off decay rule. Someone approved a different text.
            rec.sign_off_voided_by_edit = True
            rec.status = "draft"
    unmarked = [e for p, e in by_path.items() if p not in seen]
    if unmarked:
        raise UnprovenancedElementError(spec.spec_id, unmarked)


# ------------------------------------------------------------------ editorial provenance
class ProvenanceFinding(BaseModel):
    """Something wrong with an EDITORIAL record, reported instead of raised.

    A finding, not an exception, and the difference is the whole reason the second channel
    exists. `bind_provenance` may refuse a file because every enforced element is a line the
    runtime acts on, so an unmarked one is a live unattributed rule. Editorial elements are
    read by a human, and refusing to load a spec because its author has annotated four of its
    nine coding rules would mean nobody ever annotates the fifth.

    The findings still have to go somewhere a reader will see them, or "advisory" becomes
    "ignored": `specview` prints them in WHAT WE MADE UP, beside the elements they concern.
    """
    element: str
    problem: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.element}: {self.problem}"


#: A missing editorial record is not silence, it is this. Rendered verbatim to the reviewer,
#: because "we did not write down where this came from" and "we wrote down that a model made
#: it up" are different states and only one of them is somebody having looked.
ORIGIN_NOT_RECORDED = "origin_not_recorded"


def editorial_records(spec: "ExtractionSpec") -> tuple[dict[str, ProvenanceRecord],
                                                       list[ProvenanceFinding]]:
    """Parse the editorial channel leniently. Returns (valid records by element, findings).

    Never raises, for anything. A record with an unknown origin, an uncitable manual claim or
    a duplicate element becomes a finding and is dropped from the index; the rest of the file
    still loads. That tolerance is why the raw field is `list[dict]` rather than
    `list[ProvenanceRecord]` — pydantic validating the model would turn a typo in a half-
    written annotation into an unloadable spec, which is the failure this channel is for.
    """
    index: dict[str, ProvenanceRecord] = {}
    findings: list[ProvenanceFinding] = []
    for raw in spec.editorial_provenance:
        element = str(raw.get("element", "")) if isinstance(raw, dict) else ""
        if not isinstance(raw, dict) or not element:
            findings.append(ProvenanceFinding(
                element=element or "?",
                problem=f"not a provenance record ({raw!r:.80}); expected a mapping with at "
                        f"least element/origin/basis"))
            continue
        if element in index:
            findings.append(ProvenanceFinding(
                element=element, problem="two editorial records name this element; the second "
                                         "was dropped. Merge them."))
            continue
        try:
            rec = ProvenanceRecord.model_validate(raw)
            _validate_record(spec.spec_id, rec)
        except Exception as e:  # pydantic ValidationError or ProvenanceError, both advisory
            findings.append(ProvenanceFinding(
                element=element,
                problem=str(e).replace("\n", " ").strip()[:400]))
            continue
        index[element] = rec
    return index, findings


def editorial_findings(spec: "ExtractionSpec",
                       known_elements: Iterable[str] | None = None,
                       attributed_elsewhere: Iterable[str] | None = None,
                       ) -> list[ProvenanceFinding]:
    """Every editorial complaint about this spec, including the missing ones.

    `known_elements` is the set of element ids the reviewer-facing document actually renders
    — `specview.element_ids()`. Passing it turns on the two findings that need to know what
    exists: a record naming nothing (the editorial twin of `StaleProvenanceError`, demoted to
    a finding because an editorial id is a rendering detail and a renamed one must not brick
    the loader), and a statement nobody has attributed. Omit it and only the record-shape
    findings are reported.

    `attributed_elsewhere` is the ids the ENFORCED block already speaks for — the keyword
    lists, the sample sizes, the answer checks — which the review document renders as one
    sentence apiece under an id of its own. Without it this function would report a keyword
    list as unattributed while the document beside it printed the attribution, and two
    disagreeing accounts of what is attributed is worse than either one alone. Only
    `specview` knows the correspondence, so only `specview` can supply it.
    """
    index, findings = editorial_records(spec)
    if known_elements is None:
        return findings
    known = list(dict.fromkeys(str(k) for k in known_elements))
    known_set = set(known)
    covered = set(index) | {str(x) for x in (attributed_elsewhere or ())}
    for element in index:
        if element not in known_set:
            findings.append(ProvenanceFinding(
                element=element,
                problem="names no statement in this specification (renamed? deleted?). The "
                        "record is attached to nothing and still reads as an attribution."))
    for element in known:
        if element not in covered:
            findings.append(ProvenanceFinding(element=element, problem=ORIGIN_NOT_RECORDED))
    return findings


class ExtractionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    spec_id: str
    spec_version: str = "0.1.0"
    source_authority: dict[str, Any] = Field(default_factory=dict)
    #: Per enforced element, where it came from and how far it has been checked. Enforced by
    #: `load_spec`: a spec that leaves one enforced element unmarked does not load at all.
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    #: The advisory channel, for the statements a reviewer reads rather than the runtime.
    #: Same record shape, kept as raw mappings so that a malformed one is a finding rather
    #: than an unloadable file — see `editorial_records`. Missing entries are the normal
    #: state and are reported, never raised.
    editorial_provenance: list[Any] = Field(default_factory=list)
    question: str
    data_source: Literal["notes", "outside_notes"] = "notes"

    fields: list[OutputField] = Field(default_factory=list)
    decision_rule: list[str] = Field(default_factory=list)
    evidence_rules: dict[str, Any] = Field(default_factory=dict)
    when_not_to_use: list[str] = Field(default_factory=list)
    conflict_rules: list[Any] = Field(default_factory=list)
    answer_checks: list[Any] = Field(default_factory=list)
    proof_obligation: ProofObligation = Field(default_factory=ProofObligation)
    abstention: dict[str, str] = Field(default_factory=dict)
    special_codes_not_mar: list[Any] = Field(default_factory=list)
    boundary_cases: list[Any] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    #: Which ICD-O-3 code table this variable codes into, by name in `assets/codes/` — e.g.
    #: `icdo3_lung`. A TASK CONTRACT declaration: which code system a value belongs to is part
    #: of what the answer means, and it is not the runtime's guess to make.
    #:
    #: ADVISORY, not enforced. The table is rendered into the prompt so the model codes into a
    #: domain it can see rather than one it half-remembers, and read by the evaluators so an
    #: out-of-domain code is counted. Nothing refuses an answer over it — see
    #: `docs/DETERMINISTIC_RULES_REMOVED.md`, where five checks that did refuse over content
    #: destroyed a correct value 58 times against 21 helps.
    #:
    #: It carries no provenance record for the same reason: `provenance` is per ENFORCED element,
    #: and this changes what the model is shown, not what the gate decides. The table's own
    #: `source_authority` carries its provenance, and says it was recalled by a model rather than
    #: transcribed.
    value_domain: str = ""
    applicability_guard: dict[str, Any] = Field(default_factory=dict)
    agent_policy: str = ""
    downstream_warning: list[str] = Field(default_factory=list)

    @field_validator("provenance", mode="before")
    @classmethod
    def _refuse_the_editorial_shorthand(cls, v: Any) -> Any:
        """A `{element: origin}` mapping here is a reach for the other channel, said in words.

        The editorial validator below normalises that shorthand and reports its missing basis;
        this one cannot, because a bare label is what `provenance` was added to replace. It is
        refused either way — but pydantic's own "Input should be a valid list" sends the author
        to check their YAML syntax, and the mistake is never in the syntax.
        """
        if isinstance(v, dict):
            raise ValueError(
                "provenance must be a LIST of records, each naming an ENFORCED element with an "
                "origin AND a basis. A {element: origin} mapping has nowhere to put the basis, "
                "and an origin with no basis is a label — the marking this field replaced. If "
                "the element is a statement a reviewer reads rather than one the runtime "
                "applies (a decision_rule, an evidence rule, a conflict rule, a boundary case), "
                "write it in `editorial_provenance`, which takes this shorthand and reports "
                "what is missing instead of refusing the file.")
        return v

    @field_validator("editorial_provenance", mode="before")
    @classmethod
    def _tolerate_any_editorial_shape(cls, v: Any) -> list[Any]:
        """Coerce whatever is written there into a list, without judging it.

        The judging happens in `editorial_records`, where it produces findings. If this
        validator raised on a mapping or a stray string the advisory channel would be able to
        make a spec unloadable, which is precisely what it exists not to do.
        """
        if v is None:
            return []
        if isinstance(v, dict):
            # `{element: origin}` and `{element: {...}}` shorthands, normalised so the author
            # gets a real finding about the missing basis rather than a parser error.
            return [{"element": k, **(dict(x) if isinstance(x, dict) else {"origin": x})}
                    for k, x in v.items()]
        return list(v) if isinstance(v, (list, tuple)) else [v]

    # ---------------------------------------------------------------- freezing
    def canonical(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]

    @property
    def provenance_free_hash(self) -> str:
        """The spec minus its provenance block. What a reviewer's signature is taken over.

        Two hashes rather than one, because they answer different questions. `spec_hash`
        identifies the artifact a run was conducted under, and it must move when provenance
        moves: signing the keyword list changes what the run may claim, so it changes the
        run's identity. A signature, though, has to be recomputable by whoever later checks
        it — and a hash that included the signature could never be recomputed at all.
        """
        d = self.model_dump(mode="json")
        d.pop("provenance", None)
        # Both channels come out. A registrar signs a RULE, not the paperwork about the rule,
        # and an editorial note added three hundred lines away must not void that signature —
        # the same reason position is excluded from the element hash.
        d.pop("editorial_provenance", None)
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def identity(self) -> dict:
        return {"spec_id": self.spec_id, "spec_version": self.spec_version, "spec_hash": self.spec_hash}

    # ---------------------------------------------------------------- provenance
    @property
    def provenance_index(self) -> dict[str, ProvenanceRecord]:
        return {r.element: r for r in self.provenance}

    def elements_used(self, value: dict | None, status: str) -> list[str]:
        """The enforced elements a run with this answer actually applied.

        Mirrors `graph.gate_answer` rather than approximating it, because an approximation
        drifts and this number is a claim about a specific run:

          * the field format / allowable-value / answer_check rules run ONLY on a FOUND
            answer, and only for fields that carry a value — `check_field_formats` skips an
            empty one, so its constraints were never applied and the run neither earns nor
            owes anything for them;
          * the gate, the stratum keyword lists and the sample sizes are read only on the
            absence path, where `check_gate` and the ledger's stratum results are computed;
          * the stratum match rules and `establishes` are used by every run whatever it
            answers, because the ledger assigns every document to a stratum the moment it is
            built, and that assignment is what any later coverage question is answered from;
          * `for_negative.required_keywords` and `required_coverage` count as used by every
            run too — beyond the gate, `as_prompt_block` renders them into the prompt as a
            binding obligation, so they shape the search that produced the answer.
        """
        answered = {k for k, v in (value or {}).items() if str(v if v is not None else "").strip()}
        found = status == "FOUND"
        used: list[str] = []
        for e in enforced_elements(self):
            if e.kind in ("field_format", "field_allowable_values", "answer_check"):
                if found and e.field in answered:
                    used.append(e.path)
            elif e.kind in ("gate_threshold", "sample_threshold"):
                if not found:
                    used.append(e.path)
            elif e.kind == "required_keywords" and e.path.startswith(
                    "proof_obligation.for_negative.strata"):
                if not found:
                    used.append(e.path)
            else:
                used.append(e.path)
        return used

    def provenance_for_run(self, value: dict | None, status: str,
                           gate_validated: bool | None = None) -> dict:
        """The provenance block a run manifest carries: the WEAKEST status among what it used.

        The weakest, not the average and not the best, because a proof is only as good as its
        weakest premise. And `reportable_as_validated` is a separate boolean from
        `gate_validated` on purpose: the coverage gate proves the SEARCH was done — it says
        nothing about whether the search terms were the right ones. Measured on all 1,788
        real charts, STORE.400's five required keywords miss the stated diagnosis in 4,005
        documents across 567 patients (31.7%), and a run over any of those patients can pass
        the gate cleanly. A run that leaned on a model-authored draft is a run whose question
        nobody has checked, and it must not be filterable as validated on that basis.
        """
        used = self.elements_used(value, status)
        index = self.provenance_index
        recs = [index[p] for p in used if p in index]
        weakest = weakest_status([r.status for r in recs])
        counts_status: dict[str, int] = {}
        counts_origin: dict[str, int] = {}
        for r in recs:
            counts_status[r.status] = counts_status.get(r.status, 0) + 1
            counts_origin[r.origin] = counts_origin.get(r.origin, 0) + 1
        permits = weakest == "clinician_reviewed"
        return {
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "provenance_free_hash": self.provenance_free_hash,
            "weakest_status": weakest,
            # Naming them, not just counting them: a verdict whose subject is unknown cannot
            # be acted on, and the point of the block is to say which line to go and fix.
            "weakest_elements": [r.element for r in recs if r.status == weakest],
            "elements_used": used,
            "counts_by_status": counts_status,
            "counts_by_origin": counts_origin,
            "falsified_elements": [r.element for r in recs
                                   if r.measured.get("verdict") == "falsified"],
            "voided_sign_offs": [r.element for r in recs if r.sign_off_voided_by_edit],
            "gate_validated": gate_validated,
            "reportable_as_validated": bool(permits and (gate_validated is not False)),
            "why": ("provenance permits a validated claim" if permits else
                    f"the weakest element this run used is {weakest!r}; the coverage gate can "
                    "show the search was done, not that the search terms were right"),
        }

    # ---------------------------------------------------------------- prompting
    def as_prompt_block(self, *, view: str = "full") -> str:
        """Render the spec for the model. Ordering matters: rules before examples.

        ``clinical_contract`` deliberately withholds retrieval implementation:
        task-specific keywords, raw document-type requirements, and coverage work lists.
        It keeps the semantics of positive/negative answers. Runtime profiles may reveal a
        separately versioned retrieval or coverage asset later without changing the clinical
        contract the model was asked to apply.
        """
        if view not in {"full", "clinical_contract"}:
            raise ValueError(f"unknown prompt view {view!r}")
        include_retrieval = view == "full"
        L: list[str] = [f"# EXTRACTION SPECIFICATION  ({self.spec_id} v{self.spec_version})", ""]
        L += [f"QUESTION: {self.question}", ""]
        if self.data_source == "outside_notes":
            L += [
                "!! DATA SOURCE WARNING: this variable is NOT derivable from clinical notes.",
                "   It lives in institutional registration / follow-up systems.",
                "   You must answer SPEC_INSUFFICIENT. You may report clues you found, as evidence,",
                "   but you must not assign a value.",
                "",
            ]
        if self.agent_policy:
            L += ["AGENT POLICY (binding):", _indent(self.agent_policy), ""]
        if self.fields:
            L.append("OUTPUT FIELDS:")
            for f in self.fields:
                bits = [f"  - {f.name} ({f.type}"]
                if f.format:
                    bits.append(f", format={f.format}")
                bits.append(")")
                line = "".join(bits)
                if f.description:
                    line += f" — {f.description}"
                L.append(line)
                if f.allowable_values:
                    vals = ", ".join(str(v) for v in f.allowable_values[:40])
                    L.append(f"      allowable: {vals}")
            L.append("")
        if self.decision_rule:
            L += ["DECISION RULES:"] + [f"  {i+1}. {r}" for i, r in enumerate(self.decision_rule)] + [""]
        if self.evidence_rules:
            L.append("EVIDENCE RULES:")
            for k, v in self.evidence_rules.items():
                L.append(f"  {k}:")
                for item in (v if isinstance(v, list) else [v]):
                    L.append(f"    - {item}")
            L.append("")
        if self.when_not_to_use:
            L += ["WHEN THIS SPEC DOES NOT APPLY:"] + [f"  - {x}" for x in self.when_not_to_use] + [""]
        if self.conflict_rules:
            L.append("CONFLICT RESOLUTION:")
            for c in self.conflict_rules:
                if isinstance(c, dict):
                    L.append(f"  - IF {c.get('if','?')} THEN {c.get('then','?')}")
                else:
                    L.append(f"  - {c}")
            L.append("")
        L.append("PROOF OBLIGATION:")
        L.append(f"  positive answer: {self.proof_obligation.for_positive}")
        if include_retrieval and self.proof_obligation.required_coverage:
            L.append("  BEFORE you may answer negative/absent you MUST have done all of:")
            for r in self.proof_obligation.required_coverage:
                L.append(f"    - {r}")
        if include_retrieval and self.proof_obligation.required_keywords:
            L.append(f"  required searches: {', '.join(self.proof_obligation.required_keywords)}")
        if include_retrieval and self.proof_obligation.required_doc_types:
            L.append(f"  document types that must be reviewed: {', '.join(self.proof_obligation.required_doc_types)}")
        st = self.proof_obligation.for_negative.get("statement") if self.proof_obligation.for_negative else None
        if st:
            L.append(_indent(st, 2))
        L.append("")
        L.append("ABSTENTION — these are different answers, choose deliberately:")
        for k, v in (self.abstention or {}).items():
            L.append(f"  {k}: {v}")
        L.append("")
        if self.boundary_cases:
            L.append("BOUNDARY CASES (these are settled; follow them):")
            for b in self.boundary_cases:
                L.append(f"  - {json.dumps(b, ensure_ascii=False) if isinstance(b, dict) else b}")
            L.append("")
        if include_retrieval and self.search_hints:
            L += ["SEARCH HINTS (suggestions, not a required path):",
                  "  " + ", ".join(self.search_hints), ""]
        return "\n".join(L)


def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + ln for ln in str(s).strip().splitlines())


def load_spec(path: str | Path) -> ExtractionSpec:
    """Load and freeze a spec. Raises if any ENFORCED element is unprovenanced.

    The check lives here rather than in a model validator so that a spec assembled in memory
    — a test fixture, an ablation arm built by transforming another spec — can still be
    constructed. Loading a FILE is the act that puts a rule into production, and it is the
    act that has to refuse.

    The EDITORIAL channel is deliberately not checked here and cannot stop a load. Call
    `editorial_findings(spec, specview.element_ids(spec))` to read it; the review document
    does, and prints what it finds.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = ExtractionSpec.model_validate(data)
    bind_provenance(spec)
    if spec.value_domain:
        # FAIL CLOSED ON A TYPO. A declared table that does not exist would otherwise render an
        # empty value domain into the prompt, and the run would look exactly like one that had
        # been given the codes. Same reason `acr.contract.skills` raises on a missing skill: a supplier of
        # guidance that silently supplies none is worse than one that is absent, because the
        # manifest reports that it was supplied.
        from .code_tables import load_table
        load_table(spec.value_domain)
    return spec


def load_specs(directory: str | Path) -> dict[str, ExtractionSpec]:
    out: dict[str, ExtractionSpec] = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        s = load_spec(p)
        out[s.spec_id] = s
    return out
