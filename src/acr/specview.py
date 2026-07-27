"""Render a spec for the clinician who owns its decisions, and record when one is confirmed.

Almost every genuinely clinical decision in this system is written in YAML. `establishes:
[primary_site]` on a group of imaging documents IS the ruling "radiology may tell us where
the tumour started". It is a sentence a thoracic oncologist would settle in four seconds,
and it was wrong for weeks because it was never a sentence — it was a key. P03 was coded
C349 (lung, NOS) while "right upper lobe" sat in seven other note types, and the spec's own
prose said outright that radiology can localise a mass. Nobody clinical had ever read the
file, because the file is 179 lines of regexes, group names and Clopper-Pearson bounds.

So this module answers one question: what would a certified registrar have to be shown in
order to catch that? Three properties follow from it, and `tests/test_specview.py` asserts
each one rather than trusting the prose:

  * Nothing engineering-shaped reaches the page. No regex, no group name, no gate key, no
    ICD-O code without its name. A reader who hits `cT(X|0|is|1mi|...)` stops reading, and
    the sentence they stopped before is where the clinical content was.

  * The made-up list is COMPLETE. Provenance defaults to `model_authored` — not to the
    manual named in `source_authority`, because naming the CoC manual at the top of a file
    is not evidence that the sentence three hundred lines down came out of it. The opposite
    default lets a physician approve a fabricated rule believing a standards body wrote it.
    That default makes section 6 long today. That length is the finding.

  * A sign-off dies when the text it approved changes. A reviewer's assent is to a specific
    wording; carrying it across an edit would manufacture clinical approval that nobody gave.

The element hash covers the spec_id, the element's kind and its RAW content — never its
rendered prose. Improving a sentence in this renderer must not invalidate a registrar's
signature, and moving a rule from position 4 to position 2 must not either. Editing its
words must.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

#: The enforced channel's identity function for an answer_check, imported rather than
#: re-derived: two spellings of "which check is this" would drift, and the drift would show
#: up as an attributed rule quietly reverting to unattributed.
from acr.spec import _answer_check_key

MODEL_AUTHORED = "model_authored"

SIGNED = "signed"
STALE = "stale"
UNSIGNED = "unsigned"

SECTION_TITLES = (
    "1. What this answers, and the patients it does not cover",
    "2. What we will accept as proof, and what we will refuse",
    "3. What we do when the chart contradicts itself",
    "4. When we refuse to answer, and what that refusal means downstream",
    "5. DECISIONS WE NEED YOU TO CONFIRM",
    "6. WHAT WE MADE UP",
    "7. HOW OFTEN THIS FIRES",
)

#: Vocabulary that must never reach a reviewer, mapped to what it actually means.
#:
#: Not a style rule. `may_mention` and `max_elusion_upper` are the words of the people who
#: wrote the bug; the reviewer is the person being asked to catch it, and every one of these
#: tokens is an invitation to skip the paragraph. The map is applied to EVERY fragment the
#: document emits — including prose harvested from the spec's own comments — so a term that
#: leaks is a term missing from this table, not a rendering bug somewhere else.
#:
#: Deliberately NOT here: the spec file's own top-level key names (`proof_obligation`,
#: `answer_checks`, `decision_rule`, ...). Those appear exactly once, in the appendix that
#: tells the reviewer where each part of the file went, and a reader who wants to check the
#: source needs the key to be spelled the way the file spells it.
JARGON: dict[str, str] = {
    "search_then_read_hits_and_sample_misses":
        "search it, read every hit, then read a sample of what the search missed",
    "exhaustive_until_witness": "read documents until one qualifying document is found, then stop",
    "exhaustive_per_window": "read every document in each follow-up window",
    "claim_scope_must_be_reported": "the answer must state the date through which it holds",
    "validate_by_sampling": "read a random sample to check the group really is silent",
    "required_doc_types_read": "document types that must be reviewed",
    "min_sample_of_misses": "how many unsearched documents we read anyway",
    "stratified_exclusion": "proof by covering the whole chart, group by group",
    "surveillance_schedule": "follow-up schedule",
    "max_tolerated_hits": "how many surprises we tolerate",
    "qualifying_doc_types": "document types that count",
    "max_elusion_upper": "the tolerated residual chance of a missed document",
    "empty_window_policy": "what an empty follow-up window means",
    "required_keywords": "required search terms",
    "cannot_establish": "documents we treat as unable to settle the answer",
    "required_coverage": "work that must be done first",
    "doc_type_matches": "document types",
    "clopper-pearson": "the standard statistical bound",
    "clopper": "the standard statistical bound",
    "can_establish": "documents that can settle the answer on their own",
    "not_less_specific": "do not code the vaguer value",
    "nos_requires_search": "an unknown value must be searched for first",
    "origin_not_specimen": "code where it started, not where it was cut",
    "specimen_markers": "wording that marks a specimen",
    "contradicted_by": "contradicted by",
    "requires_anchor": "cannot be evaluated without",
    "observable_period": "the period the chart can speak to",
    "partition_by": "divided by",
    "may_mention": "documents that may mention the answer",
    "allowable_values": "permitted values",
    "min_sample": "how many documents we read anyway",
    "nos_values": "unknown-value codes",
    "for_negative": "the rules for saying it is not there",
    "for_positive": "the rules for saying it is there",
    "doc_type": "document type",
    "stratified": "grouped",
    "stratify": "group",
    "stratum": "document group",
    "strata": "document groups",
    "elusion": "chance of a missed document",
    "witness": "one sufficient document",
}

_JARGON_RE = re.compile("|".join(re.escape(k) for k in sorted(JARGON, key=len, reverse=True)),
                        re.IGNORECASE)

#: ICD-O-3 names for every code the shipped specs use, plus the lung morphologies the
#: 2026-07-26 corpus measurement found the keyword list missing. A code the table does not
#: know is rendered with a warning rather than bare: a reviewer cannot approve `8046` and a
#: reviewer who is shown `8046` and nothing else will approve it anyway.
ICDO_NAMES: dict[str, str] = {
    "C340": "main bronchus", "C341": "upper lobe of lung", "C342": "middle lobe of lung",
    "C343": "lower lobe of lung", "C348": "overlapping lesion of lung",
    "C349": "lung, site within the lung not stated",
    "8000": "cancer, type not stated", "8010": "carcinoma, type not stated",
    "8012": "large cell carcinoma", "8013": "large cell neuroendocrine carcinoma",
    "8041": "small cell carcinoma", "8046": "non-small cell carcinoma, type not stated",
    "8070": "squamous cell carcinoma", "8071": "keratinising squamous cell carcinoma",
    "8072": "non-keratinising squamous cell carcinoma", "8083": "basaloid squamous cell carcinoma",
    "8140": "adenocarcinoma, type not stated", "8240": "carcinoid tumour",
    "8246": "neuroendocrine carcinoma", "8249": "atypical carcinoid tumour",
    "8250": "lepidic adenocarcinoma", "8253": "invasive mucinous adenocarcinoma",
    "8260": "papillary adenocarcinoma", "8480": "mucinous adenocarcinoma",
    "8550": "acinar adenocarcinoma", "8560": "adenosquamous carcinoma",
    "9050": "mesothelioma", "9680": "diffuse large B-cell lymphoma",
}

_CODE_RE = re.compile(r"\b(C\d{3}|[89]\d{3})\b")


# ------------------------------------------------------------------- measured consequence
@dataclass(frozen=True)
class Measurement:
    """One number from the 2026-07-26 corpus pass, with the configuration it was measured on.

    `applies_when` is what stops section 7 from becoming fiction. The 31.7% miss rate is a
    property of five specific search terms; reprinting it beside a sixth term would be a
    fabricated number wearing a measured number's clothes, and it would be the most
    persuasive sentence in the document.
    """
    key: str
    text: str
    #: Search terms whose presence in the spec makes this finding apply to it.
    needs_terms: tuple[str, ...] = ()
    #: spec_id this was measured on, when the finding is specific to one criterion.
    only_spec: str | None = None
    #: The exact term list measured, when the finding depends on the whole list.
    measured_list: tuple[str, ...] | None = None


CORPUS_HEADER = (
    "Measured on 2026-07-26 over the entire real corpus — 1,787 patients, 276,054 documents, "
    "read exhaustively with no sampling. The comparison was against each patient's known "
    "diagnosis, so a \"miss\" below means a document that states the answer in plain clinical "
    "prose and that this criterion's searches would never have opened."
)

MEASUREMENTS: tuple[Measurement, ...] = (
    Measurement(
        key="shb_keyword_miss",
        only_spec="STORE.400_522_523.site_histology_behavior",
        measured_list=("pathology", "biopsy", "final diagnosis", "specimen", "carcinoma"),
        text=(
            "**The required search terms miss the diagnosis on almost a third of patients.** "
            "Across the corpus, 4,005 progress notes, discharge summaries and consult notes "
            "state the patient's diagnosis in words none of the five required terms would "
            "find. That is 567 of 1,787 patients — **31.7%** — each holding at least one note "
            "that answers the question and that we would never have opened. In the group of "
            "documents we treat as inert, the same check finds 2,743 documents on 517 patients "
            "(28.9%).\n\n"
            "The cause is one missing word. The list has *carcinoma* and not *cancer*. "
            "Pathologists write carcinoma; almost nobody else does. Clinicians write \"small "
            "cell lung cancer\" (1,333 missed documents), \"squamous cell cancer\" (795), "
            "plain \"cancer\" (687), \"NSCLC\" (388), \"non-small cell\" (384), \"carcinoid\" "
            "(165). Adding the single term *cancer* recovers 3,605 of the 4,005.\n\n"
            "The notes carrying the missed answers are the ordinary ones: discharge summaries "
            "(631), general progress notes (624), haematology-oncology outpatient progress "
            "notes (558), primary-care progress notes (471), emergency department notes (447).\n\n"
            "Twelve of these documents were re-read by hand. All twelve stated the diagnosis "
            "and contained none of the five terms — \"scheduled to see oncology today for her "
            "small cell lung cancer\", \"stage iii squamous cell lung cancer\", \"residual "
            "typical carcinoid tumor\"."),
    ),
    Measurement(
        key="stem_pathology",
        needs_terms=("pathology",),
        text=(
            "**Searching for _pathology_ instead of _patholog_ loses 9,697 documents.** "
            "Corpus-wide the stem matches 37,721 documents and the full word 28,024. On 1,531 "
            "of 1,788 charts the stem returns strictly more; on no chart does it return fewer. "
            "The full word is the one written into the required list."),
    ),
    Measurement(
        key="resection_dead",
        needs_terms=("resection",),
        text=(
            "**_resection_ returns nothing at all on 43.1% of charts** — 770 of 1,788 — "
            "because most of these patients were never resected. It is still the fourth most "
            "useful term for rescuing the misses that remain after *cancer* is added."),
    ),
    Measurement(
        key="cytology_yield",
        needs_terms=("cytology",),
        text=("**_cytology_ matches 8,734 documents corpus-wide and none at all on 367 charts.**"),
    ),
    Measurement(
        key="carcinoma_yield",
        needs_terms=("carcinoma",),
        text=(
            "**_carcinoma_ is the single highest-yield term in the list** — on its own it is "
            "the only term that finds 8,325 of the answer-bearing documents — and it is also "
            "the term whose absence of a plain-English twin causes the miss rate above."),
    ),
    Measurement(
        key="final_diagnosis_dead",
        needs_terms=("final diagnosis",),
        text=(
            "**_final diagnosis_ is the only term that finds 6 documents out of 31,725.** It "
            "is a pathology-report heading, and it is required in a group of documents that "
            "contains no pathology reports. Removing it changes nothing."),
    ),
)

#: Where the 12% comes from, and why nothing below it is achievable. Applies to any criterion
#: that reads 25 documents and tolerates no surprises.
SAMPLING_ARITHMETIC = (
    "Reading 25 documents at random and finding nothing relevant supports a residual rate of "
    "**11.3%**, not zero — that is the strongest statement 25 documents can make. Any "
    "tolerance set below 11.3% cannot be met no matter how much work is done, so the number "
    "chosen and the number of documents read are one decision, not two."
)

UNMEASURED_NOTE = (
    "**Nothing in this criterion has been measured against the corpus.** One criterion has "
    "been — site, histology and behaviour — and its required search terms turned out to miss "
    "the stated diagnosis for 31.7% of patients. Silence here is not evidence that this "
    "criterion does better; it is evidence that nobody has looked."
)


# ------------------------------------------------------------------------------- elements
@dataclass(frozen=True)
class Element:
    """One reviewable statement, with an identity a reviewer can sign and an edit can break."""
    element_id: str
    kind: str
    text: str
    raw: Any
    provenance: str = MODEL_AUTHORED
    attributed_to: str = ""
    basis: str = ""
    section: int = 0
    #: False when NO provenance record names this statement in either channel. The default
    #: origin stays `model_authored` — that default is load-bearing and is asserted — but a
    #: default is not a finding, and section 6 has to be able to tell the reviewer which of
    #: the two it is looking at: somebody wrote down that a model made this up, or nobody
    #: wrote down anything.
    recorded: bool = False

    @property
    def element_hash(self) -> str:
        return _element_hash(self.spec_id, self.kind, self.raw)

    spec_id: str = ""


def _canonical(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)


def _element_hash(spec_id: str, kind: str, raw: Any) -> str:
    """spec_id + kind + content. Not position, and not rendered prose.

    Position is excluded so inserting a rule at the top does not silently unsign the six
    below it — nothing about those six changed. The rendered prose is excluded so that
    improving a sentence in this file does not destroy a registrar's signature. The kind is
    included so that moving a sentence out of the conflict rules and into the decision rules
    — which changes when it fires — does invalidate it.
    """
    blob = f"{spec_id}\x00{kind}\x00{_canonical(raw)}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class Decision:
    """A choice with a defensible alternative, phrased so a clinician can refuse it."""
    element_id: str
    question: str
    choice: str
    who: str
    basis: str
    if_you_disagree: str
    fires: str = ""
    rank: int = 50


# ------------------------------------------------------------------- plain-English helpers
def _plain(text: Any) -> str:
    """Strip the engineering vocabulary out of any fragment on its way to the page."""
    s = str(text or "").strip()
    if not s:
        return ""
    s = _JARGON_RE.sub(lambda m: JARGON[m.group(0).lower()], s)
    s = re.sub(r"\s+", " ", s)
    return s


def _gloss_codes(text: str) -> str:
    """`C349` -> `C349 (lung, site within the lung not stated)`.

    Skipped where the spec already glossed it inline, so the decision rule that reads
    "8000 (cancer, NOS) and 8010 (carcinoma, NOS) are NOT interchangeable" is left alone.
    """
    def repl(m: re.Match) -> str:
        code = m.group(0)
        if text[m.end():m.end() + 2] == " (":
            return code
        name = ICDO_NAMES.get(code)
        if name:
            return f"{code} ({name})"
        return (f"{code} (name not on file — this document cannot tell you what you would "
                f"be approving)")
    return _CODE_RE.sub(repl, text)


def _human_field(name: str) -> str:
    parts = str(name).split("_")
    return " ".join(p.upper() if len(p) == 1 else p for p in parts)


def _human_list(items: Iterable[Any], conj: str = "and") -> str:
    xs = [str(i).strip() for i in items if str(i).strip()]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + f" {conj} " + xs[-1]


def _quoted(items: Iterable[Any]) -> str:
    return _human_list([f"_{i}_" for i in items])


def _sentence(s: str) -> str:
    s = _plain(s)
    if s and s[-1] not in ".?!:":
        s += "."
    return s


# ------------------------------------------------------------ harvesting the WHY comments
_ELEMENT_START = re.compile(r"^\s*(-\s|[A-Za-z_][\w.\"']*\s*:)")
_CODEY = re.compile(r"[`(]\)|`|->|\.py\b|\bP3d\b|§|_check|\bre\.|\(\)")


def _comment_map(text: str) -> tuple[list[str], dict[int, str]]:
    """Line -> the comment block sitting directly above it.

    The house rule in this repo is that a decision is documented at the point it is made, in
    a comment. Those comments are the only record of WHY most of these choices were made, and
    `yaml.safe_load` throws every one of them away — so a review document built from the
    parsed spec alone can state what was decided and never on what basis.
    """
    lines = text.splitlines()
    buf: list[str] = []
    attached: dict[int, str] = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("#"):
            buf.append(s.lstrip("#").strip())
            continue
        if not s:
            buf = []
            continue
        if buf:
            attached[i] = " ".join(x for x in buf if x).strip()
        buf = []
    return lines, attached


def _basis_at(lines: Sequence[str], attached: dict[int, str], needle: str) -> str:
    """The comment above whichever line the content first appears on."""
    probe = re.sub(r"\s+", " ", str(needle)).strip()[:44]
    if len(probe) < 6:
        return ""
    hit = -1
    for i, ln in enumerate(lines):
        if probe in re.sub(r"\s+", " ", ln):
            hit = i
            break
    if hit < 0:
        return ""
    j = hit
    while j >= 0 and not _ELEMENT_START.match(lines[j]):
        j -= 1
    return _clean_comment(attached.get(j if j >= 0 else hit, ""))


def _clean_comment(raw: str) -> str:
    """Keep the sentences a clinician can use; drop the ones addressed to a maintainer.

    A comment like "graph._check_gate loops over required_keywords" is true, load-bearing and
    completely useless to the person being asked whether radiology may localise a primary.
    Sentences naming a function, a file or a backticked identifier go; the rest survives and
    then goes through the jargon map like everything else.
    """
    if not raw:
        return ""
    keep = [s for s in re.split(r"(?<=[.;])\s+", raw)
            if s.strip() and not _CODEY.search(s)]
    out = _plain(" ".join(keep))
    return out[:420].rstrip()


# ------------------------------------------------------------------------ format sanity
_REGEX_META = set(r"\[](){}|+*?^$.")


def _format_is_registry_notation(fmt: str) -> bool:
    """`CCYYMMDD` is a registry's way of writing "eight digits". It is not a regex.

    `check_field_formats` applies it with `re.fullmatch`, so the field accepts the literal
    string "CCYYMMDD" and rejects 20100612 and every other real date. STORE.390 and
    STORE.1860_1880 both ship it and both are still broken today. A review document that
    renders such a field as though it worked is worse than no document, so the test for it
    is deliberately crude and deliberately loud: no metacharacters, and letters that no
    plausible value would contain.
    """
    if not fmt:
        return False
    if any(ch in _REGEX_META for ch in fmt):
        return False
    return bool(re.fullmatch(r"[A-Z]{4,}", fmt))


# ------------------------------------------------------------------------ source groups
@dataclass(frozen=True)
class SourceGroup:
    index: int
    label: str
    policy: str
    establishes: list[str]
    keywords: list[str]
    numbers: dict[str, Any]
    basis: str
    claim: str
    raw: dict


_POLICY_PROSE = {
    "exhaustive": "we read every document in this group",
    "exhaustive_until_witness": "we read documents in this group until one of them answers, then stop",
    "exhaustive_per_window": "we read every document in each follow-up window",
    "search_then_read_hits_and_sample_misses":
        "we search this group, read every document the search hits, and read a sample of the rest",
    "validate_by_sampling": "we do not read this group; we read a random sample to check it really is silent",
}


def _group_label(raw: dict) -> str:
    m = raw.get("match") or {}
    if m.get("rest"):
        return "Every other document in the chart"
    types = list(m.get("doc_type_matches") or [])
    if not types:
        return "Documents this group does not say how to select"
    shown = ", ".join(t.replace("-", " ") for t in types)
    return f"Any document whose type name contains: {shown}"


def _source_groups(spec) -> list[SourceGroup]:
    fn = _for_negative(spec)
    out: list[SourceGroup] = []
    buckets: list[tuple[str, dict]] = [("", s) for s in (fn.get("strata") or [])]
    for claim in (fn.get("claims") or []):
        cid = str(claim.get("id", ""))
        buckets += [(cid, s) for s in (claim.get("strata") or [])]
    for i, (claim, s) in enumerate(buckets, start=1):
        nums = {k: s[k] for k in ("min_sample", "min_sample_of_misses", "max_tolerated_hits")
                if k in s}
        out.append(SourceGroup(
            index=i, label=_group_label(s), policy=str(s.get("policy", "")),
            establishes=list(s.get("establishes") or []),
            keywords=list(s.get("required_keywords") or []),
            numbers=nums, basis="", claim=claim, raw=s))
    return out


def _for_negative(spec) -> dict:
    po = getattr(spec, "proof_obligation", None)
    return dict(getattr(po, "for_negative", {}) or {}) if po else {}


def _extra(spec) -> dict:
    return dict(getattr(spec, "model_extra", None) or {})


# ------------------------------------------------------------------------ element harvest
def elements(spec, source_path: str | Path | None = None) -> list[Element]:
    """Every statement in the spec that a reviewer could agree or disagree with."""
    lines: list[str] = []
    attached: dict[int, str] = {}
    if source_path:
        lines, attached = _comment_map(Path(source_path).read_text(encoding="utf-8"))

    declared = _provenance_map(spec)
    out: list[Element] = []

    def add(eid: str, kind: str, text: str, raw: Any, section: int, basis_needle: Any = None,
            enforced: str | Sequence[str] | None = None):
        # Either channel may speak for this statement. The editorial record wins when both
        # exist, because it was written about the sentence the reviewer is reading; the
        # enforced record is consulted by `enforced` path so that a keyword list already
        # attributed in the enforced block does not get listed here as unattributed.
        paths = [enforced] if isinstance(enforced, str) else list(enforced or [])
        rec = declared.get(eid) or next((declared[p] for p in paths if p in declared), None)
        prov, who = (rec if rec else (MODEL_AUTHORED, ""))
        basis = _basis_at(lines, attached, basis_needle if basis_needle is not None else raw)
        out.append(Element(element_id=eid, kind=kind, text=_plain(text), raw=raw,
                           provenance=prov, attributed_to=who, basis=basis,
                           section=section, spec_id=spec.spec_id, recorded=rec is not None))

    add("authority", "authority", _authority_line(spec), dict(spec.source_authority or {}), 1)
    add("question", "question", spec.question, str(spec.question).strip(), 1)
    if spec.agent_policy:
        add("guidance", "guidance", spec.agent_policy, str(spec.agent_policy).strip(), 1)

    for f in spec.fields:
        add(f"answer.{f.name}", "answer", _field_line(f), f.model_dump(mode="json"), 1,
            enforced=[f"fields[{f.name}].format", f"fields[{f.name}].allowable_values"])
    for i, w in enumerate(spec.when_not_to_use, start=1):
        add(f"not-for.{i}", "not-for", w, w, 1)

    for i, r in enumerate(spec.decision_rule, start=1):
        add(f"rule.{i}", "rule", r, r, 2)
    for i, e in enumerate((spec.evidence_rules or {}).get("counts_as_evidence") or [], start=1):
        add(f"accept.{i}", "accept", e, e, 2)
    for i, e in enumerate((spec.evidence_rules or {}).get("does_not_count") or [], start=1):
        add(f"refuse.{i}", "refuse", e, e, 2)
    pos = spec.proof_obligation.positive_statement
    if pos:
        add("proof.positive", "proof", pos, pos, 2)
    for fname, groups in (spec.proof_obligation.witness_strata or {}).items():
        add(f"proof.witness.{fname}", "proof",
            f"A citation for {_human_field(fname)} is only accepted from: "
            f"{_human_list(groups)}.", {fname: groups}, 2)

    for g in _source_groups(spec):
        base = f"source-group.{g.index}"
        # The enforced channel splits one document group across up to four records (how it is
        # matched, what it may establish, what must be searched, how much is sampled). The
        # reviewer sees one sentence and two sub-bullets, so the group sentence answers to
        # whichever of match/establishes carries a record.
        ebase = (f"proof_obligation.for_negative"
                 + (f".claims[{g.claim}]" if g.claim else "") + f".strata[{g.raw.get('name')}]")
        add(base, "source-group", _group_sentence(g), g.raw, 2,
            basis_needle=(g.raw.get("name") or g.label),
            enforced=[f"{ebase}.match", f"{ebase}.partition_by", f"{ebase}.establishes"])
        if g.keywords:
            add(f"{base}.terms", "search-terms",
                f"Before we may say the answer is not in this group we must have searched it "
                f"for {_quoted(g.keywords)}.", list(g.keywords), 2,
                basis_needle=g.keywords[0], enforced=f"{ebase}.required_keywords")
        if g.numbers:
            add(f"{base}.sample", "sampling", _sample_sentence(g), dict(g.numbers), 2,
                enforced=[f"{ebase}.min_sample", f"{ebase}.min_sample_of_misses"])

    fn = _for_negative(spec)
    if fn.get("required_keywords"):
        add("search-terms", "search-terms",
            f"Before any \"not documented\" answer is accepted, the chart must have been "
            f"searched for {_quoted(fn['required_keywords'])}.",
            list(fn["required_keywords"]), 2, basis_needle=fn["required_keywords"][0],
            enforced="proof_obligation.for_negative.required_keywords")
    if fn.get("confidence") is not None or (fn.get("gate") or {}).get("max_elusion_upper") is not None:
        add("certainty", "certainty", _certainty_sentence(fn),
            {"confidence": fn.get("confidence"),
             "max_elusion_upper": (fn.get("gate") or {}).get("max_elusion_upper")}, 2)
    for claim in (fn.get("claims") or []):
        cid = str(claim.get("id", "claim"))
        add(f"claim.{cid}", "claim", _claim_sentence(claim), claim, 2)
    if fn.get("reason"):
        add("proof.not-applicable", "proof", fn["reason"], fn["reason"], 1)

    for i, c in enumerate(spec.conflict_rules, start=1):
        add(f"conflict.{i}", "conflict", _conflict_line(c), c, 3)

    neg = fn.get("statement")
    if neg:
        add("proof.negative", "proof", neg, str(neg).strip(), 4)
    for k, v in (spec.abstention or {}).items():
        add(f"refusal.{k}", "refusal", f"{_refusal_label(k)} — {v}", {k: v}, 4)
    for i, s in enumerate(spec.special_codes_not_mar, start=1):
        add(f"caution.{i}", "caution", s, s, 4)
    for i, d in enumerate(spec.downstream_warning, start=1):
        add(f"downstream.{i}", "downstream", d, d, 4)
    if fn.get("on_failure"):
        add("on-refusal", "refusal", _on_failure_line(fn["on_failure"]), fn["on_failure"], 4)

    for i, b in enumerate(spec.boundary_cases, start=1):
        add(f"boundary.{i}", "boundary", _boundary_line(b), b, 5,
            basis_needle=(b.get("case") if isinstance(b, dict) else b))
    for i, c in enumerate(spec.answer_checks, start=1):
        add(f"check.{i}", "check", _check_line(c), c, 5,
            basis_needle=(c.get("message") if isinstance(c, dict) else c),
            enforced=(f"answer_checks[{_answer_check_key(c)}]" if isinstance(c, dict) else None))

    for key, val in _extra(spec).items():
        if key in ("provenance", "keyword_field_coverage"):
            continue
        if isinstance(val, dict):
            for sub, sv in val.items():
                add(f"{_slug(key)}.{_slug(sub)}", "convention",
                    f"Where the record says \"{sub}\", we record {sv}.", {sub: sv}, 5)
        elif isinstance(val, list):
            for i, item in enumerate(val, start=1):
                add(f"{_slug(key)}.{i}", "convention", str(item), item, 5)
    if spec.search_hints:
        add("hints", "hints",
            f"Suggested (not required) search terms: {_quoted(spec.search_hints)}.",
            list(spec.search_hints), 2)
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _authority_line(spec) -> str:
    sa = spec.source_authority or {}
    bits = []
    if sa.get("document"):
        bits.append(f"Written against {sa['document']}")
    if sa.get("items"):
        bits.append(f"for the registry items {_human_list(sa['items'])}")
    if sa.get("applicability"):
        bits.append(f"and applies to {sa['applicability']}")
    return ("; ".join(bits) + ".") if bits else "No source document is named."


def _field_line(f) -> str:
    bits = [f"**{_human_field(f.name)}** — {f.description or 'no description given'}"]
    if f.allowable_values:
        vals = list(f.allowable_values)
        named = [f"{v} ({ICDO_NAMES[str(v)]})" if str(v) in ICDO_NAMES else str(v) for v in vals]
        if len(vals) > 12:
            bits.append(f"The spec permits {len(vals)} values and names none of them in the "
                        f"file: {_human_list(named)}. A reviewer cannot confirm a value list "
                        f"they cannot read.")
        else:
            bits.append(f"Permitted values: {_human_list(named)}.")
    if f.format and _format_is_registry_notation(f.format):
        bits.append(f"**This field is broken today.** The file declares its shape as "
                    f"`{f.format}`, which is registry notation, but the software applies it "
                    f"as a literal pattern — so it will reject every valid value of this "
                    f"field, including every real date. Unfixed as of 2026-07-26.")
    elif f.format:
        bits.append("Its shape is checked automatically against a fixed pattern.")
    if f.nullable is False:
        bits.append("The registry does not allow this field to be left blank.")
    return " ".join(bits)


def _group_sentence(g: SourceGroup) -> str:
    bits = [f"**{g.label}.** "]
    bits.append(_POLICY_PROSE.get(g.policy, f"handling: {g.policy}") + ".")
    if g.establishes:
        bits.append(f" A document from this group may be cited as the evidence for "
                    f"{_human_list([_human_field(x) for x in g.establishes])}.")
    elif g.raw.get("establishes") == []:
        bits.append(" Nothing in this group may be cited as evidence for anything.")
    if g.claim:
        bits.append(f" (Used only for the part of the answer about "
                    f"{_human_field(g.claim).replace('-', ' ')}.)")
    return "".join(bits)


def _sample_sentence(g: SourceGroup) -> str:
    n = g.numbers.get("min_sample_of_misses") or g.numbers.get("min_sample")
    tol = g.numbers.get("max_tolerated_hits")
    bits = []
    if n:
        bits.append(f"We read {n} documents from this group at random.")
    if tol == 0:
        bits.append("If even one of them turns out to be relevant, the answer is rejected and "
                    "the work starts again.")
    elif tol is not None:
        bits.append(f"Up to {tol} of them may turn out to be relevant before the answer is "
                    f"rejected.")
    return " ".join(bits)


def _certainty_sentence(fn: dict) -> str:
    conf = fn.get("confidence")
    cap = (fn.get("gate") or {}).get("max_elusion_upper")
    bits = []
    if cap is not None:
        bits.append(f"We accept a \"not documented\" answer when the residual chance that a "
                    f"relevant document was missed is at or below {cap * 100:.0f}%.")
    if conf is not None:
        bits.append(f"That chance is computed at {conf * 100:.0f}% confidence.")
    return " ".join(bits)


def _claim_sentence(claim: dict) -> str:
    cid = _human_field(str(claim.get("id", ""))).replace("-", " ")
    bits = [f"**{cid.capitalize()}** must be proved separately."]
    if claim.get("requires_anchor"):
        bits.append("It cannot even be assessed until the other half is established.")
    w = claim.get("window")
    if w:
        bits.append("It is proved over a defined stretch of time, not over the chart as a whole.")
    if claim.get("claim_scope_must_be_reported"):
        bits.append("The answer must state the date through which it holds.")
    return " ".join(bits)


def _conflict_line(c: Any) -> str:
    if isinstance(c, dict):
        return f"If {c.get('if', '?')}, then {c.get('then', '?')}"
    return str(c)


def _refusal_label(k: str) -> str:
    return {"EVIDENCE_INSUFFICIENT": "\"The chart does not say\"",
            "SPEC_INSUFFICIENT": "\"These instructions do not cover this case\""}.get(k, k)


def _on_failure_line(of: dict) -> str:
    val = of.get("value") or {}
    bits = ["When the work cannot be completed we answer \"the chart does not say\""]
    if val:
        bits.append("and record " + _human_list([f"{_human_field(k)} = {v}" for k, v in val.items()]))
    if of.get("must_report"):
        bits.append("and we must report why, and which stretches of time were left uncovered")
    return " ".join(bits) + "."


def _boundary_line(b: Any) -> str:
    if not isinstance(b, dict):
        return str(b)
    case = b.get("case", "")
    verdict = {k: v for k, v in b.items() if k not in ("case", "why")}
    ans = _human_list([f"{_human_field(k)} = {v}" for k, v in verdict.items()])
    out = f"\"{case}\" → {ans or 'no answer recorded'}"
    if b.get("why"):
        out += f". Because: {b['why']}"
    return out


def _check_line(c: Any) -> str:
    if not isinstance(c, dict):
        return str(c)
    fld = _human_field(c.get("field", ""))
    kind = str(c.get("kind", ""))
    if kind == "not_less_specific":
        return (f"{fld}: the values {_human_list(c.get('nos_values') or [])} are refused "
                f"whenever the evidence the run itself quoted contains any of "
                f"{_quoted(c.get('contradicted_by') or [])}. {c.get('message', '')}")
    if kind == "nos_requires_search":
        return (f"{fld}: the values {_human_list(c.get('nos_values') or [])} are refused "
                f"unless the chart was first searched for "
                f"{_quoted(c.get('required_searches') or [])}. {c.get('message', '')}")
    if kind == "origin_not_specimen":
        return (f"{fld}: a value is refused when the quote supporting it is the description "
                f"of a specimen rather than of the tumour's origin. {c.get('message', '')}")
    return f"{fld}: {kind}. {c.get('message', '')}"


# ------------------------------------------------------------------------------ provenance
def _provenance_map(spec) -> dict[str, tuple[str, str]]:
    """Both provenance channels, flattened to element key -> (origin, basis).

    Keys from two namespaces land in one dict on purpose, and they cannot collide: the
    enforced channel is keyed by runtime paths
    (`proof_obligation.for_negative.strata[...].required_keywords`) and the editorial channel
    by the ids this module prints (`rule.1`, `conflict.2`, `boundary.3`). `elements()` looks
    up its own id first and the enforced path second.

    Both are first-class fields on `ExtractionSpec`, not extras, so neither shows up in
    `model_extra`. This used to read `_extra(spec).get("provenance")`, which found nothing for
    every shipped spec and defaulted every element to `model_authored` regardless of what was
    actually declared; a properly-sourced element would have silently landed in "what we made
    up" beside one that really was invented.

    The editorial side is read through `spec_mod.editorial_records`, which drops malformed
    records and returns them as findings rather than raising. A record that fails the honesty
    rules -- a `store_manual` origin naming no section, say -- therefore does NOT take its
    element out of section 6. That is the point: the escape from "we made this up" is a
    citation, not a label.
    """
    out: dict[str, tuple[str, str]] = {}
    for rec in getattr(spec, "provenance", None) or []:
        out[str(rec.element)] = (str(rec.origin), str(getattr(rec, "basis", "") or ""))
    for element, rec in editorial_index(spec).items():
        out[element] = (str(rec.origin), str(getattr(rec, "basis", "") or ""))
    return out


def editorial_index(spec) -> dict:
    """The valid editorial records, by element id. Malformed ones are dropped, not raised."""
    from acr.spec import editorial_records
    try:
        index, _ = editorial_records(spec)
    except Exception:  # a spec object from somewhere else that has no editorial channel
        return {}
    return index


def element_ids(spec, source_path: str | Path | None = None) -> list[str]:
    """Every id the review document prints — the editorial channel's namespace.

    `acr.spec.editorial_findings` needs this to tell an unattributed statement from a record
    naming a statement that no longer exists. It lives here because this module decides what
    a reviewable statement is; the loader deliberately does not know.
    """
    return [e.element_id for e in elements(spec, source_path=source_path)]


def provenance_findings(spec, source_path: str | Path | None = None) -> list:
    """`editorial_findings`, told which statements the ENFORCED block already covers.

    The single call every caller should use. Going to `acr.spec.editorial_findings` directly
    with only the id list reports a keyword list as unattributed even when the enforced block
    attributes it, because only this module knows that `source-group.2.terms` and
    `...strata[may_mention].required_keywords` are the same rule seen from two sides.
    """
    from acr.spec import editorial_findings
    els = elements(spec, source_path=source_path)
    return editorial_findings(spec, [e.element_id for e in els],
                              attributed_elsewhere=[e.element_id for e in els if e.recorded])


def _who(el: Element, spec, status: str, record: dict | None) -> str:
    if status == SIGNED and record:
        return f"Confirmed by {record['reviewer']} on {record['signed_at'][:10]}."
    if status == STALE and record:
        return (f"{record['reviewer']} confirmed a different wording on "
                f"{record['signed_at'][:10]}; that approval no longer applies to the text "
                f"below.")
    if el.provenance == MODEL_AUTHORED and not el.recorded:
        return ("Nobody clinical, and the origin is not recorded. Nothing in the file says "
                "where this sentence came from, so it is treated as model-authored — that is "
                "the safe reading, not a checked one.")
    if el.provenance == MODEL_AUTHORED:
        return ("Nobody clinical. The file records that a language model drafted this and no "
                "reviewer has confirmed it."
                + (f" {_sentence(el.attributed_to)}" if el.attributed_to else ""))
    if el.provenance == "source_authority":
        doc = (spec.source_authority or {}).get("document", "the cited standard")
        return f"Taken from {doc}." + (f" Recorded by {el.attributed_to}." if el.attributed_to else "")
    if el.provenance == "measured":
        return "Chosen from a measurement on this corpus rather than from a guideline."
    who = f"Recorded provenance: {el.provenance}."
    return who + (f" By {el.attributed_to}." if el.attributed_to else "")


# ------------------------------------------------------------------------------- sign-off
def _ledger_path(directory: str | Path, spec_id: str) -> Path:
    return Path(directory) / f"{spec_id}.jsonl"


def load_signoffs(directory: str | Path, spec_id: str) -> list[dict]:
    p = _ledger_path(directory, spec_id)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def record_signoff(directory: str | Path, spec, element_id: str, *, reviewer: str,
                   source_path: str | Path | None = None, note: str = "") -> dict:
    """Append one reviewer's assent to one element. Append-only, like every other ledger here.

    Rewriting or de-duplicating this file would destroy the only record that somebody once
    approved a wording that has since changed — which is exactly the history a re-review
    needs to see.
    """
    els = {e.element_id: e for e in elements(spec, source_path=source_path)}
    el = els.get(element_id)
    if el is None:
        import difflib
        near = difflib.get_close_matches(element_id, list(els), n=5, cutoff=0.3)
        raise KeyError(
            f"no element {element_id!r} in {spec.spec_id}. "
            + (f"did you mean {', '.join(near)}? " if near else "")
            + f"ids in this spec: {', '.join(sorted(els))}")
    rec = {
        "signed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reviewer": reviewer,
        "spec_id": spec.spec_id,
        "spec_version": spec.spec_version,
        "spec_hash": spec.spec_hash,
        "element_id": el.element_id,
        "element_kind": el.kind,
        "element_hash": el.element_hash,
        "note": note,
    }
    p = _ledger_path(directory, spec.spec_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def signoff_status(el: Element, signoffs: Sequence[dict]) -> tuple[str, dict | None]:
    """SIGNED only when the approved hash is still the element's hash.

    Matching is by hash and not by element id, so a rule that moved keeps its approval and a
    rule that was reworded loses it. STALE is reported rather than dropped: "somebody
    approved a different version of this" is information the next reviewer needs.
    """
    mine = [s for s in signoffs
            if s.get("spec_id") == el.spec_id and s.get("element_id") == el.element_id]
    exact = [s for s in signoffs
             if s.get("spec_id") == el.spec_id and s.get("element_hash") == el.element_hash]
    if exact:
        return SIGNED, sorted(exact, key=lambda s: s["signed_at"])[-1]
    if mine:
        return STALE, sorted(mine, key=lambda s: s["signed_at"])[-1]
    return UNSIGNED, None


# ------------------------------------------------------------------------------ decisions
def decisions(spec, source_path: str | Path | None = None,
              els: Sequence[Element] | None = None) -> list[Decision]:
    """The subset of elements where a different defensible clinical choice exists.

    Not every made-up element is a decision. If section 5 listed all of them it would be
    section 6 with numbers, and a registrar with ten minutes would read neither. The filter
    is: could a competent clinician choose otherwise, and would the outputs differ? A
    restatement of a CoC coding rule fails that test; a tolerance, a witness restriction, a
    hedge-acceptance rule, an imputation constant and a settled boundary ruling all pass it.
    """
    els = list(els if els is not None else elements(spec, source_path=source_path))
    by_id = {e.element_id: e for e in els}
    fn = _for_negative(spec)
    out: list[Decision] = []

    def push(eid: str, question: str, choice: str, consequence: str, rank: int, fires: str = ""):
        el = by_id.get(eid)
        out.append(Decision(
            element_id=eid, question=_plain(question), choice=_plain(choice),
            who=_who(el, spec, UNSIGNED, None) if el else "Nobody clinical.",
            basis=(el.basis if el else ""), if_you_disagree=_plain(consequence),
            fires=fires, rank=rank))

    # 1. A placeholder is the reviewer's question by construction, so it goes first.
    for g in _source_groups(spec):
        sched = g.raw.get("surveillance_schedule")
        if isinstance(sched, str) and "PLACEHOLDER" in sched.upper():
            push(f"source-group.{g.index}",
                 "How often must a patient have been seen for us to believe that no "
                 "recurrence happened between visits?",
                 "Undecided. The specification ships an explicit placeholder here and the "
                 "software has no follow-up schedule to work from, so this half of the "
                 "criterion cannot run at all.",
                 "Everything: give a schedule — for example three-monthly for two years, "
                 "then six-monthly to year five, then yearly — and a gap longer than the "
                 "interval becomes a gap in evidence rather than a silent assumption that "
                 "nothing happened. A constant interval is indefensible in both directions: "
                 "too coarse in the first two years, too fine after five.",
                 rank=0)

    # 2. Who may say what. This is the decision that was wrong on P03.
    for g in _source_groups(spec):
        eid = f"source-group.{g.index}"
        if g.raw.get("establishes") == []:
            push(eid,
                 f"Is it right that nothing in this group of documents can ever contribute to "
                 f"any part of the answer?",
                 f"{g.label}. We treat every one of these as incapable of contributing, and "
                 f"we only sample them to check that assumption.",
                 "If any of these can carry a finding that matters — a spine film showing a "
                 "bone metastasis, say — then sampling it will keep rejecting good answers, "
                 "and the honest fix is to move that type into the group we search.",
                 rank=5)
        elif g.establishes and len(g.establishes) < _n_fields(spec):
            missing = [f.name for f in spec.fields if f.name not in g.establishes]
            push(eid,
                 f"May {_lower_first(g.label)} be used to establish "
                 f"{_human_list([_human_field(x) for x in g.establishes])} — and is it right "
                 f"that they may not be used for "
                 f"{_human_list([_human_field(x) for x in missing])}?",
                 f"Yes to the first, no to the second. {_group_sentence(g)}",
                 "If you say imaging may not be used to say where the tumour started, the "
                 "answer becomes the unspecified-site code whenever the pathology report "
                 "says only \"right lung\" — which is most of them. If you say it may be "
                 "used for more than this, that evidence is currently being thrown away.",
                 rank=10)

    # 3. The words we must have searched before we are allowed to say "not documented".
    terms_ids = [e.element_id for e in els if e.kind == "search-terms"]
    for eid in terms_ids:
        el = by_id[eid]
        m = _measurement_for(spec, list(el.raw))
        push(eid,
             "Are these the words a clinician at this hospital would actually use to state "
             "this answer?",
             el.text,
             "Add a word and more of the chart gets read before we are allowed to say the "
             "answer is absent; remove one and less does. This list is the whole of what "
             "stands between a real answer and a \"not documented\".",
             rank=15,
             fires=(m.text.split("\n\n")[0] if m else ""))

    # 4. The tolerance, shown with the sample size that produces it.
    if "certainty" in by_id:
        cap = (fn.get("gate") or {}).get("max_elusion_upper")
        n = _sample_size(spec)
        push("certainty",
             f"Is a residual {cap * 100:.0f}% chance that a relevant document was missed "
             f"acceptable for this variable?" if cap is not None else
             "Is this level of certainty acceptable for this variable?",
             (by_id['certainty'].text + f" Reading {n} documents at random and finding nothing "
              f"is what produces that number." if n else by_id["certainty"].text),
             "Lower it and more documents must be read for every patient; below about 11% it "
             "cannot be met at all at the current sample size, and every answer fails. Raise "
             "it and \"not documented\" gets easier to claim.",
             rank=20, fires=SAMPLING_ARITHMETIC)

    # 5..6. The mechanical checks, grouped by kind. One decision per kind, not per rule:
    # four separate items about undivided T categories are one clinical question wearing four
    # hats, and splitting them is how a ten-minute document becomes a forty-minute one.
    checks = [(e, e.raw) for e in els if e.kind == "check" and isinstance(e.raw, dict)]
    for kind, question, consequence in (
        ("not_less_specific",
         "When the record hedges or is only partly specific, must we still code the more "
         "specific value?",
         "If a hedge is not enough, these answers stop being rejected and the vaguer code "
         "stands — a report reading \"favor squamous cell carcinoma\" would be coded as the "
         "unspecified non-small cell code."),
        ("nos_requires_search",
         "Before we are allowed to record an unknown or unspecified value, which parts of the "
         "chart must have been searched?",
         "Drop the requirement and an unknown code becomes available to a run that simply "
         "did not look; add to it and more searching is forced before any unknown is "
         "accepted."),
        ("origin_not_specimen",
         "Should a value be refused when the only quote supporting it describes the specimen "
         "rather than the tumour's origin?",
         "Remove it and the biopsy site gets coded as the site of origin, which is the "
         "commonest single coding error on this variable."),
    ):
        group = [(e, c) for e, c in checks if c.get("kind") == kind]
        if not group:
            continue
        body = "\n".join(f"  - {_plain(e.text)}" for e, _ in group)
        push(group[0][0].element_id, question, "Currently:\n" + body, consequence, rank=25)

    # 7. Conventions the spec invents outright (seasonal dates, and anything like them).
    for e in els:
        if e.kind == "convention":
            push(e.element_id,
                 f"Is this an acceptable convention: {_lower_first(e.text).rstrip('.')}?",
                 e.text,
                 "This value is invented. It lands in a date field that feeds survival time, "
                 "so choosing a different month moves every affected patient's follow-up by "
                 "up to three months.",
                 rank=30)

    # 8. Rulings the spec declares settled.
    for e in els:
        if e.kind == "boundary":
            push(e.element_id,
                 "Is this ruling correct?",
                 e.text,
                 "The spec tells the run to follow these without further thought, so a wrong "
                 "one is wrong on every patient it touches and will never be questioned.",
                 rank=35)

    # 9. Which patients get no answer at all.
    for e in els:
        if e.kind == "not-for":
            push(e.element_id,
                 "Should these patients really be excluded from this variable?",
                 e.text,
                 "An excluded patient gets no answer and no coverage claim; they drop out of "
                 "every denominator downstream, silently, unless somebody counts them.",
                 rank=40)

    # 10. Which document wins when two disagree — a registrar's judgement, not an engineer's.
    for e in els:
        if e.kind == "conflict":
            push(e.element_id,
                 "When these two disagree, is this the right one to believe?",
                 e.text,
                 "Reverse it and the other document's value is reported instead; both are in "
                 "the chart, so this choice alone decides the answer for those patients.",
                 rank=45)

    # 11. A field whose declared shape rejects every valid value is a decision, not a typo:
    # somebody has to say what the field should actually accept.
    for f in spec.fields:
        if f.format and _format_is_registry_notation(f.format):
            push(f"answer.{f.name}",
                 f"What should {_human_field(f.name)} actually accept?",
                 f"As shipped, nothing. The file declares its shape as `{f.format}` — registry "
                 f"notation — and the software applies it literally, so it will reject every "
                 f"valid value of this field. Still unfixed on 2026-07-26.",
                 "Nothing downstream can use this field until it is answered. Confirm that "
                 "eight digits with an unknown month or day written as 99 is what the registry "
                 "expects, and the pattern can be written correctly.",
                 rank=1)

    out.sort(key=lambda d: (d.rank, d.element_id))
    return out


def _n_fields(spec) -> int:
    return max(1, len(spec.fields))


def _lower_first(s: str) -> str:
    s = _plain(s).lstrip("*")
    return (s[0].lower() + s[1:]) if s else s


def _sample_size(spec) -> int | None:
    for g in _source_groups(spec):
        n = g.numbers.get("min_sample_of_misses") or g.numbers.get("min_sample")
        if n:
            return int(n)
    return None


def _measurement_for(spec, terms: Sequence[str]) -> Measurement | None:
    for m in MEASUREMENTS:
        if m.only_spec == spec.spec_id and m.measured_list is not None:
            return m if tuple(terms) == m.measured_list else None
    return None


# --------------------------------------------------------------------------------- render
def render_review(spec, source_path: str | Path | None = None,
                  signoffs: Sequence[dict] = ()) -> str:
    els = elements(spec, source_path=source_path)
    ds = decisions(spec, source_path=source_path, els=els)
    L: list[str] = []
    A = L.append

    A(f"# {_plain(spec.question).rstrip('.')}")
    A("")
    A(f"A review copy of the extraction specification `{spec.spec_id}` "
      f"(version {spec.spec_version}, content fingerprint `{spec.spec_hash}`).")
    A("")
    A("You are being asked to mark this up. Section 5 is the list of choices we need you to "
      "confirm or overturn; section 6 is everything in here that we invented and nobody "
      "clinical has ever checked. If you read only two sections, read those two.")
    A("")

    _s1(A, spec, els)
    _s2(A, spec, els, signoffs)
    _s3(A, spec, els, signoffs)
    _s4(A, spec, els, signoffs)
    _s5(A, spec, ds, els, signoffs)
    _s6(A, spec, els, signoffs)
    _s7(A, spec)
    _appendix(A, spec, source_path)

    body = "\n".join(L)
    return _gloss_codes(body) + "\n"


def _h(A, i: int) -> None:
    A("")
    A(f"## {SECTION_TITLES[i]}")
    A("")


def _pick(els: Sequence[Element], *kinds: str) -> list[Element]:
    return [e for e in els if e.kind in kinds]


def _mark(el: Element, signoffs: Sequence[dict]) -> str:
    status, rec = signoff_status(el, signoffs)
    if status == SIGNED:
        return f"  _[{el.element_id} — confirmed by {rec['reviewer']}, {rec['signed_at'][:10]}]_"
    if status == STALE:
        return (f"  _[{el.element_id} — {rec['reviewer']} confirmed a different wording on "
                f"{rec['signed_at'][:10]}; that approval no longer applies]_")
    return f"  _[{el.element_id}]_"


def _bullets(A, els: Sequence[Element], signoffs: Sequence[dict], empty: str) -> None:
    if not els:
        A(f"_{empty}_")
        A("")
        return
    for e in els:
        A(f"- {_sentence(e.text)}{_mark(e, signoffs)}")
    A("")


def _s1(A, spec, els) -> None:
    _h(A, 0)
    if spec.data_source == "outside_notes":
        A("> **This variable is not in the chart at all.** It is a fact about the patient's "
          "relationship to this hospital — where they were diagnosed and where they were "
          "treated — and it lives in registration, referral and billing systems. No amount "
          "of reading notes can establish it. The run is required to refuse it and to report "
          "clues only; a human registrar assigns the code.")
        A("")
    by = {e.element_id: e for e in els}
    A(f"**The question.** {_sentence(spec.question)}")
    A("")
    if "authority" in by:
        A(f"{_sentence(by['authority'].text)}")
        A("")
    if "guidance" in by:
        A(f"**Standing instruction to the reviewer software.** {_sentence(by['guidance'].text)}")
        A("")
    if "proof.not-applicable" in by:
        A(f"{_sentence(by['proof.not-applicable'].text)}")
        A("")
    A("**What comes out.**")
    A("")
    for e in _pick(els, "answer"):
        A(f"- {e.text}")
    A("")
    A("**Patients this does not cover.**")
    A("")
    nf = _pick(els, "not-for")
    if nf:
        for e in nf:
            A(f"- {_sentence(e.text)}")
    else:
        A("_The specification names no patients it excludes. If there are any, nobody has "
          "written them down, and the run will attempt an answer for every patient it is "
          "given._")
    A("")


def _s2(A, spec, els, signoffs) -> None:
    _h(A, 1)
    by = {e.element_id: e for e in els}
    if "proof.positive" in by:
        A(f"**What is enough to give an answer.** {_sentence(by['proof.positive'].text)}"
          f"{_mark(by['proof.positive'], signoffs)}")
        A("")
    wit = [e for e in els if e.element_id.startswith("proof.witness.")]
    if wit:
        A("**And the answer must be quoted from the right kind of document:**")
        A("")
        _bullets(A, wit, signoffs, "")
    A("**What we accept as evidence.**")
    A("")
    _bullets(A, _pick(els, "accept"), signoffs,
             "The specification never says what counts as evidence. Anything the run chooses "
             "to quote will be accepted.")
    A("**What we refuse.**")
    A("")
    _bullets(A, _pick(els, "refuse"), signoffs,
             "The specification never says what does not count.")
    A("**The coding rules the run is told to follow.**")
    A("")
    _bullets(A, _pick(els, "rule"), signoffs, "None are stated.")

    groups = _pick(els, "source-group")
    if groups:
        A("**How much of the chart must be read before we are allowed to say \"it is not "
          "documented\".** Every document in the chart falls into exactly one of these groups.")
        A("")
        for g in groups:
            A(f"- {_sentence(g.text)}{_mark(g, signoffs)}")
            for suffix in (".terms", ".sample"):
                sub = by.get(g.element_id + suffix)
                if sub:
                    A(f"    - {_sentence(sub.text)}{_mark(sub, signoffs)}")
        A("")
    for eid in ("search-terms", "certainty"):
        if eid in by:
            A(f"- {_sentence(by[eid].text)}{_mark(by[eid], signoffs)}")
    if "search-terms" in by or "certainty" in by:
        A("")
    claims = _pick(els, "claim")
    if claims:
        A("**This answer has two halves and both must be proved.**")
        A("")
        _bullets(A, claims, signoffs, "")
    if "hints" in by:
        A(f"{_sentence(by['hints'].text)}{_mark(by['hints'], signoffs)}")
        A("")


def _s3(A, spec, els, signoffs) -> None:
    _h(A, 2)
    _bullets(A, _pick(els, "conflict"), signoffs,
             "The specification says nothing about contradictions. Whichever document the run "
             "happens to read first will decide the answer, and nothing records that there "
             "was a disagreement.")


def _s4(A, spec, els, signoffs) -> None:
    _h(A, 3)
    by = {e.element_id: e for e in els}
    A("There are two different refusals and they mean different things to whoever uses this "
      "data:")
    A("")
    _bullets(A, _pick(els, "refusal"), signoffs, "Neither refusal is described.")
    if "proof.negative" in by:
        A(f"**What the run is told at the point of refusing.** "
          f"{_sentence(by['proof.negative'].text)}{_mark(by['proof.negative'], signoffs)}")
        A("")
    caut = _pick(els, "caution")
    if caut:
        A("**Why the missing values cannot be ignored downstream.**")
        A("")
        _bullets(A, caut, signoffs, "")
    down = _pick(els, "downstream")
    if down:
        A("**Warnings attached to this variable for anyone analysing it.**")
        A("")
        _bullets(A, down, signoffs, "")
    if not caut and not down:
        A("_Nothing is recorded about what a refusal does to an analysis that uses this "
          "variable. A refused answer is not missing at random — it clusters on the patients "
          "whose care was fragmented — and no warning here means nobody downstream will be "
          "told._")
        A("")


def _s5(A, spec, ds, els, signoffs) -> None:
    _h(A, 4)
    if not ds:
        A("_Nothing in this specification was identified as a clinical choice with a "
          "defensible alternative. Treat that as a finding about this document, not as "
          "reassurance._")
        A("")
        return
    by = {e.element_id: e for e in els}
    A(f"{len(ds)} of them. They are ordered by how much they change, not by where they sit "
      f"in the file. If you have ten minutes, the first three are the ones that move the most "
      f"answers.")
    A("")
    for n, d in enumerate(ds, start=1):
        el = by.get(d.element_id)
        status, rec = signoff_status(el, signoffs) if el else (UNSIGNED, None)
        A(f"### {n}. {d.question}")
        A("")
        A(f"- **What we do now:** {d.choice}")
        A(f"- **Who decided:** {_who(el, spec, status, rec) if el else d.who}")
        if d.basis:
            A(f"- **On what basis:** {_sentence(d.basis)}")
        else:
            A("- **On what basis:** nothing is recorded.")
        A(f"- **If you decide otherwise:** {d.if_you_disagree}")
        if d.fires:
            A(f"- **How often it matters:** {d.fires}")
        A(f"- To confirm this one: `acr spec signoff --spec {spec.spec_id} --reviewer "
          f"\"your name\" --element {d.element_id}`")
        A("")


def _s6(A, spec, els, signoffs) -> None:
    _h(A, 5)
    mine = [e for e in els if e.provenance == MODEL_AUTHORED]
    silent = [e for e in mine if not e.recorded]
    total = len(els)
    A(f"**{len(mine)} of the {total} statements in this specification have no source outside "
      f"this repository.** A language model wrote them, and the fact that a standards manual "
      f"is named at the top of the file is not evidence that any particular sentence below "
      f"came out of it.")
    A("")
    if silent:
        A(f"Of those, **{len(silent)} have no provenance record at all** — marked _origin not "
          f"recorded_ below. Model-authored is the default the software applies when nobody "
          f"has written anything down, and a default is not a finding: nobody has checked "
          f"where these came from either way. The remaining {len(mine) - len(silent)} carry a "
          f"record that says outright that a model wrote them.")
        A("")
    _s6_findings(A, spec, els)
    if not mine:
        A("_Every element in this specification carries a recorded source. Check the sources, "
          "not this list._")
        A("")
        return
    A("This section is the reason the document exists. It is long because the state of the "
      "specification is that nothing has been attributed, not because the list has been "
      "padded. It shrinks as soon as somebody records where a statement came from — either "
      "by signing it off below, or by naming the section it was taken from in the "
      "`editorial_provenance:` block of the spec file. A label with no section, item or rule "
      "number does not shrink it, and that is deliberate.")
    A("")
    order = ["question", "answer", "authority", "guidance", "rule", "accept", "refuse",
             "proof", "source-group", "search-terms", "sampling", "certainty", "claim",
             "conflict", "refusal", "caution", "downstream", "boundary", "check",
             "convention", "not-for", "hints"]
    labels = {
        "question": "The question itself",
        "answer": "The fields that come out, and what values they may take",
        "authority": "The standard this claims to follow",
        "guidance": "Standing instructions to the software",
        "rule": "Coding rules",
        "accept": "What counts as evidence",
        "refuse": "What does not count as evidence",
        "proof": "What must be proved before an answer or a refusal",
        "source-group": "How the chart is divided up, and which documents may be cited",
        "search-terms": "The words we search for",
        "sampling": "How many documents we read at random",
        "certainty": "How certain we require ourselves to be",
        "claim": "The separate halves of the answer",
        "conflict": "Which document wins when two disagree",
        "refusal": "What each refusal means",
        "caution": "Warnings about missing values",
        "downstream": "Warnings to whoever analyses this",
        "boundary": "Settled rulings on specific situations",
        "check": "Automatic rejections applied to a submitted answer",
        "convention": "Conventions invented outright",
        "not-for": "Patients excluded",
        "hints": "Suggested search terms",
    }
    for kind in order + sorted({e.kind for e in mine} - set(order)):
        group = [e for e in mine if e.kind == kind]
        if not group:
            continue
        A(f"**{labels.get(kind, kind)}**")
        A("")
        for e in group:
            tag = "" if e.recorded else " _(origin not recorded)_"
            A(f"- `{e.element_id}` — {_sentence(e.text)}{tag}{_mark(e, signoffs)[2:]}")
        A("")


def _s6_findings(A, spec, els) -> None:
    """Editorial records that do not hold up, printed where the reader can act on them.

    Only the record-shape complaints appear here. The missing-record finding is not repeated
    as a list, because it IS the list below — every statement in section 6 tagged _(origin not
    recorded)_ is one, and printing them twice would bury the four that are different.
    """
    from acr.spec import editorial_records
    try:
        _, findings = editorial_records(spec)
    except Exception:
        return
    if not findings:
        return
    A(f"**{len(findings)} attribution(s) in this file do not hold up and have been "
      f"ignored.** An attribution that fails its own rules is worse than none: it reads as a "
      f"source to anybody skimming. The statements they named are still listed below as "
      f"unattributed.")
    A("")
    for f in findings:
        A(f"- `{f.element}` — {_plain(f.problem)}")
    A("")


def _s7(A, spec) -> None:
    _h(A, 6)
    A(CORPUS_HEADER)
    A("")
    terms = _all_terms(spec)
    printed = False

    for m in MEASUREMENTS:
        if m.only_spec and m.only_spec != spec.spec_id:
            continue
        if m.measured_list is not None:
            lists = _keyword_lists(spec)
            if not any(tuple(x) == m.measured_list for x in lists):
                A(f"> **A measurement was made on this criterion and no longer describes it.** "
                  f"The numbers below were measured against the required terms "
                  f"{_quoted(m.measured_list)}. This specification now requires "
                  f"{_quoted(sorted({t for x in lists for t in x})) or 'a different list'}, so "
                  f"the result has been withheld rather than reprinted beside a list it was "
                  f"not measured on. Re-run the measurement.")
                A("")
                printed = True
                continue
        if m.needs_terms and not all(t in terms for t in m.needs_terms):
            continue
        A(m.text)
        A("")
        printed = True

    if _sample_size(spec):
        A(SAMPLING_ARITHMETIC)
        A("")
        printed = True
    if not printed or all(m.only_spec != spec.spec_id for m in MEASUREMENTS):
        A(UNMEASURED_NOTE)
        A("")
        shared = sorted(set(terms) & {t for m in MEASUREMENTS if m.measured_list
                                      for t in m.measured_list})
        if shared:
            A(f"This criterion shares {_quoted(shared)} with the list that failed.")
            A("")


def _all_terms(spec) -> set[str]:
    terms = {str(t).lower() for t in (spec.search_hints or [])}
    for x in _keyword_lists(spec):
        terms |= {str(t).lower() for t in x}
    return terms


def _keyword_lists(spec) -> list[list[str]]:
    fn = _for_negative(spec)
    out = [list(g.keywords) for g in _source_groups(spec) if g.keywords]
    if fn.get("required_keywords"):
        out.append(list(fn["required_keywords"]))
    return out


def _appendix(A, spec, source_path) -> None:
    A("")
    A("## Appendix — where every part of the specification file went")
    A("")
    A("So that nothing in the file can be quietly left out of this document.")
    A("")
    routed = {
        "spec_id": "the header of this document",
        "spec_version": "the header of this document",
        "source_authority": "section 1",
        "data_source": "section 1",
        "question": "section 1 and the title",
        "fields": "section 1, \"what comes out\"",
        "agent_policy": "section 1",
        "when_not_to_use": "section 1 and section 5",
        "decision_rule": "section 2",
        "evidence_rules": "section 2",
        "proof_obligation": "sections 2, 4 and 5",
        "search_hints": "section 2",
        "conflict_rules": "sections 3 and 5",
        "abstention": "section 4",
        "special_codes_not_mar": "section 4",
        "downstream_warning": "section 4",
        "boundary_cases": "section 5",
        "answer_checks": "section 5",
        "date_imputation": "section 5",
        "provenance": "section 6 (it decides what is listed there)",
        "editorial_provenance":
            "section 6 (it decides what is listed there). This is the advisory half: the "
            "runtime never reads it, a missing entry is reported rather than refused, and an "
            "entry that names no section does not take its statement off the list",
        "keyword_field_coverage":
            "nowhere — it is an engineering cross-check that the search terms reach every "
            "field, asserted in the test suite, and carries no clinical decision",
    }
    keys: list[str] = []
    if source_path:
        import yaml
        raw = yaml.safe_load(Path(source_path).read_text(encoding="utf-8")) or {}
        keys = list(raw)
    for k in keys:
        A(f"- `{k}` — {routed.get(k, 'NOT TRANSLATED. This document does not know what this is; read it in the file.')}")
    A("")
