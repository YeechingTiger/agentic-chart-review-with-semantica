"""Turn the engineering vocabulary of a spec file into the words a clinician reads.

Nothing engineering-shaped may reach the page. No regex, no group name, no gate key, no
ICD-O code without its name. A reader who hits `cT(X|0|is|1mi|...)` stops reading, and the
sentence they stopped before is where the clinical content was — so every fragment the
review document emits passes through here, including prose harvested from the spec's own
comments. A term that leaks is a term missing from the table below, not a bug elsewhere.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

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


def plain(text: Any) -> str:
    """Strip the engineering vocabulary out of any fragment on its way to the page."""
    s = str(text or "").strip()
    if not s:
        return ""
    s = _JARGON_RE.sub(lambda m: JARGON[m.group(0).lower()], s)
    s = re.sub(r"\s+", " ", s)
    return s


def gloss_codes(text: str) -> str:
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


def human_field(name: str) -> str:
    parts = str(name).split("_")
    return " ".join(p.upper() if len(p) == 1 else p for p in parts)


def human_list(items: Iterable[Any], conj: str = "and") -> str:
    xs = [str(i).strip() for i in items if str(i).strip()]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + f" {conj} " + xs[-1]


def quoted(items: Iterable[Any]) -> str:
    return human_list([f"_{i}_" for i in items])


def sentence(s: str) -> str:
    s = plain(s)
    if s and s[-1] not in ".?!:":
        s += "."
    return s


_REGEX_META = set(r"\[](){}|+*?^$.")


def format_is_registry_notation(fmt: str) -> bool:
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


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def lower_first(s: str) -> str:
    s = plain(s).lstrip("*")
    return (s[0].lower() + s[1:]) if s else s
