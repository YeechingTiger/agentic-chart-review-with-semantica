"""Enumerate every statement in a spec a reviewer could argue with, with an identity and an
origin.

The made-up list is COMPLETE, and that is this module's job rather than the renderer's.
Provenance defaults to `model_authored` — not to the manual named in `source_authority`,
because naming the CoC manual at the top of a file is not evidence that the sentence three
hundred lines down came out of it. The opposite default lets a physician approve a fabricated
rule believing a standards body wrote it. That default makes section 6 long today. That
length is the finding.

Two provenance channels are read here and only here, because the correspondence between them
is a fact about what this module chose to call a statement: the enforced channel is keyed by
runtime paths and the editorial channel by the ids printed on the page, and one keyword list
is both.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The enforced channel's identity function for an answer_check, imported rather than
#: re-derived: two spellings of "which check is this" would drift, and the drift would show
#: up as an attributed rule quietly reverting to unattributed.
from acr.contract.spec import _answer_check_key

from .basis import basis_at, comment_map
from .prose import (
    ICDO_NAMES,
    format_is_registry_notation,
    human_field,
    human_list,
    plain,
    quoted,
    slug,
)

MODEL_AUTHORED = "model_authored"


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
    blob = f"{spec_id}\x00{kind}\x00{_canonical(raw)}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


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


def source_groups(spec) -> list[SourceGroup]:
    fn = for_negative(spec)
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


def for_negative(spec) -> dict:
    po = getattr(spec, "proof_obligation", None)
    return dict(getattr(po, "for_negative", {}) or {}) if po else {}


def extra_keys(spec) -> dict:
    return dict(getattr(spec, "model_extra", None) or {})


def elements(spec, source_path: str | Path | None = None) -> list[Element]:
    """Every statement in the spec that a reviewer could agree or disagree with."""
    lines: list[str] = []
    attached: dict[int, str] = {}
    if source_path:
        lines, attached = comment_map(Path(source_path).read_text(encoding="utf-8"))

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
        basis = basis_at(lines, attached, basis_needle if basis_needle is not None else raw)
        out.append(Element(element_id=eid, kind=kind, text=plain(text), raw=raw,
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
            f"A citation for {human_field(fname)} is only accepted from: "
            f"{human_list(groups)}.", {fname: groups}, 2)

    for g in source_groups(spec):
        base = f"source-group.{g.index}"
        # The enforced channel splits one document group across up to four records (how it is
        # matched, what it may establish, what must be searched, how much is sampled). The
        # reviewer sees one sentence and two sub-bullets, so the group sentence answers to
        # whichever of match/establishes carries a record.
        ebase = ("proof_obligation.for_negative"
                 + (f".claims[{g.claim}]" if g.claim else "") + f".strata[{g.raw.get('name')}]")
        add(base, "source-group", group_sentence(g), g.raw, 2,
            basis_needle=(g.raw.get("name") or g.label),
            enforced=[f"{ebase}.match", f"{ebase}.partition_by", f"{ebase}.establishes"])
        if g.keywords:
            add(f"{base}.terms", "search-terms",
                f"Before we may say the answer is not in this group we must have searched it "
                f"for {quoted(g.keywords)}.", list(g.keywords), 2,
                basis_needle=g.keywords[0], enforced=f"{ebase}.required_keywords")
        if g.numbers:
            add(f"{base}.sample", "sampling", _sample_sentence(g), dict(g.numbers), 2,
                enforced=[f"{ebase}.min_sample", f"{ebase}.min_sample_of_misses"])

    fn = for_negative(spec)
    if fn.get("required_keywords"):
        add("search-terms", "search-terms",
            f"Before any \"not documented\" answer is accepted, the chart must have been "
            f"searched for {quoted(fn['required_keywords'])}.",
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

    for key, val in extra_keys(spec).items():
        if key in ("provenance", "keyword_field_coverage"):
            continue
        if isinstance(val, dict):
            for sub, sv in val.items():
                add(f"{slug(key)}.{slug(sub)}", "convention",
                    f"Where the record says \"{sub}\", we record {sv}.", {sub: sv}, 5)
        elif isinstance(val, list):
            for i, item in enumerate(val, start=1):
                add(f"{slug(key)}.{i}", "convention", str(item), item, 5)
    if spec.search_hints:
        add("hints", "hints",
            f"Suggested (not required) search terms: {quoted(spec.search_hints)}.",
            list(spec.search_hints), 2)
    return out


def _authority_line(spec) -> str:
    sa = spec.source_authority or {}
    bits = []
    if sa.get("document"):
        bits.append(f"Written against {sa['document']}")
    if sa.get("items"):
        bits.append(f"for the registry items {human_list(sa['items'])}")
    if sa.get("applicability"):
        bits.append(f"and applies to {sa['applicability']}")
    return ("; ".join(bits) + ".") if bits else "No source document is named."


def _field_line(f) -> str:
    bits = [f"**{human_field(f.name)}** — {f.description or 'no description given'}"]
    if f.allowable_values:
        vals = list(f.allowable_values)
        named = [f"{v} ({ICDO_NAMES[str(v)]})" if str(v) in ICDO_NAMES else str(v) for v in vals]
        if len(vals) > 12:
            bits.append(f"The spec permits {len(vals)} values and names none of them in the "
                        f"file: {human_list(named)}. A reviewer cannot confirm a value list "
                        f"they cannot read.")
        else:
            bits.append(f"Permitted values: {human_list(named)}.")
    if f.format and format_is_registry_notation(f.format):
        bits.append(f"**This field is broken today.** The file declares its shape as "
                    f"`{f.format}`, which is registry notation, but the software applies it "
                    f"as a literal pattern — so it will reject every valid value of this "
                    f"field, including every real date. Unfixed as of 2026-07-26.")
    elif f.format:
        bits.append("Its shape is checked automatically against a fixed pattern.")
    if f.nullable is False:
        bits.append("The registry does not allow this field to be left blank.")
    return " ".join(bits)


def group_sentence(g: SourceGroup) -> str:
    bits = [f"**{g.label}.** "]
    bits.append(_POLICY_PROSE.get(g.policy, f"handling: {g.policy}") + ".")
    if g.establishes:
        bits.append(f" A document from this group may be cited as the evidence for "
                    f"{human_list([human_field(x) for x in g.establishes])}.")
    elif g.raw.get("establishes") == []:
        bits.append(" Nothing in this group may be cited as evidence for anything.")
    if g.claim:
        bits.append(f" (Used only for the part of the answer about "
                    f"{human_field(g.claim).replace('-', ' ')}.)")
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
    cid = human_field(str(claim.get("id", ""))).replace("-", " ")
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
        bits.append("and record " + human_list([f"{human_field(k)} = {v}" for k, v in val.items()]))
    if of.get("must_report"):
        bits.append("and we must report why, and which stretches of time were left uncovered")
    return " ".join(bits) + "."


def _boundary_line(b: Any) -> str:
    if not isinstance(b, dict):
        return str(b)
    case = b.get("case", "")
    verdict = {k: v for k, v in b.items() if k not in ("case", "why")}
    ans = human_list([f"{human_field(k)} = {v}" for k, v in verdict.items()])
    out = f"\"{case}\" → {ans or 'no answer recorded'}"
    if b.get("why"):
        out += f". Because: {b['why']}"
    return out


def _check_line(c: Any) -> str:
    if not isinstance(c, dict):
        return str(c)
    fld = human_field(c.get("field", ""))
    kind = str(c.get("kind", ""))
    if kind == "not_less_specific":
        return (f"{fld}: the values {human_list(c.get('nos_values') or [])} are refused "
                f"whenever the evidence the run itself quoted contains any of "
                f"{quoted(c.get('contradicted_by') or [])}. {c.get('message', '')}")
    if kind == "nos_requires_search":
        return (f"{fld}: the values {human_list(c.get('nos_values') or [])} are refused "
                f"unless the chart was first searched for "
                f"{quoted(c.get('required_searches') or [])}. {c.get('message', '')}")
    if kind == "origin_not_specimen":
        return (f"{fld}: a value is refused when the quote supporting it is the description "
                f"of a specimen rather than of the tumour's origin. {c.get('message', '')}")
    return f"{fld}: {kind}. {c.get('message', '')}"


def _provenance_map(spec) -> dict[str, tuple[str, str]]:
    """Both provenance channels, flattened to element key -> (origin, basis).

    Keys from two namespaces land in one dict on purpose, and they cannot collide: the
    enforced channel is keyed by runtime paths
    (`proof_obligation.for_negative.strata[...].required_keywords`) and the editorial channel
    by the ids this module prints (`rule.1`, `conflict.2`, `boundary.3`). `elements()` looks
    up its own id first and the enforced path second.

    Both are first-class fields on `ExtractionSpec`, not extras, so neither shows up in
    `model_extra`. This used to read `extra_keys(spec).get("provenance")`, which found nothing for
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


def _editorial(spec) -> tuple[dict, list]:
    """`acr.contract.spec.editorial_records`, tolerant of a spec object that has no editorial channel.

    Imported inside the call rather than at module scope: `acr.contract.spec`'s own docstrings point
    at this package, and a top-level import in both directions is an import cycle waiting for
    somebody to add one more name.
    """
    from acr.contract.spec import editorial_records
    try:
        return editorial_records(spec)
    except Exception:  # a spec object from somewhere else that has no editorial channel
        return {}, []


def editorial_index(spec) -> dict:
    """The valid editorial records, by element id. Malformed ones are dropped, not raised."""
    return _editorial(spec)[0]


def editorial_problems(spec) -> list:
    """The editorial records that did not hold up, for printing beside the statements they name.

    The renderer asks this module rather than `acr.contract.spec` directly, for the same reason
    `provenance_findings` exists: what counts as a statement is decided here, and two routes
    to "what is attributed" is how the document and the findings come to disagree.
    """
    return _editorial(spec)[1]


def element_ids(spec, source_path: str | Path | None = None) -> list[str]:
    """Every id the review document prints — the editorial channel's namespace.

    `acr.contract.spec.editorial_findings` needs this to tell an unattributed statement from a record
    naming a statement that no longer exists. It lives here because this module decides what
    a reviewable statement is; the loader deliberately does not know.
    """
    return [e.element_id for e in elements(spec, source_path=source_path)]


def provenance_findings(spec, source_path: str | Path | None = None) -> list:
    """`editorial_findings`, told which statements the ENFORCED block already covers.

    The single call every caller should use. Going to `acr.contract.spec.editorial_findings` directly
    with only the id list reports a keyword list as unattributed even when the enforced block
    attributes it, because only this module knows that `source-group.2.terms` and
    `...strata[may_mention].required_keywords` are the same rule seen from two sides.
    """
    from acr.contract.spec import editorial_findings
    els = elements(spec, source_path=source_path)
    return editorial_findings(spec, [e.element_id for e in els],
                              attributed_elsewhere=[e.element_id for e in els if e.recorded])


def n_fields(spec) -> int:
    return max(1, len(spec.fields))


def sample_size(spec) -> int | None:
    for g in source_groups(spec):
        n = g.numbers.get("min_sample_of_misses") or g.numbers.get("min_sample")
        if n:
            return int(n)
    return None


def all_terms(spec) -> set[str]:
    terms = {str(t).lower() for t in (spec.search_hints or [])}
    for x in keyword_lists(spec):
        terms |= {str(t).lower() for t in x}
    return terms


def keyword_lists(spec) -> list[list[str]]:
    fn = for_negative(spec)
    out = [list(g.keywords) for g in source_groups(spec) if g.keywords]
    if fn.get("required_keywords"):
        out.append(list(fn["required_keywords"]))
    return out
