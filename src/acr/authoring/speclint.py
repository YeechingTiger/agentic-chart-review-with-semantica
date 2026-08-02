"""Spec completeness, in four tiers that cost four different things.

A spec can be wrong in ways that need nothing but the file, in ways that need the corpus, in
ways that need an answer key, and in ways that need a registrar. Collapsing those is how "the
spec looks fine" gets said about a spec that cannot answer its own question — so the tiers are
separate here, they are labelled with what they cost, and only the first one runs by default.

  TIER 1  FORMAL         the file alone. No corpus, no key, no clinical knowledge. Runs.
  TIER 2  CORPUS         needs the charts. Implemented, requires --corpus, never runs without it.
  TIER 3  ANSWER KEY     needs ground truth. REFUSED unless both --tier3 and --answer-key.
  TIER 4  HUMAN          cannot be checked by anything here. Printed as a list, honestly.

Two tier-1 checks earn the module on their own.

F1 is not `re.compile(fmt)`. `re.compile("CCYYMMDD")` SUCCEEDS: it is a perfectly valid
pattern that matches exactly one string, the literal "CCYYMMDD". STORE.390 and
STORE.1860_1880 both declare it, `answer_checks.check_field_formats` applies it with
`re.fullmatch`, and so every date those two specs can legally produce is rejected as
malformed. A linter that only asks whether the pattern parses reports both files clean.

F8 is arithmetic nobody had run. With n clean draws the Clopper-Pearson 95% upper bound is
1 - 0.05**(1/n); at n=25 that is 0.1129, so `max_elusion_upper: 0.12` is met by 0.0071 and
`0.10` cannot be met at all — the gate rejects forever and no rejection message says the
number is the reason. This repository has already shipped that gate once (commit 173f453).
So the check computes the minimum n for the declared bound, AND the bound the declared n
actually earns, and prints both: the numbers in the specs are model-invented, and a table is
what a human needs in order to replace an invention with a decision.

Nothing in here reads a chart, calls a model, or opens an answer key.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAIL = "FAIL"
NOTE = "NOTE"

F1 = "F1 FORMAT_IS_A_PATTERN"
F2 = "F2 STRATUM_TOTALITY"
F3 = "F3 ESTABLISHES_NAMES_REAL_FIELDS"
F4 = "F4 KEYWORD_FIELD_COVERAGE"
F5 = "F5 EVIDENCE_CLOSURE"
F6 = "F6 ABSTENTION_TOTALITY"
F7 = "F7 CONFLICT_MATRIX"
F8 = "F8 GATE_SATISFIABILITY"
F9 = "F9 ANSWER_CHECK_INTEGRITY"
F10 = "F10 LOCAL_TYPE_NAMES_IN_CONTRACT"
TIER1_CHECKS = (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)

ABSTENTIONS = ("EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT")


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    spec_id: str
    where: str
    message: str


class AnswerKeyRefused(RuntimeError):
    """A tier-3 check was reached without both the flag and the key.

    Raised rather than skipped. An answer key that a plain lint can wander into is an answer
    key the lint will eventually be scored against, and a checker scored against its own
    ground truth stops being a checker.
    """


# ------------------------------------------------------------------------------ statistics
def bound_at_n(n: int, confidence: float) -> float:
    """The elusion upper bound n clean draws earn. Same closed form as `coverage`.

    `confidence` has no default anywhere in this module. A confidence level chosen by the
    linter is a number nobody signed for, silently applied to a gate somebody has to defend.
    """
    if n <= 0:
        return 1.0
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def min_n_for_bound(bound: float, confidence: float) -> int:
    """Smallest zero-hit sample that can reach `bound`. Below it the gate cannot be passed.

    Solving 1 - alpha**(1/n) <= bound gives n >= ln(alpha)/ln(1-bound); the ceiling is taken
    and then verified, because a float that lands a hair under the true root would hand back
    an n that is one short — which is precisely the failure this function exists to catch.
    """
    if not 0.0 < bound < 1.0:
        raise ValueError(f"bound must be in (0,1), got {bound!r}")
    alpha = 1.0 - confidence
    n = max(1, math.ceil(math.log(alpha) / math.log(1.0 - bound)))
    while bound_at_n(n, confidence) > bound:
        n += 1
    while n > 1 and bound_at_n(n - 1, confidence) <= bound:
        n -= 1
    return n


# -------------------------------------------------------------------------------- helpers
def _for_negative(spec) -> dict:
    po = getattr(spec, "proof_obligation", None)
    return dict(getattr(po, "for_negative", {}) or {}) if po else {}


def _scopes(spec) -> list[tuple[str, dict]]:
    """(label, holder) for the top-level obligation and each conjunctive claim.

    Claims are separate scopes because they are separate proofs: STORE.1860_1880's witness
    claim and its coverage claim have different strata and fail independently, and folding
    them together would report a hole in one as covered by the other.
    """
    fn = _for_negative(spec)
    out = [("for_negative", fn)]
    for c in (fn.get("claims") or []):
        out.append((f"claims[{c.get('id')}]", dict(c)))
    return out


def _strata(spec) -> list[tuple[str, dict]]:
    return [(label, s) for label, holder in _scopes(spec) for s in (holder.get("strata") or [])]


def _gate(spec) -> dict:
    return dict(_for_negative(spec).get("gate") or {})


def _declared_unanswerable(spec) -> bool:
    """The spec itself says no coverage over notes could ever settle this variable.

    STORE.610 is the case: Class of Case lives in registration and billing systems, so the
    checks below would otherwise report a design decision as a defect. It is still printed —
    downgraded to a NOTE, never dropped.
    """
    return _for_negative(spec).get("mode") == "not_applicable"


def _format_state(fmt: str) -> tuple[str, str]:
    """('ok'|'uncompilable'|'literal', why). The middle state is the one that hides.

    A pattern containing no regex metacharacter is not a pattern: `re.fullmatch` accepts
    exactly one string, the pattern itself. That is what registry notation looks like once
    the software treats it as a regex.
    """
    try:
        re.compile(fmt)
    except re.error as e:
        return "uncompilable", str(e)
    if not (set(fmt) & set(r"\[](){}|+*?.^$")):
        return "literal", f"no regex metacharacter; matches only the literal {fmt!r}"
    return "ok", ""


def _accepts(f, value: Any) -> bool:
    """Does this field's declared value space contain `value`?

    A field whose `format` is broken is treated as accepting anything. The fault belongs to
    F1, and letting it also fail F6 and F9 would report one authoring mistake three times and
    bury the two checks that are about something else.
    """
    s = str(value).strip()
    fmt = getattr(f, "format", None)
    if fmt:
        return True if _format_state(fmt)[0] != "ok" else bool(re.fullmatch(fmt, s))
    allowed = getattr(f, "allowable_values", None)
    if allowed:
        return s in [str(v) for v in allowed]
    return True                      # unconstrained: F1 has already said so


def _declared_examples(spec, field_name: str) -> list[str]:
    """Values the spec itself writes down for a field — boundary cases and NOS codes.

    These are the spec's own witnesses against its own format, and quoting them is what turns
    "this looks like registry notation" into "this rejects the answer the file demands".
    """
    out: list[str] = []
    for bc in (spec.boundary_cases or []):
        if isinstance(bc, dict) and isinstance(bc.get(field_name), (str, int)):
            out.append(str(bc[field_name]))
    for chk in (spec.answer_checks or []):
        if isinstance(chk, dict) and chk.get("field") == field_name:
            out += [str(v) for v in (chk.get("nos_values") or [])]
    return out


# ------------------------------------------------------------------- TIER 1, one check each
def _f1_formats(spec) -> Iterable[Finding]:
    for f in spec.fields:
        if not f.format and not f.allowable_values:
            yield Finding(F1, FAIL, spec.spec_id, f"fields[{f.name}]",
                          f"declares neither format nor allowable_values (type: {f.type}), so "
                          f"check_field_formats enforces nothing: any string the model emits is legal.")
            continue
        if not f.format:
            continue
        state, why = _format_state(f.format)
        if state == "ok":
            continue
        rejected = [v for v in _declared_examples(spec, f.name)
                    if state == "literal" and not re.fullmatch(f.format, v)]
        kind = ("does not compile as a Python regex" if state == "uncompilable"
                else "is registry notation, not a regex")
        yield Finding(F1, FAIL, spec.spec_id, f"fields[{f.name}].format",
                      f"format {f.format!r} {kind} ({why}). check_field_formats applies it with "
                      f"re.fullmatch, so this field rejects every value it can legally hold."
                      + (f" The spec's own declared values are rejected by it: {rejected}."
                         if rejected else ""))


def _f2_totality(spec) -> Iterable[Finding]:
    for label, holder in _scopes(spec):
        strata = holder.get("strata") or []
        mode = holder.get("mode") or _for_negative(spec).get("mode")
        if not strata:
            if holder.get("claims"):
                continue      # a conjunctive obligation holds its strata inside the claims
            yield Finding(F2, NOTE, spec.spec_id, label,
                          f"mode {mode!r} declares no strata at all: no document is classified, so no "
                          f"absence claim over this scope carries an elusion bound.")
        elif not any((s.get("match") or {}).get("rest") for s in strata):
            yield Finding(F2, FAIL, spec.spec_id, label,
                          f"strata {[s.get('name') for s in strata]} leave documents unclassified: no "
                          f"stratum takes `rest: true`, and coverage.assign_strata DROPS a document "
                          f"matching none of them. It is never read, never sampled and adds no elusion, "
                          f"so whatever it says is invisible to the gate.")


def _f3_establishes(spec) -> Iterable[Finding]:
    names = {f.name for f in spec.fields}
    for label, s in _strata(spec):
        bad = [x for x in (s.get("establishes") or []) if x not in names]
        if bad:
            yield Finding(F3, FAIL, spec.spec_id, f"{label}.strata[{s.get('name')}].establishes",
                          f"names {bad}, which this spec does not declare as fields, so "
                          f"coverage.witness_policy never matches them and the stratum admits a "
                          f"citation for nothing. Declared: {sorted(names)}.")


def _f4_keyword_coverage(spec) -> Iterable[Finding]:
    """Generalised from STORE.700_880, the only spec that declares the mapping.

    The gate-enforced list is `for_negative.required_keywords` and nothing else: graph.check_gate
    loops over exactly that key. A stratum-level list feeds `_keyword_hits_among_drawn` and cannot
    discharge a field, so a spec whose only terms live there has no reachability claim at all.
    """
    names = [f.name for f in spec.fields]
    required = list(getattr(spec.proof_obligation, "required_keywords", []) or [])
    sev = NOTE if _declared_unanswerable(spec) else FAIL
    if _gate(spec).get("required_keywords_all_searched") and not required:
        yield Finding(F4, FAIL, spec.spec_id, "for_negative.required_keywords",
                      "the gate asserts required_keywords_all_searched over an EMPTY list. "
                      "graph.check_gate iterates it, finds nothing and passes: the switch reads as "
                      "enforcement and enforces nothing.")
    cov = (spec.model_extra or {}).get("keyword_field_coverage")
    if not isinstance(cov, dict):
        if not required:
            # A contract that declares NO retrieval has no reachability claim to verify, and
            # demanding the map anyway is demanding retrieval back. That is the direction this
            # tree moved on 2026-08-02: which terms reach which field is a measurement over a
            # development set, made by the experience layer, not a sentence the contract asserts.
            # Reported so the absence is visible, because silence here would read as a spec that
            # had been checked.
            yield Finding(F4, NOTE, spec.spec_id, "keyword_field_coverage",
                          f"undeclared, and no required searches are declared either, so this "
                          f"contract makes no reachability claim about its {len(names)} field(s) "
                          f"{names}. Whatever reaches them has to come from a certified "
                          f"experience asset and be measured there.")
            return
        yield Finding(F4, sev, spec.spec_id, "keyword_field_coverage",
                      f"undeclared, so nothing states which required search reaches which field. "
                      f"Reachability is unverifiable for all {len(names)} fields {names}; required "
                      f"terms declared: {required}.")
        return
    used: set[str] = set()
    for n in names:
        terms = list(cov.get(n) or [])
        unrequired = [t for t in terms if t not in required]
        used |= {t for t in terms if t in required}
        if not terms:
            yield Finding(F4, sev, spec.spec_id, f"keyword_field_coverage[{n}]",
                          f"field {n} is reachable by no required search term.")
        elif unrequired:
            yield Finding(F4, sev, spec.spec_id, f"keyword_field_coverage[{n}]",
                          f"claims coverage of {n} by {unrequired}, which the gate never requires "
                          f"anyone to search, so the field is covered on paper only.")
    for t in required:
        if t not in used:
            yield Finding(F4, sev, spec.spec_id, "for_negative.required_keywords",
                          f"required search {t!r} reaches no field: the gate demands it and the spec "
                          f"cannot say what it is for.")


def _f5_evidence_closure(spec) -> Iterable[Finding]:
    """A field no evidence rule can establish is unanswerable by construction.

    `establishes` is the machine-readable half of `counts_as_evidence` — the half
    `admissibility_for_citations` actually rules on — so it is what closure is tested against.
    Where a spec declares none, the coupling is prose, unverifiable here, and reported as such
    rather than assumed good: an UNDECLARED admissibility is not an admitted one.
    """
    clauses = list((spec.evidence_rules or {}).get("counts_as_evidence") or [])
    strata = _strata(spec)
    uses_establishes = any("establishes" in s for _, s in strata)
    sev = NOTE if _declared_unanswerable(spec) else FAIL
    for f in spec.fields:
        srcs = [s.get("name") for _, s in strata if f.name in (s.get("establishes") or [])]
        if not clauses:
            yield Finding(F5, sev, spec.spec_id, f"fields[{f.name}]",
                          f"no counts_as_evidence clause exists anywhere in this spec, so nothing "
                          f"states what could establish {f.name}: unanswerable by construction.")
        elif uses_establishes and not srcs:
            yield Finding(F5, FAIL, spec.spec_id, f"fields[{f.name}]",
                          f"no stratum's `establishes` names {f.name} while other strata here do use "
                          f"`establishes`, so every citation for it resolves to REFUSED in "
                          f"coverage.admissibility_for_citations.")
        elif not uses_establishes:
            # The gate and the admissibility rule do not read the same key. `coverage.py` passes
            # `require_can_establish_nonempty` on the mere PRESENCE OF THE NAME —
            # `ok = "can_establish" in by` — while `admissibility_for_citations` rules on
            # `establishes`. A spec can therefore satisfy a switch that says "some document type
            # here can settle this" while nothing states what any type settles, and every citation
            # the run makes resolves to UNDECLARED. Prose coupling is a NOTE; prose coupling
            # underneath a gate that asserts the coupling exists is the same fault F4 already
            # refuses over an empty required_keywords list.
            g = _gate(spec)
            asserted = (g.get("require_can_establish_nonempty")
                        or g.get("per_claim_can_establish_nonempty"))
            yield Finding(F5, sev if asserted else NOTE, spec.spec_id, f"fields[{f.name}]",
                          f"{len(clauses)} counts_as_evidence clause(s) exist but no stratum declares "
                          f"`establishes`, so admissibility is UNDECLARED for every document and the "
                          f"coupling to {f.name} is prose only."
                          + (" The gate asserts can_establish is non-empty, which coverage.py checks "
                             "by stratum NAME alone: it passes on a stratum that establishes nothing."
                             if asserted else ""))


def _f6_abstention_totality(spec) -> Iterable[Finding]:
    """value-space u EVIDENCE_INSUFFICIENT u SPEC_INSUFFICIENT must exhaust the outcomes.

    Tested against the spec's own boundary cases, because those are outcomes the author has
    already committed to: an answer the file demands and the file cannot represent is the cheapest
    possible proof that the outcome space is not closed.
    """
    by_name = {f.name: f for f in spec.fields}
    for bc in (spec.boundary_cases or []):
        if not isinstance(bc, dict) or not by_name:
            continue
        where = f"boundary_cases[{str(bc.get('case', '?'))[:40]}]"
        for key, val in bc.items():
            text = str(val)
            if key in ("case", "why") or val is None or any(a in text for a in ABSTENTIONS):
                continue
            if key in by_name:
                if not _accepts(by_name[key], val):
                    yield Finding(F6, FAIL, spec.spec_id, where,
                                  f"declares {key}={text!r}, outside that field's declared value "
                                  f"space and not an abstention.")
                continue
            # A free-text answer counts as representable if ANY token in it is a legal value:
            # "clinical_t cT2a, clinical_n cN0" is prose about real codes, and demanding that the
            # whole string parse would flag every multi-field case in the tree.
            toks = [t.strip(" .,;:'\"()") for t in re.split(r"[\s,;]+", text) if t.strip()]
            if not any(_accepts(f, t) for t in toks for f in by_name.values()):
                yield Finding(F6, FAIL, spec.spec_id, where,
                              f"the answer {text!r} is in no field's value space and names no "
                              f"abstention: the spec demands an outcome it cannot record.")


def _identity_tokens(s: dict) -> list[str]:
    return [str(t).lower() for t in ((s.get("match") or {}).get("doc_type_matches") or [])]


def _f7_conflicts(spec) -> Iterable[Finding]:
    """Two evidence source classes for one field, and nothing that orders them.

    The ordering has to be findable in the conflict rules by NAMING both sources, because a rule
    that names neither cannot be applied by anyone -- model or human -- to the pair it is supposed
    to arbitrate. A `rest: true` stratum has no identity to name at all, which is itself the
    finding: no conflict rule about it can ever be written.
    """
    rules = " ".join(f"{r.get('if', '')} {r.get('then', '')}" if isinstance(r, dict) else str(r)
                     for r in (spec.conflict_rules or [])).lower()
    strata = _strata(spec)
    pairs: dict[tuple[str, str], list[str]] = {}
    nameless: set[str] = set()
    for f in spec.fields:
        srcs = [s for _, s in strata if f.name in (s.get("establishes") or [])]
        for i, a in enumerate(srcs):
            for b in srcs[i + 1:]:
                ta, tb = _identity_tokens(a), _identity_tokens(b)
                if any(t in rules for t in ta) and any(t in rules for t in tb):
                    continue
                key = (str(a.get("name")), str(b.get("name")))
                pairs.setdefault(key, []).append(f.name)
                nameless |= {k for k, t in zip(key, (ta, tb)) if not t}
    for (a, b), fields in pairs.items():
        blind = [n for n in (a, b) if n in nameless]
        yield Finding(F7, FAIL, spec.spec_id, f"strata[{a}] x strata[{b}]",
                      f"UNDECLARED CONFLICT between {a!r} and {b!r} on {fields}: both declare they "
                      f"can establish these fields and no conflict_rule names both sources, so which "
                      f"one wins when they disagree is undecided."
                      + (f" {blind} match `rest: true` and have no document type to name, so no "
                         f"conflict rule about them can be written at all." if blind else ""))


def gate_rows(spec) -> list[dict]:
    """One row per declared elusion cap: what it demands, and what the samples can deliver.

    `binding_n` skips can_establish, matching coverage.evaluate_gate, which takes the worst
    bound over the strata that are NOT read exhaustively. The weakest sampled stratum sets the
    number for the whole gate, so a single small sample decides satisfiability however large
    the others are.
    """
    rows, fn = [], _for_negative(spec)
    for label, holder in _scopes(spec):
        cap = (holder.get("gate") or {}).get("max_elusion_upper")
        if cap is None:
            continue
        conf = holder.get("confidence", fn.get("confidence"))
        scoped = _strata(spec) if label == "for_negative" else \
            [(label, s) for s in (holder.get("strata") or [])]
        samples = {str(s.get("name")): s.get("min_sample", s.get("min_sample_of_misses"))
                   for _, s in scoped if s.get("name") != "can_establish"
                   and isinstance(s.get("min_sample", s.get("min_sample_of_misses")), int)}
        binding = min(samples.values()) if samples else None
        row = {"spec_id": spec.spec_id, "scope": label, "bound": float(cap), "confidence": conf,
               "samples": samples, "binding_n": binding, "min_n_required": None,
               "bound_earned": None, "margin": None,
               "verdict": "no confidence declared" if conf is None else "vacuous (nothing sampled)"}
        if conf is not None:
            row["min_n_required"] = min_n_for_bound(float(cap), float(conf))
            if binding is not None:
                row["bound_earned"] = bound_at_n(binding, float(conf))
                row["margin"] = float(cap) - row["bound_earned"]
                row["verdict"] = "SATISFIABLE" if row["margin"] >= 0 else "UNSATISFIABLE"
        rows.append(row)
    return rows


def _f8_gate(spec) -> Iterable[Finding]:
    for r in gate_rows(spec):
        where, n, cap = f"{r['scope']}.gate.max_elusion_upper", r["binding_n"], r["bound"]
        if r["confidence"] is None:
            yield Finding(F8, FAIL, spec.spec_id, where,
                          f"declares max_elusion_upper {cap} with no `confidence` beside it. A bound "
                          f"is meaningless without the level it holds at, and a level chosen by the "
                          f"checker is a number nobody agreed to.")
        elif n is None:
            yield Finding(F8, NOTE, spec.spec_id, where,
                          f"cap {cap} is priced over no sampled stratum: evaluate_gate takes max() of "
                          f"an empty set, defaults to 0.0, and the cap passes with nothing drawn.")
        elif r["verdict"] == "UNSATISFIABLE":
            yield Finding(F8, FAIL, spec.spec_id, where,
                          f"UNSATISFIABLE: {n} clean draws earn {r['bound_earned']:.4f} at confidence "
                          f"{r['confidence']}, above the cap {cap}. No amount of work passes this "
                          f"gate; {r['min_n_required']} draws are the minimum that reach the cap "
                          f"(sampled strata: {r['samples']}).")
        else:
            yield Finding(F8, NOTE, spec.spec_id, where,
                          f"satisfiable by {r['margin']:.4f}: {n} draws earn {r['bound_earned']:.4f} "
                          f"against a cap of {cap}; {r['min_n_required']} is the minimum n that "
                          f"reaches it.")


def _f9_answer_checks(spec) -> Iterable[Finding]:
    by_name = {f.name: f for f in spec.fields}
    hints = [h.lower() for h in (spec.search_hints or [])]
    for chk in (spec.answer_checks or []):
        if not isinstance(chk, dict):
            continue
        name = chk.get("field")
        where = f"answer_checks[{name}.{chk.get('kind')}]"
        if name not in by_name:
            yield Finding(F9, FAIL, spec.spec_id, where,
                          f"checks field {name!r}, which this spec does not declare, so check_answer "
                          f"looks it up in an answer that can never carry it. Declared: {sorted(by_name)}.")
            continue
        for v in (chk.get("nos_values") or []):
            if not _accepts(by_name[name], v):
                yield Finding(F9, FAIL, spec.spec_id, where,
                              f"nos_value {str(v)!r} is outside the declared value space of {name}, so "
                              f"the rule guards a code the field cannot hold.")
        # A required_searches term nobody was told to run is a rejection the agent cannot act on.
        for term in (chk.get("required_searches") or []):
            if not any(term.lower() in h or h in term.lower() for h in hints):
                yield Finding(F9, NOTE, spec.spec_id, where,
                              f"demands a search for {term!r} that no search_hint mentions: the answer "
                              f"is rejected for not doing something the prompt never asked for.")


def _f10_local_type_names(spec) -> Iterable[Finding]:
    """A stratum still selecting documents by substring over raw local type names.

    Two findings in one, and the placement one is why this is Tier 1 rather than Tier 2.

    PLACEMENT. `docs/CHART_REVIEW_KNOWLEDGE_AND_SEARCH_LAYERS.md` gives the Task Contract
    everything about what the answer MEANS and forbids it "keywords, raw local note types,
    sampling thresholds". `doc_type_matches: ["Pathology", "Cytology"]` is a raw local note
    type, in the Task Contract, and no corpus is needed to see that.

    CORRECTNESS. The expression is a case-insensitive SUBSTRING, and measured on this corpus
    it was wrong in both directions: it matched Speech-Language-Pathology-Note and missed
    Non-Gyn-Cyto-FNA (1,285 documents), FN-Aspirate-Report (881) and SURG-PATH-RESULT (231).
    107 of the 219 patients with no matching type name held one of those reports anyway --
    stratified as unable to establish histology while a cytology diagnosis sat in the chart.
    `T2 dead_doc_types` cannot see any of this: a token that selects SOMETHING is not dead,
    and every token above selects something. It just also misses the documents that matter.

    The repair is `means:` prose plus a Site Mapping (`acr site-mapping build`), not a longer
    substring list. STORE.700_880 already tried the longer list -- 24 tokens, with a comment
    explaining that `Pathology` was the wrong instrument -- which is the same conclusion
    reached twice and answered twice with the same tool.
    """
    for label, s in _strata(spec):
        toks = _identity_tokens(s)
        if not toks:
            continue
        yield Finding(F10, FAIL, spec.spec_id,
                      f"{label}.strata[{s.get('name')}].match.doc_type_matches",
                      f"{len(toks)} raw local type-name substring(s) {toks[:4]}"
                      f"{'...' if len(toks) > 4 else ''} decide this stratum. Raw local note "
                      f"types do not belong in a Task Contract, and as a case-insensitive "
                      f"substring this expression was measured wrong in both directions on "
                      f"this corpus. Replace it with `means:` prose and a Site Mapping.")


def lint_spec(spec) -> list[Finding]:
    """Tier 1 only. Deterministic, offline, and ordered by check so a diff of two runs reads."""
    out: list[Finding] = []
    for fn in (_f1_formats, _f2_totality, _f3_establishes, _f4_keyword_coverage,
               _f5_evidence_closure, _f6_abstention_totality, _f7_conflicts, _f8_gate,
               _f9_answer_checks, _f10_local_type_names):
        out.extend(fn(spec))
    return out


# --------------------------------------------------------------- TIER 2, corpus, never free
TIER2_CHECKS = {
    "dead_doc_types": "stratum doc_type_matches tokens that select no document in the corpus",
    "dead_terms": "required search terms that match no document text anywhere",
    "ungateable_patients": "fraction of patients for whom this spec can never pass its own gate",
}


def tier2_checks(spec, corpus_root: str | Path, max_patients: int) -> list[Finding]:
    """Everything above needs the charts, so nothing here runs without being handed them.

    `corpus_root` and `max_patients` are required positionally. A default corpus path is how a
    lint quietly starts reading real charts; a default patient cap is how it quietly reads all
    of them. Both are the caller's decision and are recorded in the report.
    """
    from ..chartstore.corpus import Corpus  # local: tier 1 must not import the corpus
    corpus = Corpus(Path(corpus_root))
    pids = corpus.patient_ids()[:max_patients]
    charts = [corpus.chart(p) for p in pids]
    out: list[Finding] = []

    vocab = {t.lower() for c in charts for t in c.doc_types}
    for label, s in _strata(spec):
        for tok in (t for t in _identity_tokens(s) if not any(t in v for v in vocab)):
            out.append(Finding("T2 dead_doc_types", FAIL, spec.spec_id,
                               f"{label}.strata[{s.get('name')}].match",
                               f"doc_type_matches {tok!r} selects no document type in {len(pids)} "
                               f"patients: the stratum is empty by typo, not by finding."))

    for t in (getattr(spec.proof_obligation, "required_keywords", []) or []):
        if not any(c.search(t, max_hits=1) for c in charts):
            out.append(Finding("T2 dead_terms", FAIL, spec.spec_id, "for_negative.required_keywords",
                               f"{t!r} matches no document in {len(pids)} patients; the gate demands "
                               f"a search that can only ever return nothing."))

    # A patient the gate can never pass is not a hard case, it is a spec that does not apply: the
    # run will burn its whole step budget and end in a rejection nobody can act on.
    gate, blocked = _gate(spec), []
    need = {n for r in gate_rows(spec) if (n := r["binding_n"])}
    ce = [t for _, s in _strata(spec) if s.get("name") == "can_establish"
          for t in _identity_tokens(s)]
    for c, pid in zip(charts, pids):
        docs, total = c.list_documents(limit=100_000)
        why = [f"only {total} documents, a sample of {n} cannot be drawn" for n in need if total < n]
        if (gate.get("require_can_establish_nonempty") and ce
                and not any(any(t in d.doc_type.lower() for t in ce) for d in docs)):
            why.append("no can_establish document")
        if why:
            blocked.append((pid, why))
    if blocked:
        out.append(Finding("T2 ungateable_patients", FAIL, spec.spec_id, "gate",
                           f"{len(blocked)}/{len(pids)} patients "
                           f"({len(blocked) / max(1, len(pids)):.1%}) can never pass this gate: "
                           f"{blocked[:5]}"))
    return out


# ----------------------------------------------------------- TIER 3, answer key, opt-in only
TIER3_CHECKS = {
    "field_recall": "fields the spec can never get right because the answer is not in its strata",
    "boundary_case_agreement": "declared boundary answers against the key's answers",
    "abstention_calibration": "EVIDENCE_INSUFFICIENT where the key holds a value, and the reverse",
    "conflict_rule_outcomes": "cases the key settles the opposite way to the declared ordering",
}


def tier3_checks(spec, answer_key: str | Path | None, enabled: bool) -> list[Finding]:
    """A stub that refuses. Both conditions are required and neither has a default.

    The point is the refusal, not the checks: an answer key reachable from a plain lint is an
    answer key the lint gets scored against, and then the linter's clean report and the
    system's accuracy stop being independent measurements.
    """
    if not enabled or not answer_key:
        raise AnswerKeyRefused(
            "tier 3 needs BOTH --tier3 and --answer-key <path>; refusing. Checks that would "
            f"run: {', '.join(TIER3_CHECKS)}")
    raise NotImplementedError(
        f"tier 3 is a stub: {list(TIER3_CHECKS)} against {answer_key}. Nothing in this "
        "repository has an answer key yet, and implementing scoring against a key that does "
        "not exist would be inventing the key.")


# --------------------------------------------------------------------- TIER 4, human, honest
#: What this linter cannot check, said plainly. Every line below is a question no property of
#: the file can settle, and the most useful thing a checker prints may be the list of things
#: it is not entitled to have an opinion about. Order matters: the first two decide whether a
#: clean tier-1 report means anything at all.
TIER4_HUMAN = [
    "Whether the question the spec asks is the question the registry needs answered.",
    "Whether the evidence rules are clinically right: whether an oncology note really can fix a first diagnosis date, whether cytology really can establish behaviour.",
    "Whether the thresholds are acceptable: nobody has agreed that a 12% residual chance of a missed document is tolerable for any of these variables.",
    "Whether the code tables are the published tables, or a language model's recollection of them.",
    "Whether the document-type groupings match how this institution actually names and files documents.",
    "Whether a passing gate means the chart was adequately searched, which is a claim about clinical practice and not about statistics.",
    "Whether the decision rules reproduce the manual, including the rules the model never had in context.",
    "Whether abstaining is the right behaviour for this variable, or whether an abstention downstream is worse than a wrong value.",
]


# --------------------------------------------------------------------------------- report
def _grid(bounds: Iterable[float], sizes: Iterable[int], confidence: float) -> list[str]:
    """The table a human needs in order to replace 25 and 0.12 with numbers they chose."""
    lines = [f"  minimum clean draws needed, at confidence {confidence}:"]
    for b in sorted(set(bounds)):
        lines.append(f"    max_elusion_upper {b:<6} -> n >= {min_n_for_bound(b, confidence)}")
    lines.append(f"  bound actually earned by n clean draws, at confidence {confidence}:")
    for n in sorted(set(sizes)):
        lines.append(f"    n = {n:<4} -> elusion_upper <= {bound_at_n(n, confidence):.4f}")
    return lines


def render_report(specs: list, *, corpus, answer_key, tier3_enabled: bool,
                  bounds: Iterable[float] = (), sizes: Iterable[int] = ()) -> str:
    """The whole four-tier report as text. Plain text, not rich markup: a wrapped line breaks
    grep, and this output exists to be pasted into a review."""
    lines: list[str] = []
    fails, all_rows = 0, []
    for spec in specs:
        findings, rows = lint_spec(spec), gate_rows(spec)
        all_rows += rows
        fails += sum(1 for f in findings if f.severity == FAIL)
        lines += ["=" * 100,
                  f"{spec.spec_id}   [{len(spec.fields)} fields, spec_hash {spec.spec_hash}]",
                  "-" * 100,
                  "TIER 1 - FORMAL (the file alone: no corpus, no answer key, no clinician)"]
        for check in TIER1_CHECKS:
            got = [f for f in findings if f.check == check]
            n_fail = sum(1 for f in got if f.severity == FAIL)
            lines.append(f"  {check:<36} "
                         + (f"FAIL ({n_fail})" if n_fail else ("PASS" if not got else "PASS (notes)")))
            lines += [f"      [{f.severity}] {f.where}\n             {f.message}" for f in got]
        if rows:
            head = (f"    {'scope':<34}{'cap':>6}{'conf':>7}{'n':>5}{'min n':>7}"
                    f"{'earned':>9}{'margin':>9}  verdict")
            lines += ["  GATE SATISFIABILITY", head]
            for r in rows:
                earned = "-" if r["bound_earned"] is None else format(r["bound_earned"], ".4f")
                margin = "-" if r["margin"] is None else format(r["margin"], "+.4f")
                row = (f"    {r['scope']:<34}{r['bound']:>6}{r['confidence']!s:>7}"
                       f"{r['binding_n']!s:>5}{r['min_n_required']!s:>7}"
                       f"{earned:>9}{margin:>9}  {r['verdict']}")
                lines += [row, f"      per-stratum draws: {r['samples'] or 'none'}"]
    why_grid = ("  candidate rows come from --bound / --n; every other row is what the specs "
                "already declare, so this table invents no threshold of its own.")
    lines += ["=" * 100,
              "SAMPLE SIZE / BOUND TABLE - the numbers in these specs are model-invented", why_grid]
    for c in sorted({float(r["confidence"]) for r in all_rows if r["confidence"] is not None}):
        lines += _grid([r["bound"] for r in all_rows] + list(bounds),
                       [n for r in all_rows if (n := r["binding_n"])] + list(sizes), c)
    t2 = ("TIER 2 - AGAINST THE CORPUS, no answer key: NOT RUN "
          + (f"(--corpus given: {corpus})" if corpus else "(needs --corpus <path>)"))
    lines += ["", t2]
    lines += [f"    {name:<24} {what}" for name, what in TIER2_CHECKS.items()]
    t3 = (f"TIER 3 - AGAINST AN ANSWER KEY: "
          f"{'enabled' if (tier3_enabled and answer_key) else 'REFUSED'} (needs BOTH --tier3 and "
          f"--answer-key; a key reachable from a plain lint is a key the lint gets scored against)")
    lines += ["", t3]
    lines += [f"    {name:<24} {what}" for name, what in TIER3_CHECKS.items()]
    lines += ["", "TIER 4 - IRREDUCIBLY HUMAN: not checked here, and not checkable here"]
    lines += [f"    - {item}" for item in TIER4_HUMAN]
    lines += ["", f"{len(specs)} spec(s), {fails} tier-1 failure(s)."]
    return "\n".join(lines)
