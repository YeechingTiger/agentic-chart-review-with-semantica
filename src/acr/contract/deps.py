"""L4.5: which variables a guideline rule reads, in both directions, and what a move breaks.

`concordance.py` decides a rule from variables that arrive. This module decides whether they
CAN arrive, and whether an answer already written down was computed from the specs that are
on disk now. Both questions have a silent failure mode that returns a number instead of an
error, and a number is what gets reported.

  FORWARD    a declared input nothing can supply is a GAP that stops the rule. Left alone it
             arrives as an absent variable, `_and`/`_or` fold it into UNKNOWN, and an
             `any_of` whose other branch happens to be TRUE returns CONCORDANT for a
             recommendation whose action variable was never wired up. `gated_assess` refuses
             that verdict; it never invents one.

  EXCEPTIONS "there are none" and "we forgot" are byte-identical YAML. `load_guideline_deps`
             forces them apart by refusing to load a recommendation that declares neither an
             exception nor a reason there is none — the difference between the two is a
             patient who declined chemotherapy counted as a care gap.

  BACKWARD   editing a spec must make every concordance result computed under the old one
             LOOK wrong. A stale CONCORDANT is worse than no answer, because no answer
             announces itself.

There are no model calls here and no path to one, for the same reason `concordance.py` has
none: this is a rule layer, and a reachable model is an invitation to ask it to guess a
binding. `registry_catalog` is the only router — nothing below re-derives spec_id -> field.
"""
from __future__ import annotations

import difflib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any

import yaml

from ..core import site
from .concordance import (
    _VAR_KEYS,
    INPUT_SOURCES,
    ConcordanceResult,
    Guideline,
    assess_one,
    load_guideline,
)
from .registry_catalog import VariableCatalog, normalise_name

# ------------------------------------------------------------------------------ vocabulary
SOURCE_EXTRACTION_SPEC = "extraction_spec"
SOURCE_REGISTRY = "registry_limited_dataset"
SOURCE_NOT_YET_EXTRACTABLE = "not_yet_extractable"

GAP_NOT_YET_EXTRACTABLE = "not_yet_extractable"
GAP_NO_SUCH_SPEC_FIELD = "no_such_spec_field"
GAP_SPEC_ID_MISMATCH = "spec_id_mismatch"
GAP_SPEC_CANNOT_ANSWER_FROM_NOTES = "spec_cannot_answer_from_notes"
GAP_UNKNOWN_SOURCE = "unknown_source"

#: Which half of the rule reads a variable. L5 holds exception variables to the exception
#: standard and driving variables to the coverage standard, so a variable read by two blocks
#: must carry both classes -- assigning it to one drops a driving variable from the other.
ELIGIBILITY = "eligibility"
ACTION = "action"
TIMING = "timing"
EXCEPTION = "exception"
CLASS_ORDER = (ELIGIBILITY, ACTION, TIMING, EXCEPTION)

#: Ops that compare two dates and therefore make their variables *timing* variables wherever
#: they sit. Every key of `concordance._VAR_KEYS` is one today, and
#: `test_every_temporal_op_in_the_engine_is_known_to_this_module` fails if a new one is added
#: there and not here: unlisted, its endpoints would silently lose the `timing` class.
TEMPORAL_OPS = frozenset({"days_between", "on_or_before"})

#: The sentinel and the key it must be written under. It may NOT be written into `exceptions`
#: itself -- `concordance.parse_guideline` iterates that key, so a bare string there makes the
#: file unloadable by the scorer, and a grammar only this module can read is a second format.
NONE_DECLARED = "none_declared"
NONE_DECLARED_KEY = "exceptions_none_declared"

#: Restatements of the field name. The reason has to say why none is CORRECT, because that
#: sentence is the only thing a clinical reviewer can disagree with.
_SHRUGS = frozenset({"", "-", "?", "n/a", "na", "no", "nil", "none", "none declared",
                     NONE_DECLARED, "not applicable", "tbd", "todo", "unknown"})
_MIN_REASON_WORDS = 5

#: Written twice on purpose: `cli.py` imports this module, so this module cannot import
#: `cli.py`. `test_the_schema_strings_this_module_scans_for_are_the_ones_the_cli_writes` is
#: the tripwire -- a drifted constant makes every artifact invisible, and invisible reads as
#: "nothing is stale".
EXTRACT_SCHEMA = "acr.extract/1"
CONCORD_SCHEMA = "acr.concord/1"
DEPS_SCHEMA = "acr.contract.deps/1"
PROVENANCE_KEY = "dependency_provenance"

RULE_DEPENDENCY_GAP = "dependency_gap"

CURRENT = "CURRENT"
STALE = "STALE"
#: The dangerous middle state. Provenance that cannot be reached is not evidence of currency,
#: and calling it STALE would cry wolf, so it gets its own verdict.
UNVERIFIABLE = "UNVERIFIABLE"


class UndeclaredExceptionsError(ValueError):
    """A recommendation says nothing about its exceptions. Loading it anyway is failure D."""


# ================================================================== FORWARD: the dependency
@dataclass(frozen=True)
class InputDep:
    """One `required_inputs` entry, and the supply route it does or does not have."""

    name: str
    source: str
    item: str = ""
    declared_spec_id: str | None = None
    spec_id: str | None = None
    spec_hash: str = ""
    resolved: bool = False
    gap_kind: str = ""
    gap_detail: str = ""
    predicate_classes: tuple[str, ...] = ()

    def gap(self) -> dict | None:
        if self.resolved:
            return None
        return {"name": self.name, "kind": self.gap_kind, "detail": self.gap_detail,
                "declared_source": self.source, "declared_spec_id": self.declared_spec_id,
                "item": self.item}

    def to_dict(self) -> dict:
        return asdict(self) | {"predicate_classes": list(self.predicate_classes)}


@dataclass(frozen=True)
class RecommendationDeps:
    recommendation_id: str
    inputs: tuple[InputDep, ...] = ()
    exceptions_declared: tuple[str, ...] = ()
    exceptions_none_declared_reason: str = ""

    @property
    def resolved(self) -> list[str]:
        return [i.name for i in self.inputs if i.resolved]

    @property
    def gaps(self) -> list[dict]:
        # resolved + gaps must PARTITION required_inputs: an input in neither bucket is one
        # the impact analysis will never notice changing.
        return [g for g in (i.gap() for i in self.inputs) if g]

    @property
    def predicate_classes(self) -> dict[str, list[str]]:
        return {i.name: list(i.predicate_classes) for i in self.inputs}

    @property
    def spec_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(i.spec_id for i in self.inputs if i.resolved and i.spec_id))

    def to_dict(self) -> dict:
        return {"recommendation_id": self.recommendation_id,
                "inputs": [i.to_dict() for i in self.inputs],
                "resolved": self.resolved, "gaps": self.gaps,
                "predicate_classes": self.predicate_classes,
                "spec_ids": list(self.spec_ids),
                "exceptions_declared": list(self.exceptions_declared),
                "exceptions_none_declared_reason": self.exceptions_none_declared_reason}


@dataclass(frozen=True)
class SpecDeps:
    spec_id: str
    current_hash: str
    recommendations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuidelineDeps:
    guideline: Guideline
    catalog: VariableCatalog
    per_recommendation: tuple[RecommendationDeps, ...] = ()

    def for_recommendation(self, rec_id: str) -> RecommendationDeps:
        for rd in self.per_recommendation:
            if rd.recommendation_id == rec_id:
                return rd
        raise KeyError(rec_id)

    def backward(self) -> dict[str, dict]:
        """spec_id -> the recommendations and the named inputs that read it.

        Every catalogued spec gets an entry, including the ones nothing reads: raising
        KeyError for those would make "nothing depends on this" indistinguishable from "you
        typo'd the spec id", and only one of those is safe to edit.
        """
        out: dict[str, dict] = {
            sid: {"spec_id": sid, "current_hash": s.spec_hash, "spec_version": s.spec_version,
                  "recommendations": [], "inputs": []}
            for sid, s in sorted(self.catalog.specs.items())}
        for rd in self.per_recommendation:
            for i in rd.inputs:
                # Every resolved spec_id came out of this catalogue, so it is already a key.
                if not (i.resolved and i.spec_id):
                    continue
                e = out[i.spec_id]
                if rd.recommendation_id not in e["recommendations"]:
                    e["recommendations"].append(rd.recommendation_id)
                e["inputs"].append([rd.recommendation_id, i.name])
        return out

    def for_spec(self, spec_id: str) -> SpecDeps:
        if spec_id not in self.catalog.specs:
            raise KeyError(f"no spec {spec_id!r} in {self.catalog.directory or 'the catalogue'}; "
                           f"known: {', '.join(sorted(self.catalog.specs))}")
        b = self.backward()[spec_id]
        return SpecDeps(spec_id, b["current_hash"], tuple(b["recommendations"]))

    def manifest(self) -> dict:
        """The whole graph, hashes included, as one JSON document."""
        g, recs = self.guideline, self.per_recommendation
        return {
            "schema": DEPS_SCHEMA,
            "guideline": {"guideline_id": g.guideline_id, "guideline_hash": g.guideline_hash,
                          "guideline_version": g.guideline_version},
            "specs": {sid: s.identity() | {"data_source": s.data_source}
                      for sid, s in sorted(self.catalog.specs.items())},
            "forward": {"per_recommendation": {rd.recommendation_id: rd.to_dict() for rd in recs},
                        "totals": {"inputs": sum(len(rd.inputs) for rd in recs),
                                   "gaps": sum(len(rd.gaps) for rd in recs)}},
            "backward": self.backward(),
            "exceptions_declared_per_rec": {rd.recommendation_id: list(rd.exceptions_declared)
                                            for rd in self.per_recommendation},
            "exceptions_none_declared_reason": {
                rd.recommendation_id: rd.exceptions_none_declared_reason
                for rd in self.per_recommendation if rd.exceptions_none_declared_reason},
        }


# ------------------------------------------------------------------------ predicate classes
def _classify(cond: Any, block: str, out: dict[str, set[str]]) -> None:
    """Record which block reads each variable, reaching inside `any_of`/`all_of`/`not`.

    A scan that stopped at the top level would classify neither branch of the stage I
    `any_of` and report both its variables as read by nothing.

    The class comes from the OP the variable sits in, not from the variable's type:
    `is_present(date_of_first_systemic_therapy)` in `applies_when` is a population test --
    the trigger event happened -- not a window, and calling it timing would put it in the
    wrong half of the L5 split.
    """
    if not isinstance(cond, dict):
        return
    op = cond.get("op")
    if op in ("all_of", "any_of", "not"):
        for s in cond.get("conditions") or [cond.get("condition")]:
            _classify(s, block, out)
        return
    for key in _VAR_KEYS.get(str(op), ("var",)):
        name = cond.get(key)
        if not name:
            continue
        s = out.setdefault(str(name), set())
        s.add(block)
        if op in TEMPORAL_OPS:
            s.add(TIMING)


def _predicate_classes(rec) -> dict[str, tuple[str, ...]]:
    acc: dict[str, set[str]] = {}
    for c in rec.applies_when:
        _classify(c, ELIGIBILITY, acc)
    for c in rec.satisfied_when:
        _classify(c, ACTION, acc)
    for e in rec.exceptions:
        for c in e.when:
            _classify(c, EXCEPTION, acc)
    return {n: tuple(k for k in CLASS_ORDER if k in v) for n, v in acc.items()}


# ------------------------------------------------------------------------------- resolution
def _resolve_extraction_input(name: str, declared: str | None, catalog: VariableCatalog,
                              idx: dict) -> tuple[str, str, str | None]:
    """(gap_kind, detail, spec_id) for an `extraction_spec` input; gap_kind "" if it resolves."""
    targets = idx.get(normalise_name(name)) or []
    fields = sorted({sid for sid, f in targets if f == name})
    if not fields:
        # The C3412-shaped bug at the binding layer: the name is written down, nothing
        # produces it, and every case comes back NOT_ASSESSABLE naming a variable the
        # operator believes was extracted. Name the fields the DECLARED spec produces -- the
        # intended one is nearly always among them, which is what makes this actionable.
        near = ([f.name for f in catalog.specs[declared].fields]
                if declared in catalog.specs else
                difflib.get_close_matches(normalise_name(name), catalog.known_names(), n=3))
        return GAP_NO_SUCH_SPEC_FIELD, (
            f"no spec in {catalog.directory or 'the catalogue'} declares a field named {name!r}"
            f" (the name reaches {', '.join(sorted({s for s, _ in targets})) or 'nothing'}). "
            f"Candidates: {', '.join(near) or '(none)'}"), None
    if declared and str(declared) not in fields:
        # The name resolves, but to a different spec than the guideline says. Trusting the
        # name and ignoring the declaration would bind the rule to a spec no reviewer
        # approved.
        return GAP_SPEC_ID_MISMATCH, (
            f"{name!r} is declared as coming from {declared!r}, but the catalogue routes that "
            f"field to {', '.join(fields)}. One of the two is wrong and neither may be guessed."
        ), None
    sid = str(declared) if declared else fields[0]
    if catalog.specs[sid].data_source != "notes":
        # graph.py forces SPEC_INSUFFICIENT / WRONG_DATA_SOURCE at finalize for these, so the
        # variable provably never arrives. Calling the binding `resolved` promises data that
        # no run can produce -- the supply route is the registry feed, and saying so is the
        # difference between a gap that can be closed and one that cannot.
        return GAP_SPEC_CANNOT_ANSWER_FROM_NOTES, (
            f"spec {sid} declares data_source={catalog.specs[sid].data_source!r}, so every agent "
            f"run over it is forced to SPEC_INSUFFICIENT / WRONG_DATA_SOURCE at finalize. Declare "
            f"this input `source: {SOURCE_REGISTRY}` instead of naming a spec that cannot answer."
        ), None
    return "", "", sid


def _input_dep(decl: dict, catalog: VariableCatalog, idx: dict,
               classes: dict[str, tuple[str, ...]]) -> InputDep:
    name = str(decl.get("name") or "")
    source = str(decl.get("source") or "")
    declared = decl.get("spec_id")
    base = InputDep(name=name, source=source, item=str(decl.get("item") or ""),
                    declared_spec_id=str(declared) if declared else None,
                    predicate_classes=classes.get(name, ()))
    if source == SOURCE_REGISTRY:
        # Not a gap. The value comes from the registry feed, which is a supply route and not
        # a hole -- STORE.610 exists and answers SPEC_INSUFFICIENT by design.
        return _replace(base, resolved=True)
    if source == SOURCE_NOT_YET_EXTRACTABLE:
        return _replace(base, gap_kind=GAP_NOT_YET_EXTRACTABLE, gap_detail=(
            f"declared with no extractor behind it (registry item: {base.item or 'unstated'}). "
            f"Nothing can supply it, so the rule cannot be scored."))
    if source != SOURCE_EXTRACTION_SPEC:
        # `parse_guideline` does not validate, so a typo'd source reaches here. Defaulting it
        # to anything is the resolver inventing a supply route.
        return _replace(base, gap_kind=GAP_UNKNOWN_SOURCE, gap_detail=(
            f"source {source!r} is not one of {sorted(INPUT_SOURCES)}; it names no supply route"))
    kind, detail, sid = _resolve_extraction_input(name, base.declared_spec_id, catalog, idx)
    if kind:
        return _replace(base, gap_kind=kind, gap_detail=detail)
    return _replace(base, resolved=True, spec_id=sid, spec_hash=catalog.specs[sid].spec_hash)


def build_dependencies(guideline: Guideline, catalog: VariableCatalog) -> GuidelineDeps:
    """The forward graph. Does not enforce the exception rule -- `load_guideline_deps` does,
    because that rule is about what the FILE says and needs the raw YAML to see it."""
    idx = catalog._index()
    raw_recs = {str(r.get("id")): r for r in (guideline.raw.get("recommendations") or [])
                if isinstance(r, dict)}
    out = []
    for rec in guideline.recommendations:
        classes = _predicate_classes(rec)
        reason = _none_declared_text((raw_recs.get(rec.id) or {}).get(NONE_DECLARED_KEY))
        out.append(RecommendationDeps(
            recommendation_id=rec.id,
            inputs=tuple(_input_dep(d, catalog, idx, classes)
                         for d in rec.required_inputs if isinstance(d, dict)),
            exceptions_declared=tuple(e.id for e in rec.exceptions),
            exceptions_none_declared_reason=reason))
    return GuidelineDeps(guideline, catalog, tuple(out))


# ============================================================== THE EXCEPTION RULE
def _none_declared_text(raw: Any) -> str:
    """The reason, from either spelling. A mapping is allowed so a reviewer can be named."""
    if isinstance(raw, dict):
        return " ".join(str(raw.get("reason") or "").split())
    return "" if raw is None else " ".join(str(raw).split())


def _enforce_exception_declaration(doc: dict, where: str) -> None:
    """Refuse a guideline whose recommendations are silent about their exceptions.

    Silence is indistinguishable from "we forgot", and the cost of getting it wrong is a
    patient who declined treatment counted as a care gap -- failure D, and the most common
    way a concordance study produces a wrong and damaging number.
    """
    for r in (doc.get("recommendations") or []):
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "<unnamed>")
        exc = r.get("exceptions")
        declared_none = NONE_DECLARED_KEY in r
        if isinstance(exc, str):
            raise UndeclaredExceptionsError(
                f"recommendation {rid}: `exceptions: {exc}` is refused. "
                f"concordance.parse_guideline iterates `exceptions`, so a bare string there "
                f"would be iterated character by character and the file becomes unloadable by "
                f"the scorer. Write the sentence under `{NONE_DECLARED_KEY}` instead.")
        if exc and declared_none:
            raise UndeclaredExceptionsError(
                f"recommendation {rid} declares both `exceptions` and `{NONE_DECLARED_KEY}`. "
                f"One of them is false and nothing here can tell which.")
        if exc:
            continue
        if not declared_none:
            raise UndeclaredExceptionsError(
                f"{where}: recommendation {rid} declares no exceptions, and an empty or absent "
                f"`exceptions` list is byte-identical to having forgotten them. Add "
                f"`{NONE_DECLARED_KEY}: <why none is correct>` if there genuinely are none.")
        text = _none_declared_text(r.get(NONE_DECLARED_KEY))
        if text.lower() in _SHRUGS or len(text.split()) < _MIN_REASON_WORDS:
            raise UndeclaredExceptionsError(
                f"recommendation {rid}: `{NONE_DECLARED_KEY}: {text!r}` restates the field name "
                f"instead of giving a reason. Say why none is CORRECT -- that sentence is the "
                f"only thing a clinical reviewer can disagree with.")


def load_guideline_deps(path: str | Path, *, specs_dir: str | Path = str(site.specs_root()),
                        catalog: VariableCatalog | None = None) -> GuidelineDeps:
    """Load a guideline for dependency analysis, refusing an undeclared exception list.

    The exception check runs on the raw YAML and BEFORE `load_guideline`, because
    `exceptions: none_declared` crashes `parse_guideline` before any check could see it.
    """
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    _enforce_exception_declaration(doc, str(p))
    cat = catalog or VariableCatalog.from_directory(specs_dir)
    return build_dependencies(load_guideline(p), cat)


# ================================================== a gap stops the rule from evaluating
def gated_assess(variables: dict[str, Any], deps: GuidelineDeps, *,
                 recommendation_ids: Iterable[str] | None = None) -> list[ConcordanceResult]:
    """`concordance.assess`, with a refusal in front of it.

    The gate adds a refusal; it must not add an opinion. With every input resolved the result
    is the engine's, byte for byte.
    """
    wanted = set(recommendation_ids) if recommendation_ids is not None else None
    out: list[ConcordanceResult] = []
    for rec in deps.guideline.recommendations:
        if wanted is not None and rec.id not in wanted:
            continue
        r = assess_one(rec, variables, deps.guideline)
        rd = deps.for_recommendation(rec.id)
        if not rd.gaps:
            out.append(r)
            continue
        notes = list(r.notes)
        for g in rd.gaps:
            if g["name"] in (variables or {}):
                # The data is real; the DECLARATION is now wrong, and a wrong declaration is
                # what the whole forward direction is enforcing. Neither trust it silently
                # nor discard it -- say so.
                notes.append(f"stale declaration: {g['name']} was supplied by the caller but "
                             f"the guideline declares it {g['kind']}; fix the declaration")
        names = [g["name"] for g in rd.gaps]
        if r.outcome == "NOT_APPLICABLE":
            # The one verdict a gap may not overturn, and the reason is arithmetic, not
            # policy: a gapped variable never arrives, an absent variable is UNKNOWN, and
            # UNKNOWN cannot produce FALSE under `_and`. So a NOT_APPLICABLE was settled by a
            # variable that DID arrive, and no unbuilt extractor can make small cell into
            # NSCLC. Forcing NOT_ASSESSABLE here would move determinately-out-of-population
            # patients into the unknown pile, inflating the unknowns exactly the way scoring
            # them would inflate the rate.
            notes.append(f"{RULE_DEPENDENCY_GAP}: {len(names)} declared input(s) have no supply "
                         f"route ({', '.join(names)}), but the population test was settled by "
                         f"variables that did arrive, so this exclusion stands")
            out.append(_replace(r, notes=tuple(notes)))
            continue
        # Every unresolved input, not just the first: a list that stopped early would send an
        # operator to build one extractor and rerun into the same refusal.
        blocking = tuple(dict.fromkeys(names + list(r.blocking_inputs)))
        out.append(_replace(
            r, outcome="NOT_ASSESSABLE", rule_applied=RULE_DEPENDENCY_GAP,
            reason=(f"the recommendation cannot be evaluated: {len(names)} declared input(s) "
                    f"have no supply route ({', '.join(names)}). Care that was never checked "
                    f"must not be scored as care that was delivered."),
            blocking_inputs=blocking, exception_id=None, notes=tuple(notes)))
    return out


# ========================================================= BACKWARD: hash invalidation
def provenance_block(guideline: Guideline, specs: dict) -> dict:
    """What a concord.json must carry to be checkable without reaching for its extract."""
    return {"guideline": {"guideline_id": guideline.guideline_id,
                          "guideline_version": guideline.guideline_version,
                          "guideline_hash": guideline.guideline_hash},
            "specs": {sid: s.identity() for sid, s in sorted(specs.items())}}


@dataclass(frozen=True)
class ArtifactVerdict:
    path: str
    verdict: str
    reason: str
    changed: tuple[dict, ...] = ()
    affected_results: int = 0
    affected_outcomes: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return asdict(self) | {"changed": list(self.changed)}


def _recorded_hashes(doc: dict, path: Path) -> tuple[str, dict[str, str], str]:
    """(guideline_hash, {spec_id: spec_hash}, unverifiable_reason)."""
    prov = doc.get(PROVENANCE_KEY) or {}
    ghash = str((prov.get("guideline") or doc.get("guideline") or {}).get("guideline_hash") or "")
    if not ghash:
        return "", {}, "the artifact records no guideline_hash, so nothing can be compared"
    if prov.get("specs"):
        return ghash, {k: str(v.get("spec_hash") or "") for k, v in prov["specs"].items()}, ""
    src = doc.get("extract_input") or ""
    p = Path(src) if src else None
    if not p or not p.exists():
        return ghash, {}, (f"the extract this concord.json reaches its spec hashes through is "
                           f"gone ({src or 'unrecorded'}), so the spec identity behind every "
                           f"result is simply unknown")
    specs = json.loads(p.read_text(encoding="utf-8")).get("specs") or {}
    return ghash, {k: str(v.get("spec_hash") or "") for k, v in specs.items()}, ""


def classify_artifact(path: str | Path, deps: GuidelineDeps) -> ArtifactVerdict:
    """CURRENT / STALE / UNVERIFIABLE for one concord.json, against the files on disk now.

    The artifact is not rewritten, not deleted and not touched: the same bytes must stop
    reading as current the moment a spec they were computed from changes.
    """
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != CONCORD_SCHEMA:
        return ArtifactVerdict(str(p), UNVERIFIABLE,
                               f"not a {CONCORD_SCHEMA} artifact (schema={doc.get('schema')!r})")
    ghash, spec_hashes, why = _recorded_hashes(doc, p)
    if why:
        return ArtifactVerdict(str(p), UNVERIFIABLE, why)

    needed = {sid for rd in deps.per_recommendation for sid in rd.spec_ids}
    missing = sorted(needed - set(spec_hashes))
    if missing:
        # Provenance covering three of four specs cannot certify the fourth. Checking only
        # what happens to be recorded would let a spec drop out of the manifest and take its
        # own staleness check with it.
        return ArtifactVerdict(str(p), UNVERIFIABLE, (
            f"the provenance records no hash for {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} read by this guideline"))

    changed: list[dict] = []
    g = deps.guideline
    if ghash != g.guideline_hash:
        changed.append({"id": g.guideline_id, "kind": "guideline",
                        "recorded": ghash, "current": g.guideline_hash})
    for sid in sorted(needed):
        cur = deps.catalog.specs[sid].spec_hash
        if spec_hashes.get(sid) != cur:
            changed.append({"id": sid, "kind": "spec",
                            "recorded": spec_hashes.get(sid), "current": cur})
    if not changed:
        return ArtifactVerdict(str(p), CURRENT,
                               "every spec hash and the guideline hash match the files on disk")

    moved = {c["id"] for c in changed if c["kind"] == "spec"}
    guideline_moved = any(c["kind"] == "guideline" for c in changed)
    affected = {rd.recommendation_id for rd in deps.per_recommendation
                if guideline_moved or moved & set(rd.spec_ids)}
    counts: dict[str, int] = {}
    n = 0
    for row in doc.get("patients") or []:
        for r in row.get("results") or []:
            if r.get("recommendation_id") in affected:
                n += 1
                k = str(r.get("outcome"))
                counts[k] = counts.get(k, 0) + 1
    return ArtifactVerdict(str(p), STALE, (
        f"{', '.join(c['id'] for c in changed)} changed since this artifact was written; "
        f"{n} result(s) were computed under the old definition"),
        tuple(changed), n, counts)


@dataclass(frozen=True)
class SpecImpact:
    spec_id: str
    recommendations: tuple[str, ...] = ()
    artifacts: tuple[ArtifactVerdict, ...] = ()

    @property
    def stale_results(self) -> int:
        return sum(a.affected_results for a in self.artifacts if a.verdict == STALE)

    def to_dict(self) -> dict:
        return {"spec_id": self.spec_id, "recommendations": list(self.recommendations),
                "artifacts": [a.to_dict() for a in self.artifacts],
                "stale_results": self.stale_results,
                # Counted beside the stale ones and never folded into them: an artifact whose
                # provenance cannot be reached is not evidence that nothing moved.
                "unverifiable": sum(1 for a in self.artifacts if a.verdict == UNVERIFIABLE)}


def find_concord_artifacts(root: str | Path) -> list[Path]:
    """Every concord artifact under `root`, found by its SCHEMA and never by its filename.

    `concord --out` takes any filename, so a filename glob would miss a renamed artifact --
    and a missed artifact is reported as nothing rather than as stale.
    """
    out = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("schema") == CONCORD_SCHEMA:
                out.append(p)
        except (OSError, ValueError, AttributeError):
            continue
    return out


def impact_of_spec(spec_id: str, deps: GuidelineDeps, root: str | Path) -> SpecImpact:
    """What editing one spec would invalidate: which rules read it, and which answers move."""
    sd = deps.for_spec(spec_id)
    return SpecImpact(spec_id, sd.recommendations,
                      tuple(classify_artifact(p, deps) for p in find_concord_artifacts(root)))
