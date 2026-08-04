"""What a run DID, and what the reference says it should have done — the two things every plane
reads and no plane owns.

EXTRACTED FROM `contract/spec_repair.py` ON 2026-08-03, and the reason is a measurement rather than
taste. That file was 1,152 lines about proposing and validating patches to a specification, which is
one plane's business. But six imports reached into it from three OTHER planes:

    review/conflict_refinement.py      BehaviorSignature
    diagnosis/attribution.py           BehaviorSignature, ChartObservableGold, safe_case_id
    commands/cli_attribute.py          load_gold
    commands/cli_gold.py               the module

So when the tree was cut into four distributions, three of them imported a module assigned to the
fourth and none declared it as a dependency. The suites passed anyway, because the verification
environment had all four installed — which is exactly how a boundary rots without anyone noticing.
`tools/verify_structure.py` is the check that now says so.

TWO THINGS LIVE HERE, and they are here together because they are the same thing seen from two
sides.

A BEHAVIOUR SIGNATURE is what one run did, reduced to something two runs can be COMPARED on: the
answer, plus what it cited, which rules it claimed, and how it got there — hashed, so that "these
two runs behaved the same" is a comparison and not an impression. Deliberately not the answer alone:
two runs agreeing on a value while citing different documents have not behaved the same, and the
distinction is the whole point of the object.

CHART-OBSERVABLE GOLD is what the reference says, plus the honest qualifier that a reference is not
always truth. A registry value may be DERIVABLE from the chart, PARTIALLY_DERIVABLE, NOT_DERIVABLE
or UNRESOLVED, and an adjudication may find the KEY itself wrong. A plane that treats a reference as
truth cannot report the case where the reference is the thing that is broken — and this repository
has three charts (SYNK01-03) that exist because that case is real.

Portability is enforced here rather than trusted: `_portable_run_id` and `_portable_source` refuse
to copy a filesystem path or a run id into a signature, because a signature exists to LEAVE the
machine that made it, and a path is both a leak and a thing that will not resolve anywhere else.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.site import looks_like_a_person_id

GOLD_SCHEMA = "acr.chart_observable_gold/1"

DERIVABLE = "DERIVABLE"

PARTIALLY_DERIVABLE = "PARTIALLY_DERIVABLE"

NOT_DERIVABLE = "NOT_DERIVABLE"

UNRESOLVED = "UNRESOLVED"

DERIVABILITY_VALUES = (DERIVABLE, PARTIALLY_DERIVABLE, NOT_DERIVABLE, UNRESOLVED)

KEY_CORRECT = "key_correct"

KEY_WRONG = "key_wrong"

OUTSIDE_CHART = "outside_chart"

ADJUDICATION_UNRESOLVED = "unresolved"

ADJUDICATION_VALUES = (KEY_CORRECT, KEY_WRONG, OUTSIDE_CHART, ADJUDICATION_UNRESOLVED)

FOUND = "FOUND"

EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"

SPEC_INSUFFICIENT = "SPEC_INSUFFICIENT"

NOT_APPLICABLE = "NOT_APPLICABLE"

SEMANTIC_STATUSES = (FOUND, EVIDENCE_INSUFFICIENT, SPEC_INSUFFICIENT, NOT_APPLICABLE)

class SpecRepairError(ValueError):
    """A develop-plane artifact is incomplete, inconsistent, or unsafe to use."""

class GoldNotUsable(SpecRepairError):
    """The supplied registry label has not become chart-observable gold."""

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)

def _hash(*parts: Any) -> str:
    return hashlib.sha256("\0".join(_canonical(p) for p in parts).encode()).hexdigest()[:16]

def _normalise(value: Any) -> Any:
    """Normalise values for equality without turning missing into an empty string."""
    if isinstance(value, str):
        return " ".join(value.split()).strip().lower()
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value

def _safe_case_id(value: Any) -> str:
    case_id = str(value or "").strip()
    if not case_id:
        raise SpecRepairError("case_id is required")
    if looks_like_a_person_id(case_id):
        raise SpecRepairError(
            "case_id looks like a real person_id; pseudonymise it before creating a "
            "develop-plane artifact")
    return case_id

def safe_case_id(value: Any) -> str:
    """Validate a portable, pseudonymous case identifier."""
    return _safe_case_id(value)

def artifact_hash(*parts: Any) -> str:
    """Return the stable short digest used to bind repair artifacts."""
    return _hash(*parts)

def _query_terms(query: object) -> list[str]:
    """One search argument -> the terms it searched for. `evals._query_terms`'s rule, restated.

    NOT imported: `contract/` may not depend on `evaluation/` (`tests/test_layering.py` pins the
    direction). So the rule is duplicated deliberately, and
    `tests/test_search_terms_agree_everywhere.py` is what keeps the two honest — a duplicate with a
    test that compares them is a different thing from a duplicate nobody checks.
    """
    if query is None:
        return []
    if isinstance(query, (list, tuple, set)):
        return [str(t) for t in query]
    return [str(query)]


def _portable_source(source: str) -> str:
    """Name a controlled input without copying a path that may contain a person id."""
    return f"manifest:{_hash(source)}" if source else ""

def _portable_run_id(run_id: str) -> str:
    return f"run:{_hash(run_id)}" if looks_like_a_person_id(run_id) else run_id

@dataclass(frozen=True)
class GoldField:
    """One field's chart-observable answer, including a correct abstention."""

    status: str
    value: Any = None
    reason_code: str = ""

    def __post_init__(self) -> None:
        if self.status not in SEMANTIC_STATUSES:
            raise SpecRepairError(
                f"gold field status {self.status!r} is not one of {SEMANTIC_STATUSES}")
        if self.status == FOUND and self.value is None:
            raise SpecRepairError("a FOUND gold field needs a value")
        if self.status != FOUND and self.value not in (None, "", [], {}):
            raise SpecRepairError(
                f"a {self.status} gold field must not carry an asserted value: {self.value!r}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoldField:
        return cls(status=str(value.get("status") or ""), value=value.get("value"),
                   reason_code=str(value.get("reason_code") or ""))

    def to_dict(self) -> dict:
        return {"status": self.status, "value": self.value, "reason_code": self.reason_code}

@dataclass(frozen=True)
class GoldEvidence:
    """An adjudicated witness or contradiction; quotes remain outside the repository."""

    note_id: str
    fields: tuple[str, ...]
    stance: str = "supports"
    quote: str = ""
    document_role: str = ""
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if not self.note_id.strip():
            raise SpecRepairError("gold evidence needs note_id")
        if not self.fields:
            raise SpecRepairError(f"gold evidence {self.note_id!r} needs at least one field")
        if self.stance not in ("supports", "contradicts"):
            raise SpecRepairError("gold evidence stance must be supports or contradicts")
        if (self.start is None) != (self.end is None):
            raise SpecRepairError("gold evidence offsets require both start and end")
        if self.start is not None and (self.start < 0 or self.end <= self.start):
            raise SpecRepairError("gold evidence offsets must satisfy 0 <= start < end")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoldEvidence:
        return cls(note_id=str(value.get("note_id") or ""),
                   fields=tuple(str(x) for x in (value.get("fields") or ())),
                   stance=str(value.get("stance") or "supports"),
                   quote=str(value.get("quote") or ""),
                   document_role=str(value.get("document_role") or ""),
                   start=(int(value["start"]) if value.get("start") is not None else None),
                   end=(int(value["end"]) if value.get("end") is not None else None))

    def to_dict(self) -> dict:
        return {"note_id": self.note_id, "fields": list(self.fields), "stance": self.stance,
                "quote": self.quote, "document_role": self.document_role,
                "start": self.start, "end": self.end}

@dataclass(frozen=True)
class ChartObservableGold:
    """Registry reference plus the answer an agent is entitled to derive from this chart."""

    case_id: str
    spec_id: str
    registry_value: Mapping[str, Any]
    registry_source_version: str
    chart_derivability: str
    chart_answer: Mapping[str, GoldField]
    gold_evidence: tuple[GoldEvidence, ...]
    adjudication: str
    semantic_hash: str = ""
    subgroups: tuple[str, ...] = ()
    adjudication_rationale: str = ""

    def __post_init__(self) -> None:
        _safe_case_id(self.case_id)
        if not self.spec_id.strip():
            raise SpecRepairError(f"{self.case_id}: spec_id is required")
        if self.chart_derivability not in DERIVABILITY_VALUES:
            raise SpecRepairError(
                f"{self.case_id}: chart_derivability must be one of {DERIVABILITY_VALUES}")
        if self.adjudication not in ADJUDICATION_VALUES:
            raise SpecRepairError(
                f"{self.case_id}: adjudication must be one of {ADJUDICATION_VALUES}")
        if self.chart_derivability in (DERIVABLE, PARTIALLY_DERIVABLE) and not self.chart_answer:
            raise SpecRepairError(
                f"{self.case_id}: chart-derivable gold needs field-level chart_answer")
        if self.adjudication == OUTSIDE_CHART and self.chart_derivability != NOT_DERIVABLE:
            raise SpecRepairError(
                f"{self.case_id}: outside_chart adjudication requires NOT_DERIVABLE")
        unknown = {f for e in self.gold_evidence for f in e.fields} - set(self.chart_answer)
        if unknown:
            raise SpecRepairError(
                f"{self.case_id}: gold evidence names unknown field(s) {sorted(unknown)}")

    @property
    def usable_for_repair(self) -> bool:
        return (self.chart_derivability in (DERIVABLE, PARTIALLY_DERIVABLE)
                and self.adjudication in (KEY_CORRECT, KEY_WRONG)
                and (self.adjudication != KEY_WRONG or bool(self.adjudication_rationale.strip()))
                and bool(self.chart_answer))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChartObservableGold:
        ans = value.get("chart_answer") or {}
        if not isinstance(ans, Mapping):
            raise SpecRepairError("chart_answer must be an object keyed by field")
        return cls(
            case_id=_safe_case_id(value.get("case_id")),
            spec_id=str(value.get("spec_id") or ""),
            registry_value=dict(value.get("registry_value") or {}),
            registry_source_version=str(value.get("registry_source_version") or ""),
            chart_derivability=str(value.get("chart_derivability") or UNRESOLVED),
            chart_answer={str(k): GoldField.from_dict(v) for k, v in ans.items()},
            gold_evidence=tuple(GoldEvidence.from_dict(v)
                                for v in (value.get("gold_evidence") or ())),
            adjudication=str(value.get("adjudication") or ADJUDICATION_UNRESOLVED),
            semantic_hash=str(value.get("semantic_hash") or ""),
            subgroups=tuple(str(x) for x in (value.get("subgroups") or ())),
            adjudication_rationale=str(value.get("adjudication_rationale") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "spec_id": self.spec_id,
            "registry_value": dict(self.registry_value),
            "registry_source_version": self.registry_source_version,
            "chart_derivability": self.chart_derivability,
            "chart_answer": {k: v.to_dict() for k, v in self.chart_answer.items()},
            "gold_evidence": [e.to_dict() for e in self.gold_evidence],
            "adjudication": self.adjudication, "semantic_hash": self.semantic_hash,
            "subgroups": list(self.subgroups),
            "adjudication_rationale": self.adjudication_rationale,
        }

def load_gold(path: str | Path) -> dict[str, ChartObservableGold]:
    """Load the explicit gold contract; legacy answer keys are refused, not guessed."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise SpecRepairError(f"gold file cannot be read: {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise SpecRepairError(f"gold file {p}: {e}") from e
    if not isinstance(raw, Mapping) or raw.get("schema") != GOLD_SCHEMA:
        raise SpecRepairError(
            f"{p}: expected schema {GOLD_SCHEMA!r}; an unresolved registry reference is not "
            "chart-observable gold until derivability and adjudication are recorded explicitly")
    rows = raw.get("cases")
    if not isinstance(rows, list):
        raise SpecRepairError(f"{p}: cases must be a list")
    out: dict[str, ChartObservableGold] = {}
    for i, row in enumerate(rows):
        try:
            gold = ChartObservableGold.from_dict(row)
        except (TypeError, SpecRepairError) as e:
            raise SpecRepairError(f"{p}: cases[{i}]: {e}") from e
        if gold.case_id in out:
            raise SpecRepairError(f"{p}: duplicate case_id {gold.case_id!r}")
        out[gold.case_id] = gold
    return out

def gold_document(cases: Iterable[ChartObservableGold]) -> dict:
    rows = list(cases)
    return {"schema": GOLD_SCHEMA, "cases": [r.to_dict() for r in rows],
            "summary": {
                "n_cases": len(rows),
                "n_repair_eligible": sum(r.usable_for_repair for r in rows),
                "by_derivability": dict(Counter(r.chart_derivability for r in rows)),
                "by_adjudication": dict(Counter(r.adjudication for r in rows)),
            }}

def audit_gold(cases: Iterable[ChartObservableGold]) -> dict:
    rows = list(cases)
    findings: list[dict] = []
    if not rows:
        findings.append({"case_id": "<cohort>", "severity": "BLOCK",
                         "finding": "gold cohort is empty"})
    for row in rows:
        if row.chart_derivability == UNRESOLVED:
            findings.append({"case_id": row.case_id, "severity": "BLOCK",
                             "finding": "chart derivability is unresolved"})
        if row.adjudication == ADJUDICATION_UNRESOLVED:
            findings.append({"case_id": row.case_id, "severity": "BLOCK",
                             "finding": "registry/chart disagreement is not adjudicated"})
        if row.adjudication == KEY_WRONG and not row.adjudication_rationale.strip():
            findings.append({"case_id": row.case_id, "severity": "BLOCK",
                             "finding": "key_wrong adjudication needs registrar rationale"})
        if row.usable_for_repair and not any(e.stance == "supports" for e in row.gold_evidence):
            findings.append({"case_id": row.case_id, "severity": "REVIEW",
                             "finding": "repair-eligible case has no supporting gold evidence"})
        for evidence in row.gold_evidence:
            if not evidence.quote and evidence.start is None:
                findings.append({
                    "case_id": row.case_id, "severity": "REVIEW",
                    "finding": f"gold evidence {evidence.note_id!r} has no quote or span offsets",
                })
        if not row.registry_source_version:
            findings.append({"case_id": row.case_id, "severity": "REVIEW",
                             "finding": "registry source version is absent"})
    return {"schema": "acr.gold_audit/1", "summary": gold_document(rows)["summary"],
            "n_findings": len(findings), "findings": findings,
            "repair_ready": not any(f["severity"] == "BLOCK" for f in findings)}

def _rule_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    att = manifest.get("rule_attribution") or {}
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            for token in re.findall(
                    r"\b(?:decision_rule|conflict_rule|boundary_case|answer_check)\.[A-Za-z0-9_.-]+",
                    value):
                found.add(token)
        elif isinstance(value, Mapping):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(att)
    return tuple(sorted(found))

def _evidence_refs(manifest: Mapping[str, Any]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    answer = manifest.get("answer") or {}
    rows = manifest.get("evidence") or answer.get("evidence") or []
    out = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        note_id = str(row.get("note_id") or row.get("document_id") or "")
        if not note_id:
            continue
        fields = tuple(sorted(str(x) for x in (row.get("fields") or ())))
        out.add((note_id, str(row.get("stance") or "supports"), fields))
    return tuple(sorted(out))

def _field_results(manifest: Mapping[str, Any]) -> dict[str, dict]:
    answer = manifest.get("answer") or {}
    explicit = answer.get("field_results")
    if isinstance(explicit, Mapping):
        return {
            str(name): {
                "status": str((row or {}).get("status") or ""),
                "value": (row or {}).get("value"),
                "reason_code": str((row or {}).get("reason_code") or ""),
            }
            for name, row in explicit.items()
        }
    status = str(answer.get("status") or "")
    return {str(name): {"status": status, "value": value, "reason_code": ""}
            for name, value in (answer.get("value") or {}).items()}

@dataclass(frozen=True)
class BehaviorSignature:
    """The behaviour of one deepagents run, excluding free-form reasoning."""

    case_id: str
    spec_id: str
    spec_hash: str
    field_results: Mapping[str, Mapping[str, Any]]
    evidence_refs: tuple[tuple[str, str, tuple[str, ...]], ...]
    rules_applied: tuple[str, ...]
    searches: tuple[str, ...]
    gate_validated: bool
    open_threads: tuple[str, ...]
    termination: str
    degradation: Mapping[str, int]
    source: str = ""
    run_id: str = ""
    entity_anchor: Mapping[str, Any] = field(default_factory=dict)
    temporal_anchor: Mapping[str, Any] = field(default_factory=dict)
    coverage_result: Mapping[str, Any] = field(default_factory=dict)
    proof_obligations: tuple[str, ...] = ()
    answer_status: str = ""
    run_conditions: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], *, source: str = "",
                      case_id: str | None = None) -> BehaviorSignature:
        threads = manifest.get("open_threads") or {}
        raw_open = threads.get("open") or threads.get("unresolved") or ()
        if isinstance(raw_open, int):
            open_threads = tuple(f"unresolved:{i}" for i in range(raw_open))
        else:
            open_threads = tuple(sorted(
                str((x or {}).get("thread_id") or (x or {}).get("id") or x)
                if isinstance(x, Mapping) else str(x) for x in raw_open))
        answer = manifest.get("answer") or {}
        coverage = (
            manifest.get("coverage_attested") or answer.get("coverage_attested") or {}
        )
        trace_searches = []
        trace_path = manifest.get("trace")
        if source and trace_path:
            tp = Path(str(trace_path))
            if not tp.is_absolute():
                tp = Path(source).parent / tp
            if tp.is_file():
                try:
                    for line in tp.read_text(encoding="utf-8").splitlines():
                        ev = json.loads(line)
                        if ev.get("kind") == "tool" and str(ev.get("tool") or "").endswith(
                                ("search", "search_notes")):
                            # FLATTENED — `search_notes` is batched, so a list-valued query is
                            # several terms and `str(q)` made them one opaque token. This was the
                            # THIRD independent implementation of "which terms did this run
                            # search"; `evals._query_terms` is the one that decides.
                            trace_searches.extend(
                                _query_terms((ev.get("args") or {}).get("query")))
                except (OSError, json.JSONDecodeError):
                    trace_searches = []
        searches = tuple(sorted({
            str(x) for x in (coverage.get("searched_terms") or ()) if str(x).strip()
        } | set(trace_searches)))
        values = answer.get("value") if isinstance(answer.get("value"), Mapping) else {}
        explicit_entity = answer.get("entity_anchor")
        explicit_temporal = answer.get("temporal_anchor")
        entity_anchor = (
            dict(explicit_entity) if isinstance(explicit_entity, Mapping)
            else {
                str(k): v for k, v in values.items()
                if any(token in str(k).lower()
                       for token in ("entity", "tumor", "tumour", "lesion", "primary_site",
                                     "origin", "specimen"))
            }
        )
        temporal_anchor = (
            dict(explicit_temporal) if isinstance(explicit_temporal, Mapping)
            else {
                str(k): v for k, v in values.items()
                if any(token in str(k).lower()
                       for token in ("date", "time", "diagnosis", "recurrence", "persistent"))
            }
        )
        coverage_result = answer.get("coverage_attested")
        if not isinstance(coverage_result, Mapping):
            coverage_result = manifest.get("coverage_attested")
        if not isinstance(coverage_result, Mapping):
            coverage_result = {}
        obligations = list(open_threads)
        obligations += [
            f"coverage_unreachable:{x}" for x in (manifest.get("coverage_unreachable") or ())
        ]
        termination = str(
            manifest.get("spend_stopped") or manifest.get("expansion_stopped")
            or answer.get("negative_basis") or manifest.get("termination") or "")
        degradation = {str(k): int(v) for k, v in (manifest.get("degradation") or {}).items()
                       if isinstance(v, (int, bool))}
        return cls(
            case_id=_safe_case_id(
                case_id if case_id is not None
                else manifest.get("patient_id") or manifest.get("case_id")),
            spec_id=str(manifest.get("spec_id") or ""),
            spec_hash=str(manifest.get("spec_hash") or ""),
            field_results=_field_results(manifest),
            evidence_refs=_evidence_refs(manifest),
            rules_applied=_rule_ids(manifest),
            searches=searches,
            gate_validated=bool(manifest.get("gate_validated")),
            open_threads=open_threads,
            termination=termination,
            degradation=degradation,
            source=source,
            run_id=str(manifest.get("run_id") or (Path(source).stem if source else "")),
            entity_anchor=entity_anchor,
            temporal_anchor=temporal_anchor,
            coverage_result=dict(coverage_result),
            proof_obligations=tuple(sorted(dict.fromkeys(obligations))),
            answer_status=str(answer.get("status") or ""),
            run_conditions={
                "model": str(manifest.get("model") or ""),
                "model_temperature": manifest.get("model_temperature"),
                "max_model_calls": manifest.get("max_model_calls"),
                "max_usd": (manifest.get("spend") or {}).get("max_usd"),
                "sample_seed": manifest.get("sample_seed"),
            },
        )

    @classmethod
    def load(cls, path: str | Path, *, case_id: str | None = None) -> BehaviorSignature:
        p = Path(path)
        return cls.from_manifest(json.loads(p.read_text(encoding="utf-8")), source=str(p),
                                 case_id=case_id)

    @property
    def behaviour(self) -> dict:
        return {
            "field_results": _normalise(self.field_results),
            "evidence_refs": self.evidence_refs,
            "rules_applied": self.rules_applied,
            "searches": tuple(_normalise(x) for x in self.searches),
            "entity_anchor": _normalise(self.entity_anchor),
            "temporal_anchor": _normalise(self.temporal_anchor),
            "coverage_result": _normalise(self.coverage_result),
            "answer_status": self.answer_status,
            "gate_validated": self.gate_validated,
            "proof_obligations": self.proof_obligations,
            "termination": self.termination,
        }

    @property
    def signature_hash(self) -> str:
        return _hash(self.spec_id, self.behaviour)

    @property
    def is_grounded(self) -> bool:
        found = any(str(r.get("status") or "") == FOUND for r in self.field_results.values())
        return bool(self.gate_validated and not self.proof_obligations
                    and (not found or self.evidence_refs)
                    and not any(self.degradation.values()))

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "spec_id": self.spec_id, "spec_hash": self.spec_hash,
            "signature_hash": self.signature_hash,
            "field_results": {k: dict(v) for k, v in self.field_results.items()},
            "evidence_refs": [{"note_id": n, "stance": s, "fields": list(f)}
                              for n, s, f in self.evidence_refs],
            "rules_applied": list(self.rules_applied), "searches": list(self.searches),
            "gate_validated": self.gate_validated, "open_threads": list(self.open_threads),
            "termination": self.termination, "degradation": dict(self.degradation),
            "source": _portable_source(self.source), "run_id": _portable_run_id(self.run_id),
            "entity_anchor": dict(self.entity_anchor),
            "temporal_anchor": dict(self.temporal_anchor),
            "coverage_result": dict(self.coverage_result),
            "proof_obligations": list(self.proof_obligations),
            "answer_status": self.answer_status,
            "run_conditions": dict(self.run_conditions),
        }
