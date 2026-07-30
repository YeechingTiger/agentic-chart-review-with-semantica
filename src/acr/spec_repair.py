"""Develop-plane specification repair from chart-observable gold and recorded trajectories.

This module never runs an agent and never changes a specification.  It turns already-recorded
deepagents manifests into behavioural clusters, compares those clusters with adjudicated
chart-observable truth, and produces the contrastive packet from which a narrowly scoped spec
edit can be proposed.  Keeping those operations deterministic is the boundary: the answer key
may influence a proposal in DEVELOP, but it is never reachable from the RUN plane.

The analogy to program repair is intentionally exact but limited.  A trajectory is grouped by
what it *did* (answer, evidence, rules and coverage), not by its prose reasoning.  A cluster is
selected only when it both matches the chart-observable answer and earned the existing gate.
Agreement alone is not correctness, and a registry value that cannot be established from the
available chart is not an instruction to make the agent guess.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOLD_SCHEMA = "acr.chart_observable_gold/1"
BEHAVIOUR_SCHEMA = "acr.behavior_distribution/1"
PACKET_SCHEMA = "acr.contrastive_failure_packet/1"
PROPOSAL_SCHEMA = "acr.spec_patch_proposal/1"
VALIDATION_SCHEMA = "acr.paired_validation/1"

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

RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
SPEC_AMBIGUITY = "SPEC_AMBIGUITY"
NO_CORRECT_BEHAVIOUR = "NO_CORRECT_BEHAVIOUR"
GOLD_NOT_CHART_OBSERVABLE = "GOLD_NOT_CHART_OBSERVABLE"
GOLD_UNRESOLVED = "GOLD_UNRESOLVED"

SEMANTIC = "semantic"
ASSET = "asset"
CHANGE_CLASSES = (SEMANTIC, ASSET)

PARAMETERS = (
    "evidence_eligibility",
    "precedence_conflict_rule",
    "temporal_scope",
    "entity_association",
    "abstention_boundary",
    "document_type_policy",
    "keyword_retrieval_asset",
    "skill_instruction",
    "deterministic_answer_check",
)
RETRIEVAL_PARAMETERS = {
    "document_type_policy", "keyword_retrieval_asset", "skill_instruction",
}

_PERSON_ID = re.compile(r"1168\d{12}")


class SpecRepairError(ValueError):
    """A develop-plane artifact is incomplete, inconsistent, or unsafe to use."""


class GoldNotUsable(SpecRepairError):
    """The supplied registry label has not become chart-observable gold."""


class InvalidProposal(SpecRepairError):
    """A proposed edit violates the gradient-routing boundary."""


class SealedSetReuse(SpecRepairError):
    """A sealed cohort was used after its one permitted certification read."""


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
    if _PERSON_ID.search(case_id):
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


def _portable_source(source: str) -> str:
    """Name a controlled input without copying a path that may contain a person id."""
    return f"manifest:{_hash(source)}" if source else ""


def _portable_run_id(run_id: str) -> str:
    return f"run:{_hash(run_id)}" if _PERSON_ID.search(run_id) else run_id


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
                            q = (ev.get("args") or {}).get("query")
                            if q is not None:
                                trace_searches.append(str(q))
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


def matches_gold(signature: BehaviorSignature, gold: ChartObservableGold) -> bool:
    if not gold.usable_for_repair or signature.spec_id != gold.spec_id:
        return False
    for field_name, expected in gold.chart_answer.items():
        actual = signature.field_results.get(field_name)
        actual_status = (
            str(actual.get("status") or "") if actual is not None
            else signature.answer_status
        )
        if actual_status != expected.status:
            return False
        if (expected.status == FOUND
                and (actual is None
                     or _normalise(actual.get("value")) != _normalise(expected.value))):
            return False
    return True


def overclaims(signature: BehaviorSignature, gold: ChartObservableGold) -> bool:
    if not gold.usable_for_repair:
        return False
    for field_name, expected in gold.chart_answer.items():
        actual = signature.field_results.get(field_name) or {}
        actual_status = str(actual.get("status") or signature.answer_status)
        if expected.status != FOUND and actual_status == FOUND:
            return True
    return False


@dataclass(frozen=True)
class BehaviorCluster:
    signature_hash: str
    count: int
    mass: float
    representative: BehaviorSignature
    sources: tuple[str, ...]
    run_conditions: tuple[Mapping[str, Any], ...]
    gold_correct: bool | None
    grounded_correct: bool | None
    overclaim: bool | None

    def to_dict(self) -> dict:
        return {
            "signature_hash": self.signature_hash, "count": self.count, "mass": self.mass,
            "representative": self.representative.to_dict(), "sources": list(self.sources),
            "run_conditions": [dict(x) for x in self.run_conditions],
            "gold_correct": self.gold_correct, "grounded_correct": self.grounded_correct,
            "overclaim": self.overclaim,
        }


@dataclass(frozen=True)
class BehaviorDistribution:
    case_id: str
    spec_id: str
    clusters: tuple[BehaviorCluster, ...]
    n_runs: int
    behavioral_entropy: float
    gold_consistency: float | None
    grounded_consistency: float | None
    overclaim_rate: float | None
    gold_usable: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "spec_id": self.spec_id, "n_runs": self.n_runs,
            "behavioral_entropy": self.behavioral_entropy,
            "gold_consistency": self.gold_consistency,
            "grounded_consistency": self.grounded_consistency,
            "overclaim_rate": self.overclaim_rate, "gold_usable": self.gold_usable,
            "clusters": [c.to_dict() for c in self.clusters],
        }


def cluster_behaviors(signatures: Sequence[BehaviorSignature],
                      gold: ChartObservableGold | None = None) -> BehaviorDistribution:
    if not signatures:
        raise SpecRepairError("cannot cluster an empty trajectory set")
    case_ids = {s.case_id for s in signatures}
    spec_ids = {s.spec_id for s in signatures}
    if len(case_ids) != 1 or len(spec_ids) != 1:
        raise SpecRepairError(
            f"one distribution must contain one case/spec, got cases={sorted(case_ids)}, "
            f"specs={sorted(spec_ids)}")
    by: dict[str, list[BehaviorSignature]] = {}
    for s in signatures:
        by.setdefault(s.signature_hash, []).append(s)
    n = len(signatures)
    usable = bool(gold and gold.usable_for_repair)
    clusters = []
    for key, rows in by.items():
        rep = rows[0]
        correct = matches_gold(rep, gold) if usable and gold else None
        grounded = bool(correct and rep.is_grounded) if correct is not None else None
        over = overclaims(rep, gold) if usable and gold else None
        clusters.append(BehaviorCluster(
            signature_hash=key, count=len(rows), mass=len(rows) / n, representative=rep,
            sources=tuple(_portable_source(r.source) for r in rows),
            run_conditions=tuple(dict(r.run_conditions) for r in rows),
            gold_correct=correct,
            grounded_correct=grounded, overclaim=over))
    clusters.sort(key=lambda c: (-c.count, c.signature_hash))
    entropy = -sum(c.mass * math.log2(c.mass) for c in clusters if c.mass)
    return BehaviorDistribution(
        case_id=signatures[0].case_id, spec_id=signatures[0].spec_id,
        clusters=tuple(clusters), n_runs=n, behavioral_entropy=round(entropy, 6),
        gold_consistency=(round(sum(c.mass for c in clusters if c.gold_correct), 6)
                          if usable else None),
        grounded_consistency=(round(sum(c.mass for c in clusters if c.grounded_correct), 6)
                              if usable else None),
        overclaim_rate=(round(sum(c.mass for c in clusters if c.overclaim), 6)
                        if usable else None),
        gold_usable=usable,
    )


def behavior_document(distributions: Iterable[BehaviorDistribution]) -> dict:
    rows = list(distributions)
    return {
        "schema": BEHAVIOUR_SCHEMA, "distributions": [d.to_dict() for d in rows],
        "summary": {
            "n_cases": len(rows), "n_runs": sum(d.n_runs for d in rows),
            "mean_behavioral_entropy": (
                round(sum(d.behavioral_entropy for d in rows) / len(rows), 6) if rows else None),
            "mean_gold_consistency": _mean(
                [d.gold_consistency for d in rows if d.gold_consistency is not None]),
            "mean_grounded_consistency": _mean(
                [d.grounded_consistency for d in rows
                 if d.grounded_consistency is not None]),
            "mean_overclaim_rate": _mean(
                [d.overclaim_rate for d in rows if d.overclaim_rate is not None]),
        },
    }


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def load_signatures(paths: Iterable[str | Path], *,
                    case_map: Mapping[str, str] | None = None) -> list[BehaviorSignature]:
    out = []
    mapping = dict(case_map or {})
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("*.manifest.json"))
            if not found:
                raise SpecRepairError(f"no *.manifest.json under {p}")
            for x in found:
                doc = json.loads(x.read_text(encoding="utf-8"))
                original = str(doc.get("patient_id") or doc.get("case_id") or "")
                out.append(BehaviorSignature.from_manifest(
                    doc, source=str(x), case_id=mapping.get(original)))
        else:
            doc = json.loads(p.read_text(encoding="utf-8"))
            original = str(doc.get("patient_id") or doc.get("case_id") or "")
            out.append(BehaviorSignature.from_manifest(
                doc, source=str(p), case_id=mapping.get(original)))
    return out


def _difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    out = {}
    for key in sorted(set(a) | set(b)):
        av, bv = a.get(key), b.get(key)
        if _normalise(av) != _normalise(bv):
            out[key] = {"selected": av, "rejected": bv}
    return out


def _spec_sections(spec: Any) -> dict:
    return {
        "question": str(getattr(spec, "question", "") or ""),
        "decision_rule": list(getattr(spec, "decision_rule", ()) or ()),
        "evidence_rules": dict(getattr(spec, "evidence_rules", {}) or {}),
        "conflict_rules": list(getattr(spec, "conflict_rules", ()) or ()),
        "proof_obligation": getattr(
            getattr(spec, "proof_obligation", None), "model_dump", lambda **_: {})(
                mode="json"),
        "abstention": dict(getattr(spec, "abstention", {}) or {}),
        "boundary_cases": list(getattr(spec, "boundary_cases", ()) or ()),
        "search_hints": list(getattr(spec, "search_hints", ()) or ()),
        "answer_checks": list(getattr(spec, "answer_checks", ()) or ()),
    }


@dataclass(frozen=True)
class ContrastiveFailurePacket:
    case_id: str
    spec_id: str
    spec_hash: str
    disposition: str
    selected: Mapping[str, Any] | None
    rejected: Mapping[str, Any] | None
    differences: Mapping[str, Any]
    gold: Mapping[str, Any]
    spec_sections: Mapping[str, Any]
    repair_permitted: bool
    why: str

    def to_dict(self) -> dict:
        return {
            "schema": PACKET_SCHEMA, "case_id": self.case_id, "spec_id": self.spec_id,
            "spec_hash": self.spec_hash, "disposition": self.disposition,
            "selected": dict(self.selected) if self.selected else None,
            "rejected": dict(self.rejected) if self.rejected else None,
            "differences": dict(self.differences), "gold": dict(self.gold),
            "spec_sections": dict(self.spec_sections),
            "repair_permitted": self.repair_permitted, "why": self.why,
        }


def diagnose(distribution: BehaviorDistribution, gold: ChartObservableGold, spec: Any,
             ) -> ContrastiveFailurePacket:
    if distribution.case_id != gold.case_id or distribution.spec_id != gold.spec_id:
        raise SpecRepairError("distribution and gold identify different case/spec")
    sections = _spec_sections(spec)
    spec_hash = str(getattr(spec, "spec_hash", "") or "")
    if gold.chart_derivability == NOT_DERIVABLE or gold.adjudication == OUTSIDE_CHART:
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, GOLD_NOT_CHART_OBSERVABLE,
            None, None, {}, gold.to_dict(), sections, False,
            "the registry value is outside the available chart; changing the spec would train "
            "the agent to guess")
    if not gold.usable_for_repair:
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, GOLD_UNRESOLVED,
            None, None, {}, gold.to_dict(), sections, False,
            "chart derivability or registry adjudication is unresolved")

    selected = [c for c in distribution.clusters if c.grounded_correct]
    rejected = [c for c in distribution.clusters if not c.grounded_correct]
    sel = max(selected, key=lambda c: c.count, default=None)
    rej = max(rejected, key=lambda c: c.count, default=None)
    if sel and rej:
        missing_gold_notes = {
            e.note_id for e in gold.gold_evidence if e.stance == "supports"
        } - {n for n, _, _ in rej.representative.evidence_refs}
        disposition = RETRIEVAL_FAILURE if missing_gold_notes else SPEC_AMBIGUITY
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, disposition,
            sel.representative.to_dict(), rej.representative.to_dict(),
            _difference(sel.representative.behaviour, rej.representative.behaviour),
            gold.to_dict(), sections, True,
            ("the rejected behaviour did not surface adjudicated supporting evidence"
             if disposition == RETRIEVAL_FAILURE else
             "the same spec induced both a grounded-correct and a rejected interpretation"))
    if not sel:
        witness = [e.to_dict() for e in gold.gold_evidence if e.stance == "supports"]
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, NO_CORRECT_BEHAVIOUR,
            {"source": "gold_evidence", "evidence": witness,
             "chart_answer": {k: v.to_dict() for k, v in gold.chart_answer.items()}}
            if witness else None,
            rej.representative.to_dict() if rej else None,
            {}, gold.to_dict(), sections, bool(witness),
            ("no trajectory was grounded-correct; an adjudicated witness is available"
             if witness else
             "no trajectory was grounded-correct and no gold witness exists; locate evidence "
             "or adjudicate chart derivability before proposing text"))
    return ContrastiveFailurePacket(
        gold.case_id, gold.spec_id, spec_hash, "NO_REPAIR_NEEDED",
        sel.representative.to_dict(), None, {}, gold.to_dict(), sections, False,
        "all observed behaviour is grounded-correct")


@dataclass(frozen=True)
class SpecPatchProposal:
    """A proposed edit to one registered parameter; never an applied edit."""

    case_id: str
    spec_id: str
    failure_class: str
    parameter_id: str
    quoted_current_text: str
    selected_vs_rejected_difference: Mapping[str, Any]
    minimal_patch: str
    expected_behavior_change: str
    change_class: str
    source_basis: str
    cases_addressed: tuple[str, ...]
    blast_radius: Mapping[str, Any]
    requires_clinician_signoff: bool

    def __post_init__(self) -> None:
        _safe_case_id(self.case_id)
        if not self.spec_id.strip():
            raise InvalidProposal("spec_id is required")
        if not self.failure_class.strip():
            raise InvalidProposal("failure_class is required")
        if self.parameter_id not in PARAMETERS:
            raise InvalidProposal(
                f"parameter_id {self.parameter_id!r} is not registered; choose one of {PARAMETERS}")
        if self.change_class not in CHANGE_CLASSES:
            raise InvalidProposal(f"change_class must be one of {CHANGE_CLASSES}")
        if not self.minimal_patch.strip():
            raise InvalidProposal("minimal_patch is empty")
        if not self.expected_behavior_change.strip():
            raise InvalidProposal("expected_behavior_change is required")
        if not self.source_basis.strip():
            raise InvalidProposal("source_basis is required")
        if not self.quoted_current_text.strip():
            raise InvalidProposal("quoted_current_text is required for the citation mask")
        if not self.cases_addressed or self.case_id not in self.cases_addressed:
            raise InvalidProposal("cases_addressed must contain the proposal case_id")
        if not self.blast_radius:
            raise InvalidProposal("blast_radius must state the expected scope or its uncertainty")
        if self.failure_class == RETRIEVAL_FAILURE and self.parameter_id not in RETRIEVAL_PARAMETERS:
            raise InvalidProposal(
                f"a retrieval failure may only change {sorted(RETRIEVAL_PARAMETERS)}, not "
                f"{self.parameter_id}")
        if self.failure_class == RETRIEVAL_FAILURE and self.change_class != ASSET:
            raise InvalidProposal("a retrieval failure may produce only an asset change")
        if self.change_class == SEMANTIC and not self.requires_clinician_signoff:
            raise InvalidProposal("semantic patches always require clinician sign-off")
        if self.change_class == ASSET and self.parameter_id not in RETRIEVAL_PARAMETERS:
            raise InvalidProposal(
                f"asset change cannot target semantic parameter {self.parameter_id}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, spec_text: str = "") -> SpecPatchProposal:
        proposal = cls(
            case_id=_safe_case_id(value.get("case_id")),
            spec_id=str(value.get("spec_id") or ""),
            failure_class=str(value.get("failure_class") or ""),
            parameter_id=str(value.get("parameter_id") or ""),
            quoted_current_text=str(value.get("quoted_current_text") or ""),
            selected_vs_rejected_difference=dict(
                value.get("selected_vs_rejected_difference") or {}),
            minimal_patch=str(value.get("minimal_patch") or ""),
            expected_behavior_change=str(value.get("expected_behavior_change") or ""),
            change_class=str(value.get("change_class") or ""),
            source_basis=str(value.get("source_basis") or ""),
            cases_addressed=tuple(str(x) for x in (value.get("cases_addressed") or ())),
            blast_radius=dict(value.get("blast_radius") or {}),
            requires_clinician_signoff=bool(value.get("requires_clinician_signoff")),
        )
        if spec_text and " ".join(proposal.quoted_current_text.split()) not in \
                " ".join(spec_text.split()):
            raise InvalidProposal(
                "quoted_current_text does not occur verbatim in the current spec")
        return proposal

    def to_dict(self) -> dict:
        return {
            "schema": PROPOSAL_SCHEMA, "case_id": self.case_id, "spec_id": self.spec_id,
            "failure_class": self.failure_class, "parameter_id": self.parameter_id,
            "quoted_current_text": self.quoted_current_text,
            "selected_vs_rejected_difference": dict(self.selected_vs_rejected_difference),
            "minimal_patch": self.minimal_patch,
            "expected_behavior_change": self.expected_behavior_change,
            "change_class": self.change_class, "source_basis": self.source_basis,
            "cases_addressed": list(self.cases_addressed),
            "blast_radius": dict(self.blast_radius),
            "requires_clinician_signoff": self.requires_clinician_signoff,
            # A proposal has not been certified merely because it is syntactically an asset
            # change. Adoption belongs to the held-out asset certification command.
            "may_apply_automatically": False,
            "eligible_for_automatic_adoption_after_certification": (
                self.change_class == ASSET and not self.requires_clinician_signoff),
        }


def validate_proposal_for_packet(
        proposal: SpecPatchProposal,
        packet: ContrastiveFailurePacket) -> SpecPatchProposal:
    """Bind a proposal to the exact failure packet it claims to repair."""
    if not packet.repair_permitted:
        raise InvalidProposal(f"{packet.case_id}: repair is not permitted: {packet.why}")
    expected = (packet.case_id, packet.spec_id, packet.disposition)
    got = (proposal.case_id, proposal.spec_id, proposal.failure_class)
    if got != expected:
        raise InvalidProposal(
            "proposal case/spec/failure does not match its packet: "
            f"expected {expected!r}, got {got!r}")
    return proposal


class ContrastiveSpecRepairer:
    """One-call proposal generator behind an injectable model seam."""

    def __init__(self, proposer: Callable[[Mapping[str, Any]], Mapping[str, Any]]):
        self.proposer = proposer

    def propose(self, packet: ContrastiveFailurePacket, *, spec_text: str) -> SpecPatchProposal:
        if not packet.repair_permitted:
            raise InvalidProposal(f"{packet.case_id}: repair is not permitted: {packet.why}")
        raw = self.proposer(packet.to_dict())
        if not isinstance(raw, Mapping):
            raise InvalidProposal("proposal model returned a non-object")
        return validate_proposal_for_packet(
            SpecPatchProposal.from_dict(raw, spec_text=spec_text), packet)


@dataclass(frozen=True)
class InstancePair:
    case_id: str
    subgroup: tuple[str, ...]
    before_correct: bool
    after_correct: bool
    before_grounded: bool
    after_grounded: bool
    before_overclaim: bool
    after_overclaim: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "subgroups": list(self.subgroup),
            "before_correct": self.before_correct, "after_correct": self.after_correct,
            "before_grounded": self.before_grounded, "after_grounded": self.after_grounded,
            "before_overclaim": self.before_overclaim, "after_overclaim": self.after_overclaim,
        }


@dataclass(frozen=True)
class PairedValidationReport:
    pairs: tuple[InstancePair, ...]
    mean_correct_delta: float
    mean_grounded_delta: float
    overclaim_delta: float
    regressions: tuple[str, ...]
    subgroup_regressions: tuple[Mapping[str, Any], ...]
    accepted: bool
    refusal_reasons: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": VALIDATION_SCHEMA, "n_cases": len(self.pairs),
            "mean_correct_delta": self.mean_correct_delta,
            "mean_grounded_delta": self.mean_grounded_delta,
            "overclaim_delta": self.overclaim_delta,
            "regressions": list(self.regressions),
            "subgroup_regressions": [dict(x) for x in self.subgroup_regressions],
            "accepted": self.accepted, "refusal_reasons": list(self.refusal_reasons),
            "metrics": {arm: dict(values) for arm, values in self.metrics.items()},
            "per_instance": [p.to_dict() for p in self.pairs],
        }


def _best_distribution_value(d: BehaviorDistribution, attr: str) -> bool:
    """Read the modal cluster, with ties resolved against acceptance."""
    best = max(c.count for c in d.clusters)
    tied = [c for c in d.clusters if c.count == best]
    return bool(tied and all(bool(getattr(c, attr)) for c in tied))


def _validation_metrics(
        distributions: Sequence[BehaviorDistribution],
        gold: Mapping[str, ChartObservableGold]) -> dict[str, float | None]:
    """Score probability mass over fields; repeated trajectories remain visible."""
    field_total = status_correct = exact_correct = 0.0
    predicted_found = correct_found = expected_found = 0.0
    expected_abstain = correct_abstain = grounded_found = 0.0
    for distribution in distributions:
        expected = gold[distribution.case_id]
        for cluster in distribution.clusters:
            signature = cluster.representative
            mass = cluster.mass
            for name, target in expected.chart_answer.items():
                actual = signature.field_results.get(name) or {}
                status = str(actual.get("status") or signature.answer_status)
                status_match = status == target.status
                value_match = (
                    target.status != FOUND
                    or _normalise(actual.get("value")) == _normalise(target.value)
                )
                field_total += mass
                status_correct += mass * status_match
                exact_correct += mass * (status_match and value_match)
                if status == FOUND:
                    predicted_found += mass
                    if target.status == FOUND and value_match:
                        correct_found += mass
                    if signature.is_grounded:
                        grounded_found += mass
                if target.status == FOUND:
                    expected_found += mass
                else:
                    expected_abstain += mass
                    correct_abstain += mass * status_match
    n = len(distributions)

    def ratio(num: float, den: float) -> float | None:
        return round(num / den, 6) if den else None

    return {
        "field_exact_accuracy": ratio(exact_correct, field_total),
        "status_accuracy": ratio(status_correct, field_total),
        "found_precision": ratio(correct_found, predicted_found),
        "found_recall": ratio(correct_found, expected_found),
        "abstention_accuracy": ratio(correct_abstain, expected_abstain),
        "evidence_validity": ratio(grounded_found, predicted_found),
        "gate_valid_correct_mass": (
            round(sum(float(d.grounded_consistency or 0) for d in distributions) / n, 6)
            if n else None
        ),
        "critical_overclaim_mass": (
            round(sum(float(d.overclaim_rate or 0) for d in distributions) / n, 6)
            if n else None
        ),
        "mean_behavioral_entropy": (
            round(sum(d.behavioral_entropy for d in distributions) / n, 6)
            if n else None
        ),
    }


def paired_validate(before: Sequence[BehaviorDistribution],
                    after: Sequence[BehaviorDistribution],
                    gold: Mapping[str, ChartObservableGold], *,
                    max_subgroup_drop: float = 0.0,
                    require_positive_mean: bool = True) -> PairedValidationReport:
    """Compare frozen before/after distributions per case and subgroup."""
    bmap, amap = ({d.case_id: d for d in before}, {d.case_id: d for d in after})
    if set(bmap) != set(amap):
        raise SpecRepairError(
            f"paired validation needs identical case ids; before-only={sorted(set(bmap)-set(amap))}, "
            f"after-only={sorted(set(amap)-set(bmap))}")
    for case_id in sorted(bmap):
        before_conditions = Counter(
            _canonical(condition)
            for cluster in bmap[case_id].clusters
            for condition in cluster.run_conditions
        )
        after_conditions = Counter(
            _canonical(condition)
            for cluster in amap[case_id].clusters
            for condition in cluster.run_conditions
        )
        if before_conditions != after_conditions:
            raise SpecRepairError(
                f"{case_id}: paired validation requires the same model, temperature, "
                "max_model_calls, max_usd and preregistered seeds in both arms")
    pairs = []
    for case_id in sorted(bmap):
        if case_id not in gold or not gold[case_id].usable_for_repair:
            raise GoldNotUsable(f"{case_id}: missing usable chart-observable gold")
        pairs.append(InstancePair(
            case_id=case_id, subgroup=gold[case_id].subgroups,
            before_correct=_best_distribution_value(bmap[case_id], "gold_correct"),
            after_correct=_best_distribution_value(amap[case_id], "gold_correct"),
            before_grounded=_best_distribution_value(bmap[case_id], "grounded_correct"),
            after_grounded=_best_distribution_value(amap[case_id], "grounded_correct"),
            before_overclaim=bool(any(c.overclaim for c in bmap[case_id].clusters)),
            after_overclaim=bool(any(c.overclaim for c in amap[case_id].clusters)),
        ))
    n = len(pairs)
    if not n:
        raise SpecRepairError("paired validation has no cases")
    correct_delta = sum(p.after_correct - p.before_correct for p in pairs) / n
    grounded_delta = sum(p.after_grounded - p.before_grounded for p in pairs) / n
    overclaim_delta = sum(p.after_overclaim - p.before_overclaim for p in pairs) / n
    regressions = tuple(p.case_id for p in pairs
                        if (p.before_correct and not p.after_correct)
                        or (p.before_grounded and not p.after_grounded)
                        or (not p.before_overclaim and p.after_overclaim))
    subgroup_rows = []
    groups = sorted({g for p in pairs for g in p.subgroup})
    for group in groups:
        rows = [p for p in pairs if group in p.subgroup]
        delta = sum(p.after_correct - p.before_correct for p in rows) / len(rows)
        if delta < -max_subgroup_drop:
            subgroup_rows.append({"subgroup": group, "n": len(rows), "correct_delta": delta})
    reasons = []
    if regressions:
        reasons.append(f"{len(regressions)} per-instance regression(s)")
    if subgroup_rows:
        reasons.append(f"{len(subgroup_rows)} subgroup regression(s)")
    if overclaim_delta > 0:
        reasons.append("critical overclaim rate increased")
    if require_positive_mean and grounded_delta <= 0:
        reasons.append("grounded-correct rate did not improve")
    before_metrics = _validation_metrics(before, gold)
    after_metrics = _validation_metrics(after, gold)
    deltas = {
        name: (
            round(float(after_metrics[name]) - float(before_metrics[name]), 6)
            if before_metrics[name] is not None and after_metrics[name] is not None else None
        )
        for name in before_metrics
    }
    return PairedValidationReport(
        pairs=tuple(pairs), mean_correct_delta=round(correct_delta, 6),
        mean_grounded_delta=round(grounded_delta, 6),
        overclaim_delta=round(overclaim_delta, 6), regressions=regressions,
        subgroup_regressions=tuple(subgroup_rows), accepted=not reasons,
        refusal_reasons=tuple(reasons),
        metrics={"before": before_metrics, "after": after_metrics, "delta": deltas})


def min_zero_error_n(max_error_rate: float, confidence: float = 0.95) -> int:
    """Exact zero-event binomial sample size: (1-p)^n <= 1-confidence."""
    if not 0 < max_error_rate < 1:
        raise SpecRepairError("max_error_rate must be strictly between 0 and 1")
    if not 0 < confidence < 1:
        raise SpecRepairError("confidence must be strictly between 0 and 1")
    return math.ceil(math.log(1 - confidence) / math.log(1 - max_error_rate))


@dataclass(frozen=True)
class SealedCertification:
    """Write-once identity for the one permitted read of a sealed cohort."""

    cohort_hash: str
    bundle_hash: str
    consumed: bool = False
    result_hash: str = ""

    def consume(self, result: Mapping[str, Any]) -> SealedCertification:
        if self.consumed:
            raise SealedSetReuse(
                f"sealed cohort {self.cohort_hash} was already consumed for bundle "
                f"{self.bundle_hash}; mint a new sealed cohort")
        return SealedCertification(self.cohort_hash, self.bundle_hash, True, _hash(result))

    def to_dict(self) -> dict:
        return {"schema": "acr.sealed_certification/1", "cohort_hash": self.cohort_hash,
                "bundle_hash": self.bundle_hash, "consumed": self.consumed,
                "result_hash": self.result_hash}
