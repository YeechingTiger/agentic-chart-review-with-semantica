"""Meta-evaluation over recorded attributions: the library, its clusters, and the calibration.

SEPARATED FROM `attribution.py` ON 2026-08-06, and the boundary is the plane rather than the topic.
`attribution.py` is one agent explaining one run. Everything here reads MANY finished attributions
and the human decisions taken on them — an accountable ledger, deterministic clusters over it, and
`meta_evaluate_attributions`, which asks whether the attributor agrees with the humans often enough
to be trusted. None of it calls a model or opens a chart.

They lived in one 2,180-line module because they share the report schema. Sharing a type is not
sharing a job: the agent is the instrument, this is the calibration of the instrument, and a reader
asking "does the attributor work" should not have to read the attributor to find out.

Persistence is `LocalArtifactStore`'s and is append-only JSONL. Nothing here writes a repository file.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contract.behaviour import safe_case_id
from ..core.local_artifacts import LocalArtifactStore
from .attribution import (
    CAUSES,
    LIFECYCLE,
    AttributionError,
    AttributionReport,
    EvidenceRef,
    _event_id,
    _now,
)


# ==================================================================== append-only case store
@dataclass(frozen=True)
class ErrorCaseEvent:
    case_id: str
    event: str
    lifecycle: str
    run_ref: Mapping[str, Any]
    reasons: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        safe_case_id(self.case_id)
        if self.lifecycle not in LIFECYCLE:
            raise AttributionError(f"lifecycle must be one of {LIFECYCLE}")

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "event": self.event, "lifecycle": self.lifecycle,
            "run_ref": dict(self.run_ref), "reasons": list(self.reasons),
            "detail": dict(self.detail), "created_at": self.created_at,
        }

@dataclass(frozen=True)
class AdjudicationEvent:
    """A human's accountable decision about one case.

    `primary_cause` IS THE HUMAN ROOT-CAUSE LABEL, and it is the only reason
    `meta_evaluate_attributions` can pair anything. It was absent: this class emitted `decision`
    (a LIFECYCLE state — what to DO about the case) and nothing from `CAUSES` (what went wrong),
    and the two sets are disjoint. So the calibration read zero pairs for every input, reported
    "need at least 30 adjudicated cases", and the repo shipped that count into its docs as an
    explanation — diagnosing a data shortage where the defect was a format mismatch.

    OPTIONAL, because not every adjudication is a root-cause label: `WONT_FIX` and
    `OUTSIDE_CHART` are decisions about what to do, and requiring a cause would make them
    unrecordable. Validated against `CAUSES` when present, because a free-text cause pairs with
    nothing — macro-F1 over labels only one side of the comparison uses is zero.
    """

    case_id: str
    decision: str
    actor: str
    actor_role: str
    rationale: str
    primary_cause: str = ""
    evidence: tuple[EvidenceRef, ...] = ()
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        safe_case_id(self.case_id)
        if self.decision not in LIFECYCLE:
            raise AttributionError(f"adjudication decision must be one of {LIFECYCLE}")
        if self.actor_role not in ("registrar", "clinician", "engineer"):
            raise AttributionError("actor_role must be registrar, clinician, or engineer")
        if not self.actor.strip() or not self.rationale.strip():
            raise AttributionError("adjudication needs actor and rationale")
        if self.primary_cause and self.primary_cause not in CAUSES:
            raise AttributionError(
                f"primary_cause {self.primary_cause!r} is not one the attributor can emit; "
                f"expected one of {CAUSES}. A label only the human side uses pairs with nothing, "
                f"and macro-F1 over it is zero.")

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "decision": self.decision, "actor": self.actor,
            "actor_role": self.actor_role, "rationale": self.rationale,
            "primary_cause": self.primary_cause,
            "evidence": [x.to_dict() for x in self.evidence], "created_at": self.created_at,
        }

@dataclass(frozen=True)
class ErrorCluster:
    cluster_id: str
    signature: Mapping[str, Any]
    case_ids: tuple[str, ...]
    primary_cause: str
    contributing_tags: tuple[str, ...] = ()
    label: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id, "signature": dict(self.signature),
            "case_ids": list(self.case_ids), "n_cases": len(self.case_ids),
            "primary_cause": self.primary_cause,
            "contributing_tags": list(self.contributing_tags),
            "label": self.label, "summary": self.summary,
        }

class ErrorCaseLibrary:
    """Four append-only JSONL ledgers below one local-only directory."""

    def __init__(self, store: LocalArtifactStore, library_id: str):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(library_id).strip())
        if not safe:
            raise AttributionError("library_id is required")
        self.store = store
        self.relative = Path("error-cases") / safe
        self.directory = store.directory(self.relative)

    def _append(self, filename: str, event: Mapping[str, Any], *identity: Any) -> bool:
        eid = _event_id(filename, *identity)
        return self.store.append_jsonl(self.relative / filename, event,
                                       idempotency_key=eid)

    def add_case(self, event: ErrorCaseEvent) -> bool:
        return self._append(
            "cases.jsonl", event.to_dict(), event.case_id, event.event,
            event.run_ref.get("sha256"), event.reasons, event.lifecycle,
        )

    def add_attribution(self, report: AttributionReport, *,
                        manifest_sha256: str) -> bool:
        row = report.to_dict()
        row["manifest_sha256"] = manifest_sha256
        return self._append(
            "attributions.jsonl", row, report.case_id,
            manifest_sha256, row,
        )

    def add_adjudication(self, event: AdjudicationEvent) -> bool:
        return self._append(
            "adjudications.jsonl", event.to_dict(), event.case_id, event.decision,
            event.actor, event.rationale, event.created_at,
        )

    def add_cluster(self, cluster: ErrorCluster) -> bool:
        return self._append(
            "clusters.jsonl", cluster.to_dict(), cluster.cluster_id, cluster.case_ids,
            cluster.signature,
        )

    def rows(self, filename: str) -> list[dict]:
        path = self.store.path(self.relative / filename, what=filename)
        if not path.exists():
            return []
        out = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AttributionError(f"{path}:{index}: truncated or invalid JSONL: {exc}") from exc
        return out

    def current_cases(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for row in self.rows("cases.jsonl"):
            out[str(row["case_id"])] = row
        for row in self.rows("attributions.jsonl"):
            prior = out.setdefault(str(row["case_id"]), {"case_id": row["case_id"]})
            prior["attribution"] = row
            prior["lifecycle"] = row.get("lifecycle", prior.get("lifecycle"))
        for row in self.rows("adjudications.jsonl"):
            prior = out.setdefault(str(row["case_id"]), {"case_id": row["case_id"]})
            prior["adjudication"] = row
            prior["lifecycle"] = row["decision"]
        return out

def cluster_reports(reports: Iterable[AttributionReport]) -> list[ErrorCluster]:
    """Group by deterministic structure; prose is not part of cluster identity."""
    grouped: dict[str, list[AttributionReport]] = defaultdict(list)
    signatures: dict[str, dict] = {}
    for report in reports:
        primary = report.primary_cause
        rule_ids = sorted({
            ref.ref for ref in primary.evidence if ref.kind == "spec_rule"
        })
        detector_ids = sorted({
            ref.ref for ref in primary.evidence if ref.kind == "detector"
        })
        signature = {
            "field": primary.field,
            "primary_cause": primary.cause,
            "parameter_id": primary.parameter_id,
            "rule_ids": rule_ids,
            "detector_ids": detector_ids,
            "termination_class": (
                "CONFIRMATION_CONFLICT" if report.confirmation_new_conflict
                else "UNRESOLVED" if primary.cause == "UNRESOLVED"
                else "OBLIGATIONS_CLOSED"
            ),
            "mode": report.mode,
        }
        key = _event_id(signature)[:16]
        signatures[key] = signature
        grouped[key].append(report)
    out = []
    for key, rows in sorted(grouped.items()):
        tags = sorted({
            cause.cause for report in rows for cause in report.contributing_causes
        })
        out.append(ErrorCluster(
            cluster_id=f"EC-{key}", signature=signatures[key],
            case_ids=tuple(sorted({r.case_id for r in rows})),
            primary_cause=rows[0].primary_cause.cause,
            contributing_tags=tuple(tags),
            label=rows[0].primary_cause.cause.replace("_", " ").title(),
            summary=(
                f"{len({r.case_id for r in rows})} case(s) share the same structured "
                f"{rows[0].primary_cause.cause} attribution signature."
            ),
        ))
    return out

def summarize_library(library: ErrorCaseLibrary) -> dict:
    reports = [
        AttributionReport.from_dict(row)
        for row in library.rows("attributions.jsonl")
    ]
    cases = library.current_cases()
    signals: dict[str, set[str]] = defaultdict(set)
    for case_id, row in cases.items():
        for reason in row.get("reasons") or ():
            signals[str(reason)].add(case_id)
    return {
        "schema": "acr.error_library_summary/1",
        "library": str(library.directory),
        "n_cases": len(cases),
        "n_attributions": len(reports),
        "lifecycle": dict(Counter(str(row.get("lifecycle") or "OPEN")
                                  for row in cases.values())),
        "primary_causes": dict(Counter(r.primary_cause.cause for r in reports)),
        "signal_clusters": [
            {"signal": signal, "case_ids": sorted(case_ids), "n_cases": len(case_ids)}
            for signal, case_ids in sorted(signals.items())
        ],
        "clusters": [c.to_dict() for c in cluster_reports(reports)],
    }

def meta_evaluate_attributions(
    predictions: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    *,
    min_cases: int = 30,
    min_macro_f1: float = 0.80,
) -> dict[str, Any]:
    """Calibrate causal attribution against accountable human root-cause labels."""
    if min_cases < 1 or not 0 <= min_macro_f1 <= 1:
        raise ValueError("min_cases must be >=1 and min_macro_f1 must be in [0,1]")
    # THE LEDGER IS APPEND-ONLY, so one case legitimately has several rows: `LIFECYCLE` includes
    # `REOPENED`, and `ErrorCaseLibrary.add_adjudication` never rewrites an event. A dict
    # comprehension over it made the LAST row win, so a follow-up `VALIDATED_FIXED` — which
    # correctly carries no cause, being a decision about what to DO — silently erased the root-cause
    # label recorded earlier, dropped the pair, AND counted the case as missing a cause. The fold
    # keeps the LATEST row that actually names one: a registrar who changes their mind appends a new
    # row with a cause, and that is the one that counts.
    gold: dict[str, str] = {}
    cases_seen: set[str] = set()
    for row in adjudications:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        cases_seen.add(case_id)
        cause = str(row.get("primary_cause")
                    or (row.get("adjudication") or {}).get("primary_cause") or "")
        if cause:
            gold[case_id] = cause
    # ROWS THAT EXIST AND CARRY NO CAUSE are a FORMAT problem, and reporting them as
    # `n_adjudicated_pairs: 0` beside "need at least N cases" reads as a data shortage — which is
    # exactly how this defect survived: the repo shipped "there are 2 records" into its docs as the
    # explanation. Counted per CASE, not per ROW: three causeless rows for one case are one case
    # that pairs with nothing, and counting rows made a complete ledger look three times as broken.
    n_rows = len(cases_seen)
    n_without_cause = len(cases_seen - set(gold))
    pairs: list[tuple[str, str]] = []
    citation_invalid = 0
    clinical_auto_confirmed = 0
    scope_violations = 0
    # `scope_violations` is read from a key NO PRODUCER WRITES: `AttributionReport.to_dict()` emits
    # `gate_rejections` and never `scope_violations`, so the condition "patient-scope violations
    # must be zero" could never appear in `reasons_not_certified` — a precondition that cannot fail
    # printing as one that passed. Whether anything measured it is now part of the report.
    scope_measured = any("scope_violations" in row for row in predictions)
    clinical = {
        "REFERENCE_OR_GOLD",
        "RETRIEVAL",
        "EVIDENCE_INTERPRETATION",
        "ENTITY_ASSOCIATION",
        "TEMPORAL_SCOPE",
        "SPEC_FORM",
        "SPEC_CONTENT",
        "EVIDENCE_GAP",
    }
    for row in predictions:
        case_id = str(row.get("case_id") or "")
        primary = row.get("primary_cause") or {}
        predicted = str(primary.get("cause") or "")
        if gold.get(case_id):
            pairs.append((predicted, gold[case_id]))
        if row.get("citation_valid") is False or row.get("gate_rejections"):
            citation_invalid += 1
        if predicted in clinical and str(primary.get("status") or "") == "CONFIRMED":
            clinical_auto_confirmed += 1
        scope_violations += int(row.get("scope_violations") or 0)

    labels = sorted({label for pair in pairs for label in pair})
    per_label = {}
    f1s = []
    for label in labels:
        tp = sum(predicted == label and expected == label for predicted, expected in pairs)
        fp = sum(predicted == label and expected != label for predicted, expected in pairs)
        fn = sum(predicted != label and expected == label for predicted, expected in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1s.append(f1)
        per_label[label] = {
            "support": sum(expected == label for _, expected in pairs),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    macro_f1 = sum(f1s) / len(f1s) if f1s else None
    sufficient = len(pairs) >= min_cases
    certified = (
        sufficient
        and macro_f1 is not None
        and macro_f1 >= min_macro_f1
        and citation_invalid == 0
        and clinical_auto_confirmed == 0
        and scope_violations == 0
        and n_without_cause == 0
    )
    return {
        "schema": "acr.attribution_meta_evaluation/1",
        "status": "CERTIFIED_SCREEN" if certified else "EXPERIMENTAL_SCREEN",
        "n_adjudicated_pairs": len(pairs),
        "min_cases": min_cases,
        "macro_f1": round(macro_f1, 6) if macro_f1 is not None else None,
        "min_macro_f1": min_macro_f1,
        "citation_invalid": citation_invalid,
        "clinical_auto_confirmed": clinical_auto_confirmed,
        "scope_violations": scope_violations,
        #: False means NOBODY LOOKED, which is not the same as zero violations.
        "scope_violations_measured": scope_measured,
        "n_adjudications_without_cause": n_without_cause,
        "per_label": per_label,
        "reasons_not_certified": [
            reason
            for condition, reason in (
                (not sufficient, f"need at least {min_cases} adjudicated cases"),
                (
                    macro_f1 is None or macro_f1 < min_macro_f1,
                    f"macro-F1 must be at least {min_macro_f1}",
                ),
                (citation_invalid > 0, "all citations must validate"),
                (
                    clinical_auto_confirmed > 0,
                    "agent may not auto-confirm clinical/semantic causes",
                ),
                (scope_violations > 0, "patient-scope violations must be zero"),
                (n_without_cause > 0,
                 f"{n_without_cause} of {n_rows} adjudicated case(s) carry no readable "
                 f"`primary_cause`, so they pair with nothing. This is a FORMAT problem, not a "
                 f"shortage of cases: record the human root cause with "
                 f"`acr attribute adjudicate --primary-cause <one of CAUSES>`."),
            )
            if condition
        ],
    }
