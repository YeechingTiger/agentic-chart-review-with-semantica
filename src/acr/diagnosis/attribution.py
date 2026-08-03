"""Offline error attribution over a completed chart-review trajectory.

This is not a second extractor and it is not an answer-key judge.  It explains a recorded run,
opens the same patient's chart only to discriminate named rival causes, and emits a structured
report whose certainty is capped by its evidence class.  In particular:

* adjudicated chart-observable gold may support a contrastive diagnosis;
* an unresolved registry value is a disagreement signal, never truth;
* without truth, anomalies remain hypotheses and cannot become semantic spec edits.

The module never writes a repository file.  Persistence is supplied by
``LocalArtifactStore`` and is append-only JSONL.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contract.behaviour import ChartObservableGold, safe_case_id
from ..contract.trace import rule_catalog
from ..core.local_artifacts import LocalArtifactStore, content_hash
from ..evaluation import evals

ATTRIBUTION_SCHEMA = "acr.attribution_report/2"
PACKET_SCHEMA = "acr.attribution_packet/1"
REGISTRY_REFERENCE_SCHEMA = "acr.registry_reference/1"

GOLD = "GOLD"
REGISTRY_REFERENCE = "REGISTRY_REFERENCE"
BLIND = "BLIND"
ATTRIBUTION_MODES = (GOLD, REGISTRY_REFERENCE, BLIND)

CAUSES = (
    "REFERENCE_OR_GOLD",
    "RETRIEVAL",
    "EVIDENCE_INTERPRETATION",
    "ENTITY_ASSOCIATION",
    "TEMPORAL_SCOPE",
    "SPEC_FORM",
    "SPEC_CONTENT",
    "SKILL_OR_PROMPT",
    "ANSWER_CHECK_OR_GATE",
    "RUNTIME_OR_PROVIDER",
    "EVIDENCE_GAP",
    "UNRESOLVED",
)
EVIDENCE_CLASSES = (
    "DETERMINISTIC", "GOLD_CONDITIONAL", "JUDGED", "HUMAN_ADJUDICATED",
)
CERTAINTY = ("CONFIRMED", "LIKELY", "POSSIBLE", "UNRESOLVED")
RELATIONS_TO_TARGET = ("EXPLAINS", "CONTRIBUTES", "UNRELATED_DEFECT", "UNKNOWN")
CAUSAL_STRENGTH = (
    "OBSERVED", "PLAUSIBLE", "COUNTERFACTUAL_SUPPORTED", "HUMAN_CONFIRMED",
)
COUNTERFACTUAL_OUTCOMES = ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "NOT_RUN")
SKEPTIC_VERDICTS = ("PASS", "PASS_WITH_LIMITATIONS", "REVISE", "UNRESOLVED")

PARAMETER_IDS = (
    "evidence_eligibility",
    "precedence_conflict_rule",
    "temporal_scope",
    "entity_association",
    "abstention_boundary",
    "document_type_policy",
    "keyword_retrieval_asset",
    "skill_instruction",
    "deterministic_answer_check",
    "agent_system_prompt",
    "answer_check_rejection_messages",
)

LIFECYCLE = (
    "OPEN", "NEEDS_ADJUDICATION", "ATTRIBUTED", "FIX_CANDIDATE",
    "VALIDATED_FIXED", "OUTSIDE_CHART", "WONT_FIX", "REOPENED",
)

AUTO_CONFIRMABLE_CAUSES = {"ANSWER_CHECK_OR_GATE", "RUNTIME_OR_PROVIDER"}
CLINICAL_CAUSES = {
    "REFERENCE_OR_GOLD", "RETRIEVAL", "EVIDENCE_INTERPRETATION",
    "ENTITY_ASSOCIATION", "TEMPORAL_SCOPE", "SPEC_FORM", "SPEC_CONTENT",
    "EVIDENCE_GAP",
}

KEY_BEARING_NAMES = {
    "answerkey", "answer_key", "expectedoutput", "expected_output", "groundtruth",
    "ground_truth", "gold", "truth", "registryvalue", "registry_value",
}

class AttributionError(ValueError):
    """An attribution packet or report crossed an evidence or authority boundary."""

def _now() -> str:
    return datetime.now(UTC).isoformat()

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)

def _event_id(*parts: Any) -> str:
    return hashlib.sha256("\0".join(_canonical(p) for p in parts).encode()).hexdigest()

def _scan_blind(value: Any, path: str = "packet") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            norm = re.sub(r"[^a-z0-9_]", "", str(key).lower())
            if norm in KEY_BEARING_NAMES:
                raise AttributionError(
                    f"{path}.{key} is key-bearing and forbidden in BLIND attribution")
            _scan_blind(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_blind(child, f"{path}[{index}]")

def _replace(value: Any, old: str, new: str) -> Any:
    if not old:
        return value
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, Mapping):
        return {str(k): _replace(v, old, new) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace(v, old, new) for v in value]
    return value

@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> ArtifactRef:
        resolved = Path(path).resolve()
        return cls(path=str(resolved), sha256=content_hash(resolved))

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class AttributionPacket:
    case_id: str
    spec_id: str
    spec_hash: str
    mode: str
    manifest_ref: ArtifactRef
    trace_ref: ArtifactRef | None
    manifest: Mapping[str, Any]
    trace: tuple[Mapping[str, Any], ...]
    rule_catalogue: tuple[Mapping[str, Any], ...]
    detector_findings: tuple[Mapping[str, Any], ...] = ()
    behavior_signature: Mapping[str, Any] = field(default_factory=dict)
    chart_gold: ChartObservableGold | None = None
    registry_reference: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        safe_case_id(self.case_id)
        if self.mode not in ATTRIBUTION_MODES:
            raise AttributionError(f"mode must be one of {ATTRIBUTION_MODES}")
        if not self.spec_id or not self.spec_hash:
            raise AttributionError("packet needs spec_id and spec_hash")
        if self.mode == GOLD:
            if self.chart_gold is None or not self.chart_gold.usable_for_repair:
                raise AttributionError(
                    "GOLD attribution requires adjudicated chart-observable gold")
            if self.registry_reference is not None:
                raise AttributionError("GOLD packet cannot also carry a registry reference")
        elif self.mode == REGISTRY_REFERENCE:
            if self.registry_reference is None:
                raise AttributionError("REGISTRY_REFERENCE mode requires a local reference")
            if self.chart_gold is not None:
                raise AttributionError(
                    "an unresolved registry reference cannot share a packet with gold")
            adjudication = str(self.registry_reference.get("adjudication") or "")
            if adjudication != "UNRESOLVED":
                raise AttributionError(
                    "registry-reference packet must remain adjudication=UNRESOLVED")
        else:
            if self.chart_gold is not None or self.registry_reference is not None:
                raise AttributionError("BLIND attribution cannot carry truth or registry values")
            _scan_blind(self.manifest, "manifest")
            _scan_blind(self.trace, "trace")
            _scan_blind(self.behavior_signature, "behavior_signature")

    @property
    def semantic_patch_allowed(self) -> bool:
        return self.mode == GOLD

    def prompt_summary(self) -> dict:
        answer = self.manifest.get("answer") or {}
        return {
            "schema": PACKET_SCHEMA,
            "case_id": self.case_id,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "mode": self.mode,
            "answer": answer,
            "gate_validated": bool(self.manifest.get("gate_validated")),
            "termination": self.manifest.get("termination")
            or self.manifest.get("termination_reason"),
            "n_trace_events": len(self.trace),
            "detector_findings": list(self.detector_findings),
            "behavior_signature": dict(self.behavior_signature),
            "truth_boundary": (
                {"chart_answer": {k: v.to_dict()
                                  for k, v in self.chart_gold.chart_answer.items()},
                 "gold_evidence": [e.to_dict() for e in self.chart_gold.gold_evidence]}
                if self.mode == GOLD and self.chart_gold else
                {"registry_disagreement": dict(self.registry_reference or {}),
                 "instruction": "UNRESOLVED reference; do not choose a winner"}
                if self.mode == REGISTRY_REFERENCE else
                {"instruction": "no truth is available; report anomalies and hypotheses only"}
            ),
        }

@dataclass(frozen=True)
class AttributionProbe:
    probe_id: str
    question: str
    alternatives: tuple[str, ...]
    expected_discriminator: str
    confirmation: bool = False
    chart_reads: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id, "question": self.question,
            "alternatives": list(self.alternatives),
            "expected_discriminator": self.expected_discriminator,
            "confirmation": self.confirmation, "chart_reads": list(self.chart_reads),
        }

@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    ref: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (
                "trace", "note", "detector", "spec_rule", "gold_witness", "human",
                "packet", "probe"):
            raise AttributionError(f"unknown evidence reference kind {self.kind!r}")
        if not str(self.ref).strip():
            raise AttributionError("an evidence reference needs a non-empty ref")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceRef:
        raw_kind = str(
            value.get("kind") or value.get("source_type") or value.get("type") or "")
        kind = {
            "trace_event": "trace", "trace_seq": "trace",
            "document": "note", "note_id": "note",
            "chart_note": "note", "rule": "spec_rule", "rule_id": "spec_rule",
            "spec": "spec_rule",
            "detector_id": "detector", "gold_evidence": "gold_witness",
            "human_adjudication": "human",
        }.get(raw_kind, raw_kind)
        ref = value.get("ref") or value.get("source_id") or value.get("id") or ""
        return cls(kind=kind, ref=str(ref),
                   detail=str(value.get("detail") or ""))

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class TargetEvent:
    """The exact outcome or process anomaly this report is trying to explain."""

    event_id: str
    kind: str
    field: str
    observed: Any
    reference_signal: Any
    source: str
    truth_status: str
    question: str

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.kind.strip() or not self.question.strip():
            raise AttributionError("target event needs event_id, kind, and question")
        if self.source not in ("GOLD", "REGISTRY_REFERENCE", "TRACE", "DETECTOR", "MANUAL"):
            raise AttributionError(f"unknown target source {self.source!r}")
        if self.truth_status not in (
                "ADJUDICATED", "UNRESOLVED_REFERENCE", "NO_TRUTH", "DETERMINISTIC"):
            raise AttributionError(f"unknown target truth_status {self.truth_status!r}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetEvent:
        return cls(
            event_id=str(value.get("event_id") or ""),
            kind=str(value.get("kind") or ""),
            field=str(value.get("field") or ""),
            observed=value.get("observed"),
            reference_signal=value.get("reference_signal"),
            source=str(value.get("source") or ""),
            truth_status=str(value.get("truth_status") or ""),
            question=str(value.get("question") or ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class CounterfactualTest:
    """A small intervention or replay used to test a claimed causal mechanism."""

    test_id: str
    target_event_id: str
    kind: str
    intervention: str
    prediction: str
    outcome: str
    observation: str
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.test_id.strip() or not self.target_event_id.strip():
            raise AttributionError("counterfactual test needs test_id and target_event_id")
        if self.outcome not in COUNTERFACTUAL_OUTCOMES:
            raise AttributionError(
                f"counterfactual outcome must be one of {COUNTERFACTUAL_OUTCOMES}")
        if not self.kind.strip() or not self.intervention.strip() or not self.prediction.strip():
            raise AttributionError(
                "counterfactual test needs kind, intervention, and prediction")
        if self.outcome != "NOT_RUN" and not self.observation.strip():
            raise AttributionError("a run counterfactual needs an observation")
        if self.outcome in ("SUPPORTED", "REFUTED") and not self.evidence:
            raise AttributionError(
                f"a {self.outcome} counterfactual needs citation-backed evidence")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CounterfactualTest:
        return cls(
            test_id=str(value.get("test_id") or ""),
            target_event_id=str(value.get("target_event_id") or ""),
            kind=str(value.get("kind") or ""),
            intervention=str(value.get("intervention") or ""),
            prediction=str(value.get("prediction") or ""),
            outcome=str(value.get("outcome") or ""),
            observation=str(value.get("observation") or ""),
            evidence=tuple(
                EvidenceRef.from_dict(row) for row in (value.get("evidence") or ())),
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "evidence": [ref.to_dict() for ref in self.evidence],
        }

@dataclass(frozen=True)
class SkepticReview:
    """Structured opposition to the investigator's proposed root cause."""

    verdict: str
    rationale: str
    objections: tuple[str, ...]
    untested_alternatives: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...] = ()
    reviewer: str = "INVESTIGATOR_SELF_CHALLENGE"

    def __post_init__(self) -> None:
        if self.verdict not in SKEPTIC_VERDICTS:
            raise AttributionError(f"skeptic verdict must be one of {SKEPTIC_VERDICTS}")
        if self.reviewer not in (
                "INVESTIGATOR_SELF_CHALLENGE", "INDEPENDENT_MODEL"):
            raise AttributionError(f"unknown skeptic reviewer {self.reviewer!r}")
        if not self.rationale.strip():
            raise AttributionError("skeptic review needs a rationale")
        if self.verdict in ("PASS", "PASS_WITH_LIMITATIONS") and not self.evidence:
            raise AttributionError("a passing skeptic review needs cited evidence")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SkepticReview:
        return cls(
            verdict=str(value.get("verdict") or ""),
            rationale=str(value.get("rationale") or ""),
            objections=tuple(str(x) for x in (value.get("objections") or ())),
            untested_alternatives=tuple(
                str(x) for x in (value.get("untested_alternatives") or ())),
            evidence=tuple(
                EvidenceRef.from_dict(row) for row in (value.get("evidence") or ())),
            reviewer=str(
                value.get("reviewer") or "INVESTIGATOR_SELF_CHALLENGE"),
        )

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "objections": list(self.objections),
            "untested_alternatives": list(self.untested_alternatives),
            "evidence": [ref.to_dict() for ref in self.evidence],
            "reviewer": self.reviewer,
        }

@dataclass(frozen=True)
class CauseFinding:
    cause: str
    status: str
    evidence_class: str
    rationale: str
    evidence: tuple[EvidenceRef, ...]
    field: str = ""
    parameter_id: str = ""
    route_owner: str = ""
    relation_to_target: str = "UNKNOWN"
    causal_strength: str = "PLAUSIBLE"
    mechanism: str = ""
    counterfactual_prediction: str = ""

    def __post_init__(self) -> None:
        if self.cause not in CAUSES:
            raise AttributionError(f"cause must be one of {CAUSES}")
        if self.status not in CERTAINTY:
            raise AttributionError(f"status must be one of {CERTAINTY}")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise AttributionError(f"evidence_class must be one of {EVIDENCE_CLASSES}")
        if not self.rationale.strip():
            raise AttributionError("cause finding needs a rationale")
        if self.cause != "UNRESOLVED" and not self.evidence:
            raise AttributionError(f"{self.cause} needs at least one evidence reference")
        if self.parameter_id and self.parameter_id not in PARAMETER_IDS:
            raise AttributionError(f"unknown parameter_id {self.parameter_id!r}")
        if self.relation_to_target not in RELATIONS_TO_TARGET:
            raise AttributionError(
                f"relation_to_target must be one of {RELATIONS_TO_TARGET}")
        if self.causal_strength not in CAUSAL_STRENGTH:
            raise AttributionError(f"causal_strength must be one of {CAUSAL_STRENGTH}")
        if (self.relation_to_target == "EXPLAINS"
                and (not self.mechanism.strip()
                     or not self.counterfactual_prediction.strip())):
            raise AttributionError(
                "an EXPLAINS cause needs a mechanism and counterfactual_prediction")
        if self.status == "CONFIRMED":
            human = self.evidence_class == "HUMAN_ADJUDICATED"
            auto = (self.evidence_class == "DETERMINISTIC"
                    and self.cause in AUTO_CONFIRMABLE_CAUSES)
            if not (human or auto):
                raise AttributionError(
                    f"{self.cause} cannot be CONFIRMED from {self.evidence_class}; "
                    "clinical and semantic causes require human adjudication")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CauseFinding:
        return cls(
            cause=str(value.get("cause") or ""),
            status=str(value.get("status") or ""),
            evidence_class=str(value.get("evidence_class") or ""),
            rationale=str(value.get("rationale") or ""),
            evidence=tuple(EvidenceRef.from_dict(x) for x in (value.get("evidence") or ())),
            field=str(value.get("field") or ""),
            parameter_id=str(value.get("parameter_id") or ""),
            route_owner=str(value.get("route_owner") or ""),
            relation_to_target=str(value.get("relation_to_target") or "UNKNOWN"),
            causal_strength=str(value.get("causal_strength") or "PLAUSIBLE"),
            mechanism=str(value.get("mechanism") or ""),
            counterfactual_prediction=str(
                value.get("counterfactual_prediction") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "cause": self.cause, "status": self.status,
            "evidence_class": self.evidence_class, "rationale": self.rationale,
            "evidence": [x.to_dict() for x in self.evidence], "field": self.field,
            "parameter_id": self.parameter_id, "route_owner": self.route_owner,
            "relation_to_target": self.relation_to_target,
            "causal_strength": self.causal_strength,
            "mechanism": self.mechanism,
            "counterfactual_prediction": self.counterfactual_prediction,
        }

@dataclass(frozen=True)
class AttributionReport:
    case_id: str
    spec_id: str
    mode: str
    primary_cause: CauseFinding
    contributing_causes: tuple[CauseFinding, ...]
    alternatives_considered: tuple[str, ...]
    probes: tuple[AttributionProbe, ...]
    termination_reason: str
    confirmation_performed: bool
    confirmation_new_conflict: bool
    model_calls: int = 0
    chart_reads: int = 0
    spend: Mapping[str, Any] = field(default_factory=dict)
    gate_rejections: tuple[str, ...] = ()
    target_event: TargetEvent | None = None
    counterfactual_tests: tuple[CounterfactualTest, ...] = ()
    skeptic_review: SkepticReview | None = None
    modules: tuple[str, ...] = ()
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        safe_case_id(self.case_id)
        if self.mode not in ATTRIBUTION_MODES:
            raise AttributionError(f"mode must be one of {ATTRIBUTION_MODES}")
        if not self.alternatives_considered:
            raise AttributionError("attribution must name the principal rival causes considered")
        if not self.confirmation_performed and self.primary_cause.cause != "UNRESOLVED":
            raise AttributionError(
                "a resolved attribution requires a final targeted confirmation round")
        if self.confirmation_new_conflict and self.primary_cause.status == "CONFIRMED":
            raise AttributionError("a new confirmation conflict forbids CONFIRMED")
        if self.target_event is not None and self.primary_cause.cause != "UNRESOLVED":
            if self.primary_cause.relation_to_target != "EXPLAINS":
                raise AttributionError(
                    "a resolved primary cause must EXPLAIN the selected target event; "
                    "contributing or unrelated defects belong in contributing_causes")
            if self.primary_cause.causal_strength == "COUNTERFACTUAL_SUPPORTED":
                supported = any(
                    test.target_event_id == self.target_event.event_id
                    and test.outcome == "SUPPORTED"
                    for test in self.counterfactual_tests
                )
                if not supported:
                    raise AttributionError(
                        "COUNTERFACTUAL_SUPPORTED requires a supported test for the target")
        if (self.skeptic_review is not None
                and self.skeptic_review.verdict in ("REVISE", "UNRESOLVED")
                and self.primary_cause.cause != "UNRESOLVED"):
            raise AttributionError(
                "a skeptic REVISE/UNRESOLVED verdict requires an UNRESOLVED report")
        if self.mode != GOLD:
            for cause in (self.primary_cause, *self.contributing_causes):
                if cause.evidence_class == "HUMAN_ADJUDICATED":
                    raise AttributionError(
                        f"{self.mode} has no human-adjudicated truth in its packet; "
                        "use JUDGED, DETERMINISTIC, or GOLD_CONDITIONAL as applicable")
                if cause.cause in CLINICAL_CAUSES and cause.status == "CONFIRMED":
                    raise AttributionError(
                        f"{self.mode} cannot confirm clinical cause {cause.cause}")

    @property
    def lifecycle(self) -> str:
        if self.mode == REGISTRY_REFERENCE:
            return "NEEDS_ADJUDICATION"
        if self.primary_cause.cause == "UNRESOLVED":
            return "OPEN"
        return "ATTRIBUTED"

    @property
    def repair_route(self) -> str:
        """The only downstream action this evidence mode and cause authorize."""
        if self.mode == REGISTRY_REFERENCE:
            return "ADJUDICATE_REFERENCE"
        if self.mode == BLIND:
            return "HUMAN_REVIEW_TEST_OBLIGATION"
        return {
            "REFERENCE_OR_GOLD": "ADJUDICATED_OUT",
            "RETRIEVAL": "RETRIEVAL_ASSET_CANDIDATE",
            "SPEC_FORM": "FORM_PATCH_CANDIDATE",
            "SPEC_CONTENT": "CLINICIAN_QUESTION",
            "SKILL_OR_PROMPT": "ENGINEER_FIX_CANDIDATE",
            "ANSWER_CHECK_OR_GATE": "ENGINEER_OR_CLINICIAN_REVIEW",
            "RUNTIME_OR_PROVIDER": "ENGINEER_FIX_CANDIDATE",
            "EVIDENCE_GAP": "OUTSIDE_CHART_REVIEW",
            "UNRESOLVED": "HUMAN_REVIEW_TEST_OBLIGATION",
        }.get(self.primary_cause.cause, "CONTRASTIVE_FAILURE_PACKET")

    def to_dict(self) -> dict:
        return {
            "schema": ATTRIBUTION_SCHEMA, "case_id": self.case_id,
            "spec_id": self.spec_id, "mode": self.mode,
            "semantic_patch_allowed": self.mode == GOLD,
            "repair_route": self.repair_route,
            "primary_cause": self.primary_cause.to_dict(),
            "contributing_causes": [x.to_dict() for x in self.contributing_causes],
            "alternatives_considered": list(self.alternatives_considered),
            "probes": [x.to_dict() for x in self.probes],
            "termination_reason": self.termination_reason,
            "confirmation_performed": self.confirmation_performed,
            "confirmation_new_conflict": self.confirmation_new_conflict,
            "model_calls": self.model_calls, "chart_reads": self.chart_reads,
            "spend": dict(self.spend), "lifecycle": self.lifecycle,
            "gate_rejections": list(self.gate_rejections),
            "target_event": self.target_event.to_dict() if self.target_event else None,
            "counterfactual_tests": [
                test.to_dict() for test in self.counterfactual_tests],
            "skeptic_review": (
                self.skeptic_review.to_dict() if self.skeptic_review else None),
            "modules": list(self.modules),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttributionReport:
        if value.get("schema") != ATTRIBUTION_SCHEMA:
            raise AttributionError(f"expected schema {ATTRIBUTION_SCHEMA}")
        return cls(
            case_id=str(value.get("case_id") or ""),
            spec_id=str(value.get("spec_id") or ""),
            mode=str(value.get("mode") or ""),
            primary_cause=CauseFinding.from_dict(value.get("primary_cause") or {}),
            contributing_causes=tuple(CauseFinding.from_dict(x)
                                      for x in (value.get("contributing_causes") or ())),
            alternatives_considered=tuple(str(x)
                                          for x in (value.get("alternatives_considered") or ())),
            probes=tuple(AttributionProbe(
                probe_id=str(x.get("probe_id") or ""),
                question=str(x.get("question") or ""),
                alternatives=tuple(str(a) for a in (x.get("alternatives") or ())),
                expected_discriminator=str(x.get("expected_discriminator") or ""),
                confirmation=bool(x.get("confirmation")),
                chart_reads=tuple(str(a) for a in (x.get("chart_reads") or ())),
            ) for x in (value.get("probes") or ())),
            termination_reason=str(value.get("termination_reason") or ""),
            confirmation_performed=bool(value.get("confirmation_performed")),
            confirmation_new_conflict=bool(value.get("confirmation_new_conflict")),
            model_calls=int(value.get("model_calls") or 0),
            chart_reads=int(value.get("chart_reads") or 0),
            spend=dict(value.get("spend") or {}),
            gate_rejections=tuple(str(x) for x in (value.get("gate_rejections") or ())),
            target_event=(
                TargetEvent.from_dict(value["target_event"])
                if value.get("target_event") else None),
            counterfactual_tests=tuple(
                CounterfactualTest.from_dict(row)
                for row in (value.get("counterfactual_tests") or ())),
            skeptic_review=(
                SkepticReview.from_dict(value["skeptic_review"])
                if value.get("skeptic_review") else None),
            modules=tuple(str(x) for x in (value.get("modules") or ())),
            created_at=str(value.get("created_at") or _now()),
        )

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
    gold = {
        str(row.get("case_id") or ""): str(
            row.get("primary_cause")
            or (row.get("adjudication") or {}).get("primary_cause")
            or ""
        )
        for row in adjudications
        if row.get("case_id")
    }
    pairs: list[tuple[str, str]] = []
    citation_invalid = 0
    clinical_auto_confirmed = 0
    scope_violations = 0
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
            )
            if condition
        ],
    }

def load_registry_references(path: str | Path) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != REGISTRY_REFERENCE_SCHEMA:
        raise AttributionError(f"{path}: expected schema {REGISTRY_REFERENCE_SCHEMA}")
    if raw.get("storage") != "LOCAL_ONLY" or raw.get("contains_phi") is not True:
        raise AttributionError(
            f"{path}: registry reference must declare LOCAL_ONLY and contains_phi=true")
    out = {}
    for row in raw.get("cases") or ():
        case_id = safe_case_id(row.get("case_id"))
        if str(row.get("adjudication") or "") != "UNRESOLVED":
            raise AttributionError(f"{case_id}: staged registry reference must be UNRESOLVED")
        if case_id in out:
            raise AttributionError(f"{path}: duplicate case_id {case_id}")
        out[case_id] = dict(row)
    return out

def trace_path_for_manifest(manifest_path: str | Path) -> Path | None:
    path = Path(manifest_path)
    candidate = path.with_name(path.name.replace(".manifest.json", ".jsonl"))
    return candidate if candidate.is_file() and candidate != path else None

def build_packet(*, manifest_path: str | Path, case_id: str, spec: Any, mode: str,
                 detector_findings: Sequence[Mapping[str, Any]] = (),
                 chart_gold: ChartObservableGold | None = None,
                 registry_reference: Mapping[str, Any] | None = None) -> AttributionPacket:
    """Build a pseudonymous packet while retaining content-addressed local source references."""
    mpath = Path(manifest_path).resolve()
    raw_manifest = json.loads(mpath.read_text(encoding="utf-8"))
    patient_id = str(raw_manifest.get("patient_id") or "")
    trace_path = trace_path_for_manifest(mpath)
    trace = []
    if trace_path:
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    manifest = _replace(raw_manifest, patient_id, case_id)
    trace = _replace(trace, patient_id, case_id)
    if str(manifest.get("spec_id") or "") != spec.spec_id:
        raise AttributionError(
            f"{mpath}: manifest spec {manifest.get('spec_id')!r} != {spec.spec_id!r}")
    if str(manifest.get("spec_hash") or "") != spec.spec_hash:
        raise AttributionError(
            f"{mpath}: manifest spec hash does not match the supplied frozen spec")
    from ..contract.behaviour import BehaviorSignature
    signature = BehaviorSignature.from_manifest(manifest, case_id=case_id).to_dict()
    return AttributionPacket(
        case_id=safe_case_id(case_id), spec_id=spec.spec_id, spec_hash=spec.spec_hash,
        mode=mode, manifest_ref=ArtifactRef.from_path(mpath),
        trace_ref=ArtifactRef.from_path(trace_path) if trace_path else None,
        manifest=manifest, trace=tuple(trace),
        rule_catalogue=tuple(r.to_dict(with_text=True) for r in rule_catalog(spec)),
        detector_findings=tuple(dict(x) for x in detector_findings),
        behavior_signature=signature, chart_gold=chart_gold,
        registry_reference=dict(registry_reference) if registry_reference else None,
    )

# ======================================================================== zero-cost screen
def field_disagreements(answer: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    values = answer.get("value") or {}
    if not isinstance(values, Mapping):
        values = {}
    return sorted(
        field for field, expected in reference.items()
        if str(values.get(field) or "") != str(expected or "")
    )

def selection_reasons(packet: AttributionPacket) -> tuple[str, ...]:
    """Deterministic reasons a case enters the expensive attribution queue."""
    m, reasons = packet.manifest, []
    answer = m.get("answer") or {}
    status = str(answer.get("status") or "")
    if status in evals.ABSTAIN_STATUSES:
        reasons.append(f"terminal_status:{status}")
    if not bool(m.get("gate_validated")):
        reasons.append("gate_not_validated")
    if m.get("coverage_unreachable"):
        reasons.append("coverage_unreachable")
    if m.get("spend_stopped") or m.get("expansion_stopped"):
        reasons.append("budget_or_expansion_stopped")
    if m.get("stalled") or m.get("rejection_loop_stopped"):
        reasons.append("rejection_loop")
    if int(m.get("no_tool_call_recoveries") or m.get("no_tool_call") or 0) > 0:
        reasons.append("no_tool_call_recovery")
    if m.get("error"):
        reasons.append("runtime_error")
    for finding in packet.detector_findings:
        severity = str(finding.get("severity") or "")
        if severity in ("IRB", "CRITICAL"):
            reasons.append(f"detector:{finding.get('detector')}")
    if packet.mode == REGISTRY_REFERENCE and packet.registry_reference:
        mismatches = field_disagreements(
            answer, packet.registry_reference.get("registry_value") or {})
        reasons.extend(f"registry_disagreement:{field}" for field in mismatches)
    if packet.mode == GOLD and packet.chart_gold:
        expected = {
            field: gold.value for field, gold in packet.chart_gold.chart_answer.items()
            if gold.status == "FOUND"
        }
        mismatches = field_disagreements(answer, expected)
        reasons.extend(f"gold_mismatch:{field}" for field in mismatches)
        for field, gold in packet.chart_gold.chart_answer.items():
            if gold.status != "FOUND" and status == "FOUND" and field in (answer.get("value") or {}):
                reasons.append(f"overclaim:{field}")
    return tuple(sorted(set(reasons)))

def derive_target_events(packet: AttributionPacket) -> tuple[TargetEvent, ...]:
    """Turn screening signals into explicit questions the investigator may select."""
    answer = packet.manifest.get("answer") or {}
    observed_values = answer.get("value") or {}
    observed_values = observed_values if isinstance(observed_values, Mapping) else {}
    events: list[TargetEvent] = []

    def add(*, kind: str, field: str = "", observed: Any = None,
            reference: Any = None, source: str, truth_status: str, question: str) -> None:
        identity = {
            "case_id": packet.case_id, "kind": kind, "field": field,
            "observed": observed, "reference": reference, "source": source,
        }
        events.append(TargetEvent(
            event_id=f"TE-{_event_id(identity)[:16]}", kind=kind, field=field,
            observed=observed, reference_signal=reference, source=source,
            truth_status=truth_status, question=question,
        ))

    if packet.mode == REGISTRY_REFERENCE and packet.registry_reference:
        reference = packet.registry_reference.get("registry_value") or {}
        for field in field_disagreements(answer, reference):
            add(
                kind="FIELD_DISAGREEMENT", field=field,
                observed=observed_values.get(field), reference=reference.get(field),
                source="REGISTRY_REFERENCE", truth_status="UNRESOLVED_REFERENCE",
                question=(
                    f"Why does the run's {field} differ from the unresolved registry "
                    "reference, without assuming either value is correct?"),
            )
    elif packet.mode == GOLD and packet.chart_gold:
        for field, gold in packet.chart_gold.chart_answer.items():
            observed = observed_values.get(field)
            if str(observed or "") != str(gold.value or ""):
                add(
                    kind="FIELD_MISMATCH", field=field, observed=observed,
                    reference={"status": gold.status, "value": gold.value},
                    source="GOLD", truth_status="ADJUDICATED",
                    question=f"Why did the run fail to produce adjudicated {field} gold?",
                )

    reasons = selection_reasons(packet)
    for reason in reasons:
        if reason.startswith(("registry_disagreement:", "gold_mismatch:", "overclaim:")):
            continue
        source = "DETECTOR" if reason.startswith("detector:") else "TRACE"
        add(
            kind="PROCESS_ANOMALY", observed=reason, source=source,
            truth_status="DETERMINISTIC" if source == "DETECTOR" else "NO_TRUTH",
            question=f"What caused the recorded process anomaly {reason!r}?",
        )
    if not events:
        add(
            kind="RUN_OUTCOME", observed=answer, source="MANUAL",
            truth_status="NO_TRUTH",
            question="What, if anything, explains the run outcome selected for review?",
        )
    unique = {event.event_id: event for event in events}
    return tuple(unique[key] for key in sorted(unique))

def batch_behavior_conflicts(packets: Sequence[AttributionPacket]) -> dict[str, tuple[str, ...]]:
    """Flag cases whose repeated runs induce more than one structured behavior."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for packet in packets:
        grouped[packet.case_id].add(
            _canonical({
                "answer": packet.behavior_signature.get("answer"),
                "evidence": packet.behavior_signature.get("evidence"),
                "rules": packet.behavior_signature.get("rules_applied"),
                "gate": packet.behavior_signature.get("gate_valid"),
            })
        )
    return {
        case: ("behavioral_entropy_nonzero",) if len(signatures) > 1 else ()
        for case, signatures in grouped.items()
    }

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
    case_id: str
    decision: str
    actor: str
    actor_role: str
    rationale: str
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

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "decision": self.decision, "actor": self.actor,
            "actor_role": self.actor_role, "rationale": self.rationale,
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

# ================================================================ deepagents attribution
@dataclass
class AttributionRuntimeContext:
    packet: AttributionPacket
    chart: Any
    max_chart_reads: int
    max_usd: float
    probes: list[AttributionProbe] = field(default_factory=list)
    active_probe: int | None = None
    trace_inspected: bool = False
    chart_audit: list[dict] = field(default_factory=list)
    draft_causes: list[dict] = field(default_factory=list)
    ruled_out: list[dict] = field(default_factory=list)
    report: AttributionReport | None = None
    accepted: bool = False
    n_model_calls: int = 0
    chart_reads: int = 0
    stopped_reason: str = ""
    submission_rejections: list[str] = field(default_factory=list)
    spend: Any = None
    candidate_targets: tuple[TargetEvent, ...] = ()
    target_event: TargetEvent | None = None
    counterfactual_tests: list[CounterfactualTest] = field(default_factory=list)
    skeptic_review: SkepticReview | None = None
    modules: tuple[Any, ...] = ()

    def attach_read(self, description: str) -> None:
        if self.active_probe is None:
            return
        current = self.probes[self.active_probe]
        self.probes[self.active_probe] = replace(
            current, chart_reads=current.chart_reads + (description,))

def _tool_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)[:24000]

def _packet_ref_exists(ref: str, packet: AttributionPacket) -> bool:
    """Resolve a model citation to an exact field in the immutable input packet."""
    path = str(ref).strip().replace("[", ".").replace("]", "")
    path = path.removeprefix("packet.")
    roots: dict[str, Any] = {
        "manifest": packet.manifest,
        "answer": packet.manifest.get("answer") or {},
        "behavior_signature": packet.behavior_signature,
        "registry_reference": packet.registry_reference,
        "packet_summary": packet.prompt_summary(),
        "mode": packet.mode,
    }
    if path in roots and roots[path] is not None:
        return True
    parts = [part for part in path.split(".") if part]
    if not parts or parts[0] not in roots:
        return False
    value = roots[parts.pop(0)]
    for part in parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return False
    return value is not None

def _citation_errors(finding: CauseFinding, ctx: AttributionRuntimeContext) -> list[str]:
    return _reference_errors(finding.evidence, ctx)

def _reference_errors(
        references: Sequence[EvidenceRef], ctx: AttributionRuntimeContext) -> list[str]:
    errors = []
    trace_seqs = {str(e.get("seq")) for e in ctx.packet.trace if e.get("seq") is not None}
    detectors = {str(e.get("detector") or "") for e in ctx.packet.detector_findings}
    rules = {str(e.get("rule_id") or "") for e in ctx.packet.rule_catalogue}
    notes = {
        str(row.get("note_id") or "") for row in ctx.chart_audit
        if row.get("tool") == "read_document"
    }
    witnesses = {
        e.note_id for e in (ctx.packet.chart_gold.gold_evidence
                            if ctx.packet.chart_gold else ())
    }
    for ref in references:
        target = ref.ref.removeprefix("seq:")
        if ref.kind == "trace" and target not in trace_seqs:
            errors.append(f"trace ref {ref.ref!r} does not name a packet event")
        elif ref.kind == "detector" and ref.ref not in detectors:
            errors.append(f"detector ref {ref.ref!r} did not fire")
        elif ref.kind == "spec_rule" and ref.ref not in rules:
            errors.append(f"spec rule {ref.ref!r} is not in the frozen catalogue")
        elif ref.kind == "note" and ref.ref not in notes:
            errors.append(f"note ref {ref.ref!r} was not read during attribution")
        elif ref.kind == "gold_witness":
            if ctx.packet.mode != GOLD or ref.ref not in witnesses:
                errors.append(f"gold witness {ref.ref!r} is unavailable in this mode")
        elif ref.kind == "human" and ctx.packet.mode != GOLD:
            errors.append("unadjudicated modes cannot cite a human adjudication")
        elif ref.kind == "packet" and not _packet_ref_exists(ref.ref, ctx.packet):
            errors.append(
                f"packet ref {ref.ref!r} is not an exact manifest/answer/"
                "behavior_signature/registry_reference path")
        elif ref.kind == "probe" and ref.ref not in {p.probe_id for p in ctx.probes}:
            errors.append(f"probe ref {ref.ref!r} was not opened in this attribution")
    return errors

def _report_from_submission(value: Mapping[str, Any], ctx: AttributionRuntimeContext
                            ) -> AttributionReport:
    primary = CauseFinding.from_dict(value.get("primary_cause") or {})
    contributing = tuple(
        CauseFinding.from_dict(x) for x in (value.get("contributing_causes") or ())
    )
    errors = []
    for finding in (primary, *contributing):
        errors.extend(_citation_errors(finding, ctx))
    if errors:
        raise AttributionError("; ".join(errors))
    confirmation = bool(value.get("confirmation_performed"))
    if confirmation and not any(p.confirmation for p in ctx.probes):
        raise AttributionError(
            "confirmation_performed=true but no confirmation AttributionProbe was opened")
    report = AttributionReport(
        case_id=ctx.packet.case_id, spec_id=ctx.packet.spec_id, mode=ctx.packet.mode,
        primary_cause=primary, contributing_causes=contributing,
        alternatives_considered=tuple(
            str(x) for x in (value.get("alternatives_considered") or ())),
        probes=tuple(ctx.probes),
        termination_reason=str(value.get("termination_reason") or "obligations_closed"),
        confirmation_performed=confirmation,
        confirmation_new_conflict=bool(value.get("confirmation_new_conflict")),
        model_calls=ctx.n_model_calls, chart_reads=ctx.chart_reads,
        spend=ctx.spend.report() if ctx.spend else {},
        target_event=ctx.target_event,
        counterfactual_tests=tuple(ctx.counterfactual_tests),
        skeptic_review=ctx.skeptic_review,
        modules=tuple(module.module_id for module in ctx.modules),
    )
    module_errors = [
        error
        for module in ctx.modules
        if module.validate is not None
        for error in module.validate(report, ctx)
    ]
    if module_errors:
        raise AttributionError("; ".join(module_errors))
    return report

def attribution_tools(ctx: AttributionRuntimeContext) -> list[Any]:
    """The complete tool surface: packet readers, same-patient chart probes, typed output."""
    from langchain_core.tools import StructuredTool

    def list_target_events() -> str:
        return _tool_payload({
            "target_events": [event.to_dict() for event in ctx.candidate_targets],
            "instruction": (
                "Select exactly one event. Other defects may be recorded, but only a defect "
                "that explains this event may become primary."),
        })

    def select_target_event(target_event_id: str) -> str:
        target = next(
            (event for event in ctx.candidate_targets
             if event.event_id == str(target_event_id)), None)
        if target is None:
            return _tool_payload({
                "accepted": False, "error": "UNKNOWN_TARGET_EVENT",
                "known": [event.event_id for event in ctx.candidate_targets],
            })
        if ctx.target_event is not None and ctx.target_event.event_id != target.event_id:
            return _tool_payload({
                "accepted": False, "error": "TARGET_ALREADY_SELECTED",
                "selected": ctx.target_event.event_id,
            })
        ctx.target_event = target
        return _tool_payload({"accepted": True, "target_event": target.to_dict()})

    def inspect_trace(start_seq: int = 0, end_seq: int = 0,
                      kinds: list[str] | None = None) -> str:
        ctx.trace_inspected = True
        wanted = set(kinds or ())
        rows = [
            event for event in ctx.packet.trace
            if (not wanted or str(event.get("kind") or "") in wanted)
            and (not start_seq or int(event.get("seq") or 0) >= start_seq)
            and (not end_seq or int(event.get("seq") or 0) <= end_seq)
        ]
        return _tool_payload({"n_matching": len(rows), "events": rows[:150],
                              "truncated": len(rows) > 150})

    def inspect_spec(rule_ids: list[str] | None = None) -> str:
        wanted = set(rule_ids or ())
        rows = [
            row for row in ctx.packet.rule_catalogue
            if not wanted or str(row.get("rule_id") or "") in wanted
        ]
        return _tool_payload({"spec_id": ctx.packet.spec_id, "rules": rows})

    def inspect_causal_stage(stage: str) -> str:
        stage_name = str(stage).strip().upper()
        vocabulary = {
            "RETRIEVAL": ("search", "list_documents", "search_documents"),
            "EVIDENCE": ("read", "evidence", "citation", "admissib"),
            "INTERPRETATION": ("reason", "hypothesis", "decision"),
            "CODING": ("answer", "code", "submit"),
            "GATE": ("rejected", "gate", "check", "coverage"),
            "OUTPUT": ("final", "accepted", "termination"),
        }
        if stage_name not in vocabulary:
            return _tool_payload({
                "error": "UNKNOWN_CAUSAL_STAGE", "known": sorted(vocabulary)})
        terms = vocabulary[stage_name]
        rows = [
            event for event in ctx.packet.trace
            if any(term in _canonical(event).lower() for term in terms)
        ]
        manifest = {}
        if stage_name in ("CODING", "OUTPUT"):
            manifest["answer"] = ctx.packet.manifest.get("answer")
            manifest["gate_validated"] = ctx.packet.manifest.get("gate_validated")
        if stage_name == "GATE":
            manifest.update({
                "rejections": ctx.packet.manifest.get("rejections"),
                "open_threads": ctx.packet.manifest.get("open_threads"),
                "coverage_unreachable": ctx.packet.manifest.get("coverage_unreachable"),
            })
        return _tool_payload({
            "stage": stage_name, "n_events": len(rows),
            "events": rows[:100], "manifest": manifest,
            "truncated": len(rows) > 100,
        })

    def open_attribution_probe(question: str, alternatives: list[str],
                               expected_discriminator: str,
                               confirmation: bool = False) -> str:
        if not ctx.trace_inspected:
            return _tool_payload({
                "error": "TRACE_FIRST",
                "message": "Inspect the recorded trajectory before opening a chart probe.",
            })
        if len(alternatives or []) < 2:
            return _tool_payload({
                "error": "RIVALS_REQUIRED",
                "message": "Name at least two rival explanations this probe distinguishes.",
            })
        probe = AttributionProbe(
            probe_id=f"probe-{len(ctx.probes) + 1}", question=str(question).strip(),
            alternatives=tuple(str(x) for x in alternatives),
            expected_discriminator=str(expected_discriminator).strip(),
            confirmation=bool(confirmation),
        )
        ctx.probes.append(probe)
        ctx.active_probe = len(ctx.probes) - 1
        return _tool_payload({"opened": probe.to_dict()})

    def _chart_ready() -> dict | None:
        if not ctx.trace_inspected:
            return {"error": "TRACE_FIRST",
                    "message": "Inspect the run trace before reading the chart."}
        if ctx.active_probe is None:
            return {"error": "PROBE_REQUIRED",
                    "message": "Call open_attribution_probe before accessing the chart."}
        return None

    def list_documents(doc_type_contains: str = "", date_from: str = "",
                       date_to: str = "", limit: int = 100, offset: int = 0) -> str:
        if blocked := _chart_ready():
            return _tool_payload(blocked)
        rows, total = ctx.chart.list_documents(
            doc_type_contains or None, date_from or None, date_to or None,
            min(max(int(limit), 1), 500), max(int(offset), 0))
        payload = {
            "total_matching": total, "documents": [r.to_dict() for r in rows],
            "patient_scope": ctx.packet.case_id,
        }
        ctx.chart_audit.append({"tool": "list_documents", "probe_id":
                                ctx.probes[ctx.active_probe].probe_id,
                                "n": len(rows)})
        ctx.attach_read(f"list:{doc_type_contains or '*'}")
        return _tool_payload(payload)

    def search_documents(query: str, doc_type_contains: str = "", date_from: str = "",
                         date_to: str = "", max_hits: int = 25) -> str:
        if blocked := _chart_ready():
            return _tool_payload(blocked)
        if not str(query).strip():
            return _tool_payload({"error": "query is required"})
        hits = ctx.chart.search(
            str(query), False, doc_type_contains or None, date_from or None,
            date_to or None, max_hits=min(max(int(max_hits), 1), 100))
        ctx.chart_audit.append({
            "tool": "search_documents", "probe_id": ctx.probes[ctx.active_probe].probe_id,
            "query": query, "n_hits": len(hits),
        })
        ctx.attach_read(f"search:{query}")
        return _tool_payload({"query": query, "hits": [asdict(h) for h in hits],
                              "patient_scope": ctx.packet.case_id})

    def read_document(note_id: str, offset: int = 0, limit: int = 6000) -> str:
        if blocked := _chart_ready():
            return _tool_payload(blocked)
        if ctx.chart_reads >= ctx.max_chart_reads:
            return _tool_payload({
                "error": "CHART_EXPANSION_LIMIT",
                "max_chart_reads": ctx.max_chart_reads,
                "message": "No more documents may be opened; resolve or submit UNRESOLVED.",
            })
        if note_id not in ctx.chart._docs:
            return _tool_payload({
                "error": "UNKNOWN_NOTE_ID",
                "message": "Use a note_id returned by list_documents or search_documents.",
            })
        result = ctx.chart.read(
            note_id, max(int(offset), 0), min(max(int(limit), 1), 20000))
        ctx.chart_reads += 1
        ctx.chart_audit.append({
            "tool": "read_document", "probe_id": ctx.probes[ctx.active_probe].probe_id,
            "note_id": note_id, "offset": result["offset"],
            "returned_chars": result["returned_chars"],
        })
        ctx.attach_read(f"note:{note_id}@{result['offset']}")
        result["patient_scope"] = ctx.packet.case_id
        return _tool_payload(result)

    def record_cause(cause: str, rationale: str, evidence: list[dict],
                     status: str = "POSSIBLE", evidence_class: str = "JUDGED",
                     field: str = "", parameter_id: str = "",
                     route_owner: str = "", relation_to_target: str = "UNKNOWN",
                     causal_strength: str = "PLAUSIBLE", mechanism: str = "",
                     counterfactual_prediction: str = "") -> str:
        try:
            finding = CauseFinding.from_dict({
                "cause": cause, "status": status, "evidence_class": evidence_class,
                "rationale": rationale, "evidence": evidence, "field": field,
                "parameter_id": parameter_id, "route_owner": route_owner,
                "relation_to_target": relation_to_target,
                "causal_strength": causal_strength,
                "mechanism": mechanism,
                "counterfactual_prediction": counterfactual_prediction,
            })
            errors = _citation_errors(finding, ctx)
            if errors:
                raise AttributionError("; ".join(errors))
        except AttributionError as exc:
            return _tool_payload({"accepted": False, "error": str(exc)})
        ctx.draft_causes.append(finding.to_dict())
        return _tool_payload({"accepted": True, "n_draft_causes": len(ctx.draft_causes)})

    def rule_out_cause(cause: str, why: str, evidence: list[dict]) -> str:
        if cause not in CAUSES:
            return _tool_payload({"accepted": False, "error": f"unknown cause {cause!r}"})
        try:
            refs = tuple(EvidenceRef.from_dict(x) for x in evidence)
            dummy = CauseFinding(cause=cause, status="POSSIBLE", evidence_class="JUDGED",
                                 rationale=why, evidence=refs)
            errors = _citation_errors(dummy, ctx)
            if errors:
                raise AttributionError("; ".join(errors))
        except AttributionError as exc:
            return _tool_payload({"accepted": False, "error": str(exc)})
        ctx.ruled_out.append({"cause": cause, "why": why,
                              "evidence": [x.to_dict() for x in refs]})
        return _tool_payload({"accepted": True, "n_ruled_out": len(ctx.ruled_out)})

    def record_counterfactual_test(kind: str, intervention: str, prediction: str,
                                   outcome: str, observation: str = "",
                                   evidence: list[dict] | None = None) -> str:
        if ctx.target_event is None:
            return _tool_payload({
                "accepted": False, "error": "TARGET_REQUIRED",
                "message": "Select the target event before testing a causal mechanism.",
            })
        try:
            refs = tuple(EvidenceRef.from_dict(row) for row in (evidence or ()))
            errors = _reference_errors(refs, ctx)
            if errors:
                raise AttributionError("; ".join(errors))
            test = CounterfactualTest(
                test_id=f"cf-{len(ctx.counterfactual_tests) + 1}",
                target_event_id=ctx.target_event.event_id,
                kind=str(kind), intervention=str(intervention),
                prediction=str(prediction), outcome=str(outcome),
                observation=str(observation), evidence=refs,
            )
        except AttributionError as exc:
            return _tool_payload({"accepted": False, "error": str(exc)})
        ctx.counterfactual_tests.append(test)
        return _tool_payload({"accepted": True, "test": test.to_dict()})

    def submit_skeptic_review(verdict: str, rationale: str,
                              objections: list[str] | None = None,
                              untested_alternatives: list[str] | None = None,
                              evidence: list[dict] | None = None) -> str:
        try:
            refs = tuple(EvidenceRef.from_dict(row) for row in (evidence or ()))
            errors = _reference_errors(refs, ctx)
            if errors:
                raise AttributionError("; ".join(errors))
            review = SkepticReview(
                verdict=str(verdict), rationale=str(rationale),
                objections=tuple(str(x) for x in (objections or ())),
                untested_alternatives=tuple(
                    str(x) for x in (untested_alternatives or ())),
                evidence=refs,
            )
        except AttributionError as exc:
            return _tool_payload({"accepted": False, "error": str(exc)})
        ctx.skeptic_review = review
        return _tool_payload({"accepted": True, "review": review.to_dict()})

    def submit_attribution(primary_cause: dict | None = None,
                           contributing_causes: list[dict] | None = None,
                           alternatives_considered: list[str] | None = None,
                           termination_reason: str = "",
                           confirmation_performed: bool = False,
                           confirmation_new_conflict: bool = False) -> str:
        try:
            report = _report_from_submission({
                "primary_cause": primary_cause or {},
                "contributing_causes": contributing_causes or [],
                "alternatives_considered": alternatives_considered or [],
                "termination_reason": termination_reason,
                "confirmation_performed": confirmation_performed,
                "confirmation_new_conflict": confirmation_new_conflict,
            }, ctx)
        except AttributionError as exc:
            ctx.submission_rejections.append(str(exc))
            return _tool_payload({
                "accepted": False, "error": str(exc),
                "message": (
                    "Correct the structured attribution. Do not raise certainty to solve a "
                    "rejection; add evidence, downgrade, or return UNRESOLVED."),
            })
        ctx.report = report
        ctx.accepted = True
        return _tool_payload({"accepted": True, "lifecycle": report.lifecycle})

    def make(function, name: str, description: str, schema: dict):
        return StructuredTool.from_function(
            func=function, name=name, description=description, args_schema=schema)

    evidence_schema = {
        "type": "object", "required": ["kind", "ref"], "additionalProperties": False,
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "trace", "note", "detector", "spec_rule", "gold_witness",
                    "human", "packet", "probe",
                ],
                "description": (
                    "Use trace for seq:<n>; note for a note_id actually read; detector for a "
                    "fired detector ID; spec_rule for an exact rule ID; packet for an exact "
                    "manifest/answer/behavior_signature/registry_reference dotted path; probe "
                    "for an opened probe ID."),
            },
            "ref": {"type": "string", "description": "Exact stable identifier or dotted path."},
            "detail": {"type": "string"},
        },
    }
    cause_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["cause", "status", "evidence_class", "rationale", "evidence"],
        "properties": {
            "cause": {"type": "string", "enum": list(CAUSES)},
            "status": {"type": "string", "enum": list(CERTAINTY)},
            "evidence_class": {"type": "string", "enum": list(EVIDENCE_CLASSES)},
            "rationale": {"type": "string"},
            "evidence": {"type": "array", "items": evidence_schema},
            "field": {"type": "string"},
            "parameter_id": {"type": "string", "enum": ["", *PARAMETER_IDS]},
            "route_owner": {"type": "string"},
            "relation_to_target": {
                "type": "string", "enum": list(RELATIONS_TO_TARGET)},
            "causal_strength": {
                "type": "string", "enum": list(CAUSAL_STRENGTH)},
            "mechanism": {"type": "string"},
            "counterfactual_prediction": {"type": "string"},
        },
    }
    return [
        make(list_target_events, "list_target_events",
             "List the exact output or process events available for causal attribution.", {
                 "type": "object", "properties": {},
             }),
        make(select_target_event, "select_target_event",
             "Select exactly one event this report will explain.", {
                 "type": "object", "required": ["target_event_id"], "properties": {
                     "target_event_id": {"type": "string"},
                 }}),
        make(inspect_trace, "inspect_trace",
             "Read selected events from the completed run before opening the chart.", {
                 "type": "object", "properties": {
                     "start_seq": {"type": "integer"}, "end_seq": {"type": "integer"},
                     "kinds": {"type": "array", "items": {"type": "string"}},
                 }}),
        make(inspect_spec, "inspect_spec",
             "Read exact frozen spec rules by stable rule ID.", {
                 "type": "object", "properties": {
                     "rule_ids": {"type": "array", "items": {"type": "string"}},
                 }}),
        make(inspect_causal_stage, "inspect_causal_stage",
             "Inspect trace events belonging to one causal stage.", {
                 "type": "object", "required": ["stage"], "properties": {
                     "stage": {"type": "string", "enum": [
                         "RETRIEVAL", "EVIDENCE", "INTERPRETATION",
                         "CODING", "GATE", "OUTPUT",
                     ]},
                 }}),
        make(open_attribution_probe, "open_attribution_probe",
             "Name rival causes and the observation that would discriminate them.", {
                 "type": "object", "required": [
                     "question", "alternatives", "expected_discriminator"],
                 "properties": {
                     "question": {"type": "string"},
                     "alternatives": {"type": "array", "items": {"type": "string"},
                                      "minItems": 2},
                     "expected_discriminator": {"type": "string"},
                     "confirmation": {"type": "boolean"},
                 }}),
        make(list_documents, "list_documents",
             "List metadata for this patient only; requires an open probe.", {
                 "type": "object", "properties": {
                     "doc_type_contains": {"type": "string"},
                     "date_from": {"type": "string"}, "date_to": {"type": "string"},
                     "limit": {"type": "integer"}, "offset": {"type": "integer"},
                 }}),
        make(search_documents, "search_documents",
             "Search this patient's chart only; requires an open probe.", {
                 "type": "object", "required": ["query"], "properties": {
                     "query": {"type": "string"}, "doc_type_contains": {"type": "string"},
                     "date_from": {"type": "string"}, "date_to": {"type": "string"},
                     "max_hits": {"type": "integer"},
                 }}),
        make(read_document, "read_document",
             "Read one note from this patient only; requires an open probe.", {
                 "type": "object", "required": ["note_id"], "properties": {
                     "note_id": {"type": "string"}, "offset": {"type": "integer"},
                     "limit": {"type": "integer"},
                 }}),
        make(record_cause, "record_cause",
             "Record a provisional, citation-backed cause.", {
                 "type": "object", "required": ["cause", "rationale", "evidence"],
                 "properties": {
                     "cause": {"type": "string", "enum": list(CAUSES)},
                     "rationale": {"type": "string"}, "evidence": {"type": "array",
                                                                  "items": evidence_schema},
                     "status": {"type": "string", "enum": list(CERTAINTY)},
                     "evidence_class": {"type": "string",
                                        "enum": list(EVIDENCE_CLASSES)},
                     "field": {"type": "string"}, "parameter_id": {"type": "string"},
                     "route_owner": {"type": "string"},
                     "relation_to_target": {
                         "type": "string", "enum": list(RELATIONS_TO_TARGET)},
                     "causal_strength": {
                         "type": "string", "enum": list(CAUSAL_STRENGTH)},
                     "mechanism": {"type": "string"},
                     "counterfactual_prediction": {"type": "string"},
                 }}),
        make(rule_out_cause, "rule_out_cause",
             "Record why a rival cause is not supported.", {
                 "type": "object", "required": ["cause", "why", "evidence"],
                 "properties": {
                     "cause": {"type": "string", "enum": list(CAUSES)},
                     "why": {"type": "string"},
                     "evidence": {"type": "array", "items": evidence_schema},
                 }}),
        make(record_counterfactual_test, "record_counterfactual_test",
             "Record a bounded intervention/replay against the selected target event.", {
                 "type": "object",
                 "required": ["kind", "intervention", "prediction", "outcome"],
                 "properties": {
                     "kind": {"type": "string"},
                     "intervention": {"type": "string"},
                     "prediction": {"type": "string"},
                     "outcome": {
                         "type": "string", "enum": list(COUNTERFACTUAL_OUTCOMES)},
                     "observation": {"type": "string"},
                     "evidence": {"type": "array", "items": evidence_schema},
                 }}),
        make(submit_skeptic_review, "submit_skeptic_review",
             "Challenge target alignment and record remaining objections before submission.", {
                 "type": "object", "required": ["verdict", "rationale"],
                 "properties": {
                     "verdict": {"type": "string", "enum": list(SKEPTIC_VERDICTS)},
                     "rationale": {"type": "string"},
                     "objections": {"type": "array", "items": {"type": "string"}},
                     "untested_alternatives": {
                         "type": "array", "items": {"type": "string"}},
                     "evidence": {"type": "array", "items": evidence_schema},
                 }}),
        make(submit_attribution, "submit_attribution",
             "Submit the final primary-plus-contributing structured attribution.", {
                 "type": "object", "required": [
                     "primary_cause", "contributing_causes", "alternatives_considered",
                     "termination_reason", "confirmation_performed"],
                 "properties": {
                     "primary_cause": cause_schema,
                     "contributing_causes": {"type": "array", "items": cause_schema},
                     "alternatives_considered": {
                         "type": "array", "items": {"type": "string"}, "minItems": 1},
                     "termination_reason": {"type": "string"},
                     "confirmation_performed": {"type": "boolean"},
                     "confirmation_new_conflict": {"type": "boolean"},
                 }}),
    ]

def _attribution_system_prompt(
        packet: AttributionPacket, modules: Sequence[Any] = (),
        eval_skills_prompt: str = "") -> str:
    boundary = {
        GOLD: (
            "The chart-observable gold in the packet was human adjudicated. You may use it to "
            "test the recorded behavior, but every root-cause claim still needs cited evidence."
        ),
        REGISTRY_REFERENCE: (
            "The registry value is an UNRESOLVED reference, not truth. A disagreement may only "
            "be NEEDS_ADJUDICATION. Do not call the agent, chart, registry, or spec clinically "
            "wrong and do not propose a semantic patch."
        ),
        BLIND: (
            "No truth is available. Report process anomalies and competing hypotheses only. "
            "Do not select a clinically correct answer and do not propose a semantic patch."
        ),
    }[packet.mode]
    module_instructions = "\n".join(
        f"- {module.module_id}@{module.version}: {module.instructions}"
        for module in modules if module.instructions
    )
    # Eval skills are METHOD, and they sit beside the stage instructions rather than replacing
    # them: a stage says what this run of the evaluator must produce, a skill says how a
    # careful reviewer goes about finding it. Neither may score — the scorer is a tool.
    # Interpolated with its own surrounding newlines rather than on a fixed template line, so
    # that a caller who supplies none (every `acr attribute` invocation) gets the prompt this
    # command rendered before eval skills existed, down to the blank lines.
    stripped = eval_skills_prompt.strip()
    eval_block = f"\n{stripped}\n" if stripped else ""
    return f"""You are the offline error-attribution agent for a completed chart-review run.
You explain this run; you do not re-run extraction and you never edit a specification.

BOUNDARY
{boundary}

WORKFLOW
1. List and select exactly one target event. This is the outcome you are explaining.
2. Call inspect_trace before any chart access. Identify what the run actually searched, read,
   cited, submitted, and why it stopped.
3. Read exact rules with inspect_spec. Do not paraphrase a rule you have not opened.
4. If the trace cannot distinguish rival causes, call open_attribution_probe with at least two
   alternatives and a concrete discriminator. Chart tools are read-only and patient-scoped.
5. For each cause, state relation_to_target. A genuine bug that would not change the selected
   event is UNRELATED_DEFECT and cannot be primary.
6. Record a bounded counterfactual test. LIKELY/CONFIRMED requires a SUPPORTED test for the
   selected target; otherwise downgrade to POSSIBLE or UNRESOLVED.
7. Perform a skeptic review. If it returns REVISE or UNRESOLVED, the report must be UNRESOLVED.
8. Run one final challenge probe with confirmation=true. If it exposes a conflict, downgrade.
9. Finish only with submit_attribution.

ACTIVE MODULES
{module_instructions}{eval_block}

CERTAINTY
Only deterministic ANSWER_CHECK_OR_GATE or RUNTIME_OR_PROVIDER facts may be automatically
CONFIRMED. Clinical, registry, evidence-meaning, and semantic-spec causes require a human
adjudication; report them LIKELY/POSSIBLE even when persuasive. Model confidence is not
evidence. Cite trace events as seq:<number>, exact spec rule IDs, detector IDs, note IDs read
during this attribution, or available gold witness note IDs.
In REGISTRY_REFERENCE and BLIND mode, HUMAN_ADJUDICATED is not an available evidence class;
no such truth is present. Use JUDGED for your interpretation or DETERMINISTIC for a fact code
can replay.
For immutable packet facts use kind=packet with an exact path beginning manifest, answer,
behavior_signature, registry_reference, or mode. For a negative-search/probe fact use
kind=probe with the exact probe ID. The submit tool's nested schema is authoritative.

Do not emit chain-of-thought. Store only concise rationales, rival causes, citations, probes,
counterfactual observations, skeptic objections, and the final structured report.
"""

def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            str(block.get("text") or "")
            if isinstance(block, Mapping) else str(block)
            for block in content
        )
    return str(content or "")

def _independent_skeptic_review(
    *,
    packet: AttributionPacket,
    report: AttributionReport,
    model: Any,
) -> tuple[SkepticReview, Mapping[str, Any] | None]:
    """Challenge a proposed cause in a separate, tool-free model call."""
    known_refs = {
        (ref.kind, ref.ref)
        for cause in (report.primary_cause, *report.contributing_causes)
        for ref in cause.evidence
    }
    known_refs.update(
        (ref.kind, ref.ref)
        for test in report.counterfactual_tests
        for ref in test.evidence
    )
    prompt = {
        "role": "independent causal skeptic",
        "task": (
            "Decide whether the proposed primary cause actually explains the selected target, "
            "rather than merely identifying a true but unrelated defect. Check the claimed "
            "mechanism, counterfactual support, rival causes, and certainty. Do not infer a "
            "clinical answer or approve a patch."
        ),
        "truth_mode": packet.mode,
        "target_event": (
            report.target_event.to_dict() if report.target_event else None),
        "primary_cause": report.primary_cause.to_dict(),
        "contributing_causes": [
            cause.to_dict() for cause in report.contributing_causes],
        "alternatives_considered": list(report.alternatives_considered),
        "counterfactual_tests": [
            test.to_dict() for test in report.counterfactual_tests],
        "allowed_evidence_refs": [
            {"kind": kind, "ref": ref} for kind, ref in sorted(known_refs)],
        "response_schema": {
            "verdict": list(SKEPTIC_VERDICTS),
            "rationale": "non-empty string",
            "objections": ["string"],
            "untested_alternatives": ["string"],
            "evidence": [{"kind": "allowed kind", "ref": "allowed ref"}],
        },
    }
    try:
        response = model.invoke([{
            "role": "user",
            "content": _tool_payload(prompt),
        }])
        text = _message_text(response).strip()
        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        raw = json.loads(text)
        review = SkepticReview.from_dict({
            **dict(raw),
            "reviewer": "INDEPENDENT_MODEL",
        })
        invalid_refs = [
            f"{ref.kind}:{ref.ref}"
            for ref in review.evidence
            if (ref.kind, ref.ref) not in known_refs
        ]
        if invalid_refs:
            raise AttributionError(
                "independent skeptic cited unavailable evidence: "
                + ", ".join(invalid_refs))
        usage = getattr(response, "usage_metadata", None)
        return review, usage
    except Exception as exc:  # noqa: BLE001 - invalid review fails closed
        return SkepticReview(
            verdict="UNRESOLVED",
            rationale=(
                "independent skeptic did not return a citation-valid review: "
                f"{type(exc).__name__}"
            ),
            objections=("independent review unavailable",),
            untested_alternatives=tuple(report.alternatives_considered),
            reviewer="INDEPENDENT_MODEL",
        ), None

def _apply_independent_skeptic(
    report: AttributionReport,
    review: SkepticReview,
) -> AttributionReport:
    if review.verdict not in ("REVISE", "UNRESOLVED"):
        return replace(report, skeptic_review=review)
    proposed = replace(
        report.primary_cause,
        status="POSSIBLE",
        causal_strength="PLAUSIBLE",
    )
    unresolved = CauseFinding(
        cause="UNRESOLVED",
        status="UNRESOLVED",
        evidence_class="JUDGED",
        rationale=(
            "the independent skeptic did not accept the proposed primary causal link"),
        evidence=(),
    )
    return replace(
        report,
        primary_cause=unresolved,
        contributing_causes=(proposed, *report.contributing_causes),
        skeptic_review=review,
        confirmation_new_conflict=True,
        termination_reason="independent skeptic conflict; human review required",
    )

def run_attribution_agent(*, packet: AttributionPacket, chart: Any, model: Any,
                          max_model_calls: int = 12, max_usd: float = 1.0,
                          max_chart_reads: int = 12,
                          attribution_profile: str = "causal-attribution-v1",
                          skeptic_model: Any | None = None,
                          eval_skills_prompt: str = "",
                          ) -> AttributionReport:
    """Run the bounded, same-patient attribution agent and return a validated report."""
    if max_model_calls < 2:
        raise AttributionError(
            "max_model_calls must be >= 2 to reserve an independent skeptic call")
    if max_usd <= 0:
        raise AttributionError("max_usd must be > 0")
    if max_chart_reads < 0:
        raise AttributionError("max_chart_reads must be >= 0")

    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, hook_config
    from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
    from langchain_core.messages import SystemMessage

    from ..core.spend import Spend
    from .attribution_modules import builtin_attribution_registry

    if chart.patient_id == packet.case_id:
        # Synthetic charts often use safe case IDs; this is fine and still one-patient scoped.
        pass
    module_registry = builtin_attribution_registry()
    modules = module_registry.profile(attribution_profile)
    ctx = AttributionRuntimeContext(
        packet=packet, chart=chart, max_chart_reads=max_chart_reads,
        max_usd=max_usd, candidate_targets=derive_target_events(packet),
        modules=modules,
    )
    ctx.spend = Spend(max_usd=max_usd,
                      model=getattr(model, "model_name", "") or str(model))
    investigator_call_limit = max_model_calls - 1

    class AttributionMiddleware(AgentMiddleware):
        def wrap_model_call(self, request: ModelRequest, handler):
            ctx.n_model_calls += 1
            if (ctx.n_model_calls >= max(1, investigator_call_limit - 4)
                    and not ctx.accepted):
                existing = request.system_message
                base = (
                    existing.content if existing is not None
                    and isinstance(existing.content, str)
                    else str(getattr(existing, "content", "") or "")
                )
                content = (
                    base
                    + "\n\nBUDGET CLOSURE: stop opening new probes. You have "
                    f"{investigator_call_limit - ctx.n_model_calls + 1} investigator call(s) "
                    "including this one. Submit now. In REGISTRY_REFERENCE/BLIND mode, "
                    "a well-cited LIKELY or "
                    "POSSIBLE hypothesis is valid; do not wait for clinical confirmation. If a "
                    "cause remains unsettled, submit UNRESOLVED instead of spending the final "
                    "turn on prose."
                )
                request = request.override(system_message=SystemMessage(content=content))
            return handler(request)

        @hook_config(can_jump_to=["model"])
        def after_model(self, state, runtime):
            messages = state.get("messages") or []
            last = messages[-1] if messages else None
            if last is not None and getattr(last, "type", None) == "ai":
                ctx.spend.add(getattr(last, "usage_metadata", None))
            if ctx.accepted:
                return None
            if why := ctx.spend.exceeded():
                ctx.stopped_reason = why
                return {"jump_to": "end"}
            if last is not None and getattr(last, "tool_calls", None):
                return None
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        "No validated attribution has been submitted. Continue with a declared "
                        "tool; if the evidence cannot settle the cause, submit UNRESOLVED."),
                }],
                "jump_to": "model",
            }

    available_tools = {tool.name: tool for tool in attribution_tools(ctx)}
    requested_tools = {
        name for module in modules for name in module.tool_names
    }
    missing_tools = sorted(requested_tools - set(available_tools))
    if missing_tools:
        raise AttributionError(
            f"profile {attribution_profile!r} requests unknown tools {missing_tools}")
    tools = [available_tools[name] for name in sorted(requested_tools)]
    middleware = [
        SummarizationMiddleware(model=model, backend=StateBackend(), keep=("messages", 20)),
        ModelCallLimitMiddleware(
            thread_limit=investigator_call_limit, exit_behavior="end"),
        AttributionMiddleware(),
    ]
    agent = create_agent(
        model, tools,
        system_prompt=_attribution_system_prompt(packet, modules, eval_skills_prompt),
        middleware=middleware,
    )
    n_per_turn = sum(
        1 for node in agent.nodes
        if node != "__start__" and not node.endswith((".before_agent", ".after_agent"))
    )
    prompt = {
        "task": "Attribute the recorded run using the declared workflow and submit a report.",
        "packet_summary": packet.prompt_summary(),
        "available_causes": list(CAUSES),
        "available_parameters": list(PARAMETER_IDS),
        "target_events": [event.to_dict() for event in ctx.candidate_targets],
        "active_modules": [module.module_id for module in modules],
    }
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": _tool_payload(prompt)}]},
            config={
                "recursion_limit": (
                    investigator_call_limit * max(n_per_turn, 2) + 10)
            },
        )
    except Exception as exc:  # noqa: BLE001 - an attribution failure becomes an explicit report
        ctx.stopped_reason = f"{type(exc).__name__}: {exc}"

    if ctx.report is None:
        why = ctx.stopped_reason or (
            f"model-call limit reached without a gate-valid attribution "
            f"({ctx.n_model_calls}/{max_model_calls})"
        )
        report = AttributionReport(
            case_id=packet.case_id, spec_id=packet.spec_id, mode=packet.mode,
            primary_cause=CauseFinding(
                cause="UNRESOLVED", status="UNRESOLVED", evidence_class="JUDGED",
                rationale=why, evidence=(),
            ),
            contributing_causes=(),
            alternatives_considered=("insufficient evidence", "budget or execution limit"),
            probes=tuple(ctx.probes), termination_reason=why,
            confirmation_performed=False, confirmation_new_conflict=False,
            model_calls=ctx.n_model_calls, chart_reads=ctx.chart_reads,
            spend=ctx.spend.report(),
            gate_rejections=tuple(ctx.submission_rejections[-20:]),
            target_event=ctx.target_event,
            counterfactual_tests=tuple(ctx.counterfactual_tests),
            skeptic_review=ctx.skeptic_review,
            modules=tuple(module.module_id for module in modules),
        )
    else:
        review, usage = _independent_skeptic_review(
            packet=packet,
            report=ctx.report,
            model=skeptic_model or model,
        )
        ctx.n_model_calls += 1
        ctx.spend.add(usage)
        reviewed = _apply_independent_skeptic(ctx.report, review)
        report = replace(
            reviewed, model_calls=ctx.n_model_calls, chart_reads=ctx.chart_reads,
            spend=ctx.spend.report(), probes=tuple(ctx.probes),
            gate_rejections=tuple(ctx.submission_rejections[-20:]))
    return report
