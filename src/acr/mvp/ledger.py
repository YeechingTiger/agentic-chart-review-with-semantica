"""Semantica-backed Decision Intelligence for verified Decision Episodes.

Only reconstructed episodes are Semantica ``Decision`` nodes. Runtime testimony, ReAct cycles,
state snapshots, submissions, gates and results remain ordinary graph nodes. Causal edges are
created only from explicit, evidenced CausalAssertions; temporal adjacency is never promoted to
causation.  All primary reads are scoped by ``run_id + analysis_id`` and an append-only
AnalysisSelection is required before ``chain(run_id)`` chooses an analysis.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from acr.mvp.decision_receipts import receipt_from_action
from acr.mvp.decision_types import DECISION_SUBJECTS, DECISION_TAXONOMY_SCHEMA, subjects_for
from acr.mvp.task_presentation import content_hash


def _json_ready(value: Any) -> Any:
    """Normalize third-party analytics into the JSON contract exposed by this module."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_ready(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value

CAUSAL_TYPES = {"CAUSED", "INFLUENCED", "PRECEDENT_FOR"}
CONFIDENCE_SEMANTICS = "RECONSTRUCTION_STABILITY"
PROJECTION_SCHEMA = "acr.semantica_projection.v3"
ATOMIC_DECISION_IDENTITY_SCHEMA = "acr.atomic_decision_identity.v1"
_SAFE_ENTITY = re.compile(
    r"^(?:DecisionFunction|DecisionSubject|Field|Standing|EvidenceRole|ConflictShape|CandidateShape):"
    r"[a-z0-9_.-]+$")
_PATIENT_LOCATOR = re.compile(r"\bSYN[A-Za-z0-9_-]*\b", re.IGNORECASE)
_NOTE_LOCATOR = re.compile(
    r"\b(?:NOTE[_:-][A-Za-z0-9][A-Za-z0-9_.:-]*|"
    r"[A-Za-z][A-Za-z0-9-]*_(?:19|20)\d{2}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_DATE_LOCATOR = re.compile(
    r"\b(?:(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}|(?:19|20)\d{6})\b",
    re.IGNORECASE,
)
_FORBIDDEN_CORE = re.compile(
    rf"(?:{_PATIENT_LOCATOR.pattern}|{_NOTE_LOCATOR.pattern}|{_DATE_LOCATOR.pattern})",
    re.IGNORECASE,
)

_BASIS_LABELS = {
    "task_contract": "the supplied task contract",
    "method_card": "the supplied method card",
    "operational_instruction": "the run instructions",
    "precedent": "an earlier decision",
    "chart": "chart evidence",
    "own_knowledge": "the model's own judgment",
}


class DecisionIntelligence(Protocol):
    def project_analysis(self, artifact: dict[str, Any]) -> str: ...
    def selected_analysis(self, run_id: str) -> str | None: ...
    def chain(self, run_id: str, analysis_id: str | None = None) -> dict[str, Any]: ...
    def save(self) -> None: ...


class NullLedger:
    """In-memory projection sink for reconstruction tests; it invents no graph semantics."""

    def __init__(self) -> None:
        self.artifacts: list[dict[str, Any]] = []
        self.selections: dict[str, str] = {}

    def project_analysis(self, artifact: dict[str, Any]) -> str:
        identity = (artifact.get("run_id"), artifact.get("analysis_id"))
        prior = next((row for row in self.artifacts
                      if (row.get("run_id"), row.get("analysis_id")) == identity), None)
        if prior is not None:
            if prior.get("analysis_artifact_hash") != artifact.get("analysis_artifact_hash"):
                raise ValueError("analysis identity was reused for different content")
            return content_hash({"run_id": identity[0], "analysis_id": identity[1],
                                 "artifact_hash": artifact.get("analysis_artifact_hash")})
        self.artifacts.append(artifact)
        return content_hash({"run_id": identity[0], "analysis_id": identity[1],
                             "artifact_hash": artifact.get("analysis_artifact_hash")})

    def select_analysis(self, run_id: str, analysis_id: str, **_kwargs: Any) -> str:
        self.selections[run_id] = analysis_id
        return f"selection:{run_id}:{analysis_id}"

    def selected_analysis(self, run_id: str) -> str | None:
        return self.selections.get(run_id)

    def chain(self, run_id: str, analysis_id: str | None = None) -> dict[str, Any]:
        selected = analysis_id or self.selected_analysis(run_id)
        available = [row["analysis_id"] for row in self.artifacts if row["run_id"] == run_id]
        if selected is None:
            return {"run_id": run_id, "status": "NO_ANALYSIS_SELECTION",
                    "available_analysis_ids": available, "episodes": [], "causal_edges": []}
        artifact = next(row for row in self.artifacts
                        if row["run_id"] == run_id and row["analysis_id"] == selected)
        return {"run_id": run_id, "analysis_id": selected, "status": "OK",
                "episodes": artifact["episodes"],
                "causal_edges": artifact.get("causal_assertions") or [],
                "suggested_links": []}

    def save(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"analyses": len(self.artifacts),
                "episodes": sum(len(row.get("episodes") or []) for row in self.artifacts),
                "causal_edges": sum(len(row.get("causal_assertions") or [])
                                    for row in self.artifacts)}


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(value).strip().lower()).strip("_") or "unknown"


def _node_id(kind: str, identity: object) -> str:
    return f"acr:{_slug(kind)}:{content_hash(identity)[:24]}"


def _edge_dict(edge: Any) -> dict[str, Any]:
    return edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)


def _count_shape(values: Any) -> str:
    count = len(values or [])
    return "none" if count == 0 else "single" if count == 1 else "multiple"


class SemanticaLedger:
    """The sole ContextGraph adapter, including v0.6.6 persistence compatibility knowledge."""

    def __init__(self, path: Path) -> None:
        try:
            from semantica.context import ContextGraph
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "SemanticaLedger requires semantica==0.6.6 installed --no-deps and the "
                "project's [ledger] extra") from exc
        self.path = Path(path)
        self.graph = ContextGraph(
            extract_entities=False,
            extract_relationships=False,
            advanced_analytics=True,
            centrality_analysis=True,
            community_detection=True,
            node_embeddings=False,
        )
        if self.path.exists():
            self.graph.load_from_file(str(self.path))
            self._rehydrate_decision_indexes()

    # ------------------------------------------------------------------ released 0.6.6 seam
    def _rehydrate_decision_indexes(self) -> None:
        upstream = getattr(self.graph, "_rebuild_decision_indexes", None)
        if callable(upstream):  # delete the private fallback after a released upgrade passes
            upstream()
            return
        self.graph._decisions = {}
        self.graph._decision_index = defaultdict(set)
        self.graph._entity_index = defaultdict(set)
        self.graph._temporal_index = []
        entities: dict[str, list[str]] = defaultdict(list)
        for edge in self.graph.edges:
            row = _edge_dict(edge)
            if row.get("type") == "involves":
                entities[str(row["source_id"])].append(str(row["target_id"]))
        for node in self.graph.find_nodes(node_type="decision"):
            meta = dict(node.get("metadata") or {})
            known = ("category", "scenario", "reasoning", "outcome", "confidence",
                     "decision_maker", "timestamp", "valid_from", "valid_until")
            record = {"id": node["id"], **{key: meta.get(key) for key in known},
                      "entities": sorted(entities.get(str(node["id"]), [])),
                      "recorded_at": meta.get("recorded_at"),
                      "metadata": {key: value for key, value in meta.items()
                                   if key not in known}}
            self.graph._decisions[node["id"]] = record
            self.graph._decision_index[record["category"]].add(node["id"])
            for entity in record["entities"]:
                self.graph._entity_index[entity].add(node["id"])
            self.graph._temporal_index.append((node["id"], record["timestamp"] or 0))
        self.graph._temporal_index.sort(key=lambda row: row[1], reverse=True)

    # --------------------------------------------------------------------------- primitives
    def _find_nodes(self, node_type: str, *, run_id: str | None = None,
                    analysis_id: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for node in self.graph.find_nodes(node_type=node_type):
            meta = node.get("metadata") or {}
            if run_id is not None and meta.get("run_id") != run_id:
                continue
            if analysis_id is not None and meta.get("analysis_id") != analysis_id:
                continue
            rows.append(node)
        return rows

    def _ensure_node(self, node_id: str, node_type: str, *, content: str,
                     **metadata: Any) -> str:
        existing = self.graph.find_node(node_id)
        if existing is not None:
            if existing.get("type") != node_type:
                raise RuntimeError(f"node identity collision for {node_id}")
            return node_id
        self.graph.add_node(node_id, node_type, content=content, **metadata)
        node = self.graph.find_node(node_id)
        if node is None or node.get("type") != node_type:
            raise RuntimeError(f"Semantica silently skipped node {node_id}")
        return node_id

    def _has_edge(self, source: str, target: str, edge_type: str) -> bool:
        return any(row.get("source_id") == source and row.get("target_id") == target
                   and row.get("type") == edge_type
                   for row in map(_edge_dict, self.graph.edges))

    def _ensure_edge(self, source: str, target: str, edge_type: str, **props: Any) -> None:
        if self._has_edge(source, target, edge_type):
            return
        self.graph.add_edge(source_id=source, target_id=target,
                            edge_type=edge_type, **props)
        if not self._has_edge(source, target, edge_type):
            raise RuntimeError(f"Semantica silently skipped {edge_type} edge {source} -> {target}")

    def _projection_manifest_node(self, run_id: str, analysis_id: str) \
            -> dict[str, Any] | None:
        return next(iter(self._find_nodes("ProjectionManifest", run_id=run_id,
                                          analysis_id=analysis_id)), None)

    # -------------------------------------------------------------------- safe core payload
    def _cycles_for_episode(self, artifact: dict[str, Any], episode: dict[str, Any]) \
            -> list[dict[str, Any]]:
        wanted = set(episode.get("source_cycle_ids") or [])
        return [cycle for cycle in artifact.get("cycles") or []
                if cycle.get("cycle_id") in wanted]

    def _controlled_entities(self, artifact: dict[str, Any], episode: dict[str, Any]) -> list[str]:
        entities = [f"DecisionFunction:{_slug(episode.get('decision_function'))}",
                    f"DecisionSubject:{_slug(episode.get('decision_subject'))}"]
        fields: set[str] = set()
        standings: set[str] = set()
        roles: set[str] = set()
        cycles = self._cycles_for_episode(artifact, episode)
        before = (cycles[0].get("state_before") if cycles else {}) or {}
        declared = before.get("declared_state") or {}
        for finding in declared.get("findings") or []:
            if finding.get("field"):
                fields.add(_slug(finding["field"]))
            if finding.get("standing"):
                standings.add(_slug(finding["standing"]))
            if finding.get("assertion_class"):
                roles.add(_slug(finding["assertion_class"]))
        entities += [f"Field:{value}" for value in sorted(fields)]
        entities += [f"Standing:{value}" for value in sorted(standings)]
        entities += [f"EvidenceRole:{value}" for value in sorted(roles)]
        shape = "none" if not episode.get("candidate_set") else (
            "single" if len(episode["candidate_set"]) == 1 else "multiple")
        entities.append(f"CandidateShape:{shape}")
        conflict_shape = ("competing_candidates" if shape == "multiple" else
                          "single_candidate" if shape == "single" else "open_search")
        entities.append(f"ConflictShape:{conflict_shape}")
        return list(dict.fromkeys(entities))

    def _situation_signature(self, artifact: dict[str, Any],
                             episode: dict[str, Any]) -> dict[str, Any]:
        """A pre-decision, locator-free signature used by Semantica retrieval."""
        cycles = self._cycles_for_episode(artifact, episode)
        first = cycles[0] if cycles else {}
        before = first.get("state_before") or {}
        observed = before.get("observed_state") or {}
        declared = before.get("declared_state") or {}
        findings = declared.get("findings") or []
        candidate_shape = _count_shape(episode.get("candidate_set"))
        evidence_roles = sorted({_slug(row.get("assertion_class")) for row in findings
                                 if row.get("assertion_class")})
        standings = sorted({_slug(row.get("standing")) for row in findings
                            if row.get("standing")})
        fields = sorted({_slug(row.get("field")) for row in findings if row.get("field")})
        return {
            "schema": "acr.decision_situation_signature.v1",
            "decision_function": _slug(episode.get("decision_function")),
            "decision_subject": _slug(episode.get("decision_subject")),
            "candidate_shape": candidate_shape,
            "conflict_shape": ("competing_candidates" if candidate_shape == "multiple" else
                               "single_candidate" if candidate_shape == "single" else
                               "open_search"),
            "surfaced_shape": _count_shape(observed.get("surfaced_notes")),
            "read_shape": _count_shape(observed.get("read_notes")),
            "finding_shape": _count_shape(findings),
            "uncertainty_shape": _count_shape(declared.get("uncertainties")),
            "fields": fields or ["none"],
            "evidence_roles": evidence_roles or ["none"],
            "standings": standings or ["none"],
        }

    def _runtime_decision_args(self, artifact: dict[str, Any],
                               episode: dict[str, Any]) -> dict[str, Any]:
        """Return the testimony attached to the atomic choice, when one exists."""
        rows = [
            action.get("args") or {}
            for cycle in self._cycles_for_episode(artifact, episode)
            for action in cycle.get("actions") or []
            if action.get("tool") in {"note_decision", "record_finding"}
        ]
        return dict(rows[-1]) if rows else {}

    def _known_note_locators(self, artifact: dict[str, Any],
                             episode: dict[str, Any]) -> set[str]:
        """Collect structured note locators so readable core text can redact them exactly."""
        locators: set[str] = set()
        for cycle in self._cycles_for_episode(artifact, episode):
            for state_name in ("state_before", "state_after"):
                state = cycle.get(state_name) or {}
                observed = state.get("observed_state") or {}
                locators.update(str(value) for key in ("surfaced_notes", "read_notes")
                                for value in observed.get(key) or [] if value)
                declared = state.get("declared_state") or {}
                locators.update(str(row["note_id"])
                                for row in declared.get("findings") or []
                                if row.get("note_id"))
            for action in cycle.get("actions") or []:
                note_id = (action.get("args") or {}).get("note_id")
                if note_id:
                    locators.add(str(note_id))
        return locators

    def _deidentify_core_text(self, artifact: dict[str, Any], episode: dict[str, Any],
                              value: Any) -> str:
        """Redact structured chart locators while retaining a readable clinical question."""
        text = " ".join(str(value or "").split())
        for locator in sorted(self._known_note_locators(artifact, episode),
                              key=len, reverse=True):
            text = re.sub(re.escape(locator), "[note]", text, flags=re.IGNORECASE)
        text = _NOTE_LOCATOR.sub("[note]", text)
        text = _PATIENT_LOCATOR.sub("[patient]", text)
        text = _DATE_LOCATOR.sub("[date]", text)
        return " ".join(text.split())

    @staticmethod
    def _join_labels(values: list[str]) -> str:
        if len(values) < 2:
            return values[0] if values else ""
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    def _human_scenario(self, artifact: dict[str, Any], episode: dict[str, Any]) -> str:
        testimony = self._runtime_decision_args(artifact, episode)
        scenario = (testimony.get("facing") or episode.get("material_question")
                    or episode.get("scenario"))
        if not scenario:
            scenario = {
                "where_to_look": "Which chart evidence should be reviewed next?",
                "is_this_it": "Does the current evidence describe the requested concept?",
                "what_it_asserts": "What does the current evidence actually assert?",
                "when_it_happened": "Which time does the current evidence establish?",
                "standing": "Can the current evidence establish the requested field?",
                "same_or_ordered": "Are these evidence items the same event, and which came first?",
                "corroborate": "Do these evidence items independently reinforce one another?",
                "which_wins": "Which conflicting evidence should control the answer?",
                "scope": "Is this case and time period within scope?",
                "infer": "Does the available evidence support this case-level inference?",
                "is_it_absent": "What does the unsuccessful search establish?",
                "enough": "Is there enough evidence to stop reviewing?",
                "what_to_answer": "What result should be submitted?",
            }.get(str(episode.get("decision_function") or ""),
                  "What material choice should be made here?")
        return self._deidentify_core_text(artifact, episode, scenario)

    def _human_reasoning(self, artifact: dict[str, Any], episode: dict[str, Any]) -> str:
        testimony = self._runtime_decision_args(artifact, episode)
        rationale = (testimony.get("because") or episode.get("decision_rationale")
                     or episode.get("model_interpretation"))
        text = self._deidentify_core_text(artifact, episode, rationale)
        if not text:
            text = "No explicit rationale was recorded"
        if text[-1] not in ".!?":
            text += "."

        basis = [str(value) for value in testimony.get("basis_sources") or []]
        if not basis:
            basis = [str(value) for value in episode.get("claimed_basis_summary") or []]
        labels = [_BASIS_LABELS[value] for value in dict.fromkeys(basis)
                  if value in _BASIS_LABELS]
        if labels:
            text += f" Basis used: {self._join_labels(labels)}."
        return text

    @staticmethod
    def _identity_text(value: Any) -> str:
        """Normalize exact-match text without turning paraphrases into false identity."""
        return " ".join(str(value or "").casefold().split())

    def _atomic_decision_identity(self, artifact: dict[str, Any],
                                  episode: dict[str, Any]) -> dict[str, Any]:
        """Identify one exact question over one exact evidence set, excluding its outcome.

        The hashes deliberately omit run, analysis, model and chosen outcome. Repeated runs that
        considered the same evidence and answered the same material question therefore share a
        decision point even when they disagree. Coarse state *shape* remains a separate retrieval
        feature and is never evidence of exact identity.
        """
        testimony = self._runtime_decision_args(artifact, episode)
        question = (testimony.get("facing") or episode.get("material_question")
                    or episode.get("scenario") or "")
        anchors: set[tuple[str, str]] = set()
        direct_evidence_items: set[tuple[str, str]] = set()
        decision_function = _slug(episode.get("decision_function"))
        decision_subject = _slug(episode.get("decision_subject"))

        # The reconstructed scenario and exact pre-choice chart state describe what was
        # actually in view. Candidate options are deliberately excluded: they are part of the
        # reconstructed choice, not the evidence, and may differ when two runs disagree.
        if episode.get("scenario"):
            anchors.add(("scenario_context", self._identity_text(episode["scenario"])))
        cycles = self._cycles_for_episode(artifact, episode)
        choice_before = (cycles[0].get("state_before") if cycles else {}) or {}
        observed = choice_before.get("observed_state") or {}
        for state_key, anchor_kind in (
                ("surfaced_notes", "surfaced_note"), ("read_notes", "read_note")):
            for note_id in observed.get(state_key) or []:
                anchors.add((anchor_kind, self._identity_text(note_id)))
        declared = choice_before.get("declared_state") or {}
        for finding in declared.get("findings") or []:
            material = {key: finding.get(key) for key in (
                "note_id", "field", "event_time", "source_start", "source_end",
                "assertion_class",
            ) if finding.get(key) is not None}
            anchors.add(("finding", content_hash(material)))
        for cycle in cycles:
            for action in cycle.get("actions") or []:
                args = action.get("args") or {}
                if args.get("note_id"):
                    direct_evidence_items.add((self._identity_text(args["note_id"]),
                                               self._identity_text(args.get("field"))))
                material = {key: args.get(key) for key in (
                    "note_id", "field", "event_time", "source_start", "source_end",
                    "assertion_class",
                ) if args.get(key) is not None}
                if material:
                    anchors.add(("evidence_item", content_hash(material)))
                for ref in args.get("cited_refs") or []:
                    normalized = self._identity_text(ref)
                    if normalized.startswith(("note:", "evidence:", "finding:")):
                        anchors.add(("evidence_ref", normalized))

        if decision_subject == "evidence_item" and direct_evidence_items:
            # For a note-level judgment, the stable question is the taxonomy function applied
            # to one exact note+field. Free-form ``facing`` wording, cited span boundaries and
            # assertion class are deliberately excluded: repeated agents can phrase or locate
            # the same audit question differently, and assertion class may itself be disputed.
            identity_basis = "EXACT_EVIDENCE_AND_SEMANTIC_QUESTION"
            question_hash = content_hash({
                "decision_function": decision_function,
                "decision_subject": decision_subject,
                "fields": sorted({field for _, field in direct_evidence_items}),
            })
            evidence_hash = content_hash([
                {"note_locator_hash": content_hash(note_id), "field": field}
                for note_id, field in sorted(direct_evidence_items)
            ])
            evidence_anchor_count = len(direct_evidence_items)
        else:
            identity_basis = "EXACT_QUESTION_AND_EVIDENCE"
            question_hash = content_hash(self._identity_text(question))
            evidence_hash = content_hash(sorted(anchors))
            evidence_anchor_count = len(anchors)
        identity_material = {
            "schema": ATOMIC_DECISION_IDENTITY_SCHEMA,
            "identity_basis": identity_basis,
            "decision_function": decision_function,
            "decision_subject": decision_subject,
            "question_hash": question_hash,
            "evidence_hash": evidence_hash,
        }
        return {
            "schema": ATOMIC_DECISION_IDENTITY_SCHEMA,
            "identity_basis": identity_basis,
            "decision_function": identity_material["decision_function"],
            "decision_subject": identity_material["decision_subject"],
            "question_hash": question_hash,
            "evidence_hash": evidence_hash,
            "evidence_anchor_count": evidence_anchor_count,
            "decision_point_hash": content_hash(identity_material),
        }

    def _safe_outcome(self, artifact: dict[str, Any], episode: dict[str, Any]) -> str:
        function = str(episode.get("decision_function") or "other")
        cycles = self._cycles_for_episode(artifact, episode)
        if function == "where_to_look":
            tools = [str(action.get("tool") or "") for cycle in cycles
                     for action in cycle.get("actions") or []
                     if action.get("tool") not in {"note_decision", "record_finding"}]
            if tools:
                return f"ACTION:{_slug(tools[0]).upper()}"
        if function == "what_to_answer":
            for cycle in cycles:
                for action in cycle.get("actions") or []:
                    if action.get("tool") == "submit_answer":
                        status = _slug((action.get("args") or {}).get("status")).upper()
                        return f"SUBMIT_STATUS:{status}"
        closed = {"can_establish", "merely_mentions", "neither", "enough", "not_enough",
                  "absent_in_chart", "absent_from_corpus", "found"}
        decision_slug = _slug(episode.get("decision"))
        if decision_slug in closed:
            return decision_slug.upper()
        candidates = list(episode.get("candidate_set") or [])
        if episode.get("decision") in candidates:
            return f"SELECT_CANDIDATE:c{candidates.index(episode['decision']) + 1}"
        return f"CHOICE_HASH:{content_hash(str(episode.get('decision') or ''))[:12]}"

    def _episode_projection(self, artifact: dict[str, Any], episode: dict[str, Any]) \
            -> dict[str, Any]:
        identity = {"schema": PROJECTION_SCHEMA, "run_id": artifact["run_id"],
                    "analysis_id": artifact["analysis_id"],
                    "episode_id": episode["episode_id"]}
        core = {"category": str(episode["decision_function"]),
                "scenario": self._human_scenario(artifact, episode),
                "reasoning": self._human_reasoning(artifact, episode),
                "outcome": self._safe_outcome(artifact, episode),
                "confidence": float(episode.get("reconstruction_stability",
                                                 artifact.get("reconstruction_stability", 0.0))),
                "entities": self._controlled_entities(artifact, episode)}
        signature = self._situation_signature(artifact, episode)
        atomic_identity = self._atomic_decision_identity(artifact, episode)
        return {"acr_node_key": content_hash({**identity, **core}), **identity, **core,
                "situation_signature": signature,
                "situation_signature_hash": content_hash(signature),
                "atomic_decision_identity": atomic_identity,
                "decision_point_hash": atomic_identity["decision_point_hash"]}

    def _desired_projection_hash(self, artifact: dict[str, Any]) -> str:
        episodes = [self._episode_projection(artifact, episode)
                    for episode in artifact.get("episodes") or []]
        triples = sorted((row["source_episode_id"], row["target_episode_id"],
                          row["relationship_type"])
                         for row in artifact.get("causal_assertions") or [])
        return content_hash({"schema": PROJECTION_SCHEMA,
                             "taxonomy_version": artifact.get(
                                 "taxonomy_version", DECISION_TAXONOMY_SCHEMA),
                             "analysis_artifact_hash": artifact["analysis_artifact_hash"],
                             "task_presentation_hash": artifact.get("task_presentation_hash"),
                             "episodes": episodes,
                             "causal_triples": triples})

    def _project_task_contract_policy(self, run_dir: Path,
                                      artifact: dict[str, Any]) \
            -> dict[str, Any] | None:
        """Store the governing contract with Semantica PolicyEngine when locally available."""
        presentation_path = Path(run_dir) / "task_presentation.json"
        if not presentation_path.exists():
            return None
        from acr.mvp.task_presentation import ContractSnapshot

        snapshot = ContractSnapshot.from_path(presentation_path)
        if snapshot.presentation_hash != artifact.get("task_presentation_hash"):
            raise ValueError("Task Presentation does not match the projected analysis")
        presentation = snapshot.to_dict()
        # The bare experimental arm intentionally withholds the clinical guideline.
        # Output-shape constraints remain part of the task prompt, but must not be
        # misrepresented as a clinical Semantica Policy applied to each decision.
        if presentation.get("arm_id") == "task_only":
            return None
        bundle = presentation.get("policy_bundle")
        if bundle:
            body = {key: value for key, value in bundle.items() if key != "bundle_hash"}
            if content_hash(body) != bundle.get("bundle_hash"):
                raise ValueError("Policy Bundle content hash does not match its payload")
            bundle_node_id = self._ensure_node(
                _node_id("policy_bundle", {
                    "bundle_id": bundle.get("bundle_id"),
                    "bundle_hash": bundle.get("bundle_hash"),
                }),
                "PolicyBundle", content=str(bundle.get("bundle_id") or "policy bundle"),
                bundle_id=bundle.get("bundle_id"),
                bundle_version=bundle.get("bundle_version"),
                bundle_hash=bundle.get("bundle_hash"),
                task_contract_content_hash=bundle.get("task_contract_content_hash"),
            )
            from semantica.context import PolicyEngine
            from semantica.context.decision_models import Policy

            engine = PolicyEngine(self.graph)
            clause_to_policy: dict[str, dict[str, str]] = {}
            now = datetime.now(UTC)
            for component in bundle.get("policies") or []:
                material = {key: component.get(key) for key in (
                    "policy_id", "category", "authority", "clause_refs", "rules")}
                if content_hash(material) != component.get("content_hash"):
                    raise ValueError(
                        f"Policy component hash mismatch for {component.get('policy_id')!r}")
                policy_id = str(component.get("policy_id") or "")
                version = str(component.get("version") or "")
                if not policy_id or not version:
                    raise ValueError("Policy Bundle component lacks policy_id/version")
                policy_node_id = f"{policy_id}:{version}"
                if self.graph.find_node(policy_node_id) is None:
                    engine.add_policy(Policy(
                        policy_id=policy_id, name=policy_id,
                        description="Atomic chart-review decision boundary.",
                        rules={"clauses": component.get("rules") or []},
                        category=str(component.get("category") or "chart_review"),
                        version=version, created_at=now, updated_at=now,
                        metadata={
                            "authority": component.get("authority"),
                            "clause_refs": component.get("clause_refs") or [],
                            "content_hash": component.get("content_hash"),
                            "bundle_hash": bundle.get("bundle_hash"),
                            "automatic_compliance_supported": False,
                        },
                    ))
                self._ensure_edge(
                    bundle_node_id, policy_node_id, "COMPOSED_OF",
                    component_content_hash=component.get("content_hash"))
                for ref in component.get("clause_refs") or []:
                    ref = str(ref)
                    if ref in clause_to_policy:
                        raise ValueError(f"Policy Bundle clause has multiple owners: {ref}")
                    clause_to_policy[ref] = {
                        "policy_id": policy_id, "version": version,
                        "node_id": policy_node_id,
                    }
            return {
                "mode": "policy_bundle", "presentation": presentation,
                "bundle_node_id": bundle_node_id,
                "clause_to_policy": clause_to_policy,
            }

        contract = presentation.get("task_contract_ref") or {}
        policy_id = str(contract.get("id") or "")
        version = str(contract.get("version") or "")
        if not policy_id or not version:
            raise ValueError("Task Presentation lacks a versioned Task Contract reference")
        policy_node_id = f"{policy_id}:{version}"
        catalog = presentation.get("known_clause_index") or []
        catalog_hash = content_hash(catalog)
        existing = self.graph.find_node(policy_node_id)
        if existing is None:
            from semantica.context import PolicyEngine
            from semantica.context.decision_models import Policy

            now = datetime.now(UTC)
            policy = Policy(
                policy_id=policy_id,
                name=policy_id,
                description=("Versioned semantic Task Contract governing chart-review "
                             "evaluation; clinical compliance requires domain adjudication."),
                rules={"catalog_hash": catalog_hash,
                       "automatic_compliance_supported": False},
                category="chart_review_task_contract", version=version,
                created_at=now, updated_at=now,
                metadata={
                    "task_contract_content_hash": contract.get("content_hash"),
                    "known_clause_index": catalog,
                    "known_clause_catalog_hash": catalog_hash,
                    "automatic_compliance_supported": False,
                    "compliance_semantics": "HUMAN_OR_DOMAIN_ENGINE_REQUIRED",
                },
            )
            PolicyEngine(self.graph).add_policy(policy)
            existing = self.graph.find_node(policy_node_id)
        if existing is None or existing.get("type") != "Policy":
            raise RuntimeError("Semantica failed to persist the Task Contract Policy")
        metadata = existing.get("metadata") or {}
        inner = metadata.get("metadata") or {}
        if (metadata.get("policy_id") != policy_id or metadata.get("version") != version
                or inner.get("known_clause_catalog_hash") != catalog_hash):
            raise ValueError("Task Contract Policy identity collides with different content")
        return {"mode": "whole_contract", "presentation": presentation,
                "policy_node_id": policy_node_id}

    @staticmethod
    def _episode_claimed_policy_refs(artifact: dict[str, Any],
                                     episode: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        wanted = set(episode.get("source_cycle_ids") or [])
        for cycle in (row for row in artifact.get("cycles") or []
                      if row.get("cycle_id") in wanted):
            for action in cycle.get("actions") or []:
                if action.get("tool") not in {"note_decision", "record_finding"}:
                    continue
                args = action.get("args") or {}
                refs.update(str(value) for value in args.get("cited_refs") or [])
                refs.update(str(value) for value in
                            args.get("checked_discriminating_fact_refs") or [])
        return refs

    # -------------------------------------------------------------------------- projection
    @staticmethod
    def _validate_atomic_artifact(artifact: dict[str, Any]) -> None:
        """Reject compound episodes before any Semantica state is mutated."""
        annotations = artifact.get("cycle_annotations")
        if not isinstance(annotations, dict):
            raise ValueError("analysis artifact needs cycle annotations for atomic decisions")
        cycles_by_id = {str(row.get("cycle_id")): row
                        for row in artifact.get("cycles") or []}
        known_cycles = set(cycles_by_id)
        for cycle_id, cycle in cycles_by_id.items():
            commits_finding = any(
                action.get("tool") == "record_finding" and action.get("ok") is True
                for action in cycle.get("actions") or [])
            annotation = annotations.get(cycle_id) or {}
            if commits_finding and (
                    annotation.get("role") != "DECISION_BEARING"
                    or annotation.get("decision_function") != "standing"):
                raise ValueError(
                    "each successful record_finding must project as a separate standing decision")
        for episode in artifact.get("episodes") or []:
            source_ids = [str(value) for value in episode.get("source_cycle_ids") or []]
            if not source_ids or any(value not in known_cycles for value in source_ids):
                raise ValueError("Decision Episode contains an unknown or empty cycle set")
            roles = [((annotations.get(value) or {}).get("role")) for value in source_ids]
            if roles.count("DECISION_BEARING") != 1:
                raise ValueError(
                    "each Semantica Decision Episode must contain exactly one atomic decision")
            if roles[0] != "DECISION_BEARING":
                raise ValueError(
                    "each Semantica Decision Episode must begin with its atomic decision")
            function = str(episode.get("decision_function") or "")
            if (annotations.get(source_ids[0]) or {}).get("decision_function") != function:
                raise ValueError("episode function disagrees with its atomic decision cycle")
            subject = str(episode.get("decision_subject") or "")
            if subject not in DECISION_SUBJECTS or subject not in subjects_for(function):
                raise ValueError(
                    f"decision subject {subject!r} is not valid for {function!r}")

    def project_analysis(self, artifact: dict[str, Any]) -> str:
        required = {"run_id", "analysis_id", "analysis_artifact_hash", "trace_manifest_hash",
                    "episodes", "cycles", "causal_assertions"}
        missing = sorted(required - set(artifact))
        if missing:
            raise ValueError(f"analysis artifact lacks required fields: {missing}")
        supplied_hash = str(artifact["analysis_artifact_hash"])
        unhashed = {key: value for key, value in artifact.items()
                    if key not in {"analysis_artifact_hash", "artifact_ref"}}
        if content_hash(unhashed) != supplied_hash:
            raise ValueError("analysis artifact hash does not match its content")
        self._validate_atomic_artifact(artifact)
        run_id, analysis_id = str(artifact["run_id"]), str(artifact["analysis_id"])
        from acr.mvp.semantica_audit import audit_location, bundle_id_for

        provenance_path, run_dir = audit_location(self.path, artifact)
        narrative_bundle_id = bundle_id_for(artifact)
        projection_hash = self._desired_projection_hash(artifact)
        existing = self._projection_manifest_node(run_id, analysis_id)
        if existing is not None:
            if (existing.get("metadata") or {}).get("projection_hash") != projection_hash:
                raise ValueError("analysis identity was already projected with different content")
            self._project_audit_narratives(
                artifact, provenance_path=provenance_path, run_dir=run_dir)
            return projection_hash

        triples = [(row["source_episode_id"], row["target_episode_id"],
                    row["relationship_type"])
                   for row in artifact.get("causal_assertions") or []]
        if len(set(triples)) != len(triples):
            raise ValueError("duplicate causal triple in one analysis is not permitted")

        analysis_node = self._ensure_node(
            _node_id("analysis", {"run": run_id, "analysis": analysis_id}),
            "AnalysisArtifact", content="decision episode analysis pointer",
            run_id=run_id, analysis_id=analysis_id,
            artifact_hash=supplied_hash, artifact_ref=artifact.get("artifact_ref"),
            trace_manifest_hash=artifact["trace_manifest_hash"],
            task_presentation_hash=artifact.get("task_presentation_hash"),
            analysis_schema=artifact.get("schema"),
            taxonomy_version=artifact.get("taxonomy_version", DECISION_TAXONOMY_SCHEMA),
            supported_runtime_receipt_schema=artifact.get(
                "supported_runtime_receipt_schema"),
            runtime_receipt_mode=(artifact.get("runtime_receipt_manifest") or {}).get("mode"),
            sealed_receipt_count=(artifact.get("runtime_receipt_manifest") or {}).get(
                "sealed_receipt_count", 0),
            decision_receipt_coverage_status=(artifact.get(
                "decision_receipt_coverage") or {}).get("status"),
            sealed_decision_count=(artifact.get("decision_receipt_coverage") or {}).get(
                "sealed_episode_count", 0),
            stability_status=artifact.get("stability_status"),
            reconstruction_stability=artifact.get("reconstruction_stability"),
            reconstructor_requested_model=(artifact.get("reconstructor_call") or {}).get(
                "requested_model"),
            reconstructor_resolved_model=(artifact.get("reconstructor_call") or {}).get(
                "resolved_model"),
            reconstructor_response_provider=(artifact.get("reconstructor_call") or {}).get(
                "response_provider"),
            reconstructor_response_id=(artifact.get("reconstructor_call") or {}).get(
                "response_id"),
            reconstructor_identity_status=(artifact.get("reconstructor_call") or {}).get(
                "identity_status"),
            provenance_path=str(provenance_path),
            decision_narrative_bundle_id=narrative_bundle_id,
            audit_schema="acr.semantica_decision_audit.v1",
        )
        policy_binding = self._project_task_contract_policy(run_dir, artifact)

        cycle_nodes: dict[str, str] = {}
        testimony_nodes: dict[str, str] = {}
        execution_nodes: dict[str, str] = {}
        reference_nodes_by_cycle: dict[str, list[tuple[str, str]]] = defaultdict(list)
        last_gate_node: str | None = None
        for cycle in artifact.get("cycles") or []:
            cycle_id = str(cycle["cycle_id"])
            result_by_event = {row.get("event_ref"): row.get("result") or {}
                               for row in cycle.get("observations") or []}
            cycle_node = self._ensure_node(
                _node_id("cycle", {"run": run_id, "analysis": analysis_id, "cycle": cycle_id}),
                "ReActCycle", content="Langtrace cycle pointer", run_id=run_id,
                analysis_id=analysis_id, cycle_id=cycle_id,
                source_seq_start=(cycle.get("source_seq_range") or [None])[0],
                source_event_time=cycle.get("source_event_time"),
                artifact_ref=artifact.get("artifact_ref"),
            )
            cycle_nodes[cycle_id] = cycle_node
            for state_name in ("state_before", "state_after"):
                state = cycle.get(state_name) or {}
                state_node = self._ensure_node(
                    _node_id("state", {"run": run_id, "analysis": analysis_id,
                                       "cycle": cycle_id, "state": state_name}),
                    "StateSnapshot", content="state snapshot pointer", run_id=run_id,
                    analysis_id=analysis_id, cycle_id=cycle_id, state_kind=state_name,
                    state_hash=content_hash(state), artifact_ref=artifact.get("artifact_ref"),
                )
                self._ensure_edge(cycle_node, state_node,
                                  "STARTED_WITH" if state_name == "state_before" else "ENDED_WITH",
                                  provenance="DETERMINISTIC_DERIVED")
            for testimony_ref in cycle.get("decision_testimony_refs") or []:
                ref = str(testimony_ref)
                receipt = next((
                    receipt_from_action(action, result_by_event.get(action.get("event_ref"), {}))
                    for action in cycle.get("actions") or []
                    if result_by_event.get(action.get("event_ref"), {}).get("decision_receipt")
                ), None)
                testimony_node = self._ensure_node(
                    _node_id("testimony", {"run": run_id, "analysis": analysis_id, "ref": ref}),
                    "DecisionTestimony", content=("server-sealed runtime testimony pointer"
                                                  if receipt else "legacy runtime testimony pointer"),
                    run_id=run_id,
                    analysis_id=analysis_id, testimony_ref=ref,
                    receipt_ref=(receipt or {}).get("receipt_id"),
                    receipt_schema=(receipt or {}).get("schema"),
                    receipt_hash=(receipt or {}).get("receipt_hash"),
                    receipt_provenance=(receipt or {}).get("provenance"),
                    artifact_ref=artifact.get("artifact_ref"),
                    testimony_provenance="SELF_REPORTED",
                    seal_provenance=("SERVER_FACT" if receipt else None),
                )
                testimony_nodes[ref] = testimony_node
                self._ensure_edge(testimony_node, cycle_node, "OCCURRED_IN",
                                  provenance="SERVER_FACT")
            for action in cycle.get("actions") or []:
                tool = action.get("tool")
                node_type = {"submit_answer": "Submission"}.get(str(tool))
                if node_type:
                    exec_node = self._ensure_node(
                        _node_id(node_type, {"run": run_id, "analysis": analysis_id,
                                            "cycle": cycle_id}),
                        node_type, content=f"{node_type.lower()} pointer", run_id=run_id,
                        analysis_id=analysis_id, cycle_id=cycle_id,
                        artifact_ref=artifact.get("artifact_ref"), provenance="SERVER_FACT")
                    execution_nodes[cycle_id] = exec_node
                    self._ensure_edge(cycle_node, exec_node, "CONTAINS_EXECUTION",
                                      provenance="SERVER_FACT")
                    result = result_by_event.get(action.get("event_ref"), {})
                    gate_node = self._ensure_node(
                        _node_id("gate", {"run": run_id, "analysis": analysis_id,
                                          "cycle": cycle_id}),
                        "GateVerdict", content="gate verdict pointer", run_id=run_id,
                        analysis_id=analysis_id, cycle_id=cycle_id,
                        accepted=bool(result.get("accepted")),
                        artifact_ref=artifact.get("artifact_ref"), provenance="SERVER_FACT")
                    self._ensure_edge(exec_node, gate_node, "EVALUATED_AS",
                                      provenance="SERVER_FACT")
                    last_gate_node = gate_node
                if tool == "record_finding":
                    args = action.get("args") or {}
                    finding_node = self._ensure_node(
                        _node_id("runtime_finding", {"run": run_id, "analysis": analysis_id,
                                                      "cycle": cycle_id}),
                        "RuntimeNoteFinding", content="runtime finding pointer", run_id=run_id,
                        analysis_id=analysis_id, cycle_id=cycle_id,
                        field=_slug(args.get("field")), standing=_slug(args.get("standing")),
                        assertion_class=_slug(args.get("assertion_class")),
                        span_resolved=bool((result_by_event.get(action.get("event_ref"), {})
                                            .get("server_fact") or {}).get("span_resolved")),
                        artifact_ref=artifact.get("artifact_ref"),
                        standing_provenance="SELF_REPORTED",
                        span_provenance="SERVER_FACT")
                    self._ensure_edge(cycle_node, finding_node, "CONTAINS_FINDING",
                                      provenance="SERVER_FACT")
                    testimony_ref = (result_by_event.get(action.get("event_ref"), {})
                                     .get("testimony_ref")
                                     or args.get("decision_testimony_ref"))
                    if str(testimony_ref) in testimony_nodes:
                        self._ensure_edge(testimony_nodes[str(testimony_ref)], finding_node,
                                          "TESTIFIES_TO", provenance="SELF_REPORTED")
                if tool == "record_evidence":
                    evidence_node = self._ensure_node(
                        _node_id("evidence_ref", {"run": run_id, "analysis": analysis_id,
                                                  "cycle": cycle_id}),
                        "EvidenceRef", content="evidence span pointer", run_id=run_id,
                        analysis_id=analysis_id, cycle_id=cycle_id,
                        locator_hash=content_hash({"cycle": cycle_id,
                                                   "event": action.get("event_ref")}),
                        artifact_ref=artifact.get("artifact_ref"), provenance="SERVER_FACT")
                    reference_nodes_by_cycle[cycle_id].append(("USES", evidence_node))
                    self._ensure_edge(cycle_node, evidence_node, "CONTAINS_EVIDENCE",
                                      provenance="SERVER_FACT")
                if tool in {"note_decision", "record_finding"}:
                    result = result_by_event.get(action.get("event_ref"), {})
                    testimony_ref = next(iter(cycle.get("decision_testimony_refs") or []), None)
                    testimony_node = testimony_nodes.get(str(testimony_ref))
                    for resolved in [*(result.get("citation_resolutions") or []),
                                     *(result.get("checked_fact_resolutions") or [])]:
                        ref = str(resolved.get("ref") or "")
                        is_fact = ref.startswith("discriminating_fact.")
                        is_rule = ref.startswith(("decision_rule.", "conflict_rule.",
                                                  "evidence_rule.", "answer_check.",
                                                  "field_", "abstention.",
                                                  "proof_obligation."))
                        node_type = "DiscriminatingFact" if is_fact else (
                            "Rule" if is_rule else "EvidenceRef")
                        metadata = {"reference_status": resolved.get("status"),
                                    "locator_hash": content_hash(ref)}
                        if is_fact or is_rule:
                            metadata["catalog_ref"] = ref
                        reference_node = self._ensure_node(
                            _node_id(node_type, {"run": run_id, "analysis": analysis_id,
                                                "ref_hash": content_hash(ref)}),
                            node_type, content=f"{node_type.lower()} pointer", run_id=run_id,
                            analysis_id=analysis_id, artifact_ref=artifact.get("artifact_ref"),
                            **metadata)
                        relation = "CHECKED" if is_fact else "CITES"
                        reference_nodes_by_cycle[cycle_id].append((relation, reference_node))
                        if testimony_node:
                            self._ensure_edge(testimony_node, reference_node, relation,
                                              provenance="SELF_REPORTED")
            if cycle.get("structural_kind") == "TERMINATION":
                observed = (cycle.get("state_after") or {}).get("observed_state") or {}
                result_node = self._ensure_node(
                    _node_id("run_result", {"run": run_id, "analysis": analysis_id,
                                             "cycle": cycle_id}),
                    "RunResult", content="run result pointer", run_id=run_id,
                    analysis_id=analysis_id, cycle_id=cycle_id,
                    status=_slug(observed.get("result_status")),
                    artifact_ref=artifact.get("artifact_ref"), provenance="SERVER_FACT")
                self._ensure_edge(cycle_node, result_node, "CONTAINS_EXECUTION",
                                  provenance="SERVER_FACT")
                if last_gate_node is not None:
                    self._ensure_edge(last_gate_node, result_node, "PRODUCED",
                                      provenance="SERVER_FACT")

        episode_nodes: dict[str, str] = {}
        projections: dict[str, dict[str, Any]] = {}
        for episode in artifact.get("episodes") or []:
            projection = self._episode_projection(artifact, episode)
            projections[str(episode["episode_id"])] = projection
            prior = next((node for node in self.graph.find_nodes(node_type="decision")
                          if (node.get("metadata") or {}).get("acr_node_key")
                          == projection["acr_node_key"]), None)
            if prior is not None:
                decision_id = str(prior["id"])
            else:
                decision_id = self.graph.record_decision(
                    category=projection["category"], scenario=projection["scenario"],
                    reasoning=projection["reasoning"], outcome=projection["outcome"],
                    confidence=projection["confidence"], entities=projection["entities"],
                    decision_maker=str(artifact.get("review_agent_identity")
                                       or "chart-review-agent"),
                    metadata={
                        "run_id": run_id, "analysis_id": analysis_id,
                        "acr_episode_id": episode["episode_id"],
                        "audit_unit": "ATOMIC_DECISION",
                        "decision_subject": episode["decision_subject"],
                        "acr_node_key": projection["acr_node_key"],
                        "projection_schema": PROJECTION_SCHEMA,
                        "taxonomy_version": artifact.get(
                            "taxonomy_version", DECISION_TAXONOMY_SCHEMA),
                        "runtime_receipt_ref": episode.get("runtime_receipt_ref"),
                        "runtime_receipt_hash": episode.get("runtime_receipt_hash"),
                        "runtime_receipt_provenance": episode.get(
                            "runtime_receipt_provenance"),
                        "projection_payload_hash": content_hash(projection),
                        "situation_signature": projection["situation_signature"],
                        "situation_signature_hash": projection["situation_signature_hash"],
                        "atomic_decision_identity": projection[
                            "atomic_decision_identity"],
                        "decision_point_hash": projection["decision_point_hash"],
                        "analysis_artifact_hash": supplied_hash,
                        "artifact_ref": artifact.get("artifact_ref"),
                        "task_presentation_hash": artifact.get("task_presentation_hash"),
                        # Full cycle identities live in the content-addressed artifact and
                        # DERIVED_FROM edges. Semantica metadata values are capped at 1000
                        # characters, so retain a verifiable compact pointer here.
                        "source_cycle_count": len(episode.get("source_cycle_ids") or []),
                        "source_cycle_ids_hash": content_hash(
                            episode.get("source_cycle_ids") or []),
                        "source_seq_start": min(
                            [int(ref.split(":", 1)[1]) for ref in episode.get("source_event_ids") or []
                             if str(ref).startswith("layer1:")] or [0]),
                        "source_event_time": next((
                            cycle.get("source_event_time")
                            for cycle in self._cycles_for_episode(artifact, episode)), None),
                        "field_provenance_hash": content_hash(
                            episode.get("field_provenance") or {}),
                        "confidence_semantics": CONFIDENCE_SEMANTICS,
                        "stability_status": episode.get("stability_status"),
                    },
                )
            episode_nodes[str(episode["episode_id"])] = decision_id
            self._ensure_edge(analysis_node, decision_id, "CONTAINS_EPISODE",
                              provenance="MODEL_RECONSTRUCTED")
            if policy_binding is not None:
                presentation = policy_binding["presentation"]
                if policy_binding["mode"] == "whole_contract":
                    self._ensure_edge(
                        decision_id, policy_binding["policy_node_id"], "APPLIED_POLICY",
                        application_semantics="AUDIT_GOVERNANCE_NOT_AGENT_CLAIM",
                        task_presentation_hash=presentation["presentation_hash"],
                        task_arm=presentation.get("arm_id"),
                        offered_clause_catalog_hash=content_hash(
                            presentation.get("offered_clause_catalog") or []),
                    )
                else:
                    self._ensure_edge(
                        decision_id, policy_binding["bundle_node_id"],
                        "GOVERNED_BY_POLICY_BUNDLE",
                        task_presentation_hash=presentation["presentation_hash"],
                        task_arm=presentation.get("arm_id"),
                    )
                    claimed = self._episode_claimed_policy_refs(artifact, episode)
                    bound = {
                        policy_binding["clause_to_policy"][ref]["node_id"]: (
                            policy_binding["clause_to_policy"][ref])
                        for ref in claimed if ref in policy_binding["clause_to_policy"]
                    }
                    for policy_node_id, component in sorted(bound.items()):
                        cited = sorted(
                            ref for ref in claimed
                            if (policy_binding["clause_to_policy"].get(ref) or {}).get(
                                "node_id") == policy_node_id)
                        self._ensure_edge(
                            decision_id, policy_node_id, "APPLIED_POLICY",
                            application_semantics=(
                                "AGENT_CLAIM_RESOLVED_TO_OFFERED_POLICY"),
                            cited_clause_refs=cited,
                            policy_id=component["policy_id"],
                            version=component["version"],
                            task_presentation_hash=presentation["presentation_hash"],
                            task_arm=presentation.get("arm_id"),
                        )
            for cycle_id in episode.get("source_cycle_ids") or []:
                self._ensure_edge(decision_id, cycle_nodes[str(cycle_id)], "DERIVED_FROM",
                                  provenance="MODEL_RECONSTRUCTED")
                cycle = next(row for row in artifact["cycles"] if row["cycle_id"] == cycle_id)
                for ref in cycle.get("decision_testimony_refs") or []:
                    node = testimony_nodes.get(str(ref))
                    if node:
                        self._ensure_edge(node, decision_id, "TESTIFIES_TO",
                                          provenance="SELF_REPORTED")
                if str(cycle_id) in execution_nodes:
                    self._ensure_edge(decision_id, execution_nodes[str(cycle_id)], "SUBMITTED_AS",
                                      provenance="SERVER_FACT")
                for relation, reference_node in reference_nodes_by_cycle.get(str(cycle_id), []):
                    self._ensure_edge(decision_id, reference_node, relation,
                                      provenance=("SELF_REPORTED" if relation in {"CITES", "CHECKED"}
                                                  else "SERVER_FACT"))

        for assertion in artifact.get("causal_assertions") or []:
            source_episode = str(assertion["source_episode_id"])
            target_episode = str(assertion["target_episode_id"])
            relationship = str(assertion["relationship_type"])
            if source_episode not in episode_nodes or target_episode not in episode_nodes:
                raise ValueError("causal assertion endpoint is outside this analysis")
            if relationship not in CAUSAL_TYPES:
                raise ValueError(f"invalid causal relationship {relationship!r}")
            source_id, target_id = episode_nodes[source_episode], episode_nodes[target_episode]
            if not self._has_edge(source_id, target_id, relationship):
                self.graph.add_causal_relationship(source_id, target_id, relationship)
            if not self._has_edge(source_id, target_id, relationship):
                raise RuntimeError("Semantica silently skipped a typed causal relationship")
            assertion_node = self._ensure_node(
                _node_id("causal_assertion", {"run": run_id, "analysis": analysis_id,
                                               "assertion": assertion["assertion_id"]}),
                "CausalAssertion", content="causal assertion pointer", run_id=run_id,
                analysis_id=analysis_id, assertion_id=assertion["assertion_id"],
                relationship_type=relationship,
                evidence_count=len(assertion.get("evidence_refs") or []),
                provenance=assertion.get("provenance"), artifact_ref=artifact.get("artifact_ref"),
            )
            self._ensure_edge(assertion_node, source_id, "ASSERTS_SOURCE")
            self._ensure_edge(assertion_node, target_id, "ASSERTS_TARGET")
            for evidence_ref in assertion.get("evidence_refs") or []:
                ref = str(evidence_ref)
                evidence_node = testimony_nodes.get(ref)
                if evidence_node is None:
                    evidence_node = self._ensure_node(
                        _node_id("evidence_ref", {"run": run_id, "analysis": analysis_id,
                                                  "ref_hash": content_hash(ref)}),
                        "EvidenceRef", content="evidence pointer", run_id=run_id,
                        analysis_id=analysis_id, locator_hash=content_hash(ref),
                        artifact_ref=artifact.get("artifact_ref"),
                    )
                self._ensure_edge(assertion_node, evidence_node, "SUPPORTED_BY")

        self._ensure_node(
            _node_id("projection_manifest", {"run": run_id, "analysis": analysis_id}),
            "ProjectionManifest", content="normalized projection manifest",
            run_id=run_id, analysis_id=analysis_id, projection_hash=projection_hash,
            projection_schema=PROJECTION_SCHEMA,
            taxonomy_version=artifact.get("taxonomy_version", DECISION_TAXONOMY_SCHEMA),
            supported_runtime_receipt_schema=artifact.get(
                "supported_runtime_receipt_schema"),
            runtime_receipt_mode=(artifact.get("runtime_receipt_manifest") or {}).get("mode"),
            sealed_receipt_count=(artifact.get("runtime_receipt_manifest") or {}).get(
                "sealed_receipt_count", 0),
            decision_receipt_coverage_status=(artifact.get(
                "decision_receipt_coverage") or {}).get("status"),
            sealed_decision_count=(artifact.get("decision_receipt_coverage") or {}).get(
                "sealed_episode_count", 0),
            decision_count=len(episode_nodes), causal_edge_count=len(triples),
        )
        self._project_audit_provenance(
            artifact, provenance_path=provenance_path, run_dir=run_dir)
        self.save()
        return projection_hash

    def _project_audit_provenance(self, artifact: dict[str, Any], *,
                                  provenance_path: Path, run_dir: Path) -> None:
        """Persist complete run-local episodes through Semantica ProvenanceManager."""
        run_id, analysis_id = str(artifact["run_id"]), str(artifact["analysis_id"])
        decision_ids = {
            str((node.get("metadata") or {})["acr_episode_id"]): str(node["id"])
            for node in self._find_nodes("decision", run_id=run_id, analysis_id=analysis_id)
            if (node.get("metadata") or {}).get("acr_episode_id")
        }
        expected = {str(row["episode_id"]) for row in artifact.get("episodes") or []}
        if set(decision_ids) != expected:
            raise RuntimeError("Semantica decision projection and audit episodes do not align")
        from acr.mvp.human_review import human_review_view
        from acr.mvp.semantica_audit import project_analysis_audit

        view = human_review_view(
            self, run_id, analysis_id, run_dir=run_dir, artifact=artifact)
        if view.get("status") != "OK":
            raise RuntimeError("a projected analysis did not produce a human review path")
        project_analysis_audit(
            artifact=artifact, decision_ids=decision_ids, human_view=view,
            provenance_path=provenance_path)

    def _project_audit_narratives(self, artifact: dict[str, Any], *,
                                  provenance_path: Path, run_dir: Path) -> None:
        """Append the current readable projection without replaying immutable core audit rows."""
        run_id, analysis_id = str(artifact["run_id"]), str(artifact["analysis_id"])
        decision_ids = {
            str((node.get("metadata") or {})["acr_episode_id"]): str(node["id"])
            for node in self._find_nodes("decision", run_id=run_id, analysis_id=analysis_id)
            if (node.get("metadata") or {}).get("acr_episode_id")
        }
        expected = {str(row["episode_id"]) for row in artifact.get("episodes") or []}
        if set(decision_ids) != expected:
            raise RuntimeError("Semantica decision projection and audit episodes do not align")
        from acr.mvp.human_review import human_review_view
        from acr.mvp.semantica_audit import project_decision_narratives

        view = human_review_view(
            self, run_id, analysis_id, run_dir=run_dir, artifact=artifact)
        project_decision_narratives(
            artifact=artifact, decision_ids=decision_ids, human_view=view,
            provenance_path=provenance_path)

    # --------------------------------------------------------------------- selection/query
    def available_analyses(self, run_id: str) -> list[str]:
        return sorted({str((node.get("metadata") or {}).get("analysis_id"))
                       for node in self._find_nodes("ProjectionManifest", run_id=run_id)})

    def projection_hash(self, run_id: str, analysis_id: str) -> str | None:
        node = self._projection_manifest_node(run_id, analysis_id)
        return str((node.get("metadata") or {}).get("projection_hash")) if node else None

    def analysis_metadata(self, run_id: str, analysis_id: str) -> dict[str, Any]:
        rows = self._find_nodes("AnalysisArtifact", run_id=run_id, analysis_id=analysis_id)
        if len(rows) != 1:
            raise ValueError(f"expected one analysis artifact for {run_id}/{analysis_id}")
        return dict(rows[0].get("metadata") or {})

    def provenance_manager(self, run_id: str, analysis_id: str):
        """Open the Semantica provenance store that owns this run-local audit trail."""
        from semantica.provenance import ProvenanceManager

        metadata = self.analysis_metadata(run_id, analysis_id)
        path = metadata.get("provenance_path")
        if not path:
            raise ValueError(f"analysis {analysis_id!r} has no Semantica provenance pointer")
        return ProvenanceManager(storage_path=str(path))

    def decision_narrative_bundle_id(self, run_id: str, analysis_id: str) -> str:
        # The analysis and its Semantica Decisions are immutable, while the human-readable
        # domain projection is revisioned append-only. Compute the current bundle from the
        # sealed artifact instead of pinning readers to a legacy projection recorded on the
        # AnalysisArtifact node.
        from acr.mvp.semantica_audit import bundle_id_for

        return bundle_id_for(self.load_analysis_artifact(run_id, analysis_id))

    def load_analysis_artifact(self, run_id: str, analysis_id: str) -> dict[str, Any]:
        metadata = self.analysis_metadata(run_id, analysis_id)
        ref = metadata.get("artifact_ref")
        if not ref:
            raise ValueError(f"analysis {analysis_id!r} has no local artifact reference")
        raw = json.loads(Path(str(ref)).read_text(encoding="utf-8"))
        supplied = str(raw.get("analysis_artifact_hash") or "")
        unhashed = {key: value for key, value in raw.items()
                    if key not in {"analysis_artifact_hash", "artifact_ref"}}
        if content_hash(unhashed) != supplied or supplied != metadata.get("artifact_hash"):
            raise ValueError("analysis artifact failed content-hash verification")
        if raw.get("run_id") != run_id or raw.get("analysis_id") != analysis_id:
            raise ValueError("analysis artifact identity does not match the graph pointer")
        return raw

    def select_analysis(self, run_id: str, analysis_id: str, *, selected_by: str,
                        reason: str, provenance: str = "HUMAN_ADJUDICATED") -> str:
        if analysis_id not in self.available_analyses(run_id):
            raise ValueError(f"analysis {analysis_id!r} is not projected for run {run_id!r}")
        selected_at = datetime.now(UTC).isoformat()
        identity = {"run": run_id, "analysis": analysis_id, "selected_by": selected_by,
                    "reason": reason, "selected_at": selected_at}
        node_id = _node_id("analysis_selection", identity)
        node_id = self._ensure_node(
            node_id, "AnalysisSelection", content="explicit analysis selection", run_id=run_id,
            analysis_id=analysis_id, selected_by=selected_by, reason=reason,
            provenance=provenance, selected_at=selected_at,
        )
        analysis_node = next(node for node in self._find_nodes(
            "AnalysisArtifact", run_id=run_id, analysis_id=analysis_id))
        self._ensure_edge(node_id, str(analysis_node["id"]), "SELECTS", provenance=provenance)
        self.save()
        return node_id

    def selected_analysis(self, run_id: str) -> str | None:
        rows = self._find_nodes("AnalysisSelection", run_id=run_id)
        if not rows:
            return None
        rows.sort(key=lambda node: str((node.get("metadata") or {}).get("selected_at") or ""))
        return str((rows[-1].get("metadata") or {}).get("analysis_id") or "") or None

    def _episode_rows(self, run_id: str, analysis_id: str) -> list[dict[str, Any]]:
        rows = []
        for node in self._find_nodes("decision", run_id=run_id, analysis_id=analysis_id):
            meta = node.get("metadata") or {}
            if not meta.get("acr_episode_id"):
                continue
            groundings: list[dict[str, Any]] = []
            for edge in map(_edge_dict, self.graph.edges):
                source = edge.get("source_id", edge.get("source"))
                if edge.get("type") != "APPLIED_POLICY" or source != node["id"]:
                    continue
                target = str(edge.get("target_id", edge.get("target")) or "")
                policy = self.graph.find_node(target) or {}
                policy_meta = policy.get("metadata") or {}
                edge_meta = edge.get("metadata") or edge.get("properties") or {}
                groundings.append({
                    "policy_id": edge_meta.get("policy_id")
                    or policy_meta.get("policy_id"),
                    "version": edge_meta.get("version") or policy_meta.get("version"),
                    "cited_clause_refs": sorted(edge_meta.get("cited_clause_refs") or []),
                    "application_semantics": edge_meta.get("application_semantics"),
                })
            groundings.sort(key=lambda row: (
                str(row.get("policy_id") or ""), str(row.get("version") or "")))
            rows.append({
                "decision_id": node["id"], "episode_id": meta["acr_episode_id"],
                "run_id": run_id, "analysis_id": analysis_id,
                "decision_function": meta.get("category"), "scenario": meta.get("scenario"),
                "reasoning": meta.get("reasoning"), "outcome": meta.get("outcome"),
                "decision_point_hash": meta.get("decision_point_hash"),
                "reconstruction_stability": meta.get("confidence"),
                "confidence_semantics": meta.get("confidence_semantics"),
                "source_seq_start": meta.get("source_seq_start"),
                "source_event_time": meta.get("source_event_time"),
                "artifact_ref": meta.get("artifact_ref"),
                "policy_groundings": groundings,
            })
        rows.sort(key=lambda row: (row.get("source_seq_start") or 0, row["episode_id"]))
        return rows

    def chain(self, run_id: str, analysis_id: str | None = None) -> dict[str, Any]:
        available = self.available_analyses(run_id)
        chosen = analysis_id or self.selected_analysis(run_id)
        if chosen is None:
            return {"run_id": run_id, "status": "NO_ANALYSIS_SELECTION",
                    "available_analysis_ids": available, "episodes": [],
                    "causal_edges": [], "suggested_links": []}
        if chosen not in available:
            raise ValueError(f"analysis {chosen!r} is not available for run {run_id!r}")
        episodes = self._episode_rows(run_id, chosen)
        by_decision = {row["decision_id"]: row["episode_id"] for row in episodes}
        causal = []
        for edge in map(_edge_dict, self.graph.edges):
            if edge.get("type") not in CAUSAL_TYPES:
                continue
            if edge.get("source_id") in by_decision and edge.get("target_id") in by_decision:
                causal.append({"source_episode_id": by_decision[edge["source_id"]],
                               "target_episode_id": by_decision[edge["target_id"]],
                               "relationship_type": edge["type"]})
        causal.sort(key=lambda row: (row["source_episode_id"], row["target_episode_id"],
                                     row["relationship_type"]))
        suggestions: list[dict[str, Any]] = []
        seen_suggestions: set[tuple[str, str, str]] = set()
        for decision_id, target_episode in by_decision.items():
            # Suggested links are intentionally direct-only.  A human chain must not
            # turn Semantica's transitive graph walks into apparent causal history,
            # and deep walks can explode combinatorially through shared entities.
            for chain in self.graph.trace_decision_chain(decision_id, max_steps=1):
                for hop in chain.get("hops") or []:
                    if hop.get("type") != "influences":
                        continue
                    source = by_decision.get(str(hop.get("from")))
                    target = by_decision.get(str(hop.get("to")))
                    if source is None or target is None:
                        continue
                    key = (source, target, "influences")
                    if key in seen_suggestions:
                        continue
                    seen_suggestions.add(key)
                    suggestions.append({
                        "source_episode_id": source, "target_episode_id": target,
                        "semantica_relationship": "influences",
                        "authority": "SUGGESTED_ONLY",
                        "edge_weight": hop.get("edge_weight"),
                        "cannot_establish": "causation",
                    })
        return {"run_id": run_id, "analysis_id": chosen, "status": "OK",
                "episodes": episodes, "causal_edges": causal,
                "suggested_links": suggestions,
                "suggested_links_note": "Semantica lowercase `influences` heuristics are excluded "
                                        "from the canonical causal chain"}

    def _decision_id_for_episode(self, episode_id: str) -> str:
        node = next((node for node in self.graph.find_nodes(node_type="decision")
                     if (node.get("metadata") or {}).get("acr_episode_id") == episode_id), None)
        if node is None:
            raise ValueError(f"unknown episode {episode_id!r}")
        return str(node["id"])

    def similar_candidates(self, episode_id: str, *, max_results: int = 5,
                           min_similarity: float = 0.3,
                           cross_run_only: bool = True) -> dict[str, Any]:
        """Use Semantica's native retrieval, then apply only ACR identity/scope guards."""
        decision_id = self._decision_id_for_episode(episode_id)
        node = self.graph.find_node(decision_id) or {}
        meta = node.get("metadata") or {}
        decision_count = len(self.graph.find_nodes(node_type="decision"))
        native_rows = self.graph.find_similar_decisions(
            str(meta.get("scenario") or ""), category=str(meta.get("category") or ""),
            max_results=decision_count, min_similarity=0.0)
        query_signature = str(meta.get("situation_signature_hash") or "")
        query_decision_point = str(meta.get("decision_point_hash") or "")
        rows: list[dict[str, Any]] = []
        for row in native_rows:
            decision = row.get("decision") or {}
            candidate_meta = decision.get("metadata") or {}
            if str(decision.get("id") or "") == decision_id:
                continue
            if cross_run_only and candidate_meta.get("run_id") == meta.get("run_id"):
                continue
            candidate_decision_point = str(
                candidate_meta.get("decision_point_hash") or "")
            same_decision_point = (
                candidate_decision_point == query_decision_point
                if query_decision_point and candidate_decision_point else None)
            # Semantica includes outcome in content similarity. Exact repeated audit points
            # must remain visible precisely when their outcomes differ, even if that lowers
            # the native score below the near-neighbour threshold.
            if float(row.get("similarity") or 0.0) < min_similarity \
                    and same_decision_point is not True:
                continue
            rows.append({
                **row,
                "same_situation_signature": bool(query_signature) and
                    candidate_meta.get("situation_signature_hash") == query_signature,
                # ``None`` preserves honest reads of legacy v2 projections, which did not
                # carry an exact question+evidence identity.
                "same_decision_point": same_decision_point,
            })
            if len(rows) == max_results:
                break
        return {
                "authority": "CANDIDATE_ONLY", "episode_id": episode_id,
                "retrieval_engine": "semantica.ContextGraph.find_similar_decisions",
                "query": {
                    "category": meta.get("category"),
                    "scenario": meta.get("scenario"),
                    "situation_signature_hash": query_signature,
                    "decision_point_hash": query_decision_point or None,
                    "cross_run_only": cross_run_only,
                    "min_similarity": min_similarity,
                },
                "candidates": rows,
                "cannot_establish": ["clinical correctness", "normative precedent"]}

    def find_divergent_decision_points(self, cohort: list[dict[str, str]], *,
                                       min_similarity: float = 0.65) -> dict[str, Any]:
        """Retrieve cross-run near-neighbours with different outcomes via Semantica.

        This is deliberately a candidate report, not an automatic guideline defect label.
        ACR supplies cohort scoping, atomic-decision guards and runtime grounding context;
        Semantica remains the similarity retrieval engine.
        """
        identities = [(str(row.get("run_id") or ""),
                       str(row.get("analysis_id") or "")) for row in cohort]
        if not identities or any(not run or not analysis for run, analysis in identities):
            raise ValueError("a comparison cohort needs run_id + analysis_id members")
        if len({run for run, _ in identities}) != len(identities):
            raise ValueError("a comparison cohort must select one analysis per run")

        members: dict[str, dict[str, Any]] = {}
        for run_id, analysis_id in identities:
            if analysis_id not in self.available_analyses(run_id):
                raise ValueError(f"unprojected cohort member {run_id}/{analysis_id}")
            artifact = self.load_analysis_artifact(run_id, analysis_id)
            episodes = {str(row.get("episode_id")): row
                        for row in artifact.get("episodes") or []}
            cycles = {str(row.get("cycle_id")): row
                      for row in artifact.get("cycles") or []}
            rows = {str(row["episode_id"]): row
                    for row in self._episode_rows(run_id, analysis_id)}
            for node in self._find_nodes(
                    "decision", run_id=run_id, analysis_id=analysis_id):
                meta = node.get("metadata") or {}
                episode_id = str(meta.get("acr_episode_id") or "")
                episode = episodes.get(episode_id)
                if not episode:
                    continue
                basis: set[str] = set()
                coverage: set[str] = set()
                for cycle_id in episode.get("source_cycle_ids") or []:
                    for action in (cycles.get(str(cycle_id)) or {}).get("actions") or []:
                        if action.get("tool") not in {"note_decision", "record_finding"}:
                            continue
                        args = action.get("args") or {}
                        basis.update(str(value) for value in args.get("basis_sources") or [])
                        if args.get("rule_coverage_claim"):
                            coverage.add(str(args["rule_coverage_claim"]))
                members[str(node["id"])] = {
                    "decision_id": str(node["id"]), "run_id": run_id,
                    "analysis_id": analysis_id, "episode_id": episode_id,
                    "decision_function": str(meta.get("category") or "other"),
                    "decision_subject": str(meta.get("decision_subject") or "other"),
                    "outcome": str(meta.get("outcome") or ""),
                    "situation_signature_hash": str(
                        meta.get("situation_signature_hash") or ""),
                    "decision_point_hash": str(meta.get("decision_point_hash") or ""),
                    "review_model": str(artifact.get("review_model") or "UNKNOWN"),
                    "task_arm": str(artifact.get("task_arm") or "UNKNOWN"),
                    "basis_sources": tuple(sorted(basis)),
                    "rule_coverage_claims": tuple(sorted(coverage)),
                    "policy_groundings": (rows.get(episode_id) or {}).get(
                        "policy_groundings") or [],
                    "scenario": str(meta.get("scenario") or ""),
                }

        divergences: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for decision_id, member in sorted(members.items()):
            native_rows = self.graph.find_similar_decisions(
                member["scenario"], category=member["decision_function"],
                max_results=len(members), min_similarity=0.0)
            for native in native_rows:
                candidate = native.get("decision") or {}
                candidate_id = str(candidate.get("id") or "")
                other = members.get(candidate_id)
                pair = tuple(sorted((decision_id, candidate_id)))
                if not other or decision_id == candidate_id or pair in seen_pairs:
                    continue
                if member["run_id"] == other["run_id"]:
                    continue
                if member["decision_subject"] != other["decision_subject"]:
                    continue
                if member["outcome"] == other["outcome"]:
                    continue
                same_decision_point = (
                    member["decision_point_hash"] == other["decision_point_hash"]
                    if member["decision_point_hash"] and other["decision_point_hash"]
                    else None
                )
                if float(native.get("similarity") or 0.0) < min_similarity \
                        and same_decision_point is not True:
                    continue
                seen_pairs.add(pair)
                pair_members = [member, other]
                outcomes = Counter(row["outcome"] for row in pair_members)
                models = Counter(row["review_model"] for row in pair_members)
                arms = Counter(row["task_arm"] for row in pair_members)
                any_own_knowledge = any(
                    "own_knowledge" in row["basis_sources"] for row in pair_members)
                any_policy = any(row["policy_groundings"] for row in pair_members)
                grounding = (
                    "UNGROUNDED_OUTCOME_DIVERGENCE"
                    if any_own_knowledge and not any_policy else
                    "POLICY_GROUNDED_DIVERGENCE" if any_policy else
                    "NO_POLICY_GROUNDING_RECORDED")
                public_members = [
                    {key: value for key, value in row.items() if key != "scenario"}
                    for row in sorted(pair_members, key=lambda value: (
                        value["run_id"], value["analysis_id"], value["episode_id"]))]
                divergences.append({
                    "decision_function": member["decision_function"],
                    "decision_subject": member["decision_subject"],
                    "similarity": float(native.get("similarity") or 0.0),
                    "content_similarity": float(native.get("content_similarity") or 0.0),
                    "structural_similarity": float(
                        native.get("structural_similarity") or 0.0),
                    "same_situation_signature": (
                        bool(member["situation_signature_hash"])
                        and member["situation_signature_hash"]
                        == other["situation_signature_hash"]),
                    "same_decision_point": same_decision_point,
                    "outcome_distribution": dict(sorted(outcomes.items())),
                    "review_model_distribution": dict(sorted(models.items())),
                    "task_arm_distribution": dict(sorted(arms.items())),
                    "grounding_status": grounding,
                    "members": public_members,
                })
        divergences.sort(key=lambda row: (
            -row["similarity"], row["decision_function"], row["decision_subject"],
            tuple(member["episode_id"] for member in row["members"])))
        return {
            "authority": "CANDIDATE_ONLY",
            "retrieval_engine": "semantica.ContextGraph.find_similar_decisions",
            "cohort": [{"run_id": run, "analysis_id": analysis}
                       for run, analysis in identities],
            "min_similarity": min_similarity,
            "divergences": divergences,
            "cannot_establish": ["clinical correctness", "guideline defect",
                                 "normative precedent"],
        }

    def impact_candidates(self, episode_id: str) -> dict[str, Any]:
        decision_id = self._decision_id_for_episode(episode_id)
        native = self.graph.analyze_decision_impact(decision_id)
        for key in ("direct_influence", "indirect_influence", "influence_scores"):
            for row in native.get(key) or []:
                candidate_id = str(row.get("decision_id") or "")
                node = self.graph.find_node(candidate_id)
                if node is not None:
                    row["decision"] = node
        return {"authority": "CANDIDATE_ONLY", "episode_id": episode_id,
                "retrieval_engine": "semantica.ContextGraph.analyze_decision_impact",
                "candidates": native,
                "cannot_establish": ["causation", "counterfactual impact"]}

    def causal_trace(self, episode_id: str, *, max_steps: int = 5) -> dict[str, Any]:
        """Use Semantica traversal, scoped to ACR's evidenced causal assertion subgraph.

        ContextGraph also maintains lowercase heuristic ``influences`` edges for general
        decision intelligence. They are useful candidates, but allowing them into an audit walk
        can create thousands of paths and falsely present adjacency as causality. We therefore
        ask Semantica for one native hop at a time and intersect it with the CausalAssertion
        provenance nodes projected for this analysis before composing the bounded trace.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        decision_id = self._decision_id_for_episode(episode_id)
        decision_node = self.graph.find_node(decision_id) or {}
        decision_meta = decision_node.get("metadata") or {}
        analysis_id = str(decision_meta.get("analysis_id") or "")

        asserted: dict[tuple[str, str, str], dict[str, Any]] = {}
        graph_edges = list(map(_edge_dict, self.graph.edges))
        for node in self._find_nodes("CausalAssertion", analysis_id=analysis_id):
            node_id = str(node["id"])
            metadata = node.get("metadata") or {}
            sources = [str(edge["target_id"]) for edge in graph_edges
                       if str(edge.get("source_id")) == node_id
                       and edge.get("type") == "ASSERTS_SOURCE"]
            targets = [str(edge["target_id"]) for edge in graph_edges
                       if str(edge.get("source_id")) == node_id
                       and edge.get("type") == "ASSERTS_TARGET"]
            relationship = str(metadata.get("relationship_type") or "")
            if len(sources) != 1 or len(targets) != 1 or relationship not in CAUSAL_TYPES:
                continue
            asserted[(sources[0], targets[0], relationship)] = {
                "assertion_id": metadata.get("assertion_id"),
                "assertion_provenance": metadata.get("provenance"),
            }

        direct_cache: dict[str, list[dict[str, Any]]] = {}
        native_direct_calls = 0

        def direct_predecessors(target_id: str) -> list[dict[str, Any]]:
            nonlocal native_direct_calls
            if target_id in direct_cache:
                return direct_cache[target_id]
            native_direct_calls += 1
            rows: dict[tuple[str, str, str], dict[str, Any]] = {}
            for chain in self.graph.trace_decision_chain(target_id, max_steps=1):
                hops = chain.get("hops") or []
                if len(hops) != 1:
                    continue
                hop = dict(hops[0])
                key = (str(hop.get("from")), str(hop.get("to")), str(hop.get("type")))
                proof = asserted.get(key)
                if proof is None or key[1] != target_id:
                    continue
                rows[key] = {**hop, **proof}
            direct_cache[target_id] = [rows[key] for key in sorted(rows)]
            return direct_cache[target_id]

        def paths_to(target_id: str, remaining: int,
                     visited: frozenset[str]) -> list[list[dict[str, Any]]]:
            paths: list[list[dict[str, Any]]] = []
            for hop in direct_predecessors(target_id):
                source_id = str(hop["from"])
                if source_id in visited:
                    continue
                prefixes = (paths_to(source_id, remaining - 1, visited | {source_id})
                            if remaining > 1 else [])
                if prefixes:
                    paths.extend([*prefix, hop] for prefix in prefixes)
                else:
                    paths.append([hop])
            return paths

        hop_paths = paths_to(decision_id, max_steps, frozenset({decision_id}))
        # An asserted audit graph should be small. Keep the response bounded even if a future
        # analysis contains combinatorial branching; the provenance graph remains queryable.
        max_chains = 100
        truncated = len(hop_paths) > max_chains
        hop_paths = hop_paths[:max_chains]
        chains = []
        for hops in hop_paths:
            weights = [float(hop.get("edge_weight", 1.0) or 0.0) for hop in hops]
            confidence = 1.0
            for weight in weights:
                confidence *= weight
            chains.append({
                "hops": hops,
                "hop_count": len(hops),
                "confidence_decay": confidence,
                "weakest_link": hops[weights.index(min(weights))],
                "distance_band": ("direct" if len(hops) == 1 else
                                  "near" if len(hops) <= 3 else "mid-range"),
                "interpretation": (
                    f"{len(hops)} explicit causal assertion hop(s); "
                    f"confidence product {confidence:.2f}."),
            })
        return {
            "authority": "EXPLICIT_CAUSAL_ASSERTIONS",
            "episode_id": episode_id,
            "retrieval_engine": "semantica.ContextGraph.trace_decision_chain",
            "scope": {"analysis_id": analysis_id,
                      "accepted_relationship_types": sorted(CAUSAL_TYPES),
                      "requires_causal_assertion_provenance": True},
            "native_direct_calls": native_direct_calls,
            "chains": chains,
            "truncated": truncated,
            "excluded_from_audit_chain": ["influences", "temporal adjacency"],
        }

    def register_policy_revision(self, policy_id: str, *, from_version: str,
                                 rules: dict[str, Any], change_reason: str) \
            -> dict[str, Any]:
        """Append one content-addressed Policy revision without rewriting past bindings."""
        from semantica.context import PolicyEngine
        from semantica.context.decision_models import Policy

        old_node_id = f"{policy_id}:{from_version}"
        old = self.graph.find_node(old_node_id)
        if old is None or old.get("type") != "Policy":
            raise ValueError(f"unknown Policy revision {old_node_id}")
        metadata = old.get("metadata") or {}
        digest = content_hash({
            "policy_id": policy_id,
            "category": metadata.get("category"),
            "rules": rules,
            "previous_version": from_version,
        })
        version = f"content-{digest[:12]}"
        new_node_id = f"{policy_id}:{version}"
        if self.graph.find_node(new_node_id) is None:
            now = datetime.now(UTC)
            prior_inner = metadata.get("metadata") or {}
            PolicyEngine(self.graph).add_policy(Policy(
                policy_id=policy_id,
                name=str(metadata.get("name") or policy_id),
                description=str(metadata.get("description") or ""),
                rules=rules,
                category=str(metadata.get("category") or "chart_review"),
                version=version, created_at=now, updated_at=now,
                metadata={
                    **prior_inner, "content_hash": digest,
                    "previous_version": from_version,
                    "change_reason": change_reason,
                },
            ))
        self._ensure_edge(
            old_node_id, new_node_id, "VERSION_OF",
            change_reason=change_reason, changed_at=datetime.now(UTC).isoformat())
        self.save()
        return {
            "policy_id": policy_id, "from_version": from_version,
            "version": version, "content_hash": digest,
            "change_reason": change_reason,
        }

    def affected_by_policy_change(self, policy_id: str, *, from_version: str,
                                  to_version: str) -> dict[str, Any]:
        """Delegate Task Contract blast-radius discovery to Semantica PolicyEngine."""
        from semantica.context import PolicyEngine

        native = PolicyEngine(self.graph).get_affected_decisions(
            policy_id, from_version, to_version)
        affected: list[dict[str, Any]] = []
        cases: dict[tuple[str, str], dict[str, Any]] = {}
        for row in native:
            enriched = dict(row)
            node = self.graph.find_node(str(row.get("decision_id") or "")) or {}
            metadata = node.get("metadata") or {}
            enriched.update({
                "run_id": metadata.get("run_id"),
                "analysis_id": metadata.get("analysis_id"),
                "episode_id": metadata.get("acr_episode_id"),
                "decision_subject": metadata.get("decision_subject"),
            })
            affected.append(enriched)
            identity = (str(metadata.get("run_id") or ""),
                        str(metadata.get("analysis_id") or ""))
            case = cases.setdefault(identity, {
                "run_id": identity[0], "analysis_id": identity[1],
                "affected_decision_count": 0, "decision_functions": set(),
            })
            case["affected_decision_count"] += 1
            case["decision_functions"].add(str(row.get("category") or "other"))
        affected_cases = []
        for identity in sorted(cases):
            case = cases[identity]
            affected_cases.append({
                **case, "decision_functions": sorted(case["decision_functions"]),
            })
        return {
            "authority": "RE_AUDIT_CANDIDATES_ONLY",
            "policy_id": policy_id, "from_version": from_version,
            "to_version": to_version,
            "scope": "DIRECT_POLICY_BINDINGS",
            "retrieval_engine": "semantica.PolicyEngine.get_affected_decisions",
            "affected_decisions": affected,
            "affected_cases": affected_cases,
            "cannot_establish": ["changed clinical answer", "automatic non-compliance"],
        }

    def insights(self, run_id: str, analysis_id: str | None = None) -> dict[str, Any]:
        chosen = analysis_id or self.selected_analysis(run_id)
        if chosen is None:
            raise ValueError("insights require an explicit or selected analysis")
        rows = self._episode_rows(run_id, chosen)
        categories = Counter(str(row["decision_function"]) for row in rows)
        values = [float(row["reconstruction_stability"] or 0.0) for row in rows]
        scoped = self.graph.__class__(
            extract_entities=False, extract_relationships=False, advanced_analytics=True,
            centrality_analysis=True, community_detection=True, node_embeddings=False)
        scoped_ids: dict[str, str] = {}
        for row in rows:
            node = self.graph.find_node(str(row["decision_id"])) or {}
            metadata = node.get("metadata") or {}
            scoped_ids[str(row["decision_id"])] = scoped.record_decision(
                category=str(metadata.get("category") or "other"),
                scenario=str(metadata.get("scenario") or ""),
                reasoning=str(metadata.get("reasoning") or ""),
                outcome=str(metadata.get("outcome") or ""),
                confidence=float(metadata.get("confidence") or 0.0),
                entities=list(metadata.get("entities") or []),
                metadata={"confidence_semantics": CONFIDENCE_SEMANTICS},
            )
        for edge in map(_edge_dict, self.graph.edges):
            if edge.get("type") in CAUSAL_TYPES and edge.get("source_id") in scoped_ids \
                    and edge.get("target_id") in scoped_ids:
                scoped.add_causal_relationship(
                    scoped_ids[str(edge["source_id"])], scoped_ids[str(edge["target_id"])],
                    str(edge["type"]))
        semantica_insights = _json_ready(scoped.get_decision_insights())
        if "confidence_stats" in semantica_insights:
            semantica_insights["confidence_stats"] = {
                **semantica_insights["confidence_stats"],
                "semantics": CONFIDENCE_SEMANTICS,
            }
        return {"run_id": run_id, "analysis_id": chosen, "episode_count": len(rows),
                "categories": dict(sorted(categories.items())),
                "reconstruction_stability": {
                    "semantics": CONFIDENCE_SEMANTICS,
                    "mean": sum(values) / len(values) if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                },
                "semantica_scoped_insights": semantica_insights,
                "scope_note": "Semantica analytics were recomputed on this run + analysis only"}

    def check_mechanical_policy(self, decision_data: dict[str, Any], *,
                                rules: dict[str, Any] | None) -> dict[str, Any]:
        if rules is None:
            raise ValueError("an explicit ACR mechanical policy is required; Semantica defaults "
                             "are not Task Contract compliance")
        result = self.graph.check_decision_rules(decision_data, rules=rules)
        return {"authority": "MECHANICAL_POLICY_ONLY", "result": result}

    # ------------------------------------------------------------------------- hygiene/I-O
    def validate_export(self) -> list[str]:
        issues: list[str] = []
        decision_ids = {str(node["id"]) for node in self.graph.find_nodes(node_type="decision")}
        for node in self.graph.find_nodes(node_type="decision"):
            meta = node.get("metadata") or {}
            for field in ("scenario", "reasoning", "outcome"):
                value = str(meta.get(field) or "")
                if _FORBIDDEN_CORE.search(value):
                    issues.append(f"decision {node['id']} core {field} contains a patient locator")
        for edge in map(_edge_dict, self.graph.edges):
            if edge.get("type") == "involves" and edge.get("source_id") in decision_ids:
                entity = str(edge.get("target_id") or "")
                if not _SAFE_ENTITY.fullmatch(entity):
                    issues.append(f"unsafe decision entity {entity!r}")
        return issues

    def save(self) -> None:
        issues = self.validate_export()
        if issues:
            raise ValueError("unsafe Semantica export: " + "; ".join(issues))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save_to_file(str(self.path))

    def stats(self) -> dict[str, Any]:
        return {"analyses": len(self.graph.find_nodes(node_type="ProjectionManifest")),
                "episodes": len(self.graph.find_nodes(node_type="decision")),
                "selections": len(self.graph.find_nodes(node_type="AnalysisSelection")),
                "causal_assertions": len(self.graph.find_nodes(node_type="CausalAssertion")),
                "semantica": self.graph.stats()}


def ingest_run(_run_dir: Path, _ledger: DecisionIntelligence) -> dict[str, Any]:
    """Raw runtime records are not Decision Episodes and therefore cannot be ingested."""
    raise RuntimeError(
        "raw trace ingestion into Semantica was removed: fetch the complete Langtrace export "
        "and run `reconstruct`, then explicitly select an analysis")
