"""Build the immutable record of what one review arm actually showed the agent.

The canonical Task Contract answers *what the run will be evaluated against*.  A
``ContractSnapshot`` answers the different question *what normative material was available to
the agent while it decided*.  Keeping those two questions separate prevents a requirements-only
run from being credited, after the fact, with a clinical rule that never appeared in its prompt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from acr.contract.outcomes import declared_statuses
from acr.contract.trace import rule_catalog

CLAIMED_AND_VERIFIED = "CLAIMED_AND_VERIFIED"
CLAIMED_NOT_OFFERED = "CLAIMED_NOT_OFFERED"
CLAIMED_UNKNOWN = "CLAIMED_UNKNOWN"
NOT_CLAIMED = "NOT_CLAIMED"

TASK_ARMS = ("task_only", "policy_bundle", "requirements_only", "detailed")


def canonical_json(value: Any) -> str:
    """The byte representation used by every content address in this read-side pipeline."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=str)


def content_hash(value: Any) -> str:
    raw = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compile_policy_bundle(spec: Any, offered: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compile exact Task Contract clauses into independently versioned decision boundaries.

    A Semantica Policy is smaller than the whole Task Contract but larger than a syntactic
    ``if`` branch.  Conflict branches sharing one discriminating fact form one boundary;
    every other catalog clause is independently changeable.  Component versions are content
    addresses so an unchanged policy keeps its identity when a bundle version changes.
    """
    catalog = [dict(row) for row in offered]
    by_ref = {str(row["rule_id"]): row for row in catalog}
    claimed: set[str] = set()
    components: list[dict[str, Any]] = []

    conflicts_by_fact: dict[str, list[str]] = {}
    for row in catalog:
        if row.get("kind") != "conflict_rule":
            continue
        try:
            structured = json.loads(str(row.get("text") or "{}"))
        except (TypeError, ValueError):
            structured = {}
        for fact in structured.get("turns_on") or []:
            conflicts_by_fact.setdefault(str(fact), []).append(str(row["rule_id"]))

    def add_component(policy_id: str, category: str, clause_refs: list[str]) -> None:
        rules = [by_ref[ref] for ref in clause_refs]
        material = {
            "policy_id": policy_id,
            "category": category,
            "authority": "NORMATIVE_TASK_CONTRACT",
            "clause_refs": clause_refs,
            "rules": rules,
        }
        digest = content_hash(material)
        components.append({
            **material,
            "version": f"content-{digest[:12]}",
            "content_hash": digest,
        })
        claimed.update(clause_refs)

    for fact, conflict_refs in sorted(conflicts_by_fact.items()):
        fact_ref = f"discriminating_fact.{fact}"
        refs = ([fact_ref] if fact_ref in by_ref else []) + sorted(conflict_refs)
        add_component(
            f"{spec.spec_id}.conflict.{fact}", "conflict_resolution", refs)

    categories = {
        "decision_rule": "candidate_or_answer_selection",
        "conflict_rule": "conflict_resolution",
        "discriminating_fact": "proof_discriminator",
        "evidence_rule": "evidence_standing",
        "proof_obligation": "proof_and_completeness",
        "field_format": "output_normalization",
        "field_allowable_values": "output_normalization",
        "abstention": "abstention",
    }
    for ref, row in sorted(by_ref.items()):
        if ref in claimed:
            continue
        kind = str(row.get("kind") or "other")
        suffix = ref.replace("_rule.", ".").replace("_", "-")
        add_component(f"{spec.spec_id}.{suffix}", categories.get(kind, kind), [ref])

    body = {
        "schema": "acr.policy_bundle.v1",
        "bundle_id": str(spec.spec_id),
        "bundle_version": str(spec.spec_version),
        "task_contract_content_hash": str(spec.spec_hash),
        "policies": sorted(components, key=lambda row: row["policy_id"]),
    }
    return {**body, "bundle_hash": content_hash(body)}


def requirements_only_text(spec: Any) -> str:
    """Render the execution arm that contains output requirements but no clinical details."""
    lines = [f"# EXTRACTION REQUIREMENTS  ({spec.spec_id} v{spec.spec_version})", "",
             f"QUESTION: {spec.question}", "", "OUTPUT FIELDS:"]
    for field in spec.fields:
        bits = [f"  - {field.name} ({field.type}"]
        if field.format:
            bits.append(f", format={field.format}")
        bits.append(")")
        lines.append("".join(bits))
        if field.description:
            lines.append(f"      {field.description}")
        if field.allowable_values:
            lines.append("      allowable: " + ", ".join(map(str, field.allowable_values)))
    lines += ["", "ALLOWED OUTCOMES:"]
    for name, declaration in declared_statuses(spec).items():
        if declaration.get("submittable", True) is not False:
            lines.append(f"  - {name}: {declaration.get('meaning')}")
    lines += ["", "Return exactly these fields through submit_answer. No decision rules, "
                      "evidence interpretation, conflict-resolution details, search hints, or "
                      "boundary-case explanations are supplied in this arm."]
    return "\n".join(lines)


def task_contract_text(spec: Any, arm_id: str) -> str:
    if arm_id in {"task_only", "requirements_only"}:
        return requirements_only_text(spec)
    if arm_id in {"policy_bundle", "detailed"}:
        return spec.as_prompt_block()
    raise ValueError(f"arm_id must be one of {TASK_ARMS}, got {arm_id!r}")


@dataclass(frozen=True, slots=True)
class ContractSnapshot:
    schema: str
    run_id: str
    arm_id: str
    task_contract_ref: dict[str, str]
    offered_clause_catalog: tuple[dict[str, Any], ...]
    known_clause_index: tuple[dict[str, str], ...]
    method_card_refs: tuple[dict[str, str], ...]
    operational_instruction_refs: tuple[dict[str, str], ...]
    rendered_prompt_artifact_ref: str
    prompt_hash: str
    presentation_hash: str
    policy_bundle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, destination: Path) -> Path:
        destination = Path(destination)
        path = (destination / "task_presentation.json"
                if destination.suffix.lower() != ".json" else destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def from_path(cls, path: Path) -> "ContractSnapshot":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        hash_payload = dict(raw)
        claimed = hash_payload.pop("presentation_hash")
        if content_hash(hash_payload) != claimed:
            raise ValueError("task presentation content hash does not match its payload")
        raw.setdefault("policy_bundle", None)
        raw["offered_clause_catalog"] = tuple(raw.get("offered_clause_catalog") or ())
        raw["known_clause_index"] = tuple(raw.get("known_clause_index") or ())
        raw["method_card_refs"] = tuple(raw.get("method_card_refs") or ())
        raw["operational_instruction_refs"] = tuple(
            raw.get("operational_instruction_refs") or ())
        return cls(**raw)

    def resolve_rule(self, raw_ref: object) -> dict[str, Any]:
        """Resolve an exact rule/fact id against *offered* material, never fuzzy text."""
        original = str(raw_ref).strip()
        ref = original
        for prefix in ("rule:", "fact:"):
            if ref.startswith(prefix):
                ref = ref[len(prefix):]
                break
        offered = {str(row["rule_id"]): row for row in self.offered_clause_catalog}
        known = {str(row["rule_id"]): row for row in self.known_clause_index}
        if ref in offered:
            row = offered[ref]
            return {"ref": ref, "status": CLAIMED_AND_VERIFIED,
                    "kind": str(row["kind"]), "text_sha": str(row["text_sha"]),
                    "rendered_locator": row.get("rendered_locator")}
        if ref in known:
            return {"ref": ref, "status": CLAIMED_NOT_OFFERED,
                    "kind": str(known[ref]["kind"])}
        kind = "discriminating_fact" if ref.startswith("discriminating_fact.") else "unknown"
        return {"ref": ref, "status": CLAIMED_UNKNOWN, "kind": kind}

    def resolve_rules(self, refs: object) -> list[dict[str, Any]]:
        values = refs if isinstance(refs, list) else []
        return [self.resolve_rule(value) for value in values]

    def resolve_asset(self, raw_ref: object) -> dict[str, Any]:
        ref = str(raw_ref).strip()
        catalog = {str(row["ref"]): row
                   for row in [*self.method_card_refs, *self.operational_instruction_refs]}
        if ref in catalog:
            kind = "method_card" if ref.startswith("card:") else "operational_instruction"
            return {"ref": ref, "status": CLAIMED_AND_VERIFIED, "kind": kind,
                    "content_hash": catalog[ref]["content_hash"]}
        prefix = ref.split(":", 1)[0]
        known_kind = prefix in {"card", "instruction"}
        return {"ref": ref, "status": CLAIMED_UNKNOWN,
                "kind": ("method_card" if prefix == "card" else
                         "operational_instruction" if prefix == "instruction" else "unknown")
                if known_kind else "unknown"}


def _asset_refs(prefix: str, assets: Mapping[str, str] | Iterable[tuple[str, str]]) \
        -> tuple[dict[str, str], ...]:
    items = assets.items() if isinstance(assets, Mapping) else assets
    return tuple({"ref": f"{prefix}:{name}", "content_hash": content_hash(text)}
                 for name, text in items)


def _citation_catalog_text(offered: tuple[dict[str, Any], ...]) -> str:
    """Render the resolver's exact ids beside the same clauses the agent receives."""
    if not offered:
        return ""
    lines = ["", "EXACT GUIDELINE CITATION IDS",
             "Use these exact ids in note_decision.cited_refs; do not invent aliases:"]
    lines.extend(f"  - {row['rule_id']}: {row['text']}" for row in offered)
    return "\n".join(lines)


def _asset_citation_text(method_cards: Mapping[str, str],
                         operational_instructions: Mapping[str, str]) -> str:
    refs = [*(f"card:{name}" for name in method_cards),
            *(f"instruction:{name}" for name in operational_instructions)]
    if not refs:
        return ""
    return ("\n\nEXACT METHOD / OPERATIONAL CITATION IDS\n"
            "Use these exact ids in cited_refs when you actually rely on them:\n"
            + "\n".join(f"  - {ref}" for ref in refs))


def build_task_presentation(
    spec: Any,
    *,
    run_id: str,
    arm_id: str,
    operational_preamble: str,
    method_cards: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    operational_instructions: Mapping[str, str] | Iterable[tuple[str, str]] = (),
) -> tuple[str, ContractSnapshot]:
    """Return the exact prompt and its immutable, content-addressed presentation record."""
    refs = rule_catalog(spec)
    # The detailed renderer is the normative contract.  The requirements-only renderer offers
    # only field/output shapes and abstention meanings; clinical rules and discriminators stay
    # known-but-unoffered so post-hoc verification fails closed.
    requirements_kinds = {"field_format", "field_allowable_values"}
    offered_refs = refs if arm_id in {"policy_bundle", "detailed"} else [
        ref for ref in refs if ref.kind in requirements_kinds]
    offered = tuple({
        **ref.to_dict(with_text=True),
        "rendered_locator": f"task_contract:{ref.view_id or ref.rule_id}",
    } for ref in offered_refs)
    known = tuple({"rule_id": ref.rule_id, "kind": ref.kind, "text_sha": ref.text_sha}
                  for ref in refs)
    extra_ops = dict(operational_instructions)
    extra_ops = {"chart_review_preamble": operational_preamble, **extra_ops}
    card_map = dict(method_cards)
    contract = task_contract_text(spec, arm_id)
    policy_bundle = (compile_policy_bundle(spec, offered)
                     if arm_id == "policy_bundle" else None)
    prompt = (operational_preamble.rstrip() + "\n" + contract
              + _citation_catalog_text(offered)
              + _asset_citation_text(card_map, extra_ops))
    payload: dict[str, Any] = {
        "schema": "acr.task_presentation.v1",
        "run_id": run_id,
        "arm_id": arm_id,
        "task_contract_ref": {
            "id": str(spec.spec_id),
            "version": str(spec.spec_version),
            "content_hash": str(spec.spec_hash),
        },
        "offered_clause_catalog": offered,
        "known_clause_index": known,
        "method_card_refs": _asset_refs("card", card_map),
        "operational_instruction_refs": _asset_refs("instruction", extra_ops),
        "rendered_prompt_artifact_ref": "prompt.txt",
        "prompt_hash": content_hash(prompt),
        "policy_bundle": policy_bundle,
    }
    payload["presentation_hash"] = content_hash(payload)
    return prompt, ContractSnapshot(**payload)
