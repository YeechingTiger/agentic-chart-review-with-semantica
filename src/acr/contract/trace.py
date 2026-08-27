"""Stable guideline identifiers used by runtime testimony and Semantica policy projection.

The old repository-level tracer lived here too. Runtime observation is now owned by Langtrace,
so this module keeps only the policy-addressing seam required by the chart-review loop.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .answer_checks import answer_check_rule_id, field_rule_id

RULE_NAMESPACES = (
    "discriminating_fact", "decision_rule", "conflict_rule", "evidence_rule",
    "answer_check", "field_format", "field_allowable_values", "abstention",
    "proof_obligation",
)
_ID_RE = re.compile(
    r"\b(?:" + "|".join(RULE_NAMESPACES)
    + r")\.[A-Za-z0-9_.#:\-]*[A-Za-z0-9_#\-]")


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "x"


def _text_of(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict) and set(raw) == {"if", "then"}:
        return f"IF {raw['if']} THEN {raw['then']}"
    return json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RuleRef:
    rule_id: str
    kind: str
    text: str
    text_sha: str
    enforced_path: str | None = None
    view_id: str | None = None
    ambiguous_id: bool = False

    def to_dict(self, with_text: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "text_sha": self.text_sha,
            "enforced_path": self.enforced_path,
            "view_id": self.view_id,
        }
        if with_text:
            row["text"] = self.text
        if self.ambiguous_id:
            row["ambiguous_id"] = True
        return row


def rule_catalog(spec: Any) -> list[RuleRef]:
    """Return every addressable clause in declaration order with a content fingerprint."""
    rows: list[RuleRef] = []

    def add(rule_id: str, kind: str, raw: Any, view_id: str | None = None) -> None:
        text = _text_of(raw)
        rows.append(RuleRef(rule_id, kind, text, _sha(text), view_id=view_id))

    for fact in getattr(spec, "discriminating_facts", []) or []:
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip():
            fact_id = str(fact["fact_id"]).strip()
            add(f"discriminating_fact.{fact_id}", "discriminating_fact",
                fact.get("asks") or "", f"fact.{fact_id}")
    for index, rule in enumerate(getattr(spec, "decision_rule", []) or [], 1):
        add(f"decision_rule.{index}", "decision_rule", rule, f"rule.{index}")
    for index, rule in enumerate(getattr(spec, "conflict_rules", []) or [], 1):
        add(f"conflict_rule.{index}", "conflict_rule", rule, f"conflict.{index}")

    evidence = getattr(spec, "evidence_rules", None) or {}
    if isinstance(evidence, dict):
        for clause, values in evidence.items():
            sequence = values if isinstance(values, (list, tuple)) else [values]
            view = {"counts_as_evidence": "accept", "does_not_count": "refuse"}.get(
                str(clause))
            for index, value in enumerate(sequence, 1):
                add(f"evidence_rule.{_slug(clause)}.{index}", "evidence_rule", value,
                    f"{view}.{index}" if view else None)

    obligation = getattr(spec, "proof_obligation", None)
    negative = getattr(obligation, "for_negative", None) or {} if obligation else {}
    scopes: list[tuple[str, dict[str, Any]]] = [
        ("", negative if isinstance(negative, dict) else {})]
    if isinstance(negative, dict):
        scopes.extend((f"claim.{claim.get('id')}.", claim)
                      for claim in negative.get("claims") or [])
    for prefix, holder in scopes:
        for stratum in holder.get("strata") or []:
            if not isinstance(stratum, dict):
                continue
            name = str(stratum.get("name", "?"))
            add(f"evidence_rule.stratum.{prefix}{name}.establishes",
                "evidence_rule_stratum", {
                    "stratum": name,
                    "establishes": list(stratum.get("establishes") or []),
                    "match": stratum.get("match") or {
                        "partition_by": stratum.get("partition_by")},
                })
    if obligation is not None:
        for field, groups in (getattr(obligation, "witness_strata", None) or {}).items():
            add(f"evidence_rule.witness.{field}", "evidence_rule_witness",
                {"field": field, "strata": list(groups)}, f"proof.witness.{field}")
        positive = getattr(obligation, "positive_statement", "")
        if positive:
            add("proof_obligation.for_positive", "proof_obligation", positive,
                "proof.positive")
    if isinstance(negative, dict) and negative.get("statement"):
        add("proof_obligation.for_negative", "proof_obligation", negative["statement"],
            "proof.negative")

    for field in getattr(spec, "fields", []) or []:
        if getattr(field, "format", None):
            add(field_rule_id("field_format", field.name), "field_format", field.format,
                f"answer.{field.name}")
        if getattr(field, "allowable_values", None):
            add(field_rule_id("field_allowable_values", field.name),
                "field_allowable_values", list(field.allowable_values),
                f"answer.{field.name}")
    for index, check in enumerate(getattr(spec, "answer_checks", []) or [], 1):
        add(answer_check_rule_id(check, index), "answer_check", check, f"check.{index}")
    for key, value in (getattr(spec, "abstention", None) or {}).items():
        add(f"abstention.{_slug(key)}", "abstention", value, f"refusal.{key}")

    try:
        from .spec import enforced_elements
        paths = {element.path for element in enforced_elements(spec)}
    except Exception:
        paths = set()
    resolved: list[RuleRef] = []
    for row in rows:
        path = None
        if row.kind == "answer_check":
            candidate = f"answer_checks[{row.rule_id.split('.', 1)[1]}]"
            path = candidate if candidate in paths else None
        elif row.kind in {"field_format", "field_allowable_values"}:
            candidate = (
                f"fields[{row.rule_id.split('.', 1)[1]}]."
                f"{row.kind.replace('field_', '')}")
            path = candidate if candidate in paths else None
        elif row.kind == "evidence_rule_stratum":
            tail = row.rule_id[len("evidence_rule.stratum."):-len(".establishes")]
            matches = [candidate for candidate in paths
                       if candidate.endswith(f".strata[{tail.split('.')[-1]}].establishes")]
            path = matches[0] if len(matches) == 1 else None
        resolved.append(RuleRef(row.rule_id, row.kind, row.text, row.text_sha,
                                path, row.view_id))

    seen: dict[str, int] = {}
    final: list[RuleRef] = []
    for row in resolved:
        count = seen.get(row.rule_id, 0) + 1
        seen[row.rule_id] = count
        if count == 1:
            final.append(row)
        else:
            final.append(RuleRef(f"{row.rule_id}#{count}", row.kind, row.text, row.text_sha,
                                 row.enforced_path, row.view_id, ambiguous_id=True))
    return final


def parse_rule_citations(source: Any, known: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return exact known and unknown guideline ids; never fuzzily repair a citation."""
    known_ids = set(known)
    if isinstance(source, (list, tuple, set)):
        text = " ".join(str(value) for value in source)
    elif isinstance(source, dict):
        text = " ".join(str(value) for value in source.values())
    else:
        text = str(source or "")
    recognised: list[str] = []
    unknown: list[str] = []
    for match in _ID_RE.findall(text):
        token = match.rstrip(".:-")
        target = recognised if token in known_ids else unknown
        if token not in target:
            target.append(token)
    return recognised, unknown
