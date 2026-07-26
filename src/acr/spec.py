"""Extraction specifications: load, validate, freeze.

A spec is the agent's *contract*. It states the decision boundary and the evidentiary
rules, but deliberately NOT the navigation path — how the agent finds the evidence is
its own business; what counts as evidence, and what must be true before it may assert a
negative, are not.

Two fields carry most of the design weight:

  proof_obligation  what must be demonstrably done before a negative/absent answer is allowed
  abstention        two distinct "I can't answer" states, which mean different things:
                      SPEC_INSUFFICIENT      the specification does not cover this case
                      EVIDENCE_INSUFFICIENT  the specification is clear, the chart is not

Every spec is content-hashed. A label is only comparable to another label produced under
the same spec_hash — that is the first of the three ground-truth layers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Status = Literal["FOUND", "EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"]


class ProofObligation(BaseModel):
    model_config = ConfigDict(extra="allow")
    for_positive: str = "A single sufficient piece of evidence is enough."
    for_negative: dict[str, Any] = Field(default_factory=dict)

    @property
    def required_coverage(self) -> list[str]:
        rc = self.for_negative.get("required_coverage", []) if self.for_negative else []
        return list(rc) if isinstance(rc, list) else []

    @property
    def required_keywords(self) -> list[str]:
        kw = self.for_negative.get("required_keywords", []) if self.for_negative else []
        return list(kw) if isinstance(kw, list) else []

    @property
    def required_doc_types(self) -> list[str]:
        dt = self.for_negative.get("required_doc_types_read", []) if self.for_negative else []
        return list(dt) if isinstance(dt, list) else []


class OutputField(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: str = "string"
    format: str | None = None
    allowable_values: list[Any] | None = None
    nullable: bool = True
    description: str = ""


class ExtractionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    spec_id: str
    spec_version: str = "0.1.0"
    source_authority: dict[str, Any] = Field(default_factory=dict)
    question: str
    data_source: Literal["notes", "outside_notes"] = "notes"

    fields: list[OutputField] = Field(default_factory=list)
    decision_rule: list[str] = Field(default_factory=list)
    evidence_rules: dict[str, Any] = Field(default_factory=dict)
    when_not_to_use: list[str] = Field(default_factory=list)
    conflict_rules: list[Any] = Field(default_factory=list)
    proof_obligation: ProofObligation = Field(default_factory=ProofObligation)
    abstention: dict[str, str] = Field(default_factory=dict)
    special_codes_not_mar: list[Any] = Field(default_factory=list)
    boundary_cases: list[Any] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    applicability_guard: dict[str, Any] = Field(default_factory=dict)
    agent_policy: str = ""
    downstream_warning: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------------- freezing
    def canonical(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]

    def identity(self) -> dict:
        return {"spec_id": self.spec_id, "spec_version": self.spec_version, "spec_hash": self.spec_hash}

    # ---------------------------------------------------------------- prompting
    def as_prompt_block(self) -> str:
        """Render the spec for the model. Ordering matters: rules before examples."""
        L: list[str] = [f"# EXTRACTION SPECIFICATION  ({self.spec_id} v{self.spec_version})", ""]
        L += [f"QUESTION: {self.question}", ""]
        if self.data_source == "outside_notes":
            L += [
                "!! DATA SOURCE WARNING: this variable is NOT derivable from clinical notes.",
                "   It lives in institutional registration / follow-up systems.",
                "   You must answer SPEC_INSUFFICIENT. You may report clues you found, as evidence,",
                "   but you must not assign a value.",
                "",
            ]
        if self.agent_policy:
            L += ["AGENT POLICY (binding):", _indent(self.agent_policy), ""]
        if self.fields:
            L.append("OUTPUT FIELDS:")
            for f in self.fields:
                bits = [f"  - {f.name} ({f.type}"]
                if f.format:
                    bits.append(f", format={f.format}")
                bits.append(")")
                line = "".join(bits)
                if f.description:
                    line += f" — {f.description}"
                L.append(line)
                if f.allowable_values:
                    vals = ", ".join(str(v) for v in f.allowable_values[:40])
                    L.append(f"      allowable: {vals}")
            L.append("")
        if self.decision_rule:
            L += ["DECISION RULES:"] + [f"  {i+1}. {r}" for i, r in enumerate(self.decision_rule)] + [""]
        if self.evidence_rules:
            L.append("EVIDENCE RULES:")
            for k, v in self.evidence_rules.items():
                L.append(f"  {k}:")
                for item in (v if isinstance(v, list) else [v]):
                    L.append(f"    - {item}")
            L.append("")
        if self.when_not_to_use:
            L += ["WHEN THIS SPEC DOES NOT APPLY:"] + [f"  - {x}" for x in self.when_not_to_use] + [""]
        if self.conflict_rules:
            L.append("CONFLICT RESOLUTION:")
            for c in self.conflict_rules:
                if isinstance(c, dict):
                    L.append(f"  - IF {c.get('if','?')} THEN {c.get('then','?')}")
                else:
                    L.append(f"  - {c}")
            L.append("")
        L.append("PROOF OBLIGATION:")
        L.append(f"  positive answer: {self.proof_obligation.for_positive}")
        if self.proof_obligation.required_coverage:
            L.append("  BEFORE you may answer negative/absent you MUST have done all of:")
            for r in self.proof_obligation.required_coverage:
                L.append(f"    - {r}")
        if self.proof_obligation.required_keywords:
            L.append(f"  required searches: {', '.join(self.proof_obligation.required_keywords)}")
        if self.proof_obligation.required_doc_types:
            L.append(f"  document types that must be reviewed: {', '.join(self.proof_obligation.required_doc_types)}")
        st = self.proof_obligation.for_negative.get("statement") if self.proof_obligation.for_negative else None
        if st:
            L.append(_indent(st, 2))
        L.append("")
        L.append("ABSTENTION — these are different answers, choose deliberately:")
        for k, v in (self.abstention or {}).items():
            L.append(f"  {k}: {v}")
        L.append("")
        if self.boundary_cases:
            L.append("BOUNDARY CASES (these are settled; follow them):")
            for b in self.boundary_cases:
                L.append(f"  - {json.dumps(b, ensure_ascii=False) if isinstance(b, dict) else b}")
            L.append("")
        if self.search_hints:
            L += ["SEARCH HINTS (suggestions, not a required path):",
                  "  " + ", ".join(self.search_hints), ""]
        return "\n".join(L)


def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + ln for ln in str(s).strip().splitlines())


def load_spec(path: str | Path) -> ExtractionSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExtractionSpec.model_validate(data)


def load_specs(directory: str | Path) -> dict[str, ExtractionSpec]:
    out: dict[str, ExtractionSpec] = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        s = load_spec(p)
        out[s.spec_id] = s
    return out
