#!/usr/bin/env python3
"""Scan one recorded run for identifiers, and say where each one sits.

A `kind: script` skill: the door is the skill, the rules stay where they live. Every pattern comes
from `acr.audit.audit_loop.PHI_PATTERNS`, which is also what the in-process audit rule uses, so this
cannot drift from it — a second copy of a PHI pattern set is two answers to "is this clean".
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from acr.audit.audit_loop import PHI_PATTERNS  # noqa: E402
from acr.core import site  # noqa: E402

#: Which field a hit landed in decides what it costs. Anything not named here is a document body,
#: which is the corpus being the corpus and is not a finding.
LOCATION = {
    "content": "MODEL_OUTPUT", "answer": "MODEL_OUTPUT", "reasoning": "MODEL_OUTPUT",
    "args": "TOOL_ARGUMENT", "path": "ARTIFACT_PATH", "source": "ARTIFACT_PATH",
}
SEVERITY = {"MODEL_OUTPUT": "IRB", "ARTIFACT_PATH": "IRB", "TOOL_ARGUMENT": "WARN"}


def fingerprint(value: str) -> str:
    """Enough to compare two hits, never enough to be the identifier.

    Keyed when a key is available. Unkeyed, a digest of a value drawn from a known shape is a
    lookup table rather than a protection, so the report says which it got.
    """
    key = os.environ.get("ACR_PSEUDONYM_KEY", "")
    if key:
        return "hmac:" + hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()[:12]
    return "unkeyed:" + hashlib.sha256(value.encode()).hexdigest()[:12]


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def main() -> int:
    target = os.environ.get("ACR_SKILL_TRACE") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not target:
        print(json.dumps({"error": "set ACR_SKILL_TRACE to a .jsonl trace or a .manifest.json"}))
        return 2
    p = pathlib.Path(target)
    if p.name.endswith(".manifest.json"):
        p = p.with_name(p.name.replace(".manifest.json", ".jsonl"))
    if not p.is_file():
        print(json.dumps({"error": f"no trace at {p}"}))
        return 2

    inert = [name for name, pat in PHI_PATTERNS.items() if pat is None]
    findings = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for field, text in walk(event):
            top = field.split(".")[0].split("[")[0]
            location = LOCATION.get(top)
            if location is None:
                continue
            for rule, pattern in PHI_PATTERNS.items():
                if pattern is None:
                    continue
                for m in pattern.finditer(text):
                    findings.append({
                        "rule": rule, "seq": event.get("seq"), "kind": event.get("kind"),
                        "field": field, "location": location,
                        "severity": SEVERITY.get(location, "INFO"),
                        "fingerprint": fingerprint(m.group(0)),
                    })

    print(json.dumps({
        "schema": "acr.phi_trace_scan/1",
        "trace": str(p),
        "findings": findings,
        "counts": {s: sum(1 for f in findings if f["severity"] == s) for s in ("IRB", "WARN")},
        # An inert rule and a clean rule print the same zero. Saying which is which is the
        # difference between "nothing is there" and "we could not look".
        "inert_rules": inert,
        "person_id_pattern_configured": bool(site.PERSON_ID_PATTERN),
    }, indent=2))
    return 1 if any(f["severity"] == "IRB" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
