"""Append one reviewer's assent to one statement, and say when an edit has voided it.

A sign-off dies when the text it approved changes. A reviewer's assent is to a specific
wording; carrying it across an edit would manufacture clinical approval that nobody gave.
Matching is therefore by element hash and not by element id, so a rule that moved keeps its
approval and a rule that was reworded loses it — and STALE is reported rather than dropped,
because "somebody approved a different version of this" is what the next reviewer needs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .statements import Element, elements

SIGNED = "signed"
STALE = "stale"
UNSIGNED = "unsigned"


def _ledger_path(directory: str | Path, spec_id: str) -> Path:
    return Path(directory) / f"{spec_id}.jsonl"


def load_signoffs(directory: str | Path, spec_id: str) -> list[dict]:
    p = _ledger_path(directory, spec_id)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def record_signoff(directory: str | Path, spec, element_id: str, *, reviewer: str,
                   source_path: str | Path | None = None, note: str = "") -> dict:
    """Append one reviewer's assent to one element. Append-only, like every other ledger here.

    Rewriting or de-duplicating this file would destroy the only record that somebody once
    approved a wording that has since changed — which is exactly the history a re-review
    needs to see.
    """
    els = {e.element_id: e for e in elements(spec, source_path=source_path)}
    el = els.get(element_id)
    if el is None:
        import difflib
        near = difflib.get_close_matches(element_id, list(els), n=5, cutoff=0.3)
        raise KeyError(
            f"no element {element_id!r} in {spec.spec_id}. "
            + (f"did you mean {', '.join(near)}? " if near else "")
            + f"ids in this spec: {', '.join(sorted(els))}")
    rec = {
        "signed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reviewer": reviewer,
        "spec_id": spec.spec_id,
        "spec_version": spec.spec_version,
        "spec_hash": spec.spec_hash,
        "element_id": el.element_id,
        "element_kind": el.kind,
        "element_hash": el.element_hash,
        "note": note,
    }
    p = _ledger_path(directory, spec.spec_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def signoff_status(el: Element, signoffs: Sequence[dict]) -> tuple[str, dict | None]:
    """SIGNED only when the approved hash is still the element's hash.

    Matching is by hash and not by element id, so a rule that moved keeps its approval and a
    rule that was reworded loses it. STALE is reported rather than dropped: "somebody
    approved a different version of this" is information the next reviewer needs.
    """
    mine = [s for s in signoffs
            if s.get("spec_id") == el.spec_id and s.get("element_id") == el.element_id]
    exact = [s for s in signoffs
             if s.get("spec_id") == el.spec_id and s.get("element_hash") == el.element_hash]
    if exact:
        return SIGNED, sorted(exact, key=lambda s: s["signed_at"])[-1]
    if mine:
        return STALE, sorted(mine, key=lambda s: s["signed_at"])[-1]
    return UNSIGNED, None
