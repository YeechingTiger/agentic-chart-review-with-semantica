"""Render one run's review chain as a readable sequence of actions, from the first step to the stop.

WHY IT IS NEEDED
----------------
`chain_report` reports the chain's **health** (resolvable ratio, depth, breaks). This tool reports
the chain **itself**: in order, what the model did, what reason it gave, what the gate said, and why
it stopped in the end. A number tells you whether the chain is broken; only the sequence of actions
tells you what this policy looks like.

Run four arms over the same patient, read them side by side, and the shape of the policy is directly
visible — more accurate than any card's description of it, because a card is intent and this is
behaviour.

Usage:
    .venv/bin/python tools/render_chain.py <manifest.json> [--max N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from acr.evaluation import evals as E
from acr.evaluation.evidence_chain import chain_report

#: How each tool reads inside the chain. These names are not for a machine, they are for the reader.
VERB = {
    "list_documents": "inventory", "document_type_summary": "type census",
    "search_notes": "search", "search_documents": "search", "search": "search",
    "read_document": "read", "read_documents_batch": "batch read", "read_section": "read section",
    "submit_answer": "submit", "gate": "gate",
}


def _arg(ev: dict) -> str:
    a = ev.get("args") or {}
    for k in ("q", "query", "term", "note_id", "doc", "doc_type_contains"):
        if a.get(k):
            return str(a[k])[:44]
    if a.get("date_from") or a.get("date_to"):
        return f"{a.get('date_from', '')}..{a.get('date_to', '')}"
    return ""


def render(path: pathlib.Path, max_rows: int = 40) -> None:
    run = E.RunRecord.from_manifest(str(path))
    rep = chain_report(run)
    m = run.manifest
    ans = m.get("answer") or {}

    print(f"\n{'=' * 78}\n{path.parent.name.split('__')[0]:20s} {path.stem}")
    print(f"{'=' * 78}")
    print(f"answer  {ans.get('status')} "
          f"{json.dumps(ans.get('value') or {}, ensure_ascii=False)[:60]}"
          f"  | gate_validated={m.get('gate_validated')}")
    print(f"chain   {rep['n_links']} steps | grounded {rep['n_grounded']}"
          f" | prose {rep['n_prose_only']}"
          f" | unsourced {rep['n_unsourced']} | depth {rep['max_depth']}")
    print(f"stopped {m.get('termination_reason') or '—'}"
          f" | steps {m.get('steps')} | llm calls {(m.get('usage') or {}).get('llm_calls')}")
    print("-" * 78)

    links = rep["links"]
    shown = links[:max_rows]
    for ln in shown:
        ev = next((e for e in run.trace if e.get("seq") == ln["seq"]), {})
        tool = ln["tool"].split(".")[-1]
        verb = VERB.get(tool, tool)
        mark = {"GROUNDED": "→", "PROSE_ONLY": "·", "UNSOURCED": " ",
                "UNRESOLVED_REF": "?", "FORWARD_REF": "!"}[ln["status"]]
        ref = f" ←{ln['ref']}" if ln.get("ref") else ""
        why = (ln["why"] or "").replace("\n", " ")[:52]
        print(f"{ln['seq']:>3} {mark} {verb:<12s} {_arg(ev):<46s}{ref}")
        if why:
            print(f"      └ {why}")
    if len(links) > max_rows:
        print(f"    … {len(links) - max_rows} more steps (raise with --max)")
    print("legend: → pointer resolves   · prose reason only   (blank) no reason   "
          "? does not resolve   ! points at the future")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args()
    for m in a.manifests:
        render(pathlib.Path(m), a.max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
