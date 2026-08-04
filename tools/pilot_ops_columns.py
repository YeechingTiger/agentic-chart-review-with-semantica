"""The operational columns of the pilot table — the ones `acr eval score` does not emit.

STRICTLY NON-OVERLAPPING WITH THE SCORER. Accuracy, abstention and gate rates come from
`acr eval score` and are not recomputed here; a second implementation of those is a second
answer to a question that already has one, and the first draft of this pilot's tooling got
SYN0002 wrong that way. What this reads is behaviour that no answer key can settle: how many
documents an arm opened, how many reads it issued, how much of its reading it explained, and
what it cost.

Cost is computed POST HOC from `usage` against `assets/pricing/prices.json`, because the runs were made
before that table existed and their manifests therefore carry `"priced": false, "usd": null`.
Same table, same arithmetic as `spend.py` — but say so in the table, because a number recomputed
downstream and a number the runtime recorded are not the same evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/xinghe/Desktop/agentic-chart-review")
ARMS = ["native", "breadth-first", "depth-first", "breadth-then-depth"]
READ_TOOLS = ("read_document", "read_documents_batch")


def _rates() -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from acr.core.spend import _rates_for
    return _rates_for("gpt-5.6-luna")


def arm_dirs(base: Path, arm: str) -> list[Path]:
    return sorted(p for p in base.glob(f"{arm}__*") if p.is_dir())


def ops(dirs: list[Path], rates: dict) -> dict:
    n = reads = docs = searches = caused = uncaused = 0
    usd = 0.0
    for m in sorted(x for d in dirs for x in d.rglob("*.manifest.json")):
        d = json.loads(m.read_text())
        n += 1
        u = d.get("usage") or {}
        p, c, o = (int(u.get("prompt_tokens", 0)), int(u.get("cached_tokens", 0)),
                   int(u.get("completion_tokens", 0)))
        if rates:
            cached_rate = rates.get("cached_input_per_1m", rates["input_per_1m"])
            usd += ((p - c) * rates["input_per_1m"] + c * cached_rate
                    + o * rates["output_per_1m"]) / 1e6

        tr = m.with_name(m.name.replace(".manifest.json", ".jsonl"))
        if not tr.is_file():
            continue
        seen: set[str] = set()
        for line in tr.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("kind") != "tool":
                continue
            tool = str(e.get("tool") or "").split(".")[-1]
            if tool == "search_notes":
                searches += 1
            if tool not in READ_TOOLS:
                continue
            reads += 1
            if str(e.get("because") or "").strip():
                caused += 1
            else:
                uncaused += 1
            args = e.get("args") or {}
            for nid in ([args["note_id"]] if args.get("note_id") else args.get("note_ids") or []):
                seen.add(str(nid))
        docs += len(seen)
    d_ = max(n, 1)
    return {"n": n,
            "reads/patient": round(reads / d_, 1),
            "docs/patient": round(docs / d_, 1),
            "searches/patient": round(searches / d_, 1),
            "caused-read frac": (round(caused / (caused + uncaused), 3)
                                 if caused + uncaused else None),
            "usd (post hoc)": round(usd, 4),
            "usd/patient": round(usd / d_, 4)}


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs" / "pilot"
    rates = _rates()
    if not rates:
        print("WARNING: no price row for gpt-5.6-luna; cost columns will read 0")
    out = {a: ops(dirs, rates) for a in ARMS if (dirs := arm_dirs(base, a))}
    if not out:
        print(f"no arm directories under {base}")
        return
    cols = list(next(iter(out.values())))
    w = max(len(c) for c in cols) + 2
    print(f"{'':<{w}}" + "".join(f"{a:>22}" for a in out))
    for c in cols:
        print(f"{c:<{w}}" + "".join(f"{out[a][c]!s:>22}" for a in out))
    print("\nAccuracy, abstention and gate rates are not in this table — those belong to "
          "`acr eval score` and are not recomputed here.")


if __name__ == "__main__":
    main()
