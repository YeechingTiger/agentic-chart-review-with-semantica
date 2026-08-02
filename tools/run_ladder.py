"""E1 + E2: the B0 baseline and the search ladder, over base and adversarial charts.

Seven arms x eighteen charts. B0 is the empty search slot — the configuration every historical
run used and that has never once been recorded AS a baseline, which is why every card so far has
been compared against a floor nobody measured.

The eighteen are the twelve original charts plus the six SYNX adversarial ones. The SYNX charts
are the only headroom that exists here: each declares `expect.naive_answer`, the date an ordinary
pass yields, so "did the trap spring" is countable rather than a matter of opinion.

Driven from Python, not a shell loop. zsh does not word-split `$var`, so `for a in $ARMS` hands
one arm whose name is every arm — a trap this repo's own BFS/DFS pilot doc records, and which I
walked into again earlier today.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"

#: B0 first — everything else is measured against it. `search=` with no value clears the slot.
ARMS: list[tuple[str, str]] = [
    ("B0-base", "controller="),
    ("native", "controller=controller-native"),
    ("breadth-first", "controller=controller-breadth-first"),
    ("depth-first", "controller=controller-depth-first"),
    ("breadth-then-depth", "controller=controller-breadth-then-depth"),
    ("information-gain", "controller=controller-information-gain"),
    ("latest-first", "controller=controller-latest-first"),
]

PATIENTS = ",".join(
    [f"SYN{i:04d}" for i in range(1, 13)] + [f"SYNX{i:02d}" for i in range(1, 7)])


def main() -> int:
    out_root = ROOT / "runs" / "ladder"
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    index = []
    for name, skills in ARMS:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "acr.commands.cli", "batch",
             "--spec", SPEC, "--skills", skills, "--patients", PATIENTS,
             "--seed", "1234", "--max-usd", "3.0",
             "--out", str(out_root / name)],
            cwd=ROOT, capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src"})
        tail = (r.stdout or "")[-400:]
        print(tail, flush=True)
        if r.returncode:
            print(f"!! {name} exited {r.returncode}\n{(r.stderr or '')[-600:]}", flush=True)
        index.append({"arm": name, "skills": skills, "rc": r.returncode,
                      "minutes": round((time.time() - t0) / 60, 1)})
        (out_root / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nALL DONE in {round((time.time() - started) / 60, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
