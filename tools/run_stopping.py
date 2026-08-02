"""E3: does a card cost accuracy through its search rule, or through its stopping rule?

Three cards, each in two versions differing only in whether the section stating when the
traversal is COMPLETE is present. Pre-registration and predicted directions are in
`docs/MODULE_LADDER_EXPERIMENT.md` — read them before reading the numbers, because the
directions differ by arm and a table that improves everywhere refutes the hypothesis.

Both members of every pair run today. `depth-first` and `information-gain` already have E2
numbers, but those came off the single-keyword tool surface; reusing them would confound the
manipulation with the tool change.

Driven from Python, not a shell loop — zsh does not word-split `$var`, so `for a in $ARMS`
hands one arm whose name is every arm.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"

#: Paired, and each pair adjacent, so a half-finished run still compares something.
ARMS: list[tuple[str, str]] = [
    ("depth-first-stop", "search=search-depth-first"),
    ("depth-first-nostop", "search=search-depth-first-nostop"),
    ("infogain-stop", "search=search-information-gain"),
    ("infogain-nostop", "search=search-information-gain-nostop"),
    ("breadth-first-stop", "search=search-breadth-first"),
    ("breadth-first-nostop", "search=search-breadth-first-nostop"),
]

PATIENTS = ",".join(
    [f"SYN{i:04d}" for i in range(1, 13)] + [f"SYNX{i:02d}" for i in range(1, 7)])


def main() -> int:
    out_root = ROOT / "runs" / "stopping"
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
            env={**os.environ, "PYTHONPATH": "src"})
        print((r.stdout or "")[-400:], flush=True)
        if r.returncode:
            print(f"!! {name} exited {r.returncode}\n{(r.stderr or '')[-800:]}", flush=True)
        index.append({"arm": name, "skills": skills, "rc": r.returncode,
                      "minutes": round((time.time() - t0) / 60, 1)})
        (out_root / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nALL DONE in {round((time.time() - started) / 60, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
