"""E4: the floor under every experiment this repo has run.

E1/E2/E3 all compared search cards against `B0-base`, described as "the empty search slot".
It is not empty. Every arm, B0 included, received a retrieval plan built from the spec's
hand-written strata BEFORE its first search — `source: spec_strata`, with document types sorted
into read_all / search / sample and five keywords supplied. On SYNX05 that plan puts the two
documents carrying the bait answer in `read_all` and demotes the type carrying the real one.

So "no card beats every card" was measured as *card on top of a prior* against *prior alone*.
The prior itself has never been on the other side of a comparison, and it is exactly the kind
of asset `assetdev.certify` exists to certify — a claim about where to look — except it is
hand-written YAML and so bypasses the permutation control and the answer-leak filter both.

`coverage_planner.plan_from_spec` says as much in its own docstring: "That is the arm the
develop plane wants to falsify."

TWO ARMS, one variable: the runtime profile.

  prior  current-stratified-coverage  strata plan + 5 keywords + retrieval detail in the prompt
  floor  guideline-only               search_terms=(), required_strata=(), clinical_contract view

Both run with `--skills search=` so no card is in play in either. Nothing else differs.

Predicted before the run: if the prior is doing useful work, `floor` reads more and answers
worse. If the prior is a liability, `floor` matches or beats it on the six SYNX charts, which
are the only ones with a wrong-but-reachable answer to be steered into. A single-patient probe
on SYNX05 returned the gold 20181107 on four self-chosen searches — n=1, and B0 with the prior
also answers that chart correctly, so it discriminates nothing. It is reported because it is
the reason this run exists, not as evidence.

DO NOT EDIT THE SPEC WHILE THIS RUNS. The E3 attempt was contaminated exactly that way: arm 1
ran on one spec hash and the rest would have run on another. `analyze_stopping.py` now collects
`spec_hash` per arm so a mixed run cannot be read as a clean one.
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

ARMS: list[tuple[str, str]] = [
    ("floor", "guideline-only"),
    ("prior", "current-stratified-coverage"),
]

PATIENTS = ",".join(
    [f"SYN{i:04d}" for i in range(1, 13)] + [f"SYNX{i:02d}" for i in range(1, 7)])


def main() -> int:
    out_root = ROOT / "runs" / "floor"
    out_root.mkdir(parents=True, exist_ok=True)
    started, index = time.time(), []
    for name, profile in ARMS:
        t0 = time.time()
        print(f"\n=== {name} ({profile}) ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "acr.commands.cli", "batch",
             "--spec", SPEC, "--runtime-profile", profile, "--skills", "search=",
             "--patients", PATIENTS, "--seed", "1234", "--max-usd", "3.0",
             "--out", str(out_root / name)],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src"})
        print((r.stdout or "")[-400:], flush=True)
        if r.returncode:
            print(f"!! {name} exited {r.returncode}\n{(r.stderr or '')[-800:]}", flush=True)
        index.append({"arm": name, "profile": profile, "rc": r.returncode,
                      "minutes": round((time.time() - t0) / 60, 1)})
        (out_root / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nALL DONE in {round((time.time() - started) / 60, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
