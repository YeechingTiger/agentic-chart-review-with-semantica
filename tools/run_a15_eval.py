"""The frozen A1.5-v1 evaluation run. Locks the protocol before it spends anything.

    python tools/run_a15_eval.py --dry-run
    python tools/run_a15_eval.py

WHY THE PROTOCOL IS WRITTEN FIRST. Everything this round measures is a property of ONE version,
and this tree already contains a document whose every number was produced by a process that
kept changing underneath it. So the commit, the contract hash, the gold annotation, the model,
the case list, the repeat count, the seed and the analyzer version are written to disk BEFORE
the first model call, and `analyze_a15.py` refuses to present numbers as a measurement without
that file.

A FAILURE FOUND DURING THIS ROUND IS RECORDED, NOT FIXED. Fixing while measuring is how a
metric becomes a diary. If the round produces changes worth making, they land as A1.5.1 and the
round is re-run against it.

REPEATS. The seeder is deterministic, so its stability is checked by replay in the analyzer
rather than by spending model calls. The reasoner is not, and SYNY04 has already been observed
getting the same chart right once and wrong twice — so three repeats per case, and both
stabilities are reported apart: mechanism stability must be 100%, LLM stability is a finding.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.contract.spec import load_spec
from acr.core.cli_common import code_sha
from acr.review.run_manifest import chart_hash

SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
RUNTIME_PROFILE = "guideline-only"
SEED = 1234
REPEATS = 3
A15_VERSION = "A1.5-v1"


def cases() -> list[str]:
    """Every chart carrying a candidate-level gold set, in stratum order."""
    idx = json.loads((ROOT / "corpus" / "index.json").read_text())
    order = {"clear": 0, "competing": 1, "no_answer": 2}
    got = [(order[r["candidate_stratum"]], r["patient_id"]) for r in idx
           if r.get("candidate_stratum")]
    return [p for _, p in sorted(got)]


def protocol(pids: list[str]) -> dict:
    spec = load_spec(SPEC)
    return {
        "a15_version": A15_VERSION,
        "analyzer": "a15-analyzer/1",
        "code_sha": code_sha(),
        "spec_id": spec.spec_id, "spec_hash": spec.spec_hash,
        "spec_version": spec.spec_version,
        "runtime_profile": RUNTIME_PROFILE,
        "model": os.environ.get("ACR_MODEL", "(unset)"),
        "seed": SEED, "repeats": REPEATS, "n_cases": len(pids),
        "cases": pids,
        "chart_hashes": {p: chart_hash(ROOT / "corpus" / "patients" / p) for p in pids},
        "gold_annotation": {p: json.loads(
            (ROOT / "corpus" / "patients" / p / "_ground_truth.json").read_text()
        )["gold_candidates"] for p in pids},
        "frozen": ["seeder sources", "value normalisation", "candidate identity",
                   "reasoner prompt", "conflict construction", "discriminator schema"],
        "note": ("A failure found during this round is recorded, not fixed. Fixing while "
                 "measuring turns a metric into a diary."),
        # WHAT THIS BATCH REPLACES, so nobody joins two datasets that are not comparable. The
        # first attempt (5e2ba7b-dirty) had 35 of 42 runs die on a TypeError in
        # `apply_updates`; its reasoner and ledger numbers are void and its directory is
        # marked. The FROZEN SURFACE is unchanged — seeder sources, normalisation, candidate
        # identity, reasoner prompt, conflict construction, discriminator schema are all
        # byte-identical — so this is still A1.5-v1. Only a crash was fixed, in a feature that
        # crashed 100% of the times it was reached and therefore had no measured behaviour.
        "supersedes": {"code_sha": "5e2ba7b-dirty",
                       "why": "35/42 RUNTIME_ERROR; reasoner and ledger numbers void"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/a15eval")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--max-usd", type=float, default=1.0, help="per run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pids = cases()
    out = ROOT / a.out
    pr = protocol(pids)
    pr["repeats"] = a.repeats
    print(json.dumps({k: v for k, v in pr.items()
                      if k not in ("chart_hashes", "gold_annotation")}, indent=2))
    print(f"\n{len(pids)} cases x {a.repeats} repeats = {len(pids) * a.repeats} runs, "
          f"ceiling ${a.max_usd:.2f} each")
    if a.dry_run:
        print("dry run — protocol shown, nothing written, nothing spent")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(pr, indent=2), encoding="utf-8")
    started = time.time()
    for rep in range(a.repeats):
        print(f"\n=== repeat {rep + 1}/{a.repeats} ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "acr.commands.cli", "batch",
             "--spec", str(SPEC.relative_to(ROOT)), "--runtime-profile", RUNTIME_PROFILE,
             "--patients", ",".join(pids), "--seed", str(SEED),
             "--max-usd", str(a.max_usd), "--candidates",
             "--out", str(out / f"rep{rep}")],
            cwd=ROOT, capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONPATH": "src"})
        print((r.stdout or "")[-1200:], flush=True)
        if r.returncode:
            print(f"!! repeat {rep} exited {r.returncode}\n{(r.stderr or '')[-600:]}", flush=True)
    print(f"\nALL DONE in {round((time.time() - started) / 60, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
