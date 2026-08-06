#!/usr/bin/env python
"""Run the real-data chain end to end and refuse to call a step proven until it produced something.

    python tools/prove_end_to_end.py --dry-run          # every free step, no model, no money
    python tools/prove_end_to_end.py --patients SYN0001 # the whole chain, one chart

WHY THIS EXISTS RATHER THAN THE PROSE IN `docs/NEW_TASK_NEW_DATA.md`. That document describes seven
phases and was verified by hand on 2026-08-03. Prose cannot fail. On 2026-08-06 the chain was walked
by hand again and two steps were broken in ways the 2188-test suite could not see:

  * `build_agent` died at construction — `ToolSurfaceError: the agent carries ['task']`. The harness
    profile that disables the injected subagent is keyed on a provider string the library does not
    publish, and the guess was written against a scripted fake whose class name happened to match.
    Every test drives that fake. A guess about a provider is only falsifiable against the provider.
  * `attribute case` ran, spent 11 of its 12 permitted model calls, and returned UNRESOLVED —
    "model-call limit reached without a gate-valid attribution". The step executed and did not work.

Both are invisible to unit tests and obvious within one real run, which is what this script is.

WHAT "PROVEN" MEANS HERE, AND WHAT IT DOES NOT

Each step asserts on an ARTIFACT it produced, never on an exit code. A command that exits 0 having
written nothing is the failure mode this repo keeps meeting — an inert check reads exactly like a
satisfied one. So `run` must yield a manifest carrying an answer, `score` must yield a row with a
denominator, `attribute` must yield a cause that is not the budget running out.

It does NOT prove the answers are clinically right. It proves the pipeline carries a real chart to a
scored answer and carries a wrong answer to a named cause. Correctness is `eval score`'s business and
its number is printed, not asserted, because a threshold here would turn a measurement into a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"
SPEC_KEY = "STORE.390.date_of_initial_diagnosis"
FIELD = "date_of_initial_diagnosis"

#: Declared here rather than defaulted in the commands, because `evals.DetectorConfig` refuses a
#: default on purpose: "thresholds belong where a reviewer reads them, not buried where they become
#: folklore". This script is a reviewer-readable place.
DETECTORS = ["--min-term-chars", "4", "--max-rejection-repeats", "3",
             "--token-band", "1000,400000", "--turn-band", "1,40"]

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def step(name: str, detail: str, ok: bool) -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return ok


def acr(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "acr.commands.cli", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="SYN0001",
                    help="comma list; the chain runs one spec across these")
    ap.add_argument("--dry-run", action="store_true",
                    help="every step that costs nothing; stop before the first model call")
    ap.add_argument("--max-usd", type=float, default=0.40)
    ap.add_argument("--keep", action="store_true", help="do not delete the work directory")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="acr-e2e-"))
    print(f"work: {work}\n")

    # ---------------------------------------------------------------- free: the corpus is loadable
    r = acr("check-corpus", "--strict")
    step("check-corpus", (r.stdout.strip().splitlines() or ["(no output)"])[-1][:90], r.returncode == 0)

    # ---------------------------------------------------------------- free: the contract is legal
    r = acr("spec", "lint", str(SPEC))
    tail = [x for x in r.stdout.strip().splitlines() if "tier-1" in x]
    step("spec lint", tail[-1][:90] if tail else "no tier-1 line", r.returncode == 0 and bool(tail))

    # ---------------------------------------------------------------- free: an answer key exists
    key = work / "key.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools/answer_key_from_corpus.py"),
                        "--spec-key", SPEC_KEY, "--fields", FIELD, "--out", str(key)],
                       cwd=ROOT, capture_output=True, text=True)
    n_key = len(json.loads(key.read_text())) if key.is_file() else 0
    step("answer key", f"{n_key} keyed instance(s)", n_key > 0)

    if a.dry_run:
        return report(work, a.keep)

    # ---------------------------------------------------------------- paid: the agent answers
    t0 = time.time()
    r = acr("batch", "--spec", str(SPEC), "--patients", a.patients,
            "--out", str(work / "runs"), "--max-usd", str(a.max_usd))
    # `--out X` writes to `X__<timestamp>__<code_sha>`, a SIBLING of X and not X itself. Globbing
    # the requested path found nothing and reported "0 answered" over a batch that had answered.
    mans = sorted(work.rglob("*.manifest.json"))
    answered = []
    for m in mans:
        d = json.loads(m.read_text())
        if (d.get("answer") or {}).get("status"):
            answered.append((m.stem, d["answer"].get("status"), d.get("gate_validated")))
    want = len(a.patients.split(","))
    if not answered:
        print((r.stdout + r.stderr).strip()[-600:])
    step("batch run",
         f"{len(answered)}/{want} answered in {time.time()-t0:.0f}s "
         f"({', '.join(f'{p}:{s}' for p, s, _ in answered) or 'none'})",
         bool(answered) and len(answered) == want)

    # the gap ledger is the plan; a run that never wrote one is not a failure, an absent KEY is
    # `all([])` IS TRUE, and this step passed on zero manifests the first time it ran — the inert
    # check reading exactly like a satisfied one, in a script whose docstring warns about it.
    gaps = [json.loads(m.read_text()).get("open_gaps") for m in mans]
    step("open-gap ledger",
         f"writes per run: {[g.get('n_writes') if isinstance(g, dict) else None for g in gaps]}",
         bool(gaps) and all(isinstance(g, dict) and "n_writes" in g for g in gaps))

    # ---------------------------------------------------------------- paid: the answer is scored
    # `work`, not `work / "runs"` — same sibling-suffix trap as above; `require_run_tree` globs.
    r = acr("eval", "score", "--runs", str(work), "--answer-key", str(key),
            "--fields", FIELD)
    row = [x for x in r.stdout.splitlines() if x.strip().startswith(FIELD)]
    step("eval score", row[-1].strip()[:90] if row else "no scored row", bool(row))

    # ---------------------------------------------------------------- paid: a cause, not a budget
    if not mans:
        step("attribute case", "skipped: no manifest to attribute", False)
        return report(work, a.keep)
    target = mans[0]
    r = acr("attribute", "case", "--run", str(target), "--case-id", "CASE-e2e",
            "--spec", str(SPEC), *DETECTORS,
            "--max-usd", str(max(a.max_usd, 0.8)), "--local-root", str(work / "local"))
    led = work / "local/error-cases/default/attributions.jsonl"
    cause = status = why = ""
    if led.is_file():
        rec = [json.loads(x) for x in led.read_text().splitlines() if x.strip()][-1]
        pc = rec.get("primary_cause") or {}
        cause, status, why = pc.get("cause", ""), pc.get("status", ""), pc.get("rationale", "")
    # THE ASSERTION THAT MATTERS. `UNRESOLVED` because the budget ran out is the step reporting
    # itself, not the run — that is exactly what the 12-call default used to produce.
    step("attribute case", f"{cause}/{status} — {why[:70]}",
         bool(cause) and not why.startswith("model-call limit reached"))

    return report(work, a.keep)


def report(work: Path, keep: bool) -> int:
    bad = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results)} step(s), {len(bad)} failing")
    if not keep:
        print(f"(work kept at {work} — run output never enters the repo)")
    return 1 if bad else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
