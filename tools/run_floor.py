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

    python tools/run_floor.py --dry-run      # validate both arms, spend nothing
    python tools/run_floor.py                # 2 arms x 18 charts

THIS SCRIPT HAD NO ARGUMENT PARSER. Every argument was ignored, so
`python tools/run_floor.py --help` — the first thing anybody types at an unfamiliar script —
launched thirty-six real batches. `run_ladder.py` grew a `--dry-run` after the arm table it
validated only at runtime cost two paid arms; the same reasoning applies to a script whose failure
mode is being asked what it does.

ONE VARIABLE IS A CLAIM THIS SCRIPT CANNOT MAKE ON ITS OWN. The header above says "TWO ARMS, one
variable: the runtime profile" — and `--runtime-profile` moves two things at once. It selects the
initial retrieval plan (`plan_from_spec`'s hand-written strata versus `plan_from_patient_inventory`)
AND whether coverage is active from the first model call. So the arms below differ in the plan and
in the policy together, and no result from them can say which half did the work. `--planner` exists
now for exactly that separation; this script passes it explicitly so the confound is visible in the
command rather than hidden in a profile lookup.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _driver_budget import budget_line, budget_report  # noqa: E402

SPEC = "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"

ARMS: list[tuple[str, str]] = [
    ("floor", "guideline-only"),
    ("prior", "current-stratified-coverage"),
]

PATIENTS = ",".join(
    [f"SYN{i:04d}" for i in range(1, 13)] + [f"SYNX{i:02d}" for i in range(1, 7)])


def preflight(arms: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Resolve every arm's runtime profile before the first model call.

    `resolve_runtime_profile` raises on a name that is not registered, so a typo costs nothing
    instead of costing the arms that ran before it. Same remedy as `run_ladder.preflight`, and the
    same reason: a driver that only discovers its own typo at runtime is a driver you must pay to
    get an error message from.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from acr.review.runtime_profiles import resolve_runtime_policy
    out = []
    for name, profile in arms:
        resolve_runtime_policy(profile)
        out.append((name, profile))
    return out


def _planner_axis(patient: str) -> dict:
    """Does `--planner` have anything to vary on this contract? See `planner_axis_is_live`."""
    sys.path.insert(0, str(ROOT / "src"))
    from acr.chartstore.corpus import Corpus
    from acr.contract.spec import load_spec
    from acr.core import site
    from acr.review.coverage_planner import planner_axis_is_live
    return planner_axis_is_live(load_spec(ROOT / SPEC), Corpus(site.corpus_root()).chart(patient))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve both arms, print what would run and what it could cost; "
                         "spend nothing")
    ap.add_argument("--max-usd", type=float, default=3.0,
                    help="PER RUN, which is what `acr batch --max-usd` means. The worst case for "
                         "the whole invocation is arms x charts x this.")
    ap.add_argument("--patients", default=PATIENTS)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--planner", default="",
                    help="'spec-strata' or 'patient-inventory'; empty means whatever the arm's "
                         "runtime profile chooses, which is what every recorded run of this script "
                         "used. Set it to break the plan/policy confound named in the header.")
    ap.add_argument("--out", default="", help="default: runs/floor")
    args = ap.parse_args(argv)

    arms = preflight(ARMS)
    n_charts = len(args.patients.split(","))
    print(f"arms={len(arms)}  charts={n_charts}  seed={args.seed}  "
          f"planner={args.planner or '(the profile decides)'}")
    for name, profile in arms:
        print(f"  {name:<8} {profile}")
    print(budget_line(budget_report(n_arms=len(arms), n_charts=n_charts,
                                    max_usd_per_run=args.max_usd)))
    axis = _planner_axis(args.patients.split(",")[0])
    print(f"planner axis on {SPEC.rsplit('/', 1)[-1]}: "
          f"{'LIVE — ' + ', '.join(axis['differs_in']) if axis['live'] else 'INERT'}")
    if not axis["live"]:
        # NOT A FOOTNOTE. This script's own header claims the spec's plan is a supplied prior worth
        # falsifying; on a contract that declares no strata the two planners build the same surface,
        # so the arms below differ only in the POLICY half and nothing here measures a plan. A
        # warning about that would be read once and then become part of the output format.
        print(f"\n{axis['why']}\n\nRefusing: the arms below would differ in coverage policy and "
              f"spec view only, and this script's header claims the plan is the variable. Point it "
              f"at a contract that declares strata, or read this as a policy experiment and say so.")
        return 2

    # AND the subcommand resolves. An arm table that validates perfectly is useless if the command
    # it is handed to does not exist — the check `run_ladder` added after the same class of failure.
    probe = subprocess.run(
        [sys.executable, "-m", "acr.commands.cli", "batch", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": "src"})
    ok = probe.returncode == 0 and "--patients" in (probe.stdout or "")
    print(f"subcommand 'batch': {'resolves' if ok else 'DOES NOT RESOLVE'}")
    if not ok:
        return 1
    if args.dry_run:
        print("dry run — both arms resolve; nothing spent")
        return 0

    out_root = pathlib.Path(args.out) if args.out else ROOT / "runs" / "floor"
    out_root.mkdir(parents=True, exist_ok=True)
    started, index = time.time(), []
    for name, profile in arms:
        t0 = time.time()
        print(f"\n=== {name} ({profile}) ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "acr.commands.cli", "batch",
             "--spec", SPEC, "--runtime-profile", profile, "--skills", "policy=",
             "--patients", args.patients, "--seed", str(args.seed),
             "--max-usd", str(args.max_usd),
             *(["--planner", args.planner] if args.planner else []),
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
