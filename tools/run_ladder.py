"""The method-card ladder: what does each card change, measured one slot at a time.

    python tools/run_ladder.py                 # the controller ladder, 3 arms
    python tools/run_ladder.py --group tactic  # one arm per tactic, plus all of them
    python tools/run_ladder.py --group all     # everything
    python tools/run_ladder.py --dry-run       # validate every arm, spend nothing

WHAT THIS FILE USED TO SAY, and why it is worth recording rather than deleting. Until
2026-08-03 the arm table named seven "controllers": `controller-native`,
`controller-breadth-first`, `controller-depth-first`, `controller-breadth-then-depth`,
`controller-latest-first`, `controller-information-gain`. FIVE OF THOSE SIX CARDS DID NOT
EXIST. Package C (commit 02d8c21) split one `search` slot into `controller` + `tactic` and
renamed every card; this driver received `s/search-/controller-/` and nothing else, so five of
seven arms would have failed `SkillStack.validate` — after paying for the first two. The E1/E2
results those old names produced are recorded in `docs/BFS_DFS_SEARCH_PILOT.md` and
`docs/MODULE_LADDER_EXPERIMENT.md`, which deliberately keep the old names because they record
what the arms were called when they ran. A DOCUMENT may be a record. An executable driver that
cannot execute is not a record, it is a landmine, so this one is rewritten.

`preflight()` is the reason it cannot happen again: every arm is assembled and validated before
the first model call, and `--dry-run` is that check on its own. `tests/test_run_ladder_arms.py`
runs it in CI.

WHAT AN ARM IS NOW. One slot varies; everything else is held. The profile is stated rather than
defaulted, because "which coverage policy was in force" is not a detail of a card experiment —
it is the thing a card experiment has to hold constant.

The eighteen charts are the twelve base plus the six SYNX adversarial ones. NOTE, and it is the
reason `tools/analyze_arms.py` refuses to fold them together: the SYNX charts were designed by
watching runs fail and the cards were written from the same failures, so a card's score on them
is a score on its own development set. They are reported separately from any headline number.
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

from acr.contract.skills import SkillStack, parse_skill_stack

SPEC = "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"

#: The subcommand, as a list, and probed by `--dry-run` rather than assumed. `chart` is a
#: COMMAND taking a patient argument, not a group, so `chart batch` parses as `chart` with
#: patient="batch" and fails only once a run is attempted. Written down once, checked once.
BATCH_CMD = ["batch"]

#: Held constant across every arm. Stated, not defaulted: a card ladder that let the coverage
#: policy drift would measure the two together.
RUNTIME_PROFILE = "guideline-only"

#: What every arm starts from, so the only difference between two arms is the clause below.
#: This is what the runtime profiles all resolve to today — a base with no controller and no
#: tactic — which is why the baseline arm's clause is the empty string rather than `controller=`.
BASE = SkillStack(general=("tool-contract", "coverage-judgement"))

_TACTICS = (
    "tactic-query-formulation",
    "tactic-coverage-pool",
    "tactic-follow-dependency",
    "tactic-orient-from-summary",
    "tactic-counterevidence",
    "tactic-sample-unknown-types",
)

#: THE CONTROLLER LADDER. A controller decides what to do next, and exactly one is drawn per
#: run — it is the single variable a controlled arm replaces. Two cards exist.
CONTROLLER_ARMS: list[tuple[str, str]] = [
    ("B0-base", ""),
    ("ctrl-reactive", "controller=controller-reactive"),
    ("ctrl-infogain", "controller=controller-information-gain"),
]

#: THE TACTIC LADDER. A tactic is a move available when its precondition holds, so an arm
#: carrying one measures "was this move called, and did calling it change anything" — not "is
#: this a better policy". `tac-all` is here because the six are meant to compose; if the set is
#: worse than its best member, they are competing rather than composing and that is a finding.
TACTIC_ARMS: list[tuple[str, str]] = (
    [("B0-base", "")]
    + [(f"tac-{t.removeprefix('tactic-')}", f"tactics={t}") for t in _TACTICS]
    + [("tac-all", "tactics=" + "|".join(_TACTICS))]
)

#: THE THIRD FACTOR. Unreachable until 2026-08-03 — `experience` was declared in `SLOTS` and
#: missing from `SkillStack`, so the one card that declares it could not be stacked. The card is
#: a METHOD for working a supplied prior; nothing certified has been through
#: `cli label scan` -> `assets evolve/certify/adopt` yet, so this arm currently measures the
#: method with no prior behind it. Kept separate so nobody reads it as the factor itself.
EXPERIENCE_ARMS: list[tuple[str, str]] = [
    ("B0-base", ""),
    ("experience-method", "experience=experience-adapter"),
]

GROUPS = {"controller": CONTROLLER_ARMS, "tactic": TACTIC_ARMS, "experience": EXPERIENCE_ARMS}

PATIENTS = ",".join(
    [f"SYN{i:04d}" for i in range(1, 13)] + [f"SYNX{i:02d}" for i in range(1, 7)])


def arms_for(group: str) -> list[tuple[str, str]]:
    """The arms to run, de-duplicated on name so `all` does not run B0 three times."""
    if group != "all":
        return GROUPS[group]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for arms in GROUPS.values():
        for name, clause in arms:
            if name not in seen:
                seen.add(name)
                out.append((name, clause))
    return out


def preflight(arms: list[tuple[str, str]]) -> list[tuple[str, SkillStack]]:
    """Assemble and validate every arm before the first model call.

    `parse_skill_stack` validates, so a card that does not exist or sits in the wrong slot
    raises here rather than after two arms have been paid for. This function is the whole
    remedy for what went wrong with the previous arm table.
    """
    out = []
    for name, clause in arms:
        stack = parse_skill_stack(clause, BASE)
        out.append((name, stack))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="controller", choices=[*GROUPS, "all"])
    ap.add_argument("--dry-run", action="store_true",
                    help="validate every arm and print what would run; spend nothing")
    ap.add_argument("--max-usd", type=float, default=3.0, help="per arm")
    ap.add_argument("--patients", default=PATIENTS)
    args = ap.parse_args()

    arms = arms_for(args.group)
    resolved = preflight(arms)
    n_charts = len(args.patients.split(","))
    print(f"group={args.group}  arms={len(arms)}  charts={n_charts}  "
          f"profile={RUNTIME_PROFILE}  ceiling=${args.max_usd:.2f}/arm")
    for name, stack in resolved:
        print(f"  {name:<22} {', '.join(stack.names()) or '(base only)'}")
    if args.dry_run:
        # AND the subcommand resolves. An arm table that validates perfectly is still useless
        # if the command it is handed to does not exist.
        probe = subprocess.run(
            [sys.executable, "-m", "acr.commands.cli", *BATCH_CMD, "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONPATH": "src"})
        ok = probe.returncode == 0 and "--patients" in (probe.stdout or "")
        print(f"\nsubcommand {' '.join(BATCH_CMD)!r}: "
              f"{'resolves' if ok else 'DOES NOT RESOLVE'}")
        if not ok:
            return 1
        print("dry run — every arm assembles and validates; nothing spent")
        return 0

    out_root = ROOT / "runs" / f"ladder-{args.group}"
    out_root.mkdir(parents=True, exist_ok=True)
    started, index = time.time(), []
    for name, clause in arms:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "acr.commands.cli", *BATCH_CMD,
             "--spec", SPEC, "--runtime-profile", RUNTIME_PROFILE,
             "--skills", clause, "--patients", args.patients,
             "--seed", "1234", "--max-usd", str(args.max_usd),
             "--out", str(out_root / name)],
            cwd=ROOT, capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONPATH": "src"})
        print((r.stdout or "")[-400:], flush=True)
        if r.returncode:
            print(f"!! {name} exited {r.returncode}\n{(r.stderr or '')[-600:]}", flush=True)
        index.append({"arm": name, "skills": clause, "profile": RUNTIME_PROFILE,
                      "rc": r.returncode, "minutes": round((time.time() - t0) / 60, 1)})
        (out_root / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nALL DONE in {round((time.time() - started) / 60, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
