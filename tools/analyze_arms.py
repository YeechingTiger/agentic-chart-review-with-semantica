"""Read out any multi-arm run under `runs/<experiment>/`.

    python tools/analyze_arms.py floor
    python tools/analyze_arms.py stopping

One analyser rather than one per experiment: the columns that matter are the same every time,
and a second copy is where the two drift apart.

IT REFUSES TO PRINT A COMPARISON ACROSS MIXED SPEC HASHES. The E3 attempt was contaminated by
an edit to the spec made while the run was in flight — arm 1 ran against one version and the
rest would have run against another, on the exact axis under measurement. Nothing caught it but
a hunch. Now the arithmetic stops instead.

AND IT REFUSES TO FOLD AN INFORMED CHART INTO A HEADLINE NUMBER. SYNX01-06 were designed by
watching runs fail and the search cards were written from the same failures — SYNX06's own
designer note says it tests "precisely the shorter-stem move the policy-reactive card
advises". A card's score on those charts is a score on its own development set, and it was
being printed in the same column as everything else. The two populations are now separated by
`informed_module_design`, counted apart, and the held-out column is the one a claim may rest
on. A REFUSAL AND NOT A FOOTNOTE: the footnote version of this warning has been written twice
in this tree and lost both times.

The held-out denominator is small — six charts — and printing it small is the point. A number
over six charts that were not used to build the thing being measured is worth more than a
number over twenty-four that were.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.contract.strata import spec_declared_keywords
from acr.core.local_artifacts import RUN_RECORD_GLOB
from acr.evaluation import evals as E

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _decision_inputs import UNKEYED  # noqa: E402

#: WAS A HARDCODED SET of STORE.390's five gate-required terms, which made `invented` (terms the run
#: invented) a number about one contract. It comes from the spec now — `spec_declared_keywords`, the
#: same list the runtime seeds a search from, so "invented" means exactly "not declared".


def design_metadata(corpus_root) -> dict[str, dict]:
    """Per-chart DESIGN facts, which no answer key carries and no real corpus has.

    `informed_module_design`, `naive_answer` and `dx_date` describe how a chart was BUILT — they
    exist because the SYNX/SYNK charts were designed by watching the agent fail, and folding them
    into a headline number scores a method on its own training set. They live in the synthetic
    corpus's `index.json` and nowhere else.

    So this returns `{}` on any other corpus, and the caller REFUSES to print the held-out column
    rather than defaulting it. That is this file's existing discipline: a chart that informed a
    method's design is never silently counted as clean, and "we do not know which charts are
    contaminated" is not the same as "none are".
    """
    index = pathlib.Path(corpus_root).parent / "index.json"
    if not index.is_file():
        return {}
    out = {}
    for row in json.loads(index.read_text(encoding="utf-8")):
        out[row["patient_id"]] = {
            # DEFAULTS TO TRUE for a chart written before the flag existed, matching the
            # generator's own default and for the same reason: the failure being guarded
            # against is a contaminated chart silently counted as clean.
            "informed": bool(row.get("informed_module_design", True)),
            # Declared only on the SYNX charts: the date an ordinary pass yields. They are the
            # only charts with a wrong-but-reachable answer, so they are the only ones where
            # "was it steered" is a question with an answer.
            "naive": (row.get("expect") or {}).get("naive_answer"),
            "dx_date": (row.get("expect") or {}).get("dx_date"),
        }
    return out


def terms_of(rec) -> list[str]:
    """Every keyword issued, one entry per term — a multi-term call counts as its terms.

    Counting CALLS does not survive the tool change: one call now carries many keywords, so
    calls/patient fell 69% in E2 with no change in what was looked for.
    """
    out = []
    for t in rec.trace:
        if t.get("kind") != "tool" or "search" not in str(t.get("tool", "")):
            continue
        q = (t.get("args") or {}).get("query")
        out += [str(w).lower() for w in ([q] if isinstance(q, str) else (q or [])) if w]
    return out


def read_ids(rec) -> set[str]:
    out: set[str] = set()
    for t in rec.trace:
        if t.get("kind") != "tool" or "read" not in str(t.get("tool", "")):
            continue
        a = t.get("args") or {}
        for k in ("note_id", "document_id", "doc_id"):
            if a.get(k):
                out.add(str(a[k]))
        for v in (a.get("note_ids") or a.get("document_ids") or []):
            out.add(str(v))
    return out


def arm(path: str, inp, design: dict, required: set[str]) -> dict:
    # `RUN_RECORD_GLOB`, not `SYN*`: the synthetic prefix meant a real cohort's manifests were
    # invisible and every arm resolved to zero records, after which `max()` over an empty sequence
    # raised. `**` because a run directory may nest.
    recs = [E.RunRecord.from_manifest(m)
            for m in sorted(glob.glob(f"{path}/**/{RUN_RECORD_GLOB}", recursive=True))]
    if not recs:
        return {}
    n = len(recs)
    field = inp.fields[0]
    correct = sprung = reached = synx = synx_ok = 0
    held = held_ok = informed = informed_ok = 0
    reads = searches = invented = 0.0
    for r in recs:
        got = inp.coded(r.manifest, field)
        want = inp.want(r.patient_id, field)
        gv = design.get(r.patient_id, {})
        ok = bool(want is not UNKEYED and got == want)
        correct += ok
        if not design:
            pass                      # no design metadata: the split is reported as unknown
        elif gv.get("informed", True):
            informed += 1
            informed_ok += ok
        else:
            held += 1
            held_ok += ok
        ids = read_ids(r)
        reads += len(ids)
        ts = terms_of(r)
        searches += len(ts)
        invented += sum(1 for t in ts if t.strip().lower() not in required)
        if gv.get("naive"):
            synx += 1
            synx_ok += ok
            sprung += bool(got == gv["naive"])
            # Proxy, and stated as one: the establishing document is dated on the gold
            # diagnosis date. True on SYNX05, where the clause sits in
            # Endo-Diab-MD-OP-Progress-Note_2018-11-07 and dx_date is 20181107.
            d = gv.get("dx_date") or ""
            if len(d) == 8 and any(f"{d[:4]}-{d[4:6]}-{d[6:]}" in i for i in ids):
                reached += 1
    return {"n": n, "acc": correct, "design_known": bool(design), "reads": reads / n, "searches": searches / n,
            "invented": invented / n, "synx": synx, "synx_ok": synx_ok,
            "sprung": sprung, "reached": reached,
            "held": held, "held_ok": held_ok,
            "informed": informed, "informed_ok": informed_ok,
            "spec": {r.spec_hash[:12] for r in recs},
            "cost": sum(r.cost_usd or 0 for r in recs)}


def main() -> int:
    import argparse

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _decision_inputs import Inputs, add_arguments

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap, runs_help="experiment name under runs/, or a path to one")
    args = ap.parse_args()

    # A NAME OR A PATH. It was `ROOT/"runs"/exp` only, so an experiment written outside the repo —
    # which `README.md` instructs, and which `runs/` being gitignored makes normal — could not be
    # read at all.
    candidate = pathlib.Path(args.runs)
    root = candidate if candidate.is_dir() else ROOT / "runs" / args.runs
    args.runs = str(root)
    inp = Inputs(args)
    inp.refuse_unless_resolved(needs_key=True)
    design = design_metadata(inp.corpus_root)
    required = {t.strip().lower() for t in spec_declared_keywords(inp.spec)}

    dirs = {re.sub(r"__.*", "", p.name): str(p) for p in sorted(root.glob("*__*")) if p.is_dir()}
    if not dirs:
        print(f"no arm directories under {root}")
        return 1
    arms = {k: a for k, v in dirs.items() if (a := arm(v, inp, design, required))}

    hashes = set().union(*(a["spec"] for a in arms.values())) if arms else set()
    if len(hashes) > 1:
        print(f"Refusing to compare: these arms ran on {len(hashes)} different spec versions "
              f"— {sorted(hashes)}")
        for k, a in arms.items():
            print(f"  {k:<24} {sorted(a['spec'])}")
        print("\nThe spec was edited half way through the run, so the difference carries two "
              "variables. Re-run it; do not read this table.")
        return 2

    if not required:
        # STATED, since otherwise `invented` silently equals `term/pt` and reads like a finding about
        # the runs. It is a fact about the CONTRACT: STORE.390 retired its `required_keywords` on
        # 2026-08-02 ("a switch that reads as enforcement and enforces nothing"), and this file's
        # hardcoded five-term set was never updated — so `invented` undercounted by five phantom terms
        # against a spec that declares none.
        print(f"'invented' = all of them: {inp.spec.spec_id} declares no required_keywords at all, "
              f"so every term is one the run chose itself. That is a fact about the contract, "
              f"not a difference between arms.\n")
    if not design:
        # THE REFUSAL, not a footnote. The held-out column is the only thing separating a method
        # scored on its own training set from a method scored on data it has not seen, and this
        # corpus carries no record of which charts informed which method.
        print(f"Refusing to split the columns: {inp.corpus_root.parent/'index.json'} does not "
              f"exist, so nothing on record says which charts were designed by watching runs "
              f"fail.\n"
              f"Neither 'held-out' nor 'informed' will be printed — not knowing which charts are "
              f"contaminated is not the same as none of them being contaminated.\n")
    print(f"{root}   spec {next(iter(hashes), '?')}   "
          f"${sum(a['cost'] for a in arms.values()):.2f}\n")
    # `invented`, not `invented`: the cell below is `a['invented']`, this file's own comment calls
    # them "terms the run invented", and `docs/MODULE_LADDER_EXPERIMENT.md` names the column
    # `invented/patient`. A header that disagrees with its own data key and with the document
    # reporting it is how a column gets read as the wrong quantity.
    print(f"{'arm':<22}{'n':>3}{'read/pt':>8}{'term/pt':>8}{'invented':>9}"
          f"{'held-out':>11}{'informed':>11}{'sprung':>7}{'reached':>10}")
    for k, a in arms.items():
        held = f"{a['held_ok']}/{a['held']}" if a["held"] else ("—" if a["design_known"] else "?")
        info = f"{a['informed_ok']}/{a['informed']}" if a["informed"] else (
            "—" if a["design_known"] else "?")
        print(f"{k:<22}{a['n']:>3}{a['reads']:>8.1f}{a['searches']:>8.1f}{a['invented']:>9.1f}"
              f"{held:>11}{info:>11}"
              f"{a['sprung']:>4}/{a['synx']:<2}{a['reached']:>7}/{a['synx']:<2}")

    n_held = max(a["held"] for a in arms.values())
    n_info = max(a["informed"] for a in arms.values())
    print()
    if not design:
        # THE REFUSAL ABOVE ALREADY SAID WE DO NOT KNOW. Falling through to "not one is held out"
        # would assert the opposite of it — that there are none — which is a different and equally
        # unfounded claim. A checker that refuses and then answers anyway has refused nothing.
        print("!! The columns were already refused above: with no design metadata, 'how many are "
              "held out' cannot be answered at all. These accuracies can be read neither as "
              "held-out results nor as informed ones.")
    elif not n_held:
        print("!! Not one chart in these runs is held out. Every accuracy above was computed on "
              "the development set of the very thing being measured, and cannot support any "
              "conclusion.")
    else:
        print(f"'held-out', {n_held} chart(s): designed from the contract's clauses alone, with no "
              f"run result ever looked at. A conclusion may rest on this column and no other.")
    if design:
        print(f"'informed', {n_info} chart(s): SYNX/SYNK were designed by watching runs fail and "
              f"the search cards were written from that same batch of failures; nobody has traced "
              f"SYN0001-0012. Reported apart, not folded together.")
    print("'sprung' means something only on charts that declare a naive_answer — those are the "
          "only ones with a wrong-but-reachable answer to be steered into.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
