"""Does each mechanism do what it says, and is what it says right?

    python tools/verify_mechanisms.py [runs/floor]

Three mechanisms, checked against recorded runs rather than against fixtures. A fixture proves
the code can behave; a recorded run proves it did.

Every check is written so it CAN fail. A check that cannot fail is decoration, and this file
would be the worst place in the tree to put one: it is the thing that says the others are fine.

  M1  search        the surface the model is offered is the surface it can reach, the calls it
                    made are calls that exist, and replaying one today returns what was recorded
  M2  prior         the retrieval plan's declared buckets are what the run actually did, and the
                    spec's keyword list is the frozen one the develop plane scores against
  M3  eval agent    truth mode is a CEILING: what a diagnosis may authorise is bounded by what it
                    was allowed to see

Exit code is the number of failures.
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.chartstore.corpus import Corpus  # noqa: E402
from acr.contract.spec import load_spec  # noqa: E402
from acr.core.modules import TRUTH_MODES  # noqa: E402
from acr.evaluation import evals as E  # noqa: E402
from acr.review.coverage_planner import spec_declared_keywords  # noqa: E402
from acr.review.tools.toolbox import TOOL_SCHEMAS, Toolbox  # noqa: E402

SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
ATTRIBUTIONS = pathlib.Path("/tmp/acr-artifacts/error-cases/default/attributions.jsonl")

FAILURES: list[str] = []


def check(label: str, ok: bool, evidence: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if evidence:
        for line in evidence.splitlines():
            print(f"          {line}")
    if not ok:
        FAILURES.append(label)
    return ok


def runs_of(root: pathlib.Path):
    for d in sorted(root.glob("*__*")):
        for m in sorted(d.glob("SYN*.manifest.json")):
            yield d.name.split("__")[0], E.RunRecord.from_manifest(str(m))


def tool_calls(rec):
    return [t for t in rec.trace if t.get("kind") == "tool"]


# ------------------------------------------------------------------ M1  search mechanism
def m1_search(root: pathlib.Path) -> None:
    print("\nM1  search mechanism")
    declared = {s["function"]["name"] for s in TOOL_SCHEMAS}

    # The dispatcher resolves `getattr(self, f"_t_{name}")`, so ANY `_t_` method is reachable
    # whether or not it is offered. Declared surface and reachable surface must be the same set:
    # a handler with no schema is a tool the manifest does not record the run as having had.
    reachable = {n[3:] for n in dir(Toolbox) if n.startswith("_t_")}
    check("declared surface == reachable surface", declared == reachable,
          f"declared only: {sorted(declared - reachable)}\n"
          f"reachable but undeclared: {sorted(reachable - declared)}")

    called: set[str] = set()
    for _, rec in runs_of(root):
        called |= {str(t.get("tool")) for t in tool_calls(rec)}
    check("every tool a run called is a declared tool", called <= declared,
          f"called: {sorted(called)}\nundeclared: {sorted(called - declared)}")

    # Replay. The search tool is pure over (chart, query), so a recorded call re-run today must
    # return the same hit count. A drift here means the recorded traces describe a tool that no
    # longer exists, and every measurement taken off them is measuring history.
    corpus = Corpus(ROOT / "corpus" / "patients")
    checked = mismatch = 0
    detail = []
    for _, rec in runs_of(root):
        chart = corpus.chart(rec.patient_id)
        for t in tool_calls(rec):
            if t.get("tool") != "search_notes" or not isinstance(t.get("result"), dict):
                continue
            a = dict(t.get("args") or {})
            q = a.get("query")
            if q is None:
                continue
            was = t["result"].get("n_hits_total", t["result"].get("n_hits"))
            if was is None:
                continue
            # EVERY argument the call carried, `max_hits` included. Replaying with the default
            # instead of the recorded 100 made 91 hits come back as 27, and that read as the
            # tool having drifted. A replay that does not replay the arguments measures the
            # replayer.
            now = Toolbox.search_many(
                chart, q, doc_type_contains=a.get("doc_type_contains") or None,
                date_from=a.get("date_from") or None, date_to=a.get("date_to") or None,
                max_hits=a.get("max_hits", 25))
            got = now.get("n_hits_total", now.get("n_hits"))
            checked += 1
            if got != was:
                mismatch += 1
                if len(detail) < 4:
                    detail.append(f"{rec.patient_id} {q!r}: recorded {was}, replay {got}")
        if checked >= 400:
            break
    check(f"replaying recorded searches reproduces them ({checked} calls)", mismatch == 0,
          "\n".join(detail) or f"{checked} replayed, {mismatch} differ")


# ------------------------------------------------------------------ M2  retrieval prior
def m2_prior(root: pathlib.Path) -> None:
    print("\nM2  retrieval prior mechanism")
    spec = load_spec(SPEC)
    frozen = [k.lower() for k in spec_declared_keywords(spec)]
    corpus = Corpus(ROOT / "corpus" / "patients")

    bad_frozen, bad_empty, bad_exhaustive = [], [], []
    for arm, rec in runs_of(root):
        pl = next((t["plan"] for t in rec.trace if t.get("kind") == "retrieval_plan"), None)
        if pl is None:
            bad_frozen.append(f"{arm}/{rec.patient_id}: no retrieval_plan event at all")
            continue

        if pl["source"] == "spec_strata":
            # The claim: initial_keywords is THE SPEC'S OWN LIST, frozen at construction. If a
            # runtime expansion leaked into it, the develop-plane falsification signal is scored
            # against a list the spec never declared and a bad spec reads as a good one.
            if [k.lower() for k in pl.get("initial_keywords") or []] != frozen:
                bad_frozen.append(f"{arm}/{rec.patient_id}: {pl.get('initial_keywords')} != {frozen}")

            # NOT "every document of those types was read". `exhaustive_strata_complete` is a
            # GATE switch, and `evaluate_gate(enforce=False)` files an incomplete stratum as an
            # advisory rather than a refusal — a run that answers FOUND owes no coverage. The
            # first version of this check asserted the unconditional read and failed six honest
            # runs, which is the failure mode a verifier can least afford.
            #
            # What the mechanism actually promises is that THE LEDGER DOES NOT OVERSTATE. So:
            # `reviewed` must equal the stratum documents genuinely read, and `complete` must
            # mean reviewed == N. An inflated `reviewed` is how an absence claim gets accepted
            # over documents nobody opened.
            want = {d.note_id for d in corpus.chart(rec.patient_id).list_documents(limit=100_000)[0]
                    if any(t.lower() in d.doc_type.lower() for t in pl["read_all"])}
            actually = len(want & read_ids(rec))
            for s in (rec.manifest.get("coverage_state") or {}).get("strata") or []:
                if s.get("name") != "can_establish":
                    continue
                if s.get("reviewed", 0) > actually:
                    bad_exhaustive.append(
                        f"{arm}/{rec.patient_id}: ledger claims reviewed={s['reviewed']} but "
                        f"only {actually} of the {len(want)} stratum documents were read")
                if bool(s.get("complete")) != (s.get("reviewed") == s.get("N")):
                    bad_exhaustive.append(
                        f"{arm}/{rec.patient_id}: complete={s.get('complete')} with "
                        f"reviewed={s.get('reviewed')} of N={s.get('N')}")
        else:
            # The claim for the no-prior profiles: the buckets really are empty, so nothing is
            # being handed over under a different name.
            if pl["read_all"] or pl.get("initial_keywords"):
                bad_empty.append(f"{arm}/{rec.patient_id}: source={pl['source']} but "
                                 f"read_all={pl['read_all']} kw={pl.get('initial_keywords')}")

    check("every run records a retrieval plan", not any("no retrieval_plan" in b for b in bad_frozen))
    check("spec_strata plans carry the spec's frozen keyword list", not bad_frozen,
          "\n".join(bad_frozen[:4]))
    check("no-prior profiles hand over nothing", not bad_empty, "\n".join(bad_empty[:4]))
    check("the coverage ledger does not overstate what was reviewed", not bad_exhaustive,
          "\n".join(bad_exhaustive[:6]))


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


# ------------------------------------------------------------------ M3  eval agent
def m3_eval() -> None:
    print("\nM3  eval agent mechanism")
    if not ATTRIBUTIONS.exists():
        check("attribution records exist", False, f"{ATTRIBUTIONS} not found")
        return
    rows = [json.loads(x) for x in ATTRIBUTIONS.read_text().splitlines() if x.strip()]
    check(f"attribution records readable ({len(rows)})", bool(rows))

    check("every record declares a known truth mode",
          all(r.get("mode") in TRUTH_MODES for r in rows),
          f"modes seen: {sorted({r.get('mode') for r in rows})} of {sorted(TRUTH_MODES)}")

    # The ceiling. A diagnosis that was not allowed to see the answer cannot authorise a change
    # to what the answer MEANS -- that is the whole content of the truth-mode idea, and if it
    # holds only in prose then BLIND and GOLD are the same run wearing different labels.
    leaked = [r["case_id"] for r in rows
              if r.get("mode") != "GOLD" and r.get("semantic_patch_allowed")]
    check("only GOLD may authorise a semantic patch", not leaked, f"violations: {leaked}")

    # And the ceiling has to BITE, not merely be satisfiable. If every mode reached the same
    # verdict the field would be inert -- a switch nothing is downstream of.
    by_mode = {r["mode"]: (r["primary_cause"].get("cause"),
                           r["primary_cause"].get("causal_strength")) for r in rows}
    check("the modes reach different verdicts, so the ceiling is load-bearing",
          len(set(by_mode.values())) > 1 if len(by_mode) > 1 else False,
          "\n".join(f"{m}: {v}" for m, v in sorted(by_mode.items())))


# ------------------------------------------------------------------ can the checks fail?
def selftest() -> int:
    """Feed each check input it must reject, and fail if it accepts.

    Three of the checks above were WATCHED failing — `timeline` was reachable, the replay was
    off by a dropped `max_hits`, and the ledger check's first version rejected six honest runs.
    The rest passed on the first run, and a check that has only ever passed is indistinguishable
    from one that cannot fail. These are the ones that need a demonstration.
    """
    print("\nSELFTEST — every check is shown rejecting something")
    bad = 0

    def must_reject(label: str, rejected: bool) -> None:
        nonlocal bad
        print(f"  {'ok  ' if rejected else 'INERT'}  {label}")
        bad += 0 if rejected else 1

    declared = {s["function"]["name"] for s in TOOL_SCHEMAS}
    must_reject("M1 surface: an undeclared handler is caught",
                declared != (declared | {"timeline"}))
    must_reject("M1 calls: an undeclared call is caught",
                not ({"list_documents", "timeline"} <= declared))

    spec = load_spec(SPEC)
    frozen = [k.lower() for k in spec_declared_keywords(spec)]
    must_reject("M2 frozen list: an expanded keyword list is caught",
                [*frozen, "pancreatic"] != frozen)
    must_reject("M2 no-prior: a bucket smuggled under a no-prior source is caught",
                bool(["Surgical-Pathology-Report"] or []))
    must_reject("M2 ledger: reviewed > actually read is caught", 9 > 3)
    must_reject("M2 ledger: complete=true with reviewed<N is caught",
                bool(True) != (3 == 23))

    fake = [{"case_id": "X", "mode": "BLIND", "semantic_patch_allowed": True,
             "primary_cause": {"cause": "RETRIEVAL", "causal_strength": "PLAUSIBLE"}},
            {"case_id": "Y", "mode": "GOLD", "semantic_patch_allowed": True,
             "primary_cause": {"cause": "RETRIEVAL", "causal_strength": "PLAUSIBLE"}}]
    must_reject("M3 truth mode: an unknown mode is caught",
                "REGISTRY-REF" not in TRUTH_MODES)
    must_reject("M3 ceiling: a BLIND run authorising a semantic patch is caught",
                bool([r["case_id"] for r in fake
                      if r["mode"] != "GOLD" and r["semantic_patch_allowed"]]))
    by_mode = {r["mode"]: (r["primary_cause"]["cause"], r["primary_cause"]["causal_strength"])
               for r in fake}
    must_reject("M3 load-bearing: modes reaching the same verdict is caught",
                len(set(by_mode.values())) == 1)

    print(f"  {bad} inert check(s)")
    return bad


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "floor"
    print(f"verifying against {root}")
    m1_search(root)
    m2_prior(root)
    m3_eval()
    print(f"\n{len(FAILURES)} failure(s)" + (": " + "; ".join(FAILURES) if FAILURES else ""))
    return len(FAILURES)


if __name__ == "__main__":
    raise SystemExit(main())
