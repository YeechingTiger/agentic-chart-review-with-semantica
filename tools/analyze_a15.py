"""The A1.5-v1 frozen evaluation: four layers, scored apart.

    python tools/analyze_a15.py runs/a15eval

WHAT THIS ROUND ASKS, and it is not whether accuracy improved. The reasoner is an observer; its
ledger never enters the main loop's prompt and the gate cannot see it, so an accuracy difference
here is variance. The question is:

    is the candidate state A1.5 produces complete, correct and stable enough to be a
    Strategic Controller's input?

WHY FOUR LAYERS AND NOT ONE NUMBER. A good final ledger can hide two opposite failures — the
seeder never surfaced a candidate, or the seeder surfaced it and the reasoner wrongly deleted
it — and those have different owners and different fixes. Scored together they cancel.

  1 SEEDER      mechanical, no model. Recomputed here from each run's own recorded evidence,
                which also proves the mechanism is deterministic rather than asserting it.
  2 REASONER    what it did to a set it was handed. Gold-answer SURVIVAL is the one that
                matters most: a wrong selection is recoverable by a Controller, a deleted gold
                candidate is not.
  3 LEDGER      is the end state usable — real conflicts visible, false ones absent.
  4 DISCRIMINATOR   six discriminators are not six successes. Canonicalised first.

STRATA ARE REPORTED APART. `clear` is the only place false competition can be measured and
`competing` the only place recall can; one pooled number would let a system that declares
alternatives everywhere look good on one and be invisible on the other. `no_answer` has ONE
chart, so it gets a mechanism description and no performance claim, and this file says so in
its own output rather than leaving it to a reader.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.contract.spec import load_spec
from acr.core.state import CandidateLedger, Evidence, EvidenceLedger
from acr.review.candidate_induction import seed_candidates

ANALYZER_VERSION = "a15-analyzer/1"
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
STRATA = ("clear", "competing", "no_answer")

#: Words that make two discriminators the same question. Canonicalising before counting,
#: because "which date is earlier", "confirm the earliest date" and "identify the first
#: diagnosis date" are one open question and counting them as three rewards verbosity.
_STOP = re.compile(r"\b(the|a|an|is|are|was|were|whether|which|of|for|to|that|this|and|or|"
                   r"be|been|it|its|on|in|at|as|by|with|from|any|more|need|needed|needs|"
                   r"require|required|requires|confirm|determine|establish|identify|clarify)\b",
                   re.IGNORECASE)


def canon(text: str) -> str:
    """A discriminator's question, stripped to the words that carry it."""
    t = _STOP.sub(" ", str(text or "").lower())
    return " ".join(sorted(set(re.findall(r"[a-z0-9]{3,}", t))))


def _corpus() -> dict:
    out = {}
    for r in json.loads((ROOT / "corpus" / "index.json").read_text()):
        gt = (r.get("ground_truth") or {}).get("STORE.390.date_of_initial_diagnosis") or {}
        out[r["patient_id"]] = {
            "gold": gt.get("value") or gt.get("status"),
            "stratum": r.get("candidate_stratum") or "",
            "gold_candidates": list(r.get("gold_candidates") or []),
            "gold_rejections": dict(r.get("gold_rejections") or {}),
            "gold_answerability": r.get("gold_answerability") or "",
            "held_out": not r.get("informed_module_design", True),
        }
    return out


def _value(c: dict) -> str:
    return str((c.get("value") or {}).get("date_of_initial_diagnosis") or "")


# ------------------------------------------------------------------ layer 1: the seeder
def reseed(spec, manifest: dict):
    """Recompute the seeding from THIS RUN'S OWN recorded evidence.

    Two jobs in one pass. It scores the seeder on its own, apart from anything the reasoner
    then did; and because it is deterministic, replaying it and comparing against what the run
    recorded is the mechanism-stability check rather than an assertion that one exists.
    """
    ev = EvidenceLedger()
    for e in manifest.get("evidence") or []:
        x = Evidence(e.get("note_id", ""), e.get("doc_type", ""), e.get("date", ""),
                     int(e.get("start", 0)), int(e.get("end", 0)), e.get("quote", ""),
                     e.get("supports", ""), e.get("stance", "supports"), e.get("entity", ""))
        x.evidence_id = e.get("evidence_id", "")
        x.event_date = e.get("event_date", "")
        x.admissibility = e.get("admissibility", "UNJUDGED")
        ev.add(x)
    led = CandidateLedger()
    seed_candidates(led, spec, ev, step=0)
    return led


def seeder_row(spec, m: dict, gold: dict) -> dict:
    led = reseed(spec, m)
    seeded = {_value(c.to_dict()): c for c in led.candidates}
    gc = set(gold["gold_candidates"])
    by_source: dict[str, set[str]] = defaultdict(set)
    for v, c in seeded.items():
        for src in c.seed_sources:
            by_source[src].add(v)
    # INDEPENDENT MARGINAL RECALL: what would be lost if this source were removed. The only
    # way to answer "what did DOCUMENT_DATE buy" — a source that only ever re-finds values
    # another source also found has bought nothing and still costs the reasoner a rejection.
    marginal = {}
    for src in ("SPAN_LITERAL", "DOCUMENT_DATE", "EVENT_DATE"):
        others = set().union(*[s for k, s in by_source.items() if k != src]) if by_source else set()
        marginal[src] = len((by_source.get(src, set()) & gc) - others)
    return {
        "n_seeded": len(seeded),
        "hit": len(gc & set(seeded)),
        "missed": sorted(gc - set(seeded)),
        "non_target": len(set(seeded) - gc),
        "by_source": {k: {"n": len(v), "gold": len(v & gc), "non_target": len(v - gc),
                          "marginal_recall": marginal[k]}
                      for k, v in sorted(by_source.items())},
        "seeded_values": sorted(seeded),
    }


# ------------------------------------------------------------------ layer 2: the reasoner
def reasoner_row(m: dict, gold: dict, seeded: set[str]) -> dict:
    led = m.get("candidates") or {}
    cands = led.get("candidates") or []
    gc, grej = set(gold["gold_candidates"]), gold["gold_rejections"]
    keep = {v for v in gc if v not in grej}          # gold candidates that must survive
    live = {_value(c) for c in cands if c["status"] in ("ACTIVE", "LEADING", "SELECTED")}
    rejected = {_value(c): c for c in cands if c["status"] == "REJECTED"}

    should_reject = set(grej) | (seeded - gc)        # gold losers + non-target noise
    r_correct = len(set(rejected) & should_reject)
    r_wrong = sorted(set(rejected) & keep)           # a gold keeper the reasoner deleted

    rule_ok = rule_absent = rule_wrong_kind = 0
    for v, c in rejected.items():
        if not str(c.get("rejection_reason") or "").strip():
            rule_absent += 1
        elif c.get("rejecting_rule"):
            rule_ok += 1
        elif v in grej:
            rule_wrong_kind += 1                     # right call, no rule id behind it
    return {
        "live": sorted(live),
        "n_live": len(live),
        "n_rejected": len(rejected),
        "gold_retained": len(keep & live),
        "gold_expected": len(keep),
        "gold_wrongly_rejected": r_wrong,
        "gold_answer_survives": gold["gold"] in live if gold["gold"] else None,
        "reject_correct": r_correct,
        "reject_total": len(rejected),
        "reject_recall_num": len(set(rejected) & set(grej)),
        "reject_recall_den": len(grej),
        "rule_cited": rule_ok, "rule_absent": rule_absent, "no_rule": rule_wrong_kind,
        "not_a_target_flagged": sum(1 for c in rejected.values() if c.get("not_a_target_value")),
    }


# ------------------------------------------------------------------ layer 3: the ledger
def ledger_row(m: dict, gold: dict) -> dict:
    led = m.get("candidates") or {}
    cands = led.get("candidates") or []
    live = [c for c in cands if c["status"] in ("ACTIVE", "LEADING", "SELECTED")]
    live_v = {_value(c) for c in live if _value(c)}
    gc = set(gold["gold_candidates"])
    return {
        "n_live": len(live_v), "n_rejected": len(cands) - len(live),
        "precision_num": len(live_v & gc), "precision_den": len(live_v),
        "recall_num": len(live_v & gc), "recall_den": len(gc),
        "has_conflict": len(led.get("conflict_sets") or []) > 0,
        # FALSE COMPETITION, defined as the user specified: a clear case that ENDS with two or
        # more live, non-equivalent, target-compatible values. Format variants, empty fields,
        # answerability and correctly-rejected candidates are all excluded by construction —
        # `live_v` is the set of distinct asserted values still standing.
        "false_competition": len(live_v) > 1,
        "answerability": led.get("answerability", ""),
        "answerability_ok": led.get("answerability", "") == gold["gold_answerability"],
        "answerability_in_candidates": sum(1 for c in cands if not c.get("value")),
        "hallucinated_value": bool(live_v) and gold["gold_answerability"] != "VALUE_AVAILABLE",
    }


# ------------------------------------------------------------ layer 4: the discriminators
def discriminator_row(m: dict, gold: dict) -> dict:
    led = m.get("candidates") or {}
    ds = led.get("discriminators") or []
    live = {c["candidate_id"] for c in (led.get("candidates") or [])
            if c["status"] in ("ACTIVE", "LEADING", "SELECTED")}
    valid = actionable = 0
    groups: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    for d in ds:
        refs_live = d.get("candidate_a") in live and d.get("candidate_b") in live
        has_fact = bool(str(d.get("unresolved_fact") or "").strip())
        unresolved = d.get("status", "UNRESOLVED") == "UNRESOLVED"
        valid += bool(refs_live and has_fact and unresolved)
        actionable += bool(refs_live and has_fact and unresolved
                           and str(d.get("evidence_needed") or "").strip()
                           and (d.get("likely_source") or [])
                           and d.get("can_be_resolved_from_current_corpus") is not None)
        groups[canon(d.get("unresolved_fact"))] += 1
        by_status[d.get("status", "UNRESOLVED")] += 1
    n_conf = len(led.get("conflict_sets") or [])
    return {
        "n": len(ds), "valid": valid, "actionable": actionable, "by_status": dict(by_status),
        "unique": len(groups),
        "redundancy": (len(ds) / len(groups)) if groups else 0.0,
        "covers_conflict": bool(valid) if n_conf else None,
        "dangling": sum(1 for d in ds
                        if d.get("candidate_a") not in live or d.get("candidate_b") not in live),
    }


# ------------------------------------------------------------------ report
def _pct(num, den) -> str:
    return f"{num}/{den}" + (f" ({100 * num / den:.0f}%)" if den else "")


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/a15eval")
    spec, corpus = load_spec(SPEC), _corpus()
    proto = root / "protocol.json"
    print(f"{root}   analyzer={ANALYZER_VERSION}")
    if proto.exists():
        pr = json.loads(proto.read_text())
        print(f"protocol: commit {pr['code_sha']}  spec {pr['spec_hash'][:12]}  "
              f"model {pr['model']}  {pr['n_cases']} cases x {pr['repeats']} repeats  "
              f"seed {pr['seed']}")
    else:
        print("!! no protocol.json — this run was not locked, and its numbers are not a "
              "measurement of a version.")

    runs: dict[str, list[dict]] = defaultdict(list)
    for m in sorted(glob.glob(f"{root}/**/*.manifest.json", recursive=True)):
        d = json.loads(pathlib.Path(m).read_text())
        if d.get("patient_id") and d.get("candidates") is not None:
            runs[d["patient_id"]].append(d)
    if not runs:
        print("no candidate-arm manifests found")
        return 1

    rows = []
    for pid, ms in sorted(runs.items()):
        g = corpus.get(pid) or {}
        if not g.get("stratum"):
            continue
        for i, m in enumerate(ms):
            s = seeder_row(spec, m, g)
            r = reasoner_row(m, g, set(s["seeded_values"]))
            rows.append({"pid": pid, "rep": i, "stratum": g["stratum"], "gold": g,
                         "seed": s, "reason": r, "ledger": ledger_row(m, g),
                         "disc": discriminator_row(m, g)})

    # ---- mechanism determinism: the seeder must be identical across repeats -------------
    print("\n=== MECHANISM STABILITY (must be 100%) ===")
    # SAME EVIDENCE IN, SAME SEEDING OUT. A first version of this check compared the seeded set
    # ACROSS REPEATS and reported 5/14 — which measured the model choosing different spans to
    # record, not the seeder. The seeder's input is the evidence a run recorded; comparing its
    # output across runs with different inputs is comparing two different questions.
    flaky = replay_mismatch = 0
    for pid, ms in sorted(runs.items()):
        g = corpus.get(pid, {"gold_candidates": []})
        for m in ms:
            a = tuple(seeder_row(spec, m, g)["seeded_values"])
            b = tuple(seeder_row(spec, m, g)["seeded_values"])
            flaky += a != b
            recorded = tuple(sorted({_value(c) for c in (m["candidates"]["candidates"] or [])
                                     if c.get("seed_method")}))
            replay_mismatch += a != recorded
    n = sum(len(v) for v in runs.values())
    print(f"  replaying the same evidence gives the same seeding: {_pct(n - flaky, n)}")
    print(f"  the replay matches what the run itself recorded:    {_pct(n - replay_mismatch, n)}")
    ev_varies = sum(1 for pid, ms in runs.items()
                    if len({tuple(sorted((e["note_id"], e["start"], e["end"])
                                         for e in (m.get("evidence") or []))) for m in ms}) > 1)
    print(f"  (the EVIDENCE the model chose to record varies on {ev_varies}/{len(runs)} cases — "
          f"that is LLM variability and is reported below, not here)")

    inv = {"answerability_in_candidates": sum(r["ledger"]["answerability_in_candidates"]
                                              for r in rows),
           "dangling_discriminator_refs": sum(r["disc"]["dangling"] for r in rows)}
    for k, v in inv.items():
        print(f"  {k}: {v}   {'OK' if v == 0 else 'VIOLATION'}")

    # ---- per stratum ---------------------------------------------------------------------
    for st in STRATA:
        sel = [r for r in rows if r["stratum"] == st]
        if not sel:
            continue
        n_cases = len({r["pid"] for r in sel})
        print(f"\n=== {st.upper()}  ({n_cases} case(s) x "
              f"{len(sel) // max(n_cases, 1)} repeat(s)) ===")
        if n_cases < 3:
            print("  !! too few cases for a performance claim. Mechanism description only.")

        def S(k, f=lambda x: x, _sel=sel):
            """Sum a field over THIS stratum. `_sel` is bound at definition: a late-bound
            closure over the loop variable would score every stratum against the last one."""
            return sum(f(r[k]) for r in _sel)

        print("  SEEDER    gold recall "
              f"{_pct(S('seed', lambda s: s['hit']), S('seed', lambda s: s['hit'] + len(s['missed'])))}"
              f"   burden {S('seed', lambda s: s['n_seeded']) / len(sel):.1f} seeded/run, "
              f"{S('seed', lambda s: s['non_target']) / len(sel):.1f} non-target/run")
        miss = sorted({v for r in sel for v in r["seed"]["missed"]})
        if miss:
            print(f"            never seeded: {miss}")
        by_src: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for r in sel:
            for k, v in r["seed"]["by_source"].items():
                for kk, vv in v.items():
                    by_src[k][kk] += vv
        for k in ("SPAN_LITERAL", "DOCUMENT_DATE", "EVENT_DATE"):
            v = by_src.get(k)
            if v:
                print(f"              {k:<14} n={v['n']:<4} gold={v['gold']:<4} "
                      f"non-target={v['non_target']:<4} marginal_recall={v['marginal_recall']}")

        print("  REASONER  gold retention "
              f"{_pct(S('reason', lambda x: x['gold_retained']), S('reason', lambda x: x['gold_expected']))}"
              f"   gold answer survives "
              f"{_pct(sum(1 for r in sel if r['reason']['gold_answer_survives']), len(sel))}")
        print(f"            rejection precision "
              f"{_pct(S('reason', lambda x: x['reject_correct']), S('reason', lambda x: x['reject_total']))}"
              f"   recall "
              f"{_pct(S('reason', lambda x: x['reject_recall_num']), S('reason', lambda x: x['reject_recall_den']))}")
        wrong = sorted({v for r in sel for v in r["reason"]["gold_wrongly_rejected"]})
        if wrong:
            print(f"            WRONGLY REJECTED a gold candidate: {wrong}")
        print(f"            rejections citing a real contract rule "
              f"{_pct(S('reason', lambda x: x['rule_cited']), S('reason', lambda x: x['reject_total']))}"
              f"   with no reason at all {S('reason', lambda x: x['rule_absent'])}")

        print(f"  LEDGER    candidate precision "
              f"{_pct(S('ledger', lambda x: x['precision_num']), S('ledger', lambda x: x['precision_den']))}"
              f"   recall "
              f"{_pct(S('ledger', lambda x: x['recall_num']), S('ledger', lambda x: x['recall_den']))}")
        if st == "clear":
            print(f"            FALSE COMPETITION "
                  f"{_pct(sum(1 for r in sel if r['ledger']['false_competition']), len(sel))}")
        if st == "competing":
            print(f"            conflict set formed "
                  f"{_pct(sum(1 for r in sel if r['ledger']['has_conflict']), len(sel))}")
        print(f"            answerability correct "
              f"{_pct(sum(1 for r in sel if r['ledger']['answerability_ok']), len(sel))}"
              f"   invented a value where none exists "
              f"{sum(1 for r in sel if r['ledger']['hallucinated_value'])}")

        d_n = S("disc", lambda x: x["n"])
        st_counts: dict[str, int] = defaultdict(int)
        for r in sel:
            for k, v in r["disc"]["by_status"].items():
                st_counts[k] += v
        print(f"  DISCRIM   {d_n} written, {S('disc', lambda x: x['unique'])} distinct questions"
              f"   valid {_pct(S('disc', lambda x: x['valid']), d_n)}"
              f"   actionable {_pct(S('disc', lambda x: x['actionable']), d_n)}")
        if st_counts:
            print(f"            status: {dict(st_counts)}   "
                  f"UNRESOLVED is the only kind a Controller can act on")
        cov = [r for r in sel if r["disc"]["covers_conflict"] is not None]
        if cov:
            print(f"            real conflicts with at least one valid discriminator "
                  f"{_pct(sum(1 for r in cov if r['disc']['covers_conflict']), len(cov))}")

    # ---- LLM stability, per case ----------------------------------------------------------
    print("\n=== LLM STABILITY (per case, across repeats) ===")
    exact = surv = 0
    worst: list[str] = []
    cases = sorted({r["pid"] for r in rows})
    for pid in cases:
        sel = [r for r in rows if r["pid"] == pid]
        lives = {tuple(sorted(r["reason"]["live"])) for r in sel}
        exact += len(lives) == 1
        surv += all(r["reason"]["gold_answer_survives"] for r in sel)
        if any(r["reason"]["gold_wrongly_rejected"] for r in sel):
            worst.append(f"{pid}: deleted a gold candidate in at least one run")
        if any(r["stratum"] == "clear" and r["ledger"]["false_competition"] for r in sel):
            worst.append(f"{pid}: false competition in at least one run")
        if any(r["stratum"] == "competing" and not r["ledger"]["has_conflict"] for r in sel):
            worst.append(f"{pid}: missed the real conflict in at least one run")
    print(f"  live candidate set identical across repeats: {_pct(exact, len(cases))}")
    print(f"  gold answer survived in EVERY repeat:        {_pct(surv, len(cases))}")
    print("  WORST CASE — an occasional error pollutes a Controller's input exactly as a "
          "systematic one does:")
    for w in worst or ["    (none)"]:
        print(f"    {w}")

    # ---- the flow table, one case ---------------------------------------------------------
    print("\n=== CANDIDATE FLOW (first repeat of each competing case) ===")
    for pid in [p for p in cases if (corpus.get(p) or {}).get("stratum") == "competing"]:
        r = next(x for x in rows if x["pid"] == pid)
        m = runs[pid][0]
        g = r["gold"]
        print(f"\n  {pid}   gold answer {g['gold']}")
        print(f"    {'value':<12}{'seed source':<16}{'gold role':<14}{'final':<10}{'rule':<18}")
        for c in (m["candidates"]["candidates"]):
            v = _value(c) or c.get("abstention") or "-"
            role = ("SELECT" if v == g["gold"] else
                    "REJECT" if v in g["gold_rejections"] else
                    "IN-GOLD" if v in g["gold_candidates"] else "NOT_TARGET")
            print(f"    {v:<12}{','.join(c['seed_sources']) or '-':<16}{role:<14}"
                  f"{c['status']:<10}{c.get('rejecting_rule') or '-':<18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
