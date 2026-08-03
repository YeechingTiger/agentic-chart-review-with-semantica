"""The six acceptance criteria for Phase A, over a paired run.

    python tools/analyze_candidates.py runs/phaseA

WHAT THIS IS NOT ASKING. Not "did accuracy improve". Phase A's reasoner is an OBSERVER — its
ledger never enters the main loop's prompt and the gate cannot see it — so the candidate arm's
search flow is the baseline's plus one call per turn that recorded evidence. An accuracy
difference here would be noise or cost, not signal, and reading it as signal is how a structural
module gets credited with a decision it did not make.

WHAT IT IS ASKING: did the candidate space stop being implicit? Six numbers, and each one has a
way of being satisfied vacuously that it is written to expose.

  1 COVERAGE          on charts where competing readings genuinely exist, were they listed?
                      Vacuous form: declare two candidates everywhere. Read it against 3.
  2 GROUNDING         does every candidate cite a recorded span, or say it is hypothesis-only?
                      Vacuous form: attach whatever span is nearest. Not detectable here; the
                      number is a floor, and it is labelled as one.
  3 FALSE ALTERNATIVE on clear charts, were candidates manufactured for form's sake?
                      This is the counterweight to 1 and they must be read together.
  4 DISCRIMINATOR     is what would settle the choice written down as a specific missing fact?
                      Vacuous form: "more information is needed". Counted separately.
  5 STABILITY         does a second call UPDATE the ledger, or rewrite it and lose the history?
                      Measured from state_history and created/updated steps, not from prose.
  6 AUTHORITY         did the reasoner reach past its remit? Any refused update, any candidate
                      whose id it invented, any call that touched something else.

Plus the two that decide whether the arm is affordable at all: cost delta and empty-ledger rate.

THE HELD-OUT SPLIT APPLIES HERE TOO. Charts marked `informed_module_design` were designed by
watching runs fail; a coverage number over them is a number over the development set. Reported
apart, same as `analyze_arms.py`.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A discriminator that is not one. The whole point of the field is a SPECIFIC missing fact, and
#: these are what it degrades into. Matched on the whole string, not as a substring, so a
#: discriminator that happens to contain "more" is not caught by accident.
_VACUOUS = re.compile(
    r"^\s*(more (information|evidence|data|context)( is)?( needed| required)?|"
    r"further (review|reading|search)|additional (evidence|information)|"
    r"unclear|unknown|n/?a|none|tbd)\s*[.。]?\s*$", re.IGNORECASE)


def _corpus() -> dict[str, dict]:
    out = {}
    for r in json.loads((ROOT / "corpus" / "index.json").read_text()):
        gt = (r.get("ground_truth") or {}).get("STORE.390.date_of_initial_diagnosis") or {}
        out[r["patient_id"]] = {
            "gold": gt.get("value") or gt.get("status"),
            "held_out": not r.get("informed_module_design", True),
            # A chart where competing readings genuinely exist: it declares a wrong-but-reachable
            # answer. That is the corpus designer's own statement that a second reading is
            # available, written before any run, so coverage is scored against it rather than
            # against what some run happened to find.
            "contested": bool((r.get("expect") or {}).get("naive_answer")),
        }
    return out


def _rows(armdir: str) -> list[dict]:
    out = []
    for m in sorted(glob.glob(f"{armdir}/*.manifest.json")):
        d = json.loads(pathlib.Path(m).read_text(encoding="utf-8"))
        if not d.get("patient_id"):
            continue
        out.append(d)
    return out


def score(armdir: str, corpus: dict) -> dict:
    rows = _rows(armdir)
    if not rows:
        return {}
    n = len(rows)
    on = [d for d in rows if d.get("candidates") is not None]
    s: dict = {"n": n, "n_with_ledger": len(on), "cost": 0.0, "calls": 0, "tokens": 0}
    for d in rows:
        s["cost"] += (d.get("spend") or {}).get("usd") or 0.0
        s["tokens"] += (d.get("usage") or {}).get("total_tokens") or 0
        s["calls"] += len(d.get("candidate_reasoner_calls") or [])

    contested = contested_multi = clear = clear_multi = 0
    grounded = total_c = empty = 0
    disc_specific = disc_vacuous = 0
    rewrites = incremental = 0
    refused = invented = failed_calls = 0
    submitted_undeclared = 0

    for d in on:
        c = d["candidates"]
        cands = c["candidates"]
        pid = d["patient_id"]
        meta = corpus.get(pid, {})
        active = [x for x in cands if x["status"] != "REJECTED"]
        if not cands:
            empty += 1
        if meta.get("contested"):
            contested += 1
            contested_multi += len(active) > 1
        else:
            clear += 1
            clear_multi += len(active) > 1

        for x in cands:
            total_c += 1
            has = bool(x["supporting_evidence_ids"] or x["contradicting_evidence_ids"])
            hypothesis = "hypothesis" in (x.get("label") or "").lower()
            grounded += bool(has or hypothesis)
            for t in x["unresolved_discriminators"]:
                if _VACUOUS.match(str(t)):
                    disc_vacuous += 1
                else:
                    disc_specific += 1
            # An UPDATE is a candidate touched after the step that created it. A ledger that is
            # rewritten every call has every candidate created and updated on the same step.
            if x["updated_at_step"] > x["created_at_step"]:
                incremental += 1
            if x["status"] == "SELECTED" and "never declared" in (x.get("label") or ""):
                submitted_undeclared += 1
        for t in c["open_discriminators"]:
            if _VACUOUS.match(str(t)):
                disc_vacuous += 1
            else:
                disc_specific += 1
        # A candidate that lost its links between calls is the rewrite signature.
        rewrites += sum(1 for x in cands
                        if x["state_history"] and not x["supporting_evidence_ids"]
                        and not x["contradicting_evidence_ids"])
        for k in (d.get("candidate_reasoner_calls") or []):
            refused += len(k.get("refused") or [])
            invented += sum(1 for r in (k.get("refused") or []) if "no candidate" in r)
            failed_calls += (not k.get("ok"))

    s.update(contested=contested, contested_multi=contested_multi,
             clear=clear, clear_multi=clear_multi,
             grounded=grounded, total_c=total_c, empty=empty,
             disc_specific=disc_specific, disc_vacuous=disc_vacuous,
             incremental=incremental, rewrites=rewrites,
             refused=refused, invented=invented, failed_calls=failed_calls,
             submitted_undeclared=submitted_undeclared)

    for group, key in (("held", True), ("info", False)):
        sel = [d for d in rows if corpus.get(d["patient_id"], {}).get("held_out") is key]
        ok = sum(1 for d in sel if _answer(d) == corpus.get(d["patient_id"], {}).get("gold"))
        s[f"{group}_ok"], s[f"{group}_n"] = ok, len(sel)
    return s


def _answer(d: dict):
    a = d.get("answer") or {}
    return (a.get("value") or {}).get("date_of_initial_diagnosis") or a.get("status")


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/phaseA")
    corpus = _corpus()
    arms = {}
    for p in sorted(root.glob("*__*")):
        if p.is_dir() and (sc := score(str(p), corpus)):
            arms[re.sub(r"__.*", "", p.name)] = sc
    if not arms:
        print(f"no arm directories under {root}/")
        return 1

    print(f"{root}\n")
    print(f"{'arm':<22}{'n':>4}{'ledger':>9}{'empty':>7}{'held':>8}{'informed':>10}"
          f"{'calls':>7}{'tokens':>12}")
    for k, a in arms.items():
        print(f"{k:<22}{a['n']:>4}{a['n_with_ledger']:>6}/{a['n']:<2}{a['empty']:>7}"
              f"{a['held_ok']:>5}/{a['held_n']:<2}{a['info_ok']:>7}/{a['info_n']:<2}"
              f"{a['calls']:>7}{a['tokens']:>12,}")

    for k, a in arms.items():
        if not a["n_with_ledger"]:
            continue
        print(f"\n=== {k} — 六条验收标准 ===")
        print(f"  1 coverage        有竞争读法的图上列出多个候选: "
              f"{a['contested_multi']}/{a['contested']}")
        if a["clear"]:
            print(f"  3 false alt.      清楚的图上凭空造出多个候选:   "
                  f"{a['clear_multi']}/{a['clear']}   (1 和 3 必须一起读)")
        else:
            print("  3 false alt.      无法测量:这一批里没有一张『清楚』的图。1 必须和 3 "
                  "一起读,\n                    单独看 1 会奖励到处多列候选 —— "
                  "下一批要混进 SYN0001-0012。")
        print(f"  2 grounding       候选连着证据或标为假设:       "
              f"{a['grounded']}/{a['total_c']}   （下界:没有检查引对没引对）")
        print(f"  4 discriminator   具体的缺失事实 / 泛泛而谈:    "
              f"{a['disc_specific']} / {a['disc_vacuous']}")
        print(f"  5 stability       增量更新的候选 / 疑似被重写:  "
              f"{a['incremental']} / {a['rewrites']}")
        print(f"  6 authority       被拒的更新 {a['refused']}  (其中编造 id {a['invented']})  "
              f"调用失败 {a['failed_calls']}")
        print(f"    提交了自己从没声明过的值:                    {a['submitted_undeclared']}")

    if len(arms) > 1:
        base = min(arms.values(), key=lambda a: a["calls"])
        rich = max(arms.values(), key=lambda a: a["calls"])
        if base is not rich and base["cost"]:
            print(f"\n成本增量: ${rich['cost'] - base['cost']:+.3f}  "
                  f"({(rich['cost'] / base['cost'] - 1) * 100:+.0f}%)   "
                  f"token {rich['tokens'] - base['tokens']:+,}")
    print("\n这一步不问准确率有没有提高。Phase A 的 reasoner 是观察者 —— 账本不进主循环的"
          "提示词,网关也看不见它 —— 所以准确率的差别是噪声或成本,不是信号。问的是:"
          "候选空间有没有从隐含变成可观察。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
