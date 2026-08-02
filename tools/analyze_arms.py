"""Read out any multi-arm run under `runs/<experiment>/`.

    python tools/analyze_arms.py floor
    python tools/analyze_arms.py stopping

One analyser rather than one per experiment: the columns that matter are the same every time,
and a second copy is where the two drift apart.

IT REFUSES TO PRINT A COMPARISON ACROSS MIXED SPEC HASHES. The E3 attempt was contaminated by
an edit to the spec made while the run was in flight — arm 1 ran against one version and the
rest would have run against another, on the exact axis under measurement. Nothing caught it but
a hunch. Now the arithmetic stops instead.
"""

from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.evaluation import evals as E  # noqa: E402

#: The spec's five gate-required terms. Anything else the run chose for itself.
REQUIRED = {"biopsy", "carcinoma", "diagnosed", "diagnosis", "malignancy"}


def gold() -> dict[str, dict]:
    out = {}
    for row in json.loads((ROOT / "corpus" / "index.json").read_text()):
        gt = (row.get("ground_truth") or {}).get("STORE.390.date_of_initial_diagnosis") or {}
        out[row["patient_id"]] = {
            "value": gt.get("value") or gt.get("status"),
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


def arm(path: str) -> dict:
    recs = [E.RunRecord.from_manifest(m) for m in sorted(glob.glob(f"{path}/SYN*.manifest.json"))]
    if not recs:
        return {}
    g, n = gold(), len(recs)
    correct = sprung = reached = synx = synx_ok = 0
    reads = searches = invented = 0.0
    for r in recs:
        got = (r.answer or {}).get("value", {}).get("date_of_initial_diagnosis") \
            or (r.answer or {}).get("status")
        gv = g.get(r.patient_id, {})
        ok = bool(got and got == gv.get("value"))
        correct += ok
        ids = read_ids(r)
        reads += len(ids)
        ts = terms_of(r)
        searches += len(ts)
        invented += sum(1 for t in ts if t not in REQUIRED)
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
    return {"n": n, "acc": correct, "reads": reads / n, "searches": searches / n,
            "invented": invented / n, "synx": synx, "synx_ok": synx_ok,
            "sprung": sprung, "reached": reached,
            "spec": {r.spec_hash[:12] for r in recs},
            "cost": sum(r.cost_usd or 0 for r in recs)}


def main() -> int:
    exp = sys.argv[1] if len(sys.argv) > 1 else "floor"
    root = ROOT / "runs" / exp
    dirs = {re.sub(r"__.*", "", p.name): str(p) for p in sorted(root.glob("*__*")) if p.is_dir()}
    if not dirs:
        print(f"runs/{exp}/ 下没有 arm 目录")
        return 1
    arms = {k: a for k, v in dirs.items() if (a := arm(v))}

    hashes = set().union(*(a["spec"] for a in arms.values())) if arms else set()
    if len(hashes) > 1:
        print(f"拒绝比较：这些臂跑在 {len(hashes)} 个不同的 spec 版本上 —— {sorted(hashes)}")
        for k, a in arms.items():
            print(f"  {k:<24} {sorted(a['spec'])}")
        print("\n跑到一半改了 spec，差值里混着两个变量。重跑，不要读这张表。")
        return 2

    print(f"runs/{exp}/   spec {next(iter(hashes), '?')}   "
          f"${sum(a['cost'] for a in arms.values()):.2f}\n")
    print(f"{'arm':<24}{'n':>3}{'读/人':>8}{'词/人':>8}{'自创':>7}"
          f"{'全部准确':>10}{'SYNX 准确':>11}{'中饵':>7}{'读到关键':>10}")
    for k, a in arms.items():
        print(f"{k:<24}{a['n']:>3}{a['reads']:>8.1f}{a['searches']:>8.1f}{a['invented']:>7.1f}"
              f"{a['acc']:>7}/{a['n']:<2}{a['synx_ok']:>8}/{a['synx']:<2}"
              f"{a['sprung']:>4}/{a['synx']:<2}{a['reached']:>7}/{a['synx']:<2}")
    print("\n六张 SYNX 是唯一有『错但够得着的答案』可以被引偏的图，所以『中饵』"
          "只在它们上面有意义。18 张图分不出 1-2 张的差别 —— 这条在跑之前就写死了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
