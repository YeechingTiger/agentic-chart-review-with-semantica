"""When a run converged on the WRONG value, would more searching have helped?

    python tools/measure_controller_value.py runs/a15eval

THE QUESTION THIS DECIDES. A1.5 was built on the assumption that a Strategic Controller
arbitrates a rich candidate set. The frozen evaluation says that assumption is wrong on these
charts: 15 of 24 competing runs resolve to a single live value, 13 resolved-claims were all
truthful, and a Controller would receive an empty list of open questions. So either the charts
are too easy for a Controller, or a Controller's job is something else — and the something else
would be deciding to KEEP SEARCHING when the candidate set is thin rather than arbitrating one
that is full.

That is answerable from the traces already on disk. For every run that selected the wrong value,
attribute the failure:

  NEVER_LOOKED         the document that carries the answer was never opened.
                       MORE SEARCHING WAS THE REMEDY. A Controller has a job here.
  READ_NOT_CITED       it was opened and nothing was recorded from it. A recognition failure —
                       a Controller could say "you read this and took nothing", which is a
                       weaker but real intervention.
  CITED_BUT_MISJUDGED  the decisive document was cited and the answer is still wrong.
                       MORE SEARCHING WOULD NOT HAVE HELPED. A Controller has no job here.
  GOLD_REJECTED        the gold value was a candidate and the reasoner rejected it. Same:
                       a judgement failure, immune to more retrieval.
  UNSEEDABLE           the gold value exists as no string in the record (a constructed value).
                       Neither searching nor arbitrating reaches it.

THE DECISIVE DOCUMENT IS GROUNDED, NOT GUESSED: it is what runs of the SAME chart that got the
answer RIGHT chose to cite. Where no run got it right, the chart is reported as unattributable
rather than assigned a cause.

If most wrong answers are NEVER_LOOKED, the Controller should be a search-continuation policy.
If most are CITED_BUT_MISJUDGED, no Controller helps and the work belongs in the reasoner.
"""

from __future__ import annotations

import glob
import json
import pathlib
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _answer(m: dict) -> str:
    a = m.get("answer") or {}
    return str((a.get("value") or {}).get("date_of_initial_diagnosis") or a.get("status") or "")


def _read_ids(trace: list[dict]) -> set[str]:
    out: set[str] = set()
    for t in trace:
        if t.get("kind") != "tool" or "read" not in str(t.get("tool") or ""):
            continue
        a = t.get("args") or {}
        if a.get("note_id"):
            out.add(str(a["note_id"]))
        for v in (a.get("note_ids") or []):
            out.add(str(v))
    return out


def _exists_in_corpus(pid: str, value: str) -> bool:
    """Is the gold value written anywhere in this chart, in any notation we can look for?"""
    if len(value) != 8 or not value.isdigit():
        return True                       # a status, not a date
    if value[4:6] == "99" or value[6:] == "99":
        return False                      # a constructed partial date is in no document
    y, mo, d = value[:4], value[4:6], value[6:]
    for form in (f"{y}-{mo}-{d}", f"{mo}/{d}/{y}", f"{int(mo)}/{int(d)}/{y}"):
        r = subprocess.run(["grep", "-rlF", form, str(ROOT / "corpus" / "patients" / pid)],
                           capture_output=True, text=True, check=False)
        if r.stdout.strip():
            return True
    return False


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/a15eval")
    gold = {r["patient_id"]: r for r in json.loads((ROOT / "corpus" / "index.json").read_text())}

    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(glob.glob(f"{root}/**/*.manifest.json", recursive=True)):
        m = json.loads(pathlib.Path(f).read_text())
        tp = pathlib.Path(f.replace(".manifest.json", ".jsonl"))
        m["_trace"] = ([json.loads(x) for x in tp.read_text().splitlines() if x.strip()]
                       if tp.is_file() else [])
        if m.get("patient_id"):
            runs[m["patient_id"]].append(m)

    verdicts: dict[str, list[str]] = defaultdict(list)
    print(f"{'chart':<9}{'picked':<14}{'gold':<14}{'attribution':<22}{'why'}")
    for pid in sorted(runs):
        g = gold.get(pid) or {}
        if not g.get("candidate_stratum"):
            continue
        gt = g["ground_truth"]["STORE.390.date_of_initial_diagnosis"]
        want = gt.get("value") or gt.get("status")
        ms = runs[pid]
        right = [m for m in ms if _answer(m) == want]
        # Grounded in an outcome: what a run that got it right cited.
        decisive = {e["note_id"] for m in right for e in (m.get("evidence") or [])}

        for m in ms:
            got = _answer(m)
            if got == want:
                continue
            led = m.get("candidates") or {}
            cands = {(c["value"].get("date_of_initial_diagnosis") or c["abstention"] or ""): c
                     for c in (led.get("candidates") or [])}
            if not _exists_in_corpus(pid, want):
                v, why = "UNSEEDABLE", "the gold value is written in no document"
            elif want in cands and cands[want]["status"] == "REJECTED":
                v, why = "GOLD_REJECTED", (cands[want].get("rejection_reason") or "")[:52]
            elif not decisive:
                v, why = "UNATTRIBUTABLE", "no run of this chart got it right"
            elif decisive & {e["note_id"] for e in (m.get("evidence") or [])}:
                v, why = "CITED_BUT_MISJUDGED", "cited the decisive document anyway"
            elif decisive & _read_ids(m["_trace"]):
                v, why = "READ_NOT_CITED", "opened it and recorded nothing from it"
            else:
                v, why = "NEVER_LOOKED", f"never opened {sorted(decisive)[:1]}"
            verdicts[v].append(pid)
            print(f"{pid:<9}{got[:12]:<14}{str(want)[:12]:<14}{v:<22}{why}")

    total = sum(len(v) for v in verdicts.values())
    print(f"\n{total} wrong answers across {sum(len(v) for v in runs.values())} runs\n")
    helps = sum(len(verdicts.get(k, [])) for k in ("NEVER_LOOKED", "READ_NOT_CITED"))
    doesnt = sum(len(verdicts.get(k, [])) for k in ("CITED_BUT_MISJUDGED", "GOLD_REJECTED"))
    for k in ("NEVER_LOOKED", "READ_NOT_CITED", "CITED_BUT_MISJUDGED", "GOLD_REJECTED",
              "UNSEEDABLE", "UNATTRIBUTABLE"):
        if verdicts.get(k):
            print(f"  {k:<22}{len(verdicts[k]):>3}   {sorted(set(verdicts[k]))}")
    print(f"\n  more searching WOULD have helped: {helps}/{total}")
    print(f"  more searching would NOT have:    {doesnt}/{total}")
    print("\nThe first number is the size of a Strategic Controller's job on this variable. The "
          "second is work that belongs in the reasoner and that no amount of retrieval policy "
          "reaches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
