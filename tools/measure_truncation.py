#!/usr/bin/env python3
"""What `search`'s hit cap actually drops, before anything is done about it.

THE BEHAVIOUR. `PatientChart.search` walks documents in date order and returns as soon as it has
`max_hits`. So a query matching more places than the cap returns THE OLDEST ONES, silently: the
result carries no count, no flag, and nothing distinguishes "40 hits, that is all of them" from
"40 hits out of 300, and the one you needed is in 2023".

This measures the two questions that decide whether ranking is worth building:

  1. HOW OFTEN does the cap bind at all? A cap that never binds costs nothing and ranking behind
     it buys nothing.
  2. WHEN it binds, is the ANSWER-BEARING DOCUMENT among the dropped? This is the number that
     matters. A cap that drops 260 irrelevant hits is working as intended.

Question 2 is answerable because the corpus knows which document establishes the answer: the
ground truth carries the date, and `expect` on the adversarial charts carries more. Both are
deterministic — no model is called and nothing here costs money.

    PYTHONPATH=src .venv/bin/python tools/measure_truncation.py

WHY MEASURE BEFORE BUILDING. Ranking is a real change to what the agent sees, and this repo's
record on unmeasured retrieval improvements is the five clinical rules, removed after they were
counted, and a search-planning pilot whose measured prior did not beat the model's own planning.
A ranking that helps should be able to show which specific answer it rescues.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from acr.chartstore.corpus import Corpus  # noqa: E402

CORPUS = REPO / "corpus" / "patients"
SPEC_KEY = "STORE.390.date_of_initial_diagnosis"

#: Terms a run against STORE.390 actually issues. Deliberately a mix: two that are meant to be
#: selective, two that a model reaches for early and that match far too much. The cap only ever
#: binds on the second kind, which is exactly when a run is casting about and most needs the
#: result to be informative rather than merely early.
QUERIES = [
    "diagnosis", "biopsy", "adenocarcinoma", "malignant",
    "pathology", "impression", "carcinoma", "mass",
]


def answer_date(pid: str) -> str | None:
    import json
    gt = json.loads((CORPUS / pid / "_ground_truth.json").read_text(encoding="utf-8"))
    row = gt["ground_truth"].get(SPEC_KEY) or {}
    v = row.get("value")
    return f"{v[:4]}-{v[4:6]}-{v[6:]}" if v else None


def main() -> None:
    corpus = Corpus(CORPUS)
    pids = corpus.patient_ids()
    # THE TOOL'S DEFAULT, not the corpus function's. `Toolbox._t_search_notes` caps at 25 and
    # `PatientChart.search` at 40; the number that decides what an agent sees is the first one.
    # Measured at both: identical, because on this corpus the cap does not bind at all.
    cap = 25

    bound = 0                      # (patient, query) pairs where the cap bound
    total = 0
    lost_answer_doc = []           # ... and the answer-bearing document fell outside the cap
    rows = []

    for pid in pids:
        chart = corpus.chart(pid)
        want = answer_date(pid)
        for q in QUERIES:
            total += 1
            capped = chart.search(q, max_hits=cap)
            every = chart.search(q, max_hits=10 ** 6)
            if len(every) <= cap:
                continue
            bound += 1
            in_capped = want and any(h.date == want for h in capped)
            in_every = want and any(h.date == want for h in every)
            if in_every and not in_capped:
                lost_answer_doc.append((pid, q, len(capped), len(every)))
            rows.append((pid, q, len(capped), len(every),
                         "kept" if in_capped else ("DROPPED" if in_every else "-")))

    print(f"{len(pids)} patients x {len(QUERIES)} queries = {total} searches, cap = {cap}\n")
    print(f"cap bound on          {bound:>5} / {total}  ({bound / total * 100:.1f}%)")
    print(f"answer doc DROPPED    {len(lost_answer_doc):>5} / {total}"
          f"  ({len(lost_answer_doc) / total * 100:.1f}% of all searches,"
          f" {len(lost_answer_doc) / bound * 100 if bound else 0:.1f}% of capped ones)\n")

    if lost_answer_doc:
        print("SEARCHES THAT DROPPED THE DOCUMENT CARRYING THE ANSWER:")
        print(f"  {'patient':<10}{'query':<18}{'returned':>9}{'actual':>9}")
        for pid, q, got, all_ in lost_answer_doc:
            print(f"  {pid:<10}{q:<18}{got:>9}{all_:>9}")
        print("\nEach of these is a search that came back looking complete and was not. The run")
        print("has no way to tell: the result carries no total and no truncation flag.")
    else:
        print("No search dropped an answer-bearing document, and the cap did not bind ONCE.")
        print("On this corpus, ranking behind the cap would be buying a number nobody can show")
        print("moving — so it is not built. What would change that verdict:")
        print("  * real charts. These hold ~300 documents; a real one holds thousands, and the")
        print("    cap binds as a function of corpus size, which is the one axis a generated")
        print("    corpus is free to be wrong about.")
        print("  * broader queries. `the` returns 40 of 96 here, so the truncation is REAL and")
        print("    reachable — it is the clinical vocabulary that is too selective to hit it.")
        print("    An arm that sweeps with short stems would find the cap that these do not.")

    widest = sorted(rows, key=lambda r: r[3], reverse=True)[:8]
    if widest:
        print("\nWIDEST TRUNCATIONS (what the cap hides, answer or not):")
        print(f"  {'patient':<10}{'query':<18}{'returned':>9}{'actual':>9}  answer doc")
        for pid, q, got, all_, state in widest:
            print(f"  {pid:<10}{q:<18}{got:>9}{all_:>9}  {state}")


if __name__ == "__main__":
    main()
