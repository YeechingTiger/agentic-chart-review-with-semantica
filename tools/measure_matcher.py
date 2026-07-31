#!/usr/bin/env python3
"""What each tolerance in the search matcher actually buys, counted on this corpus.

WHY THIS EXISTS. `corpus._notation_tolerant` folds separators, quote marks and date renderings,
and refuses to fold synonyms. Every one of those is a judgement about how much a matcher should
guess on the model's behalf, and the only honest way to hold that line is to be able to say what
each fold found that a literal match did not — and, just as important, what the refused one
WOULD have found.

Run it after changing the matcher. A tolerance that buys nothing on real text is a tolerance
that will be wrong on text nobody measured.

    PYTHONPATH=src .venv/bin/python tools/measure_matcher.py

TWO NUMBERS PER PROBE, AND THE SECOND IS THE INTERESTING ONE. `literal` is `re.escape`; `folded`
is what the matcher does now. A probe where they are equal is a tolerance this corpus cannot
exercise — which does NOT mean it is worthless, because the synthetic corpus writes ASCII
hyphens and ISO dates by construction and real dictation does not. Those rows are marked, so a
reader does not read "no gain here" as "no gain".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from acr.corpus import _notation_tolerant  # noqa: E402

CORPUS = REPO / "corpus" / "patients"

#: (probe, what tolerance it exercises). Phrases are the ones a run actually issues against
#: STORE.390 and STORE.400, plus one per tolerance the matcher claims.
PROBES: list[tuple[str, str]] = [
    ("non-small cell", "separator: hyphen and line wrap"),
    ("right upper lobe", "separator: line wrap"),
    ("squamous cell carcinoma", "separator: line wrap"),
    ("final diagnosis", "separator: line wrap"),
    ("c/w", "separator: solidus"),
    ("s/p", "separator: solidus"),
    ("patient's", "quote mark"),
    ("2023-04-12", "date rendering"),
    ("2019-03-12", "date rendering"),
    ("adenocarcinoma", "single token — control, must not change"),
    ("cancer", "single token — control, must not change"),
]

#: Terms whose EXPANSION IS REFUSED. Counted so the refusal keeps costing something visible: if
#: `cancer` and `carcinoma` ever came to overlap almost completely on real text, the argument for
#: keeping them apart would be weaker and somebody should be able to see that from here.
SYNONYM_PROBE = ("cancer", "carcinoma", "malignan", "tumor", "tumour", "neoplas", "adenocarcinoma")


def docs() -> list[Path]:
    return sorted(CORPUS.rglob("*.txt"))


def count(paths: list[tuple[Path, str]], pattern: str) -> int:
    rx = re.compile(pattern, re.IGNORECASE)
    return sum(1 for _, text in paths if rx.search(text))


def main() -> None:
    if not CORPUS.is_dir():
        raise SystemExit(f"no corpus at {CORPUS}")
    paths = [(p, p.read_text(encoding="utf-8", errors="replace")) for p in docs()]
    print(f"{len(paths)} documents in {CORPUS}\n")

    print(f"{'probe':<26}{'tolerance':<38}{'literal':>9}{'folded':>9}{'gain':>8}")
    print("-" * 90)
    unexercised = []
    for probe, why in PROBES:
        lit = count(paths, re.escape(probe))
        fold = count(paths, _notation_tolerant(probe))
        mark = "" if fold != lit else "  (not exercised here)"
        if fold == lit and "control" not in why:
            unexercised.append(probe)
        print(f"{probe:<26}{why:<38}{lit:>9}{fold:>9}{fold - lit:>8}{mark}")

    print("\nWHAT IS REFUSED, and what refusing it costs.")
    print("Share of documents containing each term. A fixed synonym list would have to be")
    print("complete to be safe, and these are the numbers saying it cannot be:\n")
    n = len(paths)
    seen_any = 0
    for term in SYNONYM_PROBE:
        c = count(paths, re.escape(term))
        print(f"  {term:<18}{c:>7}{c / n * 100:>8.1f}%")
    rx_any = re.compile("|".join(re.escape(t) for t in SYNONYM_PROBE), re.IGNORECASE)
    seen_any = sum(1 for _, text in paths if rx_any.search(text))
    print(f"  {'ANY of them':<18}{seen_any:>7}{seen_any / n * 100:>8.1f}%")
    print(f"  {'NONE of them':<18}{n - seen_any:>7}{(n - seen_any) / n * 100:>8.1f}%"
          "   <- what a synonym list would still miss")

    if unexercised:
        print("\nTOLERANCES THIS CORPUS CANNOT EXERCISE:", ", ".join(unexercised))
        print("Not evidence they are worthless. This corpus is generated: it writes ASCII")
        print("hyphens, straight quotes and ISO dates by construction. Text that has been")
        print("through Word, a PDF or a dictation front end does not, which is exactly where")
        print("these were added to work — and exactly where nobody here can measure them.")


if __name__ == "__main__":
    main()
