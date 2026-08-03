"""Build the document bitmaps `improvement/derive.py` prices terms against.

    python tools/build_termcache.py \\
        --labels ~/.acr/devlabels/<run>/labels.jsonl \\
        --spec assets/specs/STORE.400_522_523.site_histology_behavior.yaml \\
        --fields primary_site,histology,behavior

WHY THIS FILE EXISTS. `derive.py:125` says the bitmaps are "built once by
`termcache/build_cache.py`" — and that script is in no repository here. One comment referenced it and
nothing produced it, so `acr derive terms` died on `FileNotFoundError: …/termcache/meta.json` the
first time anyone ran the experience chain end to end. A consumer whose producer does not ship is a
stage that cannot run, and it stayed invisible because reaching it costs money: no test goes here.

WHAT A BITMAP IS FOR. Pricing a retrieval term means asking, for every document in a development
set, whether the term appears in it and whether that document actually bears an answer. Doing that
against the corpus once per candidate term rescans thousands of files per question. Doing it once
into a bit per (document, term) makes every later question set arithmetic. `derive` then refuses to
price a term the cache does not carry, rather than quietly rescanning — which is the right refusal,
and it is why this script's needle list has to be complete before the pricing runs.

## The matcher is the corpus's own, deliberately

A term "appears in" a document exactly when `chartstore.corpus` says it does. Reimplementing that
here with a lowercase substring test would be four lines and wrong in a way nobody would see: the
corpus matcher is notation-tolerant (it splits on a set of separator characters, widens quotes, and
special-cases dates), so `adeno-carcinoma`, `adeno carcinoma` and `adeno_carcinoma` are one term to a
run and would be three to a naive cache. A price computed against a matcher the runtime does not use
is a price for a search nobody performs.

This project has the scar: `tools/analyze_arms.py` reimplemented "which terms did this run search"
and disagreed with `RunRecord.searched_terms`, and the number it printed had already been written
into a document.

## Two fields, and what each one means

`hits` — one integer per document, bit `j` set when needle `j` appears. `oracle` — one bool per
document, true when the LABELLING said that document can establish an answer for one of the fields
under study. The oracle comes from the labelling and not from the ground truth on purpose: this is
the develop plane, and a term priced against the answer key would be a term that scores on data it
will never see again.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import pickle
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.chartstore.corpus import Corpus
from acr.contract.spec import load_spec
from acr.core import site
from acr.review.coverage import strata_from_spec

#: How many patients per pickle chunk. Small enough that a partial build leaves readable chunks,
#: large enough that a corpus of a few hundred patients does not become a few hundred files.
CHUNK = 25


def _nid(doc) -> str:
    """`list_documents` yields dicts on this corpus and objects on others; take either."""
    return str(doc["note_id"] if isinstance(doc, dict) else doc.note_id)


def needles_from(labels_path: pathlib.Path, spec_path: str, min_chars: int) -> list[str]:
    """Every term worth a bit: proposed by the reading model, plus the spec's current list.

    The spec's own keywords are included even when no label proposed them, because pricing is
    RELATIVE — `derive` asks what a candidate buys OVER the list already in the spec, and a
    comparison against a list the cache cannot see is not a comparison.
    """
    proposed: set[str] = set()
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for t in row.get("retrieval_terms") or []:
            term = (t if isinstance(t, str) else t.get("term", "")).strip().lower()
            if len(term) >= min_chars:
                proposed.add(term)
    current = {k.strip().lower()
               for st in strata_from_spec(load_spec(spec_path))
               for k in st.required_keywords}
    return sorted(proposed | current)


def answer_bearing(labels_path: pathlib.Path, fields: list[str]) -> set[tuple[str, str]]:
    """`(patient_id, note_id)` the labelling says can establish one of these fields.

    `can_establish` only. A note that merely MENTIONS the answer is not answer-bearing: a term
    priced on mentions rewards retrieving the places that talk about a thing over the places that
    settle it, which is the failure the whole admissibility vocabulary exists to name.
    """
    out: set[tuple[str, str]] = set()
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        adm = row.get("admissibility") or {}
        per_field = adm.get("fields") or adm.get("verdicts") or {}
        verdicts = ([per_field.get(f) for f in fields] if isinstance(per_field, dict)
                    else [adm.get("verdict")])
        if any(str(v) == "can_establish" for v in verdicts if v):
            out.add((str(row.get("patient_id")), str(row.get("note_id"))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="labels.jsonl from a completed scan")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--fields", required=True, help="comma list, the fields the oracle is about")
    ap.add_argument("--corpus", default=None, help="default: acr.core.site.corpus_root()")
    ap.add_argument("--out", default=None, help="default: acr.core.site.TERMCACHE_ROOT")
    ap.add_argument("--min-term-chars", type=int, default=4)
    ap.add_argument("--patients", default="", help="comma list; default every patient with a label")
    args = ap.parse_args()

    labels = pathlib.Path(args.labels).expanduser()
    out = pathlib.Path(args.out or site.TERMCACHE_ROOT).expanduser()
    corpus = Corpus(pathlib.Path(args.corpus) if args.corpus else site.corpus_root())
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    needles = needles_from(labels, args.spec, args.min_term_chars)
    if not needles:
        print("no needles: the labelling proposed no term and the spec declares none. "
              "Nothing to price, so nothing to cache.", file=sys.stderr)
        return 2
    oracle = answer_bearing(labels, fields)

    labelled = {json.loads(l)["patient_id"]
                for l in labels.read_text(encoding="utf-8").splitlines() if l.strip()}
    pids = ([p.strip() for p in args.patients.split(",") if p.strip()]
            or sorted(labelled))

    (out / "v1").mkdir(parents=True, exist_ok=True)
    for old in (out / "v1").glob("chunk*.pkl.gz"):
        old.unlink()          # a stale chunk beside a fresh one is two answers

    n_docs = n_bits = 0
    chunk: list[dict] = []
    written = 0
    for i, pid in enumerate(pids):
        chart = corpus.chart(pid)
        hits, orc = [], []
        # ONE search per needle over this chart, and the chart's own matcher decides. `search`
        # returns the notes that match, so a set membership test per document is the bit.
        matched: dict[str, set[str]] = {
            n: {h.note_id for h in chart.search(n, False, None, None, None, max_hits=100000)}
            for n in needles}
        # `(rows, total)`, and PAGINATED — the default limit is 100, so a chart of 321
        # documents would silently cache the first hundred and price every term against
        # a third of the evidence.
        docs, total = chart.list_documents(limit=10 ** 9)
        assert len(docs) == total, f"{pid}: paged {len(docs)} of {total} documents"
        for doc in docs:
            mask = 0
            for j, n in enumerate(needles):
                if _nid(doc) in matched[n]:
                    mask |= 1 << j
                    n_bits += 1
            hits.append(mask)
            orc.append((pid, _nid(doc)) in oracle)
            n_docs += 1
        chunk.append({"pid": pid, "hits": hits, "oracle": orc})
        if len(chunk) >= CHUNK or i == len(pids) - 1:
            path = out / "v1" / f"chunk{written:03d}.pkl.gz"
            with gzip.open(path, "wb") as fh:
                pickle.dump(chunk, fh)
            written += 1
            chunk = []

    (out / "meta.json").write_text(json.dumps({
        "schema": "acr.termcache/1",
        "needles": needles,
        "n_patients": len(pids),
        "n_documents": n_docs,
        "spec": load_spec(args.spec).spec_id,
        "fields": fields,
        "labels": str(labels),
        # The matcher's identity, so a cache built against a different one is visible rather than
        # merely wrong. `derive` compares nothing today; a reader can.
        "matcher": "acr.chartstore.corpus.PatientChart.search",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(needles)} needles x {n_docs} documents over {len(pids)} patients "
          f"-> {written} chunk(s) in {out}")
    print(f"  {n_bits} bits set, {len(oracle)} answer-bearing document(s) per the labelling")
    if not oracle:
        print("  WARNING: the labelling marked NOTHING can_establish for these fields. Every term "
              "will price at zero answers rescued, which is a fact about the labelling and not "
              "about the terms.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
