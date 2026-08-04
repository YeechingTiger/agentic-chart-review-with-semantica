"""Is the agent load-bearing on this variable, or is it decoration for a query?

    python tools/measure_agency.py runs/a15eval

THE QUESTION, and it is the one that decides whether this whole system is worth building for a
given variable. If a variable's evidence is well indexed and the notes use the contract's own
words, you do not need an agent — you need a query. The agent earns its keep in proportion to
how badly the record fights the contract's language, and that is a per-variable fact with a
measurable answer that nobody here has ever computed.

TWO MEASUREMENTS, and they answer different halves.

  A. REACHABILITY (corpus x contract, NO MODEL). Take the contract's own vocabulary — the words
     it uses in its question, decision rules and evidence rules — and search the chart with them.
     Is the document that actually carried the answer among the hits? If yes, one query would
     have found it and everything the agent did was overhead. If no, no amount of querying the
     contract's own words reaches the answer, and something has to read.

     "The document that carried the answer" is not guessed: it is the document a run CITED in a
     run that got the answer right. Grounded in an outcome rather than in my reading of the
     chart.

  B. HOW THE RUN ACTUALLY GOT THERE. Per cited document, from the trace: did a search using a
     CONTRACT word return it, a search using a word the model invented, a type sweep, or nothing
     at all. A run whose decisive evidence all arrived on contract words did nothing a query
     could not.

WHAT WOULD FALSIFY THE PROJECT, stated before looking: if reachability is high and decisive
evidence arrives on contract terms across most variables, then for those variables this system
is an expensive way to run a keyword search, and the honest recommendation is a query.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.chartstore.corpus import Corpus

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _decision_inputs import UNKEYED  # noqa: E402

#: NO HARDCODED SPEC. This was `specs_root()/"STORE.390…"` with no flag, and `docs/
#: NEW_TASK_NEW_DATA.md` step 12 calls this script a mandatory decision point BEFORE investing in a
#: new task — which it could not run on. The contract comes from `--spec`, or is inferred from the
#: runs, which already record it. See `tools/_decision_inputs.py`.

#: Words too common to be a retrieval term in a clinical corpus. A vocabulary that includes
#: "date" or "report" hits every document and would make reachability look total.
_STOP = set(["the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "for", "to", "that", "this", "and", "or", "not", "no", "if", "it", "its", "on", "in", "at", "as", "by", "with", "from", "any", "all", "more", "most", "must", "may", "can", "cannot", "should", "would", "when", "where", "which", "who", "whom", "whose", "what", "how", "why", "then", "than", "there", "here", "their", "them", "they", "you", "your", "we", "our", "use", "used", "using", "first", "second", "later", "earlier", "same", "other", "another", "such", "only", "also", "both", "each", "per", "than", "into", "over", "under", "about", "above", "below", "after", "before", "during", "while", "until", "unless", "whether", "either", "neither", "date", "dates", "day", "days", "month", "months", "year", "years", "time", "times", "record", "records", "document", "documents", "report", "reports", "note", "notes", "text", "case", "cases", "patient", "patients", "chart", "charts", "value", "values", "field", "fields", "statement", "statements", "source", "sources", "evidence", "rule", "rules", "answer", "answers"])


def _show(want) -> str:
    """`None` is the key asserting that abstention is correct — print it as such, not as empty."""
    return "(abstain)" if want is None else str(want)


def contract_vocabulary(spec) -> list[str]:
    """The words the CONTRACT uses. What a query built from the contract alone would search.

    Taken from the question, the decision rules and the evidence rules — the blocks a reader
    would mine for search terms — and not from the strata or keyword lists, because those were
    removed from this contract on purpose and re-introducing them here would measure an asset
    the runtime no longer has.
    """
    text = " ".join([
        str(spec.question),
        *[str(x) for x in (spec.decision_rule or [])],
        *[str(v) for vals in (spec.evidence_rules or {}).values()
          for v in (vals if isinstance(vals, list) else [vals])],
    ])
    words = {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower()) if w not in _STOP}
    return sorted(words)


def tool_events(trace: list[dict]) -> list[dict]:
    return [t for t in trace if t.get("kind") == "tool"]


def how_reached(trace: list[dict], note_id: str, vocab: set[str]) -> str:
    """The FIRST way this document became visible to the run, in event order."""
    for t in tool_events(trace):
        tool, args, res = t.get("tool"), t.get("args") or {}, t.get("result")
        if not isinstance(res, dict):
            continue
        if tool == "search_notes":
            hits = []
            for block in (res.get("by_term") or {}).values():
                hits += [h.get("note_id") for h in (block.get("hits") or [])]
            hits += [h.get("note_id") for h in (res.get("hits") or [])]
            if note_id in hits:
                q = args.get("query")
                terms = [q] if isinstance(q, str) else list(q or [])
                on_contract = [t2 for t2 in terms if str(t2).lower() in vocab]
                return "CONTRACT_TERM" if on_contract else "INVENTED_TERM"
        elif tool in ("list_documents", "document_type_summary"):
            ids = [d.get("note_id") for d in (res.get("documents") or [])]
            if note_id in ids:
                return "TYPE_SWEEP" if args.get("doc_type_contains") else "FULL_INVENTORY"
    return "UNSOURCED"


def main() -> int:
    import argparse

    from _decision_inputs import Inputs, add_arguments, load_trace

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap)
    args = ap.parse_args()
    inp = Inputs(args)
    inp.refuse_unless_resolved(needs_key=True)
    spec = inp.spec
    field = inp.fields[0]
    vocab = contract_vocabulary(spec)
    vset = set(vocab)
    corpus = Corpus(inp.corpus_root)

    print(f"contract vocabulary: {len(vocab)} words")
    print(f"  {', '.join(vocab[:22])}{' ...' if len(vocab) > 22 else ''}\n")

    runs: dict[str, list[dict]] = defaultdict(list)
    for m in inp.manifests:
        d = json.loads(pathlib.Path(m).read_text())
        d["_trace"] = load_trace(m)
        if d.get("patient_id"):
            runs[d["patient_id"]].append(d)

    print(f"{'chart':<9}{'gold':<11}{'right':<7}{'decisive doc':<40}{'reachable?':<12}"
          f"{'how the run got there'}")
    verdicts = []
    for pid in sorted(runs):
        want = inp.want(pid, field)
        if want is UNKEYED:
            continue                    # the key says nothing about this run; not a finding
        ms = runs[pid]

        # THE DECISIVE DOCUMENT, grounded in an outcome: what a run that got it RIGHT cited.
        # `inp.coded` for both halves, so a correct ABSTENTION (key `None`) matches a run that
        # abstained instead of being compared against a status string that can never equal it.
        right = [m for m in ms if inp.coded(m, field) == want]
        cited: dict[str, int] = defaultdict(int)
        for m in right:
            for e in m.get("evidence") or []:
                cited[e["note_id"]] += 1
        if not cited:
            print(f"{pid:<9}{_show(want):<11}{f'0/{len(ms)}':<7}"
                  f"{'(no run got it right — nothing to ground on)':<40}")
            continue
        decisive = max(cited, key=lambda k: cited[k])

        chart = corpus.chart(pid)
        hit = False
        for term in vocab:
            if any(h.note_id == decisive for h in chart.search(term, False, None, None, None,
                                                               max_hits=400)):
                hit = True
                break
        hows = {how_reached(m["_trace"], decisive, vset) for m in right}
        print(f"{pid:<9}{_show(want):<11}{f'{len(right)}/{len(ms)}':<7}{decisive[:38]:<40}"
              f"{('YES' if hit else 'NO'):<12}{','.join(sorted(hows))}")
        verdicts.append((pid, hit, hows))

    n = len(verdicts)
    reach = sum(1 for _, h, _ in verdicts if h)
    contract_only = sum(1 for _, h, hw in verdicts if h and hw <= {"CONTRACT_TERM"})
    print(f"\nreachable by the contract's own words: {reach}/{n}")
    print(f"and the run in fact got there on a contract word alone: {contract_only}/{n}")
    print(f"needed an invented term, a type sweep, or a full inventory: {n - contract_only}/{n}")
    print("\nA chart in the first row is one where a query would have done. A chart in the last "
          "row is where the agent is load-bearing.\nThis is a per-variable fact, and it is the "
          "number that decides whether this system is worth its cost on a given variable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
