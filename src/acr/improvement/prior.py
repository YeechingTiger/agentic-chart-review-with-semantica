"""Fold a corpus scan into a retrieval prior. Data in, data out — no model, no agent.

    labels.jsonl (one row per note, from `acr label scan`)  ->  RetrievalPrior

THE AGGREGATOR THAT DID NOT EXIST. The scan produced the raw material and the runtime had a complete
renderer for the finished prior (`review.document_concepts.experience_block`), and nothing folded one
into the other: `measured_yield` appeared exactly once in the tree, in the function that reads it.
The chain's terminal step was `assets adopt`, which writes a keyword list into the CONTRACT — moving
`spec_hash`, so `analyze_arms.py` then refuses to compare the arm against its own baseline. A prior
that is an ASSET instead of a contract edit is comparable, which is the whole point.

DECOUPLED FROM THE AGENT, deliberately and structurally. This module imports no part of `review`:
`tests/test_layering.py` forbids it, and the two planes meet through `contract.retrieval_prior` and a
JSON file on disk. Nothing here can call a model, read a prompt, or influence a run except by writing
an artifact somebody later chooses to pass in.

## What the arithmetic has to get right

`can_establish` only, for the answer-bearing denominator. `merely_mentions` is recorded beside it and
never added to it: ranking on "bears on the question" rewards retrieving the places that TALK about a
thing over the places that SETTLE it, which is the failure the admissibility vocabulary exists to
name. `tools/build_termcache.py` states the same rule for the same reason.

Two numbers per term, never one score. A term that matches every note has perfect recall and no
value, and `derive.price_terms` already prices candidates on marginal cost against gain. A prior
reporting recall alone would recommend precisely the terms `derive` refuses.

The denominator travels with every rate. One note of a type that established the answer out-rates
forty notes that established it thirty times, and only `n_scanned` tells them apart.

## Two bases, and the weaker one is the default

Without a corpus, a term is counted where the READING MODEL PROPOSED it. That is a lower bound: a
scan capped at eight terms per note cannot propose a ninth, so a term missing from a note's list is
not evidence the term is missing from the note. With `corpus=`, every term is asked of every scanned
note through the corpus's own matcher — the real number, and the only basis on which one term may be
said to beat another. The basis is stored per term so the two can never be silently compared.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contract.retrieval_prior import (
    DocTypeYield,
    FieldPrior,
    Measured,
    RetrievalPrior,
    RetrievalPriorError,
    TermYield,
    prior_digest,
)

#: The one verdict that makes a note answer-bearing. `merely_mentions` is counted separately and
#: never folded in — see the module docstring.
CAN_ESTABLISH = "can_establish"
MERELY_MENTIONS = "merely_mentions"

#: How many hits to allow when asking the corpus whether a term occurs in a chart. Large enough that
#: a common term in a 300-note chart is not truncated: a capped search would understate the COST
#: column, which is the half that stops a matches-everything term being recommended.
_MAX_HITS = 100_000


def _rows(labels: str | Path | Sequence[Mapping[str, Any]]) -> list[dict]:
    """Labels from a JSONL path, or already-parsed rows. A blank line is not a row."""
    if isinstance(labels, (str, Path)):
        p = Path(labels).expanduser()
        if not p.is_file():
            raise RetrievalPriorError(f"no labelling at {p}")
        out = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RetrievalPriorError(f"{p}:{i}: {e}") from e
        return out
    return [dict(r) for r in labels]


def _verdict(row: Mapping[str, Any], field: str) -> str:
    """This note's standing for one field.

    Two shapes, because `Admissibility.to_dict` writes `verdicts` and `tools/build_termcache.py`
    documents having met `fields` in the wild. Reading only one would silently score every note
    `neither` — which is a prior that says the corpus is empty.
    """
    adm = row.get("admissibility") or {}
    per_field = adm.get("verdicts")
    if not isinstance(per_field, Mapping):
        per_field = adm.get("fields") if isinstance(adm.get("fields"), Mapping) else {}
    return str((per_field or {}).get(field) or "")


def _proposed_terms(row: Mapping[str, Any]) -> set[str]:
    """The terms the reading model offered for this note, lowercased.

    Only verified terms reach a label — `RetrievalTerm` has no `verified` flag because an
    unverified proposal is not a row — so every term here was found in this note's text.
    """
    out: set[str] = set()
    for t in (row.get("retrieval_terms") or []):
        term = (t if isinstance(t, str) else (t or {}).get("term", ""))
        term = str(term).strip().lower()
        if term:
            out.add(term)
    return out


def _corpus_presence(corpus, rows: Sequence[Mapping[str, Any]],
                     terms: Sequence[str]) -> dict[str, set[tuple[str, str]]]:
    """`term -> {(patient, note)}`, asked of the corpus's own matcher.

    THE CORPUS DECIDES, not a substring test written here. `chart.search` is notation-tolerant
    (separators, quote widening, dates), so `adeno-carcinoma`, `adeno carcinoma` and
    `adeno_carcinoma` are one term to a run and would be three to a naive scan. A prior priced
    against a matcher the runtime does not use is a prior for a search nobody performs — the point
    `tools/build_termcache.py` makes, and the scar `tools/analyze_arms.py` carries.
    """
    scanned = {(str(r.get("patient_id") or ""), str(r.get("note_id") or "")) for r in rows}
    by_patient: dict[str, set[str]] = defaultdict(set)
    for pid, nid in scanned:
        by_patient[pid].add(nid)
    out: dict[str, set[tuple[str, str]]] = {t: set() for t in terms}
    for pid, note_ids in sorted(by_patient.items()):
        chart = corpus.chart(pid)
        for term in terms:
            for hit in chart.search(term, False, None, None, None, max_hits=_MAX_HITS):
                nid = str(getattr(hit, "note_id", ""))
                # RESTRICTED TO THE SCANNED NOTES. A term found in a note the scan never read has
                # no verdict, so counting it would put a note in the cost column whose standing
                # nobody established — inflating cost with unknowns.
                if nid in note_ids:
                    out[term].add((pid, nid))
    return out


def build_prior(labels: str | Path | Sequence[Mapping[str, Any]], *,
                fields: Sequence[str],
                min_patients: int,
                asset_id: str,
                version: str = "1",
                corpus=None,
                labelling_id: str = "") -> RetrievalPrior:
    """One scan, folded per variable. Refuses rather than emitting a prior nobody should trust.

    `min_patients` HAS NO DEFAULT. What counts as enough subjects to generalise from is a policy
    choice about this corpus, and a default would be the same class of mistake as the thing this
    module measures — `evals.DetectorConfig` states the identical rule for its thresholds. Two is
    the smallest number that can be split into two non-empty halves, which is what
    `assets split` requires downstream; a real prior wants far more.

    `corpus` is optional and upgrades every term's basis from `proposed_by_reader` to
    `corpus_matched`. See the module docstring for why the default is the weaker one.
    """
    if not fields:
        raise RetrievalPriorError(
            "fields is required: folding whatever verdicts happen to be present makes the prior's "
            "subject depend on the labelling rather than on the question")
    rows = _rows(labels)
    if not rows:
        raise RetrievalPriorError("the labelling is empty; there is nothing to fold")

    patients = sorted({str(r.get("patient_id") or "") for r in rows} - {""})
    if len(patients) < int(min_patients):
        raise RetrievalPriorError(
            f"this labelling covers {len(patients)} patient(s) and --min-patients is "
            f"{int(min_patients)}. A prior measured on fewer subjects than that describes those "
            f"subjects, not the corpus; every rate below would be a fact about one chart.")

    spec_ids = sorted({str(r.get("spec_id") or "") for r in rows} - {""})
    if len(spec_ids) > 1:
        raise RetrievalPriorError(
            f"these labels answer {len(spec_ids)} different requirements — {spec_ids}. A label is "
            f"conditional on the requirement it was read against, so folding two together produces "
            f"rates for a question nobody asked. Split the labelling.")

    models = sorted({str(r.get("model") or "") for r in rows} - {""})
    prompts = sorted({str(r.get("prompt_hash") or "") for r in rows} - {""})

    all_terms: set[str] = set()
    for r in rows:
        all_terms |= _proposed_terms(r)
    presence = (_corpus_presence(corpus, rows, sorted(all_terms))
                if corpus is not None and all_terms else None)
    basis = "corpus_matched" if presence is not None else "proposed_by_reader"

    field_priors: list[FieldPrior] = []
    for name in fields:
        answer_bearing: set[tuple[str, str]] = set()
        n_notes = 0
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])   # scanned, est, mention
        for r in rows:
            v = _verdict(r, name)
            if not v:
                # A row with no verdict for THIS field says nothing about it. Not counted as a
                # scanned note either: the denominator must be notes whose standing was decided.
                continue
            n_notes += 1
            key = (str(r.get("patient_id") or ""), str(r.get("note_id") or ""))
            dt = str(r.get("doc_type") or "").strip()
            if dt:
                per_type[dt][0] += 1
            if v == CAN_ESTABLISH:
                answer_bearing.add(key)
                if dt:
                    per_type[dt][1] += 1
            elif v == MERELY_MENTIONS and dt:
                per_type[dt][2] += 1

        decided = {(str(r.get("patient_id") or ""), str(r.get("note_id") or ""))
                   for r in rows if _verdict(r, name)}
        terms: list[TermYield] = []
        for term in sorted(all_terms):
            if presence is not None:
                surfaced = presence.get(term, set()) & decided
            else:
                surfaced = {(str(r.get("patient_id") or ""), str(r.get("note_id") or ""))
                            for r in rows if _verdict(r, name) and term in _proposed_terms(r)}
            hit_answer = len(surfaced & answer_bearing)
            terms.append(TermYield(term=term,
                                   n_surfaced_answer_bearing=hit_answer,
                                   n_surfaced_other=len(surfaced) - hit_answer,
                                   basis=basis))
        field_priors.append(FieldPrior(
            field_name=name,
            n_answer_bearing=len(answer_bearing),
            n_notes=n_notes,
            # A term nothing surfaced carries no information for this field and is dropped from the
            # per-field list. It is NOT dropped from the corpus-matched sweep above, because its
            # absence there is itself the measurement.
            terms=tuple(t for t in terms if t.n_surfaced > 0),
            doc_types=tuple(sorted(
                (DocTypeYield(doc_type=dt, n_scanned=c[0], n_can_establish=c[1],
                              n_merely_mentions=c[2])
                 for dt, c in per_type.items() if c[0] > 0),
                key=lambda d: (-d.rate, -d.n_scanned, d.doc_type))),
        ))

    return RetrievalPrior(
        asset_id=asset_id,
        version=version,
        # NEVER `certified`. `assetdev.certify` grants that on held-out subjects by writing a
        # certificate beside the asset; a builder that could stamp it would let an uncertified
        # prior claim it, which is the failure `assets adopt` refuses at the other end.
        status="measured",
        measured=Measured(
            n_patients=len(patients),
            n_notes=len(rows),
            patient_digests=tuple(prior_digest(p) for p in patients),
            spec_id=spec_ids[0] if spec_ids else "",
            model=models[0] if len(models) == 1 else ("mixed" if models else ""),
            prompt_hash=prompts[0] if len(prompts) == 1 else ("mixed" if prompts else ""),
            labelling_id=labelling_id or (str(labels) if isinstance(labels, (str, Path)) else ""),
        ),
        fields=tuple(field_priors),
    )
