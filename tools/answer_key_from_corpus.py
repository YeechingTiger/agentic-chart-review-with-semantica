#!/usr/bin/env python3
"""Turn the synthetic corpus's own `_ground_truth.json` files into an `acr eval score` key.

WHY THIS EXISTS
---------------
`acr eval score` is the deterministic scorer, and it takes the key as
`{instance_id: {"fields": {...}, "subgroups": [...]}}`. The synthetic corpus ships its truth in
a different shape, one file per patient, keyed by spec id. Without a converter, every person
scoring a run against this corpus writes their own comparison loop — and a second scorer is a
second answer to a question that already has one. The first draft of the search pilot did
exactly that and got a case wrong that `evals.score` gets right (below).

ABSTENTION IS A KEY VALUE, NOT A MISSING ONE
--------------------------------------------
`evals.score` reads `None` as "abstaining is the CORRECT answer here", which is the whole reason
`abstention_correctness` is a separate row from `task_completion`. So a patient whose
`_ground_truth` status is not FOUND must arrive as an explicit `None`, never as an absent key.

That distinction is where a hand-rolled scorer goes wrong. SYN0002's key carries
`status: EVIDENCE_INSUFFICIENT` AND `primary_site: C187` — the site is what a registrar with
outside records knows, not what this chart can establish. Read naively, that row scores an
abstaining run as having missed C187, which is precisely backwards: the run was right. Here the
status decides, and the recorded value is dropped when the status says the chart cannot
establish it.

SUBGROUPS ARE WHERE THE INTERESTING FAILURE HIDES
-------------------------------------------------
`eval compare` fails a comparison when any SUBGROUP rate falls even if every headline rate rose,
because the aggregate improvement is what gets shipped and the subgroup collapse is what reaches
a patient. Four subgroups are emitted per patient: the primary site's ICD-O chapter, whether the
chart's own key says the value is establishable, the designer's own words for what the chart
exercises, and — since 2026-08-03 — `held_out`. An arm that improves overall by giving up on
abstention cases shows up as a subgroup regression rather than as a win, and an arm that gains
only on the charts its own cards were written from shows up the same way.

USAGE
-----
    PYTHONPATH=src .venv/bin/python tools/answer_key_from_corpus.py \
        --spec-key STORE.390.date_of_initial_diagnosis \
        --fields date_of_initial_diagnosis \
        --out /tmp/key-390.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "corpus" / "patients"


def _subgroups(truth: dict, row: dict) -> list[str]:
    """Strata a rate must not be allowed to fall inside while the headline rises."""
    out = []
    site = str((row or {}).get("primary_site") or "")
    if site[:1] == "C" and site[1:3].isdigit():
        out.append(f"icdo_chapter:C{site[1:3]}")
    out.append("establishable:" + ("yes" if row.get("status") == "FOUND" else "no"))
    # THE SUBGROUP A CLAIM RESTS ON. SYNX/SYNK were designed by watching runs fail and the
    # search cards were written from the same failures, so a card's score on them is a score on
    # its own development set. `eval compare` fails a comparison when any subgroup rate falls
    # even while the headline rises — which is exactly the shape of an arm that gains on the
    # charts it was built from and loses on the ones it was not. Defaults to `informed` for a
    # chart written before the flag existed, matching the generator.
    out.append("held_out:" + ("no" if truth.get("informed_module_design", True) else "yes"))
    pattern = str(truth.get("evidence_pattern") or "").strip()
    if pattern:
        # The corpus designer's own words for what each chart exercises. Kept verbatim rather
        # than bucketed: a bucketing invented here would be a claim about which patterns are
        # alike, and nobody has measured that.
        out.append(f"pattern:{pattern[:60]}")
    return out


#: Keys that are metadata about the row rather than a field value.
_META = ("status", "why", "observable_period", "type", "date")


def _value_of(row: dict, field: str, fields: list[str], pid: str) -> object:
    """This row's value for one field, or raise.

    The corpus writes single-field specs as `{"value": ..., "status": ...}` and multi-field
    specs as `{"primary_site": ..., "histology": ..., "status": ...}`. Accepting both is
    necessary; GUESSING between them is not, which is why a field that resolves to nothing on
    an establishable row raises instead of becoming `None`.

    That distinction is the whole guard. `None` is a live assertion here — it means abstention
    is the correct answer — so a misspelled `--fields` would otherwise produce a key claiming
    every patient should abstain, and score a perfectly correct run as wrong on all of them.
    The first run of this script did exactly that, and only the "0 with a value" line said so.
    """
    if field in row:
        return row[field]
    if len(fields) == 1 and "value" in row:
        return row["value"]
    have = sorted(k for k in row if k not in _META)
    raise SystemExit(
        f"{pid}: the key for this spec has status FOUND but carries no value for "
        f"{field!r}. It has {have or '(nothing but metadata)'}. Either --fields is misspelled "
        f"or this corpus row is malformed — refusing to emit a null, which would assert that "
        f"abstaining is the correct answer here.")


def build(corpus: Path, spec_key: str, fields: list[str]) -> dict:
    key: dict[str, dict] = {}
    for gt_path in sorted(corpus.glob("*/_ground_truth.json")):
        truth = json.loads(gt_path.read_text(encoding="utf-8"))
        pid = truth.get("patient_id") or gt_path.parent.name
        row = (truth.get("ground_truth") or {}).get(spec_key)
        if row is None:
            continue
        establishable = row.get("status") == "FOUND"
        key[pid] = {
            "fields": {
                # None means "abstention is correct". Never omit the field: an absent key and a
                # null key are different assertions, and only one of them can be scored.
                f: (_value_of(row, f, fields, pid) if establishable else None)
                for f in fields
            },
            "subgroups": _subgroups(truth, row),
        }
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--spec-key", required=True,
                    help="the key inside `ground_truth`, e.g. STORE.390.date_of_initial_diagnosis")
    ap.add_argument("--fields", required=True,
                    help="comma list, and it must match `acr eval score --fields` exactly")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    key = build(Path(a.corpus), a.spec_key, fields)
    if not key:
        raise SystemExit(f"no patient carries {a.spec_key!r}; check the spec key spelling")

    Path(a.out).write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
    n_abstain = sum(1 for v in key.values() if all(x is None for x in v["fields"].values()))
    print(f"{len(key)} patients -> {a.out}")
    print(f"  {len(key) - n_abstain} with a value, {n_abstain} where abstention is correct")


if __name__ == "__main__":
    main()
