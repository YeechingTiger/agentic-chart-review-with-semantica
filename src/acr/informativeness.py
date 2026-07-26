"""Informativeness check for derived specification artifacts.

A derived artifact can be structurally perfect and semantically empty. `note_type_filters`
in the reference platform is the worked example: every metadata field correct, provenance
recorded, timestamp fresh — and almost every value the literal string "unknown", because
the note-type catalogue was never wired up. Nothing downstream complained, because
"unknown" reads like a conclusion.

So this module counts two things and refuses to add them together:

    n_null_lists              a determination that never ran      -> fix the code
    n_undeterminable_entries  a determination that ran and found
                              no classifiable type                -> data property

Merging them is exactly the defect being guarded against: the remedies differ, so the
counts must stay apart. Legacy "unknown" is treated as its own, worst category — it is
the disguise that let the failure hide.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

LEGACY_SENTINELS = {"unknown", "n/a", "na", "none", "null", "unspecified"}
UNDETERMINABLE = "UNDETERMINABLE"

# A derived artifact is worth using only if it actually discriminates.
MIN_NON_NULL_FRACTION = 0.50
MIN_HIGH_PRIORITY_FRACTION = 0.50
MIN_DISTINCT_TYPES = 2


@dataclass
class Informativeness:
    criteria_total: int = 0
    criteria_with_high_priority_type: int = 0
    non_null_type_fraction: float = 0.0
    n_null_lists: int = 0                 # pipeline faults
    n_undeterminable_entries: int = 0     # genuine data property
    n_legacy_unknown_entries: int = 0     # schema violation AND a disguised fault
    distinct_types_observed: int = 0
    verdict: str = "OK"
    red_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def assess_note_type_filters(filters: dict[str, Any]) -> Informativeness:
    """`filters` is the mapping criterion_id -> {priority -> list | null}."""
    inf = Informativeness()
    real_types: set[str] = set()
    total_entries = 0
    real_entries = 0

    for crit, prios in (filters or {}).items():
        inf.criteria_total += 1
        has_real_high = False

        for prio, lst in (prios or {}).items():
            if lst is None:
                inf.n_null_lists += 1
                continue
            for raw in lst:
                total_entries += 1
                val = str(raw).strip()
                if val.lower() in LEGACY_SENTINELS:
                    inf.n_legacy_unknown_entries += 1
                elif val == UNDETERMINABLE:
                    inf.n_undeterminable_entries += 1
                else:
                    real_entries += 1
                    real_types.add(val)
                    if prio == "high":
                        has_real_high = True

        if has_real_high:
            inf.criteria_with_high_priority_type += 1

    inf.distinct_types_observed = len(real_types)
    inf.non_null_type_fraction = (real_entries / total_entries) if total_entries else 0.0

    r = inf.red_reasons
    if inf.n_legacy_unknown_entries:
        r.append(
            f"{inf.n_legacy_unknown_entries} legacy 'unknown' entries — illegal under v2. "
            "'unknown' cannot distinguish a failed determination from an unclassifiable "
            "document; use null or UNDETERMINABLE."
        )
    if inf.n_null_lists:
        r.append(f"{inf.n_null_lists} priority lists are null — the derivation did not run. Fix the pipeline.")
    if inf.non_null_type_fraction < MIN_NON_NULL_FRACTION:
        r.append(
            f"only {inf.non_null_type_fraction:.1%} of entries carry a real note type "
            f"(threshold {MIN_NON_NULL_FRACTION:.0%})"
        )
    if inf.criteria_total:
        frac = inf.criteria_with_high_priority_type / inf.criteria_total
        if frac < MIN_HIGH_PRIORITY_FRACTION:
            r.append(
                f"only {inf.criteria_with_high_priority_type}/{inf.criteria_total} criteria have any "
                f"real high-priority note type (threshold {MIN_HIGH_PRIORITY_FRACTION:.0%})"
            )
    if inf.distinct_types_observed < MIN_DISTINCT_TYPES:
        r.append(
            f"only {inf.distinct_types_observed} distinct real note type(s) observed — "
            "the artifact cannot discriminate between criteria"
        )

    inf.verdict = "RED" if r else "OK"
    return inf


def load_frontmatter(path: str | Path) -> dict:
    """Read YAML frontmatter out of a `references/*.md` artifact."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        raise ValueError(f"no YAML frontmatter in {path}")
    return yaml.safe_load(m.group(1)) or {}


def assess_file(path: str | Path) -> Informativeness:
    fm = load_frontmatter(path)
    return assess_note_type_filters(fm.get("filters") or {})


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Check whether a derived artifact is actually informative.")
    ap.add_argument("path", help="path to a note_type_filters.md artifact")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    inf = assess_file(a.path)
    if a.json:
        print(json.dumps(inf.to_dict(), indent=2))
    else:
        print(f"verdict: {inf.verdict}")
        print(f"  criteria                        {inf.criteria_total}")
        print(f"  with a real high-priority type  {inf.criteria_with_high_priority_type}")
        print(f"  non-null type fraction          {inf.non_null_type_fraction:.1%}")
        print(f"  distinct real types             {inf.distinct_types_observed}")
        print(f"  null lists (pipeline fault)     {inf.n_null_lists}")
        print(f"  UNDETERMINABLE (data property)  {inf.n_undeterminable_entries}")
        print(f"  legacy 'unknown' (illegal)      {inf.n_legacy_unknown_entries}")
        for x in inf.red_reasons:
            print(f"  ! {x}")
    return 1 if inf.verdict == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
