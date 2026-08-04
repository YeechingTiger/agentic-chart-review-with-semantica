"""The inputs the three DECISION-POINT scripts need, taken from the command line.

`docs/NEW_TASK_NEW_DATA.md` calls two of these mandatory before investing anything:

    step 12  `measure_agency.py`           — is an agent the right tool for this variable at all?
    step 13  `measure_controller_value.py` — are the failures retrieval failures or misreadings?
    step 22  `analyze_arms.py`             — did the intervention earn its place?

All three were hardcoded to one contract and one corpus. `SPEC = specs_root()/"STORE.390…"` with no
flag; gold read from `corpus_root().parent/"index.json"`, a file only `tools/generate_corpus.py`
writes; the answer plucked from a literal `date_of_initial_diagnosis` key; `analyze_arms` globbing
`SYN*.manifest.json` and resolving its run root under the repo. So on any corpus but the shipped
synthetic one:

    $ ACR_CORPUS=/tmp/fakecorp/patients python tools/measure_agency.py /tmp/emptyruns
    FileNotFoundError: /tmp/fakecorp/index.json

A decision point that cannot run on the data you are deciding about is not a decision point. The doc
said to run them before spending money on a new task, and on a new task they could not run.

## Gold comes from the ANSWER KEY, not from a corpus sidecar

`--answer-key` takes the same file `acr eval score --answer-key` takes. The point is that these
scripts now consume the artifact the EVALUATION plane already consumes, rather than a corpus sidecar
that exists only here. On a real corpus that file must be authored: `acr gold` stages registry values
as LOCAL UNRESOLVED references and does not emit this format. `None` in the key means ABSTENTION IS CORRECT, and it stays `None`
here — collapsing it to a status string is how a correct abstention becomes a failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acr.contract.spec import load_spec  # noqa: E402
from acr.core import site  # noqa: E402
from acr.core.local_artifacts import RUN_RECORD_GLOB  # noqa: E402


def add_arguments(ap: argparse.ArgumentParser, *, runs_help: str = "run record or directory") -> None:
    """The four flags every decision point needs. Named identically across all three."""
    ap.add_argument("runs", nargs="?", default="runs/a15eval", help=runs_help)
    ap.add_argument("--spec", default=None,
                    help="contract YAML; default: the only spec the runs agree on")
    ap.add_argument("--answer-key", default=None,
                    help="the file `acr eval score --answer-key` takes. Without it, gold-dependent "
                         "columns report NOT MEASURED rather than guessing.")
    ap.add_argument("--fields", default="",
                    help="comma list; default: every field the spec declares")
    ap.add_argument("--corpus", default=None, help="default: acr.core.site.corpus_root()")


@dataclass(frozen=True)
class _Run:
    """The two attributes `evals._key_row` reads."""

    patient_id: str
    spec_id: str


class Inputs:
    """Resolved inputs, plus the two lookups every one of the three scripts performs."""

    def __init__(self, args: argparse.Namespace):
        self.root = pathlib.Path(args.runs)
        self.corpus_root = pathlib.Path(args.corpus) if args.corpus else site.corpus_root()
        self.manifests = sorted(glob.glob(f"{self.root}/**/{RUN_RECORD_GLOB}", recursive=True))
        self.spec_path = self._spec_path(args.spec)
        self.spec = load_spec(self.spec_path) if self.spec_path else None
        self.fields = ([f.strip() for f in args.fields.split(",") if f.strip()]
                       or self._declared_fields())
        self.key = self._load_key(args.answer_key)

    # -- resolution ---------------------------------------------------------------------------
    def _spec_path(self, given: str | None) -> pathlib.Path | None:
        if given:
            return pathlib.Path(given)
        # INFERRED FROM THE RUNS, not defaulted to a name. A default spec is how these scripts
        # came to be about one contract: the runs already record which one produced them, and a
        # cohort spanning two contracts is a comparison that must not be made silently.
        ids = {json.loads(pathlib.Path(m).read_text(encoding="utf-8")).get("spec_id")
               for m in self.manifests}
        ids.discard(None)
        if len(ids) != 1:
            return None
        want = ids.pop()
        for p in sorted(site.specs_root().glob("*.yaml")):
            try:
                if load_spec(p).spec_id == want:
                    return p
            except Exception:                     # noqa: BLE001 — a bad spec is not this file's to report
                continue
        return None

    def _declared_fields(self) -> list[str]:
        if self.spec is None:
            return []
        return [str(getattr(f, "name", f)) for f in (getattr(self.spec, "fields", []) or [])]

    def _load_key(self, given: str | None) -> dict:
        if not given:
            return {}
        raw = json.loads(pathlib.Path(given).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit(f"{given}: expected an answer key object, got {type(raw).__name__}")
        return raw

    # -- the two lookups ----------------------------------------------------------------------
    def want(self, patient: str, field: str):
        """The key's value for one field. `None` means abstaining is correct; `KeyError` never."""
        # `evals._key_row` is the one implementation of "does this key speak about this run" —
        # including the `spec_ids` plural that lets an ABLATION arm score against the base key.
        # Reimplementing it here is how the two scorers drifted the first time.
        from acr.evaluation.evals import _key_row
        spec_id = self.spec.spec_id if self.spec else ""
        row = _key_row(self.key, f"{patient}__{spec_id}",
                       _Run(patient_id=patient, spec_id=spec_id))
        if row is None:
            return _UNKEYED
        fields = row.get("fields")
        if not isinstance(fields, dict) or field not in fields:
            return _UNKEYED
        return fields[field]

    def coded(self, manifest: dict, field: str):
        """What a run coded for one field. `None` for an abstention, in the key's own convention."""
        answer = manifest.get("answer") or {}
        kind = str(answer.get("status_kind") or "")
        abstained = (kind != "value") if kind else str(
            answer.get("status") or "") in _ABSTAIN
        if abstained:
            return None
        v = (answer.get("value") or {}).get(field)
        return None if v in (None, "") else str(v)

    def refuse_unless_resolved(self, *, needs_key: bool) -> None:
        """Stop with a message naming the missing flag, rather than a traceback three frames in."""
        if not self.manifests:
            raise SystemExit(f"no {RUN_RECORD_GLOB} under {self.root}")
        if self.spec is None:
            raise SystemExit(
                "could not determine the contract: the runs name more than one spec_id (or a "
                "spec_id no file in assets/specs/ declares). Pass --spec.")
        if not self.fields:
            raise SystemExit(f"{self.spec.spec_id} declares no fields; pass --fields.")
        if needs_key and not self.key:
            raise SystemExit(
                "this report is about right and wrong answers, so it needs --answer-key — the "
                "same file `acr eval score --answer-key` takes. On the synthetic corpus, build one "
                "with `tools/answer_key_from_corpus.py`. On a real corpus you must author it: "
                "`acr gold` stages registry values as LOCAL UNRESOLVED references and audits "
                "derivability — it does not emit this format, and naming it here as the producer "
                "would be a consumer with no producer, which is the defect class this file is "
                "part of fixing.")


#: Returned by `want` when the key says nothing about this run/field. Distinct from `None`, which
#: is the key ASSERTING that abstention is correct — the one distinction a status string collapses.
class _Unkeyed:
    def __repr__(self) -> str:
        return "UNKEYED"

    def __bool__(self) -> bool:
        return False


_UNKEYED = _Unkeyed()
UNKEYED = _UNKEYED

_ABSTAIN = {"EVIDENCE_INSUFFICIENT", "NO_ANSWER", "SPEC_INSUFFICIENT", "ABSTAIN",
            "CORPUS_INSUFFICIENT"}


def load_trace(manifest_path: str) -> list[dict]:
    """The trace beside a manifest. `RunRecord.from_manifest`'s rule: the SIBLING, move-safely."""
    p = pathlib.Path(manifest_path)
    tp = p.with_name(p.name.replace(".manifest.json", ".jsonl"))
    if not tp.is_file():
        return []
    return [json.loads(x) for x in tp.read_text(encoding="utf-8").splitlines() if x.strip()]
