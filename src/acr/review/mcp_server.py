"""MCP server: the chart-review capability as one tool surface.

Design doc section 3. One server, so the same capability works in Claude Code, claude.ai and
any SDK agent, instead of being locked inside this repo's bespoke runtime.

Nothing here reimplements chart access, coverage accounting or the gate. `chart.*` calls run
through the existing `Toolbox`, so coverage is recorded as a side effect of real tool use
rather than being self-reported; `gate.check` calls `graph.gate_answer`, the same function
`ChartReviewAgent` uses. A second gate that could disagree with the first is the `state.py`
two-ledger failure one layer up — two verdicts, nothing raising when they diverge.

THREE INVARIANTS, ENFORCED HERE AND NOWHERE ELSE
------------------------------------------------
1. **The sampling seed is server-held.** `coverage.pending_samples` takes a `run_id` and
   nothing else. A caller cannot pass a seed, a sample size, or a list of documents. The
   seed is derived by HMAC from a server secret over `(patient, spec_id)` — not over the
   run_id — so re-minting a run for the same question re-derives the *same* seed. Deriving
   from a caller-visible run_id would let a caller mint runs until the draw looked
   convenient, which is the circularity the forced sampler exists to prevent, restored via
   the front door.

2. **`gate.check` is the only path to `validated`.** No tool accepts a validation claim from
   the caller. `gate.check` strips `gate_validated`, `coverage_attested`, `proof_basis` and
   the rest from the submitted answer before evaluating it, and reports what it stripped —
   silently discarding them would let a caller believe self-attestation worked.

3. **`registry.truth` is quarantined.** It needs a credential the extraction path does not
   have (`ACR_REGISTRY_TRUTH_TOKEN`, presented by the caller, compared in constant time) and
   it refuses any patient this session has already served a `chart.*` or `coverage.*` call
   for. The refusal is SYMMETRIC: once truth has been served for a patient, extraction calls
   for that patient are refused too. A one-directional check is defeated by reordering — ask
   for the answer first, then review the chart — and the harm is identical, because the harm
   is ground truth being in the context while the chart is read.

   The quarantine is keyed on a CANONICAL patient identity resolved once at the boundary
   (`_canonical_patient`), never on the string the caller typed. It used to be keyed on the
   raw string while the filesystem was reached by joining that same string onto the corpus
   root — two resolvers for one question, and `./SYN0002`, `SYN0002/` and
   `../patients/SYN0002` were three patients to the ledger and one directory to `open()`.
   Truth came back and the chart stayed open. See `_canonical_patient` for the full class.

   It is priced on the CREDENTIAL CHECK, not on the payload. A call that verifies the token,
   resolves the patient and then returns UNKNOWN_VARIABLE has still answered a registry
   question about that patient — it reports the answer key's variable set in `available`, and
   NO_GROUND_TRUTH vs UNKNOWN_VARIABLE says whether the patient is in the eval cohort at all.
   Charging only calls that returned a value left that probe free and unlimited. See
   `_CHARGE_QUARANTINE`.

WHAT INVARIANT 3 DOES NOT DO
----------------------------
Quarantine-on-call is damage limitation, not prevention. Four things it does not give you:

* `registry.truth` RETURNS the answer and quarantines afterwards, so by the time the ledger
  is written the truth is already in the caller's context; later refusals only stop the SAME
  session reading more chart.
* The quarantine is PER PATIENT, and a truth payload is not only a label — the answer key
  carries a free-text `why` ("histology must NOT be inferred from imaging") that is corpus
  design guidance and generalises. Reading truth for one patient leaves the session extracting
  the other eleven with that rule in context. Pinned by
  `test_truth_for_one_patient_rides_into_every_other_chart_in_the_session`.
* The ledger is process memory and is persisted nowhere, so a reconnect, a second client or a
  restart starts clean — and no OTHER front end consults it at all. `extraction_touched` and
  `truth_served` appear in this module and nowhere else in `src/`, so the CLI/LangGraph path
  reviews a chart with no knowledge that truth was served for it over MCP.
* The credential is a token read by this same object from this same environment: one process
  holds the extraction surface and the eval credential at once.

A boundary would be a second server under a second identity with no corpus text on its
surface, with the ledger outside the process. This is not that. Pinned, so the green section
cannot be read as a stronger claim than it is, by
`test_the_quarantine_is_damage_limitation_and_the_test_suite_says_so`.

Handlers are plain methods returning plain dicts, reachable through `ChartReviewService.call`,
which is the single entry point the MCP adapter also uses. Tests drive `call` directly; a live
MCP client is never required to exercise the invariants.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..chartstore.corpus import Corpus, PatientChart

# THE SAME GATE — the identical function object `graph` re-exports — taken from the module
# that owns it, so langgraph never enters this server's closure for a judgement about answers.
from ..contract.answer_contract import (
    SPEC_SECTIONS,
    assert_answer_is_reportable,
    build_spec_gap,
    strip_value_from_spec_insufficient,
)
from ..contract.spec import ExtractionSpec, load_specs
from ..core.state import EvidenceLedger
from .answer_gate import check_gate, gate_answer, keyword_hits_among_drawn
from .coverage import (
    CoverageLedger,
    ForcedSampler,
    assign_strata,
    derive_sample_seed,
    strata_from_spec,
)
from .tools import Toolbox

GROUND_TRUTH_FILE = "_ground_truth.json"

# A stratum policy maps onto exactly one plan bucket. An unrecognised policy becomes
# `search`, never `sample`: unjudged means unjudged, and the safe default for unjudged is to
# look. `coverage_planner.plan_coverage` makes the same choice for the same reason.
POLICY_BUCKET = {
    "exhaustive": "read_all",
    "exhaustive_until_witness": "read_all",
    "search_then_read_hits_and_sample_misses": "search",
    "validate_by_sampling": "sample",
}

# Arguments that would let the caller steer a draw it is not allowed to steer. Rejected
# loudly rather than dropped: a caller who thinks it set the seed and was quietly ignored
# will report a reproducible sample that nobody can reproduce.
STEERING_ARGS = frozenset({
    "seed", "sample_seed", "random_state", "rng", "n", "size", "limit",
    "note_ids", "documents", "stratum", "min_sample", "exclude",
})

# Fields a caller may not assert about its own answer. Every one of these is something the
# server decides. `witness_count` is included because a positive that inflates it is claiming
# corroboration it did not record.
SELF_ATTESTATION_KEYS = (
    "gate_validated", "validated", "accepted", "coverage_attested", "proof_obligation",
    "proof_basis", "negative_basis", "witness_count", "route_to_human", "remedy_class",
    # The spec-gap block is ASSEMBLED by the server from validated inputs (spec_section,
    # spec_quote, uncovered_fields). A caller-supplied one would be a report that skipped the
    # quote check — the exact thing refine.py's citation mask relies on having been done.
    "spec_gap", "value_withheld",
)


def _err(code: str, message: str, **extra: Any) -> dict:
    return {"error": code, "message": message, **extra}


# A handler's request that `call` charge this patient to the quarantine even though the
# handler is returning an error. Popped at the boundary, so it never reaches a client.
#
# It exists because "was anything served?" is the wrong question to price the quarantine on.
# `registry.truth` verifies the credential, resolves the patient and parses the answer key
# BEFORE it discovers the variable is missing; by then it has already learned — and reported
# in `available` — which variables this patient is scored on, and whether it is in the answer
# key at all. That is registry knowledge about this patient, obtained with the eval
# credential, and it must cost the same as a value. The line is the CREDENTIAL CHECK: below
# it nothing is disclosed and nothing is charged, so a wrong token still cannot be used to
# lock a patient out of review.
_CHARGE_QUARANTINE = "_charge_quarantine"


# Characters that make a string a PATH rather than an id. A patient id names a patient; it is
# never a route to one. Rejected structurally, before any lookup, so the error says what is
# wrong instead of "no such patient" — and so no code below is ever tempted to interpret one.
#
# `%` is refused rather than decoded. Decoding means choosing how many times to decode, and
# every answer is wrong for some caller: decode once and `%252e%252e` survives, decode until
# stable and a legitimate id containing a literal `%` is mangled into a traversal. No patient
# id in this corpus contains one, so the question does not have to be answered.
_ID_FORBIDDEN = frozenset("/\\%")


def _normalise_spelling(raw: str) -> str:
    """NFKC, because the filesystem sees the composed form.

    `ＳＹＮ０００２` (fullwidth) and `SYN0002` are different strings and — on any normalising
    layer between here and `open()` — one directory. Folding here means the ledger and the
    filesystem agree about what was named. NFKC also maps `＼` and `／` onto the ASCII
    separators, so a fullwidth traversal is caught by the structural check below rather than
    slipping past it as an exotic-but-harmless character.
    """
    return unicodedata.normalize("NFKC", raw)


@dataclass
class RunSession:
    """One (patient, spec) review. Minted by `coverage.plan`, never by the caller."""

    run_id: str
    patient_id: str
    spec: ExtractionSpec
    chart: PatientChart
    evidence: EvidenceLedger
    coverage: CoverageLedger
    toolbox: Toolbox
    plan: dict
    sample_seed: int
    seed_provenance: str
    # Set by `gate.check` and by nothing else. See invariant 2.
    validated: bool = False
    gate_checks: int = 0
    accepted_answer: dict | None = None

    def descriptor(self, minted: bool) -> dict:
        return {
            "run_id": self.run_id, "patient": self.patient_id,
            "spec_id": self.spec.spec_id, "spec_hash": self.spec.spec_hash,
            "plan": self.plan, "frozen": True, "minted": minted,
            "sample_seed": self.sample_seed, "seed_provenance": self.seed_provenance,
            "seed_held_by": "server",
            "validated": self.validated,
        }


class ChartReviewService:
    """Session state and handlers. The MCP layer is a shim over `call`."""

    def __init__(self, corpus_root: str | Path, spec_dir: str | Path = "specs", *,
                 seed_secret: bytes | None = None, truth_token: str | None = None,
                 corpus_vocabulary: bool = True):
        self.corpus = Corpus(Path(corpus_root))
        self.specs: dict[str, ExtractionSpec] = load_specs(spec_dir)
        self._charts: dict[str, PatientChart] = {}
        self._scratch: dict[str, Toolbox] = {}
        self._runs: dict[tuple[str, str], RunSession] = {}
        self._runs_by_id: dict[str, RunSession] = {}
        self._use_vocabulary = corpus_vocabulary
        self._vocabulary: list[str] | None = None

        # A secret supplied through the environment survives a restart, so an auditor can
        # replay a draw months later. A generated one cannot, and the response says so
        # instead of implying a reproducibility the process cannot deliver — the same
        # honesty `note_type_source: NOT_WIRED` exists for.
        env_secret = os.environ.get("ACR_SAMPLE_SEED_SECRET")
        if seed_secret is not None:
            self._seed_secret, self.seed_provenance = seed_secret, "supplied_secret"
        elif env_secret:
            self._seed_secret = env_secret.encode()
            self.seed_provenance = "env:ACR_SAMPLE_SEED_SECRET"
        else:
            self._seed_secret = secrets.token_bytes(32)
            self.seed_provenance = "ephemeral_process_secret_NOT_REPRODUCIBLE_ACROSS_RESTARTS"

        self._truth_token = (truth_token if truth_token is not None
                             else os.environ.get("ACR_REGISTRY_TRUTH_TOKEN"))

        # The quarantine ledger. Membership is what invariant 3 is decided on, and every
        # member is a CANONICAL id from `_patient_index` — never a caller-supplied string.
        self.extraction_touched: set[str] = set()
        self.truth_served: set[str] = set()
        self.access_log: list[dict] = []

        # Built once, lazily, by `_build_patient_index`.
        self._patient_index: dict[str, str] | None = None
        self._ambiguous_keys: set[str] = set()

    # -- patient identity ---------------------------------------------------------
    def _build_patient_index(self) -> dict[str, str]:
        """Fold every patient directory into ONE canonical identity each.

        Identity is `(st_dev, st_ino)` — what the name OPENS, not the name. Two directory
        entries for one inode (a symlink, a bind mount, a relinked shard) would otherwise be
        two quarantine keys over one chart, which is the `./SYN0002` defeat with the
        filesystem doing the aliasing instead of the string. A real directory beats a symlink
        to it and ties break lexicographically, so the canonical id is the same on every
        restart and every machine — an id that moved would silently split the ledger.

        Built once per session. Refreshing on a miss would let a caller pay for a full
        directory scan by asking for ids that do not exist; the cohort a session reviews is
        fixed anyway.
        """
        if self._patient_index is not None:
            return self._patient_index

        index: dict[str, str] = {}
        by_inode: dict[tuple[int, int], str] = {}
        try:
            entries = sorted(os.scandir(self.corpus.root),
                             key=lambda e: (e.is_symlink(), e.name))
        except OSError:
            entries = []
        for ent in entries:
            try:
                if not ent.is_dir():
                    continue
                st = ent.stat()
            except OSError:
                continue  # an unreadable entry is not a patient; it is also not fatal
            canon = by_inode.setdefault((st.st_dev, st.st_ino), ent.name)
            key = _normalise_spelling(ent.name).casefold()
            if key in index and index[key] != canon:
                # Two DIFFERENT charts whose names differ only by case or unicode form. Folding
                # them together would charge one patient's truth to another's ledger entry;
                # picking one would pick silently. Both spellings are refused instead.
                self._ambiguous_keys.add(key)
            index[key] = canon
        self._patient_index = index
        return index

    def _canonical_patient(self, raw: Any) -> tuple[str | None, dict | None]:
        """Resolve whatever the caller typed to a known patient, ONCE, at the boundary.

        Everything downstream — the quarantine ledgers, the chart cache, the run keys, the
        path `registry.truth` reads — uses the return value and never the argument. The
        argument used to be joined onto the corpus root unguarded while the quarantine was
        keyed on the raw string, so one directory had unlimited names:

            './SYN0002'  'SYN0002/'  '../patients/SYN0002'  'SYN0003/../SYN0002'
            '/abs/path/to/SYN0002'   'SYN0002//'   'SYN0002/.'   'syn0002'
            'ＳＹＮ０００２'   'SYN0002 '   'SYN0002%2F'   'SYN0002\\x00'

        Each returned ground truth and left extraction open. Rather than spell-check, the
        rule is inverted: a patient id is a MEMBER OF A SET, and membership is decided by the
        corpus, not by string surgery. Anything that is not a member is refused — including a
        spelling that would have opened a real directory, because a directory that is
        reachable is not thereby a patient.

        `Path('/corpus') / '/etc'` is `Path('/etc')`: an absolute argument discarded the root
        entirely, which made the patient field a read primitive pointed anywhere the process
        could reach. That is closed by membership too, but the structural check refuses it
        first so the failure names itself.
        """
        if not isinstance(raw, str) or raw == "":
            return None, _err("PATIENT_REQUIRED",
                              "patient must be a non-empty string naming one patient",
                              got=type(raw).__name__)

        s = _normalise_spelling(raw)
        bad: list[str] = []
        if _ID_FORBIDDEN & set(s):
            bad.append("path separator or percent-escape")
        if any(unicodedata.category(c)[0] == "C" for c in s):
            # Control and format characters: NUL truncates at the C boundary, zero-width
            # spaces and bidi marks make two ledger keys that a reviewer reads as one id.
            bad.append("control or format character")
        if s != s.strip() or s.strip(".") != s:
            # Trailing dots and spaces are stripped by some filesystems on the way to the
            # inode and by none of them on the way to a `set`.
            bad.append("leading/trailing whitespace or dot")
        if bad:
            return None, _err(
                "MALFORMED_PATIENT_ID",
                "a patient id names a patient; it is never a path, and it is not normalised "
                "on your behalf. Refused: " + ", ".join(bad),
                given=raw)

        key = s.casefold()
        # Index first: `_ambiguous_keys` is filled while building it, so checking the set
        # before the build would let the very first lookup — the only one that matters, since
        # it is the one that opens the chart — walk past an ambiguity that is not recorded yet.
        index = self._build_patient_index()
        if key in self._ambiguous_keys:
            return None, _err("AMBIGUOUS_PATIENT_ID",
                              "more than one patient directory has this name up to case and "
                              "unicode form; the quarantine cannot tell them apart",
                              given=raw)
        canon = index.get(key)
        if canon is None:
            # The known ids are deliberately NOT listed: on a real corpus they are patient
            # identifiers, and an error message is the cheapest place to enumerate a cohort.
            return None, _err("UNKNOWN_PATIENT", "no such patient in this corpus",
                              given=raw, n_known=len(index))
        return canon, None

    # -- dispatch -----------------------------------------------------------------
    HANDLERS: ClassVar[dict[str, str]] = {
        "chart.type_summary": "_h_type_summary",
        "chart.list_documents": "_h_list_documents",
        "chart.search": "_h_search",
        "chart.read": "_h_read",
        "chart.timeline": "_h_timeline",
        "coverage.plan": "_h_plan",
        "coverage.pending_samples": "_h_pending_samples",
        "gate.check": "_h_gate_check",
        "registry.truth": "_h_registry_truth",
    }

    def call(self, name: str, args: dict | None = None) -> dict:
        """The single entry point. Never raises — a tool error is an observation."""
        args = dict(args or {})
        handler = self.HANDLERS.get(name)
        if handler is None:
            return _err("UNKNOWN_TOOL", f"no such tool {name!r}",
                        available=sorted(self.HANDLERS))

        spelling = args.get("patient")
        patient, bad = self._patient_for(name, args)
        if bad:
            return bad
        if patient is not None and name not in self.RUN_SCOPED:
            # The handler must never see the spelling. Substituting here — not inside each
            # handler — is what makes "resolved once, at the boundary" true rather than
            # aspirational: a handler added later cannot forget to canonicalise, because
            # there is nothing left for it to canonicalise. Run-scoped tools are skipped:
            # they take no patient argument, and their patient came from the run, which was
            # canonical when it was minted.
            args["patient"] = patient
        family = name.split(".", 1)[0]
        blocked = self._quarantine(family, patient, name)
        if blocked:
            return blocked

        try:
            out = getattr(self, handler)(**args)
        except TypeError as e:
            return _err("BAD_ARGUMENTS", f"bad arguments for {name}: {e}")
        except Exception as e:  # noqa: BLE001 - surface it, do not kill the session
            return _err(type(e).__name__, str(e))

        # A call that served something marks the patient — and so does one that got far
        # enough to answer a registry question about them, which is what `_CHARGE_QUARANTINE`
        # signals. An UNKNOWN_DOC_TYPE still must not count as having read the chart, and a
        # registry call refused at the credential check still must not lock the patient out.
        charge = bool(out.pop(_CHARGE_QUARANTINE, False))
        if patient and (charge or not out.get("error")):
            (self.truth_served if family == "registry" else self.extraction_touched).add(patient)
            entry = {"tool": name, "patient": patient, "family": family}
            if isinstance(spelling, str) and spelling != patient:
                # The ledger keys on the canonical id, but an auditor still wants to see that
                # a caller reached this chart under another name. Dropping the spelling
                # entirely would make a probe look like ordinary traffic.
                entry["requested_as"] = spelling
            self.access_log.append(entry)
        return out

    # Tools whose patient is named by a run_id rather than a patient argument.
    RUN_SCOPED: ClassVar[frozenset[str]] = frozenset({"coverage.pending_samples", "gate.check"})

    def _patient_for(self, name: str, args: dict) -> tuple[str | None, dict | None]:
        """Which patient does this call touch? Invariant 3 is decided on the answer.

        The answer is always a canonical id or an error. There is no third case: a call whose
        patient cannot be resolved is refused before any handler runs, because a handler that
        receives an unresolved spelling is a handler that can open a chart the quarantine did
        not charge to anyone.
        """
        if name in self.RUN_SCOPED:
            run = self._runs_by_id.get(str(args.get("run_id", "")))
            if run is None:
                return None, _err("UNKNOWN_RUN_ID",
                                  "no such run in this session; mint one with coverage.plan",
                                  known_runs=sorted(self._runs_by_id))
            return run.patient_id, None
        if name == "chart.read":
            given = args.get("patient")
            if given is not None:
                canon, bad = self._canonical_patient(given)
                if bad:
                    return None, bad
                return canon, None
            return self._resolve_note_owner(str(args.get("note_id", "")), None)
        if args.get("patient") is None:
            return None, _err("PATIENT_REQUIRED", f"{name} requires a patient id")
        return self._canonical_patient(args["patient"])

    def _resolve_note_owner(self, note_id: str, patient: Any = None) -> tuple[str | None, dict | None]:
        """`chart.read(note_id)` carries no patient, but note_id is only unique WITHIN one.

        A note_id is a filename stem. Measured on this repo's 12-patient synthetic corpus:
        259 of 3,447 stems occur under more than one patient (every recurring
        `Prescriptions-Filled-RxHub_<date>`). Guessing an owner would read one patient's
        document while the quarantine ledger charged it to another — invariant 3 defeated by
        a name collision. So resolution is restricted to charts this session has already
        opened, and an ambiguity is refused rather than broken by a tiebreak.

        The note_id itself is never joined onto a directory: `PatientChart` looks it up in an
        index built from the directory listing, so `../SYN0001/x` is a missing key and not a
        path. Same rule as `_canonical_patient`, one level down.
        """
        if patient:
            return self._canonical_patient(patient)
        owners = [pid for pid, ch in self._charts.items() if note_id in ch._docs]
        if len(owners) == 1:
            return owners[0], None
        if not owners:
            return None, _err(
                "NOTE_OWNER_UNKNOWN",
                "note_id is unique only within a patient, and no open chart in this session "
                "holds it. Call chart.list_documents or chart.search first, or pass patient=.",
                note_id=note_id, open_charts=sorted(self._charts))
        return None, _err("AMBIGUOUS_NOTE_ID",
                          "this note_id exists under more than one open patient; pass patient=",
                          note_id=note_id, candidates=sorted(owners))

    def _quarantine(self, family: str, patient: str | None, name: str) -> dict | None:
        if patient is None:
            return None
        if family == "registry":
            if patient in self.extraction_touched:
                return _err(
                    "REGISTRY_TRUTH_WOULD_LEAK",
                    "this session has already served extraction calls for this patient; "
                    "ground truth reaching an extraction run voids every number downstream. "
                    "Read truth in a separate session.",
                    patient=patient,
                    served=[a["tool"] for a in self.access_log if a["patient"] == patient])
            return None
        if patient in self.truth_served:
            return _err(
                "EXTRACTION_AFTER_TRUTH_REFUSED",
                "registry ground truth for this patient was already served in this session, "
                "so any extraction from here is contaminated. Checking only one direction "
                "would be defeated by asking for the answer first.",
                patient=patient, tool=name)
        return None

    # -- shared plumbing ----------------------------------------------------------
    def _chart(self, patient: str) -> PatientChart:
        """One PatientChart per patient per session — `_text` is lru_cached on the instance,
        so sharing it is what keeps the fan-out below from re-reading files."""
        if patient not in self._charts:
            self._charts[patient] = self.corpus.chart(patient)
        return self._charts[patient]

    def _vocab(self) -> list[str] | None:
        if not self._use_vocabulary:
            return None
        if self._vocabulary is None:
            self._vocabulary = self.corpus.doc_type_vocabulary()
        return self._vocabulary

    def _toolboxes(self, patient: str) -> list[Toolbox]:
        """Every open run for this patient, or a scratch toolbox if there are none.

        A `chart.*` call is dispatched into all of them. The read genuinely happened for each
        run, and mirroring the bookkeeping by hand instead would put a second writer on the
        coverage ledger — the divergence the single-ledger rule exists to stop.
        """
        live = [r.toolbox for r in self._runs.values() if r.patient_id == patient]
        if live:
            return live
        if patient not in self._scratch:
            chart = self._chart(patient)
            docs, _ = chart.list_documents(limit=100_000)
            self._scratch[patient] = Toolbox(chart, EvidenceLedger(),
                                             CoverageLedger(docs, []), self._vocab())
        return [self._scratch[patient]]

    def _chart_call(self, patient: str, tool: str, args: dict) -> dict:
        boxes = self._toolboxes(patient)
        out: dict = {}
        for tb in boxes:
            out, _ms = tb.dispatch(tool, args)
        recorded = [r.run_id for r in self._runs.values() if r.patient_id == patient]
        if not out.get("error"):
            out["patient"] = patient
            # Without this a caller reads a hundred documents before minting a run, then
            # cannot understand why the gate says nothing was reviewed.
            out["coverage_recorded_into"] = recorded
            if not recorded:
                out["coverage_note"] = ("no run is open for this patient, so this call counts "
                                        "towards no proof obligation; call coverage.plan first")
        return out

    def _seed_for(self, patient: str, spec_id: str) -> int:
        # One implementation, shared with `ChartReviewAgent`. This construction used to live
        # only here while the agent front end drew a random seed; the same question asked
        # over MCP and over the CLI then sampled different documents, and the CLI's answer
        # was not reproducible at all.
        return derive_sample_seed(patient, spec_id, self._seed_secret)

    # -- chart.* ------------------------------------------------------------------
    def _h_type_summary(self, patient: str) -> dict:
        return self._chart_call(patient, "document_type_summary", {})

    def _h_list_documents(self, patient: str, type: str | None = None, **kw) -> dict:
        args = {"doc_type_contains": type, "date_from": kw.get("from"), "date_to": kw.get("to"),
                "limit": int(kw.get("limit", 60)), "offset": int(kw.get("offset", 0))}
        return self._chart_call(patient, "list_documents", args)

    def _h_search(self, patient: str, query: str, type: str | None = None, **kw) -> dict:
        args = {"query": query, "doc_type_contains": type, "regex": bool(kw.get("regex", False)),
                "date_from": kw.get("from"), "date_to": kw.get("to"),
                "max_hits": int(kw.get("max_hits", 25))}
        out = self._chart_call(patient, "search_notes", args)
        if not out.get("error"):
            out["note"] = "match snippets only; call chart.read for a document's text"
        return out

    def _h_read(self, note_id: str, offset: int = 0, limit: int = 4000,
                patient: str | None = None) -> dict:
        owner, bad = self._resolve_note_owner(note_id, patient)
        if bad:
            return bad
        return self._chart_call(owner, "read_document",
                                {"note_id": note_id, "offset": int(offset), "limit": int(limit)})

    def _h_timeline(self, patient: str, type: str | None = None, limit: int = 200) -> dict:
        return self._chart_call(patient, "timeline",
                                {"doc_type_contains": type, "limit": int(limit)})

    # -- coverage.* ---------------------------------------------------------------
    def _h_plan(self, patient: str, spec_id: str) -> dict:
        spec = self.specs.get(spec_id)
        if spec is None:
            return _err("UNKNOWN_SPEC", f"no spec {spec_id!r} loaded",
                        available=sorted(self.specs))
        key = (patient, spec_id)
        if key in self._runs:
            # Frozen means frozen: the same question gets the same run, the same plan and the
            # same seed. Minting a fresh run per call would hand the caller a reroll.
            return self._runs[key].descriptor(minted=False)

        chart = self._chart(patient)
        docs, _ = chart.list_documents(limit=100_000)
        strata = strata_from_spec(spec)
        seed = self._seed_for(patient, spec_id)
        coverage = CoverageLedger(docs, strata, ForcedSampler(seed))
        evidence = EvidenceLedger()
        toolbox = Toolbox(chart, evidence, coverage, self._vocab())

        assigned = assign_strata(docs, strata) if strata else {}
        plan: dict[str, Any] = {"read_all": [], "search": [], "sample": []}
        keywords: list[str] = list(getattr(spec.proof_obligation, "required_keywords", []) or [])
        for s in strata:
            bucket = POLICY_BUCKET.get(s.policy, "search")
            for t in sorted({d.doc_type for d in assigned.get(s.name, [])}):
                if t not in plan[bucket]:
                    plan[bucket].append(t)
            keywords.extend(s.required_keywords)
        plan["keywords"] = sorted({k for k in keywords if k})
        # Derived from the spec's own stratum declarations, not from a model. `source` keeps a
        # planner guess from ever being read as a curated site binding, exactly as
        # CoveragePlan.source does.
        plan["source"] = "spec_strata" if strata else "unstratified"
        plan["mode"] = "stratified_exclusion" if strata else "unstratified"
        plan["plan_hash"] = hashlib.sha256(
            json.dumps({"plan": plan, "spec_hash": spec.spec_hash, "patient": patient},
                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

        run = RunSession(run_id=f"run-{len(self._runs_by_id) + 1:04d}-{plan['plan_hash']}",
                         patient_id=patient, spec=spec, chart=chart, evidence=evidence,
                         coverage=coverage, toolbox=toolbox, plan=plan, sample_seed=seed,
                         seed_provenance=self.seed_provenance)
        self._runs[key] = run
        self._runs_by_id[run.run_id] = run
        return run.descriptor(minted=True)

    def _h_pending_samples(self, run_id: str, **steer) -> dict:
        if steer:
            # Refused, not ignored. See STEERING_ARGS.
            return _err(
                "SAMPLE_IS_SERVER_DRAWN",
                "coverage.pending_samples takes run_id only. The seed, the sample size and "
                "the documents are all server-held: a caller that chooses which unread "
                "documents to check is validating its own judgement with its own judgement.",
                rejected_arguments=sorted(steer),
                steering_arguments=sorted(STEERING_ARGS))
        run = self._runs_by_id[run_id]
        # Same order as the agent's path: credit what has already been read before deciding
        # what is still owed, or the debt never shrinks.
        run.coverage.resolve_sample_verdicts(
            run.evidence.cited_notes(),
            keyword_hits_among_drawn(run.spec, run.coverage, run.chart))
        pending = run.coverage.pending_samples()
        return {
            "run_id": run_id, "patient": run.patient_id, "spec_id": run.spec.spec_id,
            "drawn_by": "server", "sample_seed": run.sample_seed,
            "seed_provenance": run.seed_provenance,
            "n_outstanding": sum(len(v) for v in pending.values()),
            "strata": {name: [{"note_id": d.note_id, "doc_type": d.doc_type,
                               "date": d.date.isoformat()} for d in docs]
                       for name, docs in pending.items()},
            "how_to_satisfy": ("read each note_id with chart.read, then submit them as "
                               "evidence on gate.check for any that turn out to be relevant"),
        }

    # -- gate.check ---------------------------------------------------------------
    def _h_gate_check(self, run_id: str, answer: dict | None = None) -> dict:
        run = self._runs_by_id[run_id]
        run.gate_checks += 1
        submitted = dict(answer or {})

        stripped = [k for k in SELF_ATTESTATION_KEYS if k in submitted]
        for k in stripped:
            submitted.pop(k)

        cites = self._record_evidence(run, submitted.pop("evidence", None))
        status = str(submitted.get("status", ""))
        payload = {"status": status, "value": submitted.get("value") or {},
                   "reasoning": str(submitted.get("reasoning", "")),
                   # Carried through, or this surface could not report a spec gap at all
                   # while the LangGraph one could — and which runtime an operator happened
                   # to use is not a thing the improvement loop should be conditional on.
                   "spec_section": str(submitted.get("spec_section", "")),
                   "spec_quote": str(submitted.get("spec_quote", "")),
                   "uncovered_fields": list(submitted.get("uncovered_fields") or [])}

        decision = gate_answer(run.spec, payload, evidence=run.evidence,
                               coverage=run.coverage, chart=run.chart)
        out: dict[str, Any] = {
            "run_id": run_id, "patient": run.patient_id, "spec_id": run.spec.spec_id,
            "verdict": "PASS" if decision["accepted"] else "FAIL",
            "missing": decision.get("missing", []),
            # Coverage is advisory (see `coverage.evaluate_gate`): what the ledger observed and
            # did not refuse has to reach the caller, or the model is asked to judge coverage
            # without being shown what the runtime counted.
            "advisories": decision.get("advisories", []),
            "why": decision.get("why", ""),
            "evidence_recorded": cites,
            # Reported so a caller cannot conclude its self-attestation was honoured.
            "ignored_client_claims": stripped,
            "validated_by": "server:gate.check",
        }
        for k in ("how_to_satisfy", "note_ids"):
            if k in decision:
                out[k] = decision[k]
        if not decision["accepted"]:
            out["answer"] = None
            return out

        run.validated = True   # the only assignment to this field anywhere
        run.accepted_answer = self._attest(run, payload)
        out["answer"] = run.accepted_answer
        return out

    def _record_evidence(self, run: RunSession, items: Any) -> list[dict]:
        """Evidence arrives with the answer, and the server re-reads every span.

        There is no `record_evidence` tool on this surface, so the caller cannot build a
        ledger incrementally. That is deliberate: the toolbox resolves each (note_id, start,
        end) against the document on disk, so any quote text the caller sent is discarded and
        a fabricated one cannot enter the ledger. An out-of-range span clips to empty and is
        rejected.
        """
        out: list[dict] = []
        for item in (items or []):
            if not isinstance(item, dict):
                out.append({"accepted": False, "error": "evidence items must be objects"})
                continue
            # The whitelist is closed on purpose — a caller cannot smuggle a field past the
            # toolbox by naming it. So every field the ledger accepts has to be added HERE too,
            # and one that is not is dropped in silence: `entity` was, until this line, which
            # made the two front ends disagree about what an evidence row can carry while both
            # reported success.
            args = {"note_id": str(item.get("note_id", "")),
                    "start": int(item.get("start", 0) or 0),
                    "end": int(item.get("end", 0) or 0),
                    "supports": str(item.get("supports", "")),
                    "stance": str(item.get("stance", "supports")),
                    "entity": str(item.get("entity", "") or "")}
            res, _ms = run.toolbox.dispatch("record_evidence", args)
            out.append({"note_id": args["note_id"], "accepted": bool(res.get("recorded")),
                        **({"error": res["error"]} if res.get("error") else {})})
        return out

    def _attest(self, run: RunSession, payload: dict) -> dict:
        """Build the answer the server is willing to sign. Mirrors `graph._n_finalize`."""
        ans = dict(payload)
        forced_from = None
        if run.spec.data_source == "outside_notes":
            forced_from = ans.get("status")
            ans["status"] = "SPEC_INSUFFICIENT"
        if ans["status"] == "FOUND":
            # Witness proof. Deliberately no coverage ledger: a positive never claimed the
            # universe was searched, and attaching one would advertise a stronger claim than
            # was verified.
            ans["proof_basis"] = "WITNESS"
            ans["witness_count"] = len(run.evidence.items)
        elif ans["status"] == "SPEC_INSUFFICIENT":
            # Same treatment as graph._n_finalize, from the same builder. This surface never
            # crashed on the status — it simply signed a bare code, and on the outside_notes
            # path it signed the caller's VALUE along with it. A signal that exists on one
            # front end and not another is worse than one that is missing everywhere: nobody
            # goes looking for it.
            gap, remedy = build_spec_gap(
                run.spec, ans, reported_by=("runtime" if forced_from is not None else "agent"),
                gate_validated=True)
            if forced_from is not None:
                gap["forced_over_status"] = forced_from
            ans["spec_gap"] = gap
            ans["remedy_class"] = remedy
            ans["proof_basis"] = "NOT_APPLICABLE"
            ans["coverage_note"] = ("no coverage claim is made — SPEC_INSUFFICIENT is a "
                                    "statement about the specification, not about this chart")
            strip_value_from_spec_insufficient(ans)
        elif ans["status"] == "EVIDENCE_INSUFFICIENT":
            ans["negative_basis"] = "GATE_VALIDATED"
            ans["proof_obligation"] = check_gate(run.spec, run.coverage).to_dict()
            ans["coverage_attested"] = run.coverage.to_dict()
        for k in ("spec_section", "spec_quote", "uncovered_fields"):
            # They have been folded into spec_gap (or are irrelevant to this status); leaving
            # the raw inputs beside the assembled block invites a reader to trust the copy
            # that was never validated.
            ans.pop(k, None)
        ans["evidence"] = run.evidence.to_list()
        assert_answer_is_reportable(ans)   # enforced at emission, not merely intended
        return ans

    # -- registry.truth -----------------------------------------------------------
    def _h_registry_truth(self, patient: str, variable: str, token: str = "") -> dict:
        if not self._truth_token:
            return _err("REGISTRY_TRUTH_NOT_CONFIGURED",
                        "ACR_REGISTRY_TRUTH_TOKEN is not set on this server, so ground truth "
                        "is unavailable. This is the correct state for an extraction server.")
        # Constant time: the token is the only thing standing between an extraction session
        # and the answers.
        if not hmac.compare_digest(str(token), self._truth_token):
            return _err("REGISTRY_TRUTH_FORBIDDEN",
                        "registry.truth requires the evaluation credential, which the "
                        "extraction path does not hold")
        # `patient` is the canonical id `call` substituted in — a bare directory name that is
        # a member of the corpus index. Re-checking containment anyway: this join is the one
        # the audit walked through, and a later refactor that reaches this method by some
        # other route must fail loudly rather than read whatever it was handed.
        canon, bad = self._canonical_patient(patient)
        if bad:
            return bad
        pdir = (self.corpus.root / canon).resolve()
        path = (pdir / GROUND_TRUTH_FILE).resolve()
        if not path.is_relative_to(pdir):
            return _err("MALFORMED_PATIENT_ID",
                        "resolved ground-truth path escaped the patient directory",
                        given=canon)
        # From here down the credential has been verified against a resolved patient, so
        # every exit charges the quarantine. `NO_GROUND_TRUTH` and `UNKNOWN_VARIABLE` are two
        # different answers to "is this patient in the answer key, and scored on what?" —
        # free, unlimited and unlogged, they enumerate the eval cohort for a session that is
        # about to extract it. See `_CHARGE_QUARANTINE`.
        if not path.is_file():
            return _err("NO_GROUND_TRUTH", f"no {GROUND_TRUTH_FILE} for {canon!r}",
                        **{_CHARGE_QUARANTINE: True})
        try:
            gt = json.loads(path.read_text(encoding="utf-8")).get("ground_truth") or {}
        except (OSError, ValueError) as e:
            # Caught here rather than in `call`, which cannot know the credential was checked
            # and would let an unparseable answer key be a free "this patient is scored" probe.
            return _err("GROUND_TRUTH_UNREADABLE", f"{GROUND_TRUTH_FILE} for {canon!r}: {e}",
                        **{_CHARGE_QUARANTINE: True})
        if variable not in gt:
            return _err("UNKNOWN_VARIABLE", f"no ground truth for {variable!r}",
                        available=sorted(gt), **{_CHARGE_QUARANTINE: True})
        return {"patient": canon, "variable": variable, "truth": gt[variable],
                "eval_only": True,
                "warning": ("EVAL ONLY. This patient is now quarantined for this session: "
                            "every chart.* and coverage.* call for them will be refused.")}


# ------------------------------------------------------------------------ MCP declarations
def _schema(properties: dict, required: list[str]) -> dict:
    # additionalProperties:false makes the SDK reject a stray `seed` before it reaches the
    # handler. The handler refuses it too — the invariant must hold without an MCP client.
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


_PATIENT = {"type": "string",
            "description": "patient id exactly as the corpus names it. It is an id, not a "
                           "path: separators, '..', percent-escapes and padding are refused "
                           "rather than normalised, because the ground-truth quarantine is "
                           "keyed on the resolved patient and cannot follow a spelling."}
_TYPE = {"type": "string", "description": "case-insensitive substring of a document type"}

MCP_TOOLS: list[dict] = [
    {"name": "chart.type_summary",
     "description": "Document types for one patient with counts and date spans. Metadata only.",
     "inputSchema": _schema({"patient": _PATIENT}, ["patient"])},
    {"name": "chart.list_documents",
     "description": "List a patient's documents as METADATA ONLY (type, date, size). Never text.",
     "inputSchema": _schema({"patient": _PATIENT, "type": _TYPE,
                             "from": {"type": "string", "description": "YYYY-MM-DD inclusive"},
                             "to": {"type": "string", "description": "YYYY-MM-DD inclusive"},
                             "limit": {"type": "integer"}, "offset": {"type": "integer"}},
                            ["patient"])},
    {"name": "chart.search",
     "description": "Search a patient's documents. Returns match snippets with citable "
                    "character offsets, not full document text.",
     "inputSchema": _schema({"patient": _PATIENT, "query": {"type": "string"}, "type": _TYPE,
                             "regex": {"type": "boolean"}, "from": {"type": "string"},
                             "to": {"type": "string"}, "max_hits": {"type": "integer"}},
                            ["patient", "query"])},
    {"name": "chart.read",
     "description": "Read a document's text, paginated. Offsets are stable and citable. "
                    "note_id is unique only within a patient: pass patient= if this session "
                    "has more than one chart open.",
     "inputSchema": _schema({"note_id": {"type": "string"}, "offset": {"type": "integer"},
                             "limit": {"type": "integer"}, "patient": _PATIENT}, ["note_id"])},
    {"name": "chart.timeline",
     "description": "Chronological list of a patient's documents, optionally filtered by type.",
     "inputSchema": _schema({"patient": _PATIENT, "type": _TYPE,
                             "limit": {"type": "integer"}}, ["patient"])},
    {"name": "coverage.plan",
     "description": "Mint (or return) the FROZEN read/search/sample plan for one patient and "
                    "spec, with the server-held sampling seed. Calling twice returns the same "
                    "run, the same plan and the same seed.",
     "inputSchema": _schema({"patient": _PATIENT, "spec_id": {"type": "string"}},
                            ["patient", "spec_id"])},
    {"name": "coverage.pending_samples",
     "description": "The validation documents the SERVER has drawn for this run. You cannot "
                    "pass a seed, a sample size or a document list; the draw is not yours to "
                    "make. Read them, then cite the relevant ones on gate.check.",
     "inputSchema": _schema({"run_id": {"type": "string"}}, ["run_id"])},
    {"name": "gate.check",
     "description": "Submit an answer for validation. This is the ONLY way an answer becomes "
                    "validated; any gate_validated / coverage_attested / proof_basis field you "
                    "send is stripped and reported back. Returns PASS/FAIL and what is missing.",
     "inputSchema": _schema({"run_id": {"type": "string"},
                             "answer": {"type": "object", "description":
                                        "status (FOUND | EVIDENCE_INSUFFICIENT | "
                                        "SPEC_INSUFFICIENT), value, reasoning, and evidence: "
                                        "[{note_id, start, end, supports, stance}]. For "
                                        "SPEC_INSUFFICIENT add spec_section (one of "
                                        + ", ".join(SPEC_SECTIONS) + "), optionally "
                                        "spec_quote and uncovered_fields, and send no value "
                                        "— the specification cannot both fail to cover the "
                                        "case and decide it."}},
                            ["run_id", "answer"])},
    {"name": "registry.truth",
     "description": "EVAL ONLY. Registry ground truth for one patient and variable. Requires "
                    "the evaluation credential and refuses any patient this session has "
                    "served chart.* or coverage.* calls for, in either order.",
     "inputSchema": _schema({"patient": _PATIENT, "variable": {"type": "string"},
                             "token": {"type": "string"}}, ["patient", "variable", "token"])},
]


def build_mcp_server(service: ChartReviewService, name: str = "acr-chart-review"):
    """Wrap the service in an MCP server. Import is local so the service stays usable —
    and testable — in an environment where the SDK is not installed."""
    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [types.Tool(name=t["name"], description=t["description"],
                           inputSchema=t["inputSchema"]) for t in MCP_TOOLS]

    @server.call_tool()
    async def _call_tool(tool: str, arguments: dict | None) -> list[types.TextContent]:
        result = service.call(tool, arguments or {})
        return [types.TextContent(type="text",
                                  text=json.dumps(result, indent=2, default=str))]

    return server


def main() -> None:
    import argparse

    import anyio
    from mcp.server.stdio import stdio_server

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus/patients")
    ap.add_argument("--specs", default="specs")
    args = ap.parse_args()

    service = ChartReviewService(args.corpus, args.specs)
    server = build_mcp_server(service)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


if __name__ == "__main__":
    main()
