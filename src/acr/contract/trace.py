"""Run tracing, and the rule attribution that makes a trace routable.

Everything the agent does is appended to a JSONL trace: plan revisions, every tool call
with full input/output, every reflection verdict, every rejected answer, and the final
coverage attestation. The trace is the artifact that makes a label auditable — without it
you cannot tell a correct answer from a lucky one.

`to_capg()` reshapes the trace into the observation-tree form the CAPG adapter consumes,
so these runs drop straight into an existing provenance-graph pipeline.

WHICH SPEC RULE WAS IN PLAY
---------------------------
A trace of tool calls, arguments and messages says what happened and nothing about which
sentence of the specification was responsible. §6b's optimizer has to route a wrong answer
at one piece of text; with no attribution it reverse-engineers a story from the outcome,
and a loop that does that starts confidently rewriting rules that were never at fault.
So this module adds two things:

  1. `rule_catalog()` — a stable identifier for every rule in a spec (below).
  2. Attribution events on the tracer, each one MARKED with where it came from.

THE MARKING IS NOT DECORATION. Three channels feed attribution and they are not equally
trustworthy:

  DETERMINISTIC   computed by code that already knew the answer — which `answer_check`
                  rejected a submission (`answer_checks.check_answer_detail`), and which
                  stratum/`establishes` rule made a cited document admissible
                  (`coverage.admissibility_for_citations`). Reproducible from the spec and
                  the ledger, with no model in the loop.

  SELF_REPORTED   which `decision_rule` / `conflict_rule` the agent SAYS it applied. This
                  is legitimate — it is a report about which rule it used, not a claim that
                  the answer is right, so it does not carry the credibility problem that a
                  self-assessed confidence score does — but it is still a model output, and
                  a downstream consumer that cannot tell it from a computed fact will weight
                  a hallucination like a measurement.

Every event carries `provenance` on the event itself, and the manifest block keeps the two
in separately named sub-objects. Redundant on purpose: sub-objects get flattened, merged and
re-serialised downstream, and an item that loses its parent key must still say what it is.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .answer_checks import answer_check_rule_id, field_rule_id


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ===========================================================================================
#                                     RULE IDENTITY
# ===========================================================================================
#: Provenance markers. Uppercase strings rather than booleans: a boolean named `deterministic`
#: reads as `False` when it is simply absent, and "unmarked" must never render as "the model
#: said so" or as "the code computed it". An absent marker is a bug, and an unknown string is
#: visibly an unknown string.
PROV_DETERMINISTIC = "DETERMINISTIC"
PROV_SELF_REPORTED = "SELF_REPORTED"

#: The envelope `Tracer.emit` writes itself. A payload key that collides with one of these is
#: re-keyed to `payload_<key>` and the collision is recorded on the event: an emitter that
#: overwrote `kind` would re-file its event as a different type of event, and every reader in
#: this tree — `to_capg`, the manifest builders, the trace CLI — splits on `kind` first.
RESERVED_EVENT_KEYS = ("run_id", "seq", "ts", "elapsed_s", "kind")

#: Rule-id namespaces. Everything downstream splits on the first dot, so the namespace is the
#: kind and the remainder is the address within it.
RULE_NAMESPACES = (
    "decision_rule",            # spec.decision_rule[i]                     — prose, prompt only
    "conflict_rule",            # spec.conflict_rules[i]                    — prose, prompt only
    "evidence_rule",            # spec.evidence_rules.<clause>[i], strata, witness
    "answer_check",             # spec.answer_checks[]                      — enforced in code
    "field_format",             # spec.fields[].format                      — enforced in code
    "field_allowable_values",   # spec.fields[].allowable_values            — enforced in code
    "abstention",               # spec.abstention.<key>                     — prose, prompt only
    "proof_obligation",         # for_positive / for_negative statements    — prose + gate
)

_ID_RE = re.compile(r"\b(?:" + "|".join(RULE_NAMESPACES) + r")\.[A-Za-z0-9_.#:\-]*[A-Za-z0-9_#\-]")

#: Namespaces the agent is asked to cite. The other namespaces are the CODE's channel: an
#: agent naming the answer_check that rejected it is telling us nothing we did not compute.
SELF_REPORTABLE_NAMESPACES = ("decision_rule", "conflict_rule", "evidence_rule")

#: How many unrecognised identifiers to keep. Hallucinated citations are DISCARDED as
#: citations and COUNTED, but a few are kept verbatim because "the model keeps inventing
#: `decision_rule.9` for a spec with seven rules" is an actionable fact about the prompt,
#: while an unbounded list is a place for a degenerate run to dump kilobytes.
MAX_KEPT_UNRECOGNISED = 20

#: How many individual rejection rows the MANIFEST keeps. The per-rule totals and the maximum
#: consecutive run are aggregates and are never truncated, so the loop signal survives intact;
#: what is dropped is the tail of a run that submitted the same refused answer forty times,
#: which is forty copies of one message. The trace keeps every one of them.
MAX_KEPT_REJECTION_ROWS = 50


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_") or "x"


def _text_of(raw: Any) -> str:
    """One canonical string for a rule, whatever shape the YAML gave it."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict) and set(raw) == {"if", "then"}:
        return f"IF {raw['if']} THEN {raw['then']}"
    return json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RuleRef:
    """One addressable rule of one spec version.

    `text_sha` is the load-bearing field and the reason a positional id is safe to use. A
    position is stable across runs of the same spec version and nothing more; the fingerprint
    is what lets a reader of an old manifest discover that `decision_rule.3` is no longer the
    sentence it was attributing to. An id without it would silently re-point at whatever
    sentence later took the slot — the same failure `_answer_check_key` was written to avoid.
    """

    rule_id: str
    kind: str
    text: str
    text_sha: str
    #: The `spec.enforced_elements()` path, when some code path changes behaviour on this
    #: rule. None means prose: rendered into the prompt, applied by the model, checked by
    #: nobody — which is exactly the distinction §6b's last leaf turns on.
    enforced_path: str | None = None
    #: The `specview` element id for the same rule, so a routed gradient can be shown to a
    #: clinician in the document they already read.
    view_id: str | None = None
    #: True when this id had to be disambiguated because two rules collided. See rule_catalog.
    ambiguous_id: bool = False

    def to_dict(self, with_text: bool = True) -> dict:
        d = {"rule_id": self.rule_id, "kind": self.kind, "text_sha": self.text_sha,
             "enforced_path": self.enforced_path, "view_id": self.view_id}
        if with_text:
            d["text"] = self.text
        if self.ambiguous_id:
            d["ambiguous_id"] = True
        return d


def rule_catalog(spec: Any) -> list[RuleRef]:
    """Every rule in `spec`, with an identifier that is the same on every run of this version.

    THE SCHEME, and why each part of it is what it is:

      * Where the spec already carries a CONTENT identity, that identity is reused rather
        than re-derived — `answer_check.<field>.<kind>[.<first nos value>]` is
        `spec._answer_check_key`, and a stratum is addressed by its declared `name`. Content
        identities survive an insertion above them; positions do not.
      * Everything else is POSITIONAL and 1-based — `decision_rule.3`, `conflict_rule.1`,
        `evidence_rule.does_not_count.2` — because YAML lists of prose have no other stable
        handle, and inventing one from a hash of the sentence would change the id on a typo
        fix and orphan every attribution ever recorded against it. Positions are numbered in
        declaration order, which `load_spec` preserves, so the id is deterministic for a
        given spec version. `text_sha` carries the honesty: a position whose fingerprint
        moved is a position that means something else now.
      * 1-based to match what `as_prompt_block` renders ("DECISION RULES: 1. ...") and what
        `specview` shows a reviewer. An id the agent is asked to cite must be the number it
        can actually see, or the self-report channel measures our indexing convention.

    Never raises. A trace that can abort a run is worse than an imperfect identifier.
    """
    out: list[RuleRef] = []

    def add(rule_id: str, kind: str, raw: Any, view_id: str | None = None) -> None:
        t = _text_of(raw)
        out.append(RuleRef(rule_id=rule_id, kind=kind, text=t, text_sha=_sha(t),
                           view_id=view_id))

    for i, r in enumerate(getattr(spec, "decision_rule", []) or [], start=1):
        add(f"decision_rule.{i}", "decision_rule", r, f"rule.{i}")
    for i, c in enumerate(getattr(spec, "conflict_rules", []) or [], start=1):
        add(f"conflict_rule.{i}", "conflict_rule", c, f"conflict.{i}")

    ev = getattr(spec, "evidence_rules", None) or {}
    if isinstance(ev, dict):
        for clause, items in ev.items():
            seq = items if isinstance(items, (list, tuple)) else [items]
            view = {"counts_as_evidence": "accept", "does_not_count": "refuse"}.get(str(clause))
            for i, item in enumerate(seq, start=1):
                add(f"evidence_rule.{_slug(clause)}.{i}", "evidence_rule", item,
                    f"{view}.{i}" if view else None)

    po = getattr(spec, "proof_obligation", None)
    fn = (getattr(po, "for_negative", None) or {}) if po is not None else {}
    scopes: list[tuple[str, dict]] = [("", fn if isinstance(fn, dict) else {})]
    for claim in ((fn.get("claims") or []) if isinstance(fn, dict) else []):
        scopes.append((f"claim.{claim.get('id')}.", claim))
    for prefix, holder in scopes:
        for s in (holder.get("strata") or []):
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "?"))
            # The ADMISSIBILITY rule: which fields documents in this stratum may establish.
            # This is `evidence_rules.does_not_count` written where code can read it, which
            # is why it lives in the evidence_rule namespace and not a stratum one.
            add(f"evidence_rule.stratum.{prefix}{name}.establishes", "evidence_rule_stratum",
                {"stratum": name, "establishes": list(s.get("establishes") or []),
                 "match": s.get("match") or {"partition_by": s.get("partition_by")}})
    if po is not None:
        for fname, groups in (getattr(po, "witness_strata", None) or {}).items():
            add(f"evidence_rule.witness.{fname}", "evidence_rule_witness",
                {"field": fname, "strata": list(groups)}, f"proof.witness.{fname}")
        pos = getattr(po, "positive_statement", "")
        if pos:
            add("proof_obligation.for_positive", "proof_obligation", pos, "proof.positive")
    if isinstance(fn, dict) and fn.get("statement"):
        add("proof_obligation.for_negative", "proof_obligation", fn["statement"],
            "proof.negative")

    for f in getattr(spec, "fields", []) or []:
        if getattr(f, "format", None):
            add(field_rule_id("field_format", f.name), "field_format", f.format,
                f"answer.{f.name}")
        if getattr(f, "allowable_values", None):
            add(field_rule_id("field_allowable_values", f.name), "field_allowable_values",
                list(f.allowable_values), f"answer.{f.name}")

    for i, chk in enumerate(getattr(spec, "answer_checks", []) or [], start=1):
        add(answer_check_rule_id(chk, i), "answer_check", chk, f"check.{i}")

    for k, v in (getattr(spec, "abstention", None) or {}).items():
        add(f"abstention.{_slug(k)}", "abstention", v, f"refusal.{k}")

    # Enforced paths, attached by kind+address rather than re-derived, so the catalog and the
    # provenance channel cannot drift into two different opinions about what code reads.
    try:
        from .spec import enforced_elements
        enforced = list(enforced_elements(spec))
    except Exception:      # noqa: BLE001 - a spec that cannot enumerate must still trace
        enforced = []
    by_path = {e.path: e for e in enforced}
    resolved: list[RuleRef] = []
    for r in out:
        path = None
        if r.kind == "answer_check":
            path = next((p for p in by_path if p.startswith("answer_checks[")
                         and p == f"answer_checks[{r.rule_id.split('.', 1)[1]}]"), None)
        elif r.kind in ("field_format", "field_allowable_values"):
            path = f"fields[{r.rule_id.split('.', 1)[1]}].{r.kind.replace('field_', '')}"
            path = path if path in by_path else None
        elif r.kind == "evidence_rule_stratum":
            tail = r.rule_id[len("evidence_rule.stratum."):-len(".establishes")]
            cand = [p for p in by_path
                    if p.endswith(f".strata[{tail.split('.')[-1]}].establishes")]
            path = cand[0] if len(cand) == 1 else None
        resolved.append(RuleRef(r.rule_id, r.kind, r.text, r.text_sha, path, r.view_id))

    # Collisions are already refused at spec load (`enforced_elements` raises when two rules
    # resolve to one path), so this cannot fire for a loaded spec. It exists because the
    # catalog is also built over hand-made objects in tests and over half-edited drafts, and
    # two rules sharing one id is exactly the silent mis-attribution the ids exist to prevent:
    # one of them would be credited with the other's failures. Disambiguate, mark it, move on.
    seen: dict[str, int] = {}
    final: list[RuleRef] = []
    for r in resolved:
        n = seen.get(r.rule_id, 0)
        seen[r.rule_id] = n + 1
        if n:
            final.append(RuleRef(f"{r.rule_id}#{n + 1}", r.kind, r.text, r.text_sha,
                                 r.enforced_path, r.view_id, ambiguous_id=True))
        else:
            final.append(r)
    return final


def rule_index(spec: Any) -> dict[str, RuleRef]:
    return {r.rule_id: r for r in rule_catalog(spec)}


def rule_catalog_hash(catalog: Sequence[RuleRef]) -> str:
    """Identifies the catalog, so an attribution can be checked against the rules it names."""
    return _sha("\n".join(f"{r.rule_id}:{r.text_sha}" for r in catalog))


def rule_citation_block(spec: Any, max_chars: int = 160) -> str:
    """The ASK for the self-report channel: cite rules by an identifier the spec declares.

    Rendered from the catalog rather than written by hand, so the identifiers the agent is
    shown are by construction the identifiers the parser accepts. Asking for free-text rule
    names instead would guarantee a hallucination rate we then have to measure.
    """
    rows = [r for r in rule_catalog(spec)
            if r.rule_id.split(".", 1)[0] in SELF_REPORTABLE_NAMESPACES
            and r.kind in ("decision_rule", "conflict_rule", "evidence_rule")]
    if not rows:
        return ""
    L = ["RULE IDENTIFIERS FOR THIS SPECIFICATION",
         "When you submit, name the rules you ACTUALLY APPLIED. Put them on their own line "
         "inside `reasoning`, exactly like this:",
         "    rules_applied: decision_rule.1, conflict_rule.2",
         "Use only identifiers from the list below, copied exactly. An identifier that is "
         "not on the list is discarded and counted as a misattribution, so naming none is "
         "better than naming one you did not use.",
         ""]
    for r in rows:
        t = " ".join(r.text.split())
        L.append(f"  {r.rule_id}  {t[:max_chars]}{'…' if len(t) > max_chars else ''}")
    return "\n".join(L)


def parse_rule_citations(source: Any, known: Iterable[str]) -> tuple[list[str], list[str]]:
    """Pull rule identifiers out of an agent's self-report. Returns (recognised, unknown).

    Deliberately EXACT: only strings matching a declared namespace and then present in the
    catalog are recognised. No fuzzy matching, no "decision rule 3" -> `decision_rule.3`,
    no nearest-neighbour repair. A parser that guesses what the model meant is manufacturing
    the very citation the citation requirement exists to demand, and the repaired guess would
    be indistinguishable from a real one the moment it is written down.

    Unknown identifiers are RETURNED SEPARATELY and never merged into the recognised list:
    storing a hallucinated citation is worse than storing nothing, because the optimizer would
    then route a gradient at a rule that does not exist and find the spec "silent" about it.
    """
    kn = set(known)
    if isinstance(source, (list, tuple, set)):
        text = " ".join(str(x) for x in source)
    elif isinstance(source, dict):
        text = " ".join(str(v) for v in source.values())
    else:
        text = str(source or "")
    good: list[str] = []
    bad: list[str] = []
    for m in _ID_RE.findall(text):
        tok = m.rstrip(".:-")
        target = good if tok in kn else bad
        if tok not in target:
            target.append(tok)
    return good, bad


@dataclass
class Tracer:
    run_id: str
    path: Path
    events: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.time)
    _emit_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # -- rule attribution state ---------------------------------------------------
    #: The spec's rule catalog, bound once per run. Empty until `bind_spec` is called, and
    #: everything below degrades to "record it, recognise nothing" in that state rather than
    #: refusing: a front end that forgets to bind must still produce a readable trace.
    rules: dict[str, RuleRef] = field(default_factory=dict)
    spec_identity: dict = field(default_factory=dict)
    #: rule_id -> how many gate evaluations in a row this check has rejected the answer.
    #: THE REPEAT COUNT IS ITSELF A SIGNAL. A run rejected twice for coding 8046 over "favor
    #: squamous" then burned a 400k-token budget without revising: two identical rejections
    #: followed by no revision indicts the rejection MESSAGE at least as much as the rule, and
    #: that is a different parameter with a different owner (§6b: engineer, not clinician).
    #: A total alone cannot tell a loop from three unrelated rejections spread over a run.
    rule_rejection_streak: dict[str, int] = field(default_factory=dict)
    rule_rejection_total: dict[str, int] = field(default_factory=dict)
    rule_rejection_max_streak: dict[str, int] = field(default_factory=dict)
    rule_rejections: list[dict] = field(default_factory=list)
    admissibility: list[dict] = field(default_factory=list)
    self_reported_counts: dict[str, int] = field(default_factory=dict)
    unrecognised_rule_ids: dict[str, int] = field(default_factory=dict)
    #: Counted independently of the kept strings. `unrecognised_rule_ids` is capped, so its
    #: length stops growing; the misattribution RATE must not stop growing with it, or a run
    #: that invented three hundred ids would report the same twenty as one that invented
    #: twenty-one.
    n_unrecognised_total: int = 0
    n_self_report_claims: int = 0
    n_gate_evaluations: int = 0

    @classmethod
    def create(cls, out_dir: str | Path, run_id: str | None = None) -> Tracer:
        rid = run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        d.chmod(0o700)
        path = d / f"{rid}.jsonl"
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        return cls(run_id=rid, path=path)

    def emit(self, kind: str, /, **payload: Any) -> dict:
        """Append one event. `kind` is POSITIONAL-ONLY, and a colliding payload key is
        re-keyed rather than allowed to explode or to overwrite the envelope.

        Both halves of that were paid for. `emit("trigger", **trigger.to_dict())` passed
        `kind` twice — `Trigger.to_dict()` has a `kind` of its own — and the TypeError killed
        the run at the first trigger it detected, in a runtime whose entire job is to survive
        a bad tool result and keep going. Tracing is instrumentation: it may lose a field, it
        may not take the task down with it. The positional-only marker makes the collision
        impossible to raise; the re-keying below makes it impossible to hide, because a
        payload that silently overwrote `kind` would re-file the event as some other type of
        event and the reader would never know.
        """
        with self._emit_lock:
            clashes = [k for k in payload if k in RESERVED_EVENT_KEYS]
            for k in clashes:
                payload[f"payload_{k}"] = payload.pop(k)
            ev = {
                "run_id": self.run_id,
                "seq": len(self.events),
                "ts": _now(),
                "elapsed_s": round(time.time() - self.t0, 3),
                "kind": kind,
                **payload,
            }
            if clashes:
                ev["reserved_key_collisions"] = clashes
                ev["severity"] = "warning"
            self.events.append(ev)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            self.path.chmod(0o600)
        return ev

    # convenience emitters ------------------------------------------------------
    def run_start(self, **kw): return self.emit("run_start", **kw)
    def plan(self, plan, revision, rationale=""): return self.emit("plan", plan=plan, revision=revision, rationale=rationale)
    def llm(self, role, content, tool_calls=None, usage=None):
        return self.emit("llm", role=role, content=content, tool_calls=tool_calls or [], usage=usage or {})
    def tool(self, name, args, result, ok=True, ms=0.0, because=""):
        # Promoted out of `args` and into the envelope on purpose. It is already inside `args`,
        # but a consumer that has to reach into a free-form argument bag to find the causal
        # link will not do it, and a field nobody reads is a field nobody maintains.
        return self.emit("tool", tool=name, args=args, result=result, ok=ok, ms=ms,
                         because=str(because or ""))
    def reflect(self, verdict, reason, evidence_count):
        return self.emit("reflect", verdict=verdict, reason=reason, evidence_count=evidence_count)
    def rejected(self, why, missing, attempted):
        return self.emit("answer_rejected", why=why, missing=missing, attempted=attempted)
    def run_end(self, **kw): return self.emit("run_end", **kw)

    def trigger(self, **t: Any) -> dict:
        """One mechanically detected trigger, from `coverage_planner.Trigger.to_dict()`.

        THE TRIGGER'S OWN KIND IS RE-KEYED TO `trigger`. `kind` is the trace envelope's event
        type — every reader in the tree splits on it — and one key cannot mean both "this is
        a trigger event" and "this trigger is an UNLISTED_ANSWER_TERM". Naming it `trigger`
        also matches `term_provenance`, where the trigger that caused an added term has been
        under that key all along, so a develop-plane consumer joins the two on one name.
        """
        payload = dict(t)
        payload["trigger"] = payload.pop("kind", "")
        return self.emit("trigger", **payload)

    # -- rule attribution ---------------------------------------------------------
    def bind_spec(self, spec: Any) -> dict:
        """Write the rule catalog into the trace, once, before anything cites it.

        The catalog goes in the trace and not only in the manifest because an id is
        meaningless without the text it names: a reader six months from now holding
        `decision_rule.3` and a since-edited spec has nothing unless the fingerprint and the
        sentence travelled with the run.
        """
        cat = rule_catalog(spec)
        self.rules = {r.rule_id: r for r in cat}
        self.spec_identity = dict(spec.identity()) if hasattr(spec, "identity") else {}
        return self.emit("rule_catalog", provenance=PROV_DETERMINISTIC,
                         source="acr.contract.trace.rule_catalog",
                         **self.spec_identity,
                         rule_catalog_hash=rule_catalog_hash(cat),
                         n_rules=len(cat),
                         rules=[r.to_dict() for r in cat])

    def note_gate_evaluation(self) -> int:
        """One submitted answer was evaluated. Returns the new count.

        Owns `n_gate_evaluations` because it is the only thing that happens on EVERY evaluation.
        The counter used to live in `answer_check_outcome`, which was called unconditionally --
        including on clean evaluations, so that a rejection streak could see its gaps. After the
        clinical answer_checks were removed on 2026-07-30 that method only runs when something
        actually fired, and a count incremented there would have been a count of rejections
        wearing the name of evaluations. `rejections_by_rule` and the streak logic still live
        there, where they belong.
        """
        self.n_gate_evaluations += 1
        return self.n_gate_evaluations

    def answer_check_outcome(self, violations: Sequence[Any], *, status: str = "FOUND") -> dict:
        """Record which checks rejected this submission — and which did not.

        Called on EVERY evaluation that actually ran the checks, including the clean ones,
        because a streak is only a streak if the gaps are observed. Recording rejections
        alone would make "rejected, revised, rejected again for something else, rejected for
        the first rule once more" read as a three-deep loop on rule one.
        """
        # NOT incremented here any more. `note_gate_evaluation` owns the count, called once per
        # `gate_answer` regardless of what fired — this method now runs only when something did,
        # so counting here would count rejections and call them evaluations.
        # Two projections of one violation, and the difference is the quote. The trace keeps
        # it — it already holds every document the agent read, so the text is not new there —
        # and the manifest keeps `evidence_index` instead, which points into the evidence
        # ledger the manifest is already carrying. One copy of a chart quote, in one place.
        full = [v.to_dict(with_quote=True) if hasattr(v, "to_dict") else dict(v)
                for v in violations]
        fired = {r.get("rule_id", "") for r in full}
        for rid in list(self.rule_rejection_streak):
            if rid not in fired:
                self.rule_rejection_streak[rid] = 0
        for r in full:
            rid = r.get("rule_id", "")
            n = self.rule_rejection_streak.get(rid, 0) + 1
            self.rule_rejection_streak[rid] = n
            self.rule_rejection_total[rid] = self.rule_rejection_total.get(rid, 0) + 1
            self.rule_rejection_max_streak[rid] = max(
                self.rule_rejection_max_streak.get(rid, 0), n)
            r["consecutive_rejections"] = n
            r["gate_evaluation"] = self.n_gate_evaluations
            r["provenance"] = PROV_DETERMINISTIC
            r["source"] = "acr.contract.answer_checks"
            # None when no catalog was bound: "this rule is not in the spec" and "nobody told
            # me what the spec declares" are different findings and must not share a False.
            r["known_rule"] = (rid in self.rules) if self.rules else None
            self.rule_rejections.append({k: v for k, v in r.items() if k != "quote"})
        return self.emit("rule_rejection", provenance=PROV_DETERMINISTIC,
                         source="acr.contract.answer_checks",
                         gate_evaluation=self.n_gate_evaluations,
                         submitted_status=status,
                         n_violations=len(full), violations=full,
                         rejected_by=sorted(fired))

    def evidence_admissibility(self, records: Sequence[dict]) -> dict:
        """Record which evidence rule admitted or refused each cited document, at gate time.

        `enforced_by_gate` travels on every record and is currently False for all of them.
        That is not a defect to hide — nothing in `gate_answer` refuses a citation for coming
        from a stratum that cannot establish the field — and a downstream reader who assumed
        otherwise would conclude the gate had already filtered the ledger and stop looking.
        """
        self.admissibility = [dict(r) for r in records]
        return self.emit("evidence_admissibility", provenance=PROV_DETERMINISTIC,
                         source="acr.review.coverage.admissibility_for_citations",
                         gate_evaluation=self.n_gate_evaluations,
                         n_citations=len(records), citations=list(records))

    def self_reported_rules(self, source_text: Any, *, where: str) -> dict:
        """Record which rules the AGENT says it applied. Marked, counted, never promoted.

        Hallucinated identifiers are discarded as citations and counted as misattributions.
        A count is kept per identifier because the interesting failure is systematic — a
        model that repeatedly cites `decision_rule.9` in a spec with seven rules is telling
        us the prompt's numbering is not reaching it, which is a fixable thing.
        """
        good, bad = parse_rule_citations(source_text, self.rules)
        if not self.rules:
            # NO CATALOG BOUND, so nothing can be checked against anything. Counting these
            # as hallucinations would manufacture a 100% misattribution rate out of a front
            # end that forgot to call `bind_spec` — a measurement of our own wiring, reported
            # as a fact about the model. Recorded verbatim, classified as neither.
            self.n_self_report_claims += len(bad)
            return self.emit("rules_self_reported", provenance=PROV_SELF_REPORTED,
                             source=f"agent:{where}", catalog_bound=False,
                             unclassified_rule_ids=bad,
                             caveat=("no rule catalog was bound for this run, so these "
                                     "identifiers are neither recognised nor rejected"))
        self.n_self_report_claims += len(good) + len(bad)
        for rid in good:
            self.self_reported_counts[rid] = self.self_reported_counts.get(rid, 0) + 1
        self.n_unrecognised_total += len(bad)
        for rid in bad:
            if rid in self.unrecognised_rule_ids or len(self.unrecognised_rule_ids) < MAX_KEPT_UNRECOGNISED:
                self.unrecognised_rule_ids[rid] = self.unrecognised_rule_ids.get(rid, 0) + 1
        return self.emit("rules_self_reported", provenance=PROV_SELF_REPORTED,
                         source=f"agent:{where}",
                         caveat=("the agent's own report of WHICH RULE IT USED. Not a claim "
                                 "that the answer is right, and not a computed fact."),
                         rules_claimed=good,
                         unrecognised_rule_ids=bad,
                         n_unrecognised=len(bad),
                         catalog_bound=bool(self.rules))

    def rule_attribution(self) -> dict:
        """The attribution block for the run manifest.

        In the manifest AS WELL AS the trace, because attribution that can only be recovered
        by replaying a JSONL file is attribution nobody will compute: the §6b loop reads a
        directory of finished runs, and a completed run has to be able to say which rule was
        in play without being re-parsed.

        The two channels are separate top-level keys, each item additionally self-marked.
        A single merged list of "rules involved" is the one shape this must never take —
        downstream it becomes impossible to tell the check that provably fired from the rule
        the model says it thought about.
        """
        loops = sorted(rid for rid, n in self.rule_rejection_max_streak.items() if n >= 2)
        return {
            "schema_version": 1,
            **self.spec_identity,
            "rule_catalog_hash": rule_catalog_hash(list(self.rules.values())),
            # ids + fingerprints only. The rule TEXT lives in the spec, addressed by
            # spec_hash, and in the trace's `rule_catalog` event; a third copy in every
            # manifest would be the thing that goes stale.
            "rule_catalog": [r.to_dict(with_text=False) for r in self.rules.values()],
            "deterministic": {
                "provenance": PROV_DETERMINISTIC,
                "what_this_is": ("computed by code from the spec and the ledgers; no model "
                                 "judgement is involved and it replays identically"),
                "n_gate_evaluations": self.n_gate_evaluations,
                "answer_check_rejections": self.rule_rejections[:MAX_KEPT_REJECTION_ROWS],
                "rejections_truncated": len(self.rule_rejections) > MAX_KEPT_REJECTION_ROWS,
                "n_rejection_rows": len(self.rule_rejections),
                "rejected_by": sorted(self.rule_rejection_total),
                "rejections_by_rule": dict(sorted(self.rule_rejection_total.items())),
                "max_consecutive_by_rule": dict(sorted(self.rule_rejection_max_streak.items())),
                # Rules that rejected the same run twice running. Read this before reading
                # the rule itself: a repeated identical rejection that the agent never
                # answered is evidence about the MESSAGE, which is an engineer's parameter.
                "rejection_loops": loops,
                "evidence_admissibility": list(self.admissibility),
            },
            "self_reported": {
                "provenance": PROV_SELF_REPORTED,
                "what_this_is": ("the agent's report of which rule it applied. Legitimate as "
                                 "a report about its own reasoning, and not evidence that the "
                                 "rule was applied correctly or that the answer is right."),
                "rules_claimed": sorted(self.self_reported_counts),
                "claims_by_rule": dict(sorted(self.self_reported_counts.items())),
                "n_claims": self.n_self_report_claims,
                # Kept apart from `rules_claimed` on purpose: an id the spec never declared
                # is discarded as a citation and survives only as a count, so no consumer can
                # accidentally route a gradient at a rule that does not exist.
                "unrecognised_rule_ids": dict(sorted(self.unrecognised_rule_ids.items())),
                "n_unrecognised": self.n_unrecognised_total,
                "unrecognised_ids_truncated": (
                    len(self.unrecognised_rule_ids) >= MAX_KEPT_UNRECOGNISED),
            },
        }

    # export --------------------------------------------------------------------
    def write_manifest(self, manifest: dict) -> Path:
        p = self.path.with_suffix(".manifest.json")
        p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return p

    def to_capg(self, name: str = "") -> dict:
        """Observation-tree shape: {id, name, observations:[{id,name,type,parent_observation_id,...}]}."""
        obs: list[dict] = []
        root = f"{self.run_id}-root"
        obs.append({"id": root, "name": name or "chart-review", "type": "SPAN",
                    "parent_observation_id": None, "start_time": self.events[0]["ts"] if self.events else _now()})
        for ev in self.events:
            oid = f"{self.run_id}-{ev['seq']}"
            if ev["kind"] == "plan":
                # The plan is the RETRIEVAL plan now — read_all / search / sample plus the
                # term list — not a list of prose goals. Rendered as todos with the policy as
                # the status, because that is the shape the CAPG adapter consumes and because
                # "which types was this run allowed to open, at this revision" is the thing a
                # provenance graph should be able to show. `plan_todos` also accepts the old
                # list-of-goals shape, so traces recorded before this change still render.
                obs.append({"id": oid, "name": "write_todos", "type": "EVENT",
                            "parent_observation_id": root, "start_time": ev["ts"],
                            "output": {"todos": plan_todos(ev.get("plan"))}})
            elif ev["kind"] == "tool":
                obs.append({"id": oid, "name": ev["tool"], "type": "TOOL",
                            "parent_observation_id": root, "start_time": ev["ts"],
                            "input": ev.get("args"), "output": ev.get("result")})
            elif ev["kind"] == "llm":
                obs.append({"id": oid, "name": "generation", "type": "GENERATION",
                            "parent_observation_id": root, "start_time": ev["ts"],
                            "output": {"content": ev.get("content", "")}})
            elif ev["kind"] in ("reflect", "answer_rejected"):
                obs.append({"id": oid, "name": ev["kind"], "type": "EVENT",
                            "parent_observation_id": root, "start_time": ev["ts"], "output": ev})
        return {"id": self.run_id, "name": name or "chart-review", "observations": obs}


def plan_todos(plan: Any) -> list[dict]:
    """One renderer for both plan shapes, so a trace directory stays readable across the cut.

    The revisable plan used to be a list of {id, goal, rationale} that governed nothing. It
    is now `coverage_planner.CoveragePlan.to_dict()`, which governs what may be opened. Both
    shapes exist on disk under `runs/`, and a reader that handles only the new one silently
    renders 37 historical runs as empty.
    """
    if isinstance(plan, dict):
        rows = [{"content": f"{policy}: {t}", "status": policy}
                for policy in ("read_all", "search", "sample")
                for t in (plan.get(policy) or [])]
        if plan.get("keywords"):
            rows.append({"content": "search terms: " + ", ".join(plan["keywords"]),
                         "status": "keywords"})
        return rows
    return [{"content": s.get("goal", ""), "status": s.get("status", "")}
            for s in (plan or []) if isinstance(s, dict)]


def plan_summary(plan: Any) -> str:
    """One line for a terminal trace listing. Same two shapes; see `plan_todos`."""
    if isinstance(plan, dict):
        n = {k: len(plan.get(k) or []) for k in ("read_all", "search", "sample")}
        added = len(plan.get("terms_added") or [])
        proms = len(plan.get("promotions") or [])
        return (f"read_all={n['read_all']} search={n['search']} sample={n['sample']} "
                f"terms={len(plan.get('keywords') or [])}"
                + (f" (+{added} added)" if added else "")
                + (f" promotions={proms}" if proms else ""))
    return " | ".join(s.get("goal", "")[:34] for s in (plan or []) if isinstance(s, dict))


def load_trace(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
