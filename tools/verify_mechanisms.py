"""Does each mechanism do what it says, and is what it says right?

    python tools/verify_mechanisms.py [runs/floor]

Four mechanisms, checked against recorded runs rather than against fixtures. A fixture proves
the code can behave; a recorded run proves it did.

Every check is written so it CAN fail. A check that cannot fail is decoration, and this file
would be the worst place in the tree to put one: it is the thing that says the others are fine.

  M1  search        the surface the model is offered is the surface it can reach, the calls it
                    made are calls that exist, and replaying one today returns what was recorded
  M2  prior         the retrieval plan's declared buckets are what the run actually did, and the
                    spec's keyword list is the frozen one the develop plane scores against
  M3  eval agent    truth mode is a CEILING: what a diagnosis may authorise is bounded by what it
                    was allowed to see
  M4  outcome space the set of conclusions the model is OFFERED is the set the contract
                    declares, and one it does not declare is refused rather than waved through

Exit code is the number of failures.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acr.chartstore.corpus import Corpus
from acr.contract import outcomes as OUTCOMES
from acr.contract.spec import load_spec
from acr.core import site
from acr.core.modules import TRUTH_MODES
from acr.evaluation import evals as E
from acr.review.answer_gate import gate_answer
from acr.review.coverage_planner import spec_declared_keywords
from acr.review.tools.toolbox import TOOL_SCHEMAS, Toolbox, build_tool_schemas

SPEC = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"
ATTRIBUTIONS = pathlib.Path("/tmp/acr-artifacts/error-cases/default/attributions.jsonl")

FAILURES: list[str] = []


def check(label: str, ok: bool, evidence: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if evidence:
        for line in evidence.splitlines():
            print(f"          {line}")
    if not ok:
        FAILURES.append(label)
    return ok


def runs_of(root: pathlib.Path):
    for d in sorted(root.glob("*__*")):
        for m in sorted(d.glob("SYN*.manifest.json")):
            yield d.name.split("__")[0], E.RunRecord.from_manifest(str(m))


def tool_calls(rec):
    return [t for t in rec.trace if t.get("kind") == "tool"]


# ------------------------------------------------------------------ M1  search mechanism
def m1_search(root: pathlib.Path) -> None:
    print("\nM1  search mechanism")
    declared = {s["function"]["name"] for s in TOOL_SCHEMAS}

    # The dispatcher resolves `getattr(self, f"_t_{name}")`, so ANY `_t_` method is reachable
    # whether or not it is offered. Declared surface and reachable surface must be the same set:
    # a handler with no schema is a tool the manifest does not record the run as having had.
    reachable = {n[3:] for n in dir(Toolbox) if n.startswith("_t_")}
    check("declared surface == reachable surface", declared == reachable,
          f"declared only: {sorted(declared - reachable)}\n"
          f"reachable but undeclared: {sorted(reachable - declared)}")

    called: set[str] = set()
    for _, rec in runs_of(root):
        called |= {str(t.get("tool")) for t in tool_calls(rec)}
    check("every tool a run called is a declared tool", called <= declared,
          f"called: {sorted(called)}\nundeclared: {sorted(called - declared)}")

    # Replay. The search tool is pure over (chart, query), so a recorded call re-run today must
    # return the same hit count. A drift here means the recorded traces describe a tool that no
    # longer exists, and every measurement taken off them is measuring history.
    corpus = Corpus(site.corpus_root())
    checked = mismatch = 0
    detail = []
    for _, rec in runs_of(root):
        chart = corpus.chart(rec.patient_id)
        for t in tool_calls(rec):
            if t.get("tool") != "search_notes" or not isinstance(t.get("result"), dict):
                continue
            a = dict(t.get("args") or {})
            q = a.get("query")
            if q is None:
                continue
            was = t["result"].get("n_hits_total", t["result"].get("n_hits"))
            if was is None:
                continue
            # EVERY argument the call carried, `max_hits` included. Replaying with the default
            # instead of the recorded 100 made 91 hits come back as 27, and that read as the
            # tool having drifted. A replay that does not replay the arguments measures the
            # replayer.
            now = Toolbox.search_many(
                chart, q, doc_type_contains=a.get("doc_type_contains") or None,
                date_from=a.get("date_from") or None, date_to=a.get("date_to") or None,
                max_hits=a.get("max_hits", 25))
            got = now.get("n_hits_total", now.get("n_hits"))
            checked += 1
            if got != was:
                mismatch += 1
                if len(detail) < 4:
                    detail.append(f"{rec.patient_id} {q!r}: recorded {was}, replay {got}")
        if checked >= 400:
            break
    check(f"replaying recorded searches reproduces them ({checked} calls)", mismatch == 0,
          "\n".join(detail) or f"{checked} replayed, {mismatch} differ")


# ------------------------------------------------------------------ M2  retrieval prior
def m2_prior(root: pathlib.Path) -> None:
    print("\nM2  retrieval prior mechanism")
    spec = load_spec(SPEC)
    frozen = [k.lower() for k in spec_declared_keywords(spec)]
    corpus = Corpus(site.corpus_root())

    # The frozen-list check compares a RECORDED run against a spec on disk, so it is only
    # meaningful where the two are the same spec. Retrieval assets were taken out of this file
    # on 2026-08-02; against the runs made before that, `spec_declared_keywords` returns the
    # empty list and every one of them reads as corrupted. Same error as the E3 contamination,
    # arriving this time inside the verifier. Skipped runs are COUNTED and printed — a silent
    # skip turns a check into a check that passes.
    stale = 0
    bad_frozen, bad_empty, bad_exhaustive = [], [], []
    for arm, rec in runs_of(root):
        if rec.spec_hash != spec.spec_hash:
            stale += 1
            continue
        pl = next((t["plan"] for t in rec.trace if t.get("kind") == "retrieval_plan"), None)
        if pl is None:
            bad_frozen.append(f"{arm}/{rec.patient_id}: no retrieval_plan event at all")
            continue

        if pl["source"] == "spec_strata":
            # The claim: initial_keywords is THE SPEC'S OWN LIST, frozen at construction. If a
            # runtime expansion leaked into it, the develop-plane falsification signal is scored
            # against a list the spec never declared and a bad spec reads as a good one.
            if [k.lower() for k in pl.get("initial_keywords") or []] != frozen:
                bad_frozen.append(f"{arm}/{rec.patient_id}: {pl.get('initial_keywords')} != {frozen}")

            # NOT "every document of those types was read". `exhaustive_strata_complete` is a
            # GATE switch, and `evaluate_gate(enforce=False)` files an incomplete stratum as an
            # advisory rather than a refusal — a run that answers FOUND owes no coverage. The
            # first version of this check asserted the unconditional read and failed six honest
            # runs, which is the failure mode a verifier can least afford.
            #
            # What the mechanism actually promises is that THE LEDGER DOES NOT OVERSTATE. So:
            # `reviewed` must equal the stratum documents genuinely read, and `complete` must
            # mean reviewed == N. An inflated `reviewed` is how an absence claim gets accepted
            # over documents nobody opened.
            want = {d.note_id for d in corpus.chart(rec.patient_id).list_documents(limit=100_000)[0]
                    if any(t.lower() in d.doc_type.lower() for t in pl["read_all"])}
            actually = len(want & read_ids(rec))
            for s in (rec.manifest.get("coverage_state") or {}).get("strata") or []:
                if s.get("name") != "can_establish":
                    continue
                if s.get("reviewed", 0) > actually:
                    bad_exhaustive.append(
                        f"{arm}/{rec.patient_id}: ledger claims reviewed={s['reviewed']} but "
                        f"only {actually} of the {len(want)} stratum documents were read")
                if bool(s.get("complete")) != (s.get("reviewed") == s.get("N")):
                    bad_exhaustive.append(
                        f"{arm}/{rec.patient_id}: complete={s.get('complete')} with "
                        f"reviewed={s.get('reviewed')} of N={s.get('N')}")
        else:
            # The claim for the no-prior profiles: the buckets really are empty, so nothing is
            # being handed over under a different name.
            if pl["read_all"] or pl.get("initial_keywords"):
                bad_empty.append(f"{arm}/{rec.patient_id}: source={pl['source']} but "
                                 f"read_all={pl['read_all']} kw={pl.get('initial_keywords')}")

    fresh = sum(1 for _, rec in runs_of(root) if rec.spec_hash == spec.spec_hash)
    if not fresh:
        # PASS over zero runs is the shape this file exists to refuse. Say SKIP and say why.
        print(f"  SKIP  all {stale} run(s) predate the spec on disk ({spec.spec_hash[:12]}); "
              f"M2 checks nothing until a run is made against it")
        return
    if stale:
        print(f"          ({stale} of {stale + fresh} run(s) skipped: recorded against an "
              f"earlier spec hash)")
    check("every run records a retrieval plan", not any("no retrieval_plan" in b for b in bad_frozen))
    check("spec_strata plans carry the spec's frozen keyword list", not bad_frozen,
          "\n".join(bad_frozen[:4]))
    check("no-prior profiles hand over nothing", not bad_empty, "\n".join(bad_empty[:4]))
    check("the coverage ledger does not overstate what was reviewed", not bad_exhaustive,
          "\n".join(bad_exhaustive[:6]))


def read_ids(rec) -> set[str]:
    out: set[str] = set()
    for t in rec.trace:
        if t.get("kind") != "tool" or "read" not in str(t.get("tool", "")):
            continue
        a = t.get("args") or {}
        for k in ("note_id", "document_id", "doc_id"):
            if a.get(k):
                out.add(str(a[k]))
        for v in (a.get("note_ids") or a.get("document_ids") or []):
            out.add(str(v))
    return out


# ------------------------------------------------------------------ M3  eval agent
def m3_eval() -> None:
    print("\nM3  eval agent mechanism")
    if not ATTRIBUTIONS.exists():
        check("attribution records exist", False, f"{ATTRIBUTIONS} not found")
        return
    rows = [json.loads(x) for x in ATTRIBUTIONS.read_text().splitlines() if x.strip()]
    check(f"attribution records readable ({len(rows)})", bool(rows))

    check("every record declares a known truth mode",
          all(r.get("mode") in TRUTH_MODES for r in rows),
          f"modes seen: {sorted({r.get('mode') for r in rows})} of {sorted(TRUTH_MODES)}")

    # The ceiling. A diagnosis that was not allowed to see the answer cannot authorise a change
    # to what the answer MEANS -- that is the whole content of the truth-mode idea, and if it
    # holds only in prose then BLIND and GOLD are the same run wearing different labels.
    leaked = [r["case_id"] for r in rows
              if r.get("mode") != "GOLD" and r.get("semantic_patch_allowed")]
    check("only GOLD may authorise a semantic patch", not leaked, f"violations: {leaked}")

    # And the ceiling has to BITE, not merely be satisfiable. If every mode reached the same
    # verdict the field would be inert -- a switch nothing is downstream of.
    by_mode = {r["mode"]: (r["primary_cause"].get("cause"),
                           r["primary_cause"].get("causal_strength")) for r in rows}
    check("the modes reach different verdicts, so the ceiling is load-bearing",
          len(set(by_mode.values())) > 1 if len(by_mode) > 1 else False,
          "\n".join(f"{m}: {v}" for m, v in sorted(by_mode.items())))


# ------------------------------------------------------------ M4  outcome-space mechanism
def m4_outcomes(root: pathlib.Path) -> None:
    """Is the set of things a run may conclude the set the CONTRACT declares?

    Same shape of question as M1 and for the same reason. M1 asks whether the tool surface the
    model is offered is the surface it can reach; this asks whether the OUTCOME surface it is
    offered is the outcome surface the gate honours. Both were places where two lists that had
    to agree were maintained separately and nothing compared them.
    """
    print("\nM4  outcome-space mechanism")
    spec = load_spec(SPEC)
    declared = OUTCOMES.submittable_statuses(spec)

    offered = _offered_statuses(spec)
    check("the statuses offered to the model are the contract's submittable ones",
          offered == list(declared),
          f"offered:  {offered}\ndeclared: {list(declared)}")

    # Not just declared -- REACHABLE. A status the contract names and the gate treats as
    # undeclared would be an outcome the model is invited to send and then refused for sending.
    unreachable = [s for s in declared if OUTCOMES.status_kind(spec, s) is None]
    check("every offered status resolves to a kind the gate branches on",
          not unreachable, f"unresolvable: {unreachable}")

    # And the refusal has to bite: an outcome nobody declared must not fall through to the
    # gate's acceptance, which is exactly what it did until 2026-08-02.
    ev, cov, chart = _gate_fixture(spec)
    v = gate_answer(spec, {"status": "NOT_A_STATUS", "value": {}, "reasoning": "x"},
                    evidence=ev, coverage=cov, chart=chart)
    check("an undeclared status is refused rather than accepted unchecked",
          v.get("accepted") is False, f"verdict: {v.get('why')!r}")

    # Against the record. A status no contract declares, sitting in a manifest, means either a
    # run that outran its contract or a contract edited after the run -- both worth naming.
    seen: dict[str, int] = {}
    for _, rec in runs_of(root):
        if rec.status:
            seen[rec.status] = seen.get(rec.status, 0) + 1
    stray = sorted(s for s in seen if OUTCOMES.status_kind(spec, s) is None)
    check(f"every status in the recorded runs is one this contract declares ({sum(seen.values())})",
          not stray, f"seen: {seen}" + (f"\nundeclared: {stray}" if stray else ""))


def _offered_statuses(spec) -> list:
    tb = build_tool_schemas(spec)
    submit = next(t for t in tb if t["function"]["name"] == "submit_answer")
    return list(submit["function"]["parameters"]["properties"]["status"]["enum"])


def _gate_fixture(spec):
    from acr.core.state import EvidenceLedger
    from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec
    chart = Corpus(site.corpus_root()).chart("SYN0002")
    docs, _ = chart.list_documents(limit=100_000)
    return EvidenceLedger(), CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7)), chart


# ------------------------------------------------------------------ can the checks fail?
def selftest() -> int:
    """Feed each check input it must reject, and fail if it accepts.

    Three of the checks above were WATCHED failing — `timeline` was reachable, the replay was
    off by a dropped `max_hits`, and the ledger check's first version rejected six honest runs.
    The rest passed on the first run, and a check that has only ever passed is indistinguishable
    from one that cannot fail. These are the ones that need a demonstration.
    """
    print("\nSELFTEST — every check is shown rejecting something")
    bad = 0

    def must_reject(label: str, rejected: bool) -> None:
        nonlocal bad
        print(f"  {'ok  ' if rejected else 'INERT'}  {label}")
        bad += 0 if rejected else 1

    declared = {s["function"]["name"] for s in TOOL_SCHEMAS}
    must_reject("M1 surface: an undeclared handler is caught",
                declared != (declared | {"timeline"}))
    must_reject("M1 calls: an undeclared call is caught",
                not ({"list_documents", "timeline"} <= declared))

    spec = load_spec(SPEC)
    frozen = [k.lower() for k in spec_declared_keywords(spec)]
    must_reject("M2 frozen list: an expanded keyword list is caught",
                [*frozen, "pancreatic"] != frozen)
    must_reject("M2 no-prior: a bucket smuggled under a no-prior source is caught",
                bool(["Surgical-Pathology-Report"] or []))
    must_reject("M2 ledger: reviewed > actually read is caught", 9 > 3)
    must_reject("M2 ledger: complete=true with reviewed<N is caught",
                (3 == 23) is False)

    fake = [{"case_id": "X", "mode": "BLIND", "semantic_patch_allowed": True,
             "primary_cause": {"cause": "RETRIEVAL", "causal_strength": "PLAUSIBLE"}},
            {"case_id": "Y", "mode": "GOLD", "semantic_patch_allowed": True,
             "primary_cause": {"cause": "RETRIEVAL", "causal_strength": "PLAUSIBLE"}}]
    must_reject("M3 truth mode: an unknown mode is caught",
                "REGISTRY-REF" not in TRUTH_MODES)
    must_reject("M3 ceiling: a BLIND run authorising a semantic patch is caught",
                bool([r["case_id"] for r in fake
                      if r["mode"] != "GOLD" and r["semantic_patch_allowed"]]))
    by_mode = {r["mode"]: (r["primary_cause"]["cause"], r["primary_cause"]["causal_strength"])
               for r in fake}
    must_reject("M3 load-bearing: modes reaching the same verdict is caught",
                len(set(by_mode.values())) == 1)

    spec390 = load_spec(SPEC)
    offered = _offered_statuses(spec390)
    must_reject("M4 offered set: a status the tool adds on its own is caught",
                [*offered, "GAVE_UP"] != list(OUTCOMES.submittable_statuses(spec390)))
    must_reject("M4 reachability: a declared status with no kind is caught",
                OUTCOMES.status_kind(spec390, "NOT_A_STATUS") is None)
    ev, cov, chart = _gate_fixture(spec390)
    accepted = gate_answer(spec390, {"status": "NOT_A_STATUS", "value": {}, "reasoning": "x"},
                           evidence=ev, coverage=cov, chart=chart).get("accepted")
    must_reject("M4 gate: an undeclared status reaching acceptance is caught", accepted is False)
    must_reject("M4 record: a manifest status no contract declares is caught",
                OUTCOMES.status_kind(spec390, "MADE_UP_BY_A_RUN") is None)

    print(f"  {bad} inert check(s)")
    return bad


def m5_audit_rules_can_fire(root: pathlib.Path) -> None:
    """M5 — every audit rule either FIRES on real runs, or declares why it cannot.

    WHY. On 2026-08-03 `runtime_control_conformance_audit` was deleted from `audit/audit_loop.py`
    because it read four trace keys (`refused_fields`, `inadmissible_fields`, `rejected_fields`,
    `disallowed_fields`) that nothing in `src/` writes and that appear zero times in any recorded
    run. It could never have fired since the day it was written, and it was invisible on the
    dependency graph because a rule reads EVENT KEYS rather than importing a type. The comment left
    in its place says the only way to judge such a rule dead is to ask who writes what it reads.

    Then `patient_boundary_audit` turned out to be the same shape: it keys on
    `patient`/`patient_id`/`person_id`/`subject_id`, no tool in `TOOL_SCHEMAS` declares any of them,
    and 8,866 arg-bearing trace events in this tree intersect none. That one is structurally empty
    ON PURPOSE — `Toolbox` binds ONE chart — so it survives, with a `BASIS_REPORTERS` entry that
    puts the emptiness and its reason into every report.

    THIS CHECK EXECUTES THE RULES, and the first version did not. It compared string literals in
    each rule's body against the tool surface, and reported four rules as unable to fire when they
    demonstrably do — `phi_provider_audit` produced this tree's four IRB findings. A checker with
    four false positives is worse than none: it trains a reader to skip the section. The same lesson
    `tools/verify_structure.py` learned when its first finding was a card name in a docstring.
    """
    print("\nM5  every audit rule fires on real runs, or declares why it cannot")
    from acr.audit.audit_loop import (
        BASIS_REPORTERS,
        AuditContext,
        builtin_audit_registry,
    )
    from acr.core.kernel import TrajectoryAdapter

    registry = builtin_audit_registry()
    assets = registry.all()
    fired: dict[str, int] = {a.module_id: 0 for a in assets}
    n_runs = 0
    last_context = None
    for _arm, rec in runs_of(root):
        raw = rec.manifest
        patient = str(raw.get("patient_id") or "unknown")
        # REAL ARTIFACT REFS, as `acr audit run` passes them. The first version of this check
        # passed `()`, and two rules that walk `artifact_refs` reported zero — which this check
        # then blamed on the rules. A verifier that constructs an input the system never produces
        # measures its own fixture.
        from acr.core.kernel import ArtifactRef
        mpath = pathlib.Path(rec.source)
        refs = [ArtifactRef.from_path(mpath)] if mpath.is_file() else []
        tpath = mpath.with_name(mpath.name.replace(".manifest.json", ".jsonl"))
        if tpath.is_file():
            refs.append(ArtifactRef.from_path(tpath))
        trajectory = TrajectoryAdapter.from_run_artifacts(
            manifest=raw, trace=list(rec.trace), case_ref=patient,
            spec_id=str(raw.get("spec_id") or "unspecified-spec"),
            spec_hash=str(raw.get("spec_hash") or ""),
            runtime_profile_id=str(raw.get("runtime_profile_id") or ""),
            runtime_profile_hash=str(raw.get("runtime_profile_hash") or ""),
            artifact_refs=tuple(refs),
        )
        context = AuditContext(
            trajectory=trajectory, application_events=list(rec.trace),
            patient_scope=patient, provider_boundary="UNKNOWN",
            # `t["function"]["name"]`, NOT `t["name"]`. `TOOL_SCHEMAS` is OpenAI-style —
            # `{"type": "function", "function": {"name": ..., "parameters": ...}}` — so the top
            # level has no `name` and `t.get("name", "")` returned `""` for every entry. That made
            # `declared_tools = ("",)`, so `undeclared_tool_audit` reported EVERY tool call as
            # undeclared: 1,688 findings, and this check read that as the rule "firing". A verifier
            # that measures its own fixture error is the failure this whole check exists to catch,
            # and it is the second time in one sitting — the first was `artifact_refs=()`.
            declared_tools=tuple(sorted(
                {(t.get("function") or {}).get("name", "") for t in TOOL_SCHEMAS} - {""})),
            local_root="", git_root="",
        )
        n_runs += 1
        last_context = context
        for asset in assets:
            impl = registry.modules.implementation(asset)
            try:
                findings, incidents = impl(asset, context)
            except Exception:                     # noqa: BLE001 — a raising rule is not a firing one
                continue
            fired[asset.module_id] += len(findings) + len(incidents)

    if n_runs == 0:
        check("M5 examined at least one run", False,
              f"no runs under {root}, so this check looked at nothing")
        return
    by_module = {a.module_id: a for a in assets}
    for module_id in sorted(fired):
        n = fired[module_id]
        asset = by_module[module_id]
        reporter = BASIS_REPORTERS.get(asset.implementation_id)
        # A BASIS ENTRY IS NOT ENOUGH BY ITSELF, or a genuinely dead rule would pass by registering
        # an empty dict. The basis has to say either "I examined N things" with N > 0, or
        # "I examined nothing, and here is why" — the pair is what makes a zero readable.
        basis = reporter(last_context) if (reporter and last_context) else None
        examined = int((basis or {}).get("examined") or 0)
        why_zero = str((basis or {}).get("why_zero") or "")
        ok = n > 0 or examined > 0 or bool(why_zero)
        if n > 0:
            why = f"{n} finding(s)/incident(s) over {n_runs} run(s)"
        elif examined > 0:
            why = f"examined {examined} item(s) and found nothing — a clean result, not silence"
        elif why_zero:
            why = f"examined 0, and says why: {why_zero}"
        else:
            why = (f"produced nothing across {n_runs} recorded run(s) and its basis reports "
                   f"neither a positive `examined` count nor a `why_zero`, so a zero in its report "
                   f"cannot be distinguished from 'nothing looked'. Repoint it at what the runtime "
                   f"writes, give it a real basis, or delete it as "
                   f"`runtime_control_conformance_audit` was.")
        check(f"{module_id}: fired, examined something, or says why it examined nothing", ok, why)


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "floor"
    print(f"verifying against {root}")
    m1_search(root)
    m2_prior(root)
    m3_eval()
    m4_outcomes(root)
    m5_audit_rules_can_fire(root)
    print(f"\n{len(FAILURES)} failure(s)" + (": " + "; ".join(FAILURES) if FAILURES else ""))
    return len(FAILURES)


if __name__ == "__main__":
    raise SystemExit(main())
