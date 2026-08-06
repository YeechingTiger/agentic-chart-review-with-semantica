"""Model-based, local-only attribution of completed chart-review runs."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Sequence
from pathlib import Path

import typer
from rich.table import Table

from ..contract.skills import eval_skills_identity
from ..core import cli_common, site
from ..core.cli_common import API_BASE, CORPUS, MODEL, con

#: A RUN RECORD is read with `require_run_artifact`, not through the store: `runs/` is inside
#: the worktree by design and the store's `require_input` proves the opposite, so every one of
#: these call sites was unreachable. Develop artifacts — gold, answer keys, case maps — keep
#: going through the store, where the outside-the-worktree rule is the right rule.
from ..core.local_artifacts import (
    LOCAL_ROOT_ENV,
    RUN_RECORD_GLOB,
    LocalArtifactError,
    LocalArtifactStore,
    require_run_artifact,
    require_run_tree,
)
from ..diagnosis import attribution as A
from ..diagnosis import meta_evaluation as ME
from ..evaluation import evals

attribute_app = typer.Typer(add_completion=False, help=(
    "Offline root-cause attribution over another run's trace. Model-based, read-only, "
    "same-patient scoped, and LOCAL-ONLY; it never replaces extraction or edits a spec."))

LOCAL_ROOT = typer.Option(
    None, "--local-root", envvar=LOCAL_ROOT_ENV,
    help="absolute patient-artifact root outside Git")


def _store(root: str | None) -> LocalArtifactStore:
    try:
        return LocalArtifactStore(root)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _case_map(store: LocalArtifactStore, path: str) -> dict[str, str]:
    if not path:
        return {}
    try:
        p = store.require_input(path, what="case map")
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (LocalArtifactError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter("--case-map must be {pseudonymous_case_id: patient_id}")
    out = {}
    for case_id, patient_id in raw.items():
        try:
            safe = A.safe_case_id(case_id)
        except Exception as exc:
            raise typer.BadParameter(str(exc)) from exc
        if not str(patient_id).strip():
            raise typer.BadParameter(f"{safe}: patient id is empty")
        out[safe] = str(patient_id)
    return out


def _band(raw: str, name: str) -> tuple[int, int]:
    try:
        lo, hi = (int(x) for x in raw.split(",", 1))
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be lo,hi") from exc
    if lo > hi:
        raise typer.BadParameter(f"{name}: lo cannot exceed hi")
    return lo, hi


def _json_or_jsonl(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    value = json.loads(raw)
    if isinstance(value, list):
        return value
    for key in ("attributions", "adjudications", "cases"):
        if isinstance(value.get(key), list):
            return value[key]
    raise typer.BadParameter(f"{path}: expected JSON list or JSONL")


def _detectors(path: Path, *, min_term_chars: int, max_rejection_repeats: int,
               token_band: str, turn_band: str) -> list[dict]:
    record = evals.RunRecord.from_manifest(path)
    cfg = evals.DetectorConfig(
        min_term_chars=min_term_chars,
        max_rejection_repeats=max_rejection_repeats,
        token_band=_band(token_band, "--token-band"),
        turn_band=_band(turn_band, "--turn-band"),
    )
    return [f.to_dict() for f in evals.run_detectors(record, config=cfg)]


def _mode_inputs(store: LocalArtifactStore, mode: str, case_id: str, *,
                 gold: str, registry_reference: str):
    from ..contract.behaviour import load_gold
    if mode not in A.ATTRIBUTION_MODES:
        raise typer.BadParameter(f"--mode must be one of {A.ATTRIBUTION_MODES}")
    if mode == A.GOLD:
        if not gold or registry_reference:
            raise typer.BadParameter(
                "GOLD requires --gold and forbids --registry-reference")
        try:
            rows = load_gold(store.require_input(gold, what="chart-observable gold"))
        except (LocalArtifactError, OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if case_id not in rows:
            raise typer.BadParameter(f"{case_id}: absent from --gold")
        return rows[case_id], None
    if mode == A.REGISTRY_REFERENCE:
        if not registry_reference or gold:
            raise typer.BadParameter(
                "REGISTRY_REFERENCE requires --registry-reference and forbids --gold")
        try:
            rows = A.load_registry_references(
                store.require_input(registry_reference, what="registry reference"))
        except (LocalArtifactError, A.AttributionError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if case_id not in rows:
            raise typer.BadParameter(f"{case_id}: absent from --registry-reference")
        return None, rows[case_id]
    if gold or registry_reference:
        raise typer.BadParameter("BLIND forbids --gold and --registry-reference")
    return None, None


def _packet(*, store, manifest, case_id, spec_path, mode, gold_path,
            registry_path, detector_args):
    from ..contract.spec import load_spec
    try:
        mpath = require_run_artifact(manifest, what="run manifest")
        sp = load_spec(spec_path)
        chart_gold, registry = _mode_inputs(
            store, mode, case_id, gold=gold_path,
            registry_reference=registry_path)
        findings = _detectors(mpath, **detector_args)
        packet = A.build_packet(
            manifest_path=mpath, case_id=case_id, spec=sp, mode=mode,
            detector_findings=findings, chart_gold=chart_gold,
            registry_reference=registry,
        )
    except (LocalArtifactError, A.AttributionError, OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return packet


def _run_one(*, packet, patient_id, corpus, model, api_base, max_model_calls,
             max_usd, max_chart_reads, library, attribution_profile,
             eval_skills_prompt: str = ""):
    from ..chartstore.corpus import Corpus
    chart = Corpus(Path(corpus)).chart(patient_id)
    report = A.run_attribution_agent(
        packet=packet, chart=chart,
        # The deployed reasoning model accepts only its default temperature. Attribution is
        # still reproducible at the packet/gate level; model prose is never a confirmation.
        model=cli_common.chat_model(model, api_base, 1.0),
        skeptic_model=cli_common.chat_model(model, api_base, 1.0),
        max_model_calls=max_model_calls, max_usd=max_usd,
        max_chart_reads=max_chart_reads,
        attribution_profile=attribution_profile,
        # Method, not authority: the eval skills say how a careful reviewer looks for a cause.
        # Empty for `acr attribute`, which had no skills before this and must not acquire any
        # by accident — the default this command renders to the model has to stay unchanged.
        eval_skills_prompt=eval_skills_prompt,
    )
    library.add_attribution(
        report, manifest_sha256=packet.manifest_ref.sha256)
    return report


def attribute_case_payload(*, run: str, spec: str, case_id: str,
                           min_term_chars: int, max_rejection_repeats: int,
                           token_band: str, turn_band: str,
                           gold: str = "", eval_skills_prompt: str = "",
                           #: The card NAMES behind `eval_skills_prompt`. Needed because
                           #: the envelope has to say WHICH method produced a diagnosis,
                           #: and a rendered block cannot be read back into its card list.
                           eval_skills_names: Sequence[str] = (),
                           signal_type: str = "ATTRIBUTION_REPORT",
                           mode: str = "", registry_reference: str = "", case_map: str = "",
                           corpus: str = str(site.corpus_root()), model: str | None = None,
                           api_base: str | None = None, max_model_calls: int = 12,
                           max_usd: float = 1.0, max_chart_reads: int = 12,
                           library_id: str = "default",
                           attribution_profile: str = "causal-attribution-v1",
                           local_root: str | None = None) -> dict:
    """Attribute one run and return the report as a signal-shaped dict.

    Split out of `attribute_case` so `acr signal run --kind agent` reaches the same code path
    rather than a parallel one. A second path to a diagnosis is a second thing to keep honest.

    The four detector thresholds have NO defaults here, for the reason `evals.DetectorConfig`
    gives for having none itself: a threshold nobody typed is folklore. `acr attribute case`
    requires them on the command line; `acr signal run` declares its own in `cli_signal`.
    """
    store = _store(local_root)
    mapping = _case_map(store, case_map)
    # An empty mode means "derive it from what was supplied". The dispatcher has no --mode
    # switch — it either hands over an answer key or it does not — and the two mistakes that
    # would follow from guessing wrong are both refused downstream by `_mode_inputs`.
    resolved_mode = mode or (A.GOLD if gold else A.BLIND)
    packet = _packet(
        store=store, manifest=run, case_id=case_id, spec_path=spec, mode=resolved_mode,
        gold_path=gold, registry_path=registry_reference,
        detector_args={
            "min_term_chars": min_term_chars,
            "max_rejection_repeats": max_rejection_repeats,
            "token_band": token_band, "turn_band": turn_band,
        })
    raw_manifest = json.loads(Path(packet.manifest_ref.path).read_text(encoding="utf-8"))
    patient_id = mapping.get(case_id) or str(raw_manifest.get("patient_id") or "")
    if not patient_id:
        raise typer.BadParameter(
            f"{case_id}: cannot resolve patient; provide --case-map")
    library = ME.ErrorCaseLibrary(store, library_id)
    reasons = A.selection_reasons(packet) or ("manual_single_case_request",)
    library.add_case(ME.ErrorCaseEvent(
        case_id=case_id, event="SELECTED", lifecycle="OPEN",
        run_ref=packet.manifest_ref.to_dict(), reasons=reasons,
        detail={"mode": resolved_mode, "semantic_patch_allowed": resolved_mode == A.GOLD},
    ))
    report = _run_one(
        packet=packet, patient_id=patient_id, corpus=corpus, model=model,
        api_base=api_base, max_model_calls=max_model_calls, max_usd=max_usd,
        max_chart_reads=max_chart_reads, library=library,
        attribution_profile=attribution_profile,
        eval_skills_prompt=eval_skills_prompt,
    )
    return {"schema": "acr.signal/1", "signal_type": signal_type, "kind": "agent",
            "run": run, "spec": spec, "deterministic": False,
            # THE IDENTITY, not the length. `eval_skills_bytes` was a byte count: two
            # different card sets of equal length were indistinguishable, and a card edited
            # in place usually keeps its length. `attribute meta-certify` scores these
            # diagnoses against human adjudications, and "which method produced this causal
            # judgement" is the first thing that comparison needs.
            "eval_skills": eval_skills_identity(eval_skills_prompt, eval_skills_names),
            "report": report.to_dict(),
            # The library path the command prints. Returned rather than printed because a
            # dispatcher writing to stdout in the middle of building a JSON envelope is how
            # machine-readable output stops being machine-readable.
            "library_path": str(library.directory / "attributions.jsonl")}


@attribute_app.command("case")
def attribute_case(
    run: str = typer.Option(..., "--run", help="local *.manifest.json"),
    case_id: str = typer.Option(..., "--case-id", help="pseudonymous local case ID"),
    spec: str = typer.Option(..., "--spec", "-s"),
    mode: str = typer.Option(A.BLIND, "--mode"),
    gold: str = typer.Option("", "--gold"),
    registry_reference: str = typer.Option("", "--registry-reference"),
    case_map: str = typer.Option("", "--case-map"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    # 24, NOT 12. Measured 2026-08-06 attributing a real abstention (SYN0002, the corpus's
    # "biopsy performed at outside hospital" case): at 12 the agent spent 11 calls and
    # returned UNRESOLVED with "model-call limit reached without a gate-valid attribution";
    # at 30 it converged in 17 on the right cause, EVIDENCE_GAP, and the independent skeptic
    # passed it. A default that cannot reach a verdict reports the budget, not the run.
    max_model_calls: int = typer.Option(24, "--max-model-calls", min=1),
    max_usd: float = typer.Option(1.0, "--max-usd", min=0.01),
    max_chart_reads: int = typer.Option(12, "--max-chart-reads", min=0),
    min_term_chars: int = typer.Option(..., "--min-term-chars", min=1),
    max_rejection_repeats: int = typer.Option(..., "--max-rejection-repeats", min=2),
    token_band: str = typer.Option(..., "--token-band", help="lo,hi"),
    turn_band: str = typer.Option(..., "--turn-band", help="lo,hi"),
    library_id: str = typer.Option("default", "--library-id"),
    attribution_profile: str = typer.Option(
        "causal-attribution-v1", "--attribution-profile"),
    local_root: str | None = LOCAL_ROOT,
):
    """Attribute one completed run; all artifacts remain below the local root."""
    payload = attribute_case_payload(
        run=run, spec=spec, gold=gold, case_id=case_id, mode=mode,
        registry_reference=registry_reference, case_map=case_map,
        min_term_chars=min_term_chars, max_rejection_repeats=max_rejection_repeats,
        token_band=token_band, turn_band=turn_band,
        corpus=corpus, model=model, api_base=api_base,
        max_model_calls=max_model_calls, max_usd=max_usd, max_chart_reads=max_chart_reads,
        library_id=library_id, attribution_profile=attribution_profile,
        local_root=local_root,
    )
    # The command prints the attribution report, not the signal envelope. `acr signal run` is
    # where the envelope is the output; changing what this command prints would break every
    # reader of an existing attributions.jsonl workflow for no gain.
    con.print_json(json.dumps(payload["report"], ensure_ascii=False))
    con.print(f"→ {payload['library_path']}")


@attribute_app.command("batch")
def attribute_batch(
    runs: str = typer.Option(..., "--runs", help="local manifest file or directory"),
    spec: str = typer.Option(..., "--spec", "-s"),
    case_map: str = typer.Option(..., "--case-map"),
    mode: str = typer.Option(A.BLIND, "--mode"),
    gold: str = typer.Option("", "--gold"),
    registry_reference: str = typer.Option("", "--registry-reference"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    # 24, NOT 12. Measured 2026-08-06 attributing a real abstention (SYN0002, the corpus's
    # "biopsy performed at outside hospital" case): at 12 the agent spent 11 calls and
    # returned UNRESOLVED with "model-call limit reached without a gate-valid attribution";
    # at 30 it converged in 17 on the right cause, EVIDENCE_GAP, and the independent skeptic
    # passed it. A default that cannot reach a verdict reports the budget, not the run.
    max_model_calls: int = typer.Option(24, "--max-model-calls", min=1),
    max_usd: float = typer.Option(1.0, "--max-usd", min=0.01),
    max_chart_reads: int = typer.Option(12, "--max-chart-reads", min=0),
    min_term_chars: int = typer.Option(..., "--min-term-chars", min=1),
    max_rejection_repeats: int = typer.Option(..., "--max-rejection-repeats", min=2),
    token_band: str = typer.Option(..., "--token-band"),
    turn_band: str = typer.Option(..., "--turn-band"),
    library_id: str = typer.Option("default", "--library-id"),
    attribution_profile: str = typer.Option(
        "causal-attribution-v1", "--attribution-profile"),
    rerun: bool = typer.Option(
        False, "--rerun",
        help="spend again even when this library already contains case+manifest hash"),
    local_root: str | None = LOCAL_ROOT,
):
    """Screen every run for free, then invoke the agent only on selected abnormal cases."""
    store = _store(local_root)
    mapping = _case_map(store, case_map)
    try:
        # `require_run_tree`, not `store.path`: the store proves a path is under a root that must be
        # OUTSIDE the worktree, and `runs/` is inside it by design, so this command could not be
        # aimed at any run this project has produced. That is why `attribute` had never emitted a
        # proposal. See `require_run_tree` for the check that replaces it.
        root = require_run_tree(runs, what="runs")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    manifests = ([root] if root.is_file()
                 else sorted(root.rglob(RUN_RECORD_GLOB)))
    patient_to_case = {patient: case for case, patient in mapping.items()}
    packets = []
    for manifest in manifests:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        patient = str(raw.get("patient_id") or "")
        case = patient_to_case.get(patient)
        if not case:
            raise typer.BadParameter(
                f"{manifest}: patient is absent from the required local --case-map")
        packets.append(_packet(
            store=store, manifest=str(manifest), case_id=case, spec_path=spec, mode=mode,
            gold_path=gold, registry_path=registry_reference,
            detector_args={
                "min_term_chars": min_term_chars,
                "max_rejection_repeats": max_rejection_repeats,
                "token_band": token_band, "turn_band": turn_band,
            }))
    behavior = A.batch_behavior_conflicts(packets)
    library = ME.ErrorCaseLibrary(store, library_id)
    existing = {
        (str(row.get("case_id") or ""), str(row.get("manifest_sha256") or ""))
        for row in library.rows("attributions.jsonl")
    }
    selected = []
    for packet in packets:
        reasons = tuple(sorted(set(A.selection_reasons(packet) + behavior[packet.case_id])))
        if not reasons:
            continue
        selected.append((packet, reasons))
        library.add_case(ME.ErrorCaseEvent(
            case_id=packet.case_id, event="SELECTED", lifecycle="OPEN",
            run_ref=packet.manifest_ref.to_dict(), reasons=reasons,
            detail={"mode": mode, "semantic_patch_allowed": mode == A.GOLD},
        ))

    table = Table("case", "screen", "primary cause", "status", "calls", "reads")
    skipped = 0
    for packet, reasons in selected:
        if not rerun and (packet.case_id, packet.manifest_ref.sha256) in existing:
            skipped += 1
            table.add_row(packet.case_id, ", ".join(reasons)[:70],
                          "-", "ALREADY_ATTRIBUTED", "0", "0")
            continue
        patient = mapping[packet.case_id]
        report = _run_one(
            packet=packet, patient_id=patient, corpus=corpus, model=model,
            api_base=api_base, max_model_calls=max_model_calls, max_usd=max_usd,
            max_chart_reads=max_chart_reads, library=library,
            attribution_profile=attribution_profile,
        )
        table.add_row(
            packet.case_id, ", ".join(reasons)[:70],
            report.primary_cause.cause, report.primary_cause.status,
            str(report.model_calls), str(report.chart_reads))
    con.print(table)
    con.print(
        f"[bold]{len(selected)}/{len(packets)} run(s) selected[/]; "
        f"{skipped} already attributed; semantic patches emitted: 0\n→ {library.directory}")


@attribute_app.command("cluster")
def cluster(
    library_id: str = typer.Option("default", "--library-id"),
    local_root: str | None = LOCAL_ROOT,
):
    """Build deterministic clusters and append their current signatures to clusters.jsonl."""
    library = ME.ErrorCaseLibrary(_store(local_root), library_id)
    reports = [
        A.AttributionReport.from_dict(row)
        for row in library.rows("attributions.jsonl")
    ]
    clusters = ME.cluster_reports(reports)
    for row in clusters:
        library.add_cluster(row)
    con.print_json(json.dumps(
        {"n_clusters": len(clusters), "clusters": [x.to_dict() for x in clusters]},
        ensure_ascii=False))
    con.print(f"→ {library.directory / 'clusters.jsonl'}")


@attribute_app.command("summarize")
def summarize(
    library_id: str = typer.Option("default", "--library-id"),
    local_root: str | None = LOCAL_ROOT,
):
    """Print the folded local library; writes no non-JSONL summary artifact."""
    library = ME.ErrorCaseLibrary(_store(local_root), library_id)
    con.print_json(json.dumps(ME.summarize_library(library), ensure_ascii=False))


@attribute_app.command("adjudicate")
def adjudicate(
    case_id: str = typer.Option(..., "--case-id"),
    decision: str = typer.Option(..., "--decision"),
    actor: str = typer.Option(..., "--actor"),
    actor_role: str = typer.Option(..., "--actor-role"),
    rationale: str = typer.Option(..., "--rationale"),
    primary_cause: str = typer.Option(
        "", "--primary-cause",
        help="the human ROOT CAUSE, one of `attribution.CAUSES`. This is the label "
             "`meta-certify` scores the attributor against — without it the adjudication pairs "
             "with nothing and the calibration reports a shortage of cases forever."),
    library_id: str = typer.Option("default", "--library-id"),
    local_root: str | None = LOCAL_ROOT,
):
    """Append an accountable human/engineer decision; never rewrite earlier events.

    `--primary-cause` is what makes this stage part of a loop rather than a log. It is optional
    because `WONT_FIX` and `OUTSIDE_CHART` are decisions about what to DO, not root-cause labels;
    `meta-certify` now says so explicitly when rows exist and none carries one.
    """
    library = ME.ErrorCaseLibrary(_store(local_root), library_id)
    try:
        event = ME.AdjudicationEvent(
            case_id=case_id, decision=decision, actor=actor,
            actor_role=actor_role, rationale=rationale, primary_cause=primary_cause)
    except A.AttributionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    added = library.add_adjudication(event)
    con.print(
        f"{'appended' if added else 'already present'}: "
        f"{library.directory / 'adjudications.jsonl'}")
    if not primary_cause:
        con.print("[yellow]No --primary-cause, so this row pairs with nothing in `meta-certify`. "
                  f"Right for a decision about what to do; wrong if you meant to record a root "
                  f"cause. One of: {', '.join(A.CAUSES)}[/]")


@attribute_app.command("meta-certify")
def meta_certify(
    predictions: str = typer.Option(..., "--predictions"),
    adjudications: str = typer.Option(..., "--adjudications"),
    min_cases: int = typer.Option(30, "--min-cases", min=1),
    min_macro_f1: float = typer.Option(0.80, "--min-macro-f1", min=0, max=1),
):
    """Calibrate causal attribution against accountable human root-cause labels."""
    report = ME.meta_evaluate_attributions(
        _json_or_jsonl(predictions),
        _json_or_jsonl(adjudications),
        min_cases=min_cases,
        min_macro_f1=min_macro_f1,
    )
    con.print_json(json.dumps(report, ensure_ascii=False))


@attribute_app.command("case-map")
def attribute_case_map(
    runs: str = typer.Option("", "--runs", help="run record or directory; patients are read from it"),
    patients: str = typer.Option("", "--patients", help="comma list, instead of --runs"),
    out: str = typer.Option("case-map.json", "--out", help="path under the local root"),
    local_root: str | None = LOCAL_ROOT,
):
    """Mint the pseudonymous {case_id: patient_id} map that `attribute`, `eval` and `repair` require.

    THE PRODUCER THAT DID NOT SHIP. `--case-map` is a REQUIRED option on `attribute batch`,
    `attribute cluster` and half of `repair`, `_case_map` validates one in two modules, and nothing
    in this tree ever wrote one. Every consumer of a format and no producer of it is a stage that
    cannot run, and it stayed invisible for the same reason the term cache did: reaching it costs
    money, so no test goes there. `attribute` has produced zero proposals since it was written.

    WHY A MAP AT ALL, rather than using the patient id. Everything the develop plane keeps — cluster
    signatures, adjudications, the folded library — is meant to be readable by someone reasoning
    about failure MODES across a cohort, and `safe_case_id` refuses any id that
    `looks_like_a_person_id`. The map is the one file that holds both halves, and it is written
    through the store, so it lives outside Git by construction.

    WHY SALTED, and this is the part a plain hash gets wrong. An unsalted digest of an identifier is
    reversible by ENUMERATION whenever the identifier space is small or structured — and a medical
    record number is both. So the case id is an HMAC under a 256-bit key minted once per local root
    and never leaving it. Same patient, same root, same case id across batches, which is what lets
    two runs of the same chart fold into one cluster; and no relationship anyone can compute without
    the root.

    RE-RUNNING IS SAFE and additive: an existing map is read, kept, and extended. Re-minting a case
    id for a patient already in the library would split that patient's history into two cases and
    silently halve every per-case count.
    """
    store = _store(local_root)
    if bool(runs) == bool(patients):
        raise typer.BadParameter("supply exactly one of --runs or --patients")

    ids: list[str] = []
    if patients:
        ids = [p.strip() for p in patients.split(",") if p.strip()]
    else:
        try:
            root = require_run_tree(runs, what="runs")
        except LocalArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc
        records = [root] if root.is_file() else sorted(root.rglob(RUN_RECORD_GLOB))
        seen: set[str] = set()
        for r in records:
            pid = str(json.loads(r.read_text(encoding="utf-8")).get("patient_id") or "").strip()
            if pid and pid not in seen:
                seen.add(pid)
                ids.append(pid)
    if not ids:
        raise typer.BadParameter("no patient ids found; nothing to map")

    key = _case_key(store)
    existing = _read_map(store, out)
    patient_to_case = {patient: case for case, patient in existing.items()}
    minted = 0
    for pid in ids:
        if pid in patient_to_case:
            continue
        case = "CASE-" + hmac.new(key, pid.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
        # `safe_case_id` is the same gate the consumers apply. Asserting it HERE means a bad id is
        # a refusal at the moment it is minted rather than a BadParameter three commands later.
        A.safe_case_id(case)
        if case in existing:
            raise typer.BadParameter(
                f"{case} already maps to {existing[case]!r}, not {pid!r}: an HMAC collision at 48 "
                f"bits, or two patients differing only outside the id. Refusing to overwrite.")
        existing[case] = pid
        patient_to_case[pid] = case
        minted += 1

    path = store.write_json(out, existing)
    con.print(f"{len(existing)} case(s) in {path}  ([bold]{minted}[/] newly minted, "
              f"{len(existing) - minted} kept)")
    con.print("[dim]This file is the only thing that reverses a case id. It lives outside Git "
              "because it is the re-identification key, not because it is large.[/]")


#: The HMAC key's filename inside the local root. Its own file rather than a field of the map, so a
#: map may be shown to a collaborator while the key stays put.
CASE_KEY_FILE = "case-id-key.json"


def _case_key(store: LocalArtifactStore) -> bytes:
    """The site's 256-bit case-id key, minted on first use and never rotated automatically.

    Rotation would remint every case id, which detaches the whole library from its cases. If a key
    must be rotated, that is a decision with a migration, not a side effect of a command.
    """
    try:
        path = store.require_input(CASE_KEY_FILE, what="case-id key")
    except LocalArtifactError:
        key = secrets.token_hex(32)
        store.write_json(CASE_KEY_FILE, {"schema": "acr.case-id-key/1", "hmac_sha256_key": key})
        return bytes.fromhex(key)
    return bytes.fromhex(str(json.loads(path.read_text(encoding="utf-8"))["hmac_sha256_key"]))


def _read_map(store: LocalArtifactStore, out: str) -> dict[str, str]:
    """An existing map, or empty. Read through `_case_map` so a corrupt one fails here."""
    try:
        store.require_input(out, what="case map")
    except LocalArtifactError:
        return {}
    return dict(_case_map(store, out))
