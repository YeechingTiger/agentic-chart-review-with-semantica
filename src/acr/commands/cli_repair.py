"""Run the DEVELOP-plane SpecRepairLab over gold, manifests and candidate text edits."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import typer
from rich.table import Table

from ..contract import spec_repair as S
from ..core import cli_common
from ..core.cli_common import API_BASE, CORPUS, MODEL, con, read_json
from ..core.local_artifacts import (
    LOCAL_ROOT_ENV,
    LocalArtifactError,
    LocalArtifactStore,
    require_run_tree,
)

repair_app = typer.Typer(add_completion=False, help=(
    "DEVELOP only: sample deepagents behaviours, contrast them with chart-observable gold, "
    "validate minimal spec patches, and certify on a sealed cohort."))

LOCAL_ROOT = typer.Option(
    None, "--local-root", envvar=LOCAL_ROOT_ENV,
    help="absolute patient-artifact root outside Git")


def _store(root: str | None) -> LocalArtifactStore:
    try:
        return LocalArtifactStore(root)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _case_map(path: str, store: LocalArtifactStore) -> dict[str, str]:
    """Return original patient id -> pseudonymous case id; never serialise the original."""
    if not path:
        return {}
    try:
        p = store.require_input(path, what="case map")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raw = read_json(p, "case map")
    if not isinstance(raw, dict):
        raise typer.BadParameter("--case-map must be {case_id: patient_id}")
    out = {}
    for case_id, patient_id in raw.items():
        if not str(case_id).strip() or not str(patient_id).strip():
            raise typer.BadParameter("--case-map contains an empty case or patient id")
        if str(patient_id) in out:
            raise typer.BadParameter(f"patient id appears twice in --case-map: {patient_id}")
        out[str(patient_id)] = str(case_id)
    return out


def _load_distributions(runs: str, gold_path: str, store: LocalArtifactStore,
                        case_map: str = "",
                        ) -> tuple[list[S.BehaviorDistribution],
                                   dict[str, S.ChartObservableGold]]:
    try:
        gold_file = store.require_input(gold_path, what="chart-observable gold")
        # See `require_run_tree`. `gold` above stays on the store — a curated answer key IS a
        # develop artifact and belongs outside Git. A run record is not, and sending it through the
        # same proof made this whole plane unreachable.
        runs_path = require_run_tree(runs, what="recorded runs")
        gold = S.load_gold(gold_file)
        signatures = S.load_signatures(
            [runs_path], case_map=_case_map(case_map, store))
    except (S.SpecRepairError, LocalArtifactError) as e:
        raise typer.BadParameter(str(e)) from e
    grouped: dict[tuple[str, str], list[S.BehaviorSignature]] = defaultdict(list)
    for sig in signatures:
        grouped[(sig.case_id, sig.spec_id)].append(sig)
    out = []
    for (case_id, _), rows in sorted(grouped.items()):
        out.append(S.cluster_behaviors(rows, gold.get(case_id)))
    return out, gold


@repair_app.command("sample")
def sample(
    spec: str = typer.Option(..., "--spec", "-s"),
    gold: str = typer.Option(..., "--gold", help=f"{S.GOLD_SCHEMA} JSON"),
    case_map: str = typer.Option(
        "", "--case-map",
        help="optional JSON {pseudonymous_case_id: corpus_patient_id}; required for real IDs"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    initial_runs: int = typer.Option(3, "--initial-runs", min=2),
    hard_runs: int = typer.Option(
        5, "--hard-runs", min=2,
        help="total runs for a case whose initial behaviours disagree or do not ground"),
    max_steps: int = cli_common.MAX_STEPS,
    max_usd: float = cli_common.MAX_USD,
    temperature: float = typer.Option(1.0, "--temperature"),
    seed: int = typer.Option(1234, "--seed"),
    out: str = typer.Option("spec-repair", "--out",
                            help="root-relative local run-directory prefix"),
    local_root: str | None = LOCAL_ROOT,
):
    """Sample the existing deepagents runtime; expand only hard cases and change no spec."""
    if hard_runs < initial_runs:
        raise typer.BadParameter("--hard-runs must be at least --initial-runs")
    from ..chartstore.corpus import Corpus
    from ..contract.spec import load_spec
    from ..review.agent import run_patient

    store = _store(local_root)
    try:
        gold_path = store.require_input(gold, what="chart-observable gold")
        gold_rows = S.load_gold(gold_path)
    except (S.SpecRepairError, LocalArtifactError) as e:
        raise typer.BadParameter(str(e)) from e
    mapping = _case_map(case_map, store)
    reverse = {case_id: patient_id for patient_id, case_id in mapping.items()}
    sp = load_spec(spec)
    eligible = [g for g in gold_rows.values()
                if g.spec_id == sp.spec_id and g.usable_for_repair]
    if not eligible:
        raise typer.BadParameter(
            f"{gold}: no repair-eligible cases for {sp.spec_id}; run `acr gold audit`")
    c = Corpus(Path(corpus))
    try:
        store.ensure()
        run_prefix = store.path(out, what="spec-repair output")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    run_dir = cli_common.unique_run_dir(str(run_prefix))
    os.chmod(run_dir, 0o700)
    llm = cli_common.chat_model(model, api_base, temperature)
    rows = []
    for g in eligible:
        patient_id = reverse.get(g.case_id, g.case_id)
        if patient_id not in c.patient_ids():
            raise typer.BadParameter(
                f"{g.case_id}: patient {patient_id!r} is not in {corpus}; provide --case-map")
        manifests = []
        for i in range(initial_runs):
            manifests.append(run_patient(
                spec=sp, corpus=c, patient_id=patient_id, out_dir=run_dir, model=llm,
                max_model_calls=max_steps, seed=seed + i,
                run_id=f"{g.case_id}__d{i + 1}", max_usd=max_usd))
        signatures = [S.BehaviorSignature.from_manifest(m, case_id=g.case_id)
                      for m in manifests]
        distribution = S.cluster_behaviors(signatures, g)
        hard = len(distribution.clusters) > 1 or distribution.grounded_consistency != 1.0
        if hard:
            for i in range(initial_runs, hard_runs):
                manifest = run_patient(
                    spec=sp, corpus=c, patient_id=patient_id, out_dir=run_dir, model=llm,
                    max_model_calls=max_steps, seed=seed + i,
                    run_id=f"{g.case_id}__d{i + 1}", max_usd=max_usd)
                signatures.append(S.BehaviorSignature.from_manifest(
                    manifest, case_id=g.case_id))
            distribution = S.cluster_behaviors(signatures, g)
        rows.append(distribution)
        con.print(f"{g.case_id}: {distribution.n_runs} run(s), "
                  f"{len(distribution.clusters)} cluster(s), "
                  f"grounded={distribution.grounded_consistency}")
    report = S.behavior_document(rows)
    report["run_dir"] = str(run_dir)
    report["sampling"] = {"initial_runs": initial_runs, "hard_runs": hard_runs,
                          "seed_start": seed, "adaptive": True}
    p = store.write_json(run_dir / "behavior-distributions.json", report)
    con.print(f"→ {p}")


@repair_app.command("cluster")
def cluster(
    runs: str = typer.Option(..., "--runs", help="manifest file or directory"),
    gold: str = typer.Option(..., "--gold"),
    case_map: str = typer.Option("", "--case-map"),
    out: str = typer.Option("", "--out"),
    local_root: str | None = LOCAL_ROOT,
):
    """Cluster recorded trajectories by structured behaviour, not reasoning prose."""
    store = _store(local_root)
    rows, _ = _load_distributions(runs, gold, store, case_map)
    report = S.behavior_document(rows)
    t = Table("case", "runs", "clusters", "entropy", "gold", "grounded", "overclaim")
    for d in rows:
        t.add_row(d.case_id, str(d.n_runs), str(len(d.clusters)),
                  f"{d.behavioral_entropy:.3f}", str(d.gold_consistency),
                  str(d.grounded_consistency), str(d.overclaim_rate))
    con.print(t)
    if out:
        try:
            p = store.write_json(out, report)
        except LocalArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc
        con.print(f"→ {p}")


@repair_app.command("diagnose")
def diagnose(
    runs: str = typer.Option(..., "--runs"),
    gold: str = typer.Option(..., "--gold"),
    spec: str = typer.Option(..., "--spec", "-s"),
    case_map: str = typer.Option("", "--case-map"),
    out: str = typer.Option(..., "--out", help="write the packet collection here"),
    local_root: str | None = LOCAL_ROOT,
):
    """Build selected-vs-rejected contrastive packets; makes no edit and calls no model."""
    from ..contract.spec import load_spec

    store = _store(local_root)
    distributions, gold_rows = _load_distributions(runs, gold, store, case_map)
    sp = load_spec(spec)
    packets = []
    for d in distributions:
        if d.spec_id != sp.spec_id or d.case_id not in gold_rows:
            continue
        packets.append(S.diagnose(d, gold_rows[d.case_id], sp))
    doc = {"schema": "acr.contrastive_failure_packets/1",
           "packets": [p.to_dict() for p in packets],
           "summary": {"n_packets": len(packets),
                       "n_repair_permitted": sum(p.repair_permitted for p in packets),
                       "by_disposition": dict(Counter(p.disposition for p in packets))}}
    t = Table("case", "disposition", "repair?", "why")
    for p in packets:
        t.add_row(p.case_id, p.disposition, str(p.repair_permitted), p.why[:68])
    con.print(t)
    try:
        p = store.write_json(out, doc)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    con.print(f"→ {p}")


def _packet(path: str, case_id: str) -> S.ContrastiveFailurePacket:
    raw = read_json(path, "contrastive packet")
    rows = raw.get("packets") if isinstance(raw, dict) else None
    if rows is None and isinstance(raw, dict) and raw.get("schema") == S.PACKET_SCHEMA:
        rows = [raw]
    if not isinstance(rows, list):
        raise typer.BadParameter(f"{path}: expected one packet or a packet collection")
    selected = [r for r in rows if not case_id or r.get("case_id") == case_id]
    if len(selected) != 1:
        raise typer.BadParameter(
            f"--case resolved to {len(selected)} packet(s); provide one exact case id")
    r = selected[0]
    return S.ContrastiveFailurePacket(
        case_id=str(r["case_id"]), spec_id=str(r["spec_id"]),
        spec_hash=str(r.get("spec_hash") or ""), disposition=str(r["disposition"]),
        selected=r.get("selected"), rejected=r.get("rejected"),
        differences=dict(r.get("differences") or {}), gold=dict(r.get("gold") or {}),
        spec_sections=dict(r.get("spec_sections") or {}),
        repair_permitted=bool(r.get("repair_permitted")), why=str(r.get("why") or ""))


@repair_app.command("propose")
def propose(
    packet: str = typer.Option(..., "--packet"),
    spec: str = typer.Option(..., "--spec", "-s"),
    case: str = typer.Option("", "--case", help="required when packet file has multiple cases"),
    proposal: str = typer.Option(
        "", "--proposal",
        help="validate this supplied proposal instead of calling a model"),
    model: str = MODEL,
    api_base: str = API_BASE,
    max_usd: float = typer.Option(..., "--max-usd", min=0.01,
                                  help="one-call proposal ceiling; required, no default"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="print the model contract and make no provider call"),
    out: str = typer.Option(..., "--out"),
    local_root: str | None = LOCAL_ROOT,
):
    """Generate or validate one minimal, cited proposal; never applies it to the spec."""
    store = _store(local_root)
    try:
        packet_path = store.require_input(packet, what="contrastive packet")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    p = _packet(str(packet_path), case)
    spec_text = Path(spec).read_text(encoding="utf-8")
    if proposal:
        try:
            proposal_path = store.require_input(proposal, what="proposal")
        except LocalArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc
        raw = read_json(proposal_path, "proposal")
        try:
            got = S.SpecPatchProposal.from_dict(raw, spec_text=spec_text)
        except S.SpecRepairError as e:
            raise typer.BadParameter(str(e)) from e
    else:
        contract = {
            "case_id": p.case_id, "spec_id": p.spec_id,
            "failure_class": f"copy {p.disposition}",
            "parameter_id": f"one of {list(S.PARAMETERS)}",
            "quoted_current_text": "verbatim text that occurs in the spec",
            "selected_vs_rejected_difference": {},
            "minimal_patch": "replacement/addition only, not a whole rewritten spec",
            "expected_behavior_change": "",
            "change_class": f"one of {list(S.CHANGE_CLASSES)}",
            "source_basis": "",
            "cases_addressed": [p.case_id],
            "blast_radius": {"computable": False, "basis": ""},
            "requires_clinician_signoff": True,
        }
        if dry_run:
            try:
                path = store.write_json(
                    out, {"would_call_model": True, "packet": p.to_dict(),
                          "required_output": contract})
            except LocalArtifactError as exc:
                raise typer.BadParameter(str(exc)) from exc
            con.print(f"→ {path}")
            return
        client = cli_common.llm_client(model, api_base, temperature=0.0)
        prompt = (
            "You propose ONE minimal edit to an extraction specification. The packet contains "
            "a selected grounded behaviour and a rejected behaviour. Do not change the clinical "
            "target to fit the answer key. A retrieval failure may alter only a retrieval "
            "parameter. Semantic edits require clinician sign-off. quoted_current_text must be "
            "copied verbatim from spec_sections. Return only the required JSON.\n\nPACKET:\n"
            + json.dumps(p.to_dict(), ensure_ascii=False)
            + "\n\nOUTPUT CONTRACT:\n" + json.dumps(contract, ensure_ascii=False))
        raw = client.json_chat([{"role": "user", "content": prompt}],
                               schema_hint=json.dumps(contract))
        try:
            got = S.SpecPatchProposal.from_dict(raw, spec_text=spec_text)
        except S.SpecRepairError as e:
            raise typer.BadParameter(f"model proposal rejected: {e}") from e
        # A one-call command cannot prevent an unexpectedly expensive completed call, but it
        # can refuse to publish its output. The report keeps that fact visible.
        from ..core.spend import Spend
        spend = Spend(max_usd=max_usd, model=model or "")
        spend.add({"input_tokens": client.prompt_tokens, "output_tokens": client.completion_tokens,
                   "input_token_details": {"cache_read": client.cached_tokens}})
        if spend.exceeded():
            con.print(f"[red]{spend.exceeded()}; proposal not written[/]")
            raise typer.Exit(2)
    try:
        got = S.validate_proposal_for_packet(got, p)
    except S.SpecRepairError as e:
        raise typer.BadParameter(str(e)) from e
    doc = got.to_dict()
    try:
        path = store.write_json(out, doc)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    con.print(f"→ {path}")
    con.print("[yellow]candidate only[/]: this command never edits the spec; "
              + ("clinician sign-off required" if got.requires_clinician_signoff
                 else "asset certification still required"))


@repair_app.command("validate")
def validate(
    before: str = typer.Option(..., "--before", help="baseline manifest directory"),
    after: str = typer.Option(..., "--after", help="candidate manifest directory"),
    gold: str = typer.Option(..., "--gold"),
    before_case_map: str = typer.Option("", "--before-case-map"),
    after_case_map: str = typer.Option("", "--after-case-map"),
    max_subgroup_drop: float = typer.Option(
        ..., "--max-subgroup-drop",
        help="maximum tolerated absolute accuracy drop, 0..1; required"),
    allow_flat: bool = typer.Option(
        False, "--allow-flat",
        help="do not require a positive grounded-correct mean; regressions still refuse"),
    out: str = typer.Option(..., "--out"),
    local_root: str | None = LOCAL_ROOT,
):
    """Paired, per-instance validation; exits non-zero on any acceptance failure."""
    if not 0 <= max_subgroup_drop <= 1:
        raise typer.BadParameter("--max-subgroup-drop must be between 0 and 1")
    store = _store(local_root)
    b, gold_rows = _load_distributions(before, gold, store, before_case_map)
    a, _ = _load_distributions(after, gold, store, after_case_map)
    try:
        report = S.paired_validate(
            b, a, gold_rows, max_subgroup_drop=max_subgroup_drop,
            require_positive_mean=not allow_flat)
    except S.SpecRepairError as e:
        raise typer.BadParameter(str(e)) from e
    try:
        path = store.write_json(out, report.to_dict())
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    con.print(f"→ {path}")
    colour = "green" if report.accepted else "red"
    con.print(f"[bold {colour}]{'ACCEPT' if report.accepted else 'REFUSE'}[/] "
              f"grounded delta={report.mean_grounded_delta:+.3f}, "
              f"{len(report.regressions)} regression(s)")
    if not report.accepted:
        for reason in report.refusal_reasons:
            con.print(f"[red]- {reason}[/]")
        raise typer.Exit(1)


@repair_app.command("certify")
def certify(
    validation_report: str = typer.Option(..., "--validation-report"),
    sealed_cases: str = typer.Option(
        ..., "--sealed-cases", help="JSON list of pseudonymous sealed case ids"),
    bundle_hash: str = typer.Option(..., "--bundle-hash"),
    state: str = typer.Option(..., "--state",
                              help="write-once sealed certification state JSON"),
    local_root: str | None = LOCAL_ROOT,
):
    """Consume a sealed cohort once; a second read is refused instead of warned."""
    store = _store(local_root)
    try:
        report_path = store.require_input(validation_report, what="validation report")
        cases_path = store.require_input(sealed_cases, what="sealed cases")
        state_path = store.path(state, what="sealed certification state")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = read_json(report_path, "validation report")
    if not isinstance(report, dict) or report.get("schema") != S.VALIDATION_SCHEMA:
        raise typer.BadParameter(
            f"{validation_report}: expected schema {S.VALIDATION_SCHEMA}")
    cases = read_json(cases_path, "sealed cases")
    if not isinstance(cases, list) or not cases:
        raise typer.BadParameter("--sealed-cases must be a non-empty JSON list")
    for case_id in cases:
        try:
            S.safe_case_id(case_id)
        except S.SpecRepairError as e:
            raise typer.BadParameter(str(e)) from e
    cohort_hash = S.artifact_hash(sorted(str(x) for x in cases))
    if state_path.exists():
        prior = read_json(state_path, "sealed certification state")
        if prior.get("consumed"):
            raise typer.BadParameter(
                f"sealed cohort {prior.get('cohort_hash')} was already consumed; mint a new one")
        cert = S.SealedCertification(str(prior["cohort_hash"]), str(prior["bundle_hash"]))
    else:
        cert = S.SealedCertification(cohort_hash, bundle_hash)
    if (cert.cohort_hash, cert.bundle_hash) != (cohort_hash, bundle_hash):
        raise typer.BadParameter("sealed cohort or bundle does not match the stored state")
    consumed = cert.consume(report)
    store.write_json(state_path, consumed.to_dict())
    con.print(f"→ {state_path}")
    if not report.get("accepted"):
        con.print("[red]sealed report failed; cohort is consumed and may only be used for "
                  "diagnosis in the next cycle[/]")
        raise typer.Exit(1)
    con.print("[green]sealed report accepted[/]")
