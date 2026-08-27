"""Codex chart review → Langtrace cycles → Luna episodes → Semantica review graph."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from acr.contract.spec import load_spec
from acr.mvp.task_presentation import TASK_ARMS


def _ledger(path: Path):
    from acr.mvp.ledger import SemanticaLedger
    return SemanticaLedger(path)


def cmd_compile(args: argparse.Namespace) -> int:
    spec = load_spec(Path(args.spec))
    print(json.dumps({"spec_id": spec.spec_id, "spec_hash": spec.spec_hash, "ok": True}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from acr.mvp.runner import CodexCompatibilityError, run_patient
    try:
        run_dir = run_patient(
            Path(args.spec), Path(args.patient_dir), Path(args.out),
            model=args.model, base_url=args.base_url, timeout_s=args.timeout,
            codex_bin=args.codex_bin, task_arm=args.task_arm,
            langtrace_api_key=os.environ.get(args.langtrace_api_key_env),
            langtrace_api_host=args.langtrace_host,
            langtrace_project_id=args.langtrace_project_id,
        )
    except CodexCompatibilityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "runner_meta.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "status": result.get("status"),
                      "value": result.get("value"),
                      "langtrace_trace_id": meta.get("langtrace_trace_id"),
                      "task_presentation_hash": meta.get("task_presentation_hash")}))
    return 0


def _langtrace_review(run_dir: Path, args: argparse.Namespace):
    from acr.mvp.langtrace_io import LangtraceClient
    meta = json.loads((run_dir / "runner_meta.json").read_text(encoding="utf-8"))
    trace_id = getattr(args, "langtrace_trace_id", None) or meta.get("langtrace_trace_id")
    if not trace_id:
        raise RuntimeError("runner_meta.json contains no Langtrace trace id")
    return LangtraceClient(
        api_key=os.environ.get(args.langtrace_api_key_env, ""),
        api_host=args.langtrace_host,
        project_id=args.langtrace_project_id or meta.get("langtrace_project_id"),
    ).get_review(trace_id)


def cmd_trace(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if args.local_audit_copy:
        from acr.mvp.observe import decision_trace, render
        trace = decision_trace(run_dir)
        print(json.dumps(trace, ensure_ascii=False, indent=2) if args.json else render(trace))
        return 0
    from acr.mvp.timeline import build_react_cycles, build_trace_completeness
    review = _langtrace_review(run_dir, args)
    manifest = build_trace_completeness(review)
    cycles = build_react_cycles(review, manifest)
    payload = {"run_id": review.run_id, "trace_id": review.trace_id,
               "trace_completeness": manifest.to_dict(), "cycles": cycles}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{review.run_id} | Langtrace {review.trace_id} | "
              f"{manifest.event_count} events | {len(cycles)} cycles")
        for cycle in cycles:
            tool = next((action.get("tool") for action in cycle.get("actions") or []),
                        cycle["structural_kind"])
            question = f" — {cycle['declared_open_question']}" \
                if cycle.get("declared_open_question") else ""
            print(f"{cycle['source_seq_range'][0]:4d} {tool}{question}")
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    from acr.mvp.reconstruct import reconstruct_run, render
    from acr.mvp.reconstruction_llm import AuditedLiteLLM

    run_dir = Path(args.run_dir)
    review = _langtrace_review(run_dir, args)
    llm = AuditedLiteLLM(model=args.model, api_key=os.environ.get(args.api_key_env),
                         temperature=args.temperature,
                         **({"api_base": args.base_url} if args.base_url else {}))
    summary = reconstruct_run(
        review, _ledger(Path(args.ledger)), llm, passes=args.passes,
        artifact_dir=run_dir / "analyses", reconstructor_identity=args.model,
        max_attempts_per_pass=args.max_attempts)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    ledger = _ledger(Path(args.ledger))
    selection_id = ledger.select_analysis(
        args.run_id, args.analysis_id, selected_by=args.selected_by,
        reason=args.reason, provenance=args.provenance)
    print(json.dumps({"run_id": args.run_id, "analysis_id": args.analysis_id,
                      "selection_id": selection_id}))
    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    from acr.mvp.human_review import human_review_view, render
    view = human_review_view(
        _ledger(Path(args.ledger)), args.run_id, args.analysis,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        langtrace_ui_base=args.langtrace_ui_base)
    print(json.dumps(view, ensure_ascii=False, indent=2) if args.json else render(view))
    return 0


def cmd_insights(args: argparse.Namespace) -> int:
    report = _ledger(Path(args.ledger)).insights(args.run_id, args.analysis)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    report = _ledger(Path(args.ledger)).similar_candidates(
        args.episode_id, max_results=args.max_results,
        min_similarity=args.min_similarity,
        cross_run_only=not args.include_same_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    report = _ledger(Path(args.ledger)).impact_candidates(args.episode_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_causal_trace(args: argparse.Namespace) -> int:
    report = _ledger(Path(args.ledger)).causal_trace(
        args.episode_id, max_steps=args.max_steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_policy_impact(args: argparse.Namespace) -> int:
    report = _ledger(Path(args.ledger)).affected_by_policy_change(
        args.policy_id, from_version=args.from_version, to_version=args.to_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_review_ui(args: argparse.Namespace) -> int:
    from acr.mvp.review_ui import serve_review_ui

    serve_review_ui(
        Path(args.ledger), args.run_id, args.analysis,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        host=args.host, port=args.port, open_browser=not args.no_browser,
    )
    return 0


def _langtrace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--langtrace-host", default=os.environ.get("LANGTRACE_API_HOST"))
    parser.add_argument("--langtrace-api-key-env", default="LANGTRACE_API_KEY")
    parser.add_argument("--langtrace-project-id", default=os.environ.get("LANGTRACE_PROJECT_ID"))
    parser.add_argument("--langtrace-trace-id", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acr-mvp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="load and validate one Task Contract")
    compile_parser.add_argument("spec")
    compile_parser.set_defaults(fn=cmd_compile)

    run = sub.add_parser("run", help="run one chart review through the Codex App Server")
    run.add_argument("spec")
    run.add_argument("patient_dir")
    run.add_argument("--out", default="runs/mvp")
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--codex-bin", default="codex")
    run.add_argument("--task-arm", choices=TASK_ARMS, default="policy_bundle")
    _langtrace_args(run)
    run.set_defaults(fn=cmd_run)

    trace = sub.add_parser("trace", help="read the complete Langtrace export as fixed ReAct cycles")
    trace.add_argument("run_dir")
    trace.add_argument("--json", action="store_true")
    trace.add_argument("--local-audit-copy", action="store_true",
                       help="diagnostic only: read the local JSONL copy instead of Langtrace")
    _langtrace_args(trace)
    trace.set_defaults(fn=cmd_trace)

    reconstruct = sub.add_parser("reconstruct", help="annotate fixed cycles into Decision Episodes")
    reconstruct.add_argument("run_dir")
    reconstruct.add_argument("--ledger", default="runs/mvp/ledger.json")
    reconstruct.add_argument("--model", default=os.environ.get(
        "ACR_RECONSTRUCT_MODEL", "openrouter/openai/gpt-5.6-luna"))
    reconstruct.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    reconstruct.add_argument("--base-url", default=None)
    reconstruct.add_argument("--temperature", type=float, default=0.0)
    reconstruct.add_argument("--passes", type=int, default=2)
    reconstruct.add_argument("--max-attempts", type=int, default=3,
                             help="bounded verifier-feedback attempts per reconstruction pass")
    reconstruct.add_argument("--json", action="store_true")
    _langtrace_args(reconstruct)
    reconstruct.set_defaults(fn=cmd_reconstruct)

    select = sub.add_parser("select-analysis", help="append an explicit analysis selection")
    select.add_argument("run_id")
    select.add_argument("analysis_id")
    select.add_argument("--selected-by", required=True)
    select.add_argument("--reason", required=True)
    select.add_argument("--provenance", default="HUMAN_ADJUDICATED",
                        choices=["HUMAN_ADJUDICATED", "DETERMINISTIC_DERIVED"])
    select.add_argument("--ledger", default="runs/mvp/ledger.json")
    select.set_defaults(fn=cmd_select)

    chain = sub.add_parser("chain", help="render one run and one selected analysis for a human")
    chain.add_argument("run_id")
    chain.add_argument("--analysis", default=None,
                       help="inspect a provisional analysis without changing selection")
    chain.add_argument("--run-dir", default=None,
                       help="include Task Presentation and runner provenance from this run")
    chain.add_argument("--ledger", default="runs/mvp/ledger.json")
    chain.add_argument("--langtrace-ui-base", default=os.environ.get(
        "LANGTRACE_UI_BASE_URL", "http://127.0.0.1:3000"))
    chain.add_argument("--json", action="store_true")
    chain.set_defaults(fn=cmd_chain)

    insights = sub.add_parser("insights", help="analysis-scoped episode counts and stability")
    insights.add_argument("run_id")
    insights.add_argument("--analysis", default=None)
    insights.add_argument("--ledger", default="runs/mvp/ledger.json")
    insights.set_defaults(fn=cmd_insights)

    similar = sub.add_parser(
        "similar", help="find cross-run candidate situations with Semantica native retrieval")
    similar.add_argument("episode_id")
    similar.add_argument("--max-results", type=int, default=5)
    similar.add_argument("--min-similarity", type=float, default=0.3)
    similar.add_argument("--include-same-run", action="store_true")
    similar.add_argument("--ledger", default="runs/mvp/ledger.json")
    similar.set_defaults(fn=cmd_similar)

    impact = sub.add_parser(
        "impact", help="find possible downstream decisions with Semantica native analytics")
    impact.add_argument("episode_id")
    impact.add_argument("--ledger", default="runs/mvp/ledger.json")
    impact.set_defaults(fn=cmd_impact)

    causal_trace = sub.add_parser(
        "causal-trace", help="trace explicit typed decision relationships with Semantica")
    causal_trace.add_argument("episode_id")
    causal_trace.add_argument("--max-steps", type=int, default=5)
    causal_trace.add_argument("--ledger", default="runs/mvp/ledger.json")
    causal_trace.set_defaults(fn=cmd_causal_trace)

    policy_impact = sub.add_parser(
        "policy-impact", help="find decisions to re-audit after a Task Contract version change")
    policy_impact.add_argument("policy_id")
    policy_impact.add_argument("--from-version", required=True)
    policy_impact.add_argument("--to-version", required=True)
    policy_impact.add_argument("--ledger", default="runs/mvp/ledger.json")
    policy_impact.set_defaults(fn=cmd_policy_impact)

    review_ui = sub.add_parser(
        "review-ui", help="open the selected chain in Semantica Explorer Decisions")
    review_ui.add_argument("run_id")
    review_ui.add_argument("--analysis", default=None)
    review_ui.add_argument("--run-dir", default=None,
                           help="include this run's Task Presentation and runner provenance")
    review_ui.add_argument("--ledger", default="runs/mvp/ledger.json")
    review_ui.add_argument("--host", default="127.0.0.1")
    review_ui.add_argument("--port", type=int, default=8765)
    review_ui.add_argument("--no-browser", action="store_true")
    review_ui.set_defaults(fn=cmd_review_ui)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
