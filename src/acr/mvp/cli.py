"""The MVP's whole command surface: compile, run, trace, ingest, and the three read verbs.

`compile` is the existing contract loading chain — loading IS validation, so a spec that loads
is a spec that compiled. `run` drives one patient through the codex harness. `trace` renders a
finished run as a decision trace a person reads top to bottom. `ingest` distills a run into the
judgment ledger (runs with `--ledger` record themselves live and need no ingest).

Then the three verbs of the decision-precipitation design, in the order they are used:
`chain` audits one case, `decisions` compares a class across runs, and `precipitate` asks what
that class has settled and where it still diverges. Scoring joins when score.py lands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from acr.contract.spec import load_spec


def _ledger(path: Path):
    from acr.mvp.ledger import SemanticaLedger
    return SemanticaLedger(path)


def cmd_compile(args: argparse.Namespace) -> int:
    spec = load_spec(Path(args.spec))
    print(json.dumps({"spec_id": spec.spec_id, "spec_hash": spec.spec_hash, "ok": True}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from acr.mvp.runner import run_patient
    run_dir = run_patient(
        Path(args.spec), Path(args.patient_dir), Path(args.out),
        model=args.model, base_url=args.base_url, timeout_s=args.timeout,
        codex_bin=args.codex_bin,
        ledger_path=Path(args.ledger) if args.ledger else None,
    )
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "status": result.get("status"),
                      "value": result.get("value")}))
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from acr.mvp.observe import decision_trace, render
    trace = decision_trace(Path(args.run_dir))
    print(json.dumps(trace, ensure_ascii=False, indent=2) if args.json else render(trace))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from acr.mvp.ledger import ingest_run
    ledger = _ledger(Path(args.ledger))
    summary = ingest_run(Path(args.run_dir), ledger)
    print(json.dumps(summary))
    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    ledger = _ledger(Path(args.ledger))
    print(json.dumps(ledger.chain(args.case_id), indent=2))
    return 0


def cmd_precipitate(args: argparse.Namespace) -> int:
    from acr.mvp.precipitate import render, survey
    report = survey(_ledger(Path(args.ledger)), decision_type=args.type, level=args.level,
                    settled_min=args.settled_min)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    """Read a finished run back as a decision tree, into the ledger.

    The extractor is a model, so this prints the reading's own health first — how much of it
    the run actually said versus how much was inferred, which quotes did not hold up, and
    which stretches of the run no point accounts for. Believe the tree only as far as those
    numbers let you."""
    from semantica.llms import LiteLLM

    from acr.mvp.reconstruct import reconstruct_run, render
    llm = LiteLLM(model=args.model, api_key=os.environ.get(args.api_key_env),
                  temperature=args.temperature,
                  **({"api_base": args.base_url} if args.base_url else {}))
    summary = reconstruct_run(Path(args.run_dir), _ledger(Path(args.ledger)), llm,
                              passes=args.passes)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary))
    return 0


def cmd_decisions(args: argparse.Namespace) -> int:
    ledger = _ledger(Path(args.ledger))
    prefix = f"{args.level or 'big'}:{args.type}" if args.type else args.prefix
    rows = ledger.decisions(category_prefix=prefix, case_id=args.case)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="acr-mvp", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="load a contract; loading is validation")
    c.add_argument("spec")
    c.set_defaults(fn=cmd_compile)

    r = sub.add_parser("run", help="review one patient through the codex harness")
    r.add_argument("spec")
    r.add_argument("patient_dir")
    r.add_argument("--out", default="runs/mvp")
    r.add_argument("--model", default=None)
    r.add_argument("--base-url", default=None)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--codex-bin", default="codex")
    r.add_argument("--ledger", default=None,
                   help="record judgments into this semantica ledger LIVE during the run")
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("trace", help="render a run as an ordered decision trace")
    t.add_argument("run_dir")
    t.add_argument("--json", action="store_true", help="emit the trace as JSON instead of text")
    t.set_defaults(fn=cmd_trace)

    i = sub.add_parser("ingest", help="distill a run's Layer-1 trace into the ledger")
    i.add_argument("run_dir")
    i.add_argument("--ledger", default="runs/mvp/ledger.json")
    i.set_defaults(fn=cmd_ingest)

    ch = sub.add_parser("chain", help="audit: result <- gate <- submission for one case")
    ch.add_argument("case_id")
    ch.add_argument("--ledger", default="runs/mvp/ledger.json")
    ch.set_defaults(fn=cmd_chain)

    pr = sub.add_parser("precipitate",
                        help="guideline material: which decision points have settled, which "
                             "diverge, and what kind of rule each gap wants")
    pr.add_argument("--type", default=None, help="restrict to one decision type")
    pr.add_argument("--settled-min", type=int, default=3,
                    help="how many like decisions make a practice rather than an anecdote")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--level", default=None, choices=["big", "small"])
    pr.add_argument("--ledger", default="runs/mvp/ledger.json")
    pr.set_defaults(fn=cmd_precipitate)

    rc = sub.add_parser("reconstruct",
                        help="read a finished run back as a typed decision tree (needs an LLM)")
    rc.add_argument("run_dir")
    rc.add_argument("--ledger", default="runs/mvp/ledger.json")
    rc.add_argument("--model", default=os.environ.get("ACR_RECONSTRUCT_MODEL",
                                                      "openrouter/anthropic/claude-sonnet-4.5"),
                    help="a LiteLLM model id, e.g. openrouter/anthropic/claude-sonnet-4.5")
    rc.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    rc.add_argument("--base-url", default=None)
    rc.add_argument("--temperature", type=float, default=0.0)
    rc.add_argument("--passes", type=int, default=1,
                    help="extract this many times and report how far the readings drift; "
                         "only the first is stored")
    rc.add_argument("--json", action="store_true")
    rc.set_defaults(fn=cmd_reconstruct)

    d = sub.add_parser("decisions",
                       help="compare: decision points across runs, filtered by type or case")
    d.add_argument("--type", default=None, help="a decision type, e.g. enough, which_wins")
    d.add_argument("--level", default=None, choices=["big", "small"],
                   help="which level --type refers to (default big)")
    d.add_argument("--prefix", default=None,
                   help="raw category prefix (big:, small:, step, submit:, gate:, result:)")
    d.add_argument("--case", default=None, help="filter to one case id")
    d.add_argument("--ledger", default="runs/mvp/ledger.json")
    d.set_defaults(fn=cmd_decisions)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
