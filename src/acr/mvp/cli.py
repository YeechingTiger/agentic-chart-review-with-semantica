"""The MVP's whole command surface: compile, run, trace, ingest, chain.

Five verbs and no more (the design doc's answer to "29 CLIs"). `compile` is the existing
contract loading chain — loading IS validation, so a spec that loads is a spec that compiled.
`run` drives one patient through the codex harness. `trace` renders a finished run as a
decision trace a person reads top to bottom. `ingest` distills the run into the judgment
ledger, and `chain` is the audit verb reading it back. Scoring joins when score.py lands.
"""
from __future__ import annotations

import argparse
import json
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

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
