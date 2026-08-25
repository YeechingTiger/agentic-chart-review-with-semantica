"""The judgment ledger: semantica as the account book, behind an interface it cannot escape.

Three verbs, from the decision-precipitation design: **audit** (read one judgment's chain),
**compare** (read a class of judgments), **precipitate** (promote a class — later). The ledger
records; it never decides, never runs the review, never gates anything. Ingestion is post-hoc
from the Layer-1 trace, which makes write-behind true by construction: a run is complete and
scoreable before the ledger has heard of it, and a ledger failure can only ever lose bookkeeping.

The graph per run, walking upstream from the result:

    result  ◄─CAUSED─  gate verdict  ◄─CAUSED─  submission  ─uses─►  evidence spans
    (deterministic_runtime)  (rule_engine)        (model)

Refused submissions are judgments too — they enter the chain the same way, which is what makes
"the gate refused twice before accepting" a readable fact rather than a vanished one.

semantica is pinned at 0.6.6 and used strictly in-process (no REST, no MCP, no Explorer —
the PHI posture from the design docs). Every heavyweight feature is switched off at
construction; what remains is nodes, edges, decisions and JSON persistence. The `ReviewLedger`
protocol is the seam: `NullLedger` satisfies it for tests and for running without the
dependency installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class ReviewLedger(Protocol):
    def record_evidence(self, run_id: str, index: int, ev: dict[str, Any]) -> str: ...
    def record_judgment(self, run_id: str, *, category: str, scenario: str, reasoning: str,
                        outcome: str, decision_maker: str,
                        metadata: dict[str, Any] | None = None) -> str: ...
    def link_caused(self, source_id: str, target_id: str) -> None: ...
    def link_uses(self, judgment_id: str, evidence_id: str) -> None: ...
    def set_case_result(self, case_id: str, result_decision_id: str) -> None: ...
    def chain(self, case_id: str) -> list[dict[str, Any]]: ...
    def save(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


class NullLedger:
    """Counts what it is asked to record and remembers nothing else."""

    def __init__(self) -> None:
        self.counts = {"evidence": 0, "judgments": 0, "edges": 0}
        self._results: dict[str, str] = {}

    def record_evidence(self, run_id: str, index: int, ev: dict[str, Any]) -> str:
        self.counts["evidence"] += 1
        return f"ev:{run_id}:{index}"

    def record_judgment(self, run_id: str, **kw: Any) -> str:
        self.counts["judgments"] += 1
        return f"j:{run_id}:{self.counts['judgments']}"

    def link_caused(self, source_id: str, target_id: str) -> None:
        self.counts["edges"] += 1

    def link_uses(self, judgment_id: str, evidence_id: str) -> None:
        self.counts["edges"] += 1

    def set_case_result(self, case_id: str, result_decision_id: str) -> None:
        self._results[case_id] = result_decision_id

    def chain(self, case_id: str) -> list[dict[str, Any]]:
        return []

    def save(self) -> None:
        pass

    def stats(self) -> dict[str, Any]:
        return dict(self.counts)


class SemanticaLedger:
    """semantica 0.6.6's ContextGraph as the account book, JSON-persisted at `path`.

    The case→result index is a sidecar JSON beside the graph file rather than a graph query:
    it is the one lookup the audit verb starts from, and a plain dict cannot drift.
    """

    def __init__(self, path: Path) -> None:
        try:
            from semantica.context.context_graph import ContextGraph
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise ImportError(
                "SemanticaLedger needs the pinned dependency: "
                "uv pip install --no-deps semantica==0.6.6 && "
                "uv pip install rdflib networkx numpy scipy python-dateutil"
            ) from e
        self.path = Path(path)
        self.index_path = self.path.with_suffix(".index.json")
        self.graph = ContextGraph(
            extract_entities=False, extract_relationships=False, advanced_analytics=False,
            centrality_analysis=False, community_detection=False, node_embeddings=False,
        )
        if self.path.exists():
            self.graph.load_from_file(str(self.path))
        self._index: dict[str, str] = (
            json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        )

    def record_evidence(self, run_id: str, index: int, ev: dict[str, Any]) -> str:
        node_id = f"ev:{run_id}:{index}"
        self.graph.add_node(
            node_id, "Evidence", content=(ev.get("quote") or "")[:300],
            note_id=ev.get("note_id"), start=ev.get("start"), end=ev.get("end"),
            supports=ev.get("supports"), field=ev.get("field"), run_id=run_id,
        )
        return node_id

    def record_judgment(self, run_id: str, *, category: str, scenario: str, reasoning: str,
                        outcome: str, decision_maker: str,
                        metadata: dict[str, Any] | None = None) -> str:
        # semantica refuses empty strings; a judgment with no stated reasoning is still a
        # judgment, and "(none given)" keeps that fact readable instead of failing ingestion.
        return self.graph.record_decision(
            category=category, scenario=(scenario or "(none given)")[:500],
            reasoning=(reasoning or "(none given)")[:500],
            outcome=outcome, confidence=1.0, decision_maker=decision_maker,
            metadata={"run_id": run_id, **(metadata or {})},
        )

    def link_caused(self, source_id: str, target_id: str) -> None:
        self.graph.add_causal_relationship(source_id, target_id, "CAUSED")

    def link_uses(self, judgment_id: str, evidence_id: str) -> None:
        self.graph.add_edge(source_id=judgment_id, target_id=evidence_id, edge_type="uses")

    def set_case_result(self, case_id: str, result_decision_id: str) -> None:
        self._index[case_id] = result_decision_id

    def chain(self, case_id: str) -> list[dict[str, Any]]:
        """The audit verb: result ← gate ← submission(s), upstream from the case's result.

        The anchor comes from `find_node` because `get_causal_chain` returns only the nodes
        upstream OF the anchor, never the anchor itself — an audit that omitted the verdict
        being audited would read as a chain to nowhere."""
        result_id = self._index.get(case_id)
        if result_id is None:
            return []
        keys = ("decision_id", "category", "scenario", "outcome", "decision_maker")
        rows: list[dict[str, Any]] = []
        anchor = self.graph.find_node(result_id)
        if anchor:
            meta = anchor.get("metadata") or {}
            rows.append({"decision_id": anchor.get("id"),
                         **{k: meta.get(k) for k in keys[1:]}})
        for d in self.graph.get_causal_chain(result_id, direction="upstream", max_depth=10):
            row = d if isinstance(d, dict) else getattr(d, "__dict__", {"repr": repr(d)})
            rows.append({k: row.get(k) for k in keys})
        return rows

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save_to_file(str(self.path))
        self.index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def stats(self) -> dict[str, Any]:
        return {**self.graph.stats(), "cases": len(self._index)}


# ---------------------------------------------------------------------------- ingestion
def ingest_run(run_dir: Path, ledger: ReviewLedger) -> dict[str, Any]:
    """Distill one run's Layer-1 trace into the ledger. Idempotence is the caller's concern
    (ingest a run once); the run directory name is the run identity."""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    meta = next((e for e in events if e.get("kind") == "run_meta"), {})
    spec_id = meta.get("spec_id", "unknown")
    case_id = meta.get("patient_id", run_id)

    evidence_ids: list[str] = []
    last_gate_id: str | None = None
    n_submissions = 0
    for e in events:
        if e.get("kind") != "tool_call":
            continue
        tool, args, result = e.get("tool"), e.get("args") or {}, e.get("result") or {}
        if tool == "record_evidence" and e.get("ok"):
            evidence_ids.append(ledger.record_evidence(
                run_id, len(evidence_ids),
                {**args, "quote": result.get("quote", "")}))
        elif tool == "submit_answer":
            n_submissions += 1
            submission = ledger.record_judgment(
                run_id, category=f"submit:{spec_id}",
                scenario=f"case {case_id}, submission {n_submissions}",
                reasoning=str(args.get("reasoning") or ""),
                outcome=str(args.get("status")), decision_maker="model",
                metadata={"value": args.get("value")})
            for ev_id in evidence_ids:
                ledger.link_uses(submission, ev_id)
            verdict = "accepted" if result.get("accepted") else f"refused: {result.get('why')}"
            gate = ledger.record_judgment(
                run_id, category=f"gate:{spec_id}",
                scenario=f"case {case_id}, submission {n_submissions}",
                reasoning=str(result.get("why") or ""),
                outcome=verdict, decision_maker="rule_engine")
            ledger.link_caused(submission, gate)
            last_gate_id = gate

    result_path = run_dir / "result.json"
    final = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    result_node = ledger.record_judgment(
        run_id, category=f"result:{spec_id}", scenario=f"case {case_id}",
        reasoning=str(final.get("reasoning") or final.get("why") or ""),
        outcome=str(final.get("status", "NO_ANSWER")), decision_maker="deterministic_runtime",
        metadata={"value": final.get("value")})
    if last_gate_id is not None:
        ledger.link_caused(last_gate_id, result_node)
    ledger.set_case_result(case_id, result_node)
    ledger.save()
    return {"run_id": run_id, "case_id": case_id, "n_evidence": len(evidence_ids),
            "n_submissions": n_submissions, "result": final.get("status", "NO_ANSWER")}
