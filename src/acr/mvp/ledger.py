"""The judgment ledger: semantica as the account book, behind an interface it cannot escape.

Three verbs, from the decision-precipitation design: **audit** (read one judgment's chain),
**compare** (read a class of judgments), **precipitate** (promote a class — later). The ledger
records; it never decides, never runs the review, never gates anything.

Two recording paths, ONE decomposition. `RunRecorder` is the decomposition — what counts as a
judgment, which edges connect them — and both paths drive it:

  * live: the toolserver records each judgment at the moment it happens (semantica's own
    intended usage), best-effort — a ledger failure is a stderr line, never a failed run;
  * replay: `ingest_run` replays a finished trace.jsonl through the same recorder, for runs
    that had no live ledger or for rebuilding a lost one. Replay a run into a FRESH ledger:
    replaying into one that already heard the run live records everything twice.

Either way the Layer-1 trace stays the authority: the ledger can always be rebuilt from it,
and never the other way around.

The graph per run, walking upstream from the result:

    result  ◄─CAUSED─  gate verdict  ◄─CAUSED─  submission  ─uses─►  evidence spans
    (deterministic_runtime)  (rule_engine)        (model)
                                                      ▲
                                    step_N  ─INFLUENCED─┘
                                      ▲
              step_1 ─INFLUENCED─ ... ┘        (the model's note_decision points, in order)

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol



class ReviewLedger(Protocol):
    def record_evidence(self, run_id: str, index: int, ev: dict[str, Any]) -> str: ...
    def record_judgment(self, run_id: str, *, category: str, scenario: str, reasoning: str,
                        outcome: str, decision_maker: str,
                        entities: list[str] | None = None,
                        metadata: dict[str, Any] | None = None) -> str: ...
    def link_caused(self, source_id: str, target_id: str) -> None: ...
    def link_influenced(self, source_id: str, target_id: str) -> None: ...
    def link_part_of(self, child_id: str, parent_id: str) -> None: ...
    def link_uses(self, judgment_id: str, evidence_id: str) -> None: ...
    def result_node(self, case_id: str) -> str | None: ...
    def chain(self, case_id: str) -> list[dict[str, Any]]: ...
    def decisions(self, *, category_prefix: str | None = None,
                  case_id: str | None = None) -> list[dict[str, Any]]: ...
    def save(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


class NullLedger:
    """Counts what it is asked to record and remembers nothing else."""

    def __init__(self) -> None:
        self.counts = {"evidence": 0, "judgments": 0, "edges": 0}

    def record_evidence(self, run_id: str, index: int, ev: dict[str, Any]) -> str:
        self.counts["evidence"] += 1
        return f"ev:{run_id}:{index}"

    def record_judgment(self, run_id: str, **kw: Any) -> str:
        self.counts["judgments"] += 1
        return f"j:{run_id}:{self.counts['judgments']}"

    def link_caused(self, source_id: str, target_id: str) -> None:
        self.counts["edges"] += 1

    def link_influenced(self, source_id: str, target_id: str) -> None:
        self.counts["edges"] += 1

    def link_part_of(self, child_id: str, parent_id: str) -> None:
        self.counts["edges"] += 1

    def link_uses(self, judgment_id: str, evidence_id: str) -> None:
        self.counts["edges"] += 1

    def result_node(self, case_id: str) -> str | None:
        return None

    def chain(self, case_id: str) -> list[dict[str, Any]]:
        return []

    def decisions(self, *, category_prefix: str | None = None,
                  case_id: str | None = None) -> list[dict[str, Any]]:
        return []

    def save(self) -> None:
        pass

    def stats(self) -> dict[str, Any]:
        return dict(self.counts)


class SemanticaLedger:
    """semantica 0.6.6's ContextGraph as the account book, JSON-persisted at `path`.

    We keep NO parallel store. `save_to_file` writes only nodes, edges and links, so after a
    reload semantica's own decision registry (`_decisions`, and the category and entity
    indexes over it) is empty and every decision-side API — `find_precedents_by_scenario`,
    `get_decision_insights` — answers as though nothing was ever recorded. The fix is to give
    those APIs their data back rather than to hand-roll replacements for them: `_rehydrate`
    replays the persisted decision nodes, and the `involves` edges that carry their entities,
    into semantica's registry at construction. Everything downstream is then semantica's.
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
        self.graph = ContextGraph(
            extract_entities=False, extract_relationships=False, advanced_analytics=False,
            centrality_analysis=False, community_detection=False, node_embeddings=False,
        )
        if self.path.exists():
            self.graph.load_from_file(str(self.path))
            self._rehydrate()

    def _rehydrate(self) -> None:
        """Put the persisted decisions back into semantica's own registry.

        Reconstructed from what the file DOES keep: each decision node's metadata carries the
        four fields plus whatever we attached, and its `involves` edges name the entities it
        was recorded against. Timestamps come back from the node, so precedent ordering and
        `as_of` filtering survive a reload too."""
        # semantica creates its registry lazily, on the first record_decision; a ledger that
        # is only ever read (an audit over yesterday's runs) would otherwise never have one.
        if not hasattr(self.graph, "_decisions"):
            self.graph._decisions = {}
            self.graph._decision_index = defaultdict(set)
            self.graph._entity_index = defaultdict(set)
            self.graph._temporal_index = []
        entities: dict[str, list[str]] = {}
        for edge in self.graph.edges:
            e = edge.to_dict()
            if e.get("type") == "involves":
                entities.setdefault(e["source_id"], []).append(e["target_id"])
        for node in self.graph.find_nodes(node_type="decision"):
            meta = dict(node.get("metadata") or {})
            known = ("category", "scenario", "reasoning", "outcome", "confidence",
                     "decision_maker", "timestamp", "valid_from", "valid_until")
            record = {"id": node["id"],
                      **{k: meta.get(k) for k in known},
                      "entities": sorted(entities.get(node["id"], [])),
                      "recorded_at": meta.get("recorded_at"),
                      "metadata": {k: v for k, v in meta.items() if k not in known}}
            self.graph._decisions[node["id"]] = record
            self.graph._decision_index[record["category"]].add(node["id"])
            for entity in record["entities"]:
                self.graph._entity_index[entity].add(node["id"])
            self.graph._temporal_index.append((node["id"], record["timestamp"] or 0))
        self.graph._temporal_index.sort(key=lambda x: x[1], reverse=True)

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
                        entities: list[str] | None = None,
                        metadata: dict[str, Any] | None = None) -> str:
        # semantica refuses empty strings; a judgment with no stated reasoning is still a
        # judgment, and "(none given)" keeps that fact readable instead of failing ingestion.
        return self.graph.record_decision(
            category=category, scenario=(scenario or "(none given)")[:500],
            reasoning=(reasoning or "(none given)")[:500],
            outcome=outcome, confidence=1.0, decision_maker=decision_maker,
            entities=entities or None,
            metadata={"run_id": run_id, **(metadata or {})},
        )

    def link_caused(self, source_id: str, target_id: str) -> None:
        self.graph.add_causal_relationship(source_id, target_id, "CAUSED")

    def link_influenced(self, source_id: str, target_id: str) -> None:
        # The weaker causal verb, for the model's own decision points: step N shaped step N+1
        # and the eventual submission, but did not determine them the way the gate's verdict
        # determines the result.
        self.graph.add_causal_relationship(source_id, target_id, "INFLUENCED")

    def link_part_of(self, child_id: str, parent_id: str) -> None:
        # Composition, not causation. `get_causal_chain` walks only CAUSED / INFLUENCED /
        # PRECEDENT_FOR, so a small point hangs off its big point without appearing on the
        # audit chain — which is what makes the chain readable: it shows the conclusions, and
        # the steps that reached each one are one query away rather than in the way.
        self.graph.add_edge(source_id=child_id, target_id=parent_id, edge_type="PART_OF")

    def link_uses(self, judgment_id: str, evidence_id: str) -> None:
        self.graph.add_edge(source_id=judgment_id, target_id=evidence_id, edge_type="uses")

    def result_node(self, case_id: str) -> str | None:
        """The audit anchor: this case's result judgment, found by walking the graph rather
        than a sidecar index — the case is an entity, and the category says which node it is."""
        return next((n["id"] for n in self.graph.find_nodes(node_type="decision")
                     if str((n.get("metadata") or {}).get("category", "")).startswith("result:")
                     and (n.get("metadata") or {}).get("case_id") == case_id), None)

    def parts_of(self, parent_id: str) -> list[dict[str, Any]]:
        """The small points that made up one big point, in the order they happened."""
        ids = [e.to_dict()["source_id"] for e in self.graph.edges
               if e.to_dict().get("type") == "PART_OF"
               and e.to_dict()["target_id"] == parent_id]
        rows = [r for r in self.decisions() if r["decision_id"] in set(ids)]
        return sorted(rows, key=lambda r: r.get("seq") or 0)

    #: semantica scores a precedent `0.7 * content + 0.3 * structural`, and the structural
    #: half needs `advanced_analytics`, which stays off here for the PHI posture. So the score
    #: is capped at 0.7 and its own 0.5 default is unreachable: two decisions facing the
    #: IDENTICAL situation measure 0.467 on content (its similarity mixes `reasoning` and the
    #: entity list into the text, and reasoning is precisely the field meant to differ between
    #: two decisions facing the same situation), which lands at 0.327 combined and is
    #: rejected. `min_content` below is therefore stated as a floor on CONTENT similarity and
    #: converted; measured on real ledgers, an unrelated situation scores ~0.07 and an
    #: identical one ~0.47, so 0.3 separates them with room on both sides.
    _STRUCTURAL_UNAVAILABLE = 0.7

    def precedents(self, scenario: str, *, category: str | None = None,
                   entities: list[str] | None = None, limit: int = 10,
                   min_content: float = 0.3) -> list[dict[str, Any]]:
        """semantica's own precedent search, which answers here because `_rehydrate` gave it
        back its data. This is the runtime lookup — "how was this situation decided before".

        `precipitate` deliberately does NOT use it: that verb reports to a human, and it
        clusters on `scenario` alone so that two runs which faced the same situation and
        reasoned differently land in the same group instead of being split by the very field
        whose disagreement is the finding."""
        kw: dict[str, Any] = {"category": category, "limit": limit,
                              "similarity_threshold": min_content * self._STRUCTURAL_UNAVAILABLE}
        if entities:
            kw["entities"] = entities
        return self.graph.find_precedents_by_scenario(scenario, **kw)

    def decisions(self, *, category_prefix: str | None = None,
                  case_id: str | None = None) -> list[dict[str, Any]]:
        """The compare verb's raw material: every decision node matching the filters, e.g.
        category_prefix="step:coverage" for all coverage decisions across every run."""
        rows = []
        for n in self.graph.find_nodes(node_type="decision"):
            meta = n.get("metadata") or {}
            if category_prefix and not str(meta.get("category", "")).startswith(category_prefix):
                continue
            if case_id and meta.get("case_id") != case_id and case_id not in str(meta.get("scenario", "")):
                continue
            rows.append({"decision_id": n["id"],
                         **{k: meta.get(k) for k in
                            ("category", "scenario", "outcome", "decision_maker", "reasoning",
                             "case_id", "spec_id", "seq", "run_id", "options", "context",
                             "claimed_type", "used", "used_unverified")}})
        # Same seq = same tool call (a submission and its gate verdict): break the tie by
        # causal rank, so two ledgers of the same run list identically regardless of uuids.
        rank = {"small": 0, "step": 1, "big": 2, "submit": 3, "gate": 4, "result": 5}
        rows.sort(key=lambda r: (str(r.get("run_id")), r.get("seq") or 0,
                                 rank.get(str(r.get("category", "")).split(":")[0], 9)))
        return rows

    def chain(self, case_id: str) -> list[dict[str, Any]]:
        """The audit verb: result ← gate ← submission(s), upstream from the case's result.

        The anchor comes from `find_node` because `get_causal_chain` returns only the nodes
        upstream OF the anchor, never the anchor itself — an audit that omitted the verdict
        being audited would read as a chain to nowhere."""
        result_id = self.result_node(case_id)
        if result_id is None:
            return []
        keys = ("decision_id", "category", "scenario", "outcome", "decision_maker")
        rows: list[dict[str, Any]] = []
        anchor = self.graph.find_node(result_id)
        if anchor:
            meta = anchor.get("metadata") or {}
            rows.append({"decision_id": anchor.get("id"), "distance": 0,
                         **{k: meta.get(k) for k in keys[1:]}})
        # Depth 50, not 10: with note_decision steps chained by INFLUENCED, a fine-grained run
        # is result <- gate <- submission <- step_N <- ... <- step_1, one hop per step.
        for d in self.graph.get_causal_chain(result_id, direction="upstream", max_depth=50):
            row = d if isinstance(d, dict) else getattr(d, "__dict__", {"repr": repr(d)})
            meta = row.get("metadata") or {}
            rows.append({**{k: row.get(k) for k in keys},
                         "distance": meta.get("causal_distance"), "seq": meta.get("seq")})
        rows.sort(key=lambda r: (r.get("distance") or 0, -(r.get("seq") or 0)))
        return rows

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save_to_file(str(self.path))

    def stats(self) -> dict[str, Any]:
        insights = self.graph.get_decision_insights()
        cases = {str((n.get("metadata") or {}).get("case_id"))
                 for n in self.graph.find_nodes(node_type="decision")
                 if (n.get("metadata") or {}).get("case_id")}
        return {**self.graph.stats(), "cases": len(cases),
                "total_decisions": insights.get("total_decisions", 0),
                "categories": insights.get("categories", {})}


# ------------------------------------------------------------------- the one decomposition
class RunRecorder:
    """What counts as a judgment and which edges connect them — stated once, driven twice.

    The toolserver drives it live, one call per event as the review happens; `ingest_run`
    drives it by replaying a finished trace. Both produce the identical graph, which is what
    lets the trace stay authoritative while the ledger hears things at decision time."""

    def __init__(self, ledger: ReviewLedger, *, run_id: str, spec_id: str, case_id: str) -> None:
        self.ledger = ledger
        self.run_id, self.spec_id, self.case_id = run_id, spec_id, case_id
        self.evidence_ids: list[str] = []
        self.prev_step_id: str | None = None
        self.last_gate_id: str | None = None
        self.n_submissions = 0
        self.n_steps = 0

    def _common_meta(self, seq: Any) -> dict[str, Any]:
        return {"spec_id": self.spec_id, "case_id": self.case_id, "seq": seq}

    def _entities(self, *extra: str) -> list[str]:
        """What this judgment is ABOUT, in semantica's own entity vocabulary. These become
        entity nodes with `involves` edges, so "every decision touching this case" and "every
        decision that cited this rule" are graph queries rather than a sidecar we maintain."""
        return [f"case:{self.case_id}", f"spec:{self.spec_id}",
                *dict.fromkeys(e for e in extra if e)]

    def on_evidence(self, args: dict[str, Any], quote: str) -> None:
        self.evidence_ids.append(self.ledger.record_evidence(
            self.run_id, len(self.evidence_ids), {**args, "quote": quote}))

    def on_step(self, args: dict[str, Any], seq: Any,
                context: dict[str, Any] | None = None,
                used: list[dict[str, Any]] | None = None,
                grounding: list[str] | None = None) -> None:
        """A note_decision: the semantica frame verbatim (facing = scenario, because =
        reasoning, decision = outcome). Deliberately UNCLASSIFIED — the category is the bare
        `step`, because what KIND of judgment this was is decided afterwards, by
        `acr.mvp.reconstruct`, against a taxonomy still being grown from real runs.

        The inputs ride along resolved, and the grounding as the model reported it. Those two
        are collected here and nowhere else: a later reader can see which documents a run
        opened, so it would mark every citation verified and every judgment contract-grounded.
        Only the model can say it leaned on a document it never read, or on its own clinical
        knowledge."""
        self.n_steps += 1
        refs = [str(u.get("ref")) for u in (used or [])]
        unverified = [str(u.get("ref")) for u in (used or []) if u.get("verified") is False]
        step = self.ledger.record_judgment(
            self.run_id, category="step",
            scenario=str(args.get("facing") or ""),
            reasoning=str(args.get("because") or ""),
            outcome=str(args.get("decision") or ""), decision_maker="model",
            entities=self._entities(*refs[:20]),
            metadata={**self._common_meta(seq), "options": args.get("options"),
                      "context": context, "grounding": grounding or [],
                      "used": refs[:20], "used_unverified": unverified[:20]})
        if self.prev_step_id is not None:
            self.ledger.link_influenced(self.prev_step_id, step)
        self.prev_step_id = step

    def on_submission(self, args: dict[str, Any], verdict: dict[str, Any], seq: Any) -> None:
        self.n_submissions += 1
        submission = self.ledger.record_judgment(
            self.run_id, category=f"submit:{self.spec_id}",
            scenario=f"case {self.case_id}, submission {self.n_submissions}",
            reasoning=str(args.get("reasoning") or ""),
            outcome=str(args.get("status")), decision_maker="model",
            entities=self._entities(),
            metadata={**self._common_meta(seq), "value": args.get("value")})
        for ev_id in self.evidence_ids:
            self.ledger.link_uses(submission, ev_id)
        if self.prev_step_id is not None:
            self.ledger.link_influenced(self.prev_step_id, submission)
        outcome = "accepted" if verdict.get("accepted") else f"refused: {verdict.get('why')}"
        gate = self.ledger.record_judgment(
            self.run_id, category=f"gate:{self.spec_id}",
            scenario=f"case {self.case_id}, submission {self.n_submissions}",
            reasoning=str(verdict.get("why") or ""),
            outcome=outcome, decision_maker="rule_engine",
            entities=self._entities(), metadata=self._common_meta(seq))
        self.ledger.link_caused(submission, gate)
        self.last_gate_id = gate

    def on_result(self, final: dict[str, Any]) -> str:
        result_node = self.ledger.record_judgment(
            self.run_id, category=f"result:{self.spec_id}", scenario=f"case {self.case_id}",
            reasoning=str(final.get("reasoning") or final.get("why") or ""),
            outcome=str(final.get("status", "NO_ANSWER")),
            decision_maker="deterministic_runtime",
            metadata={"spec_id": self.spec_id, "case_id": self.case_id,
                      "value": final.get("value")},
            entities=self._entities())
        if self.last_gate_id is not None:
            self.ledger.link_caused(self.last_gate_id, result_node)
        return result_node

    def summary(self, final_status: str) -> dict[str, Any]:
        return {"run_id": self.run_id, "case_id": self.case_id,
                "n_evidence": len(self.evidence_ids), "n_steps": self.n_steps,
                "n_submissions": self.n_submissions, "result": final_status}


# ---------------------------------------------------------------------------- replay path
def ingest_run(run_dir: Path, ledger: ReviewLedger) -> dict[str, Any]:
    """Replay one finished run's Layer-1 trace through the recorder. For runs that had no
    live ledger, or to rebuild one — replay into a FRESH ledger, not one that already heard
    this run live. The run directory name is the run identity."""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    events = [json.loads(line)
              for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    meta = next((e for e in events if e.get("kind") == "run_meta"), {})
    rec = RunRecorder(ledger, run_id=run_id, spec_id=meta.get("spec_id", "unknown"),
                      case_id=meta.get("patient_id", run_id))
    for e in events:
        if e.get("kind") != "tool_call":
            continue
        tool, args, result = e.get("tool"), e.get("args") or {}, e.get("result") or {}
        if tool == "record_evidence" and e.get("ok"):
            rec.on_evidence(args, result.get("quote", ""))
        elif tool == "note_decision" and e.get("ok"):
            rec.on_step(args, e.get("seq"), context=result.get("context"),
                        used=result.get("used"), grounding=result.get("grounding"))
        elif tool == "submit_answer":
            rec.on_submission(args, result, e.get("seq"))

    result_path = run_dir / "result.json"
    final = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    rec.on_result(final)
    ledger.save()
    return rec.summary(final.get("status", "NO_ANSWER"))
