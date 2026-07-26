"""Run tracing.

Everything the agent does is appended to a JSONL trace: plan revisions, every tool call
with full input/output, every reflection verdict, every rejected answer, and the final
coverage attestation. The trace is the artifact that makes a label auditable — without it
you cannot tell a correct answer from a lucky one.

`to_capg()` reshapes the trace into the observation-tree form the CAPG adapter consumes,
so these runs drop straight into an existing provenance-graph pipeline.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Tracer:
    run_id: str
    path: Path
    events: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    @classmethod
    def create(cls, out_dir: str | Path, run_id: str | None = None) -> "Tracer":
        rid = run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        return cls(run_id=rid, path=d / f"{rid}.jsonl")

    def emit(self, kind: str, **payload: Any) -> dict:
        ev = {
            "run_id": self.run_id,
            "seq": len(self.events),
            "ts": _now(),
            "elapsed_s": round(time.time() - self.t0, 3),
            "kind": kind,
            **payload,
        }
        self.events.append(ev)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        return ev

    # convenience emitters ------------------------------------------------------
    def run_start(self, **kw): return self.emit("run_start", **kw)
    def plan(self, plan, revision, rationale=""): return self.emit("plan", plan=plan, revision=revision, rationale=rationale)
    def llm(self, role, content, tool_calls=None, usage=None):
        return self.emit("llm", role=role, content=content, tool_calls=tool_calls or [], usage=usage or {})
    def tool(self, name, args, result, ok=True, ms=0.0):
        return self.emit("tool", tool=name, args=args, result=result, ok=ok, ms=ms)
    def reflect(self, verdict, reason, evidence_count):
        return self.emit("reflect", verdict=verdict, reason=reason, evidence_count=evidence_count)
    def rejected(self, why, missing, attempted):
        return self.emit("answer_rejected", why=why, missing=missing, attempted=attempted)
    def run_end(self, **kw): return self.emit("run_end", **kw)

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
                obs.append({"id": oid, "name": "write_todos", "type": "EVENT",
                            "parent_observation_id": root, "start_time": ev["ts"],
                            "output": {"todos": [{"content": s.get("goal", ""), "status": s.get("status", "")}
                                                 for s in ev.get("plan", [])]}})
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


def load_trace(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
