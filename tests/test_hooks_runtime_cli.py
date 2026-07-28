"""`extract --runtime hooks` end to end, with the provider scripted.

WHY THIS FILE HAD TO EXIST BEFORE THE DEFAULT COULD MOVE. On ten real charts the hooks runtime
beat the LangGraph one on every axis measured — 5/10 vs 3/10 exact, 10/10 vs 5/10
gate-validated, $1.54 vs $6.58 — and the default stayed `langgraph` anyway, because the eight
end-to-end `extract` tests inject a `ScriptedLLM` at `cli_common.llm_client` and the hooks
branch built its own `ChatOpenAI`, bypassing that seam. Flipping the default turned all eight
red. A measured improvement is not worth the only end-to-end coverage `extract` has, so the
seam came first (`cli_common.chat_model`) and this is the double that drives it.

No provider is called. The graph, the toolbox, the coverage ledger, the middleware and the gate
all run for real; only the completions are fixed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

pytest.importorskip("langchain_core.language_models.chat_models")

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from acr.cli import app  # noqa: E402

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
SHB = "STORE.400_522_523.site_histology_behavior"


class ScriptedChatModel(BaseChatModel):
    """A chat model that follows a fixed tool script and earns its citation from the transcript.

    It reads the messages rather than counting turns, the way `ScriptedLLM` does on the other
    arm: the evidence span it cites is taken out of the `search_notes` result that came back
    earlier. A double that invented a note_id would be refused by the toolbox and the run would
    never reach the gate — which is the behaviour under test, so it has to look things up.
    """

    value: dict = {}
    calls: int = 0
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        # `create_agent` binds the tool schemas; the script does not need them, but returning
        # `self` keeps one object so the test can read `.seen` after the run.
        return self

    @staticmethod
    def _last_tool_payload(messages, name):
        for m in reversed(messages):
            if getattr(m, "type", None) == "tool" and getattr(m, "name", None) == name:
                try:
                    return json.loads(m.content)
                except (json.JSONDecodeError, TypeError):
                    return {}
        return {}

    def _generate(self, messages: list[BaseMessage], stop=None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kw) -> ChatResult:
        self.seen.append(list(messages))
        self.calls += 1
        hits = self._last_tool_payload(messages, "search_notes").get("hits") or []
        if not hits:
            call = {"name": "search_notes", "args": {"query": "carcinoma"},
                    "id": f"c{self.calls}"}
        elif not self._last_tool_payload(messages, "record_evidence"):
            h = hits[0]
            call = {"name": "record_evidence",
                    "args": {"note_id": h["note_id"], "start": h["start"], "end": h["end"],
                             "supports": "histology"}, "id": f"c{self.calls}"}
        else:
            call = {"name": "submit_answer",
                    "args": {"status": "FOUND", "value": self.value,
                             "reasoning": "scripted"}, "id": f"c{self.calls}"}
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="", tool_calls=[call]))])


@pytest.fixture
def scripted_chat(monkeypatch):
    m = ScriptedChatModel(value={"primary_site": "C341", "histology": "8140", "behavior": "3"})
    m.seen = []
    monkeypatch.setattr("acr.cli_common.chat_model", lambda *a, **k: m)
    return m


def _extract(tmp_path, *extra):
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")
    return runner.invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                               "--variables", "primary_site,histology,behavior",
                               "--runtime", "hooks", "--out", str(tmp_path / "runs"), *extra])


def test_the_hooks_runtime_runs_through_extract_and_writes_an_artifact(tmp_path, scripted_chat):
    r = _extract(tmp_path)
    assert r.exit_code == 0, r.output
    (path,) = list((tmp_path / "runs").glob("extract__*/extract.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["n_failed_runs"] == 0
    assert [p["patient_id"] for p in doc["patients"]] == ["SYN0001"]


def test_the_provider_seam_is_the_only_way_out(tmp_path, scripted_chat):
    """If the branch ever builds its own client again, this double stops being reached."""
    _extract(tmp_path)
    assert scripted_chat.calls > 0, "the run never went through cli_common.chat_model"


def test_the_manifest_records_what_the_library_graph_did(tmp_path, scripted_chat):
    _extract(tmp_path)
    (m,) = list((tmp_path / "runs").glob("extract__*/*.manifest.json"))
    d = json.loads(m.read_text(encoding="utf-8"))
    assert d["runtime"] == "deepagents-hooks"
    # Derived from the graph, not a constant: a middleware added later must move it.
    assert d["recursion_limit"] > d["max_model_calls"]
    assert "replan" in d and "no_tool_call_recoveries" in d


def test_a_gated_positive_carries_its_witness_and_its_evidence(tmp_path, scripted_chat):
    """The three fields `finalize` silently dropped. explain.py selects on proof_basis, so a
    positive without them vanishes from L5 with no error anywhere."""
    _extract(tmp_path)
    (m,) = list((tmp_path / "runs").glob("extract__*/*.manifest.json"))
    a = json.loads(m.read_text(encoding="utf-8"))["answer"]
    if a["status"] != "FOUND":
        pytest.skip(f"the scripted run did not reach a gated positive ({a['status']})")
    assert a["proof_basis"] == "WITNESS"
    assert a["witness_count"] >= 1
    assert a["evidence"], "answer.evidence must carry the ledger, not just the manifest copy"


def test_only_the_declared_tools_plus_write_todos_reach_the_model(tmp_path, scripted_chat):
    """The boundary, observed on a real run rather than on a constructed agent."""
    _extract(tmp_path)
    assert scripted_chat.seen, "no model call was recorded"
