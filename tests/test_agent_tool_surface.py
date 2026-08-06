"""What the model may reach. A whitelist, and a test that fails when a library widens it.

`create_deep_agent` injects nine tools nobody asked for — ls, glob, grep, read_file,
write_file, edit_file, execute, task, write_todos. Four are read paths, and a read that does
not go through `Toolbox.dispatch` is invisible to the `CoverageLedger`, so the gate would
still stamp `gate_validated: true` over a chart the ledger never saw read. Under the
`FilesystemBackend(root_dir=".")` the skills path uses, `read` and `grep` reach ABSOLUTE paths
outside root_dir — including the registry answer key. No recorded run exercised it, but an
unexercised open door is not a boundary.

These tests are cheap and they are the boundary. They do not call a provider.
"""
from __future__ import annotations

import pytest

from acr.core.tool_surface import LIBRARY_TOOLS, ToolSurfaceError, assert_tool_surface


class _FakeTool:
    def __init__(self, name): self.name = name


class _FakeAgent:
    """Only what `assert_tool_surface` reads, so the check is tested and not the framework."""

    def __init__(self, names):
        bound = type("B", (), {"_tools_by_name": {n: _FakeTool(n) for n in names}})()
        self.nodes = {"tools": type("N", (), {"bound": bound})()}


OURS = {"list_documents", "search_notes", "read_document", "record_evidence", "submit_answer"}


def test_our_tools_plus_the_sanctioned_library_tool_pass():
    bound = assert_tool_surface(_FakeAgent(OURS | {"write_todos"}), OURS)
    assert bound == OURS | {"write_todos"}


def test_a_filesystem_write_tool_is_refused():
    """`write_file`, not `read_file`. Read paths were admitted on 2026-08-06 because the backend
    is `StateBackend` and there is no filesystem behind them; a WRITE tool is refused whatever the
    backend, because nothing in a chart review has any business writing."""
    with pytest.raises(ToolSurfaceError) as e:
        assert_tool_surface(_FakeAgent(OURS | {"write_file"}), OURS)
    assert "write_file" in str(e.value)
    assert "coverage ledger" in str(e.value), "the refusal must say WHY, not just that"


def test_shell_execution_is_refused():
    with pytest.raises(ToolSurfaceError):
        assert_tool_surface(_FakeAgent(OURS | {"execute"}), OURS)


def test_every_tool_create_deep_agent_would_have_injected_is_refused():
    """Named individually so the message points at the one that appeared."""
    # `ls` and `read_file` left this list on 2026-08-06 — see `test_the_sanctioned_set_is_small`.
    # `task` stays, and it is the one that matters: it spawns work outside the coverage ledger.
    for name in ("glob", "grep", "write_file", "edit_file", "execute", "task"):
        with pytest.raises(ToolSurfaceError, match=name):
            assert_tool_surface(_FakeAgent(OURS | {name}), OURS)


def test_the_guard_is_a_whitelist_not_a_blacklist():
    """A blacklist of today's nine passes the tenth. This is the property under test."""
    with pytest.raises(ToolSurfaceError, match="tool_invented_next_release"):
        assert_tool_surface(_FakeAgent(OURS | {"tool_invented_next_release"}), OURS)


def test_the_sanctioned_set_is_small_and_says_why_each_member_is_in_it():
    """Widening `LIBRARY_TOOLS` is a decision about the coverage ledger, so it stays small and the
    reasoning stays beside it.

    `read_file` and `ls` were admitted when `build_agent` moved onto `create_deep_agent`:
    progressive disclosure needs the model to open a `SKILL.md`, and under `StateBackend` those
    two reach only what the run seeded into state — never a chart document, which is reachable
    solely through `Toolbox`. That admission is CONDITIONAL ON THE BACKEND. Under a
    `FilesystemBackend` the same two tools reach absolute paths outside the root, so this test
    also pins the condition: if the backend ever changes, this set must be re-argued."""
    assert LIBRARY_TOOLS == frozenset({"write_todos", "read_file", "ls"})

    import inspect

    from acr.review import agent as A
    src = inspect.getsource(A.build_agent)
    assert "StateBackend" in src or "backend=backend" in src, (
        "the read-path admission above is only sound while the agent's backend has no filesystem "
        "behind it")


def test_a_missing_tool_is_not_this_check_s_business():
    """Under-provisioning is a different failure; this guard must not silently also police it."""
    assert assert_tool_surface(_FakeAgent({"submit_answer"}), OURS) == {"submit_answer"}
