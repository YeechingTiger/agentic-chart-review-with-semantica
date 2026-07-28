"""What the model may reach, and the check that refuses anything else.

SEPARATE FROM `agent.py` FOR A TESTABILITY REASON, NOT A TIDINESS ONE. `agent.py` must import
langchain at module scope (`AuditMiddleware` subclasses `AgentMiddleware`), the test suite runs
in a venv without langchain, and `.venv-deep` has no pytest — so a boundary check living in
`agent.py` was skipped in one environment and uninvokable in the other. It ran nowhere. That
is the same defect this repo keeps producing: a check that cannot fail. The policy has no
framework dependency, so it does not need to sit behind one.

WHAT THE POLICY IS FOR. `create_deep_agent` injects nine tools nobody asked for: ls, glob,
grep, read_file, write_file, edit_file, execute, task, write_todos. Four are read paths, and a
read that does not go through `Toolbox.dispatch` is invisible to the `CoverageLedger` — the
gate would still stamp `gate_validated: true` over a chart the ledger never saw read. Under the
`FilesystemBackend(root_dir=".")` the skills path uses, `read` and `grep` reach ABSOLUTE paths
outside root_dir, the answer key among them. No recorded run exercised that; an unexercised
open door is still not a boundary.
"""
from __future__ import annotations

#: The only tool the library is permitted to add. A set, not a count: a future version that
#: swaps `write_todos` for two differently-named tools must fail loudly, not arithmetically.
LIBRARY_TOOLS = frozenset({"write_todos"})


class ToolSurfaceError(AssertionError):
    """The agent was built carrying a tool nobody in this repo declared."""


def bound_tool_names(agent) -> set[str]:
    """The names actually bound to the compiled graph's tool node."""
    return {t.name for t in agent.nodes["tools"].bound._tools_by_name.values()}


def assert_tool_surface(agent, declared: set[str]) -> set[str]:
    """Refuse an agent carrying a tool we did not pass and did not sanction.

    A WHITELIST, deliberately. The failure this guards is not a tool someone added on purpose
    — it is a library upgrade quietly widening what the model can reach, which is how nine
    filesystem and shell tools arrived the first time. A blacklist of today's nine would have
    passed the tenth.
    """
    bound = bound_tool_names(agent)
    unexpected = bound - set(declared) - LIBRARY_TOOLS
    if unexpected:
        raise ToolSurfaceError(
            f"the agent carries {sorted(unexpected)}, which this repo never declared. If a "
            f"library upgrade added them, decide whether each may bypass the coverage ledger "
            f"BEFORE adding it to LIBRARY_TOOLS — a read that does not go through "
            f"Toolbox.dispatch is invisible to the gate.")
    return bound
