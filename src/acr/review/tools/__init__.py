"""The agent's instrument panel.

These are the moves a human abstractor makes: see what documents exist and when, scan by
type, grep for terms, open a document, jump to a section, lay events on a timeline, and
write down a quote as evidence. Nothing here is clever — the cleverness is supposed to be
in the agent's sequencing of them, and that sequencing is what the trace captures.

The toolbox owns the ledgers, so coverage is recorded as a side effect of real tool use
rather than being self-reported by the model.
"""
from .toolbox import TOOL_SCHEMAS, Toolbox

__all__ = ["TOOL_SCHEMAS", "Toolbox"]
