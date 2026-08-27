# CLAUDE.md

Engineering rules for Claude Code developing this repository.

## Rules

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility
  layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative
  abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each
  new capability on top of a product that already works. Never trade a working product for
  unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve
  reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding
  packages. Do not assume a library lacks a capability without checking its documentation and
  types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now
  and is meant to be replaced later.

## Provenance

Transcribed from a screenshot of an `AGENTS.md` circulated on Threads (2026-08-05), attributed there
to the Vercel team and described as the residue of 60B tokens of use. Seven rules are legible in
that screenshot; the post says eight, and the capture appears to be cut at the top, so one may be
missing. If the original surfaces, reconcile against it rather than inventing the eighth.

## Where else to look

[`CONTEXT.md`](CONTEXT.md) defines the maintained language around runtime testimony, ReAct cycles,
Decision Episodes, Semantica decisions, policy bindings, provenance, and human adjudication.
[`README.md`](README.md) is the entry point; [`docs/CORE_STORY.md`](docs/CORE_STORY.md) explains the
design and [`docs/RUNBOOK.md`](docs/RUNBOOK.md) contains the operational workflow.
