"""The top-level CLI file composes and decides nothing.

`cli.py` was 1206 lines reaching fifteen modules directly, which meant every command group
edited the same file and the entry point imported `graph`, `intake` and `explain` just to
print `--help`. These tests hold the property that made the split worth doing: a new command
touches ONE group module, and the mounting board stays a mounting board.

Nothing here invokes a model, reads a chart or writes outside tmp_path — it reads the import
graph out of the source with `ast`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import typer.main

from acr.cli import app

SRC = Path(__file__).resolve().parents[1] / "src" / "acr"
CLI_MODULES = sorted(p.stem for p in SRC.glob("cli*.py"))


def _first_party_imports(module: str) -> set[str]:
    """Sibling `acr.*` modules this one imports, however it spells the import."""
    known = {p.stem for p in SRC.glob("*.py")} | {p.name for p in SRC.iterdir() if p.is_dir()}
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module and node.module.split(".")[0] in known:
                out.add(node.module.split(".")[0])
            elif node.module is None:
                out |= {a.name for a in node.names if a.name in known}
        elif isinstance(node, ast.Import):
            out |= {a.name.split(".", 1)[1].split(".")[0] for a in node.names
                    if a.name.startswith("acr.")
                    and a.name.split(".", 1)[1].split(".")[0] in known}
    return out - {module}


def test_the_top_level_file_reaches_no_domain_module():
    """`cli.py` may import command-group modules and sub-app owners, and nothing else.

    `assetdev` and `derive` are the two exceptions and they are exceptions of the same kind:
    each already owns its Typer sub-app, so importing them IS importing a command group. What
    must never come back is `graph`, `intake`, `concordance`, `explain`, `spec`, `state`,
    `trace` or `llm` — a top-level file that reaches those is a top-level file with opinions.
    """
    allowed = set(CLI_MODULES) | {"assetdev", "derive"}
    assert _first_party_imports("cli") <= allowed


def test_no_cycle_among_the_command_group_modules():
    """A group importing the mounting board is how a monolith reassembles itself."""
    graph = {m: _first_party_imports(m) & set(CLI_MODULES) for m in CLI_MODULES}
    assert "cli" not in {d for deps in graph.values() for d in deps}, \
        "a command group imports cli.py; the composition must only go one way"
    colour: dict[str, int] = {}
    cycles: list[list[str]] = []

    def walk(node: str, stack: list[str]) -> None:
        colour[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if colour.get(nxt) == 1:
                cycles.append(stack[stack.index(nxt):] + [nxt])
            elif not colour.get(nxt):
                walk(nxt, stack)
        stack.pop()
        colour[node] = 2

    for m in CLI_MODULES:
        if not colour.get(m):
            walk(m, [])
    assert cycles == []


def test_every_command_group_states_one_responsibility_in_its_first_sentence():
    for m in CLI_MODULES:
        doc = ast.get_docstring(ast.parse((SRC / f"{m}.py").read_text(encoding="utf-8")))
        assert doc, f"{m} has no module docstring"
        first = doc.split("\n\n")[0].strip()
        assert first.endswith("."), f"{m}: the first paragraph is not one finished sentence"


def test_every_group_mounted_in_cli_is_reachable_from_the_command_line():
    """The only thing that can break in a mounting board is an app that was not mounted."""
    top = typer.main.get_command(app).commands
    for group in ("spec", "derive", "assets", "label", "refine", "eval", "judge"):
        assert group in top, f"{group} is not mounted"
        assert top[group].commands, f"{group} mounted with no commands"
    # The commands that were top-level before the split are still top-level after it.
    for name in ("patients", "chart", "specs", "run", "batch", "consistency", "trace",
                 "extract", "concord", "explain", "ask", "deps"):
        assert name in top, f"{name} stopped being a top-level command"
