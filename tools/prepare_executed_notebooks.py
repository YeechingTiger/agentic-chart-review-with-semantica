"""Turn executed notebooks into compact, checked reading copies for GitHub.

The runnable ``.ipynb`` files retain every code cell.  Their ``.executed.ipynb`` companions retain
the saved outputs and a hash of each corresponding source cell, but omit the implementation text
so the static reading path is not dominated by notebook plumbing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NAMES = (
    "01_run_chart_review_experiments",
    "02_trace_to_decision_chain",
    "03_semantica_decision_intelligence",
)


def _source_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare(name: str) -> Path:
    source_path = NOTEBOOKS / f"{name}.ipynb"
    executed_path = NOTEBOOKS / f"{name}.executed.ipynb"
    source = nbformat.read(source_path, as_version=4)
    executed = nbformat.read(executed_path, as_version=4)
    executed.cells = [
        cell for cell in executed.cells
        if not (cell.get("metadata") or {}).get("acr_reading_copy_note")
    ]
    if len(source.cells) != len(executed.cells):
        raise ValueError(f"{name}: source and executed notebook cells do not align")

    for index, (source_cell, executed_cell) in enumerate(zip(source.cells, executed.cells)):
        if source_cell.cell_type != executed_cell.cell_type:
            raise ValueError(f"{name}: cell {index} changed type during execution")
        if source_cell.cell_type != "code":
            continue
        metadata = executed_cell.setdefault("metadata", {})
        expected_hash = _source_hash(source_cell.source)
        already_prepared = (
            executed_cell.source == ""
            and metadata.get("acr_source_cell_sha256") == expected_hash
        )
        if executed_cell.source != source_cell.source and not already_prepared:
            raise ValueError(f"{name}: cell {index} source changed during execution")
        metadata["acr_source_cell_sha256"] = expected_hash
        metadata["jupyter"] = {"source_hidden": True}
        metadata["tags"] = ["hide-input"]
        executed_cell.source = ""

    note = new_markdown_cell(
        "**Checked reading copy.** The saved outputs below come from completed real-provider "
        f"runs. Implementation cells are omitted here for readability; open "
        f"[`{name}.ipynb`](./{name}.ipynb) to inspect or rerun the code.\n",
        metadata={"acr_reading_copy_note": True},
    )
    executed.cells.insert(1, note)
    acr = executed.metadata.setdefault("acr", {})
    acr.update({
        "reading_copy": True,
        "source_notebook": f"{name}.ipynb",
        "code_inputs_omitted": True,
    })
    nbformat.write(executed, executed_path)
    return executed_path


def main() -> None:
    for name in NAMES:
        print(prepare(name))


if __name__ == "__main__":
    main()
