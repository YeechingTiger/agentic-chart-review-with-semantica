"""One key name for pseudonymisation, so fingerprints from two paths can be joined.

`audit_loop._fingerprint` read `ACR_PHI_FINGERPRINT_KEY` — a name that appeared in the whole tree
exactly once, on that line. Everything else uses `ACR_PSEUDONYM_KEY`: `evals.py` (four sites),
`assets/skills/audit-phi-in-trace/scan.py`, `README.md`, `tests/test_cli_eval_plane.py`.

A site that set the documented name got `<redacted:no-local-key>` for every finding from
`acr audit run`, so the report could not distinguish ONE identifier leaking forty times from FORTY
identifiers leaking once — which is the first triage question after the run that quoted
`MRN: … / Patient: … / DOB: …` whole into a submitted answer. And the skill's HMAC fingerprints over
the same trace never joined to the in-process ones, because they were keyed differently.
"""

from __future__ import annotations

import pathlib

from acr.audit import audit_loop
from acr.evaluation import evals


def test_both_paths_name_the_same_env_var():
    assert audit_loop.PSEUDONYM_KEY_ENV == evals.PSEUDONYM_KEY_ENV == "ACR_PSEUDONYM_KEY"


def test_the_same_string_fingerprints_identically_through_both(monkeypatch):
    """The joinability property. Two different keys produce two unrelated hex strings, and nothing
    in either report says they are about the same identifier."""
    monkeypatch.setenv("ACR_PSEUDONYM_KEY", "test-secret")
    value = "1234567890123"
    assert audit_loop._fingerprint(value) == evals.pseudonymise(value)


def test_no_code_reads_a_second_fingerprint_key_name():
    """A second name is a second keyspace.

    STRING LITERALS IN CODE, not a text grep: the retired name is still written in prose — in
    `core/site.py`'s comment and in this file's docstring — explaining why it must not come back,
    and a text grep would fail on its own explanation. `tools/verify_structure.py` learned the same
    lesson: its first finding was a card name in a DOCSTRING, which is not a consumer of it.
    """
    import ast
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted([*root.glob("src/**/*.py"), *root.glob("tools/**/*.py"),
                        *root.glob("assets/**/*.py")]):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        docstrings = {id(n.value) for n in ast.walk(tree)
                      if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "ACR_PHI_FINGERPRINT_KEY" in node.value
                    and id(node) not in docstrings):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"a second pseudonymisation key name is back in code: {offenders}"


def test_an_unset_key_still_redacts_rather_than_leaking(monkeypatch):
    monkeypatch.delenv("ACR_PSEUDONYM_KEY", raising=False)
    assert "no-local-key" in audit_loop._fingerprint("1234567890123")
