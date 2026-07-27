"""The plumbing every `acr` command group shares: the console, the model options, run identity,
and the loader that checks an artifact is the kind of artifact it claims to be.

Split out of `cli.py` when that file reached 1206 lines and fifteen outbound edges. Nothing
here decides anything about a chart, a spec or a guideline; it is the handful of things every
group would otherwise keep its own copy of, and a second copy of `code_sha` is how two runs
of the same experiment end up recorded under different identities.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from .llm import LLMClient, LLMConfig

con = Console()

CORPUS = typer.Option("corpus/patients", "--corpus", help="root directory of patient directories")
MODEL = typer.Option(None, "--model", "-m", help="LiteLLM model string, e.g. ollama_chat/qwen3.6:35b")
API_BASE = typer.Option(None, "--api-base", help="override provider base URL (vLLM, proxy, …)")

#: The artifact contract of the L0-L5 chain. Named here rather than in the command that writes
#: each one, because the command that READS an artifact has to name the same string; two copies
#: of a schema tag drift, and a drifted tag turns `_load_artifact`'s guard into a nuisance
#: somebody deletes.
EXTRACT_SCHEMA = "acr.extract/1"
CONCORD_SCHEMA = "acr.concord/1"
EXPLAIN_SCHEMA = "acr.explain/1"


def code_sha() -> str:
    """Short git sha, or 'dirty'/'nogit'. A run is only reproducible against the code that
    produced it, so the code identity belongs in the run's name."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha or "nogit") + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def unique_run_dir(base: str) -> Path:
    """runs/<label>__<utc>__<sha>/ — never reused.

    Reusing a directory name across code versions silently replaces one experiment's record
    with another's, and nothing records that a substitution happened. The same configuration
    under different code is not the same experiment, so the sha is part of the identity.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(f"{base}__{stamp}__{code_sha()}")
    d.mkdir(parents=True, exist_ok=False)
    return d


def llm_client(model, api_base, temperature=0.0) -> LLMClient:
    """THE provider seam for every command group, reached as `cli_common.llm_client(...)`.

    Called through the module rather than imported by name, so that one monkeypatch here
    silences the provider for every command. When each group held its own `_llm`, a test that
    muzzled one of them left the others free to dial out.
    """
    return LLMClient(LLMConfig.from_env(model=model, api_base=api_base, temperature=temperature))


def load_artifact(path: str, schema: str) -> dict:
    """Read a pipeline artifact, refusing one that is not the stage it is being fed to."""
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"input not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    got = doc.get("schema")
    if got != schema:
        raise typer.BadParameter(f"{p}: expected schema {schema}, got {got!r}")
    return doc


def read_json(path: str | Path, what: str) -> object:
    """A caller-supplied JSON file, with the filename in the error.

    `json.JSONDecodeError` names a line and column and not the file, and a develop-plane
    command takes four of these at once.
    """
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"{what} not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"{what} {p}: {e}") from e


def dump(doc: object, out: str) -> Path | None:
    """Write a report where the caller asked, and say where it went. `--out` empty = nowhere."""
    if not out:
        return None
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    con.print(f"→ {p}")
    return p
