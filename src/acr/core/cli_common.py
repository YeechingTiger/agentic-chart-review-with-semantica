"""The plumbing every `acr` command group shares: the console, the model options, run identity,
and the loader that checks an artifact is the kind of artifact it claims to be.

Split out of `cli.py` when that file reached 1206 lines and fifteen outbound edges. Nothing
here decides anything about a chart, a spec or a guideline; it is the handful of things every
group would otherwise keep its own copy of, and a second copy of `code_sha` is how two runs
of the same experiment end up recorded under different identities.
"""
from __future__ import annotations

import functools
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from . import site

if TYPE_CHECKING:
    from .llm import LLMClient

con = Console()

CORPUS = typer.Option(str(site.corpus_root()), "--corpus", help="root directory of patient directories")
MODEL = typer.Option(None, "--model", "-m", help="LiteLLM model string, e.g. ollama_chat/qwen3.6:35b")
API_BASE = typer.Option(None, "--api-base", help="override provider base URL (vLLM, proxy, …)")

# The hooks runtime has two backstops: a model-call limit and a priced USD ceiling. Keep both
# on the public CLI and pass them through every chart-review command; a limit hidden inside a
# Python default is a number no operator chose.
MAX_STEPS = typer.Option(24, "--max-steps",
                         help="MODEL CALLS before the run is cut off. Not plan/act/reflect "
                              "cycles — there is no reflect node; one call is one turn in which "
                              "the model may issue tool calls. The cost ceiling in spend.py is "
                              "the limit meant to bind; this is a backstop")
MAX_USD = typer.Option(5.0, "--max-usd", min=0.01,
                       help="priced per-run ceiling in USD; stops an unfinished run when reached")

#: The artifact contract of the L0-L5 chain. Named here rather than in the command that writes
#: each one, because the command that READS an artifact has to name the same string; two copies
#: of a schema tag drift, and a drifted tag turns `_load_artifact`'s guard into a nuisance
#: somebody deletes.
EXTRACT_SCHEMA = "acr.extract/1"
CONCORD_SCHEMA = "acr.concord/1"
EXPLAIN_SCHEMA = "acr.evaluation.explain/1"


@functools.lru_cache(maxsize=1)
def code_sha() -> str:
    """Short git sha, or 'dirty'/'nogit'. A run is only reproducible against the code that
    produced it, so the code identity belongs in the run's name — and, since 2026-08-03, in
    every manifest, which is what made the cost worth thinking about.

    CACHED PER PROCESS, and that is the more correct answer rather than merely the cheaper one.
    Two `git` subprocesses cost ~42ms; a batch of eighteen charts pays it eighteen times, and
    the test suite paid it once per simulated run. But the reason to cache is that the code a
    process is RUNNING was fixed when its modules were imported. A `git rev-parse` issued after
    someone commits mid-session would report a sha whose code this process never loaded, and a
    `-dirty` that appears halfway through a batch would split one experiment's runs across two
    identities for an edit none of them saw. The value at import time is the honest one.
    """
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    d = Path(f"{base}__{stamp}__{code_sha()}")
    d.mkdir(parents=True, exist_ok=False, mode=0o700)
    d.chmod(0o700)
    return d


def llm_client(model, api_base, temperature=0.0) -> LLMClient:
    """THE provider seam for every command group, reached as `cli_common.llm_client(...)`.

    Called through the module rather than imported by name, so that one monkeypatch here
    silences the provider for every command. When each group held its own `_llm`, a test that
    muzzled one of them left the others free to dial out.
    """
    from .llm import LLMClient, LLMConfig
    return LLMClient(LLMConfig.from_env(model=model, api_base=api_base, temperature=temperature))


def chat_model(model, api_base, temperature=1.0):
    """THE provider seam for the hooks runtime, the counterpart to `llm_client` above.

    Same rule, same reason: reached as `cli_common.chat_model(...)` so one monkeypatch silences
    the provider for every command that uses the library graph. The hooks branch of `extract`
    constructed its own `ChatOpenAI` inline, which is exactly the mistake `llm_client`'s
    docstring already warns about — "when each group held its own `_llm`, a test that muzzled
    one of them left the others free to dial out". The consequence here was narrower and worse
    than a stray API call: the eight end-to-end `extract` tests inject at `llm_client`, so they
    could not drive this runtime at all, and the better runtime could not become the default
    without turning them red.

    Imported lazily because langchain is not needed by any command that stays on the
    LangGraph path.
    """
    import os

    from langchain_openai import ChatOpenAI

    from .usage_telemetry import _callbacks
    model_name = model or os.getenv("ACR_MODEL_NAME") or os.getenv(
        "ACR_MODEL", "gpt-5.6-luna"
    )
    # ``ACR_MODEL`` is shared with the LiteLLM runtime, where provider-qualified names such
    # as ``openai/gpt-5.6-luna`` are required.  ``ChatOpenAI`` sends its model string to the
    # OpenAI-compatible endpoint as the deployment name, so forwarding that prefix makes an
    # otherwise healthy Azure deployment return DeploymentNotFound.  Keep one operator-facing
    # environment variable while adapting its representation at the provider seam.
    if model_name.startswith("openai/"):
        model_name = model_name.split("/", 1)[1]
    return ChatOpenAI(model=model_name,
                      base_url=api_base or os.getenv("ACR_API_BASE"),
                      api_key=os.getenv("ACR_API_KEY"),
                      temperature=temperature, timeout=600, max_retries=3,
                      callbacks=_callbacks())


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
