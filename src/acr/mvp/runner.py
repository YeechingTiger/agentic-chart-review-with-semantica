"""Drive one patient review through the codex harness, capturing both observation layers.

The runner owns everything harness-specific, so nothing else has to know codex exists:

  * it writes an isolated CODEX_HOME with a config.toml naming the model provider
    (wire_api = "responses" — codex-cli 0.149 dropped chat-completions support) and
    registering the chart toolserver as an MCP server;
  * it disables every codex-native tool that could bypass the tool boundary
    (`shell_tool`, `unified_exec`, browser/computer/image tools) — with those off, the only
    actions available to the model are this contract's five chart tools, which is what makes
    the Layer-1 trace complete by construction;
  * it captures codex's `--json` event stream verbatim as Layer 2 (archived, never scored);
  * it writes the fallback result.json when a session ends without an accepted answer,
    so every run directory is scoreable.

Provider selection is by environment, matching the repo's .env convention:
  ACR_API_BASE / ACR_API_KEY / ACR_MODEL   (OpenRouter or any Responses-speaking endpoint)
or an explicit base_url override for the local fake model (tests, this sandbox).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from acr.contract.spec import load_spec

DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "browser_use", "computer_use", "view_image", "apps",
)

TASK_PREAMBLE = """You are performing a chart review for one patient against the contract below.
You can only act through the `chart` tools. Record the evidence that establishes your answer
with record_evidence BEFORE submitting; finish by calling submit_answer. If the submission is
refused, the verdict names what is missing — fix it and submit again.
"""


def _config_toml(model: str, base_url: str, env_key: str, python: str,
                 server_env: dict[str, str]) -> str:
    env_lines = "\n".join(f'{k} = "{v}"' for k, v in server_env.items())
    return f"""\
model = "{model}"
model_provider = "acr"

[model_providers.acr]
name = "acr provider"
base_url = "{base_url}"
env_key = "{env_key}"
wire_api = "responses"

[mcp_servers.chart]
command = "{python}"
args = ["-m", "acr.mvp.toolserver"]
# "approve" = auto-approve (config/src/mcp_types.rs AppToolApproval). The default "auto" treats
# unannotated tools as writes, and with approval_policy=never every call is then denied unheard.
default_tools_approval_mode = "approve"

[mcp_servers.chart.env]
{env_lines}
"""


def run_patient(
    spec_path: Path,
    patient_dir: Path,
    out_root: Path,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 900,
    codex_bin: str = "codex",
) -> Path:
    """One review, one run directory. Returns the run directory."""
    spec = load_spec(spec_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{stamp}_{patient_dir.name}_{spec.spec_id.replace('.', '_')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    model = model or os.environ.get("ACR_MODEL", "openai/gpt-5.6-luna")
    base_url = base_url or os.environ.get("ACR_API_BASE", "https://openrouter.ai/api/v1")
    api_key = api_key if api_key is not None else os.environ.get("ACR_API_KEY", "unset")

    codex_home = run_dir / "codex_home"
    codex_home.mkdir()
    # The toolserver runs under THIS interpreter, with acr importable no matter how the parent
    # found it — `which python` once picked the system interpreter and the server died unheard.
    python = sys.executable
    import acr
    src_root = str(Path(next(iter(acr.__path__))).parent)
    pythonpath = os.pathsep.join(p for p in (src_root, os.environ.get("PYTHONPATH")) if p)
    server_env = {
        "ACR_MVP_SPEC": str(spec_path.resolve()),
        "ACR_MVP_PATIENT_DIR": str(patient_dir.resolve()),
        "ACR_MVP_RUN_DIR": str(run_dir.resolve()),
        "PYTHONPATH": pythonpath,
        "PATH": os.environ.get("PATH", ""),
    }
    (codex_home / "config.toml").write_text(
        _config_toml(model, base_url, "ACR_API_KEY", python, server_env), encoding="utf-8")

    # The agent's working directory is an empty room: nothing to read even if a
    # filesystem-facing tool were to survive the feature flags.
    workdir = run_dir / "workdir"
    workdir.mkdir()

    prompt = TASK_PREAMBLE + "\n" + spec.as_prompt_block()
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    cmd = [codex_bin, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
           "-C", str(workdir), "--output-last-message", str(run_dir / "last_message.txt")]
    for feature in DISABLED_FEATURES:
        cmd += ["--disable", feature]
    cmd += ["-c", 'approval_policy="never"', "-s", "read-only", "-"]

    env = os.environ.copy()
    env.update({
        "CODEX_HOME": str(codex_home),
        "ACR_API_KEY": api_key,
        # The local fake model must not be routed through the egress proxy.
        "NO_PROXY": env.get("NO_PROXY", "") + ",127.0.0.1,localhost",
        "no_proxy": env.get("no_proxy", "") + ",127.0.0.1,localhost",
    })

    layer2 = run_dir / "layer2_codex.jsonl"
    stderr_log = run_dir / "codex_stderr.log"
    with layer2.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd, input=prompt, stdout=out, stderr=err, env=env,
            text=True, timeout=timeout_s, check=False,
        )

    result_path = run_dir / "result.json"
    if not result_path.exists():
        # The session ended without an accepted answer. Scoreable anyway, honestly labeled.
        result_path.write_text(json.dumps({
            "status": "NO_ANSWER", "value": None, "accepted": False,
            "patient_id": patient_dir.name,
            "spec_id": spec.spec_id, "spec_hash": spec.spec_hash,
            "why": f"codex exited {proc.returncode} with no accepted submission",
        }, indent=2), encoding="utf-8")
    (run_dir / "runner_meta.json").write_text(json.dumps({
        "model": model, "base_url": base_url, "codex_returncode": proc.returncode,
        "codex_version": _codex_version(codex_bin), "timeout_s": timeout_s,
    }, indent=2), encoding="utf-8")
    return run_dir


def _codex_version(codex_bin: str) -> str:
    try:
        out = subprocess.run([codex_bin, "--version"], capture_output=True, text=True,
                             timeout=10, check=False)
        return out.stdout.strip() or out.stderr.strip()
    except OSError:
        return "unavailable"
