"""Drive one patient review through the Codex App Server harness and Langtrace.

The runner owns everything harness-specific, so nothing else has to know codex exists:

  * it writes an isolated CODEX_HOME with a config.toml naming the model provider
    (wire_api = "responses" — codex-cli 0.149 dropped chat-completions support) and
    registering the chart toolserver as an MCP server;
  * it disables every codex-native tool that could bypass the tool boundary
    (`shell_tool`, `unified_exec`, browser/computer/image and multi-agent tools) — Codex's
    inert planning and MCP-discovery helpers remain, but the only tools that can reach patient
    data are this contract's chart tools, which makes the Layer-1 trace complete by construction;
  * it uses the official Python SDK to call the Codex App Server JSON-RPC API and captures its
    typed event stream as Layer 2;
  * it publishes both layers to Langtrace, which is the read source for reconstruction;
  * it writes the fallback result.json when a session ends without an accepted answer,
    so every run directory is scoreable.

Provider selection is by environment, matching the repo's .env convention:
  ACR_API_BASE / ACR_API_KEY / ACR_MODEL   (OpenRouter or any Responses-speaking endpoint)
or an explicit base_url override for the local fake model (tests, this sandbox).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from acr.contract.spec import load_spec
from acr.mvp.codex_harness import AppServerCodexHarness
from acr.mvp.langtrace_io import LangtraceClient, LangtraceConfigurationError, LangtraceRun
from acr.mvp.task_presentation import TASK_ARMS, build_task_presentation
from acr.mvp.toolserver import TOOL_SCHEMA_INSTRUCTIONS
from acr.mvp.warrants import basis_source_lines, input_prompt_lines, rule_coverage_lines

DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "browser_use", "computer_use", "view_image",
    "image_generation", "apps", "multi_agent",
)
class CodexCompatibilityError(RuntimeError):
    """The installed Codex cannot enforce this runner's patient-data boundary."""


def require_codex_tool_boundary(codex_bin: str) -> None:
    """Refuse to run when Codex cannot disable every data-bearing native tool."""
    try:
        probe = subprocess.run(
            [codex_bin, "features", "list"], capture_output=True, text=True,
            timeout=10, check=False,
        )
    except OSError as exc:
        raise CodexCompatibilityError(f"cannot execute Codex binary {codex_bin!r}: {exc}") from exc
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise CodexCompatibilityError(
            f"cannot inspect Codex features ({detail or f'exit {probe.returncode}'})")

    available = {line.split()[0] for line in probe.stdout.splitlines() if line.split()}
    missing = sorted(set(DISABLED_FEATURES) - available)
    if missing:
        version = _codex_version(codex_bin)
        raise CodexCompatibilityError(
            f"{version or codex_bin} cannot enforce the chart-only tool boundary; "
            f"missing feature flag(s): {', '.join(missing)}. "
            "Install codex-cli 0.150.0 or newer.")

TASK_PREAMBLE = """You are performing a chart review for one patient against the contract below.
You can only act through the `chart` tools. Record the evidence that establishes your answer
with record_evidence BEFORE submitting; finish by calling submit_answer. If the submission is
refused, the verdict names what is missing — fix it and submit again.

Do not reveal private chain-of-thought. Record one independently auditable choice per Decision
Testimony: a reviewer should be able to mark that choice correct, incorrect, or uncertain without
having to split it into two judgments. Whenever you choose what to look for next, which source
governs, whether the current evidence is enough, or whether to stop, call note_decision BEFORE
acting. The one exception is a note's field Standing: record_finding is itself that atomic
Decision Testimony, so put the full basis on that call instead of creating a prior note_decision.

Do not create testimony for every tool call. One precommitted batch of keywords is one choice and
may govern several search calls. Opening every note in an already chosen read-all set is execution,
not a new choice. If a new observation makes you revise the query batch, note-type scope, selected
note set, evidence judgment, or stopping plan, record a new Decision Testimony before the revised
action. Mechanical calls with no meaningful alternative do not need invented testimony.

Give the compact, auditable Decision Testimony fields:

  facing                 — the open question and facts known BEFORE the next observation
  decision               — what you chose or what question the next action will resolve
  because                — a concise audit explanation
  basis_sources          — where the rationale came from:
{basis_sources}
  cited_refs             — exact references actually used:
{input_kinds}
                           Contract rules use decision_rule.N / conflict_rule.N /
                           evidence_rule.... exactly as rendered below.
  checked_discriminating_fact_refs — exact discriminating_fact.<fact_id> values checked
  rule_coverage_claim     — choose one:
{rule_coverage}
  provisional_inference  — any assumption added beyond supplied rules, or null (optional)
  alternatives           — candidates actually considered (optional)
  uncertainty            — unresolved ambiguity or conflict, or null (optional)

You are NOT asked to classify your decisions. Say what you decided in your own words; sorting
them into kinds is somebody else's job, later.

Be exact about claims that cannot be recovered afterwards:

  * cite what you ACTUALLY used, not everything that could support the choice. The server marks
    exact references as verified, not offered, or unknown without refusing your testimony.
  * when a choice depends on an earlier recorded choice, cite its returned decision:N reference.
    In particular, the retrieval plan after note inventory should cite the inventory decision,
    and the stop/submit choice should cite the comparison or answer decision it executes. This is
    what lets the later audit graph prove a causal link instead of guessing from time order.
  * DIRECTLY_COVERED means the cited rule determines the exact choice, not merely its goal. If a
    rule says what evidence is relevant but you choose the search keywords, notes to open, or
    stopping tactic, use COVERED_WITH_INTERPRETATION or OPERATIONAL_DISCRETION as appropriate.
  * own_knowledge is allowed and sometimes necessary. It is not automatically a guideline gap;
    state separately how the offered rules cover the situation.
  * after reading a relevant note, call record_finding for the requested field. Judge exactly
    ONE note + ONE field per call and include facing, because, basis_sources, cited_refs,
    checked_discriminating_fact_refs, rule_coverage_claim, and any provisional inference or
    uncertainty on that same call. Do not combine several notes' standings into one testimony.
    Record whether it can_establish, merely_mentions, or neither. Establishing and mention
    findings require a server-resolved source span; neither uses assertion_class=not_applicable.
  * first record each candidate note's standing independently. Only after those findings exist,
    make a separate note_decision for comparison/conflict resolution; cite the exact finding:N
    references actually compared. Recording the final evidence spans is execution of that choice,
    not another decision unless a new material alternative appears.

On every search, read and list_documents call, fill `objective` with the open question that
call is meant to resolve.

Testimony is not judged and a bad citation is recorded rather than suppressed. Runtime note
findings can be refused only when their document/span structure is impossible. An auditor must
be able to follow each decision point and distinguish observed state from your declared state.
"""


def _config_toml(model: str, base_url: str, env_key: str, python: str,
                 server_env: dict[str, str]) -> str:
    env_lines = "\n".join(f'{k} = "{v}"' for k, v in server_env.items())
    return f"""\
model = "{model}"
model_provider = "acr"
web_search = "disabled"

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


def _resolve_codex_bin(codex_bin: str) -> str:
    resolved = shutil.which(codex_bin)
    if resolved:
        return resolved
    candidate = Path(codex_bin)
    if candidate.is_file():
        return str(candidate.resolve())
    raise CodexCompatibilityError(f"cannot find Codex binary {codex_bin!r}")


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
    task_arm: str = "detailed",
    langtrace_api_key: str | None = None,
    langtrace_api_host: str | None = None,
    langtrace_project_id: str | None = None,
) -> Path:
    """One review, one run directory. Returns the run directory."""
    # Check before creating a run directory: an unsupported CLI must fail closed, without
    # leaving behind a plausible-looking NO_ANSWER run whose tool boundary was never enforced.
    codex_bin = _resolve_codex_bin(codex_bin)
    require_codex_tool_boundary(codex_bin)
    if task_arm not in TASK_ARMS:
        raise ValueError(f"task_arm must be one of {TASK_ARMS}, got {task_arm!r}")
    langtrace_api_key = (langtrace_api_key if langtrace_api_key is not None
                         else os.environ.get("LANGTRACE_API_KEY"))
    langtrace_api_host = (langtrace_api_host if langtrace_api_host is not None
                          else os.environ.get("LANGTRACE_API_HOST"))
    if not langtrace_api_key or not langtrace_api_host:
        raise LangtraceConfigurationError(
            "run requires LANGTRACE_API_KEY and LANGTRACE_API_HOST")
    spec = load_spec(spec_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (out_root / (f"{stamp}_{patient_dir.name}_{spec.spec_id.replace('.', '_')}_"
                           f"{task_arm}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    preamble = TASK_PREAMBLE.format(
        basis_sources=basis_source_lines(), input_kinds=input_prompt_lines(),
        rule_coverage=rule_coverage_lines())
    prompt, presentation = build_task_presentation(
        spec, run_id=run_dir.name, arm_id=task_arm, operational_preamble=preamble,
        operational_instructions=TOOL_SCHEMA_INSTRUCTIONS)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    presentation.write(run_dir)

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
        "ACR_MVP_TASK_PRESENTATION": str((run_dir / "task_presentation.json").resolve()),
        "PYTHONPATH": pythonpath,
        "PATH": os.environ.get("PATH", ""),
    }
    (codex_home / "config.toml").write_text(
        _config_toml(model, base_url, "ACR_API_KEY", python, server_env), encoding="utf-8")

    # The agent's working directory is an empty room: nothing to read even if a
    # filesystem-facing tool were to survive the feature flags.
    workdir = run_dir / "workdir"
    workdir.mkdir()

    env = os.environ.copy()
    env.update({
        "CODEX_HOME": str(codex_home.resolve()),
        "ACR_API_KEY": api_key,
        # Langtrace is local in the self-hosted setup and must not cross the egress proxy.
        "NO_PROXY": env.get("NO_PROXY", "") + ",127.0.0.1,localhost",
        "no_proxy": env.get("no_proxy", "") + ",127.0.0.1,localhost",
    })

    layer2 = run_dir / "layer2_codex.jsonl"
    stderr_log = run_dir / "codex_stderr.log"
    langtrace_run = LangtraceRun(
        api_key=langtrace_api_key, api_host=langtrace_api_host,
        run_id=run_dir.name, patient_id=patient_dir.name,
        spec_id=spec.spec_id, spec_hash=spec.spec_hash,
        model=model, task_arm=task_arm,
        task_presentation_hash=presentation.presentation_hash,
    )
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    if carrier.get("traceparent"):
        env["TRACEPARENT"] = carrier["traceparent"]
    if carrier.get("tracestate"):
        env["TRACESTATE"] = carrier["tracestate"]

    harness = AppServerCodexHarness(codex_bin, DISABLED_FEATURES)
    harness_result = harness.run(
        prompt, model=model, workdir=workdir, env=env,
        layer2_path=layer2, timeout_s=timeout_s,
        on_event=langtrace_run.codex_event,
    )
    langtrace_run.review_model_call(
        requested_model=model, thread_id=harness_result.thread_id,
        turn_id=harness_result.turn_id)
    (run_dir / "last_message.txt").write_text(
        harness_result.final_response or "", encoding="utf-8")
    stderr_log.write_text(
        "\n".join(x for x in (harness_result.error, harness_result.stderr) if x),
        encoding="utf-8")

    result_path = run_dir / "result.json"
    if not result_path.exists():
        # The session ended without an accepted answer. Scoreable anyway, honestly labeled.
        result_path.write_text(json.dumps({
            "status": "NO_ANSWER", "value": None, "accepted": False,
            "patient_id": patient_dir.name,
            "spec_id": spec.spec_id, "spec_hash": spec.spec_hash,
            "why": (f"Codex App Server ended {harness_result.status} with no accepted "
                    "submission"),
        }, indent=2), encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    publish_error: str | None = None
    try:
        langtrace_run.publish_run(run_dir)
    except Exception as exc:
        publish_error = f"{type(exc).__name__}: {exc}"
    finally:
        langtrace_run.finish(result_status=str(result.get("status") or "NO_ANSWER"),
                             error=publish_error or harness_result.error)
    if publish_error:
        raise RuntimeError(f"failed to publish the review trace to Langtrace: {publish_error}")

    # A trace is part of a successful run, not eventual best-effort telemetry. Read it back
    # through the same API reconstruct will use before claiming the run is complete.
    langtrace_client = LangtraceClient(
        api_key=langtrace_api_key, api_host=langtrace_api_host,
        project_id=langtrace_project_id)
    langtrace_client.get_review(langtrace_run.trace_id)
    (run_dir / "runner_meta.json").write_text(json.dumps({
        "model": model, "base_url": base_url,
        "harness": "codex-app-server", "codex_returncode": harness_result.returncode,
        "codex_version": _codex_version(codex_bin), "timeout_s": timeout_s,
        "task_arm": task_arm, "langtrace_trace_id": langtrace_run.trace_id,
        "task_presentation_hash": presentation.presentation_hash,
        "langtrace_project_id": langtrace_client.project_id,
        "langtrace_verified": True,
        "review_model_call": {
            "requested_model": model,
            "configured_provider": "acr",
            "codex_thread_id": harness_result.thread_id,
            "codex_turn_id": harness_result.turn_id,
            "resolved_model": None,
            "response_provider": None,
            "response_id": None,
            "identity_status": "CODEX_HARNESS_IDS_ONLY",
            "identity_note": ("Codex App Server does not expose the upstream provider response "
                              "identity on its public notification API"),
        },
    }, indent=2), encoding="utf-8")
    return run_dir


def _codex_version(codex_bin: str) -> str:
    try:
        out = subprocess.run([codex_bin, "--version"], capture_output=True, text=True,
                             timeout=10, check=False)
        return out.stdout.strip() or out.stderr.strip()
    except OSError:
        return "unavailable"
