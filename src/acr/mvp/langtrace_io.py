"""Write chart-review traces to Langtrace and read them back for reconstruction.

Langtrace is not a debug sidecar here.  It is the reconstruction source.  Each review emits:

* typed Codex App Server notifications;
* the MCP server's authoritative Layer-1 events;
* the merged, ordered review steps used by the reconstruction prompt.

The run directory retains JSONL as an audit copy, but :mod:`acr.mvp.reconstruct` consumes the
records returned by Langtrace's API-key trace endpoint.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from acr.mvp.task_presentation import content_hash


class LangtraceConfigurationError(RuntimeError):
    pass


class LangtraceReadError(RuntimeError):
    pass


_INIT_LOCK = threading.Lock()
_INITIALISED: tuple[str, str] | None = None


def _base_host(host: str) -> str:
    return host.rstrip("/").removesuffix("/api/trace").removesuffix("/v1/traces")


def _ingest_host(host: str) -> str:
    host = host.rstrip("/")
    if host.endswith(("/api/trace", "/v1/traces")):
        return host
    # The self-hosted Langtrace server ingests at /api/trace. The SDK otherwise assumes a
    # generic OTLP collector's /v1/traces path, which is not Langtrace's route.
    return f"{host}/api/trace"


def initialise_langtrace(api_key: str, api_host: str) -> None:
    """Initialise latest Langtrace once per process, with no third-party auto-instrumentation."""
    global _INITIALISED
    if not api_key or not api_host:
        raise LangtraceConfigurationError(
            "LANGTRACE_API_KEY and LANGTRACE_API_HOST are required; reconstruct has no local fallback")
    api_host = _ingest_host(api_host)
    identity = (api_key, api_host)
    with _INIT_LOCK:
        if _INITIALISED == identity:
            return
        if _INITIALISED is not None:
            raise LangtraceConfigurationError(
                "Langtrace was already initialised with a different host/key in this process")
        os.environ["LANGTRACE_ERROR_REPORTING"] = "False"
        os.environ["TRACE_PROMPT_COMPLETION_DATA"] = "true"
        # The SDK resolves LANGTRACE_API_HOST before its explicit ``api_host`` argument.
        # Normalize the environment first as well, otherwise a common self-hosted setting such
        # as ``http://127.0.0.1:3100`` silently exports to the SDK's generic OTLP path rather
        # than Langtrace's ``/api/trace`` endpoint.
        os.environ["LANGTRACE_API_HOST"] = api_host
        from langtrace_python_sdk import langtrace
        langtrace.init(
            api_key=api_key,
            api_host=api_host,
            batch=False,
            service_name="acr-chart-review",
            disable_logging=False,
            disable_instrumentations={"all_except": []},
        )
        # Langtrace 3.8.21 strips /api/trace from this environment variable after creating
        # its exporter. Put the caller's endpoint back so subsequent reviews in this process
        # initialise against the same identity.
        os.environ["LANGTRACE_API_HOST"] = api_host
        _INITIALISED = identity


def _attrs(span: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        span.set_attribute(key, value)


class LangtraceRun:
    """One Langtrace root span around one Codex chart review."""

    def __init__(self, *, api_key: str, api_host: str, run_id: str, patient_id: str,
                 spec_id: str, spec_hash: str, model: str, task_arm: str,
                 task_presentation_hash: str) -> None:
        initialise_langtrace(api_key, api_host)
        self.tracer = trace.get_tracer("acr.mvp", "0.1.0")
        self._cm = self.tracer.start_as_current_span(
            "acr.chart_review", kind=SpanKind.INTERNAL)
        self.root = self._cm.__enter__()
        ctx = self.root.get_span_context()
        self.trace_id = f"{ctx.trace_id:032x}"
        _attrs(self.root, {
            "langtrace.service.type": "framework",
            "langtrace.service.name": "codex",
            "acr.trace.schema": "acr.langtrace.v2",
            "acr.langtrace.trace_id": self.trace_id,
            "acr.run_id": run_id,
            "acr.patient_id": patient_id,
            "acr.spec_id": spec_id,
            "acr.spec_hash": spec_hash,
            "acr.task_arm": task_arm,
            "acr.task_presentation_hash": task_presentation_hash,
            "gen_ai.request.model": model,
        })

    def codex_event(self, event: dict[str, Any]) -> None:
        item = event.get("item") or {}
        item_type = str(item.get("type") or "event")
        with self.tracer.start_as_current_span(
                f"codex.{event.get('type', 'event')}.{item_type}") as span:
            _attrs(span, {
                "langtrace.service.type": "framework",
                "langtrace.service.name": "codex-app-server",
                "acr.codex.event_type": event.get("type"),
                "acr.codex.item_type": item_type,
                "acr.codex.event_json": event,
            })

    def review_model_call(self, *, requested_model: str, thread_id: str | None,
                          turn_id: str | None) -> None:
        """Record the identity the App Server exposes without inventing provider metadata."""
        _attrs(self.root, {
            "acr.review_model.requested_model": requested_model,
            "acr.review_model.codex_thread_id": thread_id,
            "acr.review_model.codex_turn_id": turn_id,
            "acr.review_model.identity_status": "CODEX_HARNESS_IDS_ONLY",
        })

    def publish_run(self, run_dir: Path) -> None:
        """Publish the authoritative events and merged reconstruction steps to Langtrace."""
        from acr.mvp.observe import decision_trace

        events = [json.loads(line) for line in
                  (Path(run_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        _attrs(self.root, {
            "acr.layer1.event_count": len(events),
            "acr.layer1.content_hash": content_hash(events),
        })
        for event in events:
            with self.tracer.start_as_current_span(
                    f"acr.layer1.{event.get('kind', 'event')}") as span:
                _attrs(span, {
                    "langtrace.service.type": "tool",
                    "langtrace.service.name": "acr-chart-mcp",
                    "acr.layer1.seq": event.get("seq"),
                    "acr.layer1.event_id": f"layer1:{event.get('seq')}",
                    "acr.layer1.kind": event.get("kind"),
                    "acr.layer1.event_json": event,
                })

        merged = decision_trace(Path(run_dir))
        _attrs(self.root, {
            "acr.review.step_count": len(merged["steps"]),
            "acr.review.content_hash": content_hash(merged["steps"]),
        })
        for index, step in enumerate(merged["steps"]):
            with self.tracer.start_as_current_span(
                    f"acr.review.{step.get('kind', 'step')}") as span:
                _attrs(span, {
                    "langtrace.service.type": "workflow",
                    "langtrace.service.name": "acr-review-trace",
                    "acr.review.index": index,
                    "acr.review.seq": step.get("seq"),
                    "acr.review.kind": step.get("kind"),
                    "acr.review.channel": step.get("channel"),
                    "acr.review.step_json": step,
                })
        # Written last: a fetched root may call itself complete only after every required child
        # has been synchronously exported in this span tree.
        self.root.set_attribute("acr.export.status", "COMPLETE")

    def finish(self, *, result_status: str, error: str | None = None) -> None:
        _attrs(self.root, {"acr.result.status": result_status, "error.message": error})
        if error:
            self.root.set_status(Status(StatusCode.ERROR, error))
        else:
            self.root.set_status(Status(StatusCode.OK))
        self._cm.__exit__(None, None, None)
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=30_000)


@dataclass(slots=True)
class LangtraceReviewTrace:
    trace_id: str
    run_id: str
    patient_id: str
    spec_id: str
    steps: list[dict[str, Any]]
    layer1_events: list[dict[str, Any]]
    spans: list[dict[str, Any]]
    spec_hash: str = ""
    task_arm: str = ""
    task_presentation_hash: str = ""
    review_model: str = ""


def _span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


class LangtraceClient:
    """API-key client for the self-hosted/cloud Langtrace trace API."""

    def __init__(self, *, api_key: str, api_host: str, project_id: str | None = None,
                 timeout_s: int = 30) -> None:
        if not api_key or not api_host:
            raise LangtraceConfigurationError(
                "LANGTRACE_API_KEY and LANGTRACE_API_HOST are required")
        self.api_key = api_key
        self.base = _base_host(api_host)
        self.timeout_s = timeout_s
        self.project_id = project_id or self._project_id()

    def _project_id(self) -> str:
        response = requests.get(
            f"{self.base}/api/project", headers={"x-api-key": self.api_key},
            timeout=self.timeout_s)
        response.raise_for_status()
        project_id = str((response.json().get("project") or {}).get("id") or "")
        if not project_id:
            raise LangtraceReadError("Langtrace API key resolved to no project")
        return project_id

    def get_review(self, trace_id: str) -> LangtraceReviewTrace:
        """Fetch one trace from Langtrace; there is deliberately no filesystem fallback."""
        # Langtrace's keyword filter is applied to the outer span query as well as the trace
        # selection query, so using it would return only the matching root span and discard its
        # children. Page through complete trace groups and identify ours by the explicit ACR id.
        page, total_pages = 1, 1
        spans: list[dict[str, Any]] | None = None
        while page <= total_pages and spans is None:
            response = requests.post(
                f"{self.base}/api/traces",
                headers={"x-api-key": self.api_key},
                json={"page": page, "pageSize": 100, "projectId": self.project_id},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            trace_page = response.json().get("traces") or {}
            groups = trace_page.get("result") or []
            for group in groups:
                if not isinstance(group, list):
                    continue
                if any(_span_attributes(span).get("acr.langtrace.trace_id") == trace_id
                       for span in group):
                    spans = group
                    break
            total_pages = int((trace_page.get("metadata") or {}).get("total_pages") or 1)
            page += 1
        if spans is None:
            raise LangtraceReadError(
                f"trace {trace_id} was not returned by Langtrace project {self.project_id}")

        root_attrs: dict[str, Any] = {}
        steps: list[tuple[int, dict[str, Any]]] = []
        events: list[tuple[int, dict[str, Any]]] = []
        for span in spans:
            attrs = _span_attributes(span)
            if attrs.get("acr.trace.schema") in {"acr.langtrace.v1", "acr.langtrace.v2"}:
                root_attrs = attrs
            if "acr.review.step_json" in attrs:
                steps.append((int(attrs.get("acr.review.index", 0)),
                              json.loads(str(attrs["acr.review.step_json"]))))
            if "acr.layer1.event_json" in attrs:
                event = json.loads(str(attrs["acr.layer1.event_json"]))
                events.append((int(event.get("seq", 0)), event))
        if not root_attrs or not steps or not events:
            raise LangtraceReadError(
                f"trace {trace_id} is not a complete acr.langtrace.v1 review trace")
        review = LangtraceReviewTrace(
            trace_id=trace_id,
            run_id=str(root_attrs.get("acr.run_id") or ""),
            patient_id=str(root_attrs.get("acr.patient_id") or ""),
            spec_id=str(root_attrs.get("acr.spec_id") or ""),
            steps=[step for _, step in sorted(steps)],
            layer1_events=[event for _, event in sorted(events)],
            spans=spans,
            spec_hash=str(root_attrs.get("acr.spec_hash") or ""),
            task_arm=str(root_attrs.get("acr.task_arm") or ""),
            task_presentation_hash=str(root_attrs.get("acr.task_presentation_hash") or ""),
            review_model=str(root_attrs.get("acr.review_model.requested_model")
                             or root_attrs.get("gen_ai.request.model") or ""),
        )
        from acr.mvp.timeline import build_trace_completeness
        manifest = build_trace_completeness(review)
        if manifest.export_status != "COMPLETE":
            raise LangtraceReadError(
                f"trace {trace_id} failed completeness validation: {'; '.join(manifest.issues)}")
        return review
