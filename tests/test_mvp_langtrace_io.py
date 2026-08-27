from __future__ import annotations

import os

from acr.mvp import langtrace_io


def test_initialise_uses_normalized_local_ingest_host_even_when_environment_has_base_url(
        monkeypatch):
    """The SDK reads LANGTRACE_API_HOST before its explicit argument."""
    from langtrace_python_sdk import langtrace

    observed: dict[str, object] = {}

    def fake_init(**kwargs):
        observed.update(kwargs)
        observed["environment_host_during_init"] = os.environ.get("LANGTRACE_API_HOST")

    monkeypatch.setattr(langtrace_io, "_INITIALISED", None)
    monkeypatch.setenv("LANGTRACE_API_HOST", "http://127.0.0.1:3100")
    monkeypatch.setattr(langtrace, "init", fake_init)

    langtrace_io.initialise_langtrace("local-project-key", "http://127.0.0.1:3100")

    expected = "http://127.0.0.1:3100/api/trace"
    assert observed["api_host"] == expected
    assert observed["environment_host_during_init"] == expected
