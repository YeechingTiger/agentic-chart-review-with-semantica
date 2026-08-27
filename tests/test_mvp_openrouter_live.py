"""Paid model-facing acceptance: real Luna review, Langtrace, real Luna reconstruction.

Run explicitly with ``ACR_RUN_OPENROUTER_LIVE=1``. Deterministic tests may stub malformed
extractor output or protocol endpoints, but this is the acceptance test for model behavior and
must never substitute a fake model or a local trace fallback.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from acr.mvp.langtrace_io import LangtraceClient
from acr.mvp.ledger import NullLedger
from acr.mvp.reconstruct import reconstruct_run
from acr.mvp.runner import run_patient

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
REVIEW_MODEL = "openai/gpt-5.6-luna"
RECONSTRUCT_MODEL = "openrouter/openai/gpt-5.6-luna"
CASES = {"SYN0001": "20230412", "SYNX03": "20220309"}


def _missing_live_requirement() -> str | None:
    if os.environ.get("ACR_RUN_OPENROUTER_LIVE") != "1":
        return "set ACR_RUN_OPENROUTER_LIVE=1 to authorize paid OpenRouter calls"
    if os.environ.get("ACR_OPENROUTER_KEY_ROTATED") != "1":
        return "set ACR_OPENROUTER_KEY_ROTATED=1 only after replacing the key exposed in chat"
    if os.environ.get("ACR_LANGTRACE_SELF_HOSTED") != "1":
        return "set ACR_LANGTRACE_SELF_HOSTED=1 for the authorized PHI trace deployment"
    required = ("OPENROUTER_API_KEY", "LANGTRACE_API_KEY", "LANGTRACE_API_HOST",
                "LANGTRACE_PROJECT_ID")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return "missing live-test environment: " + ", ".join(missing)
    if shutil.which("codex") is None:
        return "codex binary not on PATH"
    return None


LIVE_BLOCKER = _missing_live_requirement()


def _reusable_run(patient_id: str, arm: str) -> Path | None:
    """Optionally re-check a paid Luna artifact while always re-running reconstruction.

    This is useful when validating the test wrapper immediately after a manual pilot. Normal
    CI/live invocations omit ``ACR_LIVE_REUSE_ROOT`` and create fresh Luna runs.
    """
    root = os.environ.get("ACR_LIVE_REUSE_ROOT")
    if not root:
        return None
    candidates = sorted(Path(root).glob(f"*_{patient_id}_*_{arm}"), reverse=True)
    return next((path for path in candidates
                 if (path / "runner_meta.json").is_file()
                 and (path / "task_presentation.json").is_file()), None)


@pytest.mark.live_openrouter
@pytest.mark.skipif(LIVE_BLOCKER is not None, reason=LIVE_BLOCKER or "live setup required")
def test_two_prompt_arms_flow_from_real_luna_through_langtrace_to_real_luna_reconstruction(
        tmp_path: Path):
    from acr.mvp.reconstruction_llm import AuditedLiteLLM

    key = os.environ["OPENROUTER_API_KEY"]
    langtrace_key = os.environ["LANGTRACE_API_KEY"]
    langtrace_host = os.environ["LANGTRACE_API_HOST"]
    langtrace_project = os.environ["LANGTRACE_PROJECT_ID"]
    client = LangtraceClient(api_key=langtrace_key, api_host=langtrace_host,
                             project_id=langtrace_project)
    runs: dict[tuple[str, str], dict] = {}

    for patient_id, gold in CASES.items():
        for arm in ("task_only", "policy_bundle"):
            run_dir = _reusable_run(patient_id, arm)
            if run_dir is None:
                run_dir = run_patient(
                    SPEC, ROOT / "corpus" / "patients" / patient_id, tmp_path / "runs",
                    model=REVIEW_MODEL, base_url="https://openrouter.ai/api/v1", api_key=key,
                    task_arm=arm, langtrace_api_key=langtrace_key,
                    langtrace_api_host=langtrace_host,
                    langtrace_project_id=langtrace_project,
                )
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            meta = json.loads((run_dir / "runner_meta.json").read_text(encoding="utf-8"))
            assert result["status"] == "FOUND"
            assert result["value"]["date_of_initial_diagnosis"].isdigit()
            if arm == "policy_bundle":
                assert result["value"]["date_of_initial_diagnosis"] == gold
            assert meta["harness"] == "codex-app-server"
            assert meta["model"] == REVIEW_MODEL
            assert meta["task_arm"] == arm
            assert meta["langtrace_verified"] is True
            assert meta["review_model_call"]["requested_model"] == REVIEW_MODEL
            assert meta["review_model_call"]["codex_thread_id"]
            assert meta["review_model_call"]["codex_turn_id"]
            assert meta["review_model_call"]["identity_status"] == "CODEX_HARNESS_IDS_ONLY"

            review = client.get_review(meta["langtrace_trace_id"])
            assert review.patient_id == patient_id
            assert review.layer1_events and review.steps and review.spans
            luna_reconstructor = AuditedLiteLLM(
                model=RECONSTRUCT_MODEL, api_key=key, temperature=0.0)
            sink = NullLedger()
            summary = reconstruct_run(review, sink, luna_reconstructor, passes=2,
                                      artifact_dir=tmp_path / "analyses")
            assert summary["trace_completeness"]["export_status"] == "COMPLETE"
            assert summary["trace_id"] == meta["langtrace_trace_id"]
            assert len(summary["analyses"]) == 2
            assert all(row["n_episodes"] > 0 for row in summary["analyses"])
            assert summary["selected_analysis_id"] is None
            assert len(sink.artifacts) == 2
            assert all(row["reconstructor_call"]["identity_status"]
                       == "RETURNED_BY_PROVIDER" for row in sink.artifacts)
            assert all(row["reconstructor_call"]["resolved_model"]
                       and row["reconstructor_call"]["response_provider"]
                       and row["reconstructor_call"]["response_id"] for row in sink.artifacts)
            assert len({row["cycles_hash"] for row in sink.artifacts}) == 1
            for artifact in sink.artifacts:
                owned = [cycle_id for episode in artifact["episodes"]
                         for cycle_id in episode["source_cycle_ids"]]
                owned += artifact["mechanical_cycle_ids"]
                assert len(owned) == len(set(owned)) == len(artifact["cycles"])
                assert all(episode["field_provenance"] and episode["source_refs_by_field"]
                           for episode in artifact["episodes"])
            runs[(patient_id, arm)] = {"result": result, "summary": summary}

    # Both task categories really ran for both mirror cases. The policy arm is the production
    # contract and must hit gold; the bare arm is intentionally allowed to expose a difference.
    assert set(runs) == {(case, arm) for case in CASES
                        for arm in ("task_only", "policy_bundle")}
