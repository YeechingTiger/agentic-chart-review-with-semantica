"""One query, every hit, one model call. The control the agent has to beat.

    keyword list -> chart.search -> read every hit -> ONE model call -> a manifest

THE ARM THAT DID NOT EXIST. `tools/measure_agency.py` asks whether the answer-bearing document was
REACHABLE by a contract-word query and produces no answer, so it cannot be scored against ground
truth. Nothing else in this tree extracts a value without the agent loop, which means the central
question — does the loop earn its cost — had no control.

Measured on the shipped synthetic corpus: a keyword list hits **1–5 notes of ~310**, median **260
tokens** of text. So this arm is one call over a few hundred tokens against the agent's ~18k per call
across 6–8 calls. That ratio is the point.

## Every difference from the agent arm is a difference under test

Same contract, same `submit_answer` schema, same outcome space, same value shape, same
notation-tolerant matcher (`chart.search`, which folds separators, quote widths and date forms — a
control priced against a matcher the runtime does not use is a control for a search nobody performs).
What differs is exactly: no loop, no retrieval tools, no plan, no coverage ledger, no gate.

`gate_validated` is therefore `False` and that is not a defect of this arm — there is no gate to pass.
It means `gate_validated_rate` reads 0 here, which is the honest number.

## It must be scored by the same scorer, or the comparison is worthless

The manifest below is read by `evals.RunRecord` with no special case. If this arm needed its own
scorer, "the query-only arm lost" and "the query-only arm was scored differently" would be the same
observation. `tests/test_the_query_only_arm.py` asserts the shape against the real `evals.score`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..contract.answer_contract import (
    NO_COVERAGE_CLAIM,
    assert_answer_is_reportable,
    status_kind,
)
from ..core.cli_common import code_sha
from ..core.spend import Spend
from .run_manifest import (
    chart_hash,
    experiment_config_hash,
    model_identity,
    prompt_asset_manifest,
)
from .tools.toolbox import build_tool_schemas

#: Recorded in the manifest and folded into `experiment_config_hash`, so this arm can never collide
#: with an agent arm that happens to share a spec, model and seed.
RUNNER = "query-only"

#: How many hits one term may contribute. High enough that a common term in a 300-note chart is not
#: truncated: a capped search would hand the model a different corpus than the one the terms select,
#: and the arm's whole claim is that it saw every hit.
MAX_HITS = 100_000

INSTRUCTION = """You are given the COMPLETE text of every document in this patient's chart that
matches the search terms below. There is nothing else to read and no tool to call: answer from what
is here, or use the specification's abstention outcome if what is here does not establish an answer.

SEARCH TERMS USED: {terms}
DOCUMENTS RETRIEVED: {n_hits} of {n_docs} in the chart

Call submit_answer exactly once."""


def hit_set(chart, terms) -> list[str]:
    """Every note id any term matches, through the CORPUS's own matcher.

    `chart.search` and not a substring test written here: the runtime's matcher is notation-tolerant,
    so `adeno-carcinoma`, `adeno carcinoma` and `adeno_carcinoma` are one term to an agent run and
    would be three to a naive scan. The same reasoning `improvement/prior.py` gives for pricing terms
    against this matcher applies with more force here, where the hit set IS the arm.
    """
    out: set[str] = set()
    for term in terms:
        if not str(term).strip():
            continue
        for hit in chart.search(str(term), False, None, None, None, max_hits=MAX_HITS):
            nid = str(getattr(hit, "note_id", ""))
            if nid:
                out.add(nid)
    return sorted(out)


def _documents(chart, note_ids) -> tuple[str, int]:
    """The hit documents in full, and how many characters that came to."""
    parts, n = [], 0
    for nid in note_ids:
        try:
            text = (Path(chart.dir) / f"{nid}.txt").read_text(encoding="utf-8")
        except OSError:
            continue
        n += len(text)
        parts.append(f"===== {nid} =====\n{text}")
    return "\n\n".join(parts), n


def run_query_only(*, spec, corpus, patient_id: str, out_dir, model, terms,
                   max_usd: float = 1.0, run_id: str | None = None,
                   prior_asset: dict | None = None) -> dict:
    """One patient, one call. Returns the manifest and writes it beside the agent arms' manifests.

    `terms` is the keyword list under test — the contract's declared terms, or a measured prior's.
    Empty is allowed and is its own arm: an extractor with no terms retrieves nothing and must
    abstain, which is the floor under the floor.
    """
    t0 = time.time()
    chart = corpus.chart(patient_id)
    docs, _ = chart.list_documents(limit=100_000)
    ids = hit_set(chart, terms)
    body, n_chars = _documents(chart, ids)

    schemas = build_tool_schemas(spec)
    submit = [s for s in schemas if s["function"]["name"] == "submit_answer"]
    if not submit:
        raise ValueError("build_tool_schemas produced no submit_answer; the contract cannot be "
                         "answered")
    system = spec.as_prompt_block(view="full") + "\n\n" + INSTRUCTION.format(
        terms=", ".join(str(t) for t in terms) or "(none)",
        n_hits=len(ids), n_docs=len(docs))

    spend = Spend(max_usd=max_usd, model=model_identity(model))
    bound = model.bind_tools([s["function"] for s in submit])
    reply = bound.invoke([{"role": "system", "content": system},
                          {"role": "user", "content": body or "(no document matched any term)"}])
    if usage := getattr(reply, "usage_metadata", None):
        spend.add(usage)
    calls = list(getattr(reply, "tool_calls", ()) or ())
    submitted: dict[str, Any] = dict(calls[0].get("args") or {}) if calls else {}

    # NO GATE, so no acceptance and no coverage claim — stated rather than defaulted. A missing
    # `submit_answer` is the model declining to use its one call, which is a real outcome and not a
    # crash: it scores as an abstention with an undeclared kind, exactly as a stopped agent run does.
    status = str(submitted.get("status") or "NO_ANSWER")
    answer = {"status": status,
              "status_kind": status_kind(spec, status) or "undeclared",
              "value": dict(submitted.get("value") or {}),
              "reasoning": str(submitted.get("reasoning") or ""),
              "evidence": [],
              "proof_basis": "UNGATED",
              "witness_count": 0}
    assert_answer_is_reportable(answer, spec)

    rid = run_id or patient_id
    manifest = {
        "run_id": rid, "patient_id": patient_id, "runtime": RUNNER,
        "spec_id": spec.spec_id, "spec_hash": spec.spec_hash, "spec_version": spec.spec_version,
        "model": model_identity(model),
        "model_temperature": getattr(model, "temperature", None),
        "code_sha": code_sha(), "chart_hash": chart_hash(chart.dir),
        # The SAME block the agent arm records, so a reader comparing two arms compares one shape.
        # `tool_schemas` is the single tool that was bound; there is no skill stack, no prior in the
        # prompt and no site mapping, and each of those is an explicit `None` rather than absent.
        "prompt_assets": prompt_asset_manifest(spec, tool_schemas=submit,
                                               bound_tool_names={"submit_answer"}),
        # WHAT THIS ARM IS. The terms are the intervention, so they are recorded in full: two
        # query-only arms differing only in their term list must be tellable apart afterwards.
        "query": {"terms": [str(t) for t in terms], "n_terms": len(list(terms)),
                  "n_hits": len(ids), "n_documents_in_chart": len(docs),
                  "hit_fraction": round(len(ids) / max(len(docs), 1), 4),
                  "n_chars_read": n_chars, "note_ids": ids,
                  "prior": dict(prior_asset) if prior_asset else None},
        "usage": {"llm_calls": 1,
                  "prompt_tokens": spend.prompt, "cached_tokens": spend.cached,
                  "completion_tokens": spend.completion,
                  "total_tokens": spend.prompt + spend.completion},
        "n_model_calls": 1, "max_model_calls": 1,
        # `Spend.report()`, the same producer the agent arm uses, so `RunRecord.cost_usd` reads
        # `spend.usd` here exactly as it does there — including `None` for an unpriced model, which
        # is unknown and never 0.0.
        "spend": spend.report(),
        "answer": answer,
        # NO GATE EXISTS HERE. `False` is the honest value and it is not a failure of this arm:
        # `gate_validated_rate` reads 0 for it, which is what a reader should see.
        "gate_validated": False, "coverage_gate_validated": False,
        "coverage_note": NO_COVERAGE_CLAIM,
        "termination_reason": "ANSWER_SUBMITTED" if calls else "STOPPED_WITHOUT_ANSWER",
        "degradation": {"no_tool_call_recoveries": 0, "undeclared_tool_calls": 0,
                        "rejection_loop_stopped": False, "marker_catalogue_incomplete": False,
                        "coverage_unreachable": False, "runtime_or_provider_errors": 0,
                        "model_call_limit_without_answer": not calls},
        "evidence": [],
        "elapsed_s": round(time.time() - t0, 1),
    }
    manifest["experiment_config_hash"] = experiment_config_hash({
        "spec_hash": manifest["spec_hash"],
        "prompt_assets": manifest["prompt_assets"],
        "model": manifest["model"],
        "model_temperature": manifest.get("model_temperature"),
        "max_usd": max_usd,
        "code_sha": manifest["code_sha"],
        # THE TWO KEYS THAT KEEP THIS ARM DISTINCT. `runner` cannot collide with an agent arm even if
        # every other input matched; `terms` is the intervention itself.
        "runner": RUNNER,
        "terms": sorted(str(t) for t in terms),
    })
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rid}.manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    path.chmod(0o600)
    return manifest
