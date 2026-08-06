"""Assemble the manifest a finished run leaves behind: the numbers a reader who was not
there has to be able to audit, in the shape a filter over a DIRECTORY of runs can read.

WHY THIS IS A MODULE AND NOT THE TAIL OF `run()`
------------------------------------------------
Every field here exists because a question was asked over a directory of finished runs and
the answer was only recoverable by replaying JSONL — which is to say, was never computed.
The replan rate, the seed provenance, the degradation block: each one is a number somebody
read wrongly once. Assembling them beside the loop that produces them invites the loop's
convenience to decide the record's shape; assembling them here forces every addition to
answer "what would a reader conclude from this six months from now".

It is a pure function of what the run finished with. It reads no ledger it was not handed
and it decides nothing — in particular it never recomputes a verdict, because a manifest
that can disagree with the gate is a second gate with better formatting.

A FIELD THAT SUMMARISES TRACE EVENTS IS DERIVED FROM THEM, BY ONE FUNCTION
-------------------------------------------------------------------------
The replan block used to be read off counters `graph.py` incremented as the run went. The
trace held the same quantities as events. Two counters for one quantity, and on the first
true end-to-end run they read differently in a way nobody could see: 14 `plan_revision`
events, 13 of them `applied: true`, against a manifest saying `n_revisions_applied: 0`.

Both were arithmetically right. `applied` on the event means the revision was ADMISSIBLE;
`n_revisions_applied` means it MOVED RETRIEVAL, and on that run all 13 admitted revisions
moved nothing — the agent kept re-promoting types already at `read_all`. The damage was not
the arithmetic. It was that the manifest had no field for "how many times did the agent
reach for this channel", so a run that reached 14 times and a run that never reached
rendered identically as `replan_rate: 0.0`, and 0.0 was written up as "the model ignores the
replanning channel".

So `replan_from_trace` below is the single definition of every one of those numbers, it
reads the events and nothing else, and `build_manifest` publishes exactly what it returns.
The runtime counters still exist — `graph.py` owns them — but they are no longer published:
they are CROSS-CHECKED against the derivation, and the comparison ships inside the block, so
the next divergence is visible to a reader who never runs the test suite.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

# ==========================================================================================
#                      THE REPLAN BLOCK, DERIVED FROM THE TRACE EVENTS
# ==========================================================================================
#: Event kinds the derivation reads. Named here so a rename in `graph.py` breaks loudly at
#: the one place that reads them rather than silently deriving zero from a kind that no
#: longer exists — which is the same failure mode, one layer down, as the counter drift.
EV_REFLECT = "reflect"
EV_REVISION = "plan_revision"
EV_TRIGGER = "trigger"
EV_REFUSED_OPEN = "plan_refused_open"
EV_ASSERTED_REPLAN = "model_asserted_replan"

def _moved_retrieval(outcome: Mapping[str, Any]) -> bool:
    """Did this revision change what the agent may open or is told to search?

    The same predicate as `coverage_planner.RevisionOutcome.changed_retrieval`, restated over
    the serialised event because THAT is what a reader of a finished run holds. Thread
    bookkeeping is deliberately excluded: resolving a thread is not a replan.
    """
    return bool(outcome.get("terms_added") or outcome.get("types_promoted"))

def replan_from_trace(events: Iterable[Mapping[str, Any]]) -> dict:
    """Every replan number of one run, recomputed from that run's own trace events.

    THE ONE DEFINITION. Nothing else in this tree may count these quantities; a second
    accumulator beside this one is the defect this function exists to make impossible.

    The counts are split because they answer different questions and a single "applied"
    could not carry all of them:

      n_revision_requests      the agent asked. THE NUMBER THAT WAS MISSING. A zero
                               `replan_rate` with a high request count means the plan was
                               already adequate for what the agent wanted; a zero rate with
                               a zero request count means the channel went unused. Those are
                               opposite diagnoses and they used to render identically.
      n_revisions_admitted     the request passed monotonicity, budget and redundancy — the
                               trace event's own `applied` flag, under a name that does not
                               promise the plan moved.
      n_revisions_applied      the retrieval scope actually moved. This is the numerator of
                               `replan_rate` and its meaning is unchanged.
      n_revisions_no_op        admitted and moved nothing. Re-promoting a type already at
                               `read_all`, re-opening an open thread. Invisible before.
      n_revisions_refused      refused whole.
      n_revisions_partly_refused
                               admitted, but carrying refusal prose on part of the request.
                               The agent was told "no" about something and no number said so.
    """
    evs = list(events)
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for e in evs:
        by_kind.setdefault(str(e.get("kind", "")), []).append(e)

    reflections = len(by_kind.get(EV_REFLECT, []))
    revisions = by_kind.get(EV_REVISION, [])
    outcomes = [dict(e.get("outcome") or {}) for e in revisions]
    admitted = [o for e, o in zip(revisions, outcomes) if e.get("applied")]
    moved = [o for o in admitted if _moved_retrieval(o)]
    refused = [o for e, o in zip(revisions, outcomes) if not e.get("applied")]

    triggers: dict[str, int] = {}
    for e in by_kind.get(EV_TRIGGER, []):
        k = str(e.get("trigger", ""))
        triggers[k] = triggers.get(k, 0) + 1

    def _rate(n: int) -> float | None:
        return round(n / reflections, 4) if reflections else None

    return {
        "n_reflections": reflections,
        "n_revision_requests": len(revisions),
        "n_revisions_admitted": len(admitted),
        "n_revisions_applied": len(moved),
        "n_revisions_no_op": len(admitted) - len(moved),
        "n_revisions_refused": len(refused),
        "n_revisions_partly_refused": sum(1 for o in admitted if o.get("refused")),
        # Applied revisions that CHANGED RETRIEVAL, over reflections. Thread bookkeeping is
        # deliberately not counted: resolving a thread is not a replan, and counting it would
        # reinflate the metric with no-ops.
        "replan_rate": _rate(len(moved)),
        # NOT the replan rate and it must never be read as one. This one says the agent
        # REACHED for the channel; `replan_rate` says the plan MOVED. A design conclusion
        # about whether to ship an agent or a workflow needs both, and reading the second
        # while believing it was the first is exactly how a wrong conclusion got written.
        "request_rate": _rate(len(revisions)),
        "terms_added_by_reflection": sum(len(o.get("terms_added") or []) for o in outcomes),
        "types_promoted": sum(len(o.get("types_promoted") or []) for o in outcomes),
        "threads_opened_by_reflection": sum(len(o.get("threads_opened") or [])
                                            for o in outcomes),
        "triggers_fired": triggers,
        "plan_refused_opens": len(by_kind.get(EV_REFUSED_OPEN, [])),
        "model_asserted_replan": len(by_kind.get(EV_ASSERTED_REPLAN, [])),
        "derived_from": "trace_events",
    }

#: Derived key -> the `RunCounters` attribute that used to be published in its place. The
#: cross-check is by name so a counter somebody renames shows up as a missing attribute here
#: rather than as a silent zero — the same reason `RunCounters` is a dataclass.

def chart_hash(patient_dir) -> str:
    """A content hash over the documents a run could read. Empty when it cannot be taken.

    NAMES AND CONTENT BOTH, because both change what a run sees: the filename carries the
    document type and the date, which is what `list_documents` and every date filter work off,
    and the body is what a search matches. Hashing one without the other leaves a way to edit a
    chart invisibly.

    Why a run needs it at all: `tools/generate_corpus.py` is deterministic, so a chart whose
    content moves under a stable `patient_id` moved because somebody edited it. That is
    tolerable on the development charts and is exactly what must not happen quietly on a
    held-out one after a result has been scored against it.

    Returns "" rather than raising. A manifest that could not be written because a hash could
    not be taken is a run lost to bookkeeping.
    """
    import hashlib
    from pathlib import Path
    d = Path(patient_dir)
    if not d.is_dir():
        return ""
    h = hashlib.sha256()
    try:
        for f in sorted(d.glob("*.txt")):
            h.update(f.name.encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    except OSError:
        return ""
    return h.hexdigest()[:16]

def model_identity(model) -> str:
    """WHICH MODEL, as a string two runs of one arm agree on.

    `getattr(model, "model_name", "") or str(model)` was the fallback, and `str()` on a LangChain
    chat model that does not set `model_name` renders the Python object's repr — including its
    MEMORY ADDRESS. So `manifest["model"]` differed between two runs of the same arm, and once
    `experiment_config_hash` started being read as the arm's identity that made it a per-run id:
    `eval compare` would refuse every genuine pairing as a mixture of arms.

    Caught by running the same patient twice through the real runtime and comparing the two
    manifests, which is the only way this class of thing surfaces — every fixture that builds a
    manifest by hand types a model name and the substitution never happens.

    The class name is a real if coarse identity: stable across runs, and wrong in no way that a
    reader could mistake for a provider model id.
    """
    return str(getattr(model, "model_name", "") or type(model).__name__)

def experiment_config_hash(config: dict) -> str:
    """One value over everything that makes two runs the same arm.

    NO ALLOWLIST. Whatever the caller passes is part of the arm, because the failure this
    guards against is a field that mattered and was left out — and a hash with an allowlist
    inside it decides that question in a place nobody looks. The caller assembles the dict at
    the point where it knows what varied.

    Canonical JSON, sorted keys, so two dicts describing one arm hash alike. The alternative to
    this field is a reader comparing six identity keys by eye and being right every time.
    """
    import hashlib
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

def _open_gaps_text() -> str:
    """The gap-ledger prompt and tool description as one string, for hashing.

    Imported lazily: `run_manifest` is reachable from planes that must not import langchain, and
    `agent` does at module scope.
    """
    from .agent import OPEN_GAPS_PROMPT, OPEN_GAPS_TOOL_DESCRIPTION
    return OPEN_GAPS_PROMPT + "\n" + OPEN_GAPS_TOOL_DESCRIPTION


def prompt_asset_manifest(spec, runtime_profile_asset=None, skill_stack=None,
                          tool_schemas=None, retrieval_prior=None,
                          site_mapping=None, task_context: str = "",
                          bound_tool_names: set[str] | None = None,
                          prompt_blocks: Mapping[str, int] | None = None) -> dict:
    """The identity of every prompt block whose content can change a run's answer.

    Lives here rather than in `agent` because a manifest field belongs with the manifest, and so
    that a second front end building its own answer dict can reach the same block.

    Every entry is content-hashed and every entry carries whether anybody has signed it off. Both
    are `False` today, and that is the honest state: the code tables were recalled by a model, the
    document concepts are unmeasured by construction, and no registrar has read either. A manifest
    that recorded the names without that would let a reader assume otherwise.

    `skill_stack` IS THE STACK THAT WAS RENDERED, and it must be the same object `agent` passed to
    `skills_block`. Deriving it here from the profile instead was a real defect, caught on the
    first live run against the synthetic corpus: `--skills tactics=tactic-coverage-pool` reached
    the model and the manifest recorded the profile's default, so the artifact said two arms of a
    retrieval ablation used identical guidance while their prompts differed by a whole card. A
    manifest that names the wrong asset is worse than one that names none — the second is a gap a
    reader can see. `None` still means "the profile's stack", which is what every recorded run
    used before an override existed.
    """
    from ..contract.code_tables import table_manifest
    from ..contract.skills import skills_manifest
    from .document_concepts import concepts_manifest
    from .runtime_profiles import runtime_policy_skills

    module_id = getattr(runtime_profile_asset, "module_id", "") or ""
    try:
        stack = skill_stack if skill_stack is not None else runtime_policy_skills(module_id)
        skills = skills_manifest(stack)
    except Exception as exc:            # noqa: BLE001 - a manifest must not take down a run
        skills = [{"error": f"{type(exc).__name__}: {exc}"}]
    return {
        "value_domain": table_manifest(spec),
        # WHICH STATIC PROMPT BLOCKS THE MODEL WAS SHOWN, and how much each one contributed. The
        # prompt was a ten-term `+` chain until 2026-08-04, so no run could drop a block and no
        # manifest had anything to say about the ten; `--prompt-blocks` makes each one ablatable and
        # this is the record that turns an ablation into an arm. It rides in `prompt_assets`, which
        # `experiment_config_hash` already takes, so dropping `skills` (9,117 characters of a 20,531
        # character prompt) moves the arm hash without that function learning a new field.
        #
        # The SIZES, not a content hash: the blocks' own identities are already covered here —
        # `value_domain`, `document_concepts`, `skills`, `retrieval_prior`, `additional_task_context`
        # — and what was missing is which of them rendered at all and how large it was. A selected
        # block that rendered nothing records `0`, which is a different fact from not being selected:
        # STORE.390 declares no value domain, so a run over it renders that block empty while the
        # arm still includes it.
        #
        # `None` when the caller did not supply one, on the `retrieval_prior` precedent: `query_only`
        # builds its own prompt and never assembles this registry, and "no selection recorded" must
        # not read as "every block".
        # THE SELECTION ONLY. The per-block character counts used to ride here and this dict is
        # hashed WHOLESALE by `experiment_config_hash`, which made the arm hash a per-run id:
        # `_task` renders `TASK.format(patient=...)`, the shipped corpus has both 7- and
        # 6-character ids, and two patients of one arm therefore hashed differently (SYN0001 616
        # chars / c829dca0, SYNK01 615 / a32d5c96). Measured downstream: `derive_baseline_key`
        # returned MIXED and `eval compare` returned NOT_COMPARABLE over the DEFAULT cohort, so
        # every batch would have been unscoreable. Same defect `model_identity` was written to fix.
        #
        # The distinction is the one `ARM_PARAMETERS` / `WITHIN_ARM_PARAMETERS` already draws: the
        # selection DEFINES the arm, the sizes are a per-run observation. They live apart, and the
        # sizes are in the manifest's own `prompt_block_chars` where nothing hashes them.
        "blocks": ({"selected": list(prompt_blocks)} if prompt_blocks is not None else None),
        # THE MEASURED PRIOR THIS RUN WAS SHOWN, or an explicit `None`. Recorded because a prior is
        # prompt content: two arms differing only in which scan they were told about are otherwise
        # indistinguishable afterwards, and "accuracy rose" and "the prompt changed" would be the
        # same observation. `None` rather than an absent key — an absent entry and an entry saying
        # "no prior" are different claims and only one survives a reader asking which arm this was.
        "retrieval_prior": ({
            "asset_id": retrieval_prior.asset_id,
            "version": retrieval_prior.version,
            "status": retrieval_prior.status,
            "content_hash": retrieval_prior.content_hash,
            "n_patients": retrieval_prior.measured.n_patients,
            "n_notes": retrieval_prior.measured.n_notes,
            "spec_id": retrieval_prior.measured.spec_id,
            "labelling_model": retrieval_prior.measured.model,
        } if retrieval_prior is not None else None),
        "document_concepts": concepts_manifest(),
        "skills": skills,
        # THE TOOL SURFACE IS PROMPT CONTENT. Every schema's `description` is rendered into
        # every model call, and `submit_answer`'s is now BUILT FROM THE CONTRACT -- its enum
        # and the prose under it come from the declared outcome space -- so the surface really
        # does differ between contracts and between arms. Nothing hashed it until 2026-08-03,
        # which is the same defect this block was added to fix, one entry further along.
        #
        # `None`, not `{}`, when no surface was supplied: a caller may build its own answer
        # dict, and an absent surface is a different fact from an unhashed one.
        "tool_surface": _tool_surface(tool_schemas, bound_tool_names),
        # WHAT ELSE WENT INTO THE PROMPT, AND WHAT ELSE SHAPED THE PLAN. Both were switches on
        # `run_patient` that no manifest field recorded, so `experiment_config_hash` — the read
        # side's only discriminator since `evals.BaselineKey` started carrying it — could not tell
        # those arms from the baseline.
        #
        # `additional_task_context` is appended verbatim to the system prompt (`agent.py:1688`) and
        # its only caller is `conflict_refinement.py`, injecting the brief that DEFINES the
        # refinement arm. The hash, never the text: this is unbounded operator prose written into a
        # file that sits beside patient-derived output, and a manifest is not the place to copy it.
        "additional_task_context": _text_identity(task_context),
        # THE OPEN-GAP LEDGER'S TEXT, which this repo now owns. `_tool_surface` below records
        # `write_todos` as bound-but-not-schema-hashed, on the grounds that it "belongs to the
        # library, whose version `code_sha` already covers". That was true while the text was the
        # library's default and stopped being true on 2026-08-05, when `TodoListMiddleware` began
        # taking `system_prompt=OPEN_GAPS_PROMPT`. It is prompt content that changes behaviour —
        # the whole reason for overriding it is that the default produced zero calls in 514 runs —
        # so two arms differing only in this text would otherwise hash identically, which is the
        # measurement failure `blocks` above was split apart to avoid.
        "open_gaps_prompt": _text_identity(_open_gaps_text()),
        # `site_mapping` reaches `CoverageLedger` and `plan_from_spec`, so it changes the
        # stratification, the plan and therefore the gate. `mapping_hash` has existed since the
        # mapping did and reached no manifest.
        "site_mapping": ({"corpus_id": site_mapping.corpus_id,
                          "mapping_hash": site_mapping.mapping_hash,
                          "n_types": site_mapping.n_types,
                          "provenance": site_mapping.provenance}
                         if site_mapping is not None else None),
        "any_signed_off": False,
    }

def _text_identity(text: str) -> dict | None:
    """A content hash and a length for prompt text, or `None` when there was none.

    `None` rather than `{"n_chars": 0}`: an absent block and an empty one are the same run, but a
    manifest that predates this field and one that records "no brief" are different facts, and only
    the explicit `None` lets a reader tell them apart.
    """
    if not (text or "").strip():
        return None
    import hashlib
    return {"n_chars": len(text),
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}

def _tool_surface(schemas, bound: set[str] | None = None) -> dict | None:
    """Everything the model can reach, and an honest account of how much of it is hashed.

    THIS RECORDED SEVEN AND NINE WERE BOUND. `revise_plan` is added by `run_chart_review` and
    `write_todos` by `TodoListMiddleware`; both reach the model, and a manifest that understates the
    reachable surface is read by `undeclared-tool-audit` and by anyone asking what a run could do.

    `not_schema_hashed` is an ADMITTED limit rather than a silent one. Only `build_tool_schemas`
    produces schemas here; `revise_plan`'s description is a module constant and `write_todos` belongs
    to the library, whose version `code_sha` already covers. A reader has to be able to tell a hashed
    name from a merely listed one, or the hash looks like it covers all nine.
    """
    if schemas is None:
        return None
    import hashlib
    hashed = [s["function"]["name"] for s in schemas]
    names = sorted(set(hashed) | set(bound or ()))
    # The names go INTO the hash alongside the schemas: a tool that appears with no schema still
    # changes the surface, and two arms differing only by such a tool must not hash alike.
    blob = json.dumps({"schemas": schemas, "names": names},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"names": names, "n_tools": len(names),
            "n_schema_hashed": len(hashed),
            "not_schema_hashed": sorted(set(names) - set(hashed)),
            "content_hash": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]}
