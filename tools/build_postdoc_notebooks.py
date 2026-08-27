"""Build the three small, teaching-first postdoc walkthrough notebooks.

The notebooks deliberately tell one story each.  Plumbing cells are collapsed by default so a
reader sees the chart-review ideas and results before implementation details.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str):
    return new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str, *, hidden: bool = False):
    cell = new_code_cell(dedent(source).strip() + "\n")
    if hidden:
        cell.metadata["jupyter"] = {"source_hidden": True}
        cell.metadata["tags"] = ["hide-input"]
    return cell


def notebook(cells: list, *, title: str):
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "acr": {
                "audience": "postdoctoral researcher",
                "title": title,
                "version": 2,
                "teaching_style": "one question, one mental model, one takeaway",
            },
        },
    )


NB1 = notebook(
    [
        md(
            """
            # 01 — Run the chart-review agent

            This notebook answers one question: **what does the agent do, and what answer did it
            return?** Analysis comes later.

            ## The agent in one picture

            ```text
            task
              ↓
            choose the next useful chart action
              ↓
            call a chart tool → observe the result → update working state
              ↑                                      ↓
              └──────────────── repeat ───────────────┘
                                                      ↓
                                                submit answer
            ```

            Codex supplies the ReAct-style loop. ACR gives it a **chart-only boundary**. The agent
            cannot use a shell, browser, or arbitrary filesystem access.

            | Job | Tool | Plain meaning |
            |---|---|---|
            | Find | `list_documents` | See which notes exist. |
            | Find | `search` | Find notes containing a term. |
            | Read | `read` | Open one note. |
            | Explain a consequential choice | `note_decision` | Record the question, choice, reason, alternatives, and claimed basis. |
            | Judge evidence | `record_finding` | Say whether one note can establish the requested field. |
            | Preserve proof | `record_evidence` | Save the exact supporting span. |
            | Finish | `submit_answer` | Submit only after the evidence gate is satisfied. |

            `note_decision` is a short audit explanation, **not private chain-of-thought**.

            We use two instruction sets:

            - **Task only:** asks for the diagnosis date and output format, but withholds the
              clinical evidence/conflict rules.
            - **Task + policy:** adds explicit rules for what counts as evidence, which date wins,
              and what must be cited before submission.

            Existing real-provider runs are reused by default, so simply reading this notebook
            makes no paid call. Set `ACR_TUTORIAL_MODE=live` to run Luna again.
            """
        ),
        md(
            """
            ## Choose what to run

            The live pilot is deliberately small: two paired synthetic cases × two instruction
            sets × Luna. `SYN0001` has an early same-day physician diagnosis; `SYNX03` does not.
            That single difference tests whether the agent handles ambiguous cytology correctly.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import asyncio
            import json
            import os
            import subprocess

            from IPython.display import Markdown, display
            from acr.mvp.langtrace_io import LangtraceClient
            from acr.mvp.ledger import SemanticaLedger
            from acr.mvp.reconstruct import reconstruct_run
            from acr.mvp.reconstruction_llm import AuditedLiteLLM
            from acr.mvp.runner import run_patient

            START = Path.cwd().resolve()
            ROOT = START if (START / "pyproject.toml").is_file() else START.parent
            assert (ROOT / "pyproject.toml").is_file(), "Start Jupyter from the repo or notebooks/"

            SEED_ROOT = ROOT / "runs/policy-experiment-20260827"
            SEED_LEDGER = SEED_ROOT / "experiment-ledger.json"
            LIVE_ROOT = Path(os.environ.get("ACR_TUTORIAL_RUN_ROOT", ROOT / "runs/postdoc-study"))
            MODE = os.environ.get(
                "ACR_TUTORIAL_MODE", "reuse" if SEED_LEDGER.is_file() else "live"
            ).lower()
            assert MODE in {"reuse", "live"}
            EXPERIMENT_ROOT = SEED_ROOT if MODE == "reuse" else LIVE_ROOT
            LEDGER_PATH = SEED_LEDGER if MODE == "reuse" else LIVE_ROOT / "ledger.json"
            SPEC = ROOT / "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"

            PLAN = [
                {"case": case, "arm": arm, "model": "openai/gpt-5.6-luna"}
                for case in ("SYN0001", "SYNX03")
                for arm in ("task_only", "policy_bundle")
            ]

            def short_model(value):
                return "Terra" if "terra" in str(value).lower() else "Luna"

            def table(rows, columns):
                def clean(value):
                    return " ".join(str(value).split()).replace("|", "/")
                return "\n".join([
                    "| " + " | ".join(label for _, label in columns) + " |",
                    "|" + "|".join("---" for _ in columns) + "|",
                    *("| " + " | ".join(clean(row.get(key, "")) for key, _ in columns) + " |"
                      for row in rows),
                ])

            codex_version = subprocess.run(
                ["codex", "--version"], capture_output=True, text=True, check=True
            ).stdout.strip()
            display(Markdown(f"**Mode:** `{MODE}` · **Codex:** `{codex_version}`"))
            if MODE == "live":
                display(Markdown("**Live plan:**"))
                display(Markdown(table(PLAN, [
                    ("case", "Case"), ("arm", "Instruction set"),
                    ("model", "Review model")
                ])))
            else:
                display(Markdown(
                    "Loading the available historical examples with complete sealed metadata."
                ))
            """,
            hidden=True,
        ),
        md(
            """
            ## Run

            In live mode the collapsed cell below performs the closed loop:

            ```text
            Luna chart review → local Langtrace → two Luna reconstructions
            → verifier agreement → selected analysis → Semantica
            ```

            The reconstruction work is packaged here only so Notebooks 2 and 3 have something to
            read. This notebook does not analyze it.
            """
        ),
        code(
            r"""
            if MODE == "live":
                assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is required"
                langtrace_host = os.environ.get("LANGTRACE_API_HOST", "http://127.0.0.1:3100")
                langtrace_project = os.environ.get("LANGTRACE_PROJECT_ID", "acr_chart_review")
                langtrace_key = os.environ.get("LANGTRACE_API_KEY", "")
                client = LangtraceClient(
                    api_key=langtrace_key,
                    api_host=langtrace_host,
                    project_id=langtrace_project,
                )
                ledger = SemanticaLedger(LEDGER_PATH)
                for item in PLAN:
                    run_dir = await asyncio.to_thread(
                        run_patient,
                        SPEC,
                        ROOT / "corpus/patients" / item["case"],
                        LIVE_ROOT,
                        model=item["model"],
                        base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        task_arm=item["arm"],
                        langtrace_api_key=langtrace_key,
                        langtrace_api_host=langtrace_host,
                        langtrace_project_id=langtrace_project,
                    )
                    runner = json.loads((run_dir / "runner_meta.json").read_text())
                    review = client.get_review(runner["langtrace_trace_id"])
                    summary = reconstruct_run(
                        review,
                        ledger,
                        AuditedLiteLLM(
                            model="openrouter/openai/gpt-5.6-luna",
                            api_key=os.environ["OPENROUTER_API_KEY"],
                            temperature=0.0,
                        ),
                        passes=2,
                        artifact_dir=run_dir / "analyses",
                        reconstructor_identity="openrouter/openai/gpt-5.6-luna",
                        max_attempts_per_pass=3,
                    )
                    assert summary["drift"]["alignment_agrees"] is True
                    selected = summary["analyses"][0]["analysis_id"]
                    ledger.select_analysis(
                        review.run_id,
                        selected,
                        selected_by="postdoc-notebook-01",
                        reason="Two Luna passes agreed on episode alignment.",
                        provenance="DETERMINISTIC_DERIVED",
                    )
            else:
                assert LEDGER_PATH.is_file(), "No reusable run set was found"
            """,
            hidden=True,
        ),
        md(
            """
            ## What came back?

            This is intentionally the only result table in Notebook 1. Synthetic gold is shown
            only because these tutorial cases were designed with a known answer.
            """
        ),
        code(
            r"""
            rows = []
            skipped_incomplete = 0
            for result_path in sorted(EXPERIMENT_ROOT.glob("*/result.json")):
                run_dir = result_path.parent
                if not (run_dir / "runner_meta.json").is_file() \
                        or not (run_dir / "task_presentation.json").is_file():
                    skipped_incomplete += 1
                    continue
                result = json.loads(result_path.read_text())
                runner = json.loads((run_dir / "runner_meta.json").read_text())
                presentation = json.loads((run_dir / "task_presentation.json").read_text())
                case = str(result.get("patient_id") or run_dir.name.split("_", 2)[1])
                truth_path = ROOT / "corpus/patients" / case / "_ground_truth.json"
                truth = json.loads(truth_path.read_text()) if truth_path.is_file() else {}
                gold = (((truth.get("ground_truth") or {}).get(
                    "STORE.390.date_of_initial_diagnosis") or {}).get("value"))
                value = result.get("value") or {}
                answer = value.get("date_of_initial_diagnosis") if isinstance(value, dict) else value
                rows.append({
                    "case": case,
                    "instructions": (
                        "Task + policy" if presentation.get("arm_id") == "policy_bundle"
                        else "Task only"
                    ),
                    "model": short_model(runner.get("model")),
                    "answer": answer,
                    "expected": gold,
                    "match": "✓" if answer == gold else "✗",
                })

            assert rows, "No completed result.json files were found"
            display(Markdown(table(rows, [
                ("case", "Case"), ("instructions", "Instructions"), ("model", "Model"),
                ("answer", "Agent answer"), ("expected", "Synthetic gold"), ("match", "Match?")
            ])))
            if MODE == "reuse":
                display(Markdown(
                    f"**Read this correctly:** these are {len(rows)} historical runs with complete "
                    "metadata, not a balanced accuracy experiment. "
                    f"{skipped_incomplete} incomplete result director{'y was' if skipped_incomplete == 1 else 'ies were'} "
                    "not treated as a run. Notebook 2 now follows one run step by step."
                ))
            """
        ),
    ],
    title="Run the chart-review agent",
)


NB2 = notebook(
    [
        md(
            """
            # 02 — How did the agent reach 2022-03-09?

            We audit one real Luna review of synthetic case `SYNX03`.

            - **Task:** find the earliest diagnosis date.
            - **Agent answer:** `20220309`.
            - **Synthetic gold:** `20220309`.
            - **First conclusion:** the final answer is correct.

            A correct answer can still come from a bad process. We therefore ask: **what did the
            agent look at, what did it judge, and which step is still risky?**

            Think of the raw trace as a long receipt. It proves what happened. The Decision Chain
            groups that receipt into a short set of questions a human can judge.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import json
            import os
            import re

            from IPython.display import Markdown, display
            from acr.mvp.human_review import human_review_view
            from acr.mvp.ledger import SemanticaLedger

            START = Path.cwd().resolve()
            ROOT = START if (START / "pyproject.toml").is_file() else START.parent
            assert (ROOT / "pyproject.toml").is_file(), "Start Jupyter from the repo or notebooks/"

            candidates = [
                ROOT / "runs/notebook-live-20260827/ledger.json",
                ROOT / "runs/postdoc-study/ledger.json",
                ROOT / "runs/policy-experiment-20260827/experiment-ledger.json",
            ]
            LEDGER_PATH = Path(os.environ.get(
                "ACR_AUDIT_LEDGER", next((str(path) for path in candidates if path.is_file()), "")
            ))
            assert LEDGER_PATH.is_file(), "Run Notebook 1 first or set ACR_AUDIT_LEDGER"
            ledger = SemanticaLedger(LEDGER_PATH)

            preferred = (
                "20260827T135252029492Z_SYNX03_STORE_390_date_of_initial_diagnosis_policy_bundle"
            )
            run_id = os.environ.get("ACR_AUDIT_RUN_ID")
            if run_id is None and ledger.selected_analysis(preferred):
                run_id = preferred
            if run_id is None:
                selections = ledger.graph.find_nodes(node_type="AnalysisSelection")
                selected_runs = [
                    str((row.get("metadata") or {}).get("run_id")) for row in selections
                    if "SYNX03" in str((row.get("metadata") or {}).get("run_id"))
                ]
                assert selected_runs, "Select a reconstructed SYNX03 run first"
                run_id = selected_runs[-1]
            analysis_id = ledger.selected_analysis(run_id)
            assert analysis_id, "This walkthrough requires one explicitly selected reconstruction"

            run_dir = LEDGER_PATH.parent / run_id
            artifact = ledger.load_analysis_artifact(run_id, analysis_id)
            view = human_review_view(ledger, run_id, analysis_id, run_dir=run_dir)
            events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
            protocol_path = run_dir / "layer2_codex.jsonl"
            protocol_count = sum(1 for line in protocol_path.read_text().splitlines() if line.strip())
            result = json.loads((run_dir / "result.json").read_text())
            episodes = artifact["episodes"]
            cycles = artifact["cycles"]

            def one_line(value, limit=105):
                text = " ".join(str(value or "").split())
                return text if len(text) <= limit else text[: limit - 1] + "…"

            def table(rows, columns):
                def clean(value):
                    return one_line(value, 145).replace("|", "/")
                return "\n".join([
                    "| " + " | ".join(label for _, label in columns) + " |",
                    "|" + "|".join("---" for _ in columns) + "|",
                    *("| " + " | ".join(clean(row.get(key, "")) for key, _ in columns) + " |"
                      for row in rows),
                ])

            def tool_events(name):
                return [row for row in events if row.get("tool") == name]
            """,
            hidden=True,
        ),
        md(
            """
            ## 1. What does the raw trace look like?

            The full trace has 22 observable events. Reading every JSON field is possible, but it
            is not how a reviewer should start. First group the receipt by purpose.
            """
        ),
        code(
            r"""
            inventory_pages = [len((row.get("result") or {}).get("documents") or [])
                               for row in tool_events("list_documents")]
            queries = [str((row.get("args") or {}).get("query")) for row in tool_events("search")]
            hit_notes = {
                str(hit.get("note_id"))
                for row in tool_events("search")
                for hit in (row.get("result") or {}).get("hits") or []
            }
            process = [
                {"stage": "Inventory", "what happened": (
                    f"Two pages: {' + '.join(map(str, inventory_pages))} = "
                    f"{sum(inventory_pages)} note headers"
                )},
                {"stage": "Search", "what happened": f"Keywords: {', '.join(queries)}"},
                {"stage": "Candidates", "what happened": f"{len(hit_notes)} unique notes surfaced"},
                {"stage": "Read", "what happened": f"Opened {len(tool_events('read'))} notes"},
                {"stage": "Judge", "what happened": (
                    f"Recorded {len(tool_events('record_finding'))} evidence judgments"
                )},
                {"stage": "Proof", "what happened": (
                    f"Saved {len(tool_events('record_evidence'))} exact evidence spans"
                )},
                {"stage": "Submit", "what happened": "FOUND → 20220309"},
            ]
            display(Markdown(table(process, [("stage", "Stage"), ("what happened", "Raw trace says") ])))

            raw = tool_events("record_finding")[1]
            args = raw["args"]
            compact_event = {
                "seq": raw["seq"],
                "kind": raw["kind"],
                "tool": raw["tool"],
                "args": {
                    "note_id": args.get("note_id"),
                    "standing": args.get("standing"),
                    "because": args.get("because"),
                },
                "result": {
                    "ok": raw.get("ok"),
                    "testimony_ref": (raw.get("result") or {}).get("testimony_ref"),
                    "server_sealed_receipt": bool(
                        (raw.get("result") or {}).get("decision_receipt")
                    ),
                },
            }
            display(Markdown("### One actual event (trimmed only for display)"))
            display(Markdown(
                "```json\n" + json.dumps(compact_event, ensure_ascii=False, indent=2) + "\n```"
            ))
            """
        ),
        md(
            """
            A raw event is excellent evidence: it tells us which note, which tool, which choice,
            and whether the server sealed the record. But it does not tell the reviewer where one
            important judgment ends and the next begins.

            ## 2. Turn the receipt into eight questions

            Luna reconstructs fixed ReAct cycles into **Decision Episodes**. One episode means one
            consequential question that one human can mark right, wrong, or uncertain.

            ```text
            312-note inventory
                    ↓
            4 keyword searches
                    ↓
            3 candidate notes
                    ↓
            2/14 ❌ suspicious   3/9 ✅ biopsy   3/11 ✅ physician diagnosis
                    ↓
            earliest qualifying date = 3/9
                    ↓
            submit 20220309
            ```
            """
        ),
        code(
            r"""
            def human_question(ep):
                subject = ep.get("decision_subject")
                text = (ep.get("material_question") or "").lower()
                if subject == "retrieval_inventory":
                    return "Was the chart inventory complete?"
                if subject == "retrieval_query_batch":
                    return "Which keywords should we use to search for diagnosis evidence?"
                if subject == "retrieval_document_set":
                    return "Which surfaced notes should be opened?"
                if subject == "evidence_item" and "suspicious" in text:
                    return "Can the 2/14 suspicious cytology establish diagnosis?"
                if subject == "evidence_item" and "pathology report" in text:
                    return "Can the 3/9 definitive biopsy establish diagnosis?"
                if subject == "evidence_item" and "physician" in text:
                    return "Can the 3/11 physician diagnosis establish diagnosis?"
                if subject == "evidence_relationship":
                    return "Which of the three dates is the earliest qualifying date?"
                if subject == "case_sufficiency":
                    return "Is there enough evidence to stop and submit?"
                return one_line(ep.get("material_question"), 80)

            def human_choice(ep):
                subject = ep.get("decision_subject")
                decision = str(ep.get("decision") or "")
                if subject == "retrieval_inventory":
                    return f"Inventory all {sum(inventory_pages)} note headers"
                if subject == "retrieval_query_batch":
                    return "Search diagnosis, cancer, malignancy, carcinoma"
                if subject == "retrieval_document_set":
                    return f"Open the {len(tool_events('read'))} surfaced candidate notes"
                if decision == "merely_mentions":
                    return "No — suspicious only"
                if decision == "can_establish":
                    return "Yes — qualifying evidence"
                if subject == "evidence_relationship":
                    return "Choose 2022-03-09"
                if subject == "case_sufficiency":
                    return "Stop and submit 20220309"
                return one_line(decision, 72)

            def provenance_label(ep):
                provenance = ep.get("field_provenance") or {}
                if all(provenance.get(field) == "SELF_REPORTED"
                       for field in ("material_question", "decision", "decision_rationale")):
                    return "Agent said this at runtime"
                if provenance.get("decision") == "DETERMINISTIC_DERIVED_FROM_EXECUTION":
                    return "Action observed; reason reconstructed by Luna"
                return "Luna reconstructed"

            verdicts = {
                "retrieval_inventory": "✓ Complete after page 2",
                "retrieval_query_batch": "⚠ Main audit point: are four terms enough?",
                "retrieval_document_set": "⚠ Notes opened, but selection reason was not recorded",
                "evidence_relationship": "✓ Applies the earliest-qualifying rule correctly",
                "case_sufficiency": "△ Correct, but inherits the search-coverage risk",
            }
            evidence_verdicts = [
                "✓ Correctly rejects ambiguous cytology alone",
                "✓ Definitive pathology qualifies",
                "✓ Physician diagnosis qualifies, but is later",
            ]
            evidence_index = 0
            decision_rows = []
            for index, ep in enumerate(episodes, 1):
                subject = str(ep.get("decision_subject"))
                verdict = verdicts.get(subject, "Review")
                if subject == "evidence_item":
                    verdict = evidence_verdicts[evidence_index]
                    evidence_index += 1
                decision_rows.append({
                    "n": index,
                    "question": human_question(ep),
                    "choice": human_choice(ep),
                    "audit": f"{verdict} · {provenance_label(ep)}",
                })

            display(Markdown(table(decision_rows, [
                ("n", "#"), ("question", "Question for the reviewer"),
                ("choice", "Agent's choice"), ("audit", "Human audit reading + source"),
            ])))
            """
        ),
        md(
            """
            ## 3. Follow the clinical decision

            The three evidence judgments are the heart of the answer:

            1. `2022-02-14`: atypical cells were only **suspicious**; biopsy was recommended. This
               note alone does not establish the date.
            2. `2022-03-09`: the biopsy's final diagnosis is squamous cell carcinoma. This does
               establish the date.
            3. `2022-03-11`: the physician says the mass clinically represents malignancy. This
               also qualifies, but it is later.
            4. The task asks for the **earliest qualifying** date, so `2022-03-09` wins.

            This is the same explanation a careful human reviewer would ask a colleague to give.
            """
        ),
        md(
            """
            ## 4. Where should the human spend time?

            Not on the final date comparison. The weak point is earlier: **the agent chose four
            search terms.** Those calls really happened, and they found the three decisive notes.
            But the policy says what evidence counts; it does not prove that these four terms cover
            every possible clinical wording.
            """
        ),
        code(
            r"""
            search_episode = next(
                ep for ep in episodes if ep.get("decision_subject") == "retrieval_query_batch"
            )
            search_event_ids = set(search_episode.get("source_event_ids") or [])
            search_events = [
                row for row in events if f"layer1:{row.get('seq')}" in search_event_ids
                and row.get("tool") == "search"
            ]
            display(Markdown(
                f"- **Agent's question:** {search_episode['material_question']}\n"
                f"- **Agent's choice:** {search_episode['decision']}\n"
                f"- **Agent's stated reason:** {search_episode['decision_rationale']}\n"
                f"- **Raw proof:** events {', '.join(str(row['seq']) for row in search_events)} "
                f"executed `{', '.join(queries)}` and surfaced {len(hit_notes)} unique notes.\n"
                "- **Reviewer question:** would a different wording escape all four searches?"
            ))
            display(Markdown(
                "**If the answer is yes—or we cannot rule it out—improve the retrieval guideline "
                "here.** Changing the "
                "later date-conflict rule would not fix a missed note."
            ))
            """
        ),
        md(
            """
            ## 5. Is the Decision level better to read?

            Yes—for choosing **where to audit**. No—as a replacement for execution evidence.
            """
        ),
        code(
            r"""
            annotations = artifact["cycle_annotations"]
            if isinstance(annotations, list):
                annotations = {row["cycle_id"]: row for row in annotations}
            assigned = [cycle_id for ep in episodes for cycle_id in ep["source_cycle_ids"]]
            assigned += list(artifact.get("mechanical_cycle_ids") or [])
            assert len(assigned) == len(cycles) == len(set(assigned))
            traceable = sum(bool(ep.get("source_event_ids")) for ep in episodes)
            runtime_explained = sum(
                (ep.get("field_provenance") or {}).get("decision") == "SELF_REPORTED"
                for ep in episodes
            )
            reconstructed_only = len(episodes) - runtime_explained

            display(Markdown(
                f"```text\n{protocol_count} Codex protocol records (harness detail)\n"
                f"→ {len(events)} observable Langtrace events\n"
                f"→ {len(cycles)} fixed ReAct cycles\n"
                f"→ {len(episodes)} human-auditable decisions\n```\n\n"
                f"- {runtime_explained}/{len(episodes)} choices were explicitly stated by the "
                f"agent at runtime.\n"
                f"- {reconstructed_only}/{len(episodes)} choice was recovered from observed "
                "actions; its explanation is visibly labeled as Luna reconstruction.\n"
                f"- {traceable}/{len(episodes)} decisions link back to raw events.\n"
                f"- {len(cycles)}/{len(cycles)} cycles are accounted for exactly once."
            ))
            display(Markdown(
                "**Bottom line:** use the Decision Chain to find the questionable step. Then use "
                "the raw trace and provenance to verify what actually happened."
            ))
            """
        ),
    ],
    title="From raw trace to a human-auditable Decision Chain",
)


NB3 = notebook(
    [
        md(
            """
            # 03 — What does Semantica add?

            Think of Semantica ContextGraph as a **decision card box with an index**.

            Notebook 2 made one run readable. Notebook 3 asks what we can do after many runs have
            been stored as native Semantica Decisions. We keep only four useful questions:

            1. What is on one decision card?
            2. Can we find similar cards?
            3. Did the same question receive different answers?
            4. If a policy changes, which old cards must a human re-check?

            A card never replaces the raw trace. Its provenance pointer is the receipt number that
            lets us go back to the original run.
            """
        ),
        code(
            r"""
            from collections import defaultdict
            from pathlib import Path
            import json
            import os
            import shutil

            from IPython.display import Markdown, display
            from acr.mvp.ledger import PROJECTION_SCHEMA, SemanticaLedger

            START = Path.cwd().resolve()
            ROOT = START if (START / "pyproject.toml").is_file() else START.parent
            assert (ROOT / "pyproject.toml").is_file(), "Start Jupyter from the repo or notebooks/"

            source_paths = [
                ROOT / "runs/policy-experiment-20260827/experiment-ledger.json",
                ROOT / "runs/notebook-live-20260827/ledger.json",
                ROOT / "runs/postdoc-study/ledger.json",
            ]
            source_paths = [path for path in source_paths if path.is_file()]
            assert source_paths, "Run Notebook 1 first"

            output_dir = ROOT / "runs/postdoc-notebook-output"
            output_dir.mkdir(parents=True, exist_ok=True)
            version = PROJECTION_SCHEMA.rsplit(".", 1)[-1]
            LEDGER_PATH = Path(os.environ.get(
                "ACR_INTELLIGENCE_LEDGER", output_dir / f"readable-context-graph-{version}.json"
            ))
            ledger = SemanticaLedger(LEDGER_PATH)

            selected_members = {}
            for source_path in source_paths:
                source = SemanticaLedger(source_path)
                runs = sorted({
                    str((node.get("metadata") or {}).get("run_id"))
                    for node in source.graph.find_nodes(node_type="decision")
                })
                for run_id in runs:
                    available = source.available_analyses(run_id)
                    if not available:
                        continue
                    analysis_id = source.selected_analysis(run_id) or sorted(available)[0]
                    selected_members.setdefault(run_id, (source, analysis_id))

            # A v3 Decision projection changes Semantica decision ids. Keep the sealed v2
            # run-local provenance immutable by staging artifact pointers beside this tutorial
            # graph; the original run directories are read-only inputs.
            staging_root = output_dir / f"projection-staging-{version}"
            for run_id, (source, analysis_id) in selected_members.items():
                artifact = source.load_analysis_artifact(run_id, analysis_id)
                original_ref = Path(artifact["artifact_ref"])
                original_run = original_ref.parent.parent
                staged_run = staging_root / run_id
                staged_analyses = staged_run / "analyses"
                staged_analyses.mkdir(parents=True, exist_ok=True)
                for name in (
                    "task_presentation.json", "runner_meta.json", "result.json",
                    "trace.jsonl", "trace_manifest.json"
                ):
                    source_file = original_run / name
                    if source_file.is_file():
                        shutil.copy2(source_file, staged_run / name)
                staged_ref = staged_analyses / original_ref.name
                artifact = json.loads(json.dumps(artifact))
                artifact["artifact_ref"] = str(staged_ref)
                staged_ref.write_text(
                    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
                )
                ledger.project_analysis(artifact)

            def case_name(run_id):
                parts = str(run_id).split("_", 2)
                return parts[1] if len(parts) > 2 else "case"

            def short_model(value):
                return "Terra" if "terra" in str(value).lower() else "Luna"

            def one_line(value, limit=125):
                text = " ".join(str(value or "").split())
                return text if len(text) <= limit else text[: limit - 1] + "…"

            def table(rows, columns, limit=160):
                def clean(value):
                    return one_line(value, limit).replace("|", "/")
                return "\n".join([
                    "| " + " | ".join(label for _, label in columns) + " |",
                    "|" + "|".join("---" for _ in columns) + "|",
                    *("| " + " | ".join(clean(row.get(key, "")) for key, _ in columns) + " |"
                      for row in rows),
                ])

            def episode_for_node(node):
                meta = node.get("metadata") or {}
                artifact = ledger.load_analysis_artifact(meta["run_id"], meta["analysis_id"])
                episode = next(
                    row for row in artifact["episodes"]
                    if row["episode_id"] == meta["acr_episode_id"]
                )
                return artifact, episode

            decisions = ledger.graph.find_nodes(node_type="decision")
            display(Markdown(
                f"**Loaded:** {len(selected_members)} runs → {len(decisions)} decision cards."
            ))
            """,
            hidden=True,
        ),
        md(
            """
            ## 1. One decision card

            Semantica's native unit is simple: `category`, `scenario`, `reasoning`, `outcome`, and
            `confidence`. The main fields are human-readable and de-identified. Exact dates, note
            locators, and field-level provenance remain behind the card in the audit record.
            """
        ),
        code(
            r"""
            query_node = next(
                node for node in decisions
                if (node.get("metadata") or {}).get("decision_subject") == "evidence_item"
                and (node.get("metadata") or {}).get("outcome") == "MERELY_MENTIONS"
                and "reports atypical cells suspicious" in str(
                    (node.get("metadata") or {}).get("scenario", "")
                ).lower()
            )
            query_meta = query_node["metadata"]
            decision_card = {
                "category": query_meta["category"],
                "scenario": query_meta["scenario"],
                "reasoning": query_meta["reasoning"],
                "outcome": query_meta["outcome"],
                "confidence": query_meta["confidence"],
            }
            display(Markdown(table(
                [{"field": key.title(), "value": value}
                 for key, value in decision_card.items()],
                [("field", "Decision field"), ("value", "Recorded value")],
                limit=600,
            )))
            display(Markdown(
                "**How to read confidence:** it measures reconstruction stability across passes, "
                "not clinical correctness. The card is still something a human must judge."
            ))
            """
        ),
        md(
            """
            This is the chart-review equivalent of Semantica's vendor-selection example:

            ```python
            graph.record_decision(
                category="standing",
                scenario="Can suspicious cytology alone establish the diagnosis date?",
                reasoning="The report is ambiguous and recommends confirmatory biopsy.",
                outcome="MERELY_MENTIONS",
                confidence=0.88,  # reconstruction stability, not correctness
            )
            ```

            ## 2. Find similar decisions

            Similarity is useful because a reviewer can inspect a small set of prior judgments
            instead of searching every trace. It proposes comparison candidates; it does not say
            that either judgment is correct.
            """
        ),
        code(
            r"""
            query_episode = query_meta["acr_episode_id"]
            similar = ledger.similar_candidates(
                query_episode, max_results=6, min_similarity=0.05, cross_run_only=True
            )
            seen_runs = set()
            similar_rows = []
            for candidate in similar["candidates"]:
                node = candidate["decision"]
                meta = node.get("metadata") or {}
                if meta.get("run_id") in seen_runs:
                    continue
                seen_runs.add(meta.get("run_id"))
                candidate_artifact, _ = episode_for_node(node)
                arm = candidate_artifact.get("task_arm")
                similar_rows.append({
                    "context": (
                        f"{case_name(meta.get('run_id'))} · "
                        f"{short_model(candidate_artifact.get('review_model'))} · "
                        f"{'Task + policy' if arm == 'policy_bundle' else 'Task only'}"
                    ),
                    "scenario": node.get("scenario") or meta.get("scenario"),
                    "outcome": node.get("outcome") or meta.get("outcome"),
                    "similarity": f"{float(candidate.get('similarity') or 0):.2f}",
                })
                if len(similar_rows) == 3:
                    break

            display(Markdown(f"**Query card:** {query_meta['scenario']}"))
            display(Markdown(table(similar_rows, [
                ("context", "Run context"), ("scenario", "Comparable question"),
                ("outcome", "Outcome"),
                ("similarity", "Similarity")
            ])))
            display(Markdown(
                "**So what?** These are good cards to compare for consistency. A similarity score "
                "closer to 1 means more alike; it is not an accuracy score."
            ))
            """
        ),
        md(
            """
            ## 3. Find the same question with different answers

            “Similar” is broad. A stronger audit signal is **the same case, the same evidence note,
            and the same atomic question**, but a different outcome.

            The stored runs contain exactly that pattern for one `SYN0001` oncology note. In the
            task-only arm, no clinical policy was provided, so the models had to use their own
            judgment.
            """
        ),
        code(
            r"""
            cohort = [
                {"run_id": run_id, "analysis_id": analysis_id}
                for run_id, (_, analysis_id) in selected_members.items()
                if "task_only" in run_id
            ]
            report = ledger.find_divergent_decision_points(cohort, min_similarity=0.05)
            exact = [row for row in report["divergences"] if row.get("same_decision_point")]
            assert exact, "Expected a same-evidence task-only disagreement"
            disagreement = max(exact, key=lambda row: len(row.get("members") or []))

            disagreement_rows = []
            for member in disagreement["members"]:
                node = ledger.graph.find_node(member["decision_id"])
                meta = node.get("metadata") or {}
                disagreement_rows.append({
                    "case": case_name(member["run_id"]),
                    "model": short_model(member["review_model"]),
                    "answer": member["outcome"],
                    "reason": meta.get("reasoning"),
                    "policy": "None — task only",
                })
            display(Markdown(table(disagreement_rows, [
                ("case", "Case"), ("model", "Model"), ("answer", "Answer to same question"),
                ("reason", "Recorded reasoning"), ("policy", "Policy grounding")
            ], limit=280)))
            display(Markdown(
                "**So what?** This is a guideline-gap candidate: the same evidence received "
                "different standings when the task supplied no rule. A human should decide the "
                "desired rule, then add it to the guideline."
            ))
            """
        ),
        md(
            """
            ## 4. If a policy changes, what must be re-checked?

            Suppose we tighten this policy:

            > If a physician diagnosis predates tissue confirmation, use the earlier physician
            > date; later tissue confirmation does not reset an already established diagnosis.

            Semantica stores versioned Policy nodes and direct `APPLIED_POLICY` links. That lets us
            retrieve the historical decisions that cited this policy.
            """
        ),
        code(
            r"""
            policy_id = (
                "STORE.390.date_of_initial_diagnosis."
                "conflict.physician_statement_predating_tissue"
            )
            policy_nodes = [
                node for node in ledger.graph.find_nodes(node_type="Policy")
                if (node.get("metadata") or {}).get("policy_id") == policy_id
            ]
            assert policy_nodes, "The detailed runs contain no matching policy"
            old = policy_nodes[0]["metadata"]
            revision = ledger.register_policy_revision(
                policy_id,
                from_version=old["version"],
                rules={
                    "plain_language": (
                        "Use an earlier physician diagnosis only when it explicitly identifies "
                        "the target tumour; later tissue confirmation does not reset that date."
                    )
                },
                change_reason="Require an explicit target-tumour statement.",
            )
            impact = ledger.affected_by_policy_change(
                policy_id,
                from_version=old["version"],
                to_version=revision["version"],
            )
            functions = sorted({
                function
                for case in impact["affected_cases"]
                for function in case["decision_functions"]
            })
            friendly = {
                "where_to_look": "Search / choose notes",
                "standing": "Judge evidence",
                "which_wins": "Resolve conflicts",
                "enough": "Decide whether to stop",
                "what_to_answer": "Choose the final answer",
            }
            affected_runs = {row["run_id"] for row in impact["affected_cases"]}
            display(Markdown(
                f"- **Historical runs to revisit:** {len(affected_runs)}\n"
                f"- **Directly bound decisions:** {len(impact['affected_decisions'])}\n"
                f"- **Parts of the review involved:** "
                f"{', '.join(friendly.get(item, item) for item in functions)}\n"
                "- **Meaning:** this is a re-audit queue, not a prediction that every answer changes."
            ))
            """
        ),
        md(
            """
            ## The whole point

            | Without the ContextGraph | With the ContextGraph |
            |---|---|
            | Read one trace at a time | Retrieve a small set of comparable Decision cards |
            | Notice only final-answer differences | Localize disagreement to one evidence judgment |
            | Guess which old runs a rule touched | Ask Semantica for directly policy-bound decisions |
            | Trust a summary | Follow the card's provenance back to the raw trace |

            **Use the graph to route human attention. Use provenance and raw trace to decide what
            really happened. Use a qualified reviewer to decide what is clinically correct.**
            """
        ),
    ],
    title="Semantica Decision Intelligence",
)

# The reader should see the story and results first. Every Python cell remains available through
# the notebook's expand control, but its implementation plumbing is collapsed by default.
for teaching_notebook in (NB1, NB2, NB3):
    for teaching_cell in teaching_notebook.cells:
        if teaching_cell.cell_type == "code":
            teaching_cell.metadata["jupyter"] = {"source_hidden": True}
            teaching_cell.metadata["tags"] = ["hide-input"]


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    payloads = {
        "01_run_chart_review_experiments.ipynb": NB1,
        "02_trace_to_decision_chain.ipynb": NB2,
        "03_semantica_decision_intelligence.ipynb": NB3,
    }
    for name, payload in payloads.items():
        nbformat.write(payload, NOTEBOOKS / name)
        print(NOTEBOOKS / name)


if __name__ == "__main__":
    main()
