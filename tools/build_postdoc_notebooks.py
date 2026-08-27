"""Build the three checked postdoc walkthrough notebooks.

The notebooks are generated from ordinary Python strings so their instructional text and
executable cells can be reviewed in a normal diff. Run this file from the repository root after
editing a walkthrough, then execute the notebooks with the README's ``jupyter nbconvert`` loop.
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


def code(source: str):
    return new_code_cell(dedent(source).strip() + "\n")


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
            "acr": {"audience": "postdoctoral researcher", "title": title, "version": 1},
        },
    )


NB1 = notebook(
    [
        md(
            """
            # 01 — Run the chart-review experiment

            **Question:** what does the agent return when we vary the case and the material in the
            Task Presentation?

            This notebook is the experimental entry point. It runs, or reuses, a small factorial
            cohort over synthetic chart cases and two task arms:

            - `task_only`: the agent receives the field/output contract but no chart-review policy
              clauses. Material choices can therefore depend on its own judgment.
            - `policy_bundle`: the same task plus independently versioned evidence, conflict,
              selection, and proof clauses.

            It then reads every run back from Langtrace, reconstructs Decision Episodes with Luna,
            selects an analysis only when two reconstruction passes agree on episode alignment,
            and projects the result into one Semantica ContextGraph.

            ## What you should learn

            1. How to configure a real experiment without putting keys in the notebook.
            2. Which artifacts are created at each boundary.
            3. How answers vary by case, arm, and review model.
            4. Which observations are experimental results versus hypotheses for later audit.

            **Cost/safety:** existing local real-provider runs are reused by default. Set
            `ACR_TUTORIAL_MODE=live` explicitly to make paid OpenRouter calls. Run outputs stay
            under `runs/`, which Git ignores.
            """
        ),
        md(
            """
            ## 1. Locate the repository and choose a mode

            `reuse` makes no model or network calls. It uses the checked local cohort that produced
            the executed notebook. `live` creates a new cohort. On a fresh clone there is no local
            cohort, so the default becomes `live` and the environment checks below fail early if
            provider configuration is missing.
            """
        ),
        code(
            r"""
            from collections import Counter, defaultdict
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

            START_DIR = Path.cwd().resolve()
            ROOT = START_DIR if (START_DIR / "pyproject.toml").is_file() else START_DIR.parent
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
            OUTPUT_ROOT = ROOT / "runs/postdoc-notebook-output"
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

            def one_line(value, limit=96):
                text = " ".join(str("" if value is None else value).split())
                return text if len(text) <= limit else text[: limit - 1] + "…"

            def display_path(value):
                path = Path(value).resolve()
                try:
                    return str(path.relative_to(ROOT))
                except ValueError:
                    return str(path)

            def markdown_table(rows, columns):
                def safe(value):
                    return one_line(value, 110).replace("|", "/")
                header = "| " + " | ".join(label for _, label in columns) + " |"
                rule = "|" + "|".join("---" for _ in columns) + "|"
                body = [
                    "| " + " | ".join(safe(row.get(key, "")) for key, _ in columns) + " |"
                    for row in rows
                ]
                return "\n".join([header, rule, *body])

            codex_version = subprocess.run(
                ["codex", "--version"], capture_output=True, text=True, check=True
            ).stdout.strip()
            display({"mode": MODE, "codex": codex_version, "ledger": display_path(LEDGER_PATH)})
            """
        ),
        md(
            """
            ## 2. Declare the experimental grid

            The default `pilot` profile is two deliberately paired cases × two task arms × Luna:

            - `SYN0001`: ambiguous cytology **with** a same-day physician impression.
            - `SYNX03`: the mirror case, ambiguous cytology **without** that impression.

            Set `ACR_TUTORIAL_PROFILE=full` to add Terra as a second acting review model. Luna is
            still the reconstruction model so changes in the semantic projection are not
            confounded with a different reconstructor.

            This is a mechanism demonstration, not a powered accuracy study. Increase cases and
            repetitions only after defining the estimand and sampling plan.
            """
        ),
        code(
            r"""
            PROFILE = os.environ.get("ACR_TUTORIAL_PROFILE", "pilot").lower()
            assert PROFILE in {"pilot", "full"}
            CASE_IDS = ["SYN0001", "SYNX03"]
            TASK_ARMS = ["task_only", "policy_bundle"]
            REVIEW_MODELS = ["openai/gpt-5.6-luna"]
            if PROFILE == "full":
                REVIEW_MODELS.append("openai/gpt-5.6-terra")
            RECONSTRUCTION_MODEL = "openrouter/openai/gpt-5.6-luna"
            RECONSTRUCTION_PASSES = 2
            SPEC = ROOT / "assets/specs/STORE.390.date_of_initial_diagnosis.yaml"

            plan = [
                {"case_id": case_id, "task_arm": arm, "review_model": model}
                for case_id in CASE_IDS
                for arm in TASK_ARMS
                for model in REVIEW_MODELS
            ]
            display(Markdown(markdown_table(plan, [
                ("case_id", "Case"), ("task_arm", "Task arm"),
                ("review_model", "Acting review model"),
            ])))
            display({"planned_runs": len(plan), "reconstruction_passes_per_run": 2})
            """
        ),
        md(
            """
            ## 3. Execute the closed loop, or load the sealed cohort

            In `live` mode each row performs the whole boundary:

            `Codex App Server → chart tools → local Langtrace → fixed ReAct replay → two Luna
            reconstructions → deterministic verification → explicit selection → Semantica`.

            No key is passed as a CLI argument or written into an artifact. If the two passes do
            not agree on episode alignment, execution stops instead of silently choosing the more
            convenient reconstruction.
            """
        ),
        code(
            r"""
            if MODE == "live":
                for name in ("OPENROUTER_API_KEY", "LANGTRACE_API_KEY"):
                    assert os.environ.get(name), f"{name} is required in live mode"
                langtrace_host = os.environ.get("LANGTRACE_API_HOST", "http://127.0.0.1:3100")
                langtrace_project = os.environ.get("LANGTRACE_PROJECT_ID", "acr_chart_review")
                client = LangtraceClient(
                    api_key=os.environ["LANGTRACE_API_KEY"],
                    api_host=langtrace_host,
                    project_id=langtrace_project,
                )
                ledger = SemanticaLedger(LEDGER_PATH)
                created = []
                for cell in plan:
                    run_dir = await asyncio.to_thread(
                        run_patient,
                        SPEC,
                        ROOT / "corpus/patients" / cell["case_id"],
                        LIVE_ROOT,
                        model=cell["review_model"],
                        base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        task_arm=cell["task_arm"],
                        langtrace_api_key=os.environ["LANGTRACE_API_KEY"],
                        langtrace_api_host=langtrace_host,
                        langtrace_project_id=langtrace_project,
                    )
                    runner_meta = json.loads((run_dir / "runner_meta.json").read_text())
                    assert runner_meta["langtrace_verified"] is True
                    review = client.get_review(runner_meta["langtrace_trace_id"])
                    reconstructor = AuditedLiteLLM(
                        model=RECONSTRUCTION_MODEL,
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        temperature=0.0,
                    )
                    summary = reconstruct_run(
                        review,
                        ledger,
                        reconstructor,
                        passes=RECONSTRUCTION_PASSES,
                        artifact_dir=run_dir / "analyses",
                        reconstructor_identity=RECONSTRUCTION_MODEL,
                        max_attempts_per_pass=3,
                    )
                    assert summary["drift"]["alignment_agrees"] is True, summary["drift"]
                    selected = summary["analyses"][0]["analysis_id"]
                    ledger.select_analysis(
                        review.run_id,
                        selected,
                        selected_by="postdoc-notebook-01",
                        reason="Two Luna passes agreed on atomic episode alignment.",
                        provenance="DETERMINISTIC_DERIVED",
                    )
                    created.append({"run_id": review.run_id, "analysis_id": selected})
                display({"created_runs": len(created), "ledger": display_path(LEDGER_PATH)})
            else:
                assert LEDGER_PATH.is_file(), (
                    "No reusable cohort is present. Set ACR_TUTORIAL_MODE=live after configuring "
                    "OpenRouter and local Langtrace."
                )
                ledger = SemanticaLedger(LEDGER_PATH)
                display({
                    "reused_real_provider_ledger": display_path(LEDGER_PATH),
                    **ledger.stats(),
                })
            """
        ),
        md(
            """
            ## 4. Build one analysis row per run

            A run can have several append-only reconstructions. We use the explicit selection when
            present; otherwise, for the sealed historical cohort only, we choose the sole/first
            reconstruction and mark that fact in the table. Downstream production analysis should
            require an explicit selection.

            The synthetic corpus includes designer ground truth, so this notebook can display a
            correctness check. Real charts will not have that column unless independently
            adjudicated.
            """
        ),
        code(
            r"""
            analyses_by_run = defaultdict(set)
            for node in ledger.graph.find_nodes(node_type="decision"):
                meta = node.get("metadata") or {}
                if meta.get("run_id") and meta.get("analysis_id"):
                    analyses_by_run[str(meta["run_id"])].add(str(meta["analysis_id"]))

            edge_rows = []
            for edge in ledger.graph.edges:
                edge_rows.append(edge.to_dict() if hasattr(edge, "to_dict") else dict(edge))

            cohort = []
            for run_id, analysis_ids in sorted(analyses_by_run.items()):
                selected = ledger.selected_analysis(run_id)
                analysis_id = selected if selected in analysis_ids else sorted(analysis_ids)[0]
                artifact = ledger.load_analysis_artifact(run_id, analysis_id)
                run_dir = EXPERIMENT_ROOT / run_id
                if not (run_dir / "result.json").is_file():
                    continue
                result = json.loads((run_dir / "result.json").read_text())
                case_id = str(artifact.get("patient_id") or run_id.split("_", 2)[1])
                truth_path = ROOT / "corpus/patients" / case_id / "_ground_truth.json"
                truth = json.loads(truth_path.read_text()) if truth_path.is_file() else {}
                gold = (((truth.get("ground_truth") or {}).get(
                    "STORE.390.date_of_initial_diagnosis") or {}).get("value"))
                value = result.get("value") or {}
                answer = value.get("date_of_initial_diagnosis") if isinstance(value, dict) else value
                decision_ids = {
                    str(node["id"])
                    for node in ledger.graph.find_nodes(node_type="decision")
                    if (node.get("metadata") or {}).get("run_id") == run_id
                    and (node.get("metadata") or {}).get("analysis_id") == analysis_id
                }
                applied_policy_edges = sum(
                    edge.get("type") == "APPLIED_POLICY" and edge.get("source_id") in decision_ids
                    for edge in edge_rows
                )
                cohort.append({
                    "run_id": run_id,
                    "case": case_id,
                    "arm": artifact.get("task_arm"),
                    "review_model": artifact.get("review_model"),
                    "analysis_id": analysis_id,
                    "selection": "explicit" if selected == analysis_id else "historical fallback",
                    "answer": answer,
                    "gold": gold,
                    "correct": answer == gold if gold is not None else None,
                    "episodes": len(artifact.get("episodes") or []),
                    "react_cycles": len(artifact.get("cycles") or []),
                    "applied_policy_edges": applied_policy_edges,
                    "langtrace_events": sum(
                        1 for line in (run_dir / "trace.jsonl").read_text().splitlines() if line
                    ),
                })

            assert cohort, "The ledger contains no run with a local result.json"
            display(Markdown(markdown_table(cohort, [
                ("case", "Case"), ("arm", "Arm"), ("review_model", "Review model"),
                ("answer", "Answer"), ("gold", "Synthetic gold"), ("correct", "Match"),
                ("langtrace_events", "Events"), ("react_cycles", "Cycles"),
                ("episodes", "Episodes"), ("applied_policy_edges", "Policy bindings"),
            ])))
            """
        ),
        md(
            """
            ## 5. Inspect variation without overclaiming

            The table below groups exact returned values. Variation within the same case and task
            arm is a reproducibility signal. A difference between arms is a candidate effect of
            the offered policy material, but this small observational cohort cannot isolate that
            effect from model identity, run stochasticity, or search-path differences.
            """
        ),
        code(
            r"""
            grouped = defaultdict(list)
            for row in cohort:
                grouped[(row["case"], row["arm"])].append(row)

            variation_rows = []
            for (case_id, arm), rows in sorted(grouped.items()):
                distribution = Counter(str(row["answer"]) for row in rows)
                correct = [row["correct"] for row in rows if row["correct"] is not None]
                variation_rows.append({
                    "case": case_id,
                    "arm": arm,
                    "n": len(rows),
                    "outcomes": dict(distribution),
                    "distinct": len(distribution),
                    "gold_matches": f"{sum(correct)}/{len(correct)}" if correct else "not adjudicated",
                })
            display(Markdown(markdown_table(variation_rows, [
                ("case", "Case"), ("arm", "Arm"), ("n", "Runs"),
                ("outcomes", "Outcome distribution"), ("distinct", "Distinct outcomes"),
                ("gold_matches", "Gold matches"),
            ])))

            unstable_cells = [row for row in variation_rows if row["distinct"] > 1]
            display(Markdown(
                f"**Observed insight:** {len(unstable_cells)} case/arm cell(s) produced more than "
                "one exact answer. Notebook 3 asks whether the divergence can be localized to "
                "a comparable Decision Point rather than only observed at the final answer."
            ))
            """
        ),
        md(
            """
            ## 6. Save a compact handoff for the next notebooks

            The full trace, analysis artifact, provenance database, and ContextGraph remain the
            authorities. This JSON is only a convenience index; it is not a substitute for them.
            """
        ),
        code(
            r"""
            closure = {
                "schema": "acr.postdoc_experiment.v1",
                "mode": MODE,
                "experiment_root": display_path(EXPERIMENT_ROOT),
                "ledger_path": display_path(LEDGER_PATH),
                "cohort": cohort,
                "unstable_case_arm_cells": unstable_cells,
                "interpretation_limits": [
                    "small convenience cohort, not a powered comparison",
                    "synthetic gold is not a substitute for clinical adjudication",
                    "arm differences are not automatically causal policy effects",
                ],
            }
            closure_path = OUTPUT_ROOT / "01_experiment_summary.json"
            closure_path.write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n")
            assert all((EXPERIMENT_ROOT / row["run_id"] / "trace.jsonl").is_file() for row in cohort)
            display(Markdown(
                f"**Notebook 1 closed.** {len(cohort)} real-provider runs are indexed in "
                f"`{display_path(closure_path)}`. Continue to Notebook 2 to inspect how one answer was made."
            ))
            """
        ),
    ],
    title="Run the chart-review experiment",
)


NB2 = notebook(
    [
        md(
            """
            # 02 — From raw trace to a human-auditable Decision Chain

            **Question:** can a reviewer follow the consequential choices without losing the
            ability to inspect what actually happened?

            This notebook opens one completed chart review at four levels:

            1. Codex protocol events — very detailed harness traffic.
            2. Canonical Langtrace/Layer-1 events — observable tool calls and server facts.
            3. Deterministic ReAct cycles — state before, action, observation, state after.
            4. Decision Episodes — one material choice that one human can judge with one verdict.

            The Decision Chain is an index over the raw trace, not a replacement for it. Every
            episode must retain a path back to its cycle and event evidence. We never display or
            claim private chain-of-thought.
            """
        ),
        md(
            """
            ## 1. Select a run

            The checked demonstration uses the SYNX03 policy-guided Luna run because it contains
            the full pattern a reviewer needs to see: inventory, keyword search, candidate-note
            selection, three independent evidence judgments, conflict resolution, and a stopping
            decision. Override the paths with `ACR_AUDIT_LEDGER` and `ACR_AUDIT_RUN_ID`.
            """
        ),
        code(
            r"""
            from collections import Counter, defaultdict
            from pathlib import Path
            import json
            import os

            from IPython.display import Markdown, display
            from acr.mvp.human_review import human_review_view
            from acr.mvp.ledger import SemanticaLedger

            START_DIR = Path.cwd().resolve()
            ROOT = START_DIR if (START_DIR / "pyproject.toml").is_file() else START_DIR.parent
            assert (ROOT / "pyproject.toml").is_file(), "Start Jupyter from the repo or notebooks/"

            checked_ledger = ROOT / "runs/notebook-live-20260827/ledger.json"
            generated_ledger = ROOT / "runs/postdoc-study/ledger.json"
            seed_ledger = ROOT / "runs/policy-experiment-20260827/experiment-ledger.json"
            default_ledger = next(
                (path for path in (checked_ledger, generated_ledger, seed_ledger) if path.is_file()),
                generated_ledger,
            )
            LEDGER_PATH = Path(os.environ.get("ACR_AUDIT_LEDGER", default_ledger))
            assert LEDGER_PATH.is_file(), "Run Notebook 1 first or set ACR_AUDIT_LEDGER"
            ledger = SemanticaLedger(LEDGER_PATH)

            def one_line(value, limit=100):
                text = " ".join(str("" if value is None else value).split())
                return text if len(text) <= limit else text[: limit - 1] + "…"

            def display_path(value):
                path = Path(value).resolve()
                try:
                    return str(path.relative_to(ROOT))
                except ValueError:
                    return str(path)

            def markdown_table(rows, columns):
                def safe(value):
                    return one_line(value, 120).replace("|", "/")
                return "\n".join([
                    "| " + " | ".join(label for _, label in columns) + " |",
                    "|" + "|".join("---" for _ in columns) + "|",
                    *("| " + " | ".join(safe(row.get(key, "")) for key, _ in columns) + " |"
                      for row in rows),
                ])

            preferred_run = (
                "20260827T135252029492Z_SYNX03_STORE_390_date_of_initial_diagnosis_policy_bundle"
            )
            run_id = os.environ.get("ACR_AUDIT_RUN_ID")
            if run_id is None and ledger.selected_analysis(preferred_run):
                run_id = preferred_run
            if run_id is None:
                selections = ledger.graph.find_nodes(node_type="AnalysisSelection")
                assert selections, "The ledger has no explicitly selected analysis"
                run_id = str((selections[-1].get("metadata") or {}).get("run_id"))
            analysis_id = ledger.selected_analysis(run_id)
            assert analysis_id, "Choose/select one reconstruction before human review"
            run_dir = LEDGER_PATH.parent / run_id
            artifact = ledger.load_analysis_artifact(run_id, analysis_id)
            view = human_review_view(ledger, run_id, analysis_id, run_dir=run_dir)
            display({
                "run_id": run_id,
                "analysis_id": analysis_id,
                "case": artifact.get("patient_id") or run_id.split("_", 2)[1],
                "task_arm": artifact.get("task_arm"),
                "review_model": artifact.get("review_model"),
                "reconstructor": artifact.get("reconstructor_identity"),
            })
            """
        ),
        md(
            """
            ## 2. Measure the abstraction ladder

            These counts answer “how much does the reviewer have to read by default?” They do not
            prove the abstraction is lossless. The fidelity check comes later: each Decision
            Episode must preserve drill-down links and every deterministic cycle must be assigned
            exactly once as decision-bearing, decision-support, or mechanical.
            """
        ),
        code(
            r"""
            protocol_records = [
                json.loads(line) for line in (run_dir / "layer2_codex.jsonl").read_text().splitlines()
                if line.strip()
            ]
            layer1_events = [
                json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()
                if line.strip()
            ]
            cycles = artifact["cycles"]
            episodes = view["episodes"]
            steps = view["review_chain"]["steps"]
            ladder = [
                {"representation": "Codex protocol", "units": len(protocol_records),
                 "default human use": "Harness/debug only"},
                {"representation": "Canonical Langtrace events", "units": len(layer1_events),
                 "default human use": "Observable execution evidence"},
                {"representation": "Deterministic ReAct cycles", "units": len(cycles),
                 "default human use": "State/action replay"},
                {"representation": "Decision Episodes", "units": len(episodes),
                 "default human use": "Primary audit units"},
            ]
            display(Markdown(markdown_table(ladder, [
                ("representation", "Representation"), ("units", "Units"),
                ("default human use", "Role"),
            ])))
            """
        ),
        md(
            """
            ## 3. Read the observable raw trace

            The protocol stream contains SDK lifecycle and model transport events. We count its
            event types but deliberately do not print model reasoning payloads. The canonical
            Layer-1 stream below is the useful raw audit record: tool name, compact action, and
            server result shape.
            """
        ),
        code(
            r"""
            protocol_types = Counter(str(row.get("type") or row.get("method") or "other")
                                     for row in protocol_records)
            display(Markdown("**Codex protocol event types (content redacted):** `" +
                             json.dumps(dict(protocol_types.most_common()), ensure_ascii=False) + "`"))

            def event_action(event):
                payload = event.get("payload") or {}
                args = payload.get("args") or event.get("args") or {}
                tool = payload.get("tool") or event.get("tool") or ""
                for key in ("decision", "query", "note_id", "standing", "objective", "status"):
                    if args.get(key) is not None:
                        return f"{key}={args[key]}"
                return ""

            raw_rows = []
            for event in layer1_events:
                payload = event.get("payload") or {}
                result = payload.get("result") or event.get("result") or {}
                raw_rows.append({
                    "seq": event.get("seq"),
                    "kind": event.get("kind"),
                    "tool": payload.get("tool") or event.get("tool") or "—",
                    "action": event_action(event),
                    "result": ", ".join(sorted(result)[:6]) if isinstance(result, dict) else "",
                })
            display(Markdown(markdown_table(raw_rows, [
                ("seq", "Seq"), ("kind", "Kind"), ("tool", "Tool"),
                ("action", "Compact action"), ("result", "Result fields"),
            ])))
            """
        ),
        md(
            """
            ## 4. Replay the fixed ReAct cycles

            A cycle is not automatically a decision. Search calls executing one precommitted
            keyword batch can be support; pagination can be mechanical; a `record_finding` that
            commits one note's Standing is decision-bearing. Reconstruction may label this fixed
            skeleton, but it may not add, remove, reorder, duplicate, or move a cycle.
            """
        ),
        code(
            r"""
            annotations = artifact["cycle_annotations"]
            if isinstance(annotations, list):
                annotations = {row["cycle_id"]: row for row in annotations}
            cycle_rows = []
            for index, cycle in enumerate(cycles, 1):
                annotation = annotations[cycle["cycle_id"]]
                observed = (cycle.get("state_after") or {}).get("observed_state") or {}
                tools = [str(action.get("tool") or "") for action in cycle.get("actions") or []]
                cycle_rows.append({
                    "n": index,
                    "cycle": cycle["cycle_id"].rsplit(":", 1)[-1],
                    "role": annotation["role"],
                    "function": annotation.get("decision_function") or "—",
                    "tools": ", ".join(tools) or "—",
                    "receipt": "yes" if cycle.get("has_decision_receipt") else "no",
                    "state": (f"surfaced={len(observed.get('surfaced_notes') or [])}; "
                              f"read={len(observed.get('read_notes') or [])}; "
                              f"findings={len((cycle.get('state_after') or {}).get('declared_state', {}).get('findings') or [])}"),
                })
            display(Markdown(markdown_table(cycle_rows, [
                ("n", "#"), ("cycle", "Cycle"), ("role", "Role"),
                ("function", "Decision function"), ("tools", "Actions"),
                ("receipt", "Sealed receipt"), ("state", "State after"),
            ])))
            """
        ),
        md(
            """
            ## 5. Follow the human Decision Chain

            Read this as if a colleague were explaining the review aloud. At each step ask:

            1. Was this the right question at this point?
            2. Were the meaningful alternatives represented?
            3. Does the evidence/rule actually support the choice?
            4. What judgment remained for the model?
            5. If this step is wrong, which later steps inherit the problem?

            A resolved policy reference proves only that the reference existed. It does not prove
            semantic entailment or clinical correctness.
            """
        ),
        code(
            r"""
            narrative = ["## The run, one consequential choice at a time"]
            for index, step in enumerate(steps, 1):
                grounding = step.get("grounding_assessment") or {}
                flags = ", ".join(item["code"] for item in step.get("review_attention") or []) or "none"
                policy_refs = [
                    str(item["rule_id"]) for item in step.get("guidelines") or []
                    if item.get("rule_id")
                ]
                narrative.extend([
                    f"### {index}. {step['phase_label']} — `{step['decision_function']}/{step['decision_subject']}`",
                    f"- **Question before acting:** {step.get('question')}",
                    f"- **Choice:** {step.get('decision')}",
                    f"- **Stated reason:** {step.get('reason')}",
                    f"- **Basis:** {', '.join(step.get('basis_sources') or []) or 'not recorded'}",
                    f"- **Applied/offered clause refs shown here:** {', '.join(sorted(set(policy_refs))) or 'none'}",
                    f"- **Reference status:** {grounding.get('reference_resolution_status')}",
                    f"- **Remaining judgment:** {grounding.get('judgment_mode')}",
                    f"- **Review attention:** {flags}",
                    f"- **Resulting state:** {step.get('state_result')}",
                    "",
                ])
            conclusion = view["review_chain"]["conclusion"]
            narrative.extend([
                "## Accepted conclusion",
                f"`{json.dumps(conclusion.get('value'), ensure_ascii=False)}`",
                f"\nSubmission explanation: {conclusion.get('reasoning')}",
            ])
            display(Markdown("\n".join(narrative)))
            """
        ),
        md(
            """
            ## 6. Drill one flagged Decision back into evidence

            We choose the first step with a review-attention flag. The compact chain tells us where
            to look; the episode, runtime testimony, raw events, and field provenance tell us what
            authority each statement has. This is the safeguard against a fluent reconstruction
            hiding an execution error.
            """
        ),
        code(
            r"""
            flagged_step = next((step for step in steps if step.get("review_attention")), steps[0])
            episode_id = flagged_step["episode_ids"][0]
            episode = next(row for row in episodes if row["episode_id"] == episode_id)
            artifact_episode = next(
                row for row in artifact["episodes"] if row["episode_id"] == episode_id
            )
            reconstructed = episode["reconstruction"]
            source_event_ids = set(artifact_episode.get("source_event_ids") or [])
            source_events = [
                event for event in layer1_events if f"layer1:{event.get('seq')}" in source_event_ids
            ]
            testimony = (episode.get("runtime_testimonies") or [{}])[0]
            drill = {
                "question": flagged_step.get("question"),
                "choice": flagged_step.get("decision"),
                "runtime_testimony_ref": testimony.get("testimony_ref"),
                "runtime_because": testimony.get("because"),
                "reconstructed_rationale": reconstructed.get("decision_rationale"),
                "field_provenance": reconstructed.get("field_provenance"),
                "raw_event_ids": sorted(source_event_ids),
                "raw_tools": [((row.get("payload") or {}).get("tool") or row.get("tool"))
                              for row in source_events],
                "review_attention": flagged_step.get("review_attention"),
            }
            display(drill)
            assert episode["bearing_cycle_id"]
            assert episode["raw_langtrace_links"]
            """
        ),
        md(
            """
            ## 7. Prove the abstraction still indexes the complete cycle skeleton

            “Fewer units” is useful only if it does not silently drop a decision-bearing cycle.
            The verifier requires every cycle exactly once across Decision Episodes and mechanical
            cycles. Separately, every episode must link back to raw trace material.
            """
        ),
        code(
            r"""
            episode_cycle_ids = [
                cycle_id for row in artifact["episodes"] for cycle_id in row["source_cycle_ids"]
            ]
            mechanical_cycle_ids = artifact["mechanical_cycle_ids"]
            all_cycle_ids = [row["cycle_id"] for row in cycles]
            assert len(episode_cycle_ids + mechanical_cycle_ids) == len(all_cycle_ids)
            assert set(episode_cycle_ids + mechanical_cycle_ids) == set(all_cycle_ids)
            assert len(set(episode_cycle_ids + mechanical_cycle_ids)) == len(all_cycle_ids)
            traceable = sum(bool(row["bearing_cycle_id"]) and bool(row["raw_langtrace_links"])
                            for row in episodes)
            assert traceable == len(episodes)
            display({
                "cycles_accounted_for_exactly_once": f"{len(all_cycle_ids)}/{len(all_cycle_ids)}",
                "episodes_with_raw_drilldown": f"{traceable}/{len(episodes)}",
                "mechanical_cycles": len(mechanical_cycle_ids),
                "decision_or_support_cycles": len(episode_cycle_ids),
            })
            """
        ),
        md(
            """
            ## 8. Compare task-only with policy-bundle behavior on the same case

            If the historical experiment ledger is available, this section compares two selected
            SYN0001 runs. It does not align steps merely by sequence number; it shows the semantic
            function/subject, the chosen outcome, and whether the Decision node had a direct
            Semantica `APPLIED_POLICY` binding.

            The interesting result is not only that the final dates differ. The detailed run
            retrieved and judged a same-day physician note that the task-only run did not use as
            establishing evidence. That creates an actionable retrieval/standing/conflict audit
            question.
            """
        ),
        code(
            r"""
            comparison_path = ROOT / "runs/policy-experiment-20260827/experiment-ledger.json"
            comparison_rows = []
            if comparison_path.is_file():
                comparison_ledger = SemanticaLedger(comparison_path)
                comparison_runs = [
                    "20260827T101823421486Z_SYN0001_STORE_390_date_of_initial_diagnosis_task_only",
                    "20260827T105131502195Z_SYN0001_STORE_390_date_of_initial_diagnosis_policy_bundle",
                ]
                edge_rows = [edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)
                             for edge in comparison_ledger.graph.edges]
                for candidate_run in comparison_runs:
                    selected = comparison_ledger.selected_analysis(candidate_run)
                    candidate_view = human_review_view(
                        comparison_ledger,
                        candidate_run,
                        selected,
                        run_dir=comparison_path.parent / candidate_run,
                    )
                    arm = candidate_view["task_presentation"]["arm_id"]
                    for index, step in enumerate(candidate_view["review_chain"]["steps"], 1):
                        decision_ids = {
                            row["semantica_decision_id"] for row in step["detail_episodes"]
                        }
                        direct_bindings = sum(
                            edge.get("type") == "APPLIED_POLICY"
                            and edge.get("source_id") in decision_ids
                            for edge in edge_rows
                        )
                        comparison_rows.append({
                            "arm": arm,
                            "step": index,
                            "point": f"{step['decision_function']}/{step['decision_subject']}",
                            "choice": step.get("decision"),
                            "policy_bindings": direct_bindings,
                        })
                display(Markdown(markdown_table(comparison_rows, [
                    ("arm", "Arm"), ("step", "#"), ("point", "Decision point"),
                    ("choice", "Choice"), ("policy_bindings", "Direct policy bindings"),
                ])))
            else:
                display(Markdown(
                    "Historical paired cohort not present. Notebook 1 can generate equivalent "
                    "task-only and policy-bundle runs for comparison."
                ))
            """
        ),
        md(
            """
            ## 9. What the Decision layer gains and loses

            | Gain | Corresponding risk | Safeguard in this notebook |
            |---|---|---|
            | One verdict per consequential choice | Reconstruction may choose the wrong boundary | Fixed cycles, sealed receipts, two-pass drift, explicit selection |
            | Stable function/subject for cross-run comparison | A later taxonomy may reinterpret the run | Taxonomy is post-run and artifacts are append-only |
            | Human-readable rationale and state transition | Fluent text may overstate what happened | Field provenance and exact source refs |
            | Causal path and review routing | Temporal adjacency may be mistaken for causation | Only explicit evidenced causal assertions enter the audit chain |
            | Much shorter default reading path | Incidental low-level errors may be hidden | 100% episode drill-down plus complete cycle accounting |

            **Audit rule:** use the Decision Chain to decide *where to inspect*. Use the raw trace
            and provenance to decide *what actually happened*. Clinical correctness still requires
            a qualified reviewer.
            """
        ),
        code(
            r"""
            closure = {
                "schema": "acr.postdoc_audit_walkthrough.v1",
                "run_id": run_id,
                "analysis_id": analysis_id,
                "counts": {
                    "protocol_records": len(protocol_records),
                    "langtrace_events": len(layer1_events),
                    "react_cycles": len(cycles),
                    "decision_episodes": len(episodes),
                },
                "all_cycles_accounted_for": True,
                "all_episodes_traceable": traceable == len(episodes),
                "priority_review_count": view["review_chain"]["priority_review_count"],
                "conclusion": view["review_chain"]["conclusion"],
            }
            output = ROOT / "runs/postdoc-notebook-output/02_audit_walkthrough.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n")
            display(Markdown(
                f"**Notebook 2 closed.** The human path has {len(episodes)} auditable choices; "
                f"all {len(episodes)} retain raw drill-down. Summary: `{display_path(output)}`."
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
            # 03 — Semantica Decision Intelligence on real chart-review runs

            **Question:** what becomes possible after verified Decision Episodes from many runs are
            stored in one ContextGraph?

            This notebook demonstrates the Semantica capabilities that are actually relevant to
            ACR:

            - ContextGraph as the storage/query substrate;
            - native Decision recording and scoped insights;
            - similar-decision retrieval and divergent Decision Points;
            - explicit causal-chain traversal and candidate impact analysis;
            - Policy versioning and direct-binding impact queues;
            - native provenance lineage, integrity checks, and optional PROV export.

            It does **not** use GraphRAG, Semantica's general rule Reasoner, Ontology Hub, generic
            `KGVisualizer`, or the general graph export framework. Explorer is the interactive UI;
            the notebook focuses on reproducible analytical queries.
            """
        ),
        md(
            """
            ## 1. Load a multi-run ContextGraph

            “Decision graph” below means the decision-centered subgraph inside ContextGraph, not a
            second ACR graph. Decision nodes hold a de-identified comparable signature. Detailed
            state, testimony, actions, observations, and field provenance live in Semantica's
            run-local ProvenanceManager and are linked from the projection.
            """
        ),
        code(
            r"""
            from collections import Counter, defaultdict
            from pathlib import Path
            import json
            import os
            import shutil

            from IPython.display import Markdown, display
            from acr.mvp.ledger import SemanticaLedger

            START_DIR = Path.cwd().resolve()
            ROOT = START_DIR if (START_DIR / "pyproject.toml").is_file() else START_DIR.parent
            assert (ROOT / "pyproject.toml").is_file(), "Start Jupyter from the repo or notebooks/"
            seed = ROOT / "runs/policy-experiment-20260827/experiment-ledger.json"
            generated = ROOT / "runs/postdoc-study/ledger.json"
            default_ledger = seed if seed.is_file() else generated
            LEDGER_PATH = Path(os.environ.get("ACR_INTELLIGENCE_LEDGER", default_ledger))
            assert LEDGER_PATH.is_file(), "Run Notebook 1 first or set ACR_INTELLIGENCE_LEDGER"
            ledger = SemanticaLedger(LEDGER_PATH)

            def one_line(value, limit=105):
                text = " ".join(str("" if value is None else value).split())
                return text if len(text) <= limit else text[: limit - 1] + "…"

            def display_path(value):
                path = Path(value).resolve()
                try:
                    return str(path.relative_to(ROOT))
                except ValueError:
                    return str(path)

            def markdown_table(rows, columns):
                def safe(value):
                    return one_line(value, 130).replace("|", "/")
                return "\n".join([
                    "| " + " | ".join(label for _, label in columns) + " |",
                    "|" + "|".join("---" for _ in columns) + "|",
                    *("| " + " | ".join(safe(row.get(key, "")) for key, _ in columns) + " |"
                      for row in rows),
                ])

            stats = ledger.stats()
            display({"ledger": display_path(LEDGER_PATH), **stats})
            """
        ),
        md(
            """
            ## 2. See what is in the ContextGraph

            The important distinction is between reusable comparison nodes and audit detail:

            - `decision`, `Policy`, and typed causal edges support cross-run queries.
            - `ReActCycle`, `StateSnapshot`, `DecisionTestimony`, evidence/rule pointers, and
              `CausalAssertion` preserve the graph path back to execution evidence.
            - the full payload remains in the content-addressed artifact and provenance database,
              avoiding oversized or patient-identifying Decision signatures.
            """
        ),
        code(
            r"""
            graph_stats = stats["semantica"]
            node_rows = [
                {"type": key, "count": value}
                for key, value in sorted(
                    graph_stats["node_types"].items(), key=lambda item: (-item[1], item[0])
                )
            ]
            edge_rows = [
                {"type": key, "count": value}
                for key, value in sorted(
                    graph_stats["edge_types"].items(), key=lambda item: (-item[1], item[0])
                )
            ]
            display(Markdown("### Node types\n" + markdown_table(node_rows, [
                ("type", "Node type"), ("count", "Count")
            ])))
            display(Markdown("### Relationship types\n" + markdown_table(edge_rows, [
                ("type", "Edge type"), ("count", "Count")
            ])))
            """
        ),
        md(
            """
            ## 3. Find the same Decision Point with different outcomes

            A final-answer mismatch says only that two runs differ. A divergent Decision Point is
            more actionable: same atomic function, subject, and pre-decision situation; different
            committed outcome.

            We first restrict the cohort to one reconstruction per `task_only` run. Semantica
            supplies native similar-decision candidates; ACR applies chart-review identity and
            cohort guards. An ungrounded divergence is a **guideline-question candidate**, not an
            automatic verdict that either model was clinically wrong.
            """
        ),
        code(
            r"""
            analyses_by_run = defaultdict(set)
            for node in ledger.graph.find_nodes(node_type="decision"):
                meta = node.get("metadata") or {}
                if meta.get("run_id") and meta.get("analysis_id"):
                    analyses_by_run[str(meta["run_id"])].add(str(meta["analysis_id"]))

            task_only_cohort = []
            for candidate_run, analysis_ids in sorted(analyses_by_run.items()):
                matching = [
                    analysis_id for analysis_id in sorted(analysis_ids)
                    if ledger.load_analysis_artifact(candidate_run, analysis_id).get("task_arm")
                    == "task_only"
                ]
                if matching:
                    selected = ledger.selected_analysis(candidate_run)
                    task_only_cohort.append({
                        "run_id": candidate_run,
                        "analysis_id": selected if selected in matching else matching[0],
                    })

            divergence_report = ledger.find_divergent_decision_points(
                task_only_cohort, min_similarity=0.68
            ) if len(task_only_cohort) >= 2 else {"divergences": []}
            exact_divergences = [
                row for row in divergence_report.get("divergences", [])
                if row.get("same_situation_signature")
            ]
            if exact_divergences:
                divergence = exact_divergences[0]
                scenario_texts = {
                    (ledger.graph.find_node(str(member["decision_id"])).get("metadata") or {})
                    .get("scenario")
                    for member in divergence["members"]
                }
                exact_scenario = next(iter(scenario_texts)) if len(scenario_texts) == 1 else (
                    "Scenario strings did not agree; inspect the signature collision."
                )
                rows = [{
                    "model": member.get("review_model"),
                    "run": str(member.get("run_id"))[:24] + "…",
                    "outcome": member.get("outcome"),
                    "basis": ", ".join(member.get("basis_sources") or []),
                    "policy": ", ".join(member.get("policy_groundings") or []) or "none",
                } for member in divergence["members"]]
                display(Markdown(
                    f"**Decision Point:** `{divergence['decision_function']}/"
                    f"{divergence['decision_subject']}`  \n"
                    f"**Exact scenario:** `{exact_scenario}`  \n"
                    f"**Routing:** `{divergence['grounding_status']}`\n\n" +
                    markdown_table(rows, [
                        ("model", "Review model"), ("outcome", "Outcome"),
                        ("basis", "Claimed basis"), ("policy", "Policy grounding"),
                        ("run", "Run"),
                    ])
                ))
            else:
                divergence = None
                display(Markdown(
                    "**No exact divergent Decision Point was found in this cohort.** That is a "
                    "valid experimental result, not a query failure. Add repetitions/models in "
                    "Notebook 1 before concluding the guideline is stable."
                ))
            """
        ),
        md(
            """
            ## 4. Retrieve similar decisions as comparison candidates

            Semantica's native `ContextGraph.find_similar_decisions` returns nearby Decision
            scenarios. ACR then excludes the query run and requires the intended atomic subject.
            Similarity proposes where a human should compare; it does not establish equivalence,
            precedent, or clinical correctness.
            """
        ),
        code(
            r"""
            if divergence:
                query_episode_id = divergence["members"][0]["episode_id"]
            else:
                query_node = next(iter(ledger.graph.find_nodes(node_type="decision")))
                query_episode_id = (query_node.get("metadata") or {})["acr_episode_id"]
            similar_report = ledger.similar_candidates(
                query_episode_id, max_results=8, min_similarity=0.45
            )
            similar_rows = []
            for candidate in similar_report["candidates"]:
                decision = candidate["decision"]
                meta = decision.get("metadata") or {}
                similar_rows.append({
                    "similarity": f"{float(candidate.get('similarity') or 0):.2f}",
                    "same_signature": candidate.get("same_situation_signature"),
                    "category": decision.get("category"),
                    "subject": meta.get("decision_subject"),
                    "outcome": decision.get("outcome"),
                    "run": str(meta.get("run_id"))[:24] + "…",
                })
            assert similar_report["retrieval_engine"] == (
                "semantica.ContextGraph.find_similar_decisions"
            )
            display(Markdown(markdown_table(similar_rows, [
                ("similarity", "Similarity"), ("same_signature", "Exact signature"),
                ("category", "Function"), ("subject", "Subject"),
                ("outcome", "Outcome"), ("run", "Run"),
            ]) if similar_rows else "No cross-run candidate met the threshold."))
            """
        ),
        md(
            """
            ## 5. Traverse an evidenced causal chain

            ACR writes Semantica `CAUSED`, `INFLUENCED`, or `PRECEDENT_FOR` only when a
            `CausalAssertion` carries supporting runtime references. The query intentionally
            excludes Semantica's heuristic lowercase `influences` edges and mere temporal
            adjacency.

            We choose the latest episode with an evidenced incoming chain in an explicitly selected
            policy-guided analysis and ask what led to it.
            This is the data behind the human “why did the next step happen?” path; it is not the
            stock generic graph visualization.
            """
        ),
        code(
            r"""
            selections = ledger.graph.find_nodes(node_type="AnalysisSelection")
            assert selections, "Causal review requires an explicitly selected analysis"
            selection_candidates = []
            for selection in selections:
                meta = selection.get("metadata") or {}
                candidate_artifact = ledger.load_analysis_artifact(
                    str(meta["run_id"]), str(meta["analysis_id"])
                )
                selection_candidates.append((
                    candidate_artifact.get("task_arm") == "policy_bundle",
                    str(meta["run_id"]),
                    str(meta["analysis_id"]),
                    meta,
                ))
            selected_meta = max(selection_candidates, key=lambda row: row[:3])[3]
            causal_run = str(selected_meta["run_id"])
            causal_analysis = str(selected_meta["analysis_id"])
            scoped_decisions = [
                node for node in ledger.graph.find_nodes(node_type="decision")
                if (node.get("metadata") or {}).get("run_id") == causal_run
                and (node.get("metadata") or {}).get("analysis_id") == causal_analysis
            ]
            ordered_decisions = sorted(
                scoped_decisions,
                key=lambda node: int((node.get("metadata") or {}).get("source_seq_start") or 0),
            )
            final_node = None
            causal = None
            for candidate_node in reversed(ordered_decisions):
                candidate_episode = str(
                    (candidate_node.get("metadata") or {})["acr_episode_id"]
                )
                candidate_trace = ledger.causal_trace(candidate_episode, max_steps=12)
                if candidate_trace["chains"]:
                    final_node, causal = candidate_node, candidate_trace
                    break
            assert final_node is not None and causal is not None, (
                "The selected analysis has no evidenced causal chain"
            )
            final_episode = str((final_node.get("metadata") or {})["acr_episode_id"])
            hop_rows = []
            graph_edge_rows = [
                edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)
                for edge in ledger.graph.edges
            ]
            for chain_index, chain in enumerate(causal["chains"], 1):
                for hop_index, hop in enumerate(chain.get("hops") or [], 1):
                    source = ledger.graph.find_node(str(hop["from"])) or {}
                    target = ledger.graph.find_node(str(hop["to"])) or {}
                    assertion_node = next((
                        node for node in ledger.graph.find_nodes(node_type="CausalAssertion")
                        if (node.get("metadata") or {}).get("assertion_id")
                        == hop.get("assertion_id")
                    ), None)
                    support_count = sum(
                        edge.get("type") == "SUPPORTED_BY"
                        and assertion_node is not None
                        and edge.get("source_id") == assertion_node.get("id")
                        for edge in graph_edge_rows
                    )
                    hop_rows.append({
                        "chain": chain_index,
                        "hop": hop_index,
                        "from": (source.get("metadata") or {}).get("category"),
                        "type": hop.get("type"),
                        "to": (target.get("metadata") or {}).get("category"),
                        "assertion": hop.get("assertion_id"),
                        "provenance": hop.get("assertion_provenance"),
                        "support": support_count,
                    })
            assert hop_rows and all(row["assertion"] for row in hop_rows)
            display(Markdown(markdown_table(hop_rows, [
                ("chain", "Chain"), ("hop", "Hop"), ("from", "From"),
                ("type", "Relationship"), ("to", "To"),
                ("assertion", "Causal assertion"), ("provenance", "Provenance"),
                ("support", "Support nodes"),
            ])))
            """
        ),
        md(
            """
            ## 6. Ask for impact candidates — and keep the authority boundary visible

            Semantica can rank decisions that may have been influenced by a selected source. This
            native analysis includes graph, entity, category, and temporal signals; it is broader
            than the evidenced audit chain above. ACR therefore returns it as `CANDIDATE_ONLY`: it
            identifies decisions worth re-auditing, not a counterfactual claim that changing the
            source would change the final answer.
            """
        ),
        code(
            r"""
            first_hop = causal["chains"][0]["hops"][0]
            source_node = ledger.graph.find_node(str(first_hop["from"]))
            source_episode = str((source_node.get("metadata") or {})["acr_episode_id"])
            impact = ledger.impact_candidates(source_episode)
            native_impact = impact["candidates"]
            influence_rows = []
            for candidate in (native_impact.get("influence_scores") or [])[:8]:
                influence_rows.append({
                    "relation": "direct" if candidate.get("is_direct") else "indirect",
                    "score": f"{float(candidate.get('score') or 0):.2f}",
                    "category": candidate.get("category"),
                    "outcome": candidate.get("outcome"),
                    "run": str(((candidate.get("decision") or {}).get("metadata") or {})
                               .get("run_id"))[:24] + "…",
                })
            display(Markdown(
                f"**Authority:** `{impact['authority']}`; engine: "
                f"`{impact['retrieval_engine']}`; total candidates: "
                f"**{native_impact.get('total_influenced', 0)}**.\n\n" +
                (markdown_table(influence_rows, [
                    ("relation", "Candidate relation"), ("score", "Score"),
                    ("category", "Function"), ("outcome", "Outcome"), ("run", "Run"),
                ]) if influence_rows else "No candidate met Semantica's native thresholds.") +
                "\n\nCannot establish: " + ", ".join(impact["cannot_establish"])
            ))
            assert impact["authority"] == "CANDIDATE_ONLY"
            """
        ),
        md(
            """
            ## 7. Revise one Policy and retrieve the exact historical re-audit queue

            This cell works on a scratch copy of the ledger. It chooses a directly applied Policy,
            appends a new content-addressed version, and asks Semantica's `PolicyEngine` which
            historical decisions were bound to the old version.

            The output is precise about **historical direct bindings**. It cannot tell us whether
            the revised policy would change those decisions or their final answers; that requires
            paired reruns with a new Task Presentation.
            """
        ),
        code(
            r"""
            raw_edges = [edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)
                         for edge in ledger.graph.edges]
            applied_counts = Counter(
                str(edge["target_id"]) for edge in raw_edges if edge.get("type") == "APPLIED_POLICY"
            )
            assert applied_counts, "This cohort has no directly applied Policy"
            old_policy_node_id, binding_count = applied_counts.most_common(1)[0]
            old_policy = ledger.graph.find_node(old_policy_node_id)
            old_meta = old_policy.get("metadata") or {}
            policy_id = str(old_meta["policy_id"])
            from_version = str(old_meta["version"])

            scratch_root = ROOT / "runs/postdoc-notebook-output"
            scratch_root.mkdir(parents=True, exist_ok=True)
            scratch_path = scratch_root / "03_policy_sandbox.json"
            shutil.copy2(LEDGER_PATH, scratch_path)
            scratch = SemanticaLedger(scratch_path)
            revised_rules = json.loads(json.dumps(old_meta["rules"]))
            revised_rules["tutorial_change"] = (
                "Clarify this clause; demonstration only, not an approved clinical revision."
            )
            revision = scratch.register_policy_revision(
                policy_id,
                from_version=from_version,
                rules=revised_rules,
                change_reason="Postdoc notebook demonstration; not an approved guideline change.",
            )
            affected = scratch.affected_by_policy_change(
                policy_id,
                from_version=from_version,
                to_version=revision["version"],
            )
            affected_rows = [{
                "run": str(row.get("run_id"))[:24] + "…",
                "analysis": row.get("analysis_id"),
                "decisions": row.get("affected_decision_count"),
                "functions": ", ".join(row.get("decision_functions") or []),
            } for row in affected["affected_cases"]]
            display(Markdown(
                f"Policy `{policy_id}` had **{binding_count}** direct bindings in the source "
                f"ledger. New version: `{revision['version']}`.\n\n" +
                markdown_table(affected_rows, [
                    ("run", "Run"), ("analysis", "Analysis"),
                    ("decisions", "Affected decisions"), ("functions", "Functions"),
                ]) +
                "\n\n**Authority:** `" + affected["authority"] + "`; cannot establish: " +
                ", ".join(affected["cannot_establish"])
            ))
            """
        ),
        md(
            """
            ## 8. Inspect native provenance for one Decision

            The Decision node is a stable analytical index. Semantica ProvenanceManager retains
            who/what produced it, the reconstruction activity, parent analysis artifact, used
            cycles/testimony/basis references, checksums, and the complete episode payload with
            field-level authority.

            `verify_chain()` checks storage integrity and parent links. It does not validate that a
            medical interpretation is correct.
            """
        ),
        code(
            r"""
            provenance = ledger.provenance_manager(causal_run, causal_analysis)
            record = provenance.get_provenance(str(final_node["id"]))
            assert record, "The selected Decision has no native provenance record"
            integrity = provenance.verify_chain()
            provenance_summary = {
                "decision_id": record["entity_id"],
                "entity_type": record["entity_type"],
                "reconstructor_agent": record["agent_id"],
                "role": record["role"],
                "activity_id": record["activity_id"],
                "parent_analysis_artifact": record["parent_entity_id"],
                "used_entities": record["used_entities"],
                "checksum_prefix": str(record["checksum"])[:16],
                "episode_field_provenance": (record.get("metadata") or {})
                    .get("episode", {}).get("field_provenance"),
                "integrity": integrity,
            }
            display(provenance_summary)
            assert integrity["valid"] is True
            """
        ),
        md(
            """
            ## 9. Optional capability: export provenance as W3C PROV Turtle

            ACR does not currently use Semantica's general graph Export module. The native
            ProvenanceManager can nevertheless serialize the lineage it already stores. This is a
            useful integration experiment for an external audit archive, so we demonstrate it
            without treating it as part of the maintained pipeline.
            """
        ),
        code(
            r"""
            turtle = provenance.export_prov(format="turtle")
            turtle_path = ROOT / "runs/postdoc-notebook-output/selected-run-provenance.ttl"
            turtle_path.write_text(turtle)
            display({
                "experimental_export": display_path(turtle_path),
                "bytes": len(turtle.encode("utf-8")),
                "first_lines": turtle.splitlines()[:8],
                "production_status": "EXPLORATORY_NOT_MAINTAINED_WORKFLOW",
            })
            """
        ),
        md(
            """
            ## 10. Run-scoped Decision insights

            Semantica's insight summary is recomputed on one explicitly selected run/analysis so
            repeated reconstructions do not inflate counts. Here `confidence` means reconstruction
            stability, not clinical correctness.
            """
        ),
        code(
            r"""
            insights = ledger.insights(causal_run, causal_analysis)
            display({
                "run_id": causal_run,
                "analysis_id": causal_analysis,
                "episode_count": insights["episode_count"],
                "decision_functions": insights["categories"],
                "reconstruction_stability": insights["reconstruction_stability"],
                "semantica_scoped_insights": insights["semantica_scoped_insights"],
            })
            """
        ),
        md(
            """
            ## 11. Open the interactive Explorer view

            Use the same selected run and ledger in a terminal:

            ```bash
            acr review-ui <RUN_ID> --run-dir <LEDGER_PARENT>/<RUN_ID> \\
              --ledger <LEDGER_PATH> --port 8877
            ```

            Explorer is the delivery shell, while ACR's Decisions workspace supplies the
            chart-review narrative, step-by-step audit controls, trace links, and durable human
            review provenance. Generic Semantica graph visualization is not the primary review
            surface.
            """
        ),
        code(
            r"""
            command = (
                f"acr review-ui {causal_run} --run-dir {display_path(LEDGER_PATH.parent / causal_run)} "
                f"--ledger {display_path(LEDGER_PATH)} --port 8877"
            )
            display(Markdown(f"```bash\n{command}\n```"))

            closure = {
                "schema": "acr.postdoc_semantica_capabilities.v1",
                "ledger": display_path(LEDGER_PATH),
                "context_graph": stats,
                "claims": {
                    "native_similarity_query_executed": True,
                    "exact_divergence_found_in_this_cohort": bool(divergence),
                    "evidenced_causal_chain_found": bool(hop_rows),
                    "policy_direct_binding_queue_found": bool(affected["affected_decisions"]),
                    "provenance_chain_valid": integrity["valid"],
                    "scoped_insights_executed": insights["episode_count"] > 0,
                },
                "authority_limits": {
                    "similarity": "comparison candidates, not clinical precedent",
                    "impact": "re-audit candidates, not counterfactual answer change",
                    "provenance": "lineage/integrity, not semantic correctness",
                    "policy_change": "historical direct bindings, not automatic non-compliance",
                },
            }
            output = ROOT / "runs/postdoc-notebook-output/03_semantica_capabilities.json"
            output.write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n")
            display(Markdown(
                f"**Notebook 3 closed.** Machine-readable summary: `{display_path(output)}`."
            ))
            """
        ),
    ],
    title="Semantica Decision Intelligence on real chart-review runs",
)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "01_run_chart_review_experiments.ipynb": NB1,
        "02_trace_to_decision_chain.ipynb": NB2,
        "03_semantica_decision_intelligence.ipynb": NB3,
    }.items():
        nbformat.write(value, NOTEBOOKS / name)
        print(NOTEBOOKS / name)


if __name__ == "__main__":
    main()
