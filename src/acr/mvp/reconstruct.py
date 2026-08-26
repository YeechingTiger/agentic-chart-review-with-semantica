"""Read a finished run back as a decision tree: big points, and the small ones inside them.

A run's trace records ACTIONS — searched this, opened that, submitted — plus whatever the
model chose to narrate. It does not record JUDGMENTS in any comparable form, and it carries no
taxonomy at all, on purpose: what kind of judgment something was is decided here, afterwards,
against a vocabulary still being grown from real runs, where changing it costs one
re-extraction instead of a re-run of every review ever done.

FOUR STEPS, AND ONLY THE SECOND IS A MODEL.

  1. `run_sheet`   — deterministic. The trace and the harness's event stream, interleaved and
                     numbered, every line anchored to the seq it came from.
  2. `extract`     — one LLM call. Segments the sheet into big points, each with the small
                     points inside it, each typed and given the four semantica fields.
  3. `verify`      — deterministic. Spans must be real seqs. Claimed inputs go through the
                     same `RunFacts` the live server used. And every field the extractor
                     attributes to the model must come with a QUOTE, which is checked against
                     the sheet.
  4. `build`       — deterministic. Into semantica: `record_decision` per point, INFLUENCED
                     between big points, PART_OF from small to big.

THE QUOTE IS THE LOAD-BEARING PART. Asking a model how confident it is produces a number that
means nothing. Asking it *which line it read this off* produces a claim that either matches
the sheet or does not, so provenance is computed, not self-assessed:

    DETERMINISTIC   the server recorded it — the tool called, the gate's verdict, the span
    SELF_REPORTED   the model said it during the run, and the quote proves where
    RECONSTRUCTED   this reader inferred it; nobody said it

That distinction is not decoration. A `own_knowledge` grounding that is SELF_REPORTED means a
run told us the contract ran out; the same label RECONSTRUCTED means only that this reader
thought so. One is a question for a domain expert, the other is a guess, and a report that
mixed them would spend expert attention on noise.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from acr.mvp.decision_types import (BIG, DECISION_TYPES, SMALL, level_of, normalize_type,
                                    types_for)
from acr.mvp.observe import decision_trace
from acr.mvp.warrants import GROUNDING_KINDS, RunFacts, normalize_grounding

DETERMINISTIC = "deterministic"
SELF_REPORTED = "self_reported"
RECONSTRUCTED = "reconstructed"

#: How much of a quote must appear in the sheet for the attribution to stand. Exact substring
#: match is too brittle (a model re-types an em dash), and fuzzy matching would let a
#: paraphrase pass as a quotation, which is exactly the failure this check exists to catch.
#: Normalised whitespace and case, then containment either way, is the honest middle.
_MIN_QUOTE_CHARS = 12


class StructuredLLM(Protocol):
    """What an extractor needs. `semantica.llms.LiteLLM` and its siblings satisfy it as-is,
    which is why nothing here builds an HTTP client."""

    def generate_structured(self, prompt: str) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- 1. the run sheet
def run_sheet(run_dir: Path) -> dict[str, Any]:
    """The run as numbered, seq-anchored lines — the only thing the extractor gets to read."""
    trace = decision_trace(Path(run_dir))
    lines: list[str] = []
    seqs: set[int] = set()
    for step in trace["steps"]:
        seq = step.get("seq")
        if isinstance(seq, int):
            seqs.add(seq)
        tag = f"seq {seq}" if seq is not None else "no-seq"
        kind = step["kind"]
        if kind in ("thought", "remark"):
            lines.append(f"[{tag}] {kind} (SELF-REPORTED, between tool calls): {step['text']}")
        elif kind == "decision":
            body = (f"the model noted a decision — facing: {step.get('facing')} | "
                    f"decided: {step.get('decision')} | because: {step.get('because')}")
            if step.get("grounding"):
                body += f" | grounding it claimed: {', '.join(step['grounding'])}"
            if step.get("used"):
                body += f" | inputs it cited: {', '.join(u.get('ref', '') for u in step['used'])}"
            if step.get("options"):
                body += f" | set aside: {'; '.join(step['options'])}"
            lines.append(f"[{tag}] decision (SELF-REPORTED content, server-timed): {body}")
        elif kind == "action":
            obj = f' | objective it stated: "{step["objective"]}"' if step.get("objective") else ""
            lines.append(f"[{tag}] action (SERVER FACT): {step['tool']} "
                         f"{json.dumps(step.get('args'), ensure_ascii=False)}"
                         f" -> {step.get('observed')}{obj}")
        elif kind == "evidence":
            lines.append(f"[{tag}] evidence (SERVER FACT): {step.get('note_id')} "
                         f'"{step.get("quote")}" — {step.get("supports")}')
        elif kind == "submission":
            lines.append(f"[{tag}] submission (SERVER FACT): {step.get('status')} "
                         f"{json.dumps(step.get('value'), ensure_ascii=False)}"
                         f" | reasoning: {step.get('reasoning')}")
        elif kind == "verdict":
            word = "ACCEPTED" if step.get("accepted") else "REFUSED"
            lines.append(f"[{tag}] gate verdict (SERVER FACT): {word} — {step.get('why')}")
        elif kind == "result":
            lines.append(f"[{tag}] result (SERVER FACT): {step.get('status')}")
    return {**{k: trace[k] for k in ("run_id", "patient_id", "spec_id")},
            "lines": lines, "seqs": seqs, "steps": trace["steps"]}


def render_sheet(sheet: dict[str, Any]) -> str:
    head = (f"RUN {sheet['run_id']} | patient {sheet['patient_id']} | "
            f"contract {sheet['spec_id']}")
    return "\n".join([head, *(f"{n:3d}. {line}" for n, line in enumerate(sheet["lines"], 1))])


# ------------------------------------------------------------------------- 2. the extraction
def _type_menu(level: str) -> str:
    return "\n".join(f"    {n} — {DECISION_TYPES[n].about}"
                     for n in types_for(level) if n != "other")


def build_prompt(sheet: dict[str, Any]) -> str:
    return f"""You are reading the record of one clinical chart review that has already
finished, and rewriting it as the decisions it was made of. You are NOT reviewing the chart
and NOT judging whether the answer was right.

{render_sheet(sheet)}

Split this run into BIG POINTS. A big point is a conclusion the run reached and what it
decided to do next — the kind of thing that takes several actions to reach. Inside each big
point, list the SMALL POINTS: the single-action choices that got there (which term to search,
which document to open, what one passage says).

Big point types:
{_type_menu(BIG)}

Small point types:
{_type_menu(SMALL)}

For every point give:
  span            [first_seq, last_seq] — seqs from the sheet above. Small points: [n, n].
  decision_type   one name from the matching list, or "other" if none fits. Prefer "other"
                  over a type that is merely close; a wrong name is worse than an unnamed one.
  scenario        the state BEFORE deciding — what was open, what was known. Write it so it
                  would still make sense with the outcome erased, and so another run facing
                  the same situation would produce the same sentence: no patient details, no
                  document names, no dates.
  reasoning       why it went this way. Specific and concrete, the opposite of scenario.
  outcome         what was decided, in a short phrase.
  grounding       list from: {', '.join(GROUNDING_KINDS)}. [] if you cannot tell.
  used            inputs it rested on: note:<id>, search:<query>, evidence:<n>, rule:<name>.
  quote           VERBATIM text from a line above showing the model itself said this, or null.

About `quote` — this is the important one. If the model stated the reasoning or the grounding
during the run, copy the words it used. If you are inferring, put null. Do NOT paraphrase into
the quote field and do NOT quote a SERVER FACT line as evidence of what the model thought. An
honest null is worth more than a quote that does not hold up: every quote is checked against
the sheet, and the ones that fail are reported as failures.

Return JSON only:
{{"big_points": [{{"span": [1, 4], "decision_type": "...", "scenario": "...",
  "reasoning": "...", "outcome": "...", "grounding": [], "used": [], "quote": null,
  "small_points": [{{"span": [1, 1], "decision_type": "...", "scenario": "...",
    "reasoning": "...", "outcome": "...", "grounding": [], "used": [], "quote": null}}]}}]}}"""


def extract(sheet: dict[str, Any], llm: StructuredLLM) -> dict[str, Any]:
    return llm.generate_structured(build_prompt(sheet))


# ----------------------------------------------------------------------- 3. the verification
def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def _quote_holds(quote: Any, sheet: dict[str, Any]) -> bool:
    """Did the model actually say this? Checked against the SELF-REPORTED lines only — a
    quote lifted from a server fact proves the server observed something, never that the
    model thought it, and letting that pass would turn the trace's own record into evidence
    of reasoning nobody did."""
    if not isinstance(quote, str) or len(quote.strip()) < _MIN_QUOTE_CHARS:
        return False
    needle = _normalise(quote)
    return any(needle in _normalise(line) for line in sheet["lines"]
               if "SELF-REPORTED" in line)


def _verify_point(raw: dict[str, Any], level: str, sheet: dict[str, Any],
                  facts: RunFacts) -> dict[str, Any] | None:
    span = raw.get("span")
    if not (isinstance(span, list) and len(span) == 2
            and all(isinstance(v, int) for v in span)):
        return None
    lo, hi = min(span), max(span)
    covered = sorted(s for s in sheet["seqs"] if lo <= s <= hi)
    if not covered:
        return None   # a point anchored to nothing cannot be checked, so it is not a point

    dtype, claimed = normalize_type(raw.get("decision_type"))
    if dtype != "other" and dtype not in types_for(level):
        # A real type at the wrong level is a segmentation error, not a naming one. Keep the
        # name and let the level follow from it rather than silently discarding either.
        level = level_of(dtype, on_action=(level == SMALL))
    grounding, off_vocabulary = normalize_grounding(raw.get("grounding"))
    used = facts.resolve_all(raw.get("used"))
    quoted = _quote_holds(raw.get("quote"), sheet)
    return {
        "level": level, "decision_type": dtype, "claimed_type": claimed,
        "scenario": str(raw.get("scenario") or "").strip(),
        "reasoning": str(raw.get("reasoning") or "").strip(),
        "outcome": str(raw.get("outcome") or "").strip(),
        "grounding": grounding, "grounding_off_vocabulary": off_vocabulary,
        "used": [u["ref"] for u in used],
        "used_unverified": [u["ref"] for u in used if u.get("verified") is False],
        "span": [lo, hi], "seq": covered[0], "covers": covered,
        "quote": raw.get("quote") if quoted else None,
        # The whole point of the quote check: what the model said, versus what this reader
        # decided it must have meant.
        "provenance": SELF_REPORTED if quoted else RECONSTRUCTED,
        # A grounding claim is only worth an expert's attention when a run actually made it.
        "grounding_provenance": SELF_REPORTED if (grounding and quoted) else RECONSTRUCTED,
        "quote_rejected": bool(raw.get("quote")) and not quoted,
    }


def verify(raw: dict[str, Any], sheet: dict[str, Any],
           facts: RunFacts) -> dict[str, Any]:
    """Everything the extractor claimed, checked against the record. Never raises on bad
    extractor output — a dropped point is counted and reported, because a reconstruction that
    silently discarded half a run would read exactly like a short review."""
    points: list[dict[str, Any]] = []
    dropped = {"big": 0, "small": 0}
    for raw_big in (raw.get("big_points") or []):
        if not isinstance(raw_big, dict):
            dropped["big"] += 1
            continue
        big = _verify_point(raw_big, BIG, sheet, facts)
        if big is None:
            dropped["big"] += 1
            dropped["small"] += len(raw_big.get("small_points") or [])
            continue
        big["small_points"] = []
        for raw_small in (raw_big.get("small_points") or []):
            small = (_verify_point(raw_small, SMALL, sheet, facts)
                     if isinstance(raw_small, dict) else None)
            if small is None:
                dropped["small"] += 1
                continue
            big["small_points"].append(small)
        big["small_points"].sort(key=lambda p: p["seq"])
        points.append(big)
    points.sort(key=lambda p: p["seq"])
    return {"points": points, "dropped": dropped, **_health(points, sheet)}


def _health(points: list[dict[str, Any]], sheet: dict[str, Any]) -> dict[str, Any]:
    """The four numbers a reader needs before believing any of the rest."""
    every = [p for big in points for p in (big, *big["small_points"])]
    covered = {s for p in every for s in p["covers"]}
    own_knowledge = [p for p in every if "own_knowledge" in p["grounding"]]
    return {
        "n_big": len(points), "n_small": sum(len(b["small_points"]) for b in points),
        "types": dict(Counter(f"{p['level']}:{p['decision_type']}" for p in every)),
        # A seq nobody accounted for is a stretch of the run this reading does not explain.
        "seqs_unaccounted": sorted(sheet["seqs"] - covered),
        "self_reported": sum(1 for p in every if p["provenance"] == SELF_REPORTED),
        "reconstructed": sum(1 for p in every if p["provenance"] == RECONSTRUCTED),
        "quotes_rejected": sum(1 for p in every if p["quote_rejected"]),
        "own_knowledge": {
            "total": len(own_knowledge),
            "self_reported": sum(1 for p in own_knowledge
                                 if p["grounding_provenance"] == SELF_REPORTED)},
        "unverified_warrants": [{"seq": p["seq"], "decided": p["outcome"],
                                 "refs": p["used_unverified"]}
                                for p in every if p["used_unverified"]],
    }


# ------------------------------------------------------------------------------- 4. the graph
def build(verified: dict[str, Any], ledger: Any, *, run_id: str, spec_id: str,
          case_id: str) -> dict[str, Any]:
    """Into semantica. Big points chain by INFLUENCED; small points hang off their big point
    by PART_OF, which `get_causal_chain` does not walk — so the audit chain shows what the run
    concluded, and the steps behind any one conclusion are one query away instead of in the
    way."""
    def entities(point: dict[str, Any]) -> list[str]:
        return [f"case:{case_id}", f"spec:{spec_id}",
                f"type:{point['level']}:{point['decision_type']}", *point["used"][:20]]

    def record(point: dict[str, Any], parent_seq: int | None) -> str:
        return ledger.record_judgment(
            run_id, category=f"{point['level']}:{point['decision_type']}",
            scenario=point["scenario"], reasoning=point["reasoning"],
            outcome=point["outcome"], decision_maker="model",
            entities=entities(point),
            metadata={"spec_id": spec_id, "case_id": case_id, "seq": point["seq"],
                      "span": point["span"], "used": point["used"],
                      "used_unverified": point["used_unverified"],
                      "grounding": point["grounding"],
                      "grounding_provenance": point["grounding_provenance"],
                      "provenance": point["provenance"], "quote": point["quote"],
                      "claimed_type": point["claimed_type"], "parent_seq": parent_seq,
                      "reconstructed": True})

    previous: str | None = None
    for big in verified["points"]:
        big_id = record(big, None)
        if previous is not None:
            ledger.link_influenced(previous, big_id)
        previous = big_id
        for small in big["small_points"]:
            ledger.link_part_of(record(small, big["seq"]), big_id)
    # Hang the tree off the run's own result, or the audit verb — which walks upstream from
    # there — would never reach it, and a reconstruction nobody can reach from the answer it
    # explains is an artefact rather than a reading.
    anchor = ledger.result_node(case_id)
    if previous is not None and anchor is not None:
        ledger.link_influenced(previous, anchor)
    ledger.save()
    return {"run_id": run_id, "case_id": case_id,
            **{k: v for k, v in verified.items() if k != "points"}}


# ------------------------------------------------------------------------------ the whole verb
def reconstruct_run(run_dir: Path, ledger: Any, llm: StructuredLLM, *,
                    passes: int = 1) -> dict[str, Any]:
    """Read one finished run into the ledger as a decision tree.

    `passes` > 1 extracts the same sheet repeatedly and reports how much the readings differ,
    WITHOUT writing the extra ones. That number is the instrument checking itself: if two
    readings of one unchanged run disagree about how many decisions it contained, the taxonomy
    is not yet sharp enough to compare across runs, and no amount of downstream analysis
    fixes that. Only the first pass is stored, so the drift is a diagnostic and never data.
    """
    run_dir = Path(run_dir)
    sheet = run_sheet(run_dir)
    events = [json.loads(ln) for ln
              in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
              if ln.strip()]
    facts = RunFacts.from_trace(events)
    meta = next((e for e in events if e.get("kind") == "run_meta"), {})

    readings = [verify(extract(sheet, llm), sheet, facts) for _ in range(max(1, passes))]
    summary = build(readings[0], ledger, run_id=sheet["run_id"],
                    spec_id=meta.get("spec_id", sheet["spec_id"] or "unknown"),
                    case_id=meta.get("patient_id", sheet["patient_id"] or sheet["run_id"]))
    if len(readings) > 1:
        summary["stability"] = {
            "passes": len(readings),
            "n_big": [r["n_big"] for r in readings],
            "n_small": [r["n_small"] for r in readings],
            "types_agree": len({tuple(sorted(r["types"].items())) for r in readings}) == 1,
        }
    return summary


def render(summary: dict[str, Any]) -> str:
    """The reconstruction's own health, which a reader needs before trusting the tree."""
    out = [f"{summary['run_id']} | case {summary['case_id']}",
           f"  {summary['n_big']} big point(s), {summary['n_small']} small",
           f"  provenance: {summary['self_reported']} self-reported, "
           f"{summary['reconstructed']} reconstructed"]
    if summary.get("quotes_rejected"):
        out.append(f"  {summary['quotes_rejected']} quote(s) did not hold up against the "
                   f"record and were dropped to reconstructed")
    ok = summary["own_knowledge"]
    if ok["total"]:
        out.append(f"  own_knowledge: {ok['total']}, of which {ok['self_reported']} the run "
                   f"actually said so — only those are questions for a domain expert")
    if summary.get("seqs_unaccounted"):
        out.append(f"  UNACCOUNTED seqs (no point covers them): {summary['seqs_unaccounted']}")
    if any(summary["dropped"].values()):
        out.append(f"  dropped as unanchored: {summary['dropped']}")
    for w in summary.get("unverified_warrants", []):
        out.append(f"  FALSE WARRANT at seq {w['seq']}: decided \"{w['decided']}\" citing "
                   f"{', '.join(w['refs'])} — never read or surfaced in this run")
    if "stability" in summary:
        s = summary["stability"]
        out.append(f"  stability over {s['passes']} readings: big={s['n_big']} "
                   f"small={s['n_small']} types_agree={s['types_agree']}")
    return "\n".join(out)
