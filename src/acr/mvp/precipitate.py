"""The third verb: read a class of decision points and say what has settled and what has not.

Audit reads one case's chain; compare reads a class; **precipitate** asks the question those
two are for — *which of these decisions could become a rule?* It answers with measurement
only. It never writes a rule and never proposes wording: a guideline clause is a domain
expert's to author, and a module that invented them would be laundering a model's habits into
policy.

What it measures, per decision type:

  * **Situations** — decision points clustered by what the run said it was facing. The cluster
    is the unit a rule would govern: one recurring situation, one clause.
  * **Settled** — a situation where every run decided the same way, seen enough times to be
    worth fixing. This is a clause waiting to be written down.
  * **Divergent** — a situation where runs decided differently. This is a gap, and the report
    classifies WHICH KIND of gap, because the two need opposite remedies:

        same information, different call   -> a JUDGEMENT divergence. The runs both saw what
                                              they needed and disagreed, so the contract does
                                              not settle the question: it wants a Decision
                                              Rule (or a Conflict Rule naming the
                                              Discriminating Fact that separates them).
        different information, different   -> an INFORMATION divergence. The runs decided
        call                                  differently because they had looked at
                                              different things, so the remedy is upstream —
                                              a retrieval or Coverage rule about what must be
                                              examined before this decision is taken.

  * **Unverified warrants** — decisions citing information this run never read or surfaced.
    CONTEXT.md's warning made executable: a Warrant can be articulate and false.

Similarity is deliberately plain — token overlap over the `facing` text, no embeddings, no
model. A clustering a person cannot reproduce by hand is a clustering they cannot argue with,
and every number here has to survive being disagreed with.
"""
from __future__ import annotations

import re
from typing import Any

from acr.mvp.decision_types import BIG, SMALL

#: Words carrying no situational content. Short and blunt on purpose: a long curated stoplist
#: is a hidden model of the domain, and this module is supposed to have none.
_STOP = frozenset("""
a an the this that these those is are was were be been being do does did doing have has had
and or but if then than so of in on at to for from with without by as it its there here
i we you he she they them us our your their what which who whom when where why how
no not nor yes any some all both each more most other same such only own too very
""".split())

_SETTLED_MIN = 3        # a practice seen fewer times than this is an anecdote, not a practice
_SIMILAR = 0.34         # Jaccard over content words; two ways of saying one situation clear it
_SHARED_INPUTS = 0.5    # above this overlap, two decisions were working from the same material

#: Outcomes are grouped far more strictly than situations are clustered, and the asymmetry is
#: deliberate. "the cytology dates the case" and "the biopsy dates the case" differ by one
#: word and are opposite calls; at the situation threshold they would merge, and a merge HIDES
#: a divergence — nobody ever looks again. A split, by contrast, reports a divergence a reader
#: dismisses in a second. So the errors are not symmetric, and this leans towards splitting.
_SAME_CALL = 0.75


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return frozenset(w for w in words if w not in _STOP and len(w) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Greedy single-pass clustering on `facing` similarity, seeded in ledger order.

    Greedy rather than agglomerative because the result has to be explainable to the person
    reading the report: every member joined the first cluster it was similar enough to, and
    the cluster's own text is the first member's."""
    clusters: list[tuple[frozenset[str], list[dict[str, Any]]]] = []
    for row in rows:
        toks = _tokens(row.get("scenario") or "")
        for seed, members in clusters:
            if _jaccard(toks, seed) >= _SIMILAR:
                members.append(row)
                break
        else:
            clusters.append((toks, [row]))
    return [members for _, members in clusters]


def _outcome_groups(members: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The decisions in one situation, grouped by what was actually decided."""
    groups: list[tuple[frozenset[str], list[dict[str, Any]]]] = []
    for row in members:
        toks = _tokens(row.get("outcome") or "")
        for seed, g in groups:
            if _jaccard(toks, seed) >= _SAME_CALL:
                g.append(row)
                break
        else:
            groups.append((toks, [row]))
    groups.sort(key=lambda g: -len(g[1]))
    return [g for _, g in groups]


def _inputs(row: dict[str, Any]) -> frozenset[str]:
    return frozenset(str(u) for u in (row.get("used") or []))


def _classify_divergence(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Judgement or information? Decided by how much the two camps' cited inputs overlap.

    Compares the two largest camps only: with three-way splits the report shows the whole
    table anyway, and a single verdict over three camps would be a summary of a summary."""
    a, b = groups[0], groups[1]
    ia = frozenset().union(*(_inputs(r) for r in a)) if a else frozenset()
    ib = frozenset().union(*(_inputs(r) for r in b)) if b else frozenset()
    overlap = _jaccard(ia, ib)
    if not ia or not ib:
        kind, remedy = "unclassified", "the camps cite no inputs to compare"
    elif overlap >= _SHARED_INPUTS:
        kind = "judgement"
        remedy = ("both camps worked from the same material and still disagreed — the "
                  "contract does not settle this: it wants a Decision Rule, or a Conflict "
                  "Rule naming the Discriminating Fact that separates them")
    else:
        kind = "information"
        remedy = ("the camps had looked at different things — the remedy is upstream of the "
                  "judgement: a retrieval or Coverage rule about what must be examined "
                  "before this decision is taken")
    return {"kind": kind, "input_overlap": round(overlap, 3), "remedy": remedy,
            "inputs_only_in_majority": sorted(ia - ib)[:8],
            "inputs_only_in_minority": sorted(ib - ia)[:8]}


def _context_span(members: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """min/median/max of one server-recorded counter across a group — the measured half of
    'what did they know when they decided this'."""
    vals = sorted(v for v in ((r.get("context") or {}).get(key) for r in members)
                  if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not vals:
        return None
    return {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1]}


def survey(ledger: Any, *, decision_type: str | None = None, level: str | None = None,
           settled_min: int = _SETTLED_MIN) -> dict[str, Any]:
    """The guideline material in the ledger, one section per level and decision type.

    Reads CLASSIFIED decision points — the `big:` and `small:` categories that
    `acr.mvp.reconstruct` writes. Bare `step` rows, which is what a run records live, carry no
    type and so have nothing to be compared across; reconstruct a run before surveying it.

    Big and small are sectioned apart because a divergence in each means a different thing: a
    small point splitting is two runs taking different routes to the same place, which costs
    time; a big point splitting changed what the case concluded. Big sections come first.
    """
    levels = [level] if level else [BIG, SMALL]
    rows = [r for lv in levels for r in ledger.decisions(category_prefix=f"{lv}:")]
    if decision_type:
        rows = [r for r in rows if str(r.get("category", "")).split(":", 1)[-1] == decision_type]

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("category", "?:?")), []).append(row)

    sections = []
    order = {BIG: 0, SMALL: 1}
    for category, drows in sorted(by_type.items(),
                                  key=lambda kv: (order.get(kv[0].split(":")[0], 9),
                                                  -len(kv[1]), kv[0])):
        lvl, _, dtype = category.partition(":")
        situations = []
        for members in _cluster(drows):
            groups = _outcome_groups(members)
            situation: dict[str, Any] = {
                "facing": members[0].get("scenario"),
                "n_decisions": len(members),
                "cases": sorted({str(r.get("case_id")) for r in members if r.get("case_id")}),
                "runs": sorted({str(r.get("run_id")) for r in members if r.get("run_id")}),
                "outcomes": [{
                    "decided": g[0].get("outcome"),
                    "n": len(g),
                    "because": [r.get("reasoning") for r in g][:3],
                    "cases": sorted({str(r.get("case_id")) for r in g if r.get("case_id")}),
                    "inputs": sorted(frozenset().union(*(_inputs(r) for r in g)))[:10],
                    "searches_when_decided": _context_span(g, "n_searches"),
                    "evidence_when_decided": _context_span(g, "n_evidence"),
                    "set_aside": sorted({o for r in g for o in (r.get("options") or [])})[:5],
                } for g in groups],
            }
            if len(groups) > 1:
                situation["status"] = "divergent"
                situation["divergence"] = _classify_divergence(groups)
            elif len(members) >= settled_min:
                situation["status"] = "settled"
                situation["settled_note"] = (
                    f"{len(members)} decisions across "
                    f"{len(situation['cases'])} case(s) all went the same way — a clause "
                    f"could fix this, and the runs would lose nothing")
            else:
                situation["status"] = "thin"
            situations.append(situation)

        situations.sort(key=lambda s: ({"divergent": 0, "settled": 1, "thin": 2}[s["status"]],
                                       -s["n_decisions"]))
        unverified = [{"case_id": r.get("case_id"), "seq": r.get("seq"),
                       "decided": r.get("outcome"), "refs": r.get("used_unverified")}
                      for r in drows if r.get("used_unverified")]
        sections.append({
            "decision_type": dtype,
            "level": lvl,
            "n_decisions": len(drows),
            "n_cases": len({str(r.get("case_id")) for r in drows if r.get("case_id")}),
            "n_runs": len({str(r.get("run_id")) for r in drows if r.get("run_id")}),
            "situations": situations,
            "unverified_warrants": unverified,
        })

    return {"n_decisions": len(rows), "sections": sections,
            "settled_min": settled_min, "similarity_threshold": _SIMILAR}


def render(report: dict[str, Any]) -> str:
    """The survey as something a domain expert reads and argues with."""
    out: list[str] = [f"{report['n_decisions']} decision point(s) recorded"]
    if not report["sections"]:
        out.append("  (nothing recorded yet — run some reviews with a ledger attached)")
        return "\n".join(out)

    for sec in report["sections"]:
        out.append("")
        out.append(f"## [{sec['level']}] {sec['decision_type']} — "
                   f"{sec['n_decisions']} decision(s), "
                   f"{sec['n_cases']} case(s), {sec['n_runs']} run(s)")
        for s in sec["situations"]:
            mark = {"divergent": "DIVERGENT", "settled": "SETTLED  ", "thin": "thin     "}
            out.append("")
            out.append(f"  [{mark[s['status']]}] facing: {s['facing']}")
            out.append(f"              {s['n_decisions']} decision(s) over "
                       f"{len(s['cases'])} case(s): {', '.join(s['cases'][:6])}")
            for o in s["outcomes"]:
                out.append(f"      x{o['n']:<2} decided: {o['decided']}")
                if o["because"] and o["because"][0]:
                    out.append(f"           because: {o['because'][0]}")
                if o["inputs"]:
                    out.append(f"           used: {', '.join(o['inputs'])}")
                span = o["searches_when_decided"]
                if span:
                    out.append(f"           searches when decided: min {span['min']}, "
                               f"median {span['median']}, max {span['max']}")
                if o["set_aside"]:
                    out.append(f"           set aside: {'; '.join(o['set_aside'][:3])}")
            if s["status"] == "divergent":
                d = s["divergence"]
                out.append(f"      -> {d['kind'].upper()} divergence "
                           f"(input overlap {d['input_overlap']})")
                out.append(f"         {d['remedy']}")
                if d["inputs_only_in_majority"]:
                    out.append(f"         only the majority had: "
                               f"{', '.join(d['inputs_only_in_majority'])}")
                if d["inputs_only_in_minority"]:
                    out.append(f"         only the minority had: "
                               f"{', '.join(d['inputs_only_in_minority'])}")
            elif s["status"] == "settled":
                out.append(f"      -> {s['settled_note']}")
        if sec["unverified_warrants"]:
            out.append("")
            out.append("  UNVERIFIED WARRANTS (cited information this run never read or saw):")
            for u in sec["unverified_warrants"][:10]:
                out.append(f"    {u['case_id']} seq={u['seq']}: {', '.join(u['refs'])}")
                out.append(f"      decided: {u['decided']}")
    return "\n".join(out)
