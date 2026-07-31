"""`acr site-mapping` — build, review and diff the local-type-name to concept table.

FOUR COMMANDS, AND THREE OF THEM COST NOTHING

  types    the corpus's document-type inventory with counts. One directory scan, no model,
           no chart text. This is the entire input the classification pass is given.
  build    the model pass. One batch of ~120 type names per call; 1,516 names is 13 calls,
           once, for the whole corpus. Costs money exactly once per corpus per concept
           vocabulary.
  review   the table a registrar reads, ordered so the rows deciding the most documents come
           first, and the rows nobody could place come before those.
  diff     what the retired `doc_type_matches` substring expression would have said, against
           what the mapping says. THE ACCEPTANCE REPORT: it names every document type whose
           stratum changes, so "this fixes the abstentions" is a list of type names and
           document counts rather than a claim.

`build` is the only command that needs credentials, and `diff` is the one to run first --
it answers "is the mapping better than what it replaces" without touching a patient.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from ..chartstore.corpus import Corpus
from ..contract.site_mapping import (
    UNMAPPED,
    SiteMapping,
    build_site_mapping,
    concepts_from_strata,
)
from ..contract.spec import load_specs
from ..review.coverage import strata_from_spec

site_mapping_app = typer.Typer(add_completion=False,
                               help="Local note-type name to portable document concept.")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
    tmp.replace(path)


def _strata_for(spec_dir: str, spec_id: str):
    specs = load_specs(spec_dir)
    spec = specs.get(spec_id)
    if spec is None:
        raise typer.BadParameter(f"no spec {spec_id!r} in {spec_dir}; have {sorted(specs)}")
    strata = strata_from_spec(spec)
    if not strata:
        raise typer.BadParameter(f"spec {spec_id!r} declares no strata, so it needs no mapping")
    return spec, strata


# ------------------------------------------------------------------------------- types
@site_mapping_app.command("types")
def types_cmd(
    corpus: str = typer.Option(..., help="corpus root: a directory of patient directories"),
    out: str = typer.Option("", help="write the inventory here as JSON"),
    top: int = typer.Option(40, help="how many rows to print"),
    no_cache: bool = typer.Option(False, "--no-cache", help="rescan instead of reading the cache"),
):
    """The corpus's document-type inventory. No model, no chart text, no patient id."""
    counts = Corpus(Path(corpus)).doc_type_counts(use_cache=not no_cache)
    typer.echo(f"{len(counts)} distinct document types over {sum(counts.values())} documents")
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        typer.echo(f"  {n:7d}  {name}")
    if out:
        _write(Path(out), {"corpus": str(corpus), "n_types": len(counts),
                           "n_documents": sum(counts.values()), "counts": counts})
        typer.echo(f"wrote {out}")


# ------------------------------------------------------------------------------- build
@site_mapping_app.command("build")
def build_cmd(
    corpus: str = typer.Option(..., help="corpus root"),
    spec_id: str = typer.Option(..., "--spec", help="spec whose strata define the concepts"),
    out: str = typer.Option(..., help="where to write the mapping JSON"),
    spec_dir: str = typer.Option("assets/specs", help="directory of spec YAML"),
    built_at: str = typer.Option(..., help="ISO timestamp to stamp; passed in so the "
                                          "artifact is reproducible rather than clock-dependent"),
    batch_size: int = typer.Option(120, help="type names per model call"),
    model: str = typer.Option("", help="override ACR_MODEL"),
    no_cache: bool = typer.Option(False, "--no-cache"),
):
    """Classify every document-type name in the corpus against the spec's concepts.

    The model is given type NAMES and corpus COUNTS and nothing else -- no note id, no date,
    no patient id, no document text. See `acr.contract.site_mapping` and
    `tests/test_site_mapping.py::test_builder_is_given_type_names_and_counts_and_nothing_else`.
    """
    from ..core.llm import LLMClient, LLMConfig

    _, strata = _strata_for(spec_dir, spec_id)
    concepts = concepts_from_strata(strata)
    counts = Corpus(Path(corpus)).doc_type_counts(use_cache=not no_cache)

    typer.echo(f"{len(counts)} type names -> {len(concepts)} concepts "
               f"({', '.join(c.name for c in concepts)})")
    typer.echo(f"~{-(-len(counts) // batch_size)} model calls")

    llm = LLMClient(LLMConfig.from_env(**({"model": model} if model else {})))
    mapping = build_site_mapping(counts, concepts, llm, corpus_id=str(corpus),
                                 built_at=built_at, batch_size=batch_size)
    _write(Path(out), mapping.to_dict())

    per = {}
    for a in mapping.assignments.values():
        per.setdefault(a.concept, [0, 0])
        per[a.concept][0] += 1
        per[a.concept][1] += a.n_documents
    typer.echo(f"\nmapping_hash={mapping.mapping_hash} concepts_hash={mapping.bound_concepts_hash}")
    for concept, (n_types, n_docs) in sorted(per.items()):
        typer.echo(f"  {concept:28} {n_types:5d} types  {n_docs:7d} documents")
    typer.echo(f"wrote {out}")
    typer.echo("\nnext: `acr site-mapping diff` before any spec starts using it")


# ------------------------------------------------------------------------------- review
@site_mapping_app.command("review")
def review_cmd(
    mapping_path: str = typer.Option(..., "--mapping", help="mapping JSON from `build`"),
    out: str = typer.Option("", help="write a Markdown review table here"),
    top: int = typer.Option(60, help="rows to print"),
):
    """The table a registrar signs off, highest-impact rows first."""
    mapping = SiteMapping.from_dict(json.loads(Path(mapping_path).read_text()))
    rows = mapping.review_table()
    unmapped = [r for r in rows if r["concept"] == UNMAPPED]

    typer.echo(f"{mapping.n_types} type names, model={mapping.model}, "
               f"mapping_hash={mapping.mapping_hash}")
    typer.echo(f"{len(unmapped)} unplaced, carrying {sum(r['n_documents'] for r in unmapped)} "
               f"documents\n")
    for r in rows[:top]:
        typer.echo(f"  {r['n_documents']:7d}  {r['concept']:24} {r['doc_type']:44} {r['why']}")

    if out:
        preamble = ("Ordered so the rows deciding the most documents come first. Unplaced rows "
                    "come before those: a type name carrying 1,285 documents that no concept "
                    "describes is the most important row in the table.")
        lines = [f"# Site Mapping review — {mapping.corpus_id}", "",
                 f"- model: `{mapping.model}`", f"- built_at: {mapping.built_at}",
                 f"- mapping_hash: `{mapping.mapping_hash}`",
                 f"- concepts_hash: `{mapping.bound_concepts_hash}`",
                 f"- {mapping.n_types} type names, {len(unmapped)} unplaced", "",
                 preamble, "",
                 "| documents | concept | local type name | why |", "|---:|---|---|---|"]
        lines += [f"| {r['n_documents']} | `{r['concept']}` | `{r['doc_type']}` | {r['why']} |"
                  for r in rows]
        Path(out).write_text("\n".join(lines) + "\n")
        typer.echo(f"\nwrote {out}")


# ------------------------------------------------------------------------------- diff
@site_mapping_app.command("diff")
def diff_cmd(
    mapping_path: str = typer.Option(..., "--mapping", help="mapping JSON from `build`"),
    spec_id: str = typer.Option(..., "--spec", help="spec to compare against"),
    corpus: str = typer.Option(..., help="corpus root, for document counts"),
    spec_dir: str = typer.Option("assets/specs", help="directory of spec YAML"),
    out: str = typer.Option("", help="write the full diff here as JSON"),
    legacy_from: str = typer.Option("", help="spec YAML holding the retired doc_type_matches "
                                            "lists, if the spec has already been migrated"),
):
    """What the substring expression said, against what the mapping says.

    Run this BEFORE a spec starts selecting documents through a mapping. It reports every
    type name whose stratum changes and how many documents ride on it, so the decision to
    adopt is made against a list rather than against a claim.
    """
    import yaml

    _, strata = _strata_for(spec_dir, spec_id)
    mapping = SiteMapping.from_dict(json.loads(Path(mapping_path).read_text()))
    counts = Corpus(Path(corpus)).doc_type_counts()

    # The retired expression, read from wherever it still lives. After migration the spec no
    # longer holds it, so `--legacy-from` points at the pre-migration YAML: the comparison is
    # against what was ACTUALLY running, not against a reconstruction of it.
    src = Path(legacy_from) if legacy_from else Path(spec_dir) / f"{spec_id}.yaml"
    raw = yaml.safe_load(src.read_text()) if src.is_file() else {}
    legacy: list[tuple[str, list[str]]] = []
    fn = ((raw.get("proof_obligation") or {}).get("for_negative") or {})
    for st in (fn.get("strata") or []):
        pats = ((st.get("match") or {}).get("doc_type_matches") or [])
        if pats:
            legacy.append((st["name"], [p.lower() for p in pats]))
    rest_name = next((s.name for s in strata if s.rest), "rest")

    def legacy_stratum(doc_type: str) -> str:
        for name, pats in legacy:
            if any(p in doc_type.lower() for p in pats):
                return name
        return rest_name

    def mapped_stratum(doc_type: str) -> str:
        c = mapping.concept_for(doc_type)
        if c is None or c == UNMAPPED:
            return rest_name
        return next((s.name for s in strata if s.concept_name == c), rest_name)

    if not legacy:
        typer.echo(f"no doc_type_matches found in {src}; nothing to diff against")
        raise typer.Exit(0)

    changes = []
    for doc_type, n in counts.items():
        was, now = legacy_stratum(doc_type), mapped_stratum(doc_type)
        if was != now:
            changes.append({"doc_type": doc_type, "n_documents": n, "was": was, "now": now,
                            "why": (mapping.assignments.get(doc_type).why
                                    if doc_type in mapping.assignments else "")})
    changes.sort(key=lambda r: -r["n_documents"])

    moved_docs = sum(r["n_documents"] for r in changes)
    typer.echo(f"{len(changes)} of {len(counts)} type names change stratum, "
               f"moving {moved_docs} of {sum(counts.values())} documents\n")
    for r in changes[:60]:
        typer.echo(f"  {r['n_documents']:7d}  {r['was']:22} -> {r['now']:22} {r['doc_type']}")

    if out:
        _write(Path(out), {"spec_id": spec_id, "mapping_hash": mapping.mapping_hash,
                           "legacy_source": str(src), "n_types": len(counts),
                           "n_types_changed": len(changes), "n_documents_moved": moved_docs,
                           "changes": changes})
        typer.echo(f"\nwrote {out}")
