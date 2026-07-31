"""Local note-type name to portable document concept. THE MODEL READS THE TYPE LIST.

WHY THIS MODULE EXISTS
----------------------
A stratum used to select its documents with a spec line that read

    match: {doc_type_matches: ["Pathology", "Cytology"]}

evaluated case-insensitively, as a SUBSTRING, in `coverage.StratumSpec.matches`. Measured on
this corpus -- 1,516 distinct document-type names over 1,788 patients -- that expression:

  MISSES   Non-Gyn-Cyto-FNA (1,285 documents), FN-Aspirate-Report (881),
           SURG-PATH-RESULT (231), Microscopic-Observation-ID-Cyto-Stain (31),
           Fine-Needle-Aspiration (5). `Cyto-FNA` does not contain `Cytology`;
           `SURG-PATH-RESULT` does not contain `Pathology`.
  MATCHES  Speech-Language-Pathology-Note.

107 of the 219 patients whose `can_establish` count is zero in fact hold one of the missed
reports -- 6.0% of the cohort, told by the architecture that no document in the chart could
establish histology while an FNA diagnosis sat in it. Under `proof_obligation.for_negative`
that instruction is followed correctly and the run abstains.

The planning ablation of 2026-07-29 contains both halves of the proof:

  CASE001  no type name matched, so the cytology FNA in the chart was filed under a stratum
           named `cannot_establish`; the run returned EVIDENCE_INSUFFICIENT for histology and
           behaviour. The registry coded 8070.
  CASE006  the same chart state -- no matching type name -- but the run happened to open its
           FN-Aspirate-Report anyway, cited it as a witness, and returned the registry answer
           exactly. The document was answer-bearing; only the matcher disagreed.

Six of the eleven field misses in that arm are this one defect. None of the eleven was a
document the search failed to find.

WHY IT IS NOT A LONGER LIST OF SUBSTRINGS
-----------------------------------------
Deciding whether a free-text local type name denotes a pathologist's diagnosis is a semantic
judgement about clinical documents, and every attempt in this tree to spell it as a string
operation has produced a longer list of strings. `STORE.700_880.stage.yaml` enumerates 24 of
them by hand, above a comment that explains `Pathology` was the wrong instrument -- the same
conclusion, reached a second time, and answered with more of the same instrument.

So the concept definitions travel as prose, the model reads the corpus's type list, and it
assigns each name to a concept. That is what a Site Mapping is for:
`docs/CHART_REVIEW_KNOWLEDGE_AND_SEARCH_LAYERS.md` gives it exactly this scope -- "local
note-type name to portable document concept" -- and forbids the Task Contract from owning
"raw local note types". `doc_type_matches` was that violation, written in YAML.

WHY THE RESULT IS FROZEN INSTEAD OF RE-DERIVED PER RUN
-----------------------------------------------------
Two reasons, and the second is the load-bearing one.

1. Type names are a fact about the CORPUS, not about a patient. 1,516 names serve all 1,788
   charts, so this is one classification pass whose result every run reuses -- not a model
   call per run per patient.

2. A stratum assignment is AN INPUT TO THE COVERAGE GATE. `coverage.evaluate_gate` counts
   strata, and `judge.py` states the rule this module has to obey: a model may not re-decide
   something a deterministic procedure decides. Re-partitioning per run would make
   `exclusion_validated` a different question on every run, would leave two runs of the same
   chart incomparable, and would leave a registrar nothing to review. The gate stays
   deterministic; what the model supplies is the vocabulary the gate counts over, once,
   written down, hashed, and reviewable as a table.

`concepts_hash` binds a mapping to the concept definitions it was built against. Edit a
`means:` and the mapping stops applying rather than silently answering the question it was
built for. Fail-closed is the point: a stale mapping that still loads is a stratifier nobody
knows is stale.

NO PATIENT TEXT ENTERS THIS MODULE
----------------------------------
The builder is given type NAMES and corpus COUNTS. It is never given a document, a note id,
a date, or a patient id. That is a boundary property, asserted in
`tests/test_site_mapping.py`, not a habit: it is what makes the mapping shareable, cacheable
across patients, and safe to hand a registrar in a review document.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: A type name the model declined to place. NOT an error and NOT a synonym for "irrelevant":
#: it is "no concept in this vocabulary describes this", which is the honest answer for
#: `Unknown-Battery` and for a name that is genuinely ambiguous. Callers decide what to do
#: with it -- `coverage.assign_strata` routes it to the spec's declared `rest` stratum, so an
#: unplaced name defaults to whatever the spec author chose as the safe default rather than
#: to silence.
UNMAPPED = "UNMAPPED"

#: Domain separator for mapping hashes. Public so a reader can tell a hash of this shape from
#: some other sha256 in the same manifest.
HASH_DOMAIN = "acr.contract.site_mapping/1"


class SiteMappingError(ValueError):
    """A mapping is absent, stale, or does not cover what a stratum needs."""


# ------------------------------------------------------------------ concept vocabulary
@dataclass(frozen=True)
class Concept:
    """One portable document concept, as prose the model can act on.

    `means` is the whole interface. It replaces a substring list, so it has to say what a
    clinician would say -- what the document IS and who wrote it -- and not name local type
    strings, which is the thing this module exists to stop the Task Contract from doing.
    """

    name: str
    means: str

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise SiteMappingError("a concept needs a name")
        if not str(self.means).strip():
            raise SiteMappingError(
                f"concept {self.name!r} has no `means:` prose. A concept with no definition "
                f"is a substring list with extra steps: the model has nothing to classify "
                f"against, and a registrar has nothing to review."
            )


def concepts_hash(concepts: Sequence[Concept]) -> str:
    """Hash of the concept vocabulary a mapping was built against.

    Sorted by name, so declaration order in the spec cannot change the hash -- reordering
    strata is not a semantic change and must not invalidate a mapping. Editing a `means:`
    IS a semantic change and must.
    """
    payload = json.dumps(
        [[c.name, " ".join(str(c.means).split())] for c in sorted(concepts, key=lambda c: c.name)],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(f"{HASH_DOMAIN}|concepts|{payload}".encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the mapping itself
@dataclass(frozen=True)
class TypeAssignment:
    """One local type name, the concept it denotes, and why.

    `why` is not decoration. It is the only thing that makes a 1,516-row table reviewable: a
    registrar reading `SURG-PATH-RESULT -> definitive_pathology` needs to see the model's
    reason to catch the row where the reason is wrong but the label happens to look right.
    """

    doc_type: str
    concept: str
    why: str = ""
    n_documents: int = 0

    def to_dict(self) -> dict:
        return {"doc_type": self.doc_type, "concept": self.concept,
                "why": self.why, "n_documents": self.n_documents}


@dataclass(frozen=True)
class SiteMapping:
    """A frozen, hashed local-type-name to document-concept table."""

    corpus_id: str
    concepts: tuple[Concept, ...]
    bound_concepts_hash: str
    assignments: Mapping[str, TypeAssignment]
    model: str
    built_at: str
    #: Type names the builder was asked about and the model placed at UNMAPPED, kept
    #: separately from `assignments` so `review_table` can put them first: an unplaced name
    #: with 1,285 documents behind it is the most important row in the table.
    provenance: str = "model_assigned"

    @property
    def n_types(self) -> int:
        return len(self.assignments)

    @property
    def mapping_hash(self) -> str:
        payload = json.dumps(
            {"corpus": self.corpus_id, "concepts": self.bound_concepts_hash,
             "assign": sorted((a.doc_type, a.concept) for a in self.assignments.values())},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{HASH_DOMAIN}|mapping|{payload}".encode()).hexdigest()[:16]

    # -- lookup ---------------------------------------------------------------
    def concept_for(self, doc_type: str) -> str | None:
        """The concept for a local type name, or None if this mapping never saw the name.

        None and UNMAPPED are different answers and callers must not collapse them. None is
        "this mapping does not cover that name" -- a corpus that grew a new document type
        since the mapping was built, which is a reason to rebuild. UNMAPPED is "the model
        looked at it and no concept fits", which is a decision.
        """
        a = self.assignments.get(doc_type)
        return a.concept if a is not None else None

    def types_for(self, concept: str) -> list[str]:
        return sorted(a.doc_type for a in self.assignments.values() if a.concept == concept)

    def unmapped_types(self) -> list[str]:
        return self.types_for(UNMAPPED)

    def coverage_of(self, doc_types: Iterable[str]) -> tuple[list[str], list[str]]:
        """(known, unknown) for an actual chart's type names.

        The second list is the one that matters at run time: it is exactly the documents this
        mapping cannot speak for, and a gate that counts strata over them is counting
        something it has not classified.
        """
        known, unknown = [], []
        for t in sorted(set(doc_types)):
            (known if t in self.assignments else unknown).append(t)
        return known, unknown

    # -- staleness ------------------------------------------------------------
    def binds(self, concepts: Sequence[Concept]) -> bool:
        return self.bound_concepts_hash == concepts_hash(concepts)

    def require_binds(self, concepts: Sequence[Concept]) -> None:
        if not self.binds(concepts):
            raise SiteMappingError(
                f"site mapping {self.mapping_hash} was built against concept vocabulary "
                f"{self.bound_concepts_hash}, and the spec now declares "
                f"{concepts_hash(concepts)}. A `means:` changed, so every assignment in the "
                f"mapping answers a question that is no longer being asked. Rebuild it; do "
                f"not reuse it. (Reordering strata does not do this -- the hash is over "
                f"name/means sorted by name.)"
            )

    # -- review ---------------------------------------------------------------
    def review_table(self) -> list[dict]:
        """Rows for a human, ordered by how much of the corpus each decides.

        UNMAPPED first, then by document count. A registrar with an hour should spend it on
        the rows that move the most documents, and the rows nobody could place.
        """
        rows = [a.to_dict() for a in self.assignments.values()]
        rows.sort(key=lambda r: (r["concept"] != UNMAPPED, -r["n_documents"], r["doc_type"]))
        return rows

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": HASH_DOMAIN,
            "corpus_id": self.corpus_id,
            "model": self.model,
            "built_at": self.built_at,
            "provenance": self.provenance,
            "concepts": [{"name": c.name, "means": c.means} for c in self.concepts],
            "concepts_hash": self.bound_concepts_hash,
            "mapping_hash": self.mapping_hash,
            "n_types": self.n_types,
            "n_unmapped": len(self.unmapped_types()),
            "assignments": [a.to_dict() for a in
                            sorted(self.assignments.values(), key=lambda a: a.doc_type)],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SiteMapping:
        concepts = tuple(Concept(name=c["name"], means=c["means"])
                         for c in (d.get("concepts") or []))
        assigns = {a["doc_type"]: TypeAssignment(
            doc_type=a["doc_type"], concept=a["concept"], why=a.get("why", ""),
            n_documents=int(a.get("n_documents", 0))) for a in (d.get("assignments") or [])}
        m = cls(corpus_id=d["corpus_id"], concepts=concepts,
                bound_concepts_hash=d["concepts_hash"], assignments=assigns,
                model=d.get("model", "unknown"), built_at=d.get("built_at", ""),
                provenance=d.get("provenance", "model_assigned"))
        stored = d.get("mapping_hash")
        if stored and stored != m.mapping_hash:
            raise SiteMappingError(
                f"site mapping file records mapping_hash={stored} but its own contents hash "
                f"to {m.mapping_hash}. The file was edited after it was written. A mapping is "
                f"a measurement, not a config file: rebuild it rather than hand-correcting a "
                f"row, or the hash stops meaning anything and the provenance is a lie."
            )
        return m


# ------------------------------------------------------------------ building it
#: Kept small enough that the model classifies a batch attentively and a refusal costs one
#: batch rather than the corpus. 1,516 names at 120 is 13 calls.
DEFAULT_BATCH = 120

_INSTRUCTION = """\
You are building a Site Mapping for a clinical chart-review system: you assign each LOCAL
DOCUMENT TYPE NAME from one hospital corpus to one PORTABLE DOCUMENT CONCEPT.

The concepts, and what each one means:

{concepts}

Rules:
- Assign every name in the list below to exactly one concept `name`, or to "{unmapped}" if no
  concept genuinely describes it.
- Judge what the DOCUMENT IS and WHO WROTE IT, not whether the name contains a word. A name
  can denote a pathologist's diagnosis without containing "pathology"; a name can contain
  "pathology" and be a speech-language therapy note.
- A procedure or imaging report that RECORDS how a specimen was obtained is not the same
  document as the report in which a pathologist states the diagnosis. Say which one the name
  denotes.
- "{unmapped}" is a real answer. Use it rather than forcing a name into the nearest concept.
- Do not invent, merge, split, or reword a name. Echo each one back exactly as given.

Return JSON only:
{{"assignments": [{{"doc_type": "<exactly as given>", "concept": "<a concept name or {unmapped}>",
                   "why": "<one clause: what this document is>"}}]}}
"""


def _concept_block(concepts: Sequence[Concept]) -> str:
    return "\n".join(f"- {c.name}: {' '.join(str(c.means).split())}"
                     for c in sorted(concepts, key=lambda c: c.name))


def _batch_prompt(concepts: Sequence[Concept], batch: Sequence[tuple[str, int]]) -> list[dict]:
    listing = "\n".join(f"- {name}  ({n} documents in the corpus)" for name, n in batch)
    return [
        {"role": "system", "content": _INSTRUCTION.format(
            concepts=_concept_block(concepts), unmapped=UNMAPPED)},
        {"role": "user", "content": f"Local document type names to assign:\n{listing}"},
    ]


def build_site_mapping(type_counts: Mapping[str, int], concepts: Sequence[Concept], llm, *,
                       corpus_id: str, built_at: str, batch_size: int = DEFAULT_BATCH,
                       model: str | None = None) -> SiteMapping:
    """Classify a corpus's document-type names against a concept vocabulary.

    `type_counts` is {local type name: how many documents carry it} and is the ONLY corpus
    fact that crosses into the prompt. No note id, no date, no patient id, no document text --
    see the module docstring; `tests/test_site_mapping.py` asserts it.

    `built_at` is passed in rather than read from the clock so the artifact is reproducible
    and so a caller cannot accidentally make two identical mappings hash differently.
    """
    if not concepts:
        raise SiteMappingError("cannot build a mapping against an empty concept vocabulary")
    names = sorted(type_counts)
    if not names:
        raise SiteMappingError(f"corpus {corpus_id!r} reports no document types")
    valid = {c.name for c in concepts} | {UNMAPPED}

    assignments: dict[str, TypeAssignment] = {}
    for start in range(0, len(names), batch_size):
        batch = [(n, int(type_counts[n])) for n in names[start:start + batch_size]]
        want = {n for n, _ in batch}
        out = llm.json_chat(_batch_prompt(concepts, batch),
                            schema_hint='{"assignments":[{"doc_type","concept","why"}]}')
        for row in (out.get("assignments") or []):
            name = str(row.get("doc_type", ""))
            if name not in want:
                # Not tolerated. A returned name that was not asked about means the model
                # rewrote a type string, and a rewritten name silently maps NOTHING at run
                # time: `concept_for` misses, the document falls to `rest`, and the mapping
                # looks complete. Refusing here is how that stays visible.
                raise SiteMappingError(
                    f"model returned document type {name!r}, which was not in the batch it "
                    f"was given. Type names are corpus identifiers and must be echoed "
                    f"exactly; a paraphrased name maps no document."
                )
            concept = str(row.get("concept", "")).strip() or UNMAPPED
            if concept not in valid:
                raise SiteMappingError(
                    f"model assigned {name!r} to concept {concept!r}, which the spec does not "
                    f"declare. Declared: {sorted(valid)}."
                )
            assignments[name] = TypeAssignment(
                doc_type=name, concept=concept, why=str(row.get("why", "")).strip(),
                n_documents=int(type_counts[name]))
        missing = want - set(assignments)
        if missing:
            # Fail closed rather than defaulting the remainder to UNMAPPED. A batch that came
            # back short is a truncated completion or a model that lost the tail of a long
            # list, and quietly filing 40 unclassified names under "no concept fits" would
            # reproduce the original defect with a new label on it.
            raise SiteMappingError(
                f"batch starting at {start} returned no assignment for {len(missing)} of "
                f"{len(want)} names, e.g. {sorted(missing)[:5]}. Rerun with a smaller "
                f"batch_size; do not default the remainder."
            )

    return SiteMapping(
        corpus_id=corpus_id, concepts=tuple(concepts),
        bound_concepts_hash=concepts_hash(concepts), assignments=assignments,
        model=model or getattr(getattr(llm, "cfg", None), "model", "unknown"),
        built_at=built_at)


# ------------------------------------------------------------------ spec -> vocabulary
def concepts_from_strata(strata: Sequence[Any]) -> list[Concept]:
    """The concept vocabulary a spec's strata declare.

    A `rest: true` stratum contributes NO concept: it is not something the model classifies
    into, it is the spec author's declared destination for whatever the mapping did not place.
    Offering it as a concept would let the model file documents directly into the fallback and
    make "the mapping could not place this" indistinguishable from "the model chose the
    catch-all".
    """
    out: list[Concept] = []
    for s in strata:
        if getattr(s, "rest", False):
            continue
        name = getattr(s, "concept", None) or getattr(s, "name", None)
        means = getattr(s, "means", "") or ""
        if not str(means).strip():
            raise SiteMappingError(
                f"stratum {name!r} declares no `means:`. Since `doc_type_matches` was "
                f"retired, `means:` is how a stratum says which documents it is about; "
                f"without it the stratum selects nothing and the gate counts an empty "
                f"stratum as a satisfied one."
            )
        out.append(Concept(name=str(name), means=str(means)))
    return out
