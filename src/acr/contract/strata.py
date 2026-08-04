"""Strata: the stratification a spec DECLARES, and which documents fall in each.

Derived from a spec's `proof_obligation.for_negative`, so it belongs to the contract layer and not
to runtime policy.

Why it moved out of `coverage.py`
---------------------------------
`assetdev` and `derive` need to stratify by spec, so they imported `coverage` — a module on the
runtime plane. The direction was not wrong, but it meant that changing one function signature in
`coverage` reached into the improvement plane, and these three symbols were never a policy about
"how to search": they are "which strata this spec declares, and which stratum a document falls in",
which is part of what the answer MEANS. `tests/test_layering.py` registers those two edges as
action B.

`coverage.py` still re-exports the three names, because `CoverageLedger`'s API takes them as
parameters — that is a natural re-export, not a compatibility shim.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..chartstore.corpus import DocMeta
from .site_mapping import SiteMapping, SiteMappingError


@dataclass
class StratumSpec:
    name: str
    policy: str
    doc_type_matches: list[str] = field(default_factory=list)
    rest: bool = False
    required_keywords: list[str] = field(default_factory=list)
    min_sample: int = 25
    min_sample_of_misses: int = 25
    max_tolerated_hits: int = 0
    partition_by: str | None = None
    surveillance_schedule: Any = None
    qualifying_doc_types: list[str] = field(default_factory=list)
    empty_window_policy: dict[str, str] = field(default_factory=dict)
    # Which of the spec's fields this stratum speaks to. Empty = all of them.
    #
    # A criterion can carry fields with DIFFERENT evidence rules. STORE.400_522_523 is the
    # case in point: histology and behaviour require pathology, but primary_site does not --
    # the spec says outright "Radiology can localise a mass; it cannot establish histology or
    # behaviour." One stratification serving all three fields therefore files every CT and
    # PET under a stratum literally named `cannot_establish`, and the agent reads that as
    # "these documents are useless".
    #
    # Observed: patient P03 was coded C349 (lung NOS) when "right upper lobe"
    # was documented across seven imaging and oncology note types. The pathology said only
    # "Right Lung", and the agent would not use the imaging that would have given it C341.
    # The architecture taught it that error; the spec text said the opposite.
    establishes: list[str] = field(default_factory=list)

    # -- what this stratum is ABOUT, in prose, for a Site Mapping to classify against ------
    #
    # `means` replaces `doc_type_matches`. The substring list it replaces was measured wrong
    # on this corpus: `["Pathology", "Cytology"]` matched Speech-Language-Pathology-Note and
    # missed Non-Gyn-Cyto-FNA (1,285 documents), FN-Aspirate-Report (881) and
    # SURG-PATH-RESULT (231), so 107 patients holding an FNA diagnosis were stratified as
    # holding nothing that could establish histology. See `acr.contract.site_mapping` for the
    # measurement and for the two ablation cases that prove both halves of it.
    #
    # `concept` is the portable name the mapping assigns to; it defaults to `name` so a
    # stratum only sets it when the local stratum name and the portable concept differ.
    concept: str | None = None
    means: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> StratumSpec:
        m = d.get("match") or {}
        return cls(
            name=d["name"], policy=d["policy"],
            concept=(d.get("concept") or None),
            means=str(d.get("means") or ""),
            doc_type_matches=list(m.get("doc_type_matches") or []),
            rest=bool(m.get("rest")),
            required_keywords=list(d.get("required_keywords") or []),
            min_sample=int(d.get("min_sample", 25)),
            min_sample_of_misses=int(d.get("min_sample_of_misses", 25)),
            max_tolerated_hits=int(d.get("max_tolerated_hits", 0)),
            partition_by=d.get("partition_by"),
            surveillance_schedule=d.get("surveillance_schedule"),
            qualifying_doc_types=list(d.get("qualifying_doc_types") or []),
            empty_window_policy=dict(d.get("empty_window_policy") or {}),
            establishes=list(d.get("establishes") or []),
        )

    @property
    def concept_name(self) -> str:
        """The concept a Site Mapping assigns documents to for this stratum."""
        return self.concept or self.name

    @property
    def is_mapped(self) -> bool:
        """True when this stratum selects documents through a Site Mapping.

        Declaring `means:` is the switch. A stratum that declares it has retired its substring
        list and cannot fall back to one: see `matches`.
        """
        return bool(str(self.means).strip())

    def matches(self, doc: DocMeta, mapping: SiteMapping | None = None) -> bool:
        """Does this stratum claim `doc`?

        Three paths, and the ordering is the whole safety property:

        `rest`          claims anything, as it always did. It is a declared destination, not a
                        match, so a mapping is irrelevant to it.

        `means:`        the Site Mapping decides, and NOTHING ELSE MAY. If the mapping has no
                        opinion about this type name the answer is False, so the document falls
                        through to the spec's `rest` stratum -- the author's declared default
                        for an unclassified document. It does NOT quietly fall back to
                        `doc_type_matches`: a substring list that runs only when the mapping
                        is absent is the same defect with a lower firing rate, and it would be
                        hardest to notice exactly when the mapping is broken.

        legacy          `doc_type_matches` as a case-insensitive substring, for the strata not
                        yet migrated. `speclint` reports every one of these as a placement
                        violation (raw local type names do not belong in a Task Contract) and
                        `acr.contract.site_mapping` records what the expression measured wrong here.
        """
        if self.rest:
            return True
        if self.is_mapped:
            if mapping is None:
                raise SiteMappingError(
                    f"stratum {self.name!r} selects documents through a Site Mapping "
                    f"(concept {self.concept_name!r}) and no mapping was supplied. Refusing "
                    f"rather than stratifying: with no mapping every document falls to the "
                    f"`rest` stratum, the gate counts an empty `{self.name}` as a satisfied "
                    f"one, and a run reports a coverage proof it never performed. Build one "
                    f"with `acr site-mapping build` and pass it to assign_strata."
                )
            return mapping.concept_for(doc.doc_type) == self.concept_name
        return any(pat.lower() in doc.doc_type.lower() for pat in self.doc_type_matches)

def assign_strata(docs: Sequence[DocMeta], specs: Sequence[StratumSpec],
                  mapping: SiteMapping | None = None) -> dict[str, list[DocMeta]]:
    """First match wins; the `rest: true` stratum sweeps up whatever is left.

    `mapping` is required as soon as any stratum declares `means:` -- `StratumSpec.matches`
    raises rather than guessing. Passing one to strata that declare no `means:` is harmless
    and changes nothing, so a caller may pass it unconditionally.
    """
    out: dict[str, list[DocMeta]] = {s.name: [] for s in specs}
    ordered = [s for s in specs if not s.rest] + [s for s in specs if s.rest]
    for d in docs:
        for s in ordered:
            if s.matches(d, mapping):
                out[s.name].append(d)
                break
    return out


def spec_declared_keywords(spec) -> list[str]:
    """Every retrieval term a spec declares, in declaration order, deduplicated and lowercased.

    THE ONE IMPLEMENTATION, and it lives HERE rather than beside its first caller because two work
    planes need it: `review` seeds a run's search list from it, and `improvement` prices every
    candidate term against it. `tests/test_layering.py` forbids `improvement -> review` — work
    planes may share only `core`/`contract` types — and the alternative to moving it down was four
    copies. Three of the four disagreed: `strata_from_spec` reads the two STRATUM locations and
    misses `proof_obligation.required_keywords`, so `assetdev`, `derive` and `build_termcache` each
    priced candidates against a shorter list than the runtime searched. `STORE.700_880` declares
    `tnm` there and in no stratum: runtime 11 terms, develop plane 10, and `certify` certified
    improvement over a configuration nobody deploys.

    Three locations, because a spec may declare terms at any of them:
      `proof_obligation.required_keywords`                     — scoped to no stratum
      `for_negative.strata[].required_keywords`
      `for_negative.claims[].strata[].required_keywords`
    """
    po = getattr(spec, "proof_obligation", None)
    kws: list[str] = list(getattr(po, "required_keywords", []) or []) if po else []
    fn = (getattr(po, "for_negative", {}) or {}) if po else {}
    for st in (fn.get("strata") or []):
        kws.extend(st.get("required_keywords") or [])
    for claim in (fn.get("claims") or []):
        for st in (claim.get("strata") or []):
            kws.extend(st.get("required_keywords") or [])
    out: list[str] = []
    for k in kws:
        k = str(k).strip().lower()
        if k and k not in out:
            out.append(k)
    return out


def strata_from_spec(spec) -> list[StratumSpec]:
    """Pull the stratum declarations out of a spec's proof_obligation.

    Returns [] when the spec declares no strata — which is what the unstratified baseline
    arm runs on, and which `to_dict()` labels `mode: unstratified` so the two arms are never
    confused in the trace.
    """
    fn = getattr(spec, "proof_obligation", None)
    fn = getattr(fn, "for_negative", {}) if fn else {}
    raw = list(fn.get("strata") or [])
    for claim in (fn.get("claims") or []):
        raw.extend(claim.get("strata") or [])
    return [StratumSpec.from_dict(s) for s in raw]

