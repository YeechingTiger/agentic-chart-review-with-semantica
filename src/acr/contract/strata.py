"""Strata: the stratification a spec DECLARES, and which documents fall in each.

从 spec 的 `proof_obligation.for_negative` 推导出来的，所以属于合同层而不是运行时策略。

为什么它从 `coverage.py` 搬出来
------------------------------
`assetdev` 和 `derive` 需要按 spec 分层，于是它们 import 了 `coverage` —— 一个运行时平面的
模块。方向没错，但那意味着改 `coverage` 的一个函数签名会动到 improvement 平面，而这三个
符号从来不是"怎样搜索"的策略：它们是"这个 spec 声明了哪些层，一份文档属于哪一层"，也就是
答案含义的一部分。`tests/test_layering.py` 把这两条登记为动作 B。

`coverage.py` 仍然转出这三个名字，因为 `CoverageLedger` 的 API 就以它们为参数 —— 那是自然的
再导出，不是兼容层。
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

