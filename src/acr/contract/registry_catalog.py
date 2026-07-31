"""L0 RESOLVE — a user's variable names to the specs that produce them.

The layer exists because the unit a user asks for and the unit the system runs are not the
same. A user asks for `primary_site`; the system runs `STORE.400_522_523`, which answers
three fields in one agent pass. Resolving per variable instead of per spec would run that
agent three times over the same chart and could return three disagreeing answers for one
patient — the two-ledger failure `state.py` already had to remove once.

Three rules, and each is here because of a failure that has already happened in this repo:

  * Matching is EXACT, after whitespace/hyphen normalisation only. No substring, no prefix,
    no fuzzy match. Substring matching on a name the user typed is precisely the mechanism
    that filed `Fine-Needle-Report` outside `["Pathology", "Cytology"]` and swept
    `Speech-Language-Pathology-Note` in. `difflib` is used for the "did you mean" line in an
    error and never to resolve anything.
  * A name that reaches two different specs is an ERROR naming both, never first-wins.
    `assign_strata` may take first-match because the spec author declared the order; nothing
    declares an order over specs, so picking one would be the resolver inventing a policy.
  * An unrecognised name raises with the full known vocabulary attached. Silently dropping a
    requested variable produces a cohort extract that is short one column and says so
    nowhere, and the concordance layer downstream then reports NOT_ASSESSABLE for a variable
    the user believes was extracted.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .spec import ExtractionSpec, load_specs

#: How a requested name reached a spec. Recorded on every resolution so an extract manifest
#: shows whether the user named the field or was routed there by an alias.
MatchKind = str
BY_FIELD = "field"
BY_SPEC_ID = "spec_id"
BY_SPEC_STEM = "spec_stem"
BY_STORE_ITEM = "store_item"
BY_DECLARED_ALIAS = "declared_alias"

_STORE_ITEMS_RE = re.compile(r"^(?:store\.)?(\d+(?:_\d+)*)$", re.IGNORECASE)


class VariableResolutionError(ValueError):
    """Base for every way a requested variable fails to become a spec+field."""


class UnknownVariableError(VariableResolutionError):
    """No spec in the catalogue produces this. Carries the vocabulary that does exist."""

    def __init__(self, unknown: Sequence[str], known: Sequence[str],
                 suggestions: dict[str, list[str]] | None = None):
        self.unknown = list(unknown)
        self.known = list(known)
        self.suggestions = dict(suggestions or {})
        lines = [f"unknown variable(s): {', '.join(self.unknown)}"]
        for name in self.unknown:
            if self.suggestions.get(name):
                lines.append(f"  {name}: did you mean {', '.join(self.suggestions[name])}?")
        lines.append("known variables: " + (", ".join(self.known) or "(none)"))
        super().__init__("\n".join(lines))


class AmbiguousVariableError(VariableResolutionError):
    """One name, two specs. Naming both is the only safe answer."""

    def __init__(self, name: str, spec_ids: Sequence[str]):
        self.name = name
        self.spec_ids = sorted(spec_ids)
        super().__init__(
            f"variable {name!r} is produced by more than one spec: {', '.join(self.spec_ids)}. "
            f"Name the spec_id you want. Choosing one for you would silently bind a cohort "
            f"to a spec the request never mentioned."
        )


def normalise_name(raw: str) -> str:
    """Fold the ways a human writes one name. Not a fuzzy match — see the module docstring.

    `Primary Site`, `primary-site` and `PRIMARY_SITE` are the same token typed by three
    people. `primary` and `site` are not, and must not resolve.
    """
    return re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())


@dataclass(frozen=True)
class VariableEntry:
    """One extractable variable: a field of exactly one spec."""

    name: str
    spec_id: str
    spec_version: str
    spec_hash: str
    data_source: str
    type: str = "string"
    format: str | None = None
    allowable_values: tuple[str, ...] | None = None
    description: str = ""

    @property
    def from_notes(self) -> bool:
        """False means every run of this spec is forced to SPEC_INSUFFICIENT at finalize
        (`graph.py`, `data_source == "outside_notes"`). Callers warn; they do not refuse —
        the variable really is known, and the refusal is the run's to make, not the
        resolver's."""
        return self.data_source == "notes"

    def to_dict(self) -> dict:
        return {"name": self.name, "spec_id": self.spec_id, "spec_version": self.spec_version,
                "spec_hash": self.spec_hash, "data_source": self.data_source,
                "type": self.type, "format": self.format,
                "allowable_values": list(self.allowable_values) if self.allowable_values else None,
                "description": self.description}


@dataclass(frozen=True)
class ResolvedVariable:
    """What the user asked for, and the (spec, field) it became."""

    requested: str
    name: str
    spec_id: str
    matched_on: MatchKind
    data_source: str = "notes"

    def to_dict(self) -> dict:
        return {"requested": self.requested, "name": self.name, "spec_id": self.spec_id,
                "matched_on": self.matched_on, "data_source": self.data_source}


@dataclass(frozen=True)
class Resolution:
    """The resolved request, grouped the way work is actually done: by spec."""

    variables: tuple[ResolvedVariable, ...] = ()
    requested: tuple[str, ...] = ()

    @property
    def spec_ids(self) -> list[str]:
        """Deduplicated, in first-request order — this is the run list, one pass per spec."""
        return list(dict.fromkeys(v.spec_id for v in self.variables))

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(v.name for v in self.variables))

    def fields_for(self, spec_id: str) -> list[str]:
        return list(dict.fromkeys(v.name for v in self.variables if v.spec_id == spec_id))

    def not_from_notes(self) -> list[ResolvedVariable]:
        return [v for v in self.variables if v.data_source != "notes"]

    def to_dict(self) -> dict:
        return {"requested": list(self.requested),
                "variables": [v.to_dict() for v in self.variables],
                "spec_ids": self.spec_ids}


@dataclass
class VariableCatalog:
    """Every variable the shipped specs can produce, indexed by every name that reaches it."""

    specs: dict[str, ExtractionSpec] = field(default_factory=dict)
    directory: str = ""

    @classmethod
    def from_directory(cls, directory: str | Path = "assets/specs") -> "VariableCatalog":
        """Load `<directory>/*.yaml`, NON-recursively, exactly as `load_specs` does.

        The non-recursion is load-bearing, not an oversight inherited from `load_specs`:
        `assets/specs/ablation/STORE.400_522_523.unstratified.yaml` declares the same three field
        names (`primary_site`, `histology`, `behavior`) under a different spec_id. A
        recursive scan would make all three ambiguous and every ordinary request would fail.
        An ablation arm is a second copy of a variable on purpose; it is selected by path,
        never by name.
        """
        return cls(specs=load_specs(directory), directory=str(directory))

    # ------------------------------------------------------------------ vocabulary
    def entries(self) -> list[VariableEntry]:
        out: list[VariableEntry] = []
        for sid in sorted(self.specs):
            s = self.specs[sid]
            for f in s.fields:
                av = getattr(f, "allowable_values", None)
                out.append(VariableEntry(
                    name=f.name, spec_id=s.spec_id, spec_version=s.spec_version,
                    spec_hash=s.spec_hash, data_source=s.data_source,
                    type=getattr(f, "type", "string") or "string",
                    format=getattr(f, "format", None),
                    allowable_values=tuple(str(v) for v in av) if av else None,
                    description=getattr(f, "description", "") or ""))
        return out

    def known_names(self) -> list[str]:
        """The canonical vocabulary: field names only. This is what an error message shows —
        listing the aliases too would bury the names the rest of the system keys on."""
        return sorted({e.name for e in self.entries()})

    def known_aliases(self) -> dict[str, list[str]]:
        """alias -> the spec_ids it reaches. Every key is already normalised."""
        return {k: sorted({sid for sid, _ in v}) for k, v in self._index().items()}

    def _declared_aliases(self, s: ExtractionSpec, fname: str | None) -> list[str]:
        """Aliases a spec declares for itself.

        Nothing declares any today, and that is deliberate: inventing `site -> primary_site`
        in Python would put a naming judgement in the enforced layer, where a domain expert
        cannot read or approve it. The extension point is here so the judgement can be added
        where it belongs — the spec YAML — at the cost of a changed `spec_hash`, which is the
        correct price for changing what a label means.
        """
        if fname is None:
            raw = (s.model_extra or {}).get("aliases")
        else:
            fld = next((f for f in s.fields if f.name == fname), None)
            raw = (getattr(fld, "model_extra", None) or {}).get("aliases") if fld else None
        return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []

    def _index(self) -> dict[str, list[tuple[str, str | None]]]:
        """normalised alias -> [(spec_id, field_name or None)]. None means "the whole spec"."""
        idx: dict[str, list[tuple[str, str | None]]] = {}

        def add(key: str, spec_id: str, fname: str | None) -> None:
            k = normalise_name(key)
            if not k:
                return
            idx.setdefault(k, [])
            if (spec_id, fname) not in idx[k]:
                idx[k].append((spec_id, fname))

        for sid in sorted(self.specs):
            s = self.specs[sid]
            add(s.spec_id, sid, None)
            stem = s.spec_id.rsplit(".", 1)[-1]
            add(stem, sid, None)
            for part in s.spec_id.split("."):
                m = _STORE_ITEMS_RE.match(part)
                if m:
                    # `STORE.400_522_523` -> 400, 522, 523. A registrar names the item, and
                    # an item number reaches the spec that produces it — not a single field,
                    # because nothing in the spec says which field answers which item. The
                    # binding item->field is declared in the guideline, not here.
                    for item in m.group(1).split("_"):
                        add(item, sid, None)
                        add(f"store.{item}", sid, None)
            for a in self._declared_aliases(s, None):
                add(a, sid, None)
            for f in s.fields:
                add(f.name, sid, f.name)
                for a in self._declared_aliases(s, f.name):
                    add(a, sid, f.name)
        return idx

    def _kind(self, key: str, spec_id: str, fname: str | None) -> MatchKind:
        s = self.specs[spec_id]
        if fname is not None:
            return BY_FIELD if key == normalise_name(fname) else BY_DECLARED_ALIAS
        if key == normalise_name(s.spec_id):
            return BY_SPEC_ID
        if key == normalise_name(s.spec_id.rsplit(".", 1)[-1]):
            return BY_SPEC_STEM
        return BY_STORE_ITEM if _STORE_ITEMS_RE.match(key) else BY_DECLARED_ALIAS

    # ------------------------------------------------------------------ resolution
    def resolve(self, requested: Iterable[str] | str) -> Resolution:
        """Requested names -> (spec, field) pairs. Raises rather than dropping or guessing.

        Every unknown name in the request is collected before raising. Failing on the first
        one makes the user rerun a cohort extract once per typo.
        """
        names = _split(requested)
        if not names:
            raise VariableResolutionError(
                "no variables requested; pass --variables with a comma-separated list. "
                "Extracting nothing succeeds silently, which is worse than failing.")

        idx = self._index()
        unknown: list[str] = []
        out: list[ResolvedVariable] = []
        for raw in names:
            key = normalise_name(raw)
            targets = idx.get(key)
            if not targets:
                unknown.append(raw)
                continue
            spec_ids = {sid for sid, _ in targets}
            if len(spec_ids) > 1:
                raise AmbiguousVariableError(raw, spec_ids)
            sid = targets[0][0]
            s = self.specs[sid]
            # A whole-spec target (spec_id / stem / item) and a field target can coexist for
            # one key: `STORE.390.date_of_initial_diagnosis` has a field of the same name as
            # its own stem. Same spec, so it is not an ambiguity — and the FIELD wins,
            # because it is the more specific of the two readings. Letting the stem win
            # widened that request from one column to two (`month_day_imputed` came along
            # uninvited), which is a resolver quietly answering a question it was not asked.
            named = [f for _, f in targets if f]
            if named:
                fields, kind = named, self._kind(key, sid, named[0])
            else:
                fields, kind = [f.name for f in s.fields], self._kind(key, sid, None)
            for fname in fields:
                out.append(ResolvedVariable(raw, fname, sid, kind, s.data_source))

        if unknown:
            known = self.known_names()
            pool = sorted(set(known) | set(self._index()))
            sugg = {u: difflib.get_close_matches(normalise_name(u), pool, n=3, cutoff=0.6)
                    for u in unknown}
            raise UnknownVariableError(unknown, known, sugg)

        # Deduplicate on (name, spec_id): asking for `histology` and `STORE.522` is one
        # variable requested twice, not two columns. The first request keeps its `requested`
        # label so the manifest still shows how it was named.
        #
        # Order is request order, and within one requested name it is the spec's own field
        # order. Sorting alphabetically instead turned `primary_site, histology, behavior`
        # into `behavior, histology, primary_site` in every extract table — deterministic,
        # but it discards the ordering the spec author chose and the operator typed.
        seen: set[tuple[str, str]] = set()
        uniq = []
        for v in out:
            if (v.name, v.spec_id) not in seen:
                seen.add((v.name, v.spec_id))
                uniq.append(v)
        return Resolution(tuple(uniq), tuple(names))


def _split(requested: Iterable[str] | str) -> list[str]:
    if isinstance(requested, str):
        parts = re.split(r"[,\n]", requested)
    else:
        parts = [p for r in requested for p in re.split(r"[,\n]", str(r))]
    return [p.strip() for p in parts if p.strip()]


# ------------------------------------------------------------------ guideline bindings
def check_guideline_bindings(catalog: VariableCatalog, guideline: Any) -> list[str]:
    """Return violations; empty means every extraction-sourced input reaches a real field.

    Same contract as `validate_guideline` and `check_field_formats`: a list of strings, never
    an exception. This is the check for the failure the L4 build already hit once — a
    guideline naming `ajcc_pathologic_stage` while the spec's field is
    `pathologic_stage_group`. Nothing errors; the variable simply never arrives, every case
    comes back NOT_ASSESSABLE naming a variable the operator believes was requested, and the
    concordance denominator quietly goes to zero.
    """
    out: list[str] = []
    idx = catalog._index()
    for rec in getattr(guideline, "recommendations", ()):
        for d in getattr(rec, "required_inputs", ()):
            if not isinstance(d, dict) or d.get("source") != "extraction_spec":
                continue
            name = str(d.get("name") or "")
            p = f"recommendation {rec.id}: input {name!r}"
            targets = idx.get(normalise_name(name))
            if not targets:
                out.append(f"{p} is sourced from extraction_spec but no spec in "
                           f"{catalog.directory or 'the catalogue'} declares that field")
                continue
            declared = d.get("spec_id")
            reached = sorted({sid for sid, _ in targets})
            if declared and str(declared) not in reached:
                out.append(f"{p} declares spec_id {declared!r} but the name resolves to "
                           f"{', '.join(reached)}")
            if not any(f == name for _, f in targets):
                out.append(f"{p} resolves to {', '.join(reached)} but not to a field of that "
                           f"name, so `variables_from_answer` will never key on it")
    return out
