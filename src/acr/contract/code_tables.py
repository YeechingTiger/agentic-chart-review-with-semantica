"""Value-domain code tables: the fact table for one code system, its axes declared by the asset
rather than hard-coded in this module.

THIS MODULE REPLACES `icdo3.py`, AND NOT FOR STYLE
--------------------------------------------------
`icdo3.load_table` required every table to carry the three keys `topography` / `morphology` /
`behavior`, `prompt_block` hard-coded those three words as its section headings, `_TOPO = C\\d{3}`
and `_MORPH = \\d{4}` were module constants, and `check_codes(site, histology, behavior)` wrote
three cancer field names into the parameter list. So:

  * a LOINC lipid panel or an RxNorm drug class **could not be expressed** — `load_table` raised,
    and `spec.load_spec` is fail-closed, so the whole spec load failed;
  * three pieces of framework code — `spec`, `agent`, `run_manifest` — imported a module belonging
    to one use case, and those were all three of the inverted dependencies registered in
    `tests/test_layering.py`.

Cancer registry abstraction is **one** use case of this framework. So the axis names, the section
headings, the code-shape regexes and the notation-folding rules all move down into
`assets/codes/*.yaml`, and this module only knows that "a table has some ordered axes, and each
axis has codes and names".

THE THREE THINGS THAT SURVIVED, AND NOT ONE OF THEM IS A GATE
-------------------------------------------------------------
  1. `prompt_block()` renders the value domain into the system prompt, so the model codes into a
     table it can see rather than into one it half remembers. Two real failures prompted it: one
     run used `7205` as a morphology code (there is no 7205 in ICD-O-3), and one wrote "C341 is the
     right middle lobe" and coded on that basis (C341 is the **upper** lobe).
  2. `check_values()` returns typed problems for the evaluation plane to **count**, never to reject
     with.
  3. `normalize()` folds notation differences. This is 4 of the 6 useful firings of the old
     `field_format` check — most of what that check did was manufacture a round trip it then
     solved itself.

REPORTABILITY / ADMISSIBILITY is domain policy, not this module's authority. The `admissible: false`
in an axis and the table-level `exclusions` carry a casefinding manual's rulings, so every problem
is advisory and every table's `source_authority` asks for a human to check it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..core.repo_paths import asset_dir

CODES_DIR = asset_dir("assets/codes")

#: Problem kinds. Names rather than a boolean, because "this is not a code", "this code belongs to
#: a different scope" and "this code is not admissible" are three different findings, and a caller
#: that cannot tell them apart will report the wrong one.
MALFORMED = "MALFORMED"
NOT_IN_TABLE = "NOT_IN_TABLE"
OUT_OF_TABLE_SCOPE = "OUT_OF_TABLE_SCOPE"
NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
EXCLUDED_BY_SPEC = "EXCLUDED_BY_SPEC"


class CodeTableError(ValueError):
    """A code table is missing, or is not a shape this module can read."""


@dataclass(frozen=True)
class CodeProblem:
    kind: str
    field: str
    value: str
    message: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "field": self.field, "value": self.value,
                "message": self.message}


@dataclass(frozen=True)
class CodeAxis:
    """One axis: a set of codes, plus the asset-declared metadata needed to render and check it.

    `name` is the axis name (callers pass values keyed by it) and `field` is the name of the spec
    field this axis codes (problems are reported against it). The two are kept apart because one
    axis can land on a different field name in a different spec.
    """

    name: str
    label: str
    field: str
    codes: dict[str, dict]
    unspecified: tuple[str, ...] = ()
    code_shape: str = ""
    shape_description: str = ""
    scope_note: str = ""

    def name_of(self, code: str) -> str | None:
        e = self.codes.get(code)
        return e.get("name") if e else None

    def is_admissible(self, code: str) -> bool:
        """A code is admissible when the axis does not declare `admissible`. The default is True
        because the codes on almost every axis are usable values, and defaulting to False would make
        a table that never writes the key inadmissible as a whole, with nobody able to see why."""
        e = self.codes.get(code)
        return True if e is None else bool(e.get("admissible", True))


@dataclass(frozen=True)
class CodeTable:
    table_id: str
    version: str
    scope: str
    axes: dict[str, CodeAxis]
    exclusions: tuple[dict, ...] = ()
    guidance: tuple[str, ...] = ()
    excluded_by_spec: dict[str, dict] = field(default_factory=dict)
    excluded_by_spec_axis: str = ""
    warnings: tuple[str, ...] = ()
    _norm: dict = field(default_factory=dict)

    def normalize(self, raw: str) -> str:
        """Fold one code by the notation rules **this table** declares.

        The rules travel with the table instead of living in the module: ICD-O-3 writes `C18.7` and
        `8140/3`, whereas the hyphen inside a LOINC code is part of the code. A function with
        `[.\\s]` and `split_on='/'` hard-coded into it corrupts another system's codes.
        """
        s = str(raw or "")
        pattern = self._norm.get("strip_pattern")
        if pattern:
            s = re.sub(pattern, "", s)
        if self._norm.get("uppercase", True):
            s = s.upper()
        split_on = self._norm.get("split_on")
        if split_on and split_on in s:
            s = s.split(split_on, 1)[0]
        return s

    def trailing_part(self, raw: str) -> str | None:
        """The first character of what `split_on` dropped, for example the `3` of `8140/3`.

        Why it exists: the behaviour digit is a **field of its own** in STORE, so it must not be
        silently merged into the morphology code, but a caller still needs to be able to get at it.
        A table that declares no `split_on` always returns None.
        """
        split_on = self._norm.get("split_on")
        if not split_on:
            return None
        s = str(raw or "")
        pattern = self._norm.get("strip_pattern")
        if pattern:
            s = re.sub(pattern, "", s)
        if split_on not in s:
            return None
        tail = s.split(split_on, 1)[1]
        return tail[:1] or None

    def exclusion_term(self, axis_values: dict[str, str]) -> str | None:
        """The term of the `exclusions` row whose every axis value matches these axis values."""
        for row in self.exclusions:
            match = row.get("axis_values") or {}
            if not match:
                continue
            if all(self.normalize(str(axis_values.get(a, ""))) == self.normalize(str(v))
                   for a, v in match.items()):
                return str(row.get("term", ""))
        return None


def _axis(name: str, d: dict, norm: dict) -> CodeAxis:
    codes_raw = d.get("codes")
    if not isinstance(codes_raw, dict) or not codes_raw:
        raise CodeTableError(f"axis {name!r} has no `codes` mapping")

    def fold(raw: str) -> str:
        s = str(raw or "")
        if norm.get("strip_pattern"):
            s = re.sub(norm["strip_pattern"], "", s)
        return s.upper() if norm.get("uppercase", True) else s

    return CodeAxis(
        name=name,
        label=str(d.get("label") or name.upper()),
        field=str(d.get("field") or name),
        codes={fold(k): (v or {}) for k, v in codes_raw.items()},
        unspecified=tuple(fold(c) for c in (d.get("unspecified") or [])),
        code_shape=str(d.get("code_shape") or ""),
        shape_description=str(d.get("shape_description") or ""),
        scope_note=str(d.get("scope_note") or ""),
    )


@lru_cache(maxsize=8)
def load_table(name: str, codes_dir: str | None = None) -> CodeTable:
    """One code table, by name.

    There is no default table name. The old `load_table(name="icdo3_lung")` made "forgot to declare
    a value domain" and "declared lung" look the same, and loading the wrong table makes every case
    look like a wrong answer. The name has to be said out loud by the spec.
    """
    root = Path(codes_dir) if codes_dir else CODES_DIR
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise CodeTableError(
            f"no code table {name!r} at {path}; available: "
            f"{sorted(p.stem for p in root.glob('*.yaml'))}")
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "table_id" not in d:
        raise CodeTableError(f"{path} has no 'table_id'")
    axes_raw = d.get("axes")
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise CodeTableError(
            f"{path} declares no axes. An empty value domain renders an empty block into the "
            f"prompt, and the run then looks exactly like one that was given the codes — the "
            f"loader must fail rather than silently claim that guidance was supplied.")
    norm = dict(d.get("normalization") or {})
    return CodeTable(
        table_id=str(d["table_id"]),
        version=str(d.get("table_version", "0")),
        scope=str((d.get("source_authority") or {}).get("scope", "")),
        axes={str(k): _axis(str(k), v or {}, norm) for k, v in axes_raw.items()},
        exclusions=tuple(d.get("exclusions") or []),
        guidance=tuple(str(s) for s in (d.get("guidance") or [])),
        excluded_by_spec={str(k): (v or {})
                          for k, v in (d.get("excluded_by_spec") or {}).items()},
        excluded_by_spec_axis=str(d.get("excluded_by_spec_axis") or ""),
        warnings=tuple(str(s) for s in (d.get("warnings") or [])),
        _norm=norm,
    )


def check_values(values: dict[str, str | None], *, table: CodeTable) -> list[CodeProblem]:
    """The factual problems in a set of values given by axis name. ADVISORY — for counting, never
    for rejecting.

    Empty values are skipped: an empty value is a matter of abstention, and this module has no
    opinion on whether abstaining was right.

    A misspelled axis name raises rather than being ignored. Ignoring it silently amounts to
    reporting "the check passed" without having checked a single thing — the kind of check that
    cannot fail that this repository keeps naming.
    """
    out: list[CodeProblem] = []
    unknown = [a for a in values if a not in table.axes]
    if unknown:
        raise CodeTableError(
            f"{table.table_id} has no axis {unknown!r}; declared axes are "
            f"{sorted(table.axes)}")
    for axis_name, axis in table.axes.items():          # the table's order, not the caller's
        if axis_name not in values:
            continue
        raw = values[axis_name]
        if raw is None or not str(raw).strip():
            continue
        code = table.normalize(raw)
        if axis.code_shape and not re.fullmatch(axis.code_shape, code):
            out.append(CodeProblem(
                MALFORMED, axis.field, str(raw),
                f"{raw!r} is not shaped like a {axis.name} code in {table.table_id}"
                + (f" ({axis.shape_description})" if axis.shape_description else "")))
            continue
        if code in table.excluded_by_spec and (
                not table.excluded_by_spec_axis
                or table.excluded_by_spec_axis == axis_name):
            e = table.excluded_by_spec[code]
            out.append(CodeProblem(
                EXCLUDED_BY_SPEC, axis.field, code,
                f"{code} is {e.get('name', '')} — {e.get('why', '')}. The spec puts this outside "
                f"the variable, so the honest answer is SPEC_INSUFFICIENT rather than a coded "
                f"value. This is a scope boundary, not a coding error."))
            continue
        if code not in axis.codes:
            kind = OUT_OF_TABLE_SCOPE if axis.scope_note else NOT_IN_TABLE
            note = (f" {axis.scope_note}" if axis.scope_note else "")
            out.append(CodeProblem(
                kind, axis.field, code,
                f"{code} is not a {axis.name} value in {table.table_id}"
                + (f" ({table.scope})" if table.scope else "") + "." + note))
            continue
        if not axis.is_admissible(code):
            term = table.exclusion_term({a: values.get(a) or "" for a in table.axes})
            out.append(CodeProblem(
                NOT_ADMISSIBLE, axis.field, code,
                f"{axis.name} {code} ({axis.name_of(code) or ''}) is not an admissible value"
                + (f"; this combination is {term}" if term else "") + "."))
    return out


def prompt_block(table: CodeTable, *, max_terms: int = 0) -> str:
    """The value domain, for the system prompt. Renders the whole table by default.

    Rendered rather than summarised: a model shown only 12 of 40 codes will code into those 12.
    `max_terms` is for a caller that is measuring prompt size, not a default.

    Every line of text comes from the asset — the headings from each axis's `label`, the warnings
    from the table's `warnings`. The old version wrote "MORPHOLOGY (four digits; behaviour is a
    separate field)" and "if a diagnosis has no ICD-O-3 morphology then the finding is not a
    reportable neoplasm" into Python, and both of those sentences are nonsense over a lipid panel.
    """
    L = [f"VALUE DOMAIN — {table.table_id} v{table.version}"
         + (f" ({table.scope})" if table.scope else "")]
    if table.warnings:
        L += [""] + list(table.warnings)
    for axis in table.axes.values():
        L += ["", axis.label]
        items = list(axis.codes.items())
        if max_terms:
            items = items[:max_terms]
        for code, e in items:
            line = f"  {code}  {e.get('name', '')}"
            if e.get("aliases"):
                line += f"   [{', '.join(str(a) for a in e['aliases'])}]"
            if code in axis.unspecified:
                line += "   (unspecified — asserts the finer detail is not documented)"
            if not bool(e.get("admissible", True)):
                line += "   — NOT admissible"
            L.append(line)
    if table.exclusions:
        L += ["", "VALUES THAT ARE NOT ADMISSIBLE ANSWERS"]
        for r in table.exclusions:
            av = r.get("axis_values") or {}
            shown = "/".join(str(v) for v in av.values()) if av else "no code exists"
            L.append(f"  {r.get('term', '')}  ({shown}) — {r.get('why', '')}")
    if table.guidance:
        L += ["", "CODING SAFEGUARDS"] + [f"  - {s}" for s in table.guidance]
    return "\n".join(L)


def code_domain_block(spec) -> str:
    """The value-domain block for a spec that declares one; a spec that declares none gets "".

    The seam between the Task Contract and the prompt. The spec says **which** table its values are
    coded into (`value_domain: icdo3_lung`), because that is part of what the answer means; this
    renders it. A spec that declares none gets nothing — date and class-of-case variables have no
    code-table value domain, and a wall of morphology codes stuffed at them is only ignored.

    `load_spec` has already refused a table that does not exist, so a name that reaches here always
    resolves. The `try` is there for specs a test or an ablation builds in memory: there is no such
    guarantee, and a missing table should not bring down a run that never needed it.
    """
    name = str(getattr(spec, "value_domain", "") or "").strip()
    if not name:
        return ""
    try:
        return prompt_block(load_table(name))
    except CodeTableError:
        return ""


def table_manifest(spec) -> dict | None:
    """The identity of the value domain a run was shown, or None (it declared none).

    Content-hashed, for the same reason as `skills_manifest`: these tables are YAML meant to be
    edited by hand — the `what_a_human_must_check` in every table is an invitation to exactly that —
    so `table_version` on its own lets an edited table pass for the one the previous run used. The
    1,788-row validation that added eleven morphology codes to the lung table was an edit of that
    kind, and the manifests written before it and after it must never be taken as comparable.
    """
    import hashlib
    name = str(getattr(spec, "value_domain", "") or "").strip()
    if not name:
        return None
    try:
        t = load_table(name)
    except CodeTableError:
        return {"declared": name, "loaded": False}
    path = CODES_DIR / f"{name}.yaml"
    return {
        "declared": name, "loaded": True,
        "table_id": t.table_id, "table_version": t.version,
        "axes": {ax: len(a.codes) for ax, a in t.axes.items()},
        "n_excluded_by_spec": len(t.excluded_by_spec),
        "content_hash": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "origin": "model_recalled",
        "signed_off": False,
    }
