"""ICD-O-3 code tables: the value domain, as a FACT TABLE rather than a word list.

WHY A TABLE AND NOT A CHECK
---------------------------
Two failures on 2026-07-30 that nothing in this tree could catch, and neither is a judgement:

  A run coded histology `7205` for a hyperplastic polyp. `\\d{4}` accepts it, so
  `check_field_formats` passed it and `answer_shape_miss` never fired. There is no 7205 in
  ICD-O-3 morphology, and a hyperplastic polyp has no morphology code at all because it is not a
  neoplasm.

  A run wrote "ICD-O-3 topography C341 is right middle lobe" and coded C341 over cited evidence
  reading "right middle lobe". C341 is the UPPER lobe. The model's recollection of the code table
  was simply wrong.

Both are facts about a published classification. A regex cannot decide them and a
`contradicted_by` word list should not be asked to: the five word-list checks removed on the same
day destroyed a correct value 58 times against 21 helps
(`docs/DETERMINISTIC_RULES_REMOVED.md`). What was missing was never a cleverer matcher — it was
the table.

WHAT THIS MODULE DOES WITH IT, AND WHAT IT REFUSES TO DO
-------------------------------------------------------
Three uses, and none of them is a gate:

  1. `prompt_block()` renders the domain into the system prompt, so the model codes into a table
     it can see instead of one it half-remembers. This is what "the format belongs in the prompt"
     means when the format is a code system.
  2. `check_codes()` returns typed problems for the EVAL plane to count. An out-of-domain code
     becomes a number in a report, not a rejection in a loop.
  3. `normalize_code()` folds the punctuated form ICD-O-3 itself writes — `C18.7` to `C187`,
     `8140/3` to `8140` + `3`. That is a notation difference and not a mistake, and it is 4 of the
     6 useful firings the removed `field_format` check ever had: it was largely creating the round
     trips it then resolved.

REPORTABILITY IS A REGISTRY POLICY QUESTION. `not_reportable` in the YAML carries rulings this
file has no authority to make, which is why every problem this module returns is advisory and why
the table's own `source_authority` says a registrar must check them against the casefinding
manual.

SITE SCOPE, AND WHY IT IS A NAMED FINDING. One table per site group: `icdo3_lung` (C34, the
extraction corpus — all 1,788 gold topographies are C34x) and `icdo3_colorectal` (C18-C21, for
the CRC guideline tranche). `load_table` defaults to lung because that is what this repository
extracts.

Loading the wrong table would otherwise look like a wrong answer on every case, so `check_codes`
reports `OUT_OF_TABLE_SCOPE` — never `UNKNOWN_TOPOGRAPHY` — for a well-formed code belonging to
another site group. "This tumour is not a lung primary" and "this is not a code" are different
findings and a caller that cannot tell them apart will report the wrong one.

Each table also carries a `laterality` block, which is not a code field: it records which
subsites can coexist with which side. The left lung has no middle lobe, and a run coded C342
while its cited evidence read "left lower lobe" nine times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

CODES_DIR = Path(__file__).resolve().parents[2] / "codes"

#: Problem kinds `check_codes` can report. Names, not booleans, because "this is not a code" and
#: "this is a code for another organ" and "this is a benign entity" are three different findings
#: and a caller that cannot tell them apart will report the wrong one.
UNKNOWN_TOPOGRAPHY = "UNKNOWN_TOPOGRAPHY"
UNKNOWN_MORPHOLOGY = "UNKNOWN_MORPHOLOGY"
OUT_OF_TABLE_SCOPE = "OUT_OF_TABLE_SCOPE"
NOT_REPORTABLE_BEHAVIOR = "NOT_REPORTABLE_BEHAVIOR"
BEHAVIOR_NOT_IN_TABLE = "BEHAVIOR_NOT_IN_TABLE"
MALFORMED = "MALFORMED"
#: A code the SPEC declares out of scope, as distinct from one the table simply lacks. The lung
#: spec's `when_not_to_use` excludes haematopoietic neoplasms, and six patients in this corpus have
#: a lymphoma gold histology — those are SPEC_INSUFFICIENT cases, not coding errors and not table
#: gaps. Collapsing the three would make a scope boundary look like an incomplete table.
EXCLUDED_BY_SPEC = "EXCLUDED_BY_SPEC"


class CodeTableError(ValueError):
    """A code table is absent or does not have the shape this module reads."""


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
class CodeTable:
    table_id: str
    version: str
    scope: str
    topography: dict[str, dict]
    morphology: dict[str, dict]
    behavior: dict[str, dict]
    nos_topography: tuple[str, ...]
    nos_morphology: tuple[str, ...]
    not_reportable: tuple[dict, ...]
    safeguards: tuple[str, ...]
    excluded_by_spec: dict[str, dict]

    # -- lookup ---------------------------------------------------------------
    def topography_name(self, code: str) -> str | None:
        e = self.topography.get(normalize_code(code))
        return e.get("name") if e else None

    def morphology_name(self, code: str) -> str | None:
        e = self.morphology.get(normalize_code(code))
        return e.get("name") if e else None

    def reportable_behavior(self, digit: str) -> bool | None:
        e = self.behavior.get(str(digit).strip())
        return None if e is None else bool(e.get("reportable"))

    def benign_term_for(self, morphology: str, behavior: str) -> str | None:
        """The `not_reportable` term a (morphology, behaviour) pair names, if any."""
        m, b = normalize_code(morphology), str(behavior).strip()
        for row in self.not_reportable:
            if row.get("code") and normalize_code(str(row["code"])) == m \
                    and str(row.get("behavior", "")).strip() == b:
                return str(row.get("term", ""))
        return None


_PUNCT = re.compile(r"[.\s]")
_TOPO = re.compile(r"\AC\d{3}\Z")
_MORPH = re.compile(r"\A\d{4}\Z")


def normalize_code(raw: str) -> str:
    """Fold the punctuated form ICD-O-3 writes onto the digits-only form STORE takes.

    `C18.7` -> `C187`, `c18.7` -> `C187`, `8140/3` -> `8140` (the behaviour digit is a separate
    STORE item and is dropped here, not silently merged into the morphology).

    Notation, not judgement. The removed `field_format` check spent 4 of its 6 useful firings
    rejecting `C34.9`, `C34.11` and `C34.2` — the form the manual itself uses — so it was
    creating round trips rather than catching errors.
    """
    s = _PUNCT.sub("", str(raw or "")).upper()
    if "/" in s:
        s = s.split("/", 1)[0]
    return s


def behavior_digit(raw: str) -> str | None:
    """The behaviour digit if `raw` carries one as `8140/3`; otherwise None."""
    s = _PUNCT.sub("", str(raw or ""))
    return s.split("/", 1)[1][:1] if "/" in s and s.split("/", 1)[1] else None


@lru_cache(maxsize=8)
def load_table(name: str = "icdo3_lung", codes_dir: str | None = None) -> CodeTable:
    root = Path(codes_dir) if codes_dir else CODES_DIR
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise CodeTableError(
            f"no code table {name!r} at {path}; available: "
            f"{sorted(p.stem for p in root.glob('*.yaml'))}")
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for required in ("table_id", "topography", "morphology", "behavior"):
        if required not in d:
            raise CodeTableError(f"{path} has no {required!r}")
    return CodeTable(
        table_id=str(d["table_id"]),
        version=str(d.get("table_version", "0")),
        scope=str((d.get("source_authority") or {}).get("scope", "")),
        topography={normalize_code(k): (v or {}) for k, v in (d["topography"] or {}).items()},
        morphology={normalize_code(k): (v or {}) for k, v in (d["morphology"] or {}).items()},
        behavior={str(k): (v or {}) for k, v in (d["behavior"] or {}).items()},
        nos_topography=tuple(normalize_code(c) for c in (d.get("nos_topography") or [])),
        nos_morphology=tuple(normalize_code(c) for c in (d.get("nos_morphology") or [])),
        not_reportable=tuple(d.get("not_reportable") or []),
        safeguards=tuple(str(s) for s in (d.get("safeguards") or [])),
        excluded_by_spec={normalize_code(k): (v or {})
                          for k, v in (d.get("excluded_by_spec") or {}).items()},
    )


def check_codes(site: str | None, histology: str | None, behavior: str | None, *,
                table: CodeTable | None = None) -> list[CodeProblem]:
    """Factual problems with a coded triple. ADVISORY — for counting, never for refusing.

    Absent fields are skipped: an empty value is abstention's business, and this module has no
    opinion about whether abstaining was right.
    """
    t = table or load_table()
    out: list[CodeProblem] = []

    if site and str(site).strip():
        c = normalize_code(site)
        if not _TOPO.match(c):
            out.append(CodeProblem(MALFORMED, "primary_site", str(site),
                                   f"{site!r} is not shaped like an ICD-O-3 topography code "
                                   f"(letter C and three digits)"))
        elif c not in t.topography:
            # A well-formed code for another organ group is a DIFFERENT finding from a code that
            # does not exist. This table covers C18-C21 only.
            out.append(CodeProblem(
                OUT_OF_TABLE_SCOPE, "primary_site", c,
                f"{c} is not in {t.table_id} ({t.scope}). Either the tumour is not a colorectal "
                f"primary or the wrong table is loaded — this is not evidence that {c} is invalid."))

    if histology and str(histology).strip():
        m = normalize_code(histology)
        if not _MORPH.match(m):
            out.append(CodeProblem(MALFORMED, "histology", str(histology),
                                   f"{histology!r} is not shaped like an ICD-O-3 morphology code "
                                   f"(four digits, behaviour reported separately)"))
        elif m in t.excluded_by_spec:
            e = t.excluded_by_spec[m]
            out.append(CodeProblem(
                EXCLUDED_BY_SPEC, "histology", m,
                f"{m} is {e.get('name', '')} — {e.get('why', '')}. The spec's `when_not_to_use` "
                f"puts this outside the variable, so the honest answer is SPEC_INSUFFICIENT "
                f"rather than a site/histology/behaviour triple. This is a scope boundary, not a "
                f"coding error."))
        elif m not in t.morphology:
            out.append(CodeProblem(
                UNKNOWN_MORPHOLOGY, "histology", m,
                f"{m} is not a morphology in {t.table_id}. A four-digit number that is not in the "
                f"table is not a code; if the diagnosis has no ICD-O-3 morphology then the finding "
                f"is not a reportable neoplasm."))

    if behavior is not None and str(behavior).strip():
        b = str(behavior).strip()
        rep = t.reportable_behavior(b)
        if rep is None:
            out.append(CodeProblem(BEHAVIOR_NOT_IN_TABLE, "behavior", b,
                                   f"{b!r} is not a behaviour digit in {t.table_id}"))
        elif not rep:
            meaning = (t.behavior.get(b) or {}).get("meaning", "")
            term = t.benign_term_for(histology or "", b)
            out.append(CodeProblem(
                NOT_REPORTABLE_BEHAVIOR, "behavior", b,
                f"behaviour {b} ({meaning}) is not reportable"
                + (f"; {normalize_code(histology or '')}/{b} is {term}" if term else "")
                + ". A benign or borderline finding is not the reportable tumour."))
    return out


def prompt_block(table: CodeTable | None = None, *, max_terms: int = 0) -> str:
    """The value domain, for the system prompt. The whole table by default.

    Rendered rather than summarised: a model that is shown 12 of 40 morphologies will code into
    the 12. `max_terms` exists for a caller measuring prompt size, not as a default.
    """
    t = table or load_table()
    L = [f"ICD-O-3 VALUE DOMAIN — {t.table_id} v{t.version} ({t.scope})",
         "",
         "Code into this table. A four-digit number that is not in it is not a morphology code,",
         "and if a diagnosis has no ICD-O-3 morphology then the finding is not a reportable",
         "neoplasm — say that rather than choosing a number that looks like one.",
         "",
         "This table was recalled by a language model, not transcribed from ICD-O-3, and no",
         "registrar has checked it. Where it disagrees with a pathology report you have read,",
         "say so in your reasoning rather than silently following either one.",
         "",
         "TOPOGRAPHY"]
    topo = list(t.topography.items())[:max_terms] if max_terms else list(t.topography.items())
    L += [f"  {c}  {e.get('name', '')}"
          + (f"   [{', '.join(e['aliases'])}]" if e.get("aliases") else "")
          + ("   (NOS — asserts the subsite is not documented)" if c in t.nos_topography else "")
          for c, e in topo]
    L += ["", "MORPHOLOGY (four digits; behaviour is a separate field)"]
    morph = list(t.morphology.items())[:max_terms] if max_terms else list(t.morphology.items())
    L += [f"  {c}  {e.get('name', '')}"
          + ("   (NOS)" if c in t.nos_morphology else "")
          for c, e in morph]
    L += ["", "BEHAVIOUR"]
    L += [f"  {d}  {e.get('meaning', '')}"
          + ("" if e.get("reportable") else "   — NOT reportable")
          for d, e in t.behavior.items()]
    if t.not_reportable:
        L += ["", "COMMON FINDINGS THAT ARE NOT REPORTABLE NEOPLASMS"]
        L += [f"  {r.get('term', '')}"
              + (f"  ({normalize_code(str(r['code']))}/{r.get('behavior', '')})"
                 if r.get("code") else "  (no ICD-O-3 morphology exists)")
              + f" — {r.get('why', '')}"
              for r in t.not_reportable]
    if t.safeguards:
        L += ["", "CODING SAFEGUARDS"] + [f"  - {s}" for s in t.safeguards]
    return "\n".join(L)


def code_domain_block(spec) -> str:
    """The value domain for a spec that declares one, or "" for a spec that does not.

    The seam between the Task Contract and the prompt. A spec says WHICH table its values are
    coded into (`value_domain: icdo3_lung`) because that is part of what the answer means; this
    renders it. A spec that declares nothing gets nothing — the date and class-of-case variables
    have no ICD-O-3 domain and would only be given a wall of lung morphologies to ignore.

    `load_spec` has already refused a declared table that does not exist, so a name that reaches
    here resolves. The `try` is for a spec built in memory by a test or an ablation transform,
    where the guarantee does not hold and a missing table must not take down a run that never
    needed the block.
    """
    name = str(getattr(spec, "value_domain", "") or "").strip()
    if not name:
        return ""
    try:
        return prompt_block(load_table(name))
    except CodeTableError:
        return ""


def table_manifest(spec) -> dict | None:
    """The identity of the value domain a run was shown, or None when it declared none.

    Content-hashed for the same reason `skills_manifest` is: the tables are YAML a human is meant
    to edit — `what_a_human_must_check` in each one invites exactly that — so `table_version`
    alone would let a corrected code table masquerade as the one a previous run used. The 1,788
    row validation that added eleven morphologies to the lung table is precisely such an edit,
    and the manifests written before it must not be comparable to the ones written after.
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
        "n_topography": len(t.topography), "n_morphology": len(t.morphology),
        "n_excluded_by_spec": len(t.excluded_by_spec),
        "content_hash": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "origin": "model_recalled",
        "signed_off": False,
    }
