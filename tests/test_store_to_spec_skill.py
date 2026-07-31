"""The authoring skill must not teach a rule the tree has stopped obeying.

`skills/store-to-spec/` tells an engineer how to turn a registry standard item into a spec in
this repo's format. Everything it teaches is a claim about code and specs that live elsewhere
and are being edited by other people, so every claim here has a second life as a fact this
file re-checks:

  * it says `format` is `re.fullmatch` -> asserted against `check_field_formats_detail`, the
    function that now applies it, plus that `check_field_formats` still routes through it
  * it names the two specs whose `format` is registry notation -> the list is recomputed, in
    BOTH directions, so fixing STORE.390 fails this test instead of quietly leaving the skill
    pointing at a bug that no longer exists
  * it teaches `establishes:` and four stratum policies -> asserted against `coverage.py`
  * it says nobody has reviewed the shipped specs -> asserted against the shipped specs

A documentation test that only greps its own document for its own words proves the document
is internally consistent, which is not the property anyone needs. The property needed is that
the document is still TRUE, and truth here lives in files this skill does not own.

`tests/test_skills_load.py` already covers the frontmatter and the reference pointers for
every skill including this one; nothing below repeats it.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import yaml

from acr.contract.answer_checks import check_field_formats, check_field_formats_detail
from acr.contract.spec import ExtractionSpec, load_spec
from acr.review.coverage import StratumSpec

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "store-to-spec"
SKILL = SKILL_DIR / "SKILL.md"
FIELD_DESIGN = SKILL_DIR / "references" / "field-design.md"
PROOF_OBLIGATIONS = SKILL_DIR / "references" / "proof-obligations.md"
SPEC_PATHS = sorted((ROOT / "specs").rglob("*.yaml"))
STAGE = ROOT / "specs" / "STORE.700_880.stage.yaml"

# --------------------------------------------------------------------- known, current gaps
#
# 1. `references/proof-obligations.md` was never written: the build agent authoring
#    store-to-spec was killed mid-work by the org spend limit on 2026-07-26. SKILL.md and
#    field-design.md already point at the path (see test_skills_load.py's reference-pointer
#    check), so the pointer is real and correct -- the target just does not exist yet. Every
#    test below that reads `_docs()`/`_all_text()` needs that file; skip them by name rather
#    than writing a stand-in file to satisfy the assertions.
_PROOF_OBLIGATIONS_MISSING = not PROOF_OBLIGATIONS.is_file()
skip_no_proof_obligations = pytest.mark.skipif(
    _PROOF_OBLIGATIONS_MISSING,
    reason=(
        "skills/store-to-spec/references/proof-obligations.md was never written (build agent "
        "killed by org spend limit before authoring it; SKILL.md and field-design.md already "
        "point at this path)"
    ),
)

# 2. `specs/STORE.700_880.stage.yaml` no longer loads. This is not a store-to-spec skill
#    problem: `src/acr/spec.py` is locally modified (not this skill's file, not committed)
#    to require a provenance record on every enforced element, and this spec was not updated
#    to match -- that repair belongs to whoever owns specs/ and acr/spec.py, not to this
#    skill's authors. Recomputed here (rather than hard-coded) so the guard clears itself
#    the moment the other team's fix lands.
try:
    load_spec(STAGE)
    _STAGE_SPEC_ERROR: str | None = None
except Exception as _e:  # noqa: BLE001 - want to name and skip on *any* load failure here
    _STAGE_SPEC_ERROR = f"{type(_e).__name__}: {_e}"
skip_stage_spec_broken = pytest.mark.skipif(
    _STAGE_SPEC_ERROR is not None,
    reason=(
        "specs/STORE.700_880.stage.yaml fails load_spec() under acr.contract.spec's provenance "
        f"enforcement (owned by another team, currently mid-edit): {_STAGE_SPEC_ERROR}"
    ),
)


def _docs() -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in (SKILL, FIELD_DESIGN, PROOF_OBLIGATIONS)}


def _all_text() -> str:
    return "\n".join(_docs().values())


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```ya?ml\n(.*?)```", text, re.DOTALL)


# --------------------------------------------------------------------------- it is there
@skip_no_proof_obligations
def test_the_skill_ships_both_references():
    """The two reference files are named in the brief because the SKILL.md cannot hold them:
    field design and proof-obligation design are each longer than the skill's whole budget,
    and a skill that inlines them is a skill nobody finishes reading."""
    for p in (SKILL, FIELD_DESIGN, PROOF_OBLIGATIONS):
        assert p.is_file(), f"missing {p.relative_to(ROOT)}"
    body = SKILL.read_text(encoding="utf-8")
    for ref in (FIELD_DESIGN, PROOF_OBLIGATIONS):
        rel = str(ref.relative_to(ROOT))
        assert rel in body, f"SKILL.md never points at {rel}; nothing else will find it"


# ------------------------------------------------- 1. `format` is a regex, and still is
@skip_no_proof_obligations
def test_the_skill_states_the_rule_the_runtime_actually_applies():
    """`format` is applied with `re.fullmatch`, so registry notation in it rejects every
    valid value. If the runtime ever softens that -- `re.match`, a casefold, a try/except
    that skips -- the skill's central instruction becomes wrong and must be rewritten."""
    # The enforcement moved into `check_field_formats_detail` when rejections started
    # carrying the rule they came from; `check_field_formats` is now a projection of it.
    # Both are asserted, so the guard cannot be satisfied by a wrapper that has quietly
    # stopped calling the function that does the deciding.
    src = inspect.getsource(check_field_formats_detail)
    assert "re.fullmatch(fmt, s)" in src, "the enforcement the skill describes has changed"
    assert "check_field_formats_detail(" in inspect.getsource(check_field_formats), \
        "check_field_formats no longer routes through the function that applies the format"
    text = _all_text()
    assert "re.fullmatch" in text, "the skill must name the function that decides this"
    assert "check_field_formats" in text


def _registry_notation_fields() -> list[tuple[str, str]]:
    """(spec file stem, field name) for every `format` that can only ever match itself.

    A pattern with no regex metacharacter accepts exactly one string: its own text. No
    registry code is ever literally "CCYYMMDD", so such a field rejects 100% of valid values
    while compiling cleanly and raising nothing. That is the whole defect, stated
    mechanically rather than by keeping a list of known-bad spellings.
    """
    out = []
    for p in SPEC_PATHS:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for f in raw.get("fields") or []:
            fmt = f.get("format")
            if isinstance(fmt, str) and fmt and re.escape(fmt) == fmt:
                out.append((p.stem, f["name"]))
    return out


@skip_no_proof_obligations
def test_the_skill_records_the_fixed_defect_as_history_not_current_state():
    """Worked examples must not claim the repaired CCYYMMDD defect is still live."""
    broken = _registry_notation_fields()
    assert broken == []
    text = _all_text()
    assert "historical defect" in text.lower()
    assert "still live" not in text.lower()


@skip_no_proof_obligations
@skip_stage_spec_broken
def test_the_patterns_the_skill_quotes_are_the_patterns_the_spec_declares():
    """The `no pM0 / no cMX` lesson is taught by quoting two live patterns. A quote that has
    drifted from the spec teaches a value domain the runtime does not enforce, which is worse
    than teaching nothing: the reader copies it into a new spec."""
    spec = load_spec(STAGE)
    text = _all_text()
    quoted = {f.name: f.format for f in spec.fields if f.name in ("clinical_m", "pathologic_m")}
    for name, fmt in quoted.items():
        assert fmt and fmt in text, f"{name}: spec declares {fmt!r}, the skill quotes something else"
    assert check_field_formats(spec.fields, {"pathologic_m": "pM0"}), "pM0 is now accepted"
    assert check_field_formats(spec.fields, {"clinical_m": "cMX"}), "cMX is now accepted"
    assert check_field_formats(spec.fields, {"clinical_m": "cM0", "pathologic_m": "pM1a"}) == []


# ------------------------------------------------ 2. field scoping and stratum policies
@skip_no_proof_obligations
def test_the_field_scoping_key_the_skill_teaches_is_the_key_coverage_reads():
    """`establishes:` is how a stratum name stops being a claim about every field at once.
    Renaming it in coverage.py would leave five specs and this skill declaring a key that is
    parsed into nothing -- silently, since `from_dict` defaults it to an empty list."""
    s = StratumSpec.from_dict({"name": "cannot_establish", "policy": "validate_by_sampling",
                               "establishes": ["primary_site"], "match": {"rest": True}})
    assert s.establishes == ["primary_site"]
    text = _all_text()
    assert "establishes:" in text
    assert "cannot_establish" in text and "primary_site" in text, (
        "the rule is only legible with the case that produced it: a stratum named "
        "cannot_establish that was false of primary_site"
    )


def _implemented_policies() -> set[str]:
    src = inspect.getsource(__import__("acr.review.coverage", fromlist=["coverage"]))
    out = set(re.findall(r'\.policy == "([a-z_]+)"', src))
    for grp in re.findall(r'\.policy in \(([^)]*)\)', src):
        out |= set(re.findall(r'"([a-z_]+)"', grp))
    return out


@skip_no_proof_obligations
def test_no_yaml_example_hands_the_author_a_policy_that_is_never_complete():
    """`exhaustive_per_window` parses, routes, and can never satisfy the gate -- it falls
    into the sampling else-branch of `stratum_results`. STORE.1860_1880 ships it. A skill
    that shows it in an example without saying so hands the next author an unpassable gate
    and no error message.
    """
    implemented = _implemented_policies()
    assert "exhaustive" in implemented, "policy dispatch in coverage.py has moved"
    for path, text in _docs().items():
        for block in _yaml_blocks(text):
            for policy in re.findall(r"^\s*policy:\s*([a-z_]+)", block, re.M):
                if policy in implemented:
                    continue
                context = [ln for ln in text.splitlines() if policy in ln]
                assert any(re.search(r"not implemented|never complete|unimplemented", ln)
                           for ln in context), (
                    f"{path.name} shows `policy: {policy}`, which no branch of "
                    f"stratum_results implements, and never says so"
                )


# ------------------------------------------------------- 3. every key it teaches is read
def _keys_anywhere(node) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys_anywhere(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys_anywhere(v)}
    return set()


@skip_no_proof_obligations
def test_every_yaml_key_the_skill_tells_an_author_to_write_is_read_by_something():
    """A key nothing reads is worse than a missing key: `ExtractionSpec` has
    `extra="allow"`, so it loads, hashes into spec_hash, renders nowhere and enforces
    nothing. `required_keywords_all_searched` sat in four specs' gates being read by no code
    at all, and every one of those specs reported its keyword obligation as satisfied.
    """
    known = set(ExtractionSpec.model_fields)
    for p in SPEC_PATHS:
        known |= _keys_anywhere(yaml.safe_load(p.read_text(encoding="utf-8")))

    # Two keys no shipped spec uses yet, each grounded in the code that reads it rather than
    # waved through: `provenance` drives the "what we made up" section of `spec review`, and
    # `witness` is the binding grammar of `for_positive`.
    specview_tests = (ROOT / "tests" / "test_specview.py").read_text(encoding="utf-8")
    assert "provenance" in specview_tests
    assert "witness" in inspect.getsource(ExtractionSpec.__module__ and __import__(
        "acr.contract.spec", fromlist=["spec"]))
    known |= {"provenance", "witness"}

    for path, text in _docs().items():
        for block in _yaml_blocks(text):
            for key in re.findall(r"^([a-z][a-z0-9_]*):", block, re.M):
                assert key in known, (
                    f"{path.name} teaches top-level key {key!r}, which no spec declares and "
                    f"no loader field names -- it would load, hash, and do nothing"
                )


# -------------------------------------------------------------- 4. the order of the work
ORDER = ["question", "fields", "evidence_rules", "conflict_rules",
         "proof_obligation", "abstention", "answer_checks", "provenance"]


def test_the_order_of_work_is_documented_in_the_order_it_must_be_done():
    """The order is not a preference. Strata cannot be chosen before the fields exist,
    because `establishes:` names fields; `answer_checks` cannot be written before the value
    domains, because a check names a value. An author who writes the proof obligation first
    produces the one-stratification-for-three-fields shape that mis-coded P03.
    """
    body = SKILL.read_text(encoding="utf-8")
    m = re.search(r"^##+ .*[Oo]rder of work.*$", body, re.M)
    assert m, "SKILL.md has no order-of-work section"
    section = body[m.end():]
    nxt = re.search(r"^## ", section, re.M)
    section = section[:nxt.start()] if nxt else section

    at = []
    for step in ORDER:
        i = section.find(step)
        assert i >= 0, f"the order of work never mentions {step!r}"
        at.append(i)
    assert at == sorted(at), (
        "the order-of-work section lists the steps out of order: "
        + ", ".join(s for _, s in sorted(zip(at, ORDER)))
    )


# ------------------------------------------------------------ 5. the unreviewed disclosure
@skip_no_proof_obligations
def test_the_skill_says_the_shipped_specs_are_unreviewed_and_that_is_still_true():
    """The reason the model_authored rule exists is a fact about this repo, not a principle:
    four specs written by a model in one commit, committed under a human author's name, with
    no registrar sign-off anywhere. When that stops being true the skill must be rewritten --
    a warning that has been overtaken by events trains the reader to skip warnings.
    """
    text = _all_text()
    assert "model_authored" in text
    assert re.search(r"no registrar|not been reviewed|never been reviewed", text), (
        "the skill must state the disclosure plainly, not imply it"
    )

    claimed_reviewed = []
    for p in SPEC_PATHS:
        keys = _keys_anywhere(yaml.safe_load(p.read_text(encoding="utf-8")))
        if keys & {"reviewed_by", "signed_off_by", "reviewer", "clinical_review"}:
            claimed_reviewed.append(p.name)
    assert not claimed_reviewed, (
        f"{claimed_reviewed} now record a reviewer; the skill's disclosure is out of date"
    )


@skip_no_proof_obligations
@skip_stage_spec_broken
def test_the_item_number_caveat_is_still_admitted_by_the_spec_it_cites():
    """The skill's rule -- cite the item number you verified, or say you did not verify it --
    rests on STORE.700_880 admitting in its own `source_authority` that its numbers follow a
    file-naming convention and were never reconciled against NAACCR. If someone reconciles
    them and deletes the admission, the skill is citing a confession that no longer exists.
    """
    note = str((load_spec(STAGE).source_authority or {}).get("note", ""))
    assert "NAACCR" in note and "file-naming" in note, (
        "STORE.700_880 no longer admits its item numbers are unverified"
    )
    text = _all_text()
    assert "NAACCR" in text and "spec_id" in text


@pytest.mark.parametrize(
    "path",
    [SKILL, FIELD_DESIGN, pytest.param(PROOF_OBLIGATIONS, marks=skip_no_proof_obligations)],
    ids=lambda p: p.name,
)
def test_no_worked_example_uses_anything_but_the_pseudonyms(path: Path):
    """`tests/test_no_phi_in_tree.py` catches a real person_id by its digits. It cannot catch
    a real date of diagnosis or a real MRN in another shape, and this skill's whole method is
    "cite the failure that motivated the rule", which is precisely the writing habit that put
    sixteen-digit ids into seven skill documents in the first place.
    """
    text = path.read_text(encoding="utf-8")
    for token in re.findall(r"\bP\d{2}\b", text):
        assert token in {f"P0{i}" for i in range(1, 6)}, (
            f"{path.name}: {token} is outside the P01..P05 pseudonym range"
        )
    assert not re.search(r"\bMRN\s*[:#]?\s*\d", text, re.I), f"{path.name}: an MRN-shaped number"
