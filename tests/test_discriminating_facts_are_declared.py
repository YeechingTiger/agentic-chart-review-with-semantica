"""A conflict rule's discriminating fact must be declared, or its absence must be stated.

`conflict_rules` is prose today, so the fact a rule turns on is buried in the English of its `if`
clause and no code can read it:

    - if:   "cytology with an ambiguous term precedes a confirmatory biopsy AND a physician's
              clinical impression of cancer exists at the cytology date"
      then: "use the cytology date"

That is the whole reason the SYN0001 / SYNX03 mirror pair fails the way it does. Those two charts are
identical except for whether an oncology note records a clinical impression on the cytology date. A
run cited the cytology conflict rule, asserted *"no physician clinical impression of cancer is
documented at that date"*, and had never searched for one — and on the mirror chart, where the
impression genuinely is absent, the same shortcut answers correctly. Right answer, no reasoning, and
nothing in the runtime could tell the two apart.

Declaring the fact is what lets an obligation to check it exist BEFORE any competing value does. The
alternative — deriving it from two competing candidates — was built and measured: candidates formed no
value-against-value pair in 10 of 10 runs, `conflict_sets: []` throughout, so the fact was never
reached. See `docs/CANDIDATE_LEDGER_REMOVED.md`.

## Two things reading the real contracts changed

**A fact is shared between rules, so it is declared once.** STORE.390's rules 1 and 2 are the same
question with opposite branches — *impression present → cytology date*, *impression absent → biopsy
date*. Declaring the fact inside each rule would open two obligations for one question, and a run that
resolved one would still look incomplete on the other. So facts are declared at spec level with a
content-identity name and referenced by `turns_on`, which is also how `rule_catalog` already treats a
stratum: *"Where the spec already carries a CONTENT identity, that identity is reused rather than
re-derived."*

**Not every conflict rule turns on a fact.** STORE.390's rule 4 is *"IF documents disagree on the date
THEN take the earliest that satisfies the decision rules"* — a residual tie-break over dates already in
hand. Requiring a discriminating fact there would force one to be invented, which is the failure mode
this repo keeps catching. So a rule with no fact must say so with `no_discriminator: <reason>`, and
`load_spec` refuses a rule that declares neither. An empty container must not carry two meanings —
the same rule as `why_zero`, `_noop_why` and `n_unrecorded` elsewhere in this tree.

## Identity

`conflict_rule.N` already exists, is 1-based, matches what `as_prompt_block` renders and what the
model is already asked to cite, and carries `text_sha` so *"a position whose fingerprint moved is a
position that means something else now"*. Nothing new is minted. A fact is addressed by its declared
name, which is a content identity and survives a rule being reordered above it.
"""

from __future__ import annotations

import pytest
import yaml

from acr.contract.spec import (
    DiscriminatingFactError,
    ExtractionSpec,
    load_spec,
)
from acr.contract.trace import rule_catalog
from acr.core import site

SPEC_390 = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"


def _doc(**over) -> dict:
    """A minimal loadable contract, so a refusal is about the thing under test."""
    base = yaml.safe_load(SPEC_390.read_text(encoding="utf-8"))
    base.update(over)
    return base


def _load(doc, tmp_path):
    p = tmp_path / "T.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_spec(p)


# ------------------------------------------------------------------ the shipped contracts

def test_every_shipped_conflict_rule_declares_a_fact_or_states_it_has_none():
    """13 conflict rules across four contracts. Each must resolve one way or the other — this is the
    test that makes the schema real rather than optional."""
    import glob
    for path in sorted(glob.glob(str(site.specs_root() / "*.yaml"))):
        spec = load_spec(path)
        for i, rule in enumerate(spec.conflict_rules, start=1):
            assert isinstance(rule, dict), f"{path} conflict_rule.{i} is not a mapping"
            turns = rule.get("turns_on") or []
            assert turns or rule.get("no_discriminator"), (
                f"{path} conflict_rule.{i} declares neither `turns_on` nor `no_discriminator`; "
                f"an unstated absence is indistinguishable from an oversight")


def test_the_shared_fact_is_declared_once_and_referenced_twice():
    """STORE.390's rules 1 and 2 are one question with opposite branches. One fact, two references,
    so one obligation — not two, of which resolving either leaves the other looking open."""
    spec = load_spec(SPEC_390)
    facts = {f["fact_id"] for f in spec.discriminating_facts}
    refs = [r.get("turns_on") or [] for r in spec.conflict_rules]
    shared = [f for f in facts if sum(f in r for r in refs) >= 2]
    assert shared, "no fact is shared, but rules 1 and 2 ask the same question"


def test_the_residual_tie_break_states_that_it_turns_on_nothing():
    """Rule 4 is 'documents disagree -> take the earliest'. There is no further fact to check, and
    inventing one would be fabrication."""
    spec = load_spec(SPEC_390)
    tiebreaks = [r for r in spec.conflict_rules if not (r.get("turns_on") or [])]
    assert tiebreaks, "at least one rule is a pure tie-break on this contract"
    for r in tiebreaks:
        assert len(str(r.get("no_discriminator") or "")) > 20, (
            "the reason must be a reason, not a flag")


# ------------------------------------------------------------------ what load_spec refuses

def test_a_rule_declaring_neither_is_refused(tmp_path):
    doc = _doc()
    doc["conflict_rules"] = [{"if": "x", "then": "y"}]
    with pytest.raises(DiscriminatingFactError, match="turns_on"):
        _load(doc, tmp_path)


def test_an_empty_turns_on_with_no_reason_is_refused(tmp_path):
    """`turns_on: []` is the empty container. It must not mean both 'no fact' and 'not yet filled
    in' — the distinction `no_discriminator` exists to carry."""
    doc = _doc()
    doc["conflict_rules"] = [{"if": "x", "then": "y", "turns_on": []}]
    with pytest.raises(DiscriminatingFactError, match="no_discriminator"):
        _load(doc, tmp_path)


def test_a_reference_to_an_undeclared_fact_is_refused(tmp_path):
    """A rule pointing at a fact nobody declared would open an obligation with no question in it."""
    doc = _doc()
    doc["conflict_rules"] = [{"if": "x", "then": "y", "turns_on": ["no_such_fact"]}]
    with pytest.raises(DiscriminatingFactError, match="no_such_fact"):
        _load(doc, tmp_path)


def test_a_declared_fact_nobody_references_is_refused(tmp_path):
    """Dead weight in a contract is worse than absence: it renders into the prompt, hashes into
    `spec_hash`, and enforces nothing — `extra=allow` already lets a typo do that silently."""
    doc = _doc()
    doc["discriminating_facts"] = [
        *doc.get("discriminating_facts", []),
        {"fact_id": "orphan", "asks": "whether something", "applicable_when": "never"},
    ]
    with pytest.raises(DiscriminatingFactError, match="orphan"):
        _load(doc, tmp_path)


def test_a_fact_needs_an_asks_and_an_applicable_when(tmp_path):
    doc = _doc()
    doc["discriminating_facts"] = [{"fact_id": "bare"}]
    doc["conflict_rules"] = [{"if": "x", "then": "y", "turns_on": ["bare"]}]
    with pytest.raises(DiscriminatingFactError, match="asks"):
        _load(doc, tmp_path)


def test_a_duplicate_fact_id_is_refused(tmp_path):
    """The id is the obligation's target. Two facts under one name is two questions with one
    identity, and the ledger would dedup them into one."""
    doc = _doc()
    f = {"fact_id": "dup", "asks": "a", "applicable_when": "b"}
    doc["discriminating_facts"] = [f, dict(f, asks="c")]
    doc["conflict_rules"] = [{"if": "x", "then": "y", "turns_on": ["dup"]}]
    with pytest.raises(DiscriminatingFactError, match="dup"):
        _load(doc, tmp_path)


# ------------------------------------------------------------------ it reaches the reader

def test_the_facts_are_addressable_and_carry_a_fingerprint():
    """`rule_catalog` is how an attribution resolves a rule and how drift is detected. A fact that
    is not in it cannot be cited and cannot be seen to have changed."""
    spec = load_spec(SPEC_390)
    rows = {r.rule_id: r for r in rule_catalog(spec)}
    facts = [f["fact_id"] for f in spec.discriminating_facts]
    for fid in facts:
        rid = f"discriminating_fact.{fid}"
        assert rid in rows, f"{rid} is not addressable through rule_catalog"
        assert rows[rid].text_sha, "a fact with no fingerprint cannot be seen to drift"
        assert rows[rid].kind == "discriminating_fact"


def test_the_prompt_shows_the_fact_beside_the_rule_that_turns_on_it():
    """The model has to be able to tell which question decides which rule; a fact list rendered
    apart from its rules is a list of questions with no consequences attached."""
    spec = load_spec(SPEC_390)
    block = spec.as_prompt_block(view="full")
    assert "CONFLICT RESOLUTION:" in block
    for f in spec.discriminating_facts:
        assert f["fact_id"] in block, f"{f['fact_id']} never reaches the model"
        assert f["asks"][:40] in block


def test_a_contract_with_no_conflict_rules_is_untouched():
    """STORE.610 declares none. It must load, render and hash exactly as before."""
    spec = load_spec(site.specs_root() / "STORE.610.class_of_case.yaml")
    assert spec.conflict_rules == []
    assert spec.discriminating_facts == []
    assert isinstance(spec, ExtractionSpec)


# ------------------------------------------------------------------ provenance

def test_a_declared_fact_does_NOT_yet_need_provenance_and_that_is_deliberate():
    """`enforced_elements` states its own admission rule, and this respects it rather than overriding
    it:

        "An element enters this list when some code path CHANGES BEHAVIOUR on its value...
         Requiring provenance for a line the runtime ignores would teach authors that these
         records are paperwork. When any of them is wired up, it belongs in this list in the
         same commit."

    Nothing reads `discriminating_facts` yet. The obligation ledger is what will act on them, and
    until it does, demanding a provenance record here would be paperwork by that definition — the
    module names three other near-misses it excludes for exactly this reason.

    So this test pins the CURRENT correct state and names the commit that must change it. When
    obligations are created from these facts, they become enforced, this test flips, and nine
    provenance records get authored in that same commit.
    """
    from acr.contract.spec import enforced_elements
    spec = load_spec(SPEC_390)
    assert spec.discriminating_facts, "the facts are declared"
    paths = {e.path for e in enforced_elements(spec)}
    assert not any("discriminating_fact" in p for p in paths), (
        "a discriminating fact is in enforced_elements while nothing reads it. If the obligation "
        "ledger now acts on them, flip this test and author the provenance records in the same "
        "commit — that is what `enforced_elements`' docstring asks for.")


def test_the_spec_hash_moves_once_and_the_shipped_contracts_still_lint():
    """Adding the facts retires every baseline recorded before it — correctly, the question changed.
    What must not happen is a contract that no longer loads or lints."""
    import subprocess
    r = subprocess.run([".venv/bin/python", "-m", "acr.commands.cli", "spec", "lint",
                        "assets/specs"], capture_output=True, text=True,
                       env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert "tier-1 failure" in (r.stdout + r.stderr), r.stdout[-400:]
    n = int(str(r.stdout).split("tier-1 failure")[0].split()[-1])
    assert n <= 14, f"{n} tier-1 failures, above the CI floor of 14"


# ------------------------------------------------------------------ the refusal is a ratchet

def test_a_contract_declaring_no_facts_at_all_still_loads(tmp_path):
    """A RATCHET, NOT A CLIFF, and the difference is 12 contracts.

    Refusing every contract whose conflict rules are silent was tried and would have made the 12
    contracts under `assets/usecase/crc/` unloadable, along with test fixtures whose conflict rule is
    incidental scaffolding. Authoring discriminating facts for a use case nobody here can speak for
    is exactly the fabrication this schema exists to prevent — the same reason a residual tie-break
    is allowed to declare no fact rather than being made to invent one.

    So: a contract that has not started declaring facts is loadable, and `spec lint` reports it.
    """
    doc = _doc()
    doc.pop("discriminating_facts", None)
    doc["conflict_rules"] = [{"if": "x", "then": "y"}]
    spec = _load(doc, tmp_path)
    assert spec.discriminating_facts == []


def test_a_contract_that_has_started_declaring_must_finish(tmp_path):
    """The ratchet's teeth: one declared fact makes every rule accountable. Half-compliance is the
    state in which the completeness column silently covers some rules and not others."""
    doc = _doc()
    doc["conflict_rules"] = [
        {"if": "a", "then": "b", "turns_on": ["impression_at_ambiguous_cytology"]},
        {"if": "c", "then": "d"},                      # the unfinished one
    ]
    with pytest.raises(DiscriminatingFactError, match="must finish"):
        _load(doc, tmp_path)


def test_the_structural_refusals_fire_regardless_of_whether_facts_are_declared(tmp_path):
    """Only the "said nothing" refusal is ratcheted. A rule pointing at a fact that does not exist is
    broken in any contract, annotated or not."""
    doc = _doc()
    doc.pop("discriminating_facts", None)
    doc["conflict_rules"] = [{"if": "x", "then": "y", "turns_on": ["ghost"]}]
    with pytest.raises(DiscriminatingFactError, match="ghost"):
        _load(doc, tmp_path)
