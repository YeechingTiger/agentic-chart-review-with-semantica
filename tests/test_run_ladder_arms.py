"""Every arm has to assemble before the first penny is spent.

This test exists because of a real accident, and the second one of the same kind. Package C
(02d8c21) split one `search` slot into `controller` + `tactic` and renamed every card;
`tools/run_ladder.py` received `s/search-/controller-/` and nothing else. Five of the seven arms
then named cards that did not exist — `controller-native`, `controller-breadth-first`,
`controller-depth-first`, `controller-breadth-then-depth`, `controller-latest-first`. (The slot was
renamed again on 2026-08-03, from `controller` to `policy`; the names above are left exactly as they
were at the time of the accident rather than rewritten along with it.) What running it would do:
the first two arms spend money normally, the third raises out of `SkillStack.validate`.

A driver script that only discovers its own misspelled names at runtime is a driver script you must
pay to get an error message from. So the validation moved into `preflight()`, and this test runs it
in CI.

`tools/` is not a package, so it is loaded by path. This is the first test in this tree to reach
into `tools/`, and the reason is that what lives in there — which arms get run — is part of the
experiment design, not scaffolding.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from acr.contract.skills import SkillError
from acr.core import site

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ladder():
    return _load("run_ladder")


@pytest.mark.parametrize("group", ["policy", "tactic", "experience", "all"])
def test_every_arm_assembles_and_validates(ladder, group):
    """Naming a card that does not exist, or putting a card in the wrong slot, blows up here rather
    than on the third arm."""
    resolved = ladder.preflight(ladder.arms_for(group))
    assert resolved, group
    for name, stack in resolved:
        assert isinstance(name, str) and name
        stack.validate()          # Cheap redundancy: if preflight stops validating, this still does


def test_the_baseline_arm_is_in_every_group(ladder):
    """Every group needs its own floor, or a comparison across groups subtracts two different
    baselines."""
    for group in ladder.GROUPS:
        assert "B0-base" in {n for n, _ in ladder.arms_for(group)}, group


def test_all_does_not_run_the_baseline_three_times(ladder):
    names = [n for n, _ in ladder.arms_for("all")]
    assert len(names) == len(set(names))


def test_a_card_that_does_not_exist_still_fails_loudly(ladder):
    """Falsify the guard itself once: `preflight` is not a function that always passes."""
    with pytest.raises(SkillError):
        ladder.preflight([("bogus", "policy=policy-native")])


def test_the_policy_arms_are_every_policy_card_in_the_tree(ladder):
    """A policy card that is in the tree and not on the ladder is an intervention nothing measures.

    This test used to pin the list to two cards, on the grounds that "when a third one appears
    somebody has to say what it is". A third one appeared (`policy-hypothesis-set`, 2026-08-03, in
    place of the deleted candidate-ledger machinery), and a pinned list means adding a card requires
    editing a test — so what this test measured was whether I remembered to edit it. It reads the
    tree now: a card missing from the ladder is a card nothing tests.
    """
    on_disk = {p.name for p in (site.skills_root()).iterdir()
               if p.name.startswith("policy-")}
    named = {s.policy for _, s in ladder.preflight(ladder.POLICY_ARMS)} - {None}
    assert named == on_disk, "the policy ladder and the tree disagree"


def test_the_tactic_arms_cover_every_tactic_card_in_the_tree():
    """A tactic card in the tree and not on the ladder is an intervention nothing measures."""
    ladder = _load("run_ladder")
    on_disk = {p.name for p in (site.skills_root()).iterdir()
               if p.name.startswith("tactic-")}
    assert set(ladder._TACTICS) == on_disk, "the tactic ladder and the tree disagree"


# ============================================================ SAY WHAT IT COSTS BEFORE SPENDING IT

@pytest.fixture(scope="module")
def budget():
    return _load("_driver_budget")


def test_the_ceiling_is_reported_in_the_unit_it_actually_is(budget):
    """`--max-usd` is `acr batch`'s **per-run** ceiling, and the ladder printed it as `$3.00/arm`.

    27 charts to an arm, so that `--dry-run` line understated the worst case 27-fold. This is not a
    formatting problem: `--dry-run` exists for exactly one reason, which is to see what it costs
    before deciding whether to run it, and the number it gave was 1/27 of the real bound.
    """
    r = budget.budget_report(n_arms=7, n_charts=27, max_usd_per_run=3.0)
    assert r["runs"] == 189
    assert r["per_run_ceiling_usd"] == 3.0
    assert r["worst_case_usd"] == 567.0, "7 arms x 27 charts x $3 per-run ceiling"


def test_the_line_names_both_numbers_and_which_is_which(budget):
    """The total alone reads as a spend already committed; the per-run ceiling alone is the original
    defect. Print both, and name the unit on each."""
    line = budget.budget_line(budget.budget_report(n_arms=2, n_charts=18, max_usd_per_run=3.0))
    assert "$3.00" in line and "per run" in line
    assert "$108.00" in line and "worst case" in line
    assert "36 run" in line


def test_a_single_run_is_not_pluralised_into_a_worse_number(budget):
    r = budget.budget_report(n_arms=1, n_charts=1, max_usd_per_run=0.5)
    assert r["worst_case_usd"] == 0.5


def test_both_drivers_print_the_bound_not_the_per_run_ceiling(ladder, capsys, monkeypatch):
    """Each script doing the arithmetic itself is two answers to "what does this cost", and one of
    the two has already been wrong once.

    It looks at the **output**, not at the source. The first version of this test asserted that
    `/arm` does not appear in the source — and it tripped over its own comment explaining the old
    defect. A guard that substring-matches source treats a record of the defect as the defect.
    """
    # `run_floor` returns 2 (against STORE.390 it refuses an INERT planner axis) and `run_ladder`
    # returns 0. What this test asks about is the budget line, so it reads the output and not the
    # return code — the return code has its own test.
    for mod in (_load("run_floor"), ladder):
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
        mod.main(["--dry-run", "--max-usd", "3.0", "--patients", "SYN0001,SYN0002,SYN0003"])
        out = capsys.readouterr().out
        assert "$3.00 per run" in out, f"{mod.__name__} does not name the unit"
        assert "worst case" in out, f"{mod.__name__} does not give the bound"
        assert "/arm" not in out, f"{mod.__name__} still prints the per-run ceiling as per-arm"


# ============================================================ --help MUST NOT START SPENDING

def test_run_floor_parses_its_arguments_before_doing_anything():
    """`run_floor.py` had no argparse. `python tools/run_floor.py --help` ignored the argument and
    launched 36 real batches — somebody trying to work out how to use the script paid for it.

    It returns 2, not 0, because it finds something more pressing on the spot: see the next test.
    """
    floor = _load("run_floor")
    assert floor.main(["--dry-run"]) == 2


def test_run_floor_refuses_an_inert_planner_axis(capsys):
    """Its header says "TWO ARMS, one variable: the runtime profile" — and STORE.390 declares not
    one stratum, so `plan_from_spec` sends every type into `search`, exactly as
    `plan_from_patient_inventory` does.

    This script's whole claim is that the spec's hand-written plan is a prior worth falsifying. On
    this contract it is not the variable: the two arms differ only in coverage policy and in the
    spec view. $108 buys not one word about the plan, so this refuses rather than adding a line of
    warning — a warning is read once and then becomes part of the output format.
    """
    floor = _load("run_floor")
    assert floor.main(["--dry-run"]) == 2
    out = capsys.readouterr().out
    assert "INERT" in out and "Refusing" in out


def test_run_floor_dry_run_spawns_no_subprocess(monkeypatch):
    """Spending nothing is a thing to be proved, not declared. The original script did not even have
    a boundary that could be stubbed."""
    floor = _load("run_floor")
    calls = []
    monkeypatch.setattr(floor.subprocess, "run",
                        lambda *a, **k: calls.append(a) or _Probe())
    # It returns at the INERT refusal, so it never even reaches the `--help` probe — which is what
    # has to be proved: the refusal happens before any spending, and before any subprocess at all.
    assert floor.main(["--dry-run"]) == 2
    assert calls == []


class _Probe:
    returncode = 0
    stdout = "--patients"
    stderr = ""


def test_run_floor_still_refuses_an_arm_whose_profile_does_not_exist():
    """Falsify the guard once: `preflight` does not always pass."""
    floor = _load("run_floor")
    with pytest.raises(Exception):
        floor.preflight([("bogus", "no-such-runtime-profile")])
