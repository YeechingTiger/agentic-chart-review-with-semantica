"""Every claim this repository makes about its own shape, as a check that can fail.

    python tools/verify_structure.py
    python tools/verify_structure.py --json

WHY THIS EXISTS, and it is not tidiness. On 2026-08-03 this tree was cut into four distributions and
every suite passed — 1,319 tests green, composed and isolated. Then a five-line script asked whether
any repository imported a module assigned to another without declaring the dependency, and the answer
was SIX, all reaching into `contract/spec_repair.py`. Nothing failed, because the verification
environment had all four installed. A boundary that is only true when everything is present is not a
boundary; it is a description of one.

That is the general shape of the problem this file addresses. This project's structural claims live
in prose — a docstring saying which planes may depend on which, a README saying a plane is
self-contained, a commit message saying nine repositories became four. Prose does not fail. So every
claim worth making is written here as an assertion over the tree, and CI runs it.

THE IDIOM IS `tools/verify_mechanisms.py`'s, deliberately: named checks, an explicit verdict per
check, and the ability to report a check as INERT — passing because it had nothing to examine rather
than because the property holds. An inert check and a satisfied check print the same PASS, and this
repository has already shipped an audit rule that could never fire from the day it was written,
because nothing wrote the four trace keys it read. A checker that cannot say "I had nothing to look
at" reproduces that defect one level up.

WHAT IS DELIBERATELY NOT HERE. The tests own behaviour; this owns SHAPE. `tests/test_layering.py`
already pins which layer may import which and is a better place for it, because it fails in the same
run as the code it governs. This file asks the questions a test cannot: questions about the
DISTRIBUTION boundary, which does not exist inside a single checkout.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from split_repos import CARD_OWNERS, REPOS, TESTS_BY_SUBJECT, assign_tests, collect

PASS, FAIL, INERT = "PASS", "FAIL", "INERT"


@dataclass
class Check:
    name: str
    claim: str
    verdict: str = PASS
    examined: int = 0
    findings: list[str] = field(default_factory=list)

    def fails(self, detail: str) -> None:
        self.verdict = FAIL
        self.findings.append(detail)


def _owners() -> dict[str, str]:
    """Every source file under `src/acr/`, mapped to the distribution that ships it."""
    out: dict[str, str] = {}
    for name in REPOS:
        for f in collect(REPOS[name], name):
            if f.startswith("src/acr/") and f.endswith(".py"):
                out[f] = name
    return out


def _string_literals(path: pathlib.Path) -> set[str]:
    """Every string a module actually USES, excluding docstrings.

    The first version of check S5 asked whether a card's name appeared anywhere in a file's text, and
    its first finding was its own false positive: `core/site.py` mentions `store-to-spec` in a
    DOCSTRING explaining how the cards are distributed. A docstring naming a card is not a consumer
    of it; a string literal in code is. The distinction is the difference between a check that
    reports the tree and a check that reports its own comments.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    docstrings = {id(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                  and isinstance(n.value.value, str)}
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings}


def _imports(path: pathlib.Path):
    """(target module path, imported name, lineno) for every `from … import …` in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return
    pkg = "/".join(str(path).split("/")[:-1])
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom):
            continue
        if n.level:
            base = pkg.split("/")
            base = base[: len(base) - (n.level - 1)] if n.level > 1 else base
            target = "/".join(base + ([n.module.replace(".", "/")] if n.module else []))
        else:
            target = (n.module or "").replace(".", "/").replace("acr/", "src/acr/", 1)
        for alias in n.names:
            yield target, alias.name, n.lineno


# --------------------------------------------------------------------------------- the checks

def check_every_module_is_assigned(owners) -> Check:
    c = Check("S1-assignment",
              "every module under src/acr/ ships in exactly one distribution")
    seen: dict[str, list[str]] = {}
    for name in REPOS:
        for f in collect(REPOS[name], name):
            if f.startswith("src/acr/") and f.endswith(".py"):
                seen.setdefault(f, []).append(name)
    for f in sorted(ROOT.glob("src/acr/**/*.py")):
        rel = str(f.relative_to(ROOT))
        if "__pycache__" in rel:
            continue
        c.examined += 1
        homes = seen.get(rel, [])
        if not homes:
            c.fails(f"{rel}: assigned to NO distribution — it would silently disappear")
        elif len(homes) > 1:
            c.fails(f"{rel}: assigned to {homes} — two copies drift")
    return c


def check_no_undeclared_cross_repo_import(owners) -> Check:
    c = Check("S2-import-boundary",
              "no distribution imports a module owned by another unless it declares the dependency")
    for f, home in sorted(owners.items()):
        c.examined += 1
        for target, alias, lineno in _imports(ROOT / f):
            for candidate in (f"{target}.py", f"{target}/{alias}.py"):
                other = owners.get(candidate)
                if other and other != home and other not in REPOS[home]["deps"]:
                    c.fails(f"{f}:{lineno}  {home} → {other}  "
                            f"({candidate.replace('src/acr/', '')}: {alias})")
    return c


def check_dependencies_are_acyclic(owners) -> Check:
    c = Check("S3-acyclic", "the declared dependency graph between distributions has no cycle")
    edges = {n: set(REPOS[n]["deps"]) for n in REPOS}
    c.examined = len(edges)
    colour: dict[str, int] = {}

    def walk(n: str, path: list[str]) -> None:
        if colour.get(n) == 1:
            c.fails("cycle: " + " → ".join([*path, n]))
            return
        if colour.get(n) == 2:
            return
        colour[n] = 1
        for m in sorted(edges.get(n, ())):
            walk(m, [*path, n])
        colour[n] = 2

    for n in sorted(edges):
        walk(n, [])
    return c


def check_no_namespace_init(owners) -> Check:
    c = Check("S4-namespace",
              "no `src/acr/__init__.py` in any distribution: `acr` is a PEP 420 namespace package")
    for p in sorted(ROOT.glob("src/acr/__init__.py")):
        c.examined += 1
        c.fails(f"{p.relative_to(ROOT)} exists — it shadows every sibling's subpackage, and the "
                f"failure presents as a missing module rather than as a conflict")
    if c.examined == 0:
        c.examined = 1  # the absence IS the observation; not inert
    return c


def check_cards_ship_with_their_readers(owners) -> Check:
    c = Check("S5-card-owner",
              "a method card ships in the distribution whose code or tests name it")
    skills = ROOT / "assets" / "skills"
    if not skills.is_dir():
        c.verdict = INERT
        c.findings.append("no assets/skills/ in this checkout")
        return c
    owner_of_card = {card: repo for repo, cards in CARD_OWNERS.items() for card in cards}
    for d in sorted(skills.iterdir()):
        if not (d / "SKILL.md").is_file():
            continue
        c.examined += 1
        card = d.name
        ships_in = owner_of_card.get(card, "acr-chart-review")
        named_by = set()
        for f, home in owners.items():
            if card in _string_literals(ROOT / f):
                named_by.add(home)
        if not named_by:
            continue  # named by no code: prose the model reads, and nothing to disagree with
        if ships_in not in named_by and not any(ships_in in REPOS[n]["deps"] for n in named_by):
            c.fails(f"card {card!r} ships in {ships_in} but is named only by "
                    f"{sorted(named_by)} — a card in the wrong distribution is guidance nobody "
                    f"receives")
    return c


def check_tests_route_somewhere(owners) -> Check:
    c = Check("S6-test-routing",
              "every test file routes to exactly one distribution or to the composer")
    tests = assign_tests()
    placed = {t for v in tests.values() for t in v}
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        c.examined += 1
        if p.name not in placed:
            c.fails(f"tests/{p.name}: routed nowhere")
    for name, files in tests.items():
        dupes = {f for f in files if sum(f in v for v in tests.values()) > 1}
        for d in sorted(dupes):
            c.fails(f"tests/{d}: routed to more than one distribution")
    return c


def check_declared_subjects_exist(owners) -> Check:
    c = Check("S7-manual-routes",
              "every hand-written test route and card owner names something that exists")
    for t, repo in sorted(TESTS_BY_SUBJECT.items()):
        c.examined += 1
        if not (ROOT / "tests" / t).is_file():
            c.fails(f"TESTS_BY_SUBJECT names tests/{t}, which does not exist")
        if repo != "(composer)" and repo not in REPOS:
            c.fails(f"TESTS_BY_SUBJECT routes tests/{t} to unknown distribution {repo!r}")
    for repo, cards in sorted(CARD_OWNERS.items()):
        if repo not in REPOS:
            c.fails(f"CARD_OWNERS names unknown distribution {repo!r}")
        for card in cards:
            c.examined += 1
            if not (ROOT / "assets" / "skills" / card).is_dir():
                c.fails(f"CARD_OWNERS gives {repo} the card {card!r}, which does not exist")
    return c


def check_skill_frontmatter(owners) -> Check:
    c = Check("S8-skill-frontmatter",
              "every skill declares a valid kind, a valid category if any, and a slot if prose")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from acr.contract import skill_invoke as si
        from acr.contract import skills as sk
    except ImportError as e:
        c.verdict = INERT
        c.findings.append(f"cannot import the skill loader here: {e}")
        return c
    skills = ROOT / "assets" / "skills"
    if not skills.is_dir():
        c.verdict = INERT
        c.findings.append("no assets/skills/ in this checkout")
        return c
    for d in sorted(skills.iterdir()):
        if not (d / "SKILL.md").is_file():
            continue
        c.examined += 1
        try:
            kind = si.skill_kind(d.name, skills)
            si.skill_category(d.name, skills)
            if kind == "prose":
                sk.skill_slot(d.name, skills)
            elif kind == "script" and not si._skills._frontmatter(d.name, skills).get("entry"):
                c.fails(f"{d.name}: kind: script with no `entry`")
        except Exception as e:  # noqa: BLE001 — the loader's own error IS the finding
            c.fails(f"{d.name}: {type(e).__name__}: {e}")
    return c


CHECKS = (check_every_module_is_assigned, check_no_undeclared_cross_repo_import,
          check_dependencies_are_acyclic, check_no_namespace_init,
          check_cards_ship_with_their_readers, check_tests_route_somewhere,
          check_declared_subjects_exist, check_skill_frontmatter)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    owners = _owners()
    results = [fn(owners) for fn in CHECKS]

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print(f"{'check':<24}{'verdict':<8}{'seen':>6}  claim")
        for r in results:
            print(f"{r.name:<24}{r.verdict:<8}{r.examined:>6}  {r.claim}")
            for f in r.findings:
                print(f"      {f}")
        bad = sum(1 for r in results if r.verdict == FAIL)
        inert = sum(1 for r in results if r.verdict == INERT)
        print(f"\n{len(results)} checks, {bad} failing, {inert} inert")
        if inert:
            print("An INERT check passed because it had nothing to look at, which is not the same "
                  "as the property holding. Find out why before reading it as good news.")
    return 1 if any(r.verdict == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
