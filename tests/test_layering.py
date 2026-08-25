"""Which layer every module belongs to, and that a lower layer may not import a higher one.

Why this test exists before the directories do
----------------------------------------------
`src/acr/` was 73 flat modules, and a reader could not see which of them belonged to the chart
review agent, which to audit, which to evaluation, which to diagnosis, nor how they fit
together. Directories can say that, but the next person will reshuffle them, and after a single
move nothing stops `evals.py` from importing `agent.py` back the other way. So the layering was
written as an assertion first and the directories were moved second — the move is protected by
this test, not by the mover's memory.

A layer is not defined by "where the file sits", it is defined by **who it may depend on**. Two
rules:

    1. A lower layer must not import a higher one. Within a layer there is no restriction.
    2. Work planes may meet each other only through the types of a shared layer — that is,
       through "the shape of the input and the output". Importing the other one's functions is
       code coupling, even when the direction is legal.

Ten layers, bottom to top. The first three are the shared I/O contract, the rest are work planes:

  0 core         The common things that stay stable across tasks: AssetRef/Trajectory/
                 SignalEnvelope, the local artifact boundary, the model client, spend, state,
                 the module protocol. No domain semantics at all.
  0 chartstore   Chart data access. Same rank as kernel but named apart: kernel is the abstract
                 vocabulary, this layer is "how one patient's documents get read out", and all
                 three planes have to read.
  1 contract     The task contract and its vocabulary: spec, the answer contract, field format
                 checks, the guideline's three-valued logic, skill assembly, the rule catalog,
                 the layering declaration, value-domain code tables. "What this answer must
                 mean" lives here.
  2 review       The chart review agent itself: orchestration, the in-request hard controls, the
                 coverage policy, the tool surface, manifest serialisation. The only layer that
                 can produce an answer.
  3 audit        The safety/boundary evidence chain. Finding → Incident. Takes no TruthContext.
  3 evaluation   Quality assessment. Truth mode is a parameter, not a premise.
  3 diagnosis    Causal attribution. Bound to an explicit target event, explaining that one
                 error.
                 These three share a rank: they are three kinds of conclusion over the same
                 trajectory that cannot substitute for one another, and they must not depend on
                 each other — audit not being allowed to import evaluation is the executable
                 form of the sentence "audit is not an alias for a CODE evaluator".
  4 improvement  Repair routing, asset tuning, labelling.
  4 authoring    Onboarding for a new task or a new variable, and static checks on a spec. A
                 development plane, not on the path of a chart run.
  5 usecase      Knowledge specific to **one particular** use case. The cancer registry is one
                 of them, not the framework.
  6 cli          The entry point. May depend on any layer, may not be depended on by any layer.

`usecase` sits below cli and above everything else because a use case belongs at the edge: the
framework importing it is the framework being tied to that one use case.

Both registries are empty right now: `KNOWN_DOMAIN_COUPLING` (the framework depending on a use
case) and `KNOWN_DIRECT_COUPLING` (a direct import between work planes). They are empty and kept
anyway, because the rules are still here — the lists may only get shorter, and they are the place
where the next person who wants to break a rule is forced to write down why.
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "acr"

#: (rank, layer name, modules). Equal rank = same layer, no restriction between them; a smaller
#: rank may not import a larger one.
LAYERS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    # The move is done: every module lives in its own plane's directory and the layer is decided
    # by the **path**, so all of these lists are empty. Keeping the third slot of the tuple is
    # not a formality: a newly added top-level module — one whose plane has not been decided yet
    # — lands here, and `test_every_module_is_assigned_to_a_layer` demands that it be placed
    # explicitly.
    #
    # The directory is named `core` and not `kernel`: a package would shadow the `kernel.py` of
    # the same name, and that file is this layer's core vocabulary, which should not be renamed
    # for the sake of a directory name. `commands` and not `cli`: same reason, and
    # `acr.commands.cli:app` is the console entry point.
    (0, "core", ()),
    (0, "chartstore", ()),
    (1, "contract", ()),
    (2, "review", ()),
    # The codex-harness MVP path (docs/MVP_CODEX_DESIGN.md): same rank as review because it is
    # the same kind of thing — a plane that produces an answer — living beside the old runtime
    # until the MVP is green and the removal doc retires review/.
    (2, "mvp", ()),
    (3, "audit", ()),
    (3, "evaluation", ()),
    (3, "diagnosis", ()),
    (4, "improvement", ()),
    (4, "authoring", ()),
    (5, "usecase", ()),
)

#: Edges where the framework depends back on one particular use case. Each entry names the piece
#: of work that will delete it. This list may only get shorter — adding a line is welding the
#: framework onto the tumour registry one more time.
#:
#: Empty now. There used to be three — `spec` / `agent` / `run_manifest` each importing `icdo3` —
#: and action C deleted all three at once: `icdo3.py` was replaced by the domain-neutral
#: `code_tables.py`, and the three hard-coded axes became the `axes:` declared in
#: `assets/codes/*.yaml`. This empty dict is kept rather than deleted because the rule is still
#: here: the next time someone makes the framework import a use case module,
#: `test_no_layer_imports_a_higher_one` goes red, and this is where they are forced to write down
#: why.
KNOWN_DOMAIN_COUPLING: dict[tuple[str, str], str] = {}


#: The two layers that may be shared across planes. What they carry is exactly "the shape of the
#: input and the output": `core` is AssetRef / Trajectory / SignalEnvelope, `contract` is the spec
#: and its vocabulary. A work plane knowing another plane's product through these two layers is a
#: design requirement; importing the other plane's functions is code coupling.
SHARED_LAYERS = ("core", "chartstore", "contract")

#: The work planes. They may meet each other **only** through the types of SHARED_LAYERS and
#: through artifacts written to disk.
WORK_LAYERS = ("review", "mvp", "audit", "evaluation", "diagnosis", "improvement", "authoring",
               "usecase")

#: The direct couplings that still exist today, and the action that deletes each one. The key is
#: (source module, target module). Same as KNOWN_DOMAIN_COUPLING: it may only get shorter.
KNOWN_DIRECT_COUPLING: dict[tuple[str, str], str] = {}


def _modules() -> dict[str, pathlib.Path]:
    return {str(p.relative_to(SRC).with_suffix("")).replace("/", "."): p
            for p in SRC.rglob("*.py") if p.name != "__init__.py"}


#: Layer name -> rank. The directory move is in progress, so a module's layer has two sources, in
#: this order:
#:
#:   1. Its path — `acr/diagnosis/attribution.py` is in the diagnosis layer, nobody has to
#:      register it;
#:   2. The explicit lists in `LAYERS` — the modules not yet moved into a plane directory.
#:
#: When the move finishes the module lists in `LAYERS` all go empty, and this rank table stays: it
#: is the ordering the one rule needs, and it is the authoritative source for the directory names.
#: A move done in slices therefore does not need to edit this file in every slice.
PLANE_RANK: dict[str, int] = {name: r for r, name, _ in LAYERS} | {"commands": 6}


def _layer_of() -> tuple[dict[str, str], dict[str, int]]:
    plane, rank = {}, {}
    for r, name, names in LAYERS:
        for n in names:
            assert n not in plane, f"{n} appears twice in LAYERS"
            plane[n], rank[n] = name, r
    for m in _modules():                    # Path wins: once moved into a plane dir, the dir rules
        head = m.split(".")[0]
        if head in PLANE_RANK:
            plane[m], rank[m] = head, PLANE_RANK[head]
    return plane, rank


def _edges() -> dict[str, set[str]]:
    """Module -> the other acr modules it imports. A deferred import inside a function body counts.

    Deferred imports have to count: `run_manifest` is exactly the module that imports
    `document_concepts` from inside a function, and that is precisely the kind of dependency the
    layering is there to govern. Looking only at module-level imports would miss the whole class of
    them and then report that everything is fine.
    """
    mods = _modules()
    out: dict[str, set[str]] = collections.defaultdict(set)
    for name, path in mods.items():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            target = (node.module if node.level
                      else node.module[4:] if node.module.startswith("acr.") else None)
            if target in mods and target != name:
                out[name].add(target)
    return out


def test_every_module_is_assigned_to_a_layer():
    """A module with no layer is a module nobody can say which plane it is in — exactly the state
    this work exists to remove.

    Adding a module means deciding which layer it is in at the same time. That decision is written
    down in LAYERS, not left for the next person reading the directory tree to guess.
    """
    plane, _ = _layer_of()
    mods = set(_modules())
    unassigned = mods - set(plane)
    assert not unassigned, f"not assigned to a layer: {sorted(unassigned)}"
    stale = set(plane) - mods
    assert not stale, f"modules listed in LAYERS no longer exist: {sorted(stale)}"


def test_no_layer_imports_a_higher_one():
    """The one rule. Every violation of it has to be registered in KNOWN_DOMAIN_COUPLING first."""
    plane, rank = _layer_of()
    # cli is the entry point: its rank is above everything, and it may not be depended on — the
    # latter half is asserted separately by the next test.
    bad = []
    for src, targets in _edges().items():
        for dst in targets:
            if rank[src] < rank[dst] and (src, dst) not in KNOWN_DOMAIN_COUPLING:
                bad.append(f"{plane[src]}/{src} -> {plane[dst]}/{dst}")
    assert not bad, (
        "a lower layer imported a higher one:\n  " + "\n  ".join(sorted(bad))
        + "\n\nEither this module is in the wrong layer, or this is a real inversion. Do not make "
          "it go away by adding a line to KNOWN_DOMAIN_COUPLING — that list only takes usecase "
          "couplings.")


def test_nothing_depends_on_the_cli():
    """The CLI is the entry point. Any layer that imports it makes that layer's behaviour depend on
    whether someone came in from the command line."""
    def is_cli(m: str) -> bool:
        return m.startswith("commands.")

    offenders = [f"{s} -> {t}" for s, ts in _edges().items() for t in ts
                 if is_cli(t) and not is_cli(s)]
    assert not offenders, f"non-CLI modules depend on the CLI: {sorted(offenders)}"


@pytest.mark.parametrize("a,b", [("audit", "evaluation"), ("audit", "diagnosis"),
                                 ("evaluation", "audit"), ("evaluation", "diagnosis"),
                                 ("diagnosis", "audit")])
def test_the_three_post_run_planes_do_not_depend_on_each_other(a: str, b: str):
    """The three kinds of conclusion cannot substitute for one another, so they cannot depend on one
    another either.

    `diagnosis -> evaluation` is absent from the parameter list, and it is the one direction that is
    allowed: attribution has to read the deterministic scorer in order to know which error it is
    explaining (the README's own words are "ask the scorer"), and that dependency is a design
    requirement. The reverse is not: the moment evaluation depends on diagnosis, "is this correct"
    starts waiting on a model's opinion.
    """
    plane, _ = _layer_of()
    by_plane = {p: {m for m, pl in plane.items() if pl == p} for p in (a, b)}
    edges = _edges()
    offenders = [f"{s} -> {t}" for s in by_plane[a] for t in edges.get(s, ())
                 if t in by_plane[b]]
    assert not offenders, f"{a} depends on {b}: {sorted(offenders)}"


def test_work_planes_touch_each_other_only_through_the_io_contract():
    """The criterion for an independent module: no direct import between planes, only shared
    kernel/contract types.

    This is stricter than "a lower layer may not import a higher one". Layering only says the
    direction of a dependency is legal, it says nothing about the form of the coupling being legal:
    `derive -> coverage [assign_strata]` has the direction right, but it means that changing one
    function signature in coverage reaches into the improvement plane. Whereas what `audit`,
    `evaluation` and `diagnosis` should be reading is the same Trajectory and the same
    SignalEnvelope — that is a shape, not a function.

    Every exception has to be registered, naming the action that deletes it. The list may only get
    shorter.
    """
    plane, _ = _layer_of()
    work = set(WORK_LAYERS)
    bad = []
    for src, targets in _edges().items():
        if plane.get(src) not in work:
            continue
        for dst in targets:
            dp = plane.get(dst)
            if dp in SHARED_LAYERS or dp == plane[src] or dp is None:
                continue
            if (src, dst) not in KNOWN_DIRECT_COUPLING:
                bad.append(f"{plane[src]}/{src} -> {dp}/{dst}")
    assert not bad, (
        "a new direct coupling appeared between work planes:\n  " + "\n  ".join(sorted(bad))
        + "\n\nPlanes should share only kernel/contract types and artifacts written to disk. "
          "Either move the depended-on thing down into a shared layer, or make the caller read "
          "the product instead of calling the function.")


def test_the_direct_coupling_list_only_shrinks():
    """A registered coupling has to still be there. Fixing one without deleting its entry lets the
    next identical coupling ride in on it."""
    edges = _edges()
    gone = [f"{s} -> {t}" for (s, t) in KNOWN_DIRECT_COUPLING if t not in edges.get(s, ())]
    assert not gone, (
        f"these couplings are gone, delete them from KNOWN_DIRECT_COUPLING: {sorted(gone)}")


def test_the_declared_work_and_shared_layers_are_real():
    """Guards the two tests above: a mistyped layer name silently turns a whole assertion into an
    empty check.

    `plane.get(src) not in work` is always true for a misspelled layer name, so the test passes
    having checked nothing.
    """
    declared = {name for _, name, _ in LAYERS} | {"commands"}
    for name in SHARED_LAYERS + WORK_LAYERS:
        assert name in declared, f"{name!r} is not a layer name in LAYERS"
    assert not set(SHARED_LAYERS) & set(WORK_LAYERS)
    assert declared == set(SHARED_LAYERS) | set(WORK_LAYERS) | {"commands"}, (
        "a layer is neither shared nor a work plane — which of the two it is has to be decided "
        "explicitly")


def test_the_domain_coupling_list_only_shrinks():
    """Every registered entry has to genuinely still be there — once it is fixed it comes off the
    list.

    An exemption describing a coupling that disappeared long ago lets the next identical coupling
    ride in quietly.
    """
    edges = _edges()
    gone = [f"{s} -> {t}" for (s, t) in KNOWN_DOMAIN_COUPLING if t not in edges.get(s, ())]
    assert not gone, (
        f"these couplings no longer exist, delete them from KNOWN_DOMAIN_COUPLING: {sorted(gone)}")


#: Clinical words. Forbidden in **executable code** only, not in prose — see the note on
#: `_clinical_hits`.
CLINICAL_WORDS = ("histolog", "topograph", "morpholog", "icdo", "icd-o", "tumour", "tumor",
                  "carcinom", "oncolog", "biopsy", "primary_site", "seer", "ajcc")


def _clinical_hits(path: pathlib.Path) -> list[str]:
    """Clinical words occurring in executable code: identifier names, and string literals that are
    not docstrings.

    A whole-file grep was the first version of this test, and it went red in two places that were
    both false positives: `kernel.py:3`'s "deliberately knows nothing about tumour registries" is
    **declaring** neutrality, and `llm.py` has a `"pathology OR biopsy"` example inside a docstring.
    Neither drives any behaviour.

    This repository's standard for domain coupling is the one at `labelling.py:342` — a template
    may not **name** a disease, an organ, a document vocabulary or a coding system, because "every
    one of those words was a lie the moment the requirement moved". What lies is that name in the
    code, not the sentence explaining why it is not there. So check identifiers and live strings,
    and let docstrings and comments through (comments never reach the AST at all).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docstrings):
            names.append(node.value)
    hits = []
    for text in names:
        low = text.lower()
        for word in CLINICAL_WORDS:
            if word in low:
                hits.append(f"{word} in {text[:60]!r}")
    return hits


def test_the_core_layer_names_no_clinical_concept():
    """In the core layer, `kernel.py`'s docstring says it "deliberately knows nothing about tumour
    registries". Assert it.

    Only the core layer is checked, because it is the only layer that can be held to zero domain
    words: the contract layer legitimately carries field names, and the review layer's prompts
    carry document types. A complete domain-neutrality check is a separate piece of work; this pins
    the innermost layer only — a narrow assertion that passes beats a broad one that can only pass
    on the back of an exemption list.
    """
    plane, _ = _layer_of()
    hits = {m: h for m, path in _modules().items()
            if plane.get(m) == "core" and (h := _clinical_hits(path))}
    assert not hits, f"clinical concepts in the executable code of the core layer: {hits}"


def test_the_clinical_word_check_can_actually_fail(tmp_path: pathlib.Path):
    """Guards the test above: a check that can only pass is not a check.

    Two assertions pointing in opposite directions — a name in the code has to be caught, and the
    same word in prose has to be let through. The first version's whole-file grep caught the latter,
    so "prose is let through" is pinned here too; otherwise, once the false positive was fixed,
    nothing would stop someone changing it back to a whole-file search.
    """
    bad = tmp_path / "bad.py"
    bad.write_text('def f(histology_code):\n    return {"primary_site": histology_code}\n',
                   encoding="utf-8")
    assert _clinical_hits(bad), "a clinical identifier in code has to be caught"

    prose = tmp_path / "prose.py"
    prose.write_text('"""This module knows nothing about tumour registries or biopsy reports."""\n'
                     'def f(x):\n    """Not about histology either."""\n    return x\n',
                     encoding="utf-8")
    assert not _clinical_hits(prose), "the same word inside a docstring has to be let through"
