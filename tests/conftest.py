"""Muzzle the provider for every test that has not explicitly asked for a model.

WHY THIS FILE EXISTS
--------------------
`test_a_crashed_run_is_distinguishable_from_one_that_never_happened` patched
`cli_common.llm_client` while `extract` reaches `cli_common.chat_model`. Nothing was muzzled,
`.env` at the repo root supplied real credentials, and the test made two real paid calls on
every run — 29,216 prompt tokens and 371 completion tokens — before failing on an assertion
about the crash stub it thought it had caused.

That it was RED is not the danger. The danger is the same mistake in a test that happens to be
GREEN: a unit test that dials out and one that is correctly muzzled are indistinguishable from
the exit code. The difference shows up in the bill and on the network, which is to say nowhere
a test run reports.

So the default is inverted. Reaching a provider seam without patching it now RAISES, and a test
that wants a model declares that by patching — one line, at the seam, visible in the diff.

WHY BOTH NAMES
--------------
`llm_client` and `chat_model` are two constructors for two runtimes, and their own docstrings
each claim to be "THE provider seam". They are both seams, and patching one leaves the other
free to dial out — which is exactly the accident above. `tests/test_provider_is_muzzled.py`
asserts this list matches what `cli_common` actually exposes, so a third constructor cannot be
added quietly.

THE OPT-OUT, AND WHY IT IS NOT A HOLE
-------------------------------------
Constructing a client is not calling a model, and three tests legitimately do the first without
the second: `chat_model`'s own unit test (which fakes `ChatOpenAI` underneath it and then checks
that the model prefix was adapted), and two `signal --kind judge` tests where the client is built
before the fence refuses the dimension. Marking them `@pytest.mark.provider_seam` is the
declaration this guard is asking for — the point was never "no test may touch the seam", it was
"no test may touch it silently". A marker is one visible word in the diff.

WHAT THIS IS NOT
----------------
It is not a network sandbox. A module that constructs its own client inline, without going
through `cli_common`, walks straight past this — and that is precisely the mistake
`chat_model`'s docstring records having been made once already. The guard is worth having
anyway: it closes the path everything is supposed to use, and it fails loudly rather than
silently spending.
"""
from __future__ import annotations

import importlib
import os

import pytest

# --------------------------------------------------------------- the site's identifier shape
# `acr.core.site` ships NO default person-id pattern, on purpose: three defaults were tried and
# each was measured wrong somewhere nobody had looked (see that module's docstring). The runtime
# guards are therefore INERT until a deployment declares a shape — which would silently disarm
# every test that asserts a real-looking identifier is REFUSED or MASKED.
#
# So the test session declares one. It is set here rather than per-test because the alternative is
# sixteen tests each remembering to, and a test that forgets does not fail: it passes while
# measuring nothing. Wide on purpose, matching the runtime setting's error cost — a false negative
# here would mean a guard test that cannot see the thing it guards against.
#
# `ACR_PHI_SCAN_PATTERN` is deliberately NOT set. It is the other half of the split, and the byte
# scan over the tree wants the opposite error cost: with a pattern this wide it flags 203 content
# hashes under `assets/`. `tests/test_no_phi_in_tree.py` skips when it is unset, and a deployment
# holding real data is refused unless it sets both — see `site.require_person_id_pattern`.
os.environ.setdefault("ACR_PERSON_ID_PATTERN", r"(?<![\d.])\d{10,}")

import acr.core.site as _site

importlib.reload(_site)


#: The provider constructors on `acr.core.cli_common`. Kept here rather than imported so that
#: deleting one from the guard is a visible edit to the guard.
PROVIDER_SEAMS = ("llm_client", "chat_model")


def _refuse(name: str):
    def seam(*args, **kwargs):
        raise RuntimeError(
            f"provider seam `cli_common.{name}` was reached by a test that did not patch it. "
            f"A test that needs a model must say so: "
            f"monkeypatch.setattr('acr.core.cli_common.{name}', ...). Note that `llm_client` "
            f"and `chat_model` are DIFFERENT runtimes — patching one does not muzzle the "
            f"other, which is how this guard came to exist.")
    return seam


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "provider_seam: this test reaches a cli_common provider constructor ON PURPOSE and "
        "does not call a model through it. Says so out loud instead of being muzzled.")


@pytest.fixture(autouse=True)
def _muzzle_provider(request, monkeypatch):
    """Applied to every test. A test's own `monkeypatch.setattr` on the same name wins, because
    it runs after this fixture and both are undone at teardown."""
    if request.node.get_closest_marker("provider_seam"):
        return
    for name in PROVIDER_SEAMS:
        monkeypatch.setattr(f"acr.core.cli_common.{name}", _refuse(name))



# ------------------------------------------------- fixtures that live in a sibling repository
#
# After the 2026-08-03 split the corpus lives in `acr-corpus` and the contracts and method cards in
# `acr-chart-review`, so a repository's own suite may reach for a directory that is simply not
# checked out. `acr.core.site._resolve` raises `FileNotFoundError` naming the variable to set and the
# sibling to clone; this converts exactly that into a SKIP carrying the same message.
#
# A HOOKWRAPPER, not a plain hook. The first version of this called `item.runtest()` itself, which
# bypasses pytest's own call protocol: the monorepo suite went from 2051 passing to 65 failing,
# because the test body ran outside the machinery that manages its fixtures. A wrapper yields and
# then replaces the recorded exception, which is the only way to reclassify an outcome without
# taking over the execution.
#
# NARROW ON PURPOSE: it matches only the sentence `_resolve` produces. A genuinely missing file
# inside this repository still FAILS, because "an asset that was supposed to travel and did not" is
# the case this project most needs to hear about.
#: The three phases a resolver failure can surface in. `setup` is where a fixture that builds a
#: corpus path fails; `call` is where a test body does. Registering the wrapper for both is not
#: belt-and-braces — an error in setup is reported as ERROR and never reaches `call`, which is why
#: the isolated suites still showed `258 errors` after the call hook alone was in place.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    yield from _reclassify()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield from _reclassify()


def _reclassify():
    """Turn the resolver's own FileNotFoundError into a skip, wherever it is raised.

    A generator so both phase hooks share one body. Narrow on purpose: it matches only the sentence
    `acr.core.site._resolve` and `skill_roots` produce, so a genuinely missing file inside this
    repository still fails — "an asset that was supposed to travel and did not" is the case this
    project most needs to hear about rather than skip.
    """
    outcome = yield
    exc = outcome.excinfo[1] if outcome.excinfo else None
    if isinstance(exc, FileNotFoundError) and (
            "Set ACR_" in str(exc) or "does not exist (" in str(exc)
            or "cannot find the method cards" in str(exc)):
        outcome.force_exception(
            pytest.skip.Exception(f"fixture lives in a sibling repository: {exc}",
                                  _use_item_location=True))


#: Which shared fixtures this checkout can actually reach, computed ONCE. `None` means unavailable.
def _fixture_availability():
    from acr.core import site
    out = {}
    for name, fn in (("specs", site.specs_root), ("corpus", site.corpus_root),
                     ("skills", site.skill_roots)):
        try:
            fn(); out[name] = True
        except FileNotFoundError:
            out[name] = False
    return out


_AVAILABLE = _fixture_availability()

#: Source-text markers for each fixture. A module that mentions the resolver needs the fixture; a
#: grep is enough and, unlike importing, it works before the failure it is trying to avoid.
_NEEDS = {"specs": ("specs_root",), "corpus": ("corpus_root",), "skills": ("skill_roots", "skills_root")}


def pytest_ignore_collect(collection_path, config):
    """Skip a test module BEFORE importing it when the fixture it needs is in an absent sibling.

    WHY NOT A REPORT HOOK. Two earlier attempts reclassified the outcome after the fact —
    `pytest_runtest_call`, which never runs because the failure is at MODULE level during import,
    and `pytest_collectreport`, which does mark the report skipped but leaves pytest's own error
    count intact, so a nine-repo suite still ended `Interrupted: 272 errors during collection`. A
    collection error is fatal by design and the only way not to have one is not to import.

    So the decision is made from the SOURCE TEXT: a module that names `specs_root` needs the
    contracts. That is a grep, it needs no import, and it is wrong only in the harmless direction —
    a module that mentions a resolver in a comment gets skipped when the fixture is missing, which
    it would have been anyway.
    """
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    try:
        text = collection_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for fixture, markers in _NEEDS.items():
        if not _AVAILABLE.get(fixture, True) and any(m in text for m in markers):
            return True
    return None
