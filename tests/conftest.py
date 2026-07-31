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

import pytest

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
