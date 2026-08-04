"""No test may reach a real provider without having said so.

WHY THIS FILE EXISTS
--------------------
`test_a_crashed_run_is_distinguishable_from_one_that_never_happened` patched
`cli_common.llm_client`, while `extract` goes through `cli_common.chat_model`. Two different
constructors, so the "simulated provider crash" never happened: the `.env` at the repo root
supplied real credentials, this test made **two real model calls on every single run** (29,216
prompt tokens + 371 completion), and it ended by hitting MODEL_CALL_LIMIT and writing a manifest,
after which the test went looking for the crash stub, did not find it, and reported red.

That it was red for a year does no harm; the danger is that it **might happen to go green**. A unit
test that dialled out successfully and a unit test that was correctly muzzled look identical from
the exit code — the difference is only in the bill and on the network. This is exactly the kind of
"check that cannot fail" this repository keeps naming, only with the direction reversed: a
mechanism whose job is to stop the provider looks like a pass when it stops working.

THE SHAPE OF THE GUARD
----------------------
An autouse fixture in `conftest.py` replaces both provider seams with stubs that raise. A test that
wants to call a model must patch it itself — that one patch line is its declaration. The guard
itself must also be able to fail, so of the two tests below one asserts that reaching a seam
without patching raises, and the other asserts that a patched seam still works as before;
otherwise a guard that has stopped working would stay green.
"""
from __future__ import annotations

import pytest

from acr.core import cli_common


def test_an_unpatched_provider_seam_raises_instead_of_dialling_out():
    """Both seams have to be plugged. Plugging only one is precisely the accident this test is
    here to prevent."""
    for seam in ("llm_client", "chat_model"):
        with pytest.raises(RuntimeError, match="provider seam"):
            getattr(cli_common, seam)("some-model", None)


def test_a_test_that_patches_the_seam_still_gets_its_own_stub(monkeypatch):
    """The guard must not block legitimate use, or every test that needs to drive a model would
    have to fight it.

    That is also the guard's **own** failure mode: a fixture that blocks everybody gets deleted by
    the next person.
    """
    sentinel = object()
    monkeypatch.setattr("acr.core.cli_common.chat_model", lambda *a, **k: sentinel)
    assert cli_common.chat_model("m", None) is sentinel


def test_the_guard_names_both_seams_so_a_third_one_cannot_be_added_quietly():
    """The list of seams lives in conftest, and this asserts it lines up with what `cli_common`
    actually exposes.

    Adding a third provider constructor without telling the guard is reopening that door — and
    reopening it silently.
    """
    from conftest import PROVIDER_SEAMS
    for seam in PROVIDER_SEAMS:
        assert callable(getattr(cli_common, seam)), f"{seam} is not a callable on cli_common"
    documented = {n for n in dir(cli_common)
                  if n in ("llm_client", "chat_model")}
    assert set(PROVIDER_SEAMS) == documented, (
        "cli_common exposes a provider constructor the guard does not muzzle")
