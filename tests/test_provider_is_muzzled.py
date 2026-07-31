"""没有哪个测试可以在没明说的情况下打真实 provider。

为什么存在
--------
`test_a_crashed_run_is_distinguishable_from_one_that_never_happened` 打的是
`cli_common.llm_client`，而 `extract` 走的是 `cli_common.chat_model`。两个不同的构造函数，
于是那次"模拟 provider 崩溃"从未发生：仓库根的 `.env` 提供了真凭证，这条测试**每跑一次就
真实调用两次模型**（29,216 prompt tokens + 371 completion），最后因为撞上 MODEL_CALL_LIMIT
而写出一份 manifest，测试再去找崩溃 stub、找不到、报红。

它红了一年也无妨；危险的是它**如果碰巧变绿**。一次拨号成功的单元测试，和一次被正确 muzzle
的单元测试，从退出码上看一模一样 —— 差别只在账单和网络里。这正是这个仓库反复点名的那种
"不会失败的检查"，只是方向反过来：一个本该拦住 provider 的机制，失效时看起来像通过。

守卫的形状
--------
`conftest.py` 里一条 autouse fixture 把两个 provider 接缝都换成会抛的桩。想调模型的测试必须
自己 patch —— 那一行 patch 就是它的声明。守卫本身也必须能失败，所以下面两条测试一条断言
"没 patch 就抛"，另一条断言"patch 之后照常работа"，否则一个失效的守卫会一直绿着。
"""
from __future__ import annotations

import pytest

from acr.core import cli_common


def test_an_unpatched_provider_seam_raises_instead_of_dialling_out():
    """两个接缝都要被堵。只堵一个正是这条测试要防的那次事故。"""
    for seam in ("llm_client", "chat_model"):
        with pytest.raises(RuntimeError, match="provider seam"):
            getattr(cli_common, seam)("some-model", None)


def test_a_test_that_patches_the_seam_still_gets_its_own_stub(monkeypatch):
    """守卫不能挡住正当用法，否则每个要驱动模型的测试都得跟它搏斗。

    这也是守卫**自己**的失败模式：一个把所有人都挡住的 fixture 会被下一个人删掉。
    """
    sentinel = object()
    monkeypatch.setattr("acr.core.cli_common.chat_model", lambda *a, **k: sentinel)
    assert cli_common.chat_model("m", None) is sentinel


def test_the_guard_names_both_seams_so_a_third_one_cannot_be_added_quietly():
    """接缝的清单住在 conftest 里，而这条断言它和 `cli_common` 实际暴露的对得上。

    加第三个 provider 构造函数而不告诉守卫，就是重新打开这扇门 —— 而且是静默打开。
    """
    from conftest import PROVIDER_SEAMS
    for seam in PROVIDER_SEAMS:
        assert callable(getattr(cli_common, seam)), f"{seam} is not a callable on cli_common"
    documented = {n for n in dir(cli_common)
                  if n in ("llm_client", "chat_model")}
    assert set(PROVIDER_SEAMS) == documented, (
        "cli_common exposes a provider constructor the guard does not muzzle")
