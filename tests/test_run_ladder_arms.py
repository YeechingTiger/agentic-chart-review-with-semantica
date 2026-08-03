"""每一条实验臂在花第一分钱之前就必须装得起来。

这条测试存在的原因是一次真实的事故,而且是同一类事故的第二次。Package C(02d8c21)把一个
`search` 槽拆成 `controller` + `tactic` 并给每张卡改了名;`tools/run_ladder.py` 收到的是
`s/search-/controller-/`,别的什么都没改。结果七条臂里有五条点名的卡根本不存在
—— `controller-native`、`controller-breadth-first`、`controller-depth-first`、
`controller-breadth-then-depth`、`controller-latest-first`。跑起来会怎样:前两条臂正常花钱,
第三条 `SkillStack.validate` 抛异常。

一个只在运行时才发现自己名字写错的驱动脚本,是一个必须先付钱才能报错的驱动脚本。所以校验
搬到了 `preflight()`,而这条测试在 CI 里跑它。

`tools/` 不在包里,所以按路径加载。这是这棵树上第一条伸进 `tools/` 的测试,理由是这里的
内容——哪些臂会被跑——是实验设计的一部分,不是脚手架。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from acr.contract.skills import SkillError

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ladder():
    return _load("run_ladder")


@pytest.mark.parametrize("group", ["controller", "tactic", "experience", "all"])
def test_every_arm_assembles_and_validates(ladder, group):
    """点名一张不存在的卡、或者把卡装错槽,都在这里炸,而不是在第三条臂上炸。"""
    resolved = ladder.preflight(ladder.arms_for(group))
    assert resolved, group
    for name, stack in resolved:
        assert isinstance(name, str) and name
        stack.validate()          # 冗余但便宜:preflight 若哪天不再校验,这里还在


def test_the_baseline_arm_is_in_every_group(ladder):
    """每一组都得有自己的地板,否则组间比较是在拿两个不同的基线做减法。"""
    for group in ladder.GROUPS:
        assert "B0-base" in {n for n, _ in ladder.arms_for(group)}, group


def test_all_does_not_run_the_baseline_three_times(ladder):
    names = [n for n, _ in ladder.arms_for("all")]
    assert len(names) == len(set(names))


def test_a_card_that_does_not_exist_still_fails_loudly(ladder):
    """把守卫本身证伪一次:preflight 不是一个恒真的函数。"""
    with pytest.raises(SkillError):
        ladder.preflight([("bogus", "controller=controller-native")])


def test_the_controller_arms_are_every_controller_card_in_the_tree(ladder):
    """一张树上有、梯子上没有的 controller 卡,是一个不会被测到的干预。

    原来这条测试把名单钉成两张卡,理由是"第三张出现时要有人说明它是什么"。第三张出现了
    (`controller-hypothesis-set`,2026-08-03,替代被删掉的候选账本机制),而钉死名单意味着
    加一张卡就要改一次测试 —— 于是这条测试量的是我记不记得改它。改成对着树:少一张就是漏测。
    """
    on_disk = {p.name for p in (ROOT / "assets" / "skills").iterdir()
               if p.name.startswith("controller-")}
    named = {s.controller for _, s in ladder.preflight(ladder.CONTROLLER_ARMS)} - {None}
    assert named == on_disk, "controller 卡和梯子不一致"


def test_the_tactic_arms_cover_every_tactic_card_in_the_tree():
    """一张树上有、梯子上没有的战术卡,是一个不会被测到的干预。"""
    ladder = _load("run_ladder")
    on_disk = {p.name for p in (ROOT / "assets" / "skills").iterdir()
               if p.name.startswith("tactic-")}
    assert set(ladder._TACTICS) == on_disk, "the tactic ladder and the tree disagree"
