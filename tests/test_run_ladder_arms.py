"""每一条实验臂在花第一分钱之前就必须装得起来。

这条测试存在的原因是一次真实的事故,而且是同一类事故的第二次。Package C(02d8c21)把一个
`search` 槽拆成 `controller` + `tactic` 并给每张卡改了名;`tools/run_ladder.py` 收到的是
`s/search-/controller-/`,别的什么都没改。结果七条臂里有五条点名的卡根本不存在
—— `controller-native`、`controller-breadth-first`、`controller-depth-first`、
`controller-breadth-then-depth`、`controller-latest-first`。(槽在 2026-08-03 又从
`controller` 改叫 `policy`;上面这些名字是事故当时的原样,不随之改写。)跑起来会怎样:前两条臂正常花钱,
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
        ladder.preflight([("bogus", "policy=policy-native")])


def test_the_policy_arms_are_every_policy_card_in_the_tree(ladder):
    """一张树上有、梯子上没有的 policy 卡,是一个不会被测到的干预。

    原来这条测试把名单钉成两张卡,理由是"第三张出现时要有人说明它是什么"。第三张出现了
    (`policy-hypothesis-set`,2026-08-03,替代被删掉的候选账本机制),而钉死名单意味着
    加一张卡就要改一次测试 —— 于是这条测试量的是我记不记得改它。改成对着树:少一张就是漏测。
    """
    on_disk = {p.name for p in (site.skills_root()).iterdir()
               if p.name.startswith("policy-")}
    named = {s.policy for _, s in ladder.preflight(ladder.POLICY_ARMS)} - {None}
    assert named == on_disk, "policy 卡和梯子不一致"


def test_the_tactic_arms_cover_every_tactic_card_in_the_tree():
    """一张树上有、梯子上没有的战术卡,是一个不会被测到的干预。"""
    ladder = _load("run_ladder")
    on_disk = {p.name for p in (site.skills_root()).iterdir()
               if p.name.startswith("tactic-")}
    assert set(ladder._TACTICS) == on_disk, "the tactic ladder and the tree disagree"


# ============================================================ 花钱之前得先说清楚花多少

@pytest.fixture(scope="module")
def budget():
    return _load("_driver_budget")


def test_the_ceiling_is_reported_in_the_unit_it_actually_is(budget):
    """`--max-usd` 是 `acr batch` 的 **单次运行** 上限,梯子把它印成 `$3.00/arm`。

    27 张图一条臂,所以那条 `--dry-run` 的行把最坏情况少报了 27 倍。这不是排版问题:
    `--dry-run` 存在的唯一理由就是"先看要花多少再决定跑不跑",而它给出的数字是真实上限的 1/27。
    """
    r = budget.budget_report(n_arms=7, n_charts=27, max_usd_per_run=3.0)
    assert r["runs"] == 189
    assert r["per_run_ceiling_usd"] == 3.0
    assert r["worst_case_usd"] == 567.0, "7 臂 × 27 图 × $3 单次上限"


def test_the_line_names_both_numbers_and_which_is_which(budget):
    """只印总额会被读成"已经定了要花这么多";只印单次上限就是原来的缺陷。两个都印,并说明单位。"""
    line = budget.budget_line(budget.budget_report(n_arms=2, n_charts=18, max_usd_per_run=3.0))
    assert "$3.00" in line and "per run" in line
    assert "$108.00" in line and "worst case" in line
    assert "36 run" in line


def test_a_single_run_is_not_pluralised_into_a_worse_number(budget):
    r = budget.budget_report(n_arms=1, n_charts=1, max_usd_per_run=0.5)
    assert r["worst_case_usd"] == 0.5


def test_both_drivers_print_the_bound_not_the_per_run_ceiling(ladder, capsys, monkeypatch):
    """两个脚本各自算一遍,就是两个对"这要花多少"的答案,而其中一个已经错过一次。

    看的是 **输出**,不是源码。第一版这条测试断言源码里不出现 `/arm` —— 结果它被自己那段
    解释旧缺陷的注释绊倒了。一个对源码做子串匹配的守卫,会把对缺陷的记述当成缺陷本身。
    """
    # `run_floor` 返回 2(它对着 STORE.390 拒绝一条 INERT 的 planner 轴),`run_ladder` 返回 0。
    # 这条测试问的是预算那一行,所以只看输出,不看返回码 —— 返回码有它自己的测试。
    for mod in (_load("run_floor"), ladder):
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
        mod.main(["--dry-run", "--max-usd", "3.0", "--patients", "SYN0001,SYN0002,SYN0003"])
        out = capsys.readouterr().out
        assert "$3.00 per run" in out, f"{mod.__name__} 没说清单位"
        assert "worst case" in out, f"{mod.__name__} 没给出上限"
        assert "/arm" not in out, f"{mod.__name__} 还在把单次上限印成每臂上限"


# ============================================================ --help 不能开始花钱

def test_run_floor_parses_its_arguments_before_doing_anything():
    """`run_floor.py` 没有 argparse。`python tools/run_floor.py --help` 会忽略这个参数,
    直接启动 36 次真实的 batch —— 一个想弄清楚脚本怎么用的人会为此付钱。

    返回 2 而不是 0,因为它当场发现了一件更要紧的事:见下一条。
    """
    floor = _load("run_floor")
    assert floor.main(["--dry-run"]) == 2


def test_run_floor_refuses_an_inert_planner_axis(capsys):
    """它的表头说"两条臂,一个变量:runtime profile" —— 而 STORE.390 一个 stratum 都没声明,
    所以 `plan_from_spec` 把每个类型都送进 `search`,和 `plan_from_patient_inventory` 一模一样。

    这条脚本的整个主张是"spec 手写的计划是一个值得被证伪的先验"。在这份契约上它不是变量:
    两条臂差的只有 coverage 政策和 spec 视图。$108 买不到关于计划的任何一个字,所以这里拒绝,
    而不是加一行警告 —— 警告会被读一次,然后变成输出格式的一部分。
    """
    floor = _load("run_floor")
    assert floor.main(["--dry-run"]) == 2
    out = capsys.readouterr().out
    assert "INERT" in out and "Refusing" in out


def test_run_floor_dry_run_spawns_no_subprocess(monkeypatch):
    """"不花钱"这件事要被证明,不是被声明。原来的脚本连一个可以打桩的边界都没有。"""
    floor = _load("run_floor")
    calls = []
    monkeypatch.setattr(floor.subprocess, "run",
                        lambda *a, **k: calls.append(a) or _Probe())
    # 在 INERT 拒绝之前就返回,所以连 `--help` 探针都还没跑到 —— 这正是要证明的:
    # 拒绝发生在花钱之前,而且发生在任何子进程之前。
    assert floor.main(["--dry-run"]) == 2
    assert calls == []


class _Probe:
    returncode = 0
    stdout = "--patients"
    stderr = ""


def test_run_floor_still_refuses_an_arm_whose_profile_does_not_exist():
    """把守卫证伪一次:preflight 不是恒真的。"""
    floor = _load("run_floor")
    with pytest.raises(Exception):
        floor.preflight([("bogus", "no-such-runtime-profile")])
