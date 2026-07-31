"""每个模块属于哪一层，以及低层不许 import 高层。

为什么这条测试先于目录存在
--------------------------
`src/acr/` 是 73 个平铺模块，读的人无法看出哪个属于 chart review agent、哪个属于 audit、
哪个属于 evaluation、哪个属于 diagnosis，也无法看出它们怎么合在一起。目录能表达这件事，
但目录会被下一个人重新打乱，而且一次搬迁之后没有任何东西阻止 `evals.py` 反过来 import
`agent.py`。所以先把分层写成断言，再搬目录 —— 搬迁由这条测试保护，而不是由搬迁者的记性。

层的定义不是"文件放在哪"，是**允许依赖谁**。规则只有一条：

    低层不得 import 高层。同层之间不限制。

九层，自下而上：

  0 kernel       跨任务稳定的公共物：AssetRef/Trajectory/SignalEnvelope、本地 artifact
                 边界、模型客户端、花费、状态、module protocol。不含任何领域语义。
  1 contract     任务合同与其词表：spec、答案契约、字段格式检查、指南三值逻辑、skill 装配、
                 rule catalog。"这个答案必须意味着什么"住在这里。
  2 review       chart review agent 本身：编排、请求内硬控制、coverage 策略、工具面、
                 manifest 序列化。唯一能产出答案的一层。
  3 audit        安全/边界证据链。Finding → Incident。不接收 TruthContext。
  3 evaluation   质量评价。truth mode 是参数而非前提。
  3 diagnosis    因果归因。绑定显式 target event，解释那一个错误。
                 这三层同 rank：它们是同一条 trajectory 上三种不能互相替代的结论类型，
                 彼此不得依赖 —— audit 不许 import evaluation 正是"audit 不是 CODE
                 evaluator 的别名"这句话的可执行形式。
  4 improvement  修复路由、资产调优、标注。
  4 authoring    新任务/新变量的 onboarding 与 spec 静态检查。开发面，不在跑病历的路上。
  5 usecase      **某一个** use case 特有的知识。癌症登记是其中一个，不是框架。
  6 cli          入口。可以依赖任何层，不许被任何层依赖。

`usecase` 排在 cli 之下、其余之上，是因为一个 use case 应该坐在边缘：框架 import 它，
就是框架被那个 use case 绑住了。今天恰好有三条这样的边，它们全部登记在
`KNOWN_DOMAIN_COUPLING` 里，每条都写明哪项工作会让它消失。这不是豁免清单 —— 清单只能变短。
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "acr"

#: (rank, 层名, 模块). rank 相同 = 同层，互相不限制；rank 小的不许 import rank 大的。
LAYERS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (0, "kernel", (
        "kernel", "local_artifacts", "llm", "spend", "state", "modules", "tool_surface",
        "cli_common", "usage_telemetry")),
    (1, "contract", (
        "spec", "answer_contract", "answer_checks", "concordance", "skills",
        "registry_catalog", "deps", "spec_repair", "trace",
        # 值域加载器。在 contract 层而不是 usecase 层，因为泛化之后它只知道"表有若干个有序
        # 的轴"—— 轴名、章节标题、码形正则和记法折叠规则全部在 `codes/*.yaml` 里。癌症的
        # 部分现在是纯资产，`tests/test_icdo3.py` 测那些资产。
        "code_tables")),
    (2, "review", (
        "agent", "answer_gate", "coverage", "coverage_planner", "plan_expansion",
        "conflict_refinement", "runtime_controls", "runtime_profiles", "document_concepts",
        "tools.toolbox", "corpus", "mcp_server", "run_triggers", "site_mapping",
        "run_manifest")),
    (3, "audit", ("audit_loop",)),
    (3, "evaluation", (
        "evals", "evaluation_modules", "evaluation_pipeline", "judge", "explain")),
    (3, "diagnosis", (
        "attribution", "attribution_modules.builtins", "attribution_modules.registry")),
    (4, "improvement", ("repair_loop", "refine", "assetdev", "labelling", "derive")),
    (4, "authoring", ("intake", "speclint")),
    (5, "usecase", (
        "specview.basis", "specview.decisions", "specview.measurements", "specview.prose",
        "specview.render", "specview.signoff", "specview.statements")),
)

#: 框架反过来依赖某一个 use case 的边。每条写明哪项工作会删掉它。
#: 这个清单只能变短 —— 新增一条就是把框架又焊死在肿瘤登记上一次。
#:
#: 空了。原先三条 —— `spec` / `agent` / `run_manifest` 各自 import `icdo3` —— 由动作 C
#: 一次删掉：`icdo3.py` 换成领域中立的 `code_tables.py`，三个写死的轴变成 `codes/*.yaml`
#: 里声明的 `axes:`。留着这个空 dict 而不是删掉它，是因为规则还在：下一次有人让框架
#: import 一个 use case 模块，`test_no_layer_imports_a_higher_one` 会红，而这里是他被迫
#: 写下理由的地方。
KNOWN_DOMAIN_COUPLING: dict[tuple[str, str], str] = {}


#: 允许被跨平面共享的两层。它们承载的正是"输入输出的形式"：`kernel` 是 AssetRef /
#: Trajectory / SignalEnvelope，`contract` 是 spec 与其词表。一个工作平面通过这两层认识
#: 另一个平面的产物，是设计要求；直接 import 对方的函数，是代码耦合。
SHARED_LAYERS = ("kernel", "contract")

#: 干活的平面。它们之间**只能**通过 SHARED_LAYERS 的类型和落盘的 artifact 相见。
WORK_LAYERS = ("review", "audit", "evaluation", "diagnosis", "improvement", "authoring",
               "usecase")

#: 今天还存在的直接耦合，以及删掉它的那个动作。键是 (源模块, 目标模块)。
#: 和 KNOWN_DOMAIN_COUPLING 一样：只能变短。
KNOWN_DIRECT_COUPLING: dict[tuple[str, str], str] = {
    # 动作 A —— `corpus` 不属于 review 平面。Corpus/DocMeta 是**病历数据访问**，
    # 三个平面都要读它；它现在被归进 review 只是因为 agent 是第一个用它的人。
    # 把它下移到共享层，这三条一起消失。
    ("speclint", "corpus"): "A: corpus 下移到共享的数据访问层",
    ("derive", "corpus"): "A: 同上",
    ("labelling", "corpus"): "A: 同上",
    # 动作 B —— `strata_from_spec` / `assign_strata` 是**从 spec 推导**出来的，不是运行时
    # 策略。它们住在 coverage 里，于是任何想按 spec 分层的模块都得 import 运行时平面。
    # 拆到 contract 之后这两条消失。
    ("assetdev", "coverage"): "B: strata_from_spec/assign_strata 拆到 contract",
    ("derive", "coverage"): "B: 同上",
    # 动作 C 已完成，原先 ("agent","icdo3") 和 ("run_manifest","icdo3") 两条在此，
    # 随 `icdo3.py` -> `code_tables.py` 一起消失：值域加载器现在在 contract 层，
    # 而 contract 是允许被共享的 I/O 契约层。
}


def _modules() -> dict[str, pathlib.Path]:
    return {str(p.relative_to(SRC).with_suffix("")).replace("/", "."): p
            for p in SRC.rglob("*.py") if p.name != "__init__.py"}


def _layer_of() -> tuple[dict[str, str], dict[str, int]]:
    plane, rank = {}, {}
    for r, name, names in LAYERS:
        for n in names:
            assert n not in plane, f"{n} 在 LAYERS 里出现了两次"
            plane[n], rank[n] = name, r
    return plane, rank


def _edges() -> dict[str, set[str]]:
    """模块 -> 它 import 的其他 acr 模块。函数体内的延迟 import 也算。

    延迟 import 必须算：`run_manifest` 就是在函数里 import `document_concepts` 的，而那
    正是分层要管的依赖。只看模块层的 import 会把这一类全部漏掉，然后报告一切正常。
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
    """没被分层的模块就是没人说得清它属于哪个平面 —— 那正是这次要消除的状态。

    新增一个模块必须同时决定它属于哪一层。这个决定写在 LAYERS 里，而不是留给下一个读
    目录的人猜。
    """
    plane, _ = _layer_of()
    mods = set(_modules())
    cli = {m for m in mods if m == "cli" or m.startswith("cli_")}
    unassigned = mods - set(plane) - cli
    assert not unassigned, f"未分层: {sorted(unassigned)}"
    stale = set(plane) - mods
    assert not stale, f"LAYERS 里的模块已不存在: {sorted(stale)}"


def test_no_layer_imports_a_higher_one():
    """唯一的规则。违反它的每一条都必须先登记在 KNOWN_DOMAIN_COUPLING 里。"""
    plane, rank = _layer_of()
    # cli 是入口，rank 高于一切，且不许被依赖 —— 后者由下一个测试单独断言。
    for m in _modules():
        if m == "cli" or m.startswith("cli_"):
            plane[m], rank[m] = "cli", 6
    bad = []
    for src, targets in _edges().items():
        for dst in targets:
            if rank[src] < rank[dst] and (src, dst) not in KNOWN_DOMAIN_COUPLING:
                bad.append(f"{plane[src]}/{src} -> {plane[dst]}/{dst}")
    assert not bad, (
        "低层 import 了高层:\n  " + "\n  ".join(sorted(bad))
        + "\n\n要么这个模块分错了层，要么这是一处真的倒挂。不要靠往 "
          "KNOWN_DOMAIN_COUPLING 加一行来消掉它 —— 那个清单只收 usecase 耦合。")


def test_nothing_depends_on_the_cli():
    """CLI 是入口。任何一层 import 它，就意味着那层的行为取决于有没有人从命令行进来。"""
    offenders = [f"{s} -> {t}" for s, ts in _edges().items() for t in ts
                 if (t == "cli" or t.startswith("cli_"))
                 and not (s == "cli" or s.startswith("cli_"))]
    assert not offenders, f"非 CLI 模块依赖 CLI: {sorted(offenders)}"


@pytest.mark.parametrize("a,b", [("audit", "evaluation"), ("audit", "diagnosis"),
                                 ("evaluation", "audit"), ("evaluation", "diagnosis"),
                                 ("diagnosis", "audit")])
def test_the_three_post_run_planes_do_not_depend_on_each_other(a: str, b: str):
    """三种结论类型不能互相替代，所以也不能互相依赖。

    `diagnosis -> evaluation` 不在参数表里，是唯一被允许的方向：归因需要读确定性评分器
    才能知道自己在解释哪个错误（README 的原话是"ask the scorer"），这是设计要求的依赖。
    反过来则不行：evaluation 一旦依赖 diagnosis，"是否正确"就会开始等一个模型的意见。
    """
    plane, _ = _layer_of()
    by_plane = {p: {m for m, pl in plane.items() if pl == p} for p in (a, b)}
    edges = _edges()
    offenders = [f"{s} -> {t}" for s in by_plane[a] for t in edges.get(s, ())
                 if t in by_plane[b]]
    assert not offenders, f"{a} 依赖了 {b}: {sorted(offenders)}"


def test_work_planes_touch_each_other_only_through_the_io_contract():
    """独立模块的判据：平面之间不许直接 import，只准共享 kernel/contract 的类型。

    这比"低层不许 import 高层"严。分层只说明依赖方向合法，不说明耦合方式合法：
    `derive -> coverage [assign_strata]` 方向没错，但它意味着改 coverage 的一个函数签名会
    动到 improvement 平面。而 `audit`、`evaluation`、`diagnosis` 读的应该是同一条
    Trajectory 和同一个 SignalEnvelope —— 那是形式，不是函数。

    每条例外都必须登记，并写明哪个动作删掉它。清单只能变短。
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
        "工作平面之间出现了新的直接耦合:\n  " + "\n  ".join(sorted(bad))
        + "\n\n平面之间只应共享 kernel/contract 的类型和落盘 artifact。要么把被依赖的东西"
          "下移到共享层，要么让调用方读产物而不是调函数。")


def test_the_direct_coupling_list_only_shrinks():
    """登记的耦合必须还在。修好了不删登记项，下一次同样的耦合会搭这个便车。"""
    edges = _edges()
    gone = [f"{s} -> {t}" for (s, t) in KNOWN_DIRECT_COUPLING if t not in edges.get(s, ())]
    assert not gone, f"这些耦合已消失，从 KNOWN_DIRECT_COUPLING 删掉: {sorted(gone)}"


def test_the_declared_work_and_shared_layers_are_real():
    """守卫上面两条：层名打错会让整条断言静默变成空检查。

    `plane.get(src) not in work` 对一个拼错的层名永远为真，于是测试通过而什么都没查。
    """
    declared = {name for _, name, _ in LAYERS} | {"cli"}
    for name in SHARED_LAYERS + WORK_LAYERS:
        assert name in declared, f"{name!r} 不是 LAYERS 里的层名"
    assert not set(SHARED_LAYERS) & set(WORK_LAYERS)
    assert declared == set(SHARED_LAYERS) | set(WORK_LAYERS) | {"cli"}, (
        "有层既不共享也不干活 —— 它属于哪一类必须被明确决定")


def test_the_domain_coupling_list_only_shrinks():
    """登记的每一条都必须是真的还在那里 —— 修好了就要从清单里删掉。

    一个描述早已消失的耦合的豁免项，会让下一次同样的耦合悄悄搭上便车。
    """
    edges = _edges()
    gone = [f"{s} -> {t}" for (s, t) in KNOWN_DOMAIN_COUPLING if t not in edges.get(s, ())]
    assert not gone, (
        f"这些耦合已经不存在了，从 KNOWN_DOMAIN_COUPLING 删掉: {sorted(gone)}")


#: 临床词。只在**可执行代码**里被禁止，不在散文里 —— 见 `_clinical_hits` 的说明。
CLINICAL_WORDS = ("histolog", "topograph", "morpholog", "icdo", "icd-o", "tumour", "tumor",
                  "carcinom", "oncolog", "biopsy", "primary_site", "seer", "ajcc")


def _clinical_hits(path: pathlib.Path) -> list[str]:
    """可执行代码里出现的临床词：标识符名，以及非 docstring 的字符串字面量。

    整文件 grep 是这条测试的第一版，它红在两处而两处都是假阳性：`kernel.py:3` 的
    "deliberately knows nothing about tumour registries" 是在**声明**中立，`llm.py` 的
    docstring 里有一句 `"pathology OR biopsy"` 的示例。两者都不驱动任何行为。

    这个仓库对领域耦合的标准是 `labelling.py:342` 那条 —— 模板里不许**命名**疾病、器官、
    文档词表或编码系统，因为"every one of those words was a lie the moment the requirement
    moved"。会说谎的是代码里的那个名字，不是解释它为什么不在那里的句子。所以查标识符和
    活的字符串，放过 docstring 与注释（注释根本不进 AST）。
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


def test_the_kernel_names_no_clinical_concept():
    """kernel 的 docstring 说它"deliberately knows nothing about tumour registries"。断言它。

    只查 kernel 层，因为这是唯一一层可以要求零领域词的：contract 层会合法地出现字段名，
    review 层的 prompt 里会出现文档类型。领域中立的完整检查是另一件事，这里只钉最里层 ——
    一条能通过的窄断言，胜过一条必须靠豁免清单才能通过的宽断言。
    """
    plane, _ = _layer_of()
    hits = {m: h for m, path in _modules().items()
            if plane.get(m) == "kernel" and (h := _clinical_hits(path))}
    assert not hits, f"kernel 层的可执行代码里出现临床概念: {hits}"


def test_the_clinical_word_check_can_actually_fail(tmp_path: pathlib.Path):
    """守卫上一条：一个只会通过的检查等于没有检查。

    两个断言，方向相反 —— 代码里的名字必须被抓到，散文里的同一个词必须被放过。第一版
    整文件 grep 抓到了后者，所以这里把"放过散文"也钉住，否则修完假阳性之后没有任何东西
    阻止有人把它改回整文件搜索。
    """
    bad = tmp_path / "bad.py"
    bad.write_text('def f(histology_code):\n    return {"primary_site": histology_code}\n',
                   encoding="utf-8")
    assert _clinical_hits(bad), "代码里的临床标识符必须被抓到"

    prose = tmp_path / "prose.py"
    prose.write_text('"""This module knows nothing about tumour registries or biopsy reports."""\n'
                     'def f(x):\n    """Not about histology either."""\n    return x\n',
                     encoding="utf-8")
    assert not _clinical_hits(prose), "docstring 里的同一个词必须被放过"
