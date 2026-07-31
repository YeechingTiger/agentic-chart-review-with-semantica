"""把 `src/acr/` 的平铺模块搬进按平面命名的子包，并重写全树的引用。

为什么是脚本而不是手工
--------------------
57 个文件引用 `acr.X`，而其中一类引用**不是 import 语句**：`evals.REGISTRY` 的 `method` /
`verifier` 是字符串（`"acr.evals.score"`（搬迁后是 `"acr.evaluation.evals.score"`）），测试里的 monkeypatch 目标也是字符串
（`"acr.cli_common.llm_client"`）。只重写 AST 里的 import 会漏掉它们，而漏掉的那些在运行时
才炸 —— 或者更糟，`monkeypatch.setattr` 打在一个不存在的路径上会抛错，但一个只在字符串里
被查表的名字会静默失配。所以重写按**文本**做，覆盖 import 与字符串两类。

分片搬，每片跑一次全量测试。用法：

    .venv/bin/python tools/move_to_plane.py <plane> <module> [<module> ...]

约束：平面目录名不得与任何模块名相同（否则 `acr.cli/` 会遮蔽 `acr/cli.py`），脚本会拒绝。
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "acr"
#: 要重写引用的树。`runs/` 和 `corpus/` 不含代码，`.venv` 是第三方。
#: 本文件排除在外：它自己的 docstring 里有 `acr.evals.score` 这类**示例**，第一次运行时被
#: 自己改写成了 `acr.evaluation.evals.score`，于是文档在讲一件它不再演示的事。
SCAN = ("src", "tests", "tools")
SELF = pathlib.Path(__file__).resolve()


def modules_in_tree() -> set[str]:
    out = set()
    for p in SRC.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        out.add(str(p.relative_to(SRC).with_suffix("")).replace("/", "."))
    return out


def rewrite(text: str, plane: str, moved: set[str]) -> str:
    """把对 `moved` 里模块的引用改指到 `acr.<plane>.<module>`。

    四种形态，逐个处理，顺序无关（模式互不重叠）：

      `from acr.X import`  /  `import acr.X`  /  `"acr.X..."`   -> 插入 plane
      `from .X import`     （src/acr 里的同级相对 import）      -> `from .<plane>.X import`
      `from ..X import`    （src/acr/sub 里的相对 import）       -> `from ..<plane>.X import`

    被搬走的模块**彼此之间**的相对 import 不改：它们落在同一个新包里，`from .other import`
    仍然解析得到。这也是为什么参数是一整个平面而不是单个模块。
    """
    tops = {m.split(".")[0] for m in moved}
    for m in sorted(moved, key=len, reverse=True):
        top = m.split(".")[0]
        # 绝对形态：import 语句与字符串字面量一起覆盖
        text = re.sub(rf"\bacr\.{re.escape(m)}\b", f"acr.{plane}.{m}", text)
        if "." not in m:
            text = re.sub(rf"(?m)^(\s*)from \.{re.escape(top)} import ",
                          rf"\1from .{plane}.{top} import ", text)
            text = re.sub(rf"(?m)^(\s*)from \.\.{re.escape(top)} import ",
                          rf"\1from ..{plane}.{top} import ", text)

    # `from . import attribution as A` / `from acr import attribution`。第一版漏了这一整类，
    # 后果是 14 个测试模块在收集期就 `ImportError: cannot import name 'attribution' from 'acr'`
    # —— `cli_attribute.py` 正是这么写的。名单里混了搬走和没搬走的会被拆成两条语句。
    def split_bare(match: re.Match) -> str:
        indent, base, names_raw = match.group(1), match.group(2), match.group(3)
        parts = [p.strip() for p in names_raw.split(",") if p.strip()]
        went = [p for p in parts if p.split(" as ")[0].strip() in tops]
        stayed = [p for p in parts if p.split(" as ")[0].strip() not in tops]
        if not went:
            return match.group(0)
        lines = []
        if stayed:
            lines.append(f"{indent}from {base} import {', '.join(stayed)}")
        sep = "" if base.endswith(".") else "."
        lines.append(f"{indent}from {base}{sep}{plane} import {', '.join(went)}")
        return "\n".join(lines)

    text = re.sub(r"(?m)^(\s*)from (\.+|acr) import ([^\n(]+)$", split_bare, text)

    # 顶层单位本身（一个**包**的名字）。搬 `specview` / `tools` 这两个包时，
    # `acr.specview.basis` 被上面的按模块规则改到了，但 `from acr.specview import JARGON`
    # —— 引用包自己的 `__init__` 再导出 —— 没有，于是 test_specview 与 test_value_domains
    # 在收集期 ModuleNotFoundError。放在按模块规则**之后**：那时长名字已经变成
    # `acr.<plane>.specview.basis`，负向前视保证不会被再匹配一次。
    for top in sorted(tops, key=len, reverse=True):
        text = re.sub(rf"\bacr\.{re.escape(top)}\b(?!\.\w)", f"acr.{plane}.{top}", text)
        text = re.sub(rf"(?m)^(\s*)from \.{re.escape(top)} import ",
                      rf"\1from .{plane}.{top} import ", text)
        text = re.sub(rf"(?m)^(\s*)from \.\.{re.escape(top)} import ",
                      rf"\1from ..{plane}.{top} import ", text)
    return text


def unrewrite_within_plane(path: pathlib.Path, plane: str, moved: set[str]) -> None:
    """搬进同一平面的模块之间，把刚被改成 `.plane.X` 的相对 import 改回 `.X`。

    上一步是全树无差别替换，所以搬进来的文件也被改了，而它现在**就在**那个包里 ——
    `from .<plane>.X import` 在 `acr/<plane>/other.py` 里会去找 `acr/<plane>/<plane>/X.py`。
    """
    text = path.read_text(encoding="utf-8")
    for m in sorted(moved, key=len, reverse=True):
        top = m.split(".")[0]
        text = re.sub(rf"(?m)^(\s*)from \.{re.escape(plane)}\.{re.escape(top)} import ",
                      rf"\1from .{top} import ", text)
        # 平面内模块引用平面外模块的相对 import：现在深了一层
        text = re.sub(rf"\bacr\.{re.escape(plane)}\.{re.escape(m)}\b", f"acr.{plane}.{m}", text)
    path.write_text(text, encoding="utf-8")


def deepen_outward_imports(path: pathlib.Path, plane: str, moved: set[str],
                           depth: int = 1) -> None:
    """搬进来的文件对**平面外**模块的相对 import 现在少了一层，加一个点。

    DEPTH 决定规则，这是第一版的第二个 bug。一个整体搬走的**子包**内部的
    `from .sibling import` 依然有效，不能动 —— 而第一版把
    `attribution_modules/builtins.py` 的 `from .registry import` 改成了 `..registry`，
    于是它去找 `acr.diagnosis.registry`，那里什么也没有。

        depth 1（平面目录下的文件）
            from .other import X   -> from ..other import X   （other 在平面外）
            from .attribution ...  -> 不变                     （同平面）
            from . import evals    -> from .. import evals
        depth 2+（整体搬走的子包内的文件）
            from .registry import  -> 不变                     （子包内的兄弟）
            from ..other import    -> from ...other import   （`..` 原指 acr，现指 acr/<plane>）

    名单里混了平面内外的 `from . import a, b` 会被拆成两条语句，而不是猜一个。
    """
    text = path.read_text(encoding="utf-8")
    inside = {m.split(".")[0] for m in moved}

    if depth > 1:
        # 子包内：只有指向"原 acr 顶层"的那一级需要加深。同包兄弟（单点）保持不变。
        def sub_up(match: re.Match) -> str:
            indent, dots, name = match.group(1), match.group(2), match.group(3)
            return (match.group(0) if name in inside
                    else f"{indent}from {dots}.{name}" + match.group(4))
        text = re.sub(r"(?m)^(\s*)from (\.\.)([a-z_][a-z0-9_]*)((?:\.[a-z_][a-z0-9_]*)* import )",
                      sub_up, text)
        text = re.sub(r"(?m)^(\s*)from \.\. import ", r"\1from ... import ", text)
        path.write_text(text, encoding="utf-8")
        return

    def sub_from_x(match: re.Match) -> str:
        indent, name, rest = match.group(1), match.group(2), match.group(3)
        return match.group(0) if name in inside else f"{indent}from ..{name}{rest}"

    text = re.sub(r"(?m)^(\s*)from \.([a-z_][a-z0-9_]*)((?:\.[a-z_][a-z0-9_]*)* import )",
                  sub_from_x, text)

    def sub_from_dot(match: re.Match) -> str:
        indent, names_raw = match.group(1), match.group(2)
        parts = [p.strip() for p in names_raw.split(",") if p.strip()]
        out_names = [p for p in parts if p.split(" as ")[0].strip() not in inside]
        in_names = [p for p in parts if p.split(" as ")[0].strip() in inside]
        lines = []
        if in_names:
            lines.append(f"{indent}from . import {', '.join(in_names)}")
        if out_names:
            lines.append(f"{indent}from .. import {', '.join(out_names)}")
        return "\n".join(lines)

    text = re.sub(r"(?m)^(\s*)from \. import ([^\n]+)$", sub_from_dot, text)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    plane, names = argv[0], argv[1:]
    tree = modules_in_tree()
    # 撞名的判据是"存在 `acr/<plane>.py` 这个文件"，不是"某个模块路径首段等于 plane"。
    # 后者在**续跑**时必然成立 —— `core/` 已经存在，模块名就都以 `core.` 开头 —— 于是守卫
    # 会把一次正常的收尾当成撞名拒掉。
    if (SRC / f"{plane}.py").is_file():
        print(f"refusing: plane {plane!r} collides with acr/{plane}.py — the package would "
              f"shadow that module", file=sys.stderr)
        return 1
    missing = [n for n in names
               if n not in tree and f"{plane}.{n}" not in tree
               and not any(m.startswith(f"{plane}.{n.split('.')[0]}.") for m in tree)]
    if missing:
        print(f"no such module(s): {missing}", file=sys.stderr)
        return 1
    moved = set(names)

    dest = SRC / plane
    dest.mkdir(exist_ok=True)
    init = dest / "__init__.py"
    if not init.exists():
        init.write_text(f'"""The {plane} plane. See tests/test_layering.py for what may '
                        f'depend on it."""\n', encoding="utf-8")

    # 搬的是**顶层单位**：`attribution_modules.builtins` 属于包 `attribution_modules`，
    # 整个包一起搬。第一版按点分名逐个 git mv，把 `attribution_modules/builtins.py` 压平成了
    # `diagnosis/builtins.py` —— 包结构没了，而 `from .attribution_modules.registry import`
    # 这类引用会指向不存在的路径。
    units = sorted({n.split(".")[0] for n in names})
    for unit in units:
        src = SRC / f"{unit}.py"
        pkg = SRC / unit
        if src.is_file():
            subprocess.run(["git", "mv", str(src), str(dest / src.name)], check=True, cwd=ROOT)
        elif pkg.is_dir():
            subprocess.run(["git", "mv", str(pkg), str(dest / unit)], check=True, cwd=ROOT)
        elif (dest / f"{unit}.py").is_file() or (dest / unit).is_dir():
            # 已经在目标位置。允许，因为一次中断的搬迁（`git mv` 在未被跟踪的新文件上失败过）
            # 必须能靠重跑同一条命令收尾 —— 而 `git mv` 是原子的一批里的一步，不是全部。
            print(f"  {unit}: already under {plane}/, rewriting references only")
        else:
            print(f"refusing: {unit!r} is neither a module nor a package", file=sys.stderr)
            return 1

    for d in SCAN:                                    # 全树重写引用
        for p in (ROOT / d).rglob("*.py"):
            if ".venv" in p.parts or p.resolve() == SELF:
                continue
            text = p.read_text(encoding="utf-8")
            new = rewrite(text, plane, moved)
            if new != text:
                p.write_text(new, encoding="utf-8")

    # 搬进来的文件：修回同平面引用，加深指向平面外的引用。深度决定规则 —— 见
    # `deepen_outward_imports`。子包的 `__init__.py` 也要处理（它可能 import 兄弟模块），
    # 只有平面自己那个刚写出来的 `__init__.py` 跳过。
    for p in sorted(dest.rglob("*.py")):
        if p == init:
            continue
        depth = len(p.relative_to(dest).parts)
        unrewrite_within_plane(p, plane, moved)
        deepen_outward_imports(p, plane, moved, depth=depth)

    print(f"moved into acr/{plane}/: {sorted(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
