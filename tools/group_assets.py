"""把仓库根上平铺的资产目录收进一个 `assets/` 屋顶，并重写全树的路径引用。

为什么
----
`src/acr/` 已经按平面分成十个目录，仓库根却还是 17 个平铺目录，其中 9 个是同一类东西 ——
框架**加载**的版本化资产 —— 而看目录名分不出它们和 `src/`、`tests/`、`runs/` 有什么区别。
两个名字还在误导：`audit/` 里只有一个 `prices.json`（价格表；真正的审计输出走
`LocalArtifactStore`，按设计永不进仓库），`authoring/` 是 723 个 CRC 专用 YAML，属于**一个
use case** 的工作区。

不做第二层（`assets/contract/specs/`）：路径深度和改写量翻倍，而"哪个平面拥有它"由
`assets/README.md` 讲更便宜。屋顶本身已经把 9 个变成 1 个。

只改路径前缀，不改裸名
--------------------
第一版还改了"引号里整串就是目录名"的裸名，代价是即刻的：`migrate_code_tables.py` 里
`"codes": doc["topography"]` 的那个 `codes` 是 YAML **键名**不是路径，被改成了
`"assets/codes"`，三张码表随即加载失败。一个装作路径的键名和一个真路径在文本上无法区分，
猜错的方向又是静默的 —— 所以这里不猜。

剩下的裸名常量由 `BARE_NAME_FIXES` 点名修改：能看着上下文判断的地方才手工改，而手工改的
清单必须短到可以逐条读完。

本文件排除在改写之外。第一版把自己的 `MOVES` 键改成了 `"assets/specs"`，脚本从此搬不动任何
东西；`move_to_plane.py` 踩过同一个坑，两次都是"工具在自己扫描的树里"。
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SELF = pathlib.Path(__file__).resolve()

#: 目录名 -> assets/ 下的新名字。两处值不同于键，是在修正误导性命名。
MOVES: dict[str, str] = {
    "specs": "specs",
    "codes": "codes",
    "guidelines": "guidelines",
    "contracts": "contracts",
    "skills": "skills",
    "module_catalog": "module_catalog",
    "pipeline_catalog": "pipeline_catalog",
    "certification_catalog": "certification_catalog",
    "evaluators": "evaluators",
    "audit": "pricing",
    "authoring": "usecase",
}

#: 逐条点名的裸名修改：(文件, 原文, 新文)。短到可以读完，因为每一条都是人判断过上下文的。
BARE_NAME_FIXES: tuple[tuple[str, str, str], ...] = (
    ("src/acr/contract/code_tables.py", 'asset_dir("codes")', 'asset_dir("assets/codes")'),
    ("src/acr/contract/skills.py", 'asset_dir("skills")', 'asset_dir("assets/skills")'),
    ("src/acr/review/coverage_planner.py", 'asset_dir("skills")', 'asset_dir("assets/skills")'),
)

SCAN_DIRS = ("src", "tests", "tools", "docs")
SCAN_FILES = ("README.md", "DEPLOY.md", "RESULTS.md", ".gitignore", "pyproject.toml")


def rewrite(text: str) -> str:
    """只改 `name/` 这一种形态。前面必须不是 word/斜杠/点/连字符，否则 `assets/specs/` 自己
    和 `module_catalog/audit_rules` 这类内含子串的路径会被再改一次。"""
    for old, new in MOVES.items():
        text = re.sub(rf"(?<![\w/.-]){re.escape(old)}/", f"assets/{new}/", text)
    return text


def _self_check() -> None:
    """碰树之前验证 rewrite() 的行为。

    第一版的收窄改动用 `.replace()` 打补丁而我没有断言，`.replace()` 没匹配上，于是裸名规则
    又跑了一遍整棵树 —— 同一个错误连犯两次。一个改写整棵树的工具必须先证明自己在做什么。
    """
    cases = [
        ("specs/STORE.390.yaml", "assets/specs/STORE.390.yaml"),
        ("--spec specs/x.yaml", "--spec assets/specs/x.yaml"),
        ('"codes": doc["topography"]', '"codes": doc["topography"]'),   # YAML 键名不动
        ('asset_dir("codes")', 'asset_dir("codes")'),                   # 裸名不动
        ("assets/specs/x", "assets/specs/x"),                           # 幂等
        ("module_catalog/audit_rules/x.yaml", "assets/module_catalog/audit_rules/x.yaml"),
    ]
    for src, want in cases:
        got = rewrite(src)
        assert got == want, f"rewrite({src!r}) == {got!r}, expected {want!r}"


def main() -> int:
    _self_check()
    ASSETS.mkdir(exist_ok=True)
    for old, new in MOVES.items():
        src, dst = ROOT / old, ASSETS / new
        if dst.exists():
            print(f"  {old}: already at assets/{new}")
            continue
        if not src.is_dir():
            print(f"refusing: {old}/ is not a directory", file=sys.stderr)
            return 1
        subprocess.run(["git", "mv", str(src), str(dst)], check=True, cwd=ROOT)
        print(f"  {old}/ -> assets/{new}/")

    targets: list[pathlib.Path] = []
    for d in SCAN_DIRS:
        targets += [p for p in (ROOT / d).rglob("*")
                    if p.is_file() and p.suffix in (".py", ".md", ".yaml", ".yml", ".json")]
    targets += [ROOT / f for f in SCAN_FILES if (ROOT / f).is_file()]
    targets += list(ASSETS.rglob("*.md"))
    changed = 0
    for p in targets:
        if ".venv" in p.parts or "__pycache__" in p.parts or p.resolve() == SELF:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text = rewrite(text)
        if new_text != text:
            p.write_text(new_text, encoding="utf-8")
            changed += 1

    for rel, old, new in BARE_NAME_FIXES:
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            print(f"refusing: {rel} does not contain {old!r}", file=sys.stderr)
            return 1
        p.write_text(text.replace(old, new), encoding="utf-8")
        changed += 1
    print(f"rewrote references in {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
