"""仓库根和它下面的资产目录，不靠 `__file__` 的层数算出来。

为什么存在
--------
五个模块各自写着 `Path(__file__).resolve().parents[2]`，那个 2 编码的是"我在 src/acr/ 下面
第一层"。把模块搬进平面目录之后它指向 `src/`，而后果不是报错：`labelling.py` 的
"标注不许写进仓库内部"这条拒绝**静默失效**了，因为它比较的根变成了 `src/`。
`test_labels_root_refuses_a_path_inside_the_repository` 抓到了它 —— DID NOT RAISE。

层数是随目录结构变的量，而"仓库根"不是。所以往上走到带 `pyproject.toml` 的那一层，
搬到多深都对。找不到就抛，不回退到某个猜测：一个猜错了根的路径检查正是上面那种静默失效。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

#: 标记文件。`pyproject.toml` 而不是 `.git`：从打好的 wheel 里跑时没有 `.git`，而那时
#: 这些资产目录也不在，调用方应该拿到一个明确的错误而不是一个存在但为空的路径。
MARKER = "pyproject.toml"


class RepoRootNotFound(RuntimeError):
    """向上找不到带标记文件的目录。"""


@lru_cache(maxsize=1)
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).is_file():
            return candidate
    raise RepoRootNotFound(
        f"no {MARKER} at or above {here}. These asset directories only exist in a source "
        f"checkout; a packaged install has no repo root and callers must not guess one.")


def asset_dir(name: str) -> Path:
    """仓库根下的一个资产目录，例如 `skills` / `codes`。不检查存在性 —— 调用方各有各的报错。"""
    return repo_root() / name
