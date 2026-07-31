"""值域码表：一个码系统的事实表，轴由资产声明而不是由本模块写死。

这个模块替换了 `icdo3.py`。原因不是风格
--------------------------------------
`icdo3.load_table` 要求每张表都有 `topography` / `morphology` / `behavior` 三个键，
`prompt_block` 把这三个词当章节标题写死，`_TOPO = C\\d{3}` 和 `_MORPH = \\d{4}` 是模块常量，
`check_codes(site, histology, behavior)` 把三个癌症字段名写进了参数表。于是：

  * 一张 LOINC 血脂面板或 RxNorm 药物类**无法表达** —— `load_table` 抛错，而
    `spec.load_spec` 是 fail-closed 的，整个 spec 加载失败；
  * `spec`、`agent`、`run_manifest` 三处框架代码 import 了一个 use case 专属模块，
    这是 `tests/test_layering.py` 登记的三条倒挂依赖中的全部。

肿瘤登记是这套框架的**一个** use case。所以轴的名字、章节标题、码形正则和记法折叠规则
全部下沉到 `codes/*.yaml`，本模块只知道"表有若干个有序的轴，每个轴有码和名字"。

保留下来的三件事，一件都不是 gate
--------------------------------
  1. `prompt_block()` 把值域渲进系统提示词，让模型往一张看得见的表里编码，而不是往一张
     半记住的表里编码。两次真实失败促成了它：一次把 `7205` 当形态码（ICD-O-3 里没有
     7205），一次写下"C341 是右中叶"并据此编码（C341 是**上**叶）。
  2. `check_values()` 返回带类型的问题给评测面**计数**，永不用于拒绝。
  3. `normalize()` 折叠记法差异。这是 4 of 6 有用的旧 `field_format` firing —— 它当年
     大半时间在制造自己再解决的往返。

REPORTABILITY / ADMISSIBILITY 是领域策略，不是本模块的权限。轴里的 `admissible: false`
和表级 `exclusions` 携带的是编码手册的裁定，所以每个问题都是 advisory，而且每张表的
`source_authority` 都要求由人核对。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CODES_DIR = Path(__file__).resolve().parents[2] / "codes"

#: 问题类型。是名字而不是布尔，因为"这不是一个码"、"这个码属于别的范围"和"这个码不可采纳"
#: 是三种不同的发现，分不清的调用方会报错那一种。
MALFORMED = "MALFORMED"
NOT_IN_TABLE = "NOT_IN_TABLE"
OUT_OF_TABLE_SCOPE = "OUT_OF_TABLE_SCOPE"
NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
EXCLUDED_BY_SPEC = "EXCLUDED_BY_SPEC"


class CodeTableError(ValueError):
    """码表缺失，或者不是本模块能读的形状。"""


@dataclass(frozen=True)
class CodeProblem:
    kind: str
    field: str
    value: str
    message: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "field": self.field, "value": self.value,
                "message": self.message}


@dataclass(frozen=True)
class CodeAxis:
    """一个轴：一组码，加上渲染和检查它所需要的、由资产声明的元数据。

    `name` 是轴名（调用方按它传值），`field` 是这个轴编码的 spec 字段名（问题报在它上面）。
    两者分开，是因为一个轴可以在不同 spec 里落到不同字段名上。
    """

    name: str
    label: str
    field: str
    codes: dict[str, dict]
    unspecified: tuple[str, ...] = ()
    code_shape: str = ""
    shape_description: str = ""
    scope_note: str = ""

    def name_of(self, code: str) -> str | None:
        e = self.codes.get(code)
        return e.get("name") if e else None

    def is_admissible(self, code: str) -> bool:
        """轴不声明 `admissible` 时默认可采纳。缺省为 True，因为绝大多数轴的码都是可用值，
        而把默认设成 False 会让一张没写这个键的表整体变成不可采纳且没人看得出为什么。"""
        e = self.codes.get(code)
        return True if e is None else bool(e.get("admissible", True))


@dataclass(frozen=True)
class CodeTable:
    table_id: str
    version: str
    scope: str
    axes: dict[str, CodeAxis]
    exclusions: tuple[dict, ...] = ()
    guidance: tuple[str, ...] = ()
    excluded_by_spec: dict[str, dict] = field(default_factory=dict)
    excluded_by_spec_axis: str = ""
    warnings: tuple[str, ...] = ()
    _norm: dict = field(default_factory=dict)

    def normalize(self, raw: str) -> str:
        """按**这张表**声明的记法规则折叠一个码。

        规则跟着表走而不是写在模块里：ICD-O-3 写 `C18.7` 和 `8140/3`，而 LOINC 码里的连字符
        是码的一部分。一个写死了 `[.\\s]` 加 `split_on='/'` 的函数会把别的系统的码改坏。
        """
        s = str(raw or "")
        pattern = self._norm.get("strip_pattern")
        if pattern:
            s = re.sub(pattern, "", s)
        if self._norm.get("uppercase", True):
            s = s.upper()
        split_on = self._norm.get("split_on")
        if split_on and split_on in s:
            s = s.split(split_on, 1)[0]
        return s

    def trailing_part(self, raw: str) -> str | None:
        """`split_on` 之后被丢掉的那一段的首字符，例如 `8140/3` 的 `3`。

        存在的理由：行为位在 STORE 里是**单独一个字段**，所以不能被静默并进形态码，但调用方
        需要拿得到它。表没声明 `split_on` 就永远返回 None。
        """
        split_on = self._norm.get("split_on")
        if not split_on:
            return None
        s = str(raw or "")
        pattern = self._norm.get("strip_pattern")
        if pattern:
            s = re.sub(pattern, "", s)
        if split_on not in s:
            return None
        tail = s.split(split_on, 1)[1]
        return tail[:1] or None

    def exclusion_term(self, axis_values: dict[str, str]) -> str | None:
        """`exclusions` 里与这组轴值全部吻合的那一条的术语名。"""
        for row in self.exclusions:
            match = row.get("axis_values") or {}
            if not match:
                continue
            if all(self.normalize(str(axis_values.get(a, ""))) == self.normalize(str(v))
                   for a, v in match.items()):
                return str(row.get("term", ""))
        return None


def _axis(name: str, d: dict, norm: dict) -> CodeAxis:
    codes_raw = d.get("codes")
    if not isinstance(codes_raw, dict) or not codes_raw:
        raise CodeTableError(f"axis {name!r} has no `codes` mapping")

    def fold(raw: str) -> str:
        s = str(raw or "")
        if norm.get("strip_pattern"):
            s = re.sub(norm["strip_pattern"], "", s)
        return s.upper() if norm.get("uppercase", True) else s

    return CodeAxis(
        name=name,
        label=str(d.get("label") or name.upper()),
        field=str(d.get("field") or name),
        codes={fold(k): (v or {}) for k, v in codes_raw.items()},
        unspecified=tuple(fold(c) for c in (d.get("unspecified") or [])),
        code_shape=str(d.get("code_shape") or ""),
        shape_description=str(d.get("shape_description") or ""),
        scope_note=str(d.get("scope_note") or ""),
    )


@lru_cache(maxsize=8)
def load_table(name: str, codes_dir: str | None = None) -> CodeTable:
    """一张码表，按名字。

    没有默认表名。旧的 `load_table(name="icdo3_lung")` 让"忘了声明值域"和"声明了肺"看起来
    一样，而装错表会让每个病例都像答错。名字必须由 spec 说出来。
    """
    root = Path(codes_dir) if codes_dir else CODES_DIR
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise CodeTableError(
            f"no code table {name!r} at {path}; available: "
            f"{sorted(p.stem for p in root.glob('*.yaml'))}")
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "table_id" not in d:
        raise CodeTableError(f"{path} has no 'table_id'")
    axes_raw = d.get("axes")
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise CodeTableError(
            f"{path} declares no axes. An empty value domain renders an empty block into the "
            f"prompt, and the run then looks exactly like one that was given the codes — the "
            f"same reason `acr.skills` raises on a missing skill instead of returning ''.")
    norm = dict(d.get("normalization") or {})
    return CodeTable(
        table_id=str(d["table_id"]),
        version=str(d.get("table_version", "0")),
        scope=str((d.get("source_authority") or {}).get("scope", "")),
        axes={str(k): _axis(str(k), v or {}, norm) for k, v in axes_raw.items()},
        exclusions=tuple(d.get("exclusions") or []),
        guidance=tuple(str(s) for s in (d.get("guidance") or [])),
        excluded_by_spec={str(k): (v or {})
                          for k, v in (d.get("excluded_by_spec") or {}).items()},
        excluded_by_spec_axis=str(d.get("excluded_by_spec_axis") or ""),
        warnings=tuple(str(s) for s in (d.get("warnings") or [])),
        _norm=norm,
    )


def check_values(values: dict[str, str | None], *, table: CodeTable) -> list[CodeProblem]:
    """一组按轴名给出的值的事实性问题。ADVISORY —— 用于计数，永不用于拒绝。

    空值跳过：一个空值是弃答的事，本模块对"该不该弃答"没有意见。

    轴名拼错会抛错而不是被忽略。静默忽略等于报告"检查通过"而一个字都没查 —— 那是这个仓库
    反复点名的那种不会失败的检查。
    """
    out: list[CodeProblem] = []
    unknown = [a for a in values if a not in table.axes]
    if unknown:
        raise CodeTableError(
            f"{table.table_id} has no axis {unknown!r}; declared axes are "
            f"{sorted(table.axes)}")
    for axis_name, axis in table.axes.items():          # 按表的顺序，不按传入顺序
        if axis_name not in values:
            continue
        raw = values[axis_name]
        if raw is None or not str(raw).strip():
            continue
        code = table.normalize(raw)
        if axis.code_shape and not re.fullmatch(axis.code_shape, code):
            out.append(CodeProblem(
                MALFORMED, axis.field, str(raw),
                f"{raw!r} is not shaped like a {axis.name} code in {table.table_id}"
                + (f" ({axis.shape_description})" if axis.shape_description else "")))
            continue
        if code in table.excluded_by_spec and (
                not table.excluded_by_spec_axis
                or table.excluded_by_spec_axis == axis_name):
            e = table.excluded_by_spec[code]
            out.append(CodeProblem(
                EXCLUDED_BY_SPEC, axis.field, code,
                f"{code} is {e.get('name', '')} — {e.get('why', '')}. The spec puts this outside "
                f"the variable, so the honest answer is SPEC_INSUFFICIENT rather than a coded "
                f"value. This is a scope boundary, not a coding error."))
            continue
        if code not in axis.codes:
            kind = OUT_OF_TABLE_SCOPE if axis.scope_note else NOT_IN_TABLE
            note = (f" {axis.scope_note}" if axis.scope_note else "")
            out.append(CodeProblem(
                kind, axis.field, code,
                f"{code} is not a {axis.name} value in {table.table_id}"
                + (f" ({table.scope})" if table.scope else "") + "." + note))
            continue
        if not axis.is_admissible(code):
            term = table.exclusion_term({a: values.get(a) or "" for a in table.axes})
            out.append(CodeProblem(
                NOT_ADMISSIBLE, axis.field, code,
                f"{axis.name} {code} ({axis.name_of(code) or ''}) is not an admissible value"
                + (f"; this combination is {term}" if term else "") + "."))
    return out


def prompt_block(table: CodeTable, *, max_terms: int = 0) -> str:
    """值域，给系统提示词。默认渲染整张表。

    渲染而不是摘要：一个只被展示 12 of 40 个码的模型会往那 12 个里编码。`max_terms` 是给
    量提示词大小的调用方用的,不是默认值。

    每一行文字都来自资产 —— 标题来自轴的 `label`，提醒来自表的 `warnings`。旧版本把
    "MORPHOLOGY (four digits; behaviour is a separate field)" 和 "if a diagnosis has no
    ICD-O-3 morphology then the finding is not a reportable neoplasm" 写在 Python 里，那两句
    话对一张血脂表就是胡说。
    """
    L = [f"VALUE DOMAIN — {table.table_id} v{table.version}"
         + (f" ({table.scope})" if table.scope else "")]
    if table.warnings:
        L += [""] + list(table.warnings)
    for axis in table.axes.values():
        L += ["", axis.label]
        items = list(axis.codes.items())
        if max_terms:
            items = items[:max_terms]
        for code, e in items:
            line = f"  {code}  {e.get('name', '')}"
            if e.get("aliases"):
                line += f"   [{', '.join(str(a) for a in e['aliases'])}]"
            if code in axis.unspecified:
                line += "   (unspecified — asserts the finer detail is not documented)"
            if not bool(e.get("admissible", True)):
                line += "   — NOT admissible"
            L.append(line)
    if table.exclusions:
        L += ["", "VALUES THAT ARE NOT ADMISSIBLE ANSWERS"]
        for r in table.exclusions:
            av = r.get("axis_values") or {}
            shown = "/".join(str(v) for v in av.values()) if av else "no code exists"
            L.append(f"  {r.get('term', '')}  ({shown}) — {r.get('why', '')}")
    if table.guidance:
        L += ["", "CODING SAFEGUARDS"] + [f"  - {s}" for s in table.guidance]
    return "\n".join(L)


def code_domain_block(spec) -> str:
    """声明了值域的 spec 的值域块；没声明的 spec 得到 ""。

    Task Contract 与提示词之间的接缝。spec 说它的值编码进**哪张**表
    (`value_domain: icdo3_lung`)，因为那是答案含义的一部分；这里把它渲染出来。没声明的
    spec 什么也不给 —— 日期和 class-of-case 变量没有码表值域，塞一墙形态码只会被忽略。

    `load_spec` 已经拒绝过不存在的表，所以到这里的名字一定能解析。`try` 是为测试或 ablation
    在内存里构造的 spec 留的：那里没有这个保证，而一张缺失的表不该拖垮一次本来不需要它的运行。
    """
    name = str(getattr(spec, "value_domain", "") or "").strip()
    if not name:
        return ""
    try:
        return prompt_block(load_table(name))
    except CodeTableError:
        return ""


def table_manifest(spec) -> dict | None:
    """一次运行被展示过的值域的身份，或者 None（它没声明）。

    内容哈希，理由和 `skills_manifest` 一样：这些表是给人编辑的 YAML —— 每张表里的
    `what_a_human_must_check` 正是在邀请这件事 —— 所以单靠 `table_version` 会让一张改过的
    表冒充上一次运行用过的那张。那次给肺表加了十一个形态码的 1,788 行验证就是这种编辑，
    它之前和之后写的 manifest 绝不能被当成可比的。
    """
    import hashlib
    name = str(getattr(spec, "value_domain", "") or "").strip()
    if not name:
        return None
    try:
        t = load_table(name)
    except CodeTableError:
        return {"declared": name, "loaded": False}
    path = CODES_DIR / f"{name}.yaml"
    return {
        "declared": name, "loaded": True,
        "table_id": t.table_id, "table_version": t.version,
        "axes": {ax: len(a.codes) for ax, a in t.axes.items()},
        "n_excluded_by_spec": len(t.excluded_by_spec),
        "content_hash": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "origin": "model_recalled",
        "signed_off": False,
    }
