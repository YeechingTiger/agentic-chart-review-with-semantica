"""把 `codes/*.yaml` 从写死三个轴的旧形状迁到声明式 `axes:` 形状。

机械迁移，因为手改会掉行：三张表一共 48 个 topography、161 个 morphology、18 个 behavior
条目，加上 34 条 not_reportable。`tests/test_code_tables.py` 按迁移前的实测条目数钉住了每
张表，所以掉行会红。

旧 -> 新的对应：

    topography / nos_topography   -> axes.topography.codes / .unspecified
    morphology / nos_morphology   -> axes.morphology.codes / .unspecified
    behavior                      -> axes.behavior.codes，`reportable` -> `admissible`,
                                     `meaning` -> `name`
    not_reportable                -> exclusions，`code`+`behavior` -> axis_values
    safeguards                    -> guidance
    excluded_by_spec              -> 原样，另加 excluded_by_spec_axis: morphology
    laterality                    -> 原样保留（旧加载器从未读它，是资产里的说明数据）

三句原先写在 `prompt_block()` Python 里的话搬进每张表的 `warnings:`。它们是**关于这张表**
的断言（模型回忆而来、未经登记员核对、行为位是单独字段），本来就该跟着表走。

用法：`.venv/bin/python tools/migrate_code_tables.py`（就地改写 codes/*.yaml）
"""

from __future__ import annotations

import pathlib
import sys

import yaml

CODES = pathlib.Path(__file__).resolve().parents[1] / "codes"

#: 旧 `prompt_block()` 硬编码的提醒，现在是每张 ICD-O-3 表自己的断言。
ICDO3_WARNINGS = [
    "Code into this table. A four-digit number that is not in it is not a morphology code,",
    "and if a diagnosis has no ICD-O-3 morphology then the finding is not a reportable",
    "neoplasm — say that rather than choosing a number that looks like one.",
    "",
    "This table was recalled by a language model, not transcribed from ICD-O-3, and no",
    "registrar has checked it. Where it disagrees with a pathology report you have read,",
    "say so in your reasoning rather than silently following either one.",
]

#: ICD-O-3 的记法：`C18.7` -> `C187`，`8140/3` -> `8140` + 行为位单列。
ICDO3_NORMALIZATION = {"strip_pattern": r"[.\s]", "uppercase": True, "split_on": "/"}


def migrate(doc: dict) -> dict:
    out: dict = {"table_id": doc["table_id"], "table_version": doc.get("table_version", "0")}
    if "source_authority" in doc:
        out["source_authority"] = doc["source_authority"]
    out["normalization"] = dict(ICDO3_NORMALIZATION)
    out["warnings"] = list(ICDO3_WARNINGS)

    scope = str((doc.get("source_authority") or {}).get("scope", ""))
    axes: dict = {}
    axes["topography"] = {
        "label": "TOPOGRAPHY",
        "field": "primary_site",
        "code_shape": r"C\d{3}",
        "shape_description": "letter C and three digits",
        # 一个格式正确但属于别的器官组的码，是与"这个码不存在"不同的发现。装错表会让每个病例
        # 都像答错，所以这条注记让 check_values 报 OUT_OF_TABLE_SCOPE 而不是 NOT_IN_TABLE。
        "scope_note": (f"Either the tumour is not within {scope} or the wrong table is loaded — "
                       f"this is not evidence that the code is invalid."),
        "codes": doc["topography"],
        "unspecified": doc.get("nos_topography") or [],
    }
    axes["morphology"] = {
        "label": "MORPHOLOGY (four digits; behaviour is a separate field)",
        "field": "histology",
        "code_shape": r"\d{4}",
        "shape_description": "four digits, behaviour reported separately",
        "codes": doc["morphology"],
        "unspecified": doc.get("nos_morphology") or [],
    }
    axes["behavior"] = {
        "label": "BEHAVIOUR",
        "field": "behavior",
        "codes": {str(k): _behavior_entry(v or {}) for k, v in (doc["behavior"] or {}).items()},
    }
    out["axes"] = axes

    if doc.get("excluded_by_spec"):
        out["excluded_by_spec"] = doc["excluded_by_spec"]
        out["excluded_by_spec_axis"] = "morphology"
    if doc.get("not_reportable"):
        out["exclusions"] = [_exclusion(r) for r in doc["not_reportable"]]
    if doc.get("safeguards"):
        out["guidance"] = doc["safeguards"]
    if doc.get("laterality"):
        out["laterality"] = doc["laterality"]
    return out


def _behavior_entry(v: dict) -> dict:
    """`meaning` -> `name`、`reportable` -> `admissible`，其余键原样保留。

    重命名而不是两个键并存：`reportable` 是登记面的词，`admissible` 是"这个值能不能作为答案"
    的通用说法，而两个键同时存在会让下一个人不知道该读哪个。
    """
    e = {k: val for k, val in v.items() if k not in ("meaning", "reportable")}
    if "meaning" in v:
        e["name"] = v["meaning"]
    if "reportable" in v:
        e["admissible"] = bool(v["reportable"])
    return e


def _exclusion(r: dict) -> dict:
    """`{code, behavior, term, why}` -> `{term, why, axis_values}`。

    没有 `code` 的那几条（"这个发现根本没有 ICD-O-3 形态码"）不带 axis_values，于是
    `exclusion_term()` 不会去匹配它们 —— 一条匹配不到任何码的排除项若参与匹配，会匹配上一切。
    """
    out = {"term": r.get("term", ""), "why": r.get("why", "")}
    av = {}
    if r.get("code"):
        av["morphology"] = str(r["code"])
    if r.get("behavior") is not None and str(r.get("behavior", "")).strip():
        av["behavior"] = str(r["behavior"])
    if av:
        out["axis_values"] = av
    return out


def main() -> int:
    paths = sorted(CODES.glob("*.yaml"))
    if not paths:
        print(f"no tables under {CODES}", file=sys.stderr)
        return 1
    for path in paths:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "axes" in doc:
            print(f"{path.name}: already migrated, skipped")
            continue
        new = migrate(doc)
        path.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True, width=100),
                        encoding="utf-8")
        counts = {a: len(v["codes"]) for a, v in new["axes"].items()}
        print(f"{path.name}: {counts}, exclusions={len(new.get('exclusions', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
