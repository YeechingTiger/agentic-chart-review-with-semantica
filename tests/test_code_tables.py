"""码表加载器必须能表达一个不是肿瘤的码系统。

为什么这是一条测试而不是一次重构
--------------------------------
`icdo3.load_table` 要求每张表都有 `topography` / `morphology` / `behavior` 三个键，
`prompt_block` 把这三个词当章节标题写死，`_TOPO = C\\d{3}` 和 `_MORPH = \\d{4}` 是模块常量。
于是一张 LOINC 血脂面板或 RxNorm 药物类**根本无法表达**：`load_table` 直接抛
`CodeTableError`，而 `spec.load_spec` 是 fail-closed 的，整个 spec 加载失败。

框架因此被焊在一个 use case 上，而这件事只有从"能不能装进第二个领域"这个角度才看得见 ——
所以第一条测试就是一张血脂表，而且它先于代码存在。

肿瘤那三张表的等价性同样被钉住：迁移是机械的，但机械的迁移也会掉行。
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODES = ROOT / "assets" / "codes"

#: 一张与肿瘤无关的码表，写成资产该有的样子。两个轴，各自带自己的标题、字段名和码形。
LIPID_TABLE = {
    "table_id": "loinc_lipid_panel",
    "table_version": "1",
    "source_authority": {"scope": "fasting lipid panel, LOINC 2.77"},
    "normalization": {"uppercase": False, "strip_pattern": r"\s"},
    "axes": {
        "analyte": {
            "label": "ANALYTE (LOINC)",
            "field": "analyte_code",
            "code_shape": r"\d{4,5}-\d",
            "shape_description": "four or five digits, a hyphen, one check digit",
            "codes": {
                "2093-3": {"name": "Cholesterol, total"},
                "2571-8": {"name": "Triglyceride"},
                "13457-7": {"name": "LDL cholesterol, calculated",
                            "aliases": ["LDL-C calc"]},
            },
            "unspecified": ["2093-3"],
        },
        "interpretation": {
            "label": "INTERPRETATION",
            "field": "flag",
            "codes": {
                "H": {"name": "above reference range", "admissible": True},
                "N": {"name": "within reference range", "admissible": True},
                "X": {"name": "not reported", "admissible": False},
            },
        },
    },
    "guidance": ["A calculated LDL is not admissible when triglycerides exceed 400 mg/dL."],
}


def _write(tmp_path: pathlib.Path, doc: dict, name: str) -> pathlib.Path:
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False),
                                           encoding="utf-8")
    return tmp_path / f"{name}.yaml"


# ============================================================ THE POINT OF THE WHOLE CHANGE
def test_a_code_system_with_no_cancer_axes_loads(tmp_path):
    """今天 `icdo3.load_table` 会在这里抛 CodeTableError。"""
    from acr.contract.code_tables import load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))
    assert t.table_id == "loinc_lipid_panel"
    assert list(t.axes) == ["analyte", "interpretation"]
    assert t.axes["analyte"].codes["2093-3"]["name"] == "Cholesterol, total"


def test_the_prompt_block_uses_the_tables_own_axis_labels(tmp_path):
    """章节标题来自 YAML，不来自 Python。

    旧的 `prompt_block` 写死了 TOPOGRAPHY / MORPHOLOGY / BEHAVIOUR 三个标题和一句
    "if a diagnosis has no ICD-O-3 morphology then the finding is not a reportable neoplasm"。
    一张血脂表拿到那些标题就是胡说。
    """
    from acr.contract.code_tables import load_table, prompt_block
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    block = prompt_block(load_table("loinc_lipid_panel", codes_dir=str(tmp_path)))
    assert "ANALYTE (LOINC)" in block and "INTERPRETATION" in block
    assert "2093-3" in block and "Cholesterol, total" in block
    assert "LDL-C calc" in block                      # 别名要进提示词
    for cancer in ("TOPOGRAPHY", "MORPHOLOGY", "BEHAVIOUR", "neoplasm", "ICD-O-3", "tumour"):
        assert cancer not in block, f"血脂表的提示词里出现了 {cancer!r}"


def test_an_axis_order_declared_in_yaml_is_the_render_order(tmp_path):
    """轴的顺序是资产的决定。反过来会让"先看部位再看形态"这类阅读顺序无法表达。"""
    from acr.contract.code_tables import load_table, prompt_block
    doc = {**LIPID_TABLE, "axes": {k: LIPID_TABLE["axes"][k]
                                   for k in ("interpretation", "analyte")}}
    _write(tmp_path, doc, "reordered")
    block = prompt_block(load_table("reordered", codes_dir=str(tmp_path)))
    assert block.index("INTERPRETATION") < block.index("ANALYTE (LOINC)")


def test_normalization_is_declared_by_the_table_not_by_the_module(tmp_path):
    """`C18.7 -> C187` 是 ICD-O-3 的记法，不是所有码系统的记法。

    血脂表的 LOINC 码里那个连字符是**码的一部分**，一个写死了 `[.\\s]` 加 `split_on='/'`
    的规范化函数会把别的系统的码改坏 —— 所以规则跟着表走。
    """
    from acr.contract.code_tables import load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))
    assert t.normalize("13457-7") == "13457-7"          # 连字符保留
    assert t.normalize(" 2093-3 ") == "2093-3"          # 声明了 strip 空白
    assert t.normalize("ldl") == "ldl"                  # 声明了不转大写


def test_a_table_with_no_axes_is_refused(tmp_path):
    """空表比缺表更危险：它会渲染出一个空值域，而运行看起来就像被给过码表。

    和 `acr.contract.skills` 拒绝缺失的 skill 是同一个理由。
    """
    from acr.contract.code_tables import CodeTableError, load_table
    _write(tmp_path, {"table_id": "empty", "axes": {}}, "empty")
    with pytest.raises(CodeTableError, match="no axes"):
        load_table("empty", codes_dir=str(tmp_path))


def test_a_missing_table_names_what_is_available(tmp_path):
    from acr.contract.code_tables import CodeTableError, load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    with pytest.raises(CodeTableError, match="loinc_lipid_panel"):
        load_table("nope", codes_dir=str(tmp_path))


# ============================================================ 值检查按轴名，不按字段名
def test_check_values_is_keyed_by_axis_not_by_cancer_field_names(tmp_path):
    """旧签名是 `check_codes(site, histology, behavior)` —— 三个癌症字段名写在参数里。

    泛化后调用方按**轴名**传值，字段名从轴的 `field` 读，于是同一个函数能检查血脂面板。
    """
    from acr.contract.code_tables import check_values, load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))

    assert check_values({"analyte": "2093-3", "interpretation": "N"}, table=t) == []

    bad_shape = check_values({"analyte": "cholesterol"}, table=t)
    assert [p.kind for p in bad_shape] == ["MALFORMED"]
    assert bad_shape[0].field == "analyte_code"        # 字段名来自轴声明
    assert "four or five digits" in bad_shape[0].message

    unknown = check_values({"analyte": "9999-9"}, table=t)
    assert [p.kind for p in unknown] == ["NOT_IN_TABLE"]

    not_admissible = check_values({"interpretation": "X"}, table=t)
    assert [p.kind for p in not_admissible] == ["NOT_ADMISSIBLE"]

    # 空值是弃答的事，不是码表的事。
    assert check_values({"analyte": "", "interpretation": None}, table=t) == []


def test_an_unknown_axis_name_is_refused_rather_than_ignored(tmp_path):
    """静默忽略一个拼错的轴名，等于报告"检查通过"而什么都没检查。"""
    from acr.contract.code_tables import CodeTableError, check_values, load_table
    _write(tmp_path, LIPID_TABLE, "loinc_lipid_panel")
    t = load_table("loinc_lipid_panel", codes_dir=str(tmp_path))
    with pytest.raises(CodeTableError, match="analytee"):
        check_values({"analytee": "2093-3"}, table=t)


# ============================================================ 三张真实肿瘤表的等价性
REAL = {"icdo3_lung": {"topography": 8, "morphology": 55, "behavior": 6},
        "icdo3_colorectal": {"topography": 16, "morphology": 37, "behavior": 6},
        "icdo3_multisite": {"topography": 24, "morphology": 69, "behavior": 6}}


@pytest.mark.parametrize("name", sorted(REAL))
def test_the_real_tables_survive_migration_with_every_code(name: str):
    """机械迁移也会掉行。每个轴的条目数按迁移前的实测值钉住。"""
    from acr.contract.code_tables import load_table
    t = load_table(name)
    got = {ax: len(a.codes) for ax, a in t.axes.items()}
    assert got == REAL[name], f"{name} 迁移后条目数变了: {got}"


@pytest.mark.parametrize("name", sorted(REAL))
def test_the_real_tables_keep_their_icdo3_notation_folding(name: str):
    """`C18.7 -> C187`、`8140/3 -> 8140` 必须还在 —— 它是 4 of 6 有用的旧 firing。"""
    from acr.contract.code_tables import load_table
    t = load_table(name)
    assert t.normalize("C18.7") == "C187"
    assert t.normalize("c34.1") == "C341"
    assert t.normalize("8140/3") == "8140"


def test_the_lung_table_still_knows_which_lobe_c341_is():
    """回归：一次运行写过"C341 是右中叶"并据此编码，而 C341 是**上**叶。

    这是整个码表存在的理由之一，泛化不能把它丢掉。
    """
    from acr.contract.code_tables import load_table
    t = load_table("icdo3_lung")
    assert "upper" in (t.axes["topography"].codes["C341"].get("name") or "").lower()


def test_the_cancer_prompt_block_still_carries_its_own_warnings():
    """表自己的提醒（未经登记员核对、行为位单独一个字段）现在住在 YAML 里，必须还在提示词里。"""
    from acr.contract.code_tables import load_table, prompt_block
    block = prompt_block(load_table("icdo3_lung"))
    assert "TOPOGRAPHY" in block and "MORPHOLOGY" in block
    assert "C341" in block
    low = block.lower()
    assert "registrar" in low or "signed off" in low or "no registrar" in low
