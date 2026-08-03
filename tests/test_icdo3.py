"""ICD-O-3 码表：一个 use case 的资产，通过通用加载器读取。

这个文件测的是**癌症资产**，不是框架。分工：`test_code_tables.py` 断言加载器不认识
topography/morphology/behavior 也能工作（一张血脂表也能装进去）；这里断言肺表确实说
C341 是上叶。前者是框架，后者是这个 use case 的事实。

它存在的两次真实失败，2026-07-30 测得，都不是正则或词表能抓的：

  `C187 / 7205 / 0` —— 一次运行在肺癌病人的病历里找到一个真实的乙状结肠增生性息肉并把它
  编码了。`7205` 不是 ICD-O-3 形态码，增生性息肉根本没有形态码（它不是新生物），行为 0
  不可上报。`\\d{4}` 接受了 `7205`，于是 `check_field_formats` 放它过去，
  `answer_shape_miss` 从未触发。

  "ICD-O-3 topography C341 is right middle lobe" —— 一次运行这样断言，然后在引用证据写着
  "right middle lobe" 的情况下编了 C341。C341 是**上**叶。这是表能决定而模型的回忆没能
  决定的事。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acr.contract.code_tables import (
    CODES_DIR,
    EXCLUDED_BY_SPEC,
    MALFORMED,
    NOT_ADMISSIBLE,
    NOT_IN_TABLE,
    OUT_OF_TABLE_SCOPE,
    CodeTableError,
    check_values,
    load_table,
    prompt_block,
)


@pytest.fixture(scope="module")
def t():
    return load_table("icdo3_colorectal")


@pytest.fixture(scope="module")
def lung():
    return load_table("icdo3_lung")


def kinds(problems):
    return {p.kind for p in problems}


def triple(site=None, histology=None, behavior=None) -> dict:
    """旧 `check_codes(site, histology, behavior)` 的三个位置参数，现在是三个**轴名**。

    字段名（primary_site / histology / behavior）从每个轴的 `field` 声明读出来，所以问题
    仍然报在正确的字段上，而函数签名里不再有癌症词。
    """
    return {"topography": site, "morphology": histology, "behavior": behavior}


# ------------------------------------------------------- 那个什么都抓不住的答案
def test_the_case004_answer_is_caught_on_all_three_fields(t):
    """`C187 / 7205 / 0`：不是形态码，行为不可采纳，而 C187 本身没问题。"""
    ks = kinds(check_values(triple("C187", "7205", "0"), table=t))
    assert NOT_IN_TABLE in ks, "7205 is not an ICD-O-3 morphology"
    assert NOT_ADMISSIBLE in ks, "behaviour 0 is benign and not reportable"
    # C187 是真实的结直肠码，所以绝不能报成无效 —— 那次的错误是答错了肿瘤，而这张表无从知道。
    assert OUT_OF_TABLE_SCOPE not in ks


def test_a_four_digit_number_is_not_a_morphology_code(t):
    """被删掉的 `field_format` 永远关不上的那个口子：`\\d{4}` 什么都放过。"""
    for invented in ("7205", "9999", "1234", "8999"):
        assert NOT_IN_TABLE in kinds(check_values(triple(histology=invented), table=t)), invented


def test_a_benign_polyp_is_named_when_its_code_and_behaviour_are_given(t):
    """8210/0 是合法 ICD-O-3 码且不是可上报新生物。两个事实都重要。"""
    p = next(x for x in check_values(triple(histology="8210", behavior="0"), table=t)
             if x.kind == NOT_ADMISSIBLE)
    assert "adenomatous polyp" in p.message.lower()
    assert NOT_IN_TABLE not in kinds(check_values(triple(histology="8210", behavior="3"), table=t)), (
        "8210/3 — carcinoma arising in an adenomatous polyp — IS reportable")


def test_hyperplastic_polyp_has_no_code_at_all(t):
    """不是漏写 `code: null`，这**就是**那个发现：它不是新生物。

    迁移后表现为这条排除项没有 `axis_values` —— 一条匹配不到任何码的排除项若参与匹配，
    会匹配上一切。
    """
    row = next(r for r in t.exclusions if r["term"] == "Hyperplastic polyp")
    assert "axis_values" not in row
    assert "not a neoplasm" in row["why"]


# ------------------------------------------------------- 记法被折叠，而不是被拒绝
@pytest.mark.parametrize("raw,want", [
    ("C18.7", "C187"), ("c18.7", "C187"), ("C187", "C187"), (" C18.7 ", "C187"),
    ("8140/3", "8140"), ("8140", "8140"), ("C34.11", "C3411"),
])
def test_the_punctuated_form_the_manual_writes_is_folded(t, raw, want):
    """被删掉的 `field_format` 6 次有用触发里有 4 次在拒绝 `C34.9`/`C34.11`/`C34.2` ——
    ICD-O-3 自己就是这么写的。它在制造自己再解决的往返。

    规则现在写在表的 `normalization:` 里，不在模块常量里。
    """
    assert t.normalize(raw) == want


def test_the_behaviour_digit_is_split_out_not_merged(t):
    assert t.trailing_part("8140/3") == "3"
    assert t.trailing_part("8140") is None
    assert t.normalize("8140/3") == "8140", "the behaviour digit must not join the morphology"


# ------------------------------------------------------- 范围是一个发现，不是一个错误
def test_a_lung_code_is_out_of_scope_and_not_invalid(t):
    """本仓库的语料是**肺**登记 —— 1,788 个 gold topography 全是 C34x。结直肠表必须说
    "装错表了"而不是"答错了"，否则一张装错的表会让每个病例都像答错。"""
    p = next(x for x in check_values(triple("C341", "8140", "3"), table=t)
             if x.field == "primary_site")
    assert p.kind == OUT_OF_TABLE_SCOPE
    assert "not evidence that" in p.message
    assert "wrong table is loaded" in p.message


def test_a_malformed_topography_is_distinguished_from_an_out_of_scope_one(t):
    assert kinds(check_values(triple("C18"), table=t)) == {MALFORMED}
    assert kinds(check_values(triple("lung"), table=t)) == {MALFORMED}


def test_a_correct_colorectal_answer_has_no_problems(t):
    assert check_values(triple("C187", "8140", "3"), table=t) == []
    assert check_values(triple("C209", "8480", "3"), table=t) == []
    assert check_values(triple("C211", "8070", "3"), table=t) == []


def test_an_absent_field_is_not_a_problem(t):
    """弃答不是本模块的事。"""
    assert check_values(triple(), table=t) == []
    assert check_values(triple("C187", "", "  "), table=t) == []


def test_an_unknown_behaviour_digit_is_its_own_finding(t):
    assert kinds(check_values(triple(behavior="7"), table=t)) == {NOT_IN_TABLE}


# ------------------------------------------------------- NOS 码是真实答案
def test_the_nos_codes_are_in_the_table_and_flagged_not_excluded(t):
    """8000/8010/8046 合起来是登记面对本语料 10.8% 病例自己的答案，C349 占 9.6%。
    一张漏掉它们的表会重建已删除的 `not_less_specific`。"""
    morph = t.axes["morphology"]
    for c in ("8000", "8010", "8046"):
        assert c in morph.codes
        assert c in morph.unspecified
    assert check_values(triple(histology="8010", behavior="3"), table=t) == [], \
        "coding NOS is not a problem"


def test_nos_topography_is_marked(t):
    unspec = t.axes["topography"].unspecified
    assert "C189" in unspec and "C210" in unspec
    assert check_values(triple("C189"), table=t) == []


# ------------------------------------------------------- 提示词渲染
def test_the_prompt_block_renders_the_whole_domain(t):
    """一个只被展示 12 of 40 个形态码的模型会往那 12 个里编码。"""
    b = prompt_block(t)
    for c in t.axes["morphology"].codes:
        assert c in b, f"{c} missing from the prompt block"
    for c in t.axes["topography"].codes:
        assert c in b


def test_the_prompt_block_states_its_own_provenance(t):
    """模型回忆而来、非转录、无人签署。模型必须能拿它和自己真正读过的病理报告掰。

    这三句话迁移后住在表的 `warnings:` 里而不是 Python 里 —— 它们是关于**这张表**的断言。
    """
    b = " ".join(prompt_block(t).split())
    assert "recalled by a language model, not transcribed" in b
    assert "no registrar has checked it" in b
    assert "say so in your reasoning" in b


def test_the_prompt_block_carries_the_exclusions_and_the_safeguards(t):
    b = prompt_block(t)
    assert "Hyperplastic polyp" in b and "no code exists" in b
    assert "NOT admissible" in b
    assert "benign polyp is not the reportable tumour" in b


# ------------------------------------------------------- YAML 的来源纪律
def test_the_yaml_declares_itself_model_recalled_and_unbound():
    """和 `assets/specs/` 里那四张表同一个标准：模型回忆来的表必须自己说出来，并写明人该拿什么核对。"""
    d = yaml.safe_load((CODES_DIR / "icdo3_colorectal.yaml").read_text(encoding="utf-8"))
    sa = d["source_authority"]
    assert sa["origin"] == "model_recalled"
    assert sa["version_binding"] == "NOT_BOUND"
    assert "no clinical or registrar sign-off" in sa["status"]
    # 读**解析后**的字段，不读文件文本。注释说的是同一件事而 `yaml.safe_load` 会全部丢掉，
    # 所以一个只有注释作依据的 `origin: model_recalled` 就是 `assets/specs/` 加 `provenance:` 要
    # 取代的那种"有标签没依据"。
    assert "RECALLED BY A LANGUAGE MODEL" in " ".join(sa["basis"].split())
    assert "not a transcription of ICD-O-3" in " ".join(sa["basis"].split())
    assert "casefinding manual" in " ".join(sa["what_a_human_must_check"].split()), \
        "reportability is a registry policy question and the file must say who settles it"


def test_a_missing_table_raises_and_names_what_exists():
    with pytest.raises(CodeTableError) as e:
        load_table("icdo3_does_not_exist")
    assert "available:" in str(e.value)


def test_there_is_no_default_table_any_more():
    """旧 `load_table(name="icdo3_lung")` 有默认值，于是"忘了声明值域"和"声明了肺"长得一样。

    装错表会让每个病例都像答错（这就是 OUT_OF_TABLE_SCOPE 存在的理由），所以表名必须由
    spec 说出来。这条替换了原来断言"默认表是肺"的那一条 —— 那个默认被有意去掉了。
    """
    with pytest.raises(TypeError):
        load_table()                                    # type: ignore[call-arg]


# ==========================================================================================
# 肺表 —— 抽取语料，以及它为之而建的那些失败
# ==========================================================================================
GT = Path("$ACR_REAL_CORPUS/ground_truth.csv")


def test_the_subsite_digits_a_run_got_wrong(lung):
    """一次运行断言 "ICD-O-3 topography C341 is right middle lobe"，并在证据写着
    "right middle lobe" 时编了 C341。C341 是**上**叶。这就是要一张表的全部理由。"""
    topo = lung.axes["topography"]
    assert topo.name_of("C341") == "Upper lobe, lung"
    assert topo.name_of("C342") == "Middle lobe, lung"
    assert topo.name_of("C343") == "Lower lobe, lung"
    assert topo.name_of("C340") == "Main bronchus"
    assert "NOS" in (topo.name_of("C349") or "")


def test_the_left_lung_has_no_middle_lobe(lung):
    """一次运行在引用证据写着 "left lower lobe" 时编了 C342，九次；登记面编的是 C343。
    记成解剖事实，不是一个叶名正则。"""
    d = yaml.safe_load((CODES_DIR / "icdo3_lung.yaml").read_text(encoding="utf-8"))
    lat = d["laterality"]
    assert "C342" not in lat["left_lung_lobes"]
    assert "C342" in lat["right_lung_lobes"]
    imp = next(r for r in lat["impossible"] if r["subsite"] == "C342")
    assert imp["side"] == "left" and "no left middle lobe" in imp["why"]


def test_the_two_blastomas_are_different_diseases(lung):
    """一次运行编了 8973 而登记面编的是 8972。两个都是真码，所以只有一张同时收录两者并在
    它们之间写下区别的表才能让这次混淆浮出来。"""
    morph = lung.axes["morphology"]
    assert "pulmonary blastoma" in (morph.name_of("8972") or "").lower()
    assert "pleuropulmonary blastoma" in (morph.name_of("8973") or "").lower()
    assert morph.name_of("8972") != morph.name_of("8973")
    # 两者都不是码级错误：那次运行错在读病理，这张表不许假装不是。
    assert check_values(triple("C349", "8973", "3"), table=lung) == []


def test_the_solid_adenocarcinoma_code_a_run_missed_is_in_the_table(lung):
    """CASE003：登记 8230，运行编了 8140。两个都合法；表无法决定谁对，但能让 8230 不像是
    编造的。"""
    assert lung.axes["morphology"].name_of("8230") is not None
    assert check_values(triple("C341", "8230", "3"), table=lung) == []


def test_a_carcinoid_is_malignant_and_reportable(lung):
    """把典型类癌叫作良性是临床习惯，不是 ICD-O-3 的立场。"""
    assert check_values(triple("C341", "8240", "3"), table=lung) == []
    assert lung.axes["behavior"].is_admissible("3") is True


def test_the_nos_codes_that_the_removed_check_pushed_away_from_are_clean(lung):
    """8000/8010/8046 是登记面对本语料 10.8% 的答案，C349 占 9.6%。被删掉的
    `not_less_specific` 拒绝的正是这些，22 次触发全是。"""
    for h in ("8000", "8010", "8046"):
        assert check_values(triple("C349", h, "3"), table=lung) == []


def test_a_haematopoietic_gold_is_a_scope_boundary_not_a_table_gap(lung):
    """本语料有六个病人的 gold histology 是淋巴瘤，而 spec 的 `when_not_to_use` 排除了
    造血系统新生物。那些是 SPEC_INSUFFICIENT 病例；一张把它们报成 NOT_IN_TABLE 的表会把
    范围边界弄成像是表不完整。"""
    p = next(x for x in check_values(triple("C349", "9680", "3"), table=lung)
             if x.field == "histology")
    assert p.kind == EXCLUDED_BY_SPEC
    assert "SPEC_INSUFFICIENT" in p.message
    assert "not a coding error" in p.message
    for c in ("9591", "9699", "9702"):
        assert {x.kind for x in check_values(triple(histology=c), table=lung)} == {
            EXCLUDED_BY_SPEC}


def test_a_colorectal_code_is_out_of_scope_against_the_lung_table(lung):
    p = next(x for x in check_values(triple("C187", "8140", "3"), table=lung)
             if x.field == "primary_site")
    assert p.kind == OUT_OF_TABLE_SCOPE


@pytest.mark.skipif(not GT.is_file(), reason="registry gold is outside the repository")
def test_the_table_validates_against_every_registry_answer_in_the_corpus(lung):
    """真正要紧的那条回归测试，而且免费：对 1,788 条操作员确认过的登记答案做确定性字符串
    匹配，回路里没有模型。

    这张表的第一稿得分 1762/1788，26 个漏项都是登记面真在用的码 —— 8033 梭形细胞癌、
    8256/8257 微浸润腺癌、8550、8574、8141、8002、8023、8144、8800、9180 —— 外加六个
    spec 排除的造血病例。是这条测试找出了它们，也是它会找出下一个漏项。
    """
    import csv
    rows = list(csv.DictReader(GT.open(encoding="utf-8")))
    problems: dict[str, list[str]] = {}
    for r in rows:
        for p in check_values(
                triple(r["gt_primary_site"], r["gt_histology"], r["gt_behavior"]), table=lung):
            problems.setdefault(p.kind, []).append(p.value)

    # 剩下的每个问题都必须是那个已声明的范围边界。别的都是真漏洞。
    unexpected = {k: sorted(set(v)) for k, v in problems.items() if k != EXCLUDED_BY_SPEC}
    assert not unexpected, (
        f"the table does not cover codes the registry actually uses: {unexpected}. "
        f"Add them; a value domain that rejects the registry's own answers is worse than none.")
    assert len(problems.get(EXCLUDED_BY_SPEC, [])) == 6, (
        "six haematopoietic cases were measured in this corpus; a change here means the cohort "
        "or the spec's exclusions moved and the accuracy denominator moved with them")


# ==========================================================================================
# 接缝：Task Contract 声明表，提示词渲染它
# ==========================================================================================
def test_the_lung_spec_declares_its_code_table():
    """一个值属于哪个码系统是答案**含义**的一部分，所以由 spec 说。运行时不从语料或字段名猜。"""
    from acr.contract.spec import load_spec as _ls
    spec = _ls("assets/specs/STORE.400_522_523.site_histology_behavior.yaml")
    assert spec.value_domain == "icdo3_lung"


def test_a_spec_with_no_code_system_gets_no_block():
    """日期和 class-of-case 变量没有 ICD-O-3 值域。塞一墙肺形态码给它们忽略是提示词膨胀。"""
    from acr.contract.code_tables import code_domain_block
    from acr.contract.spec import load_specs as _lss
    blocks = {sid: code_domain_block(sp) for sid, sp in _lss("assets/specs").items()}
    assert blocks["STORE.400_522_523.site_histology_behavior"]
    assert not blocks["STORE.390.date_of_initial_diagnosis"]
    assert not blocks["STORE.610.class_of_case"]


def test_a_declared_table_that_does_not_exist_stops_the_spec_from_loading(tmp_path):
    """打错就 FAIL CLOSED。缺表否则会渲染出空值域，运行看起来就和被给过码表一模一样 ——
    和一个静默不提供任何指导却在 manifest 里报告提供了的 skill 是同一个失败。"""
    from acr.contract.code_tables import CodeTableError as CTE
    from acr.contract.spec import load_spec as _ls
    p = tmp_path / "S.2.yaml"
    p.write_text(
        "spec_id: S.2\nspec_version: 0.1.0\ndata_source: notes\nquestion: q\n"
        "value_domain: icdo3_atlantis\n"
        "fields:\n  - name: primary_site\n    type: string\n"
        "decision_rule: [r]\nevidence_rules:\n  counts_as_evidence: [anything]\n",
        encoding="utf-8")
    with pytest.raises(CTE) as e:
        _ls(p)
    assert "available:" in str(e.value)


def test_the_rendered_block_contains_the_subsite_facts_a_run_got_wrong():
    """端到端：模型真正会读到的那段文字里，C341 旁边写着 'Upper lobe'。"""
    from acr.contract.code_tables import code_domain_block
    from acr.contract.spec import load_spec as _ls
    b = code_domain_block(_ls("assets/specs/STORE.400_522_523.site_histology_behavior.yaml"))
    assert "C341  Upper lobe, lung" in b
    assert "C342  Middle lobe, lung" in b
    assert "C343  Lower lobe, lung" in b
    assert "left lung has no middle lobe" in " ".join(b.split())
    assert "8972" in b and "8973" in b
