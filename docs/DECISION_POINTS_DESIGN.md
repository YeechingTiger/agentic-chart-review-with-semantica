# 决策点设计：taxonomy、run 的分解、依据的出处

2026-08-26。只有设计，没有实现。接入点是现有的六工具 agent run。

## 0 · 三件事怎么咬合

```
taxonomy  ── 给每个决策点一个可比的身份 ──┐
                                          ├──► 同类可比 ⇒ 分歧可算 ⇒ 沉淀有对象
run 分解  ── 保证每一步都有决策点 ────────┤
                                          │
依据出处  ── 记下这个判断凭的是我们给的还是它自己的 ──► 分歧可分诊 ⇒ 修改可路由
```

第三件是新的，也是最要紧的：**一个判断如果凭的是模型自带的知识，那它就发生在合同没覆盖、
没人批准过的地方**。文档 §4.2 把这叫"最危险的角落——规范不清晰 + 执行一致 = 隐性立法"，
并说分歧检测器抓不到它。记下依据出处，是唯一能抓到它的办法。

---

## 1 · 决策点 taxonomy

### 1.1 构造原则

两条来源合并，缺一不可：

- **感知侧** —— 决策沉淀设计 §1.4 的十个决策族，是"忠实分解真人 reviewer 的认知动作"。
  该文档明说十族刻画的是**确定性检索算子覆盖不到、剩给人的**那部分。
- **合成侧** —— `THE_IDEAL_SYSTEM.md` P4 的判断层组合算子（佐证 / 仲裁 / 去重排序 / 推导）。
  这四个在十族里几乎没有（只有仲裁 ≈ 族8），因为十族假定算子是确定性的；
  在我们这里它们**不是**确定性的，它们就是判断。

只取十族会丢掉整个合成侧；只取算子会丢掉整个感知侧。

### 1.2 粒度判据

> **两个类型应当分开，当且仅当它们的分歧会被送给不同的人、改不同的东西。**

taxonomy 存在的唯一理由是路由。四个去处（对应仓库已有的资产分层）：
**工程/先验**（自动回归门）、**临床专家**（裁决→合同条款）、**抽取/先例**（回归测试集）、
**卡片作者**（散文，可背离）。

并且——**细度不住在 type 里**。完整身份是 `{level}:{decision_type}:{situation_slug}`：

| 层 | 干什么 | 性质 |
|---|---|---|
| `level` | 大点 / 小点。决定这个分歧值不值得管 | 封闭，2 个，**由 type 推出，不问模型** |
| `decision_type` | 粗分类：哪一种认知动作 | 封闭，13 + `other`，**稳定** |
| `situation_slug` | 细分类：具体什么情形。**可比性真正住在这里** | 开放，演化，每周归一化 |

把演化压力放在该演化的那一层：type 是封闭词表，改一次要重打历史、要 equivalence map；
slug 本来就该每周归一化。

slug 守三条铁律（§3.4）：受控词表 / **剥离实例细节**（时间、药名、note_id、患者特征
只进 metadata 不进 key）/ **抽象到状态不抽象到结论**。冷启动允许 provisional。

### 1.3 十三类，三组

#### A 组 · 关于一份文档 / 一段文字 —— 一次动作内可定 ⇒ **小点**

| type | 在定什么 | 例子 | 分歧了送给谁 |
|---|---|---|---|
| `where_to_look` | 去哪找、用什么词、开哪一份、要不要扩 | 先看类型分布还是直接搜 | 工程 / Retrieval Prior |
| `is_this_it` | 这段说的是不是目标概念 | "discomfort" 算不算疼痛主诉 | 专家 → 合同的概念定义 |
| `what_it_asserts` | 断言了什么：否定 / 病史 / 假设 / 转述 / **计划 vs 已执行** | "will give morphine" ≠ 已给药 | 抽取 / 先例库 |
| `when_it_happened` | 指向哪个时间：记录时间 vs 事件时间、**copy-forward** | 昨天的 note 今天还写 "POD1" | 抽取（检测）+ 专家（用哪个时间） |
| `standing` | 这份文档对**这个字段**值多少：能确立 / 仅提及 / 无关 | 处方 filled 对"是否输注过"只是仅提及 | 专家 → 合同 evidence rules |

`standing` 是 per-field 的，不是 per-document——同一份处方记录，对"是否取过药"是
can-establish，对"是否输注过"只是 merely-mentions。CONTEXT.md 已把 Standing 定为域语言。

#### B 组 · 关于两条证据之间的关系 —— 需 ≥2 条 ⇒ **大点**

两条证据之间只可能是这三种关系之一，互斥且穷尽：

| type | 在定什么 | 例子 | 分歧了送给谁 |
|---|---|---|---|
| `same_or_ordered` | 是不是同一件事；两件的话谁先 | 处方说 5-07、note 说 "early May"，一次还是两次 | 专家 → 事件同一性规则 |
| `corroborate` | 说的一致，强度叠加 | 三个来源都说做过化疗 | 专家 → 证据阶梯 |
| `which_wins` | 说的不一致，按规则选一个 | 细胞学 vs 活检哪个定日期 | 专家 → **Conflict Rule** |

#### C 组 · 关于案子当前能说什么 —— 需综合全局 ⇒ **大点**

| type | 在定什么 | 分歧了送给谁 |
|---|---|---|
| `scope` | 案子 / 实体在不在范围内；时间锚用哪个 | 专家 → 合同适用性 |
| `infer` | 无文档直接断言，从见证前提推出。欠"前提逐条 witness + 排除竞争候选" | 专家 → 推断政策（P6） |
| `is_it_absent` | 没找到意味着什么。结论分两种：**这份 chart 里没有** vs **这类记录不在这个 corpus** | 卡片作者 + 闸门（可强制那半） |
| `enough` | 手上的证据够不够作答 | 卡片作者 → coverage 卡片（**已测死不可强制**） |
| `what_to_answer` | 满不满足定义、边界怎么算、值怎么规范化、怎么拼 | 专家 → 字段定义 / 格式 / 组合规则 |

`is_it_absent` 的两种结论正好对上合同已声明的 `EVIDENCE_INSUFFICIENT` /
`CORPUS_INSUFFICIENT`，不用拆类型，用结论词表区分。

#### `other`

不认识的 type 归 `other`，**原话保留在 `claimed_type`**，永不拒绝。
`other` 的堆积是唯一能证伪这张表的东西——某个 claimed_type 反复出现就是晋升候选。

### 1.4 层级由 type 推出，不问模型

A 组 → 小点；B、C 组 → 大点。**唯一例外**：`where_to_look` 两级都允许——
"下一段整体往哪找"是大点（需综合前面搜的结果），"这一次用什么词"是小点。
其余十二类归属干净，服务端可校验（在 `search` 里报 `enough` 就是用错了）。

---

## 2 · 一个 run 分解成什么

### 2.1 两级，靠位置归属

**大点 = `note_decision`**：需要若干小步骤才能作出的判断。它同时收两头——
`facing` 写"上一批小步骤做完之后我知道了什么"（= 上一个大点的结论），
`decision` 写"所以下一步做什么"（= 开启本大点）。

**小点 = 每个动作自带**：调 `read(note_id=X)` 的那一刻，"选 X 来读"这个**选择已经完整
表达了**——不必复述。只需补短字段说明"这一小步在定什么"和"哪一类"。

> **小点密度 = 动作密度。原子粒度由构造保证，不靠模型自觉。**

### 2.2 归属规则（服务端按 seq 算，不问模型）

- 动作 `PART_OF` **紧邻在前**的那次 `note_decision`
- 大点之间 `INFLUENCED`（时间序）
- 提交 → 闸门 → 终局 用 `CAUSED`（**模型只能"影响"，代码才能"导致"**——信任边界画在图里）

### 2.3 一个 run 读出来的形状

```
大点1 [where_to_look] 先从文档类型入手，看诊断日期可能住在哪
   └ 小 [where_to_look] 哪类文档可能载有诊断日期
        -> list_documents{}                    观察: 312 份, 19 类
   └ 小 [standing]      病理报告能不能确立诊断日期
        -> read{Surgical-Pathology-Document_2023-04-12}   观察: 419 字, 读完
   └ 小 [what_it_asserts] "suspicious for" 是确诊还是存疑
        -> record_evidence{[295,305)}          观察: 记下 "suspicious"
大点2 [enough] 这些看下来证据不够，我要确定没有同期的临床印象
   └ 小 [where_to_look] 临床印象会在哪类文档里
        -> search{"malignancy"}                观察: 1 命中
   └ 小 [where_to_look] 缺席声称前要看全
        -> list_documents{}                    观察: 312 份
大点3 [which_wins] 有 impression 陪着，细胞学定日期
   └ 小 -> submit_answer{FOUND 20230412}
        ──CAUSED──► 闸门 通过 ──CAUSED──► 终局 20230412
```

审计链（`get_causal_chain` 只认 CAUSED/INFLUENCED/PRECEDENT_FOR）**只显示大点**——
`PART_OF` 不在因果边里，所以小点要展开才看得到。这是想要的：**审计读大点，细看才展开小点。**

---

## 3 · 依据的出处（grounding）

### 3.1 问题

"凭什么这么判"有五种可能的出处，而它们的意义完全不同：

| grounding | 依据是 | 可核验性 |
|---|---|---|
| `contract` | 合同明写的条款：Decision Rule / Conflict Rule / evidence rules / 字段定义 | **可核验**——规则 id 服务端可枚举 |
| `card` | 提示里的方法卡片或任务前言 | **可核验**——装配进 prompt 的卡片列表服务端知道 |
| `precedent` | 检索到的先例 | **可核验**——先例 id 必须在本次 run 被返回过 |
| `chart` | 纯粹是病历里读到的事实 | **可核验**——note / evidence 引用已在核验 |
| `own_knowledge` | **模型自带的临床或常识知识；提供的材料里没有** | **不可核验，而这正是要的** |

一个决策可以有多个 grounding（既引了规则又用了自带知识）。分析关心的是
**是否含 `own_knowledge`**。

### 3.2 为什么这一维值钱

1. **抓隐性立法**：`own_knowledge` 反复出现在同一个 `(type, slug)` 上 = 模型在没人批准
   的地方形成了惯例。这是文档 §4.2 说"分歧检测器抓不到"的那类东西，而 grounding 抓得到。
2. **合同成熟度**：`contract` 占比上升、`own_knowledge` 占比下降 = 合同在吸收 agent 的知识。
   这是 D2「B−C 分歧收敛」的另一种形式，而且**零标注**。
3. **编造依据可检测**：声称 `contract` 但引用的 rule id 不在合同里 = 凭空捏造。
   已核实 `STORE.390` 服务端可枚举 `conflict_rules`(4)、
   `discriminating_facts`(`impression_at_ambiguous_cytology`,
   `physician_statement_predating_tissue`)、`decision_rule`、`evidence_rules`。

### 3.3 图上的形状 —— `own_knowledge` 是"没有出边"

```
decision ──cites──►  Rule(conflict_rule.2)        grounding=contract
decision ──cites──►  Card(coverage-judgement)     grounding=card
decision ──uses───►  Evidence(note+span)          grounding=chart
decision ──follows─► Precedent(decision_id)       grounding=precedent
decision             （没有任何 cites 边）         grounding=own_knowledge
```

于是"哪些判断是模型自己拿的主意"变成一句图查询：**找没有 `cites` 出边的 step 节点。**
不需要读一个字的散文。

### 3.4 服务端核验（永不拒绝，只标记）

| 声称 | 检查 | 不通过时 |
|---|---|---|
| `contract` | `used` 里有 `rule:<id>`，且 id 在已加载的 spec 里存在 | 记 `grounding_unverified`，照常放行 |
| `card` | card 名在本次 run 装配的卡片清单里 | 同上 |
| `precedent` | 该 precedent id 本次 run 被返回过 | 同上 |
| `chart` | note / evidence 引用可核验（现有 `_resolve_used` 已做） | 同上 |
| `own_knowledge` | 不检查 | —— |

**永不拒绝**：一个不可核验的引用是**发现**，不是错误。拒绝它只会教模型别说实话。

---

## 4 · 接进现有的六工具面

现有：`list_documents` / `search` / `read` / `note_decision` / `record_evidence` / `submit_answer`

| 工具 | 改动 |
|---|---|
| `list_documents`<br>`search`<br>`read` | `objective`（现在被 `del` 丢弃）换成三个**必填**：<br>`deciding`（这一小步在定什么）· `decision_type`（**限 A 组**）· `grounding`（数组）<br>可选 `because`。**停止丢弃**，进 trace 的 result 与账本 |
| `record_evidence` | 加必填 `decision_type`（限 A 组：`standing`/`what_it_asserts`/<br>`when_it_happened`/`is_this_it`）· `deciding` · `grounding`。<br>`supports` 保持——它是这条证据支持什么，不是决策 |
| `note_decision` | `decision_type` **限 B/C 组**（+`where_to_look` 例外）；加必填 `grounding`。<br>其余字段（`facing`/`decision`/`because`/`used`/`options`）不变 |
| `submit_answer` | **不变**。它是终局，闸门管，不是决策点叙述 |

**没有新工具。** 小点长在动作参数里，大点用现有的 `note_decision`。

`used` 的引用词表扩两种（现有 `note:` / `search:` / `evidence:` / `rule:` / `decision:`）：
`card:<name>`、`precedent:<id>`。

### 4.1 服务端每步多算的（都是确定性的）

- `level`：由 `decision_type` 推出（A→小、B/C→大），不问模型
- `parent`：紧邻在前的 `note_decision` 的 seq
- `grounding_unverified`：§3.4 的核验结果
- 现有的 `context` 快照与 `_resolve_used` 照旧

### 4.2 prompt 侧

任务前言改成两级，并说明 grounding：
"每次动作说明你这一小步在定什么、哪一类、凭什么来源；一批步骤做完、你得出结论并决定
下一步时，叫一次 `note_decision`。**凭自己的临床知识判断没有问题，但要如实标成
`own_knowledge`**——它不会被拒绝，它帮我们看出合同哪里没写。"

最后一句是设计的关键：**只有当自认不受惩罚时，自认才有信息量。**

---

## 5 · 图上的完整形状

```
节点
  Decision(step)     category=step:{type}  metadata={level, slug, grounding,
                                                     parent_seq, context, seq, stamps}
  Decision(submit / gate / result)         （现有）
  Evidence           note_id + [start,end) + 逐字原文
  Rule               合同条款 id
  Card               方法卡片名
边
  step   ──PART_OF────►  step(大点)            小挂大
  step   ──INFLUENCED─►  step(大点→大点)        时间序
  submit ──CAUSED─────►  gate ──CAUSED──► result
  step   ──uses───────►  Evidence
  step   ──cites──────►  Rule | Card
  step   ──follows────►  Decision(先例)
  step   ──PRECEDENT_FOR►  step                 裁决之后才连
```

六戳（`case_id` / `run_id` / `step_id` / `spec_hash` / `prompt_hash` / `agent_version`）——
少任何一个，对应那类混淆就无法排除。现在有前四个，缺后两个。

---

## 6 · 这一版不做的

- **PointDefinitionRegistry**（服务端管 slug，模型只能选 ID，未知即 `UNMAPPED_POINT`）——
  冷启动阶段先允许 provisional slug，跑出真实分布再收口
- **先例检索工具**（`recall`）——先例库要人裁决才立得起来，等有量再说
- **impact 反事实 replay**、**三分诊**、**裁决工作流**——都在这一版之上
- **词表重打**——`other` 堆积到能看出模式之前不动 type

---

## 7 · 跑起来之后，用这三个数判断设计对不对

1. **同一 type 内 slug 的重复率** —— 从不重复 ⇒ type 太细或 slug 违反铁律2；
   重复但同 slug 内的决策不可比 ⇒ type 太粗
2. **`other` 里堆了什么** —— 唯一能证伪这张表的东西
3. **`own_knowledge` 的占比与集中度** —— 集中在哪几个 `(type, slug)` 上，
   那几个就是合同该补的地方；占比随合同版本下降，就是成熟度曲线
