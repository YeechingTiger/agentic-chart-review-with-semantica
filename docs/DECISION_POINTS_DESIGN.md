# 决策点设计：taxonomy、怎么写下来、run 的分解、依据的出处

2026-08-26。只有设计，没有实现。接入点是现有的六工具 agent run。

## 0 · 四件事怎么咬合

```
taxonomy      ── 给每个决策点一个可比的身份 ──┐
四字段语义    ── 保证同类的实例真的能放一起比 ─┤
                                              ├─► 分歧可算 ⇒ 沉淀有对象
run 分解      ── 保证每一步都有决策点 ────────┤
依据出处      ── 记下凭的是我们给的还是它自己的 ─► 分歧可分诊 ⇒ 修改可路由
```

第四件是最要紧的：**一个判断如果凭的是模型自带的知识，那它就发生在合同没覆盖、没人批准过
的地方**。文档 §4.2 把这叫"最危险的角落——规范不清晰 + 执行一致 = 隐性立法"，并说分歧
检测器抓不到它。记下依据出处，是唯一能抓到它的办法。

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

### 1.3 十三类，三组

#### A 组 · 关于一份文档 / 一段文字 ⇒ **小点**

**注意"什么时候发生"这一列——它是 §3 分解设计的关键。**

| type | 在定什么 | 什么时候发生 | 分歧了送给谁 |
|---|---|---|---|
| `where_to_look` | 去哪找、用什么词、开哪一份、要不要扩 | **取信息之前** | 工程 / Retrieval Prior |
| `is_this_it` | 这段说的是不是目标概念 | **看到内容之后** | 专家 → 合同的概念定义 |
| `what_it_asserts` | 否定 / 病史 / 假设 / 转述 / **计划 vs 已执行** | **看到内容之后** | 抽取 / 先例库 |
| `when_it_happened` | 记录时间 vs 事件时间、**copy-forward** | **看到内容之后** | 抽取（检测）+ 专家（用哪个） |
| `standing` | 这份文档对**这个字段**值多少 | **看到内容之后** | 专家 → 合同 evidence rules |

`standing` 是 per-field 的，不是 per-document——同一份处方记录，对"是否取过药"是
can-establish，对"是否输注过"只是 merely-mentions。CONTEXT.md 已把 Standing 定为域语言。

#### B 组 · 关于两条证据之间的关系 ⇒ **大点**

两条证据之间只可能是这三种关系之一，互斥且穷尽：

| type | 在定什么 | 分歧了送给谁 |
|---|---|---|
| `same_or_ordered` | 是不是同一件事；两件的话谁先 | 专家 → 事件同一性规则 |
| `corroborate` | 说的一致，强度叠加 | 专家 → 证据阶梯 |
| `which_wins` | 说的不一致，按规则选一个 | 专家 → **Conflict Rule** |

#### C 组 · 关于案子当前能说什么 ⇒ **大点**

| type | 在定什么 | 分歧了送给谁 |
|---|---|---|
| `scope` | 案子 / 实体在不在范围内；时间锚用哪个 | 专家 → 合同适用性 |
| `infer` | 无文档直接断言，从见证前提推出。欠"前提逐条 witness + 排除竞争候选" | 专家 → 推断政策（P6） |
| `is_it_absent` | 没找到意味着什么 | 卡片作者 + 闸门（可强制那半） |
| `enough` | 手上的证据够不够作答 | 卡片作者 → coverage 卡片（**已测死不可强制**） |
| `what_to_answer` | 满不满足定义、边界、值规范化、组合 | 专家 → 字段定义 / 格式 / 组合规则 |

#### `other`

不认识的 type 归 `other`，**原话保留在 `claimed_type`**，永不拒绝。
`other` 的堆积是唯一能证伪这张表的东西——某个 claimed_type 反复出现就是晋升候选。

### 1.4 层级由 type 推出，不问模型

A 组 → 小点；B、C 组 → 大点。**唯一例外**：`where_to_look` 两级都允许——
"下一段整体往哪找"是大点，"这一次用什么词"是小点。其余十二类归属干净，
服务端可校验（在 `search` 里报 `enough` 就是用错了）。

---

## 2 · 一个决策点怎么写下来：四个字段

semantica 的 `record_decision` 收四个语义字段。**它们各装什么，决定了整个沉淀机制成不成立**
——`scenario` 是等价类的键，写坏了同类就聚不到一起，熵就是假的。

### 2.1 判据：铁律 3

决策沉淀设计 §3.4 的第三条铁律：

> **抽象到状态不抽象到结论**——key 描述"处于什么情形"，**不预设"该选什么"**。

由它推出一个可操作的测试：

> **把这次决策的结论抹掉，`scenario` 还站得住吗？** 站得住 → `scenario`；塌了 → `reasoning` 或 `outcome`。

### 2.2 四个字段的分工

| 字段 | 装什么 | 约束 |
|---|---|---|
| `category` | `step:{decision_type}` | 封闭词表 |
| `scenario` | **局面的类型**——决策做出**之前**就为真的状态 | 剥离实例细节；不预设结论；**和 `outcome` 一起决定可比性** |
| `reasoning` | **为什么在这个局面下选这个**——引用的规则、权衡、判定 | 可以具体、可以长；**不参与聚类，只在裁决时被人读** |
| `outcome` | **选了什么** | 尽量来自封闭候选集（见 §2.5） |

一条贯穿的原则：**`scenario` 必须抽象（它是聚类的键），`reasoning` 可以随便具体
（它不进熵计算）。**

### 2.3 "已经收集了什么信息"该进哪

**拆成两半：**

| | 去哪 | 谁写 |
|---|---|---|
| **抽象形态**："已有一份歧义细胞学作为候选，存在更晚的确诊标本" | `scenario` | 模型 |
| **具体清单**：哪几份文档、几条证据、搜过什么词 | **`context` 快照** | **服务端自己测，不问模型** |

理由是铁律 2：**实例细节（时间、药名、note_id、患者特征）只进 metadata 不进 key**——
写了 `note_id` 进 `scenario`，第二个病人就聚不到一起，这个点永远不会重复出现。

这个分工有额外好处：**两半可以对账。** 模型说"已有一份歧义细胞学"，服务端快照说
`n_evidence: 0` —— 矛盾立刻可见，不用信任任何叙述。

### 2.4 "现在还缺什么信息"该进哪

**先分机械的还是判断的：**

- **机械的**（合同要求 X，手上没有 X）→ **服务端算，进 `context`**。闸门本来就知道
  value 答案欠证据、abstain 欠全量列表，不该问模型。
- **判断出来的**（"我认为需要查 impression"）→ **这是这次决策的产物，不是它的输入**：
  - 判定"不够" → **`outcome`**
  - "因为 conflict_rule.1 turns_on impression，而它没被查过" → **`reasoning`**
  - **绝不进 `scenario`** —— 写进去就是把结论塞回了前提，铁律 3 直接违反

### 2.5 outcome 词表：按 type 定，六类天然封闭

§3.5 要求 typed choice——"自由文本没法算离散度"。

| type | outcome 词表 |
|---|---|
| `standing` | `can_establish` / `merely_mentions` / `neither`（CONTEXT.md 已有的域语言） |
| `what_it_asserts` | `asserted` / `negated` / `historical` / `hypothetical` / `planned` / `reported_by_other` |
| `is_this_it` | `is_target` / `not_target` / `unclear` |
| `when_it_happened` | `event_time_stated` / `recorded_time_only` / `carried_forward` / `undatable` |
| `is_it_absent` | `absent_in_chart` / `absent_from_corpus` / `found`（对应合同已声明的两种弃答 status） |
| `enough` | `enough` / `not_enough` |
| `which_wins` | **本次的候选集之一**——不是固定词表，但是封闭的 |
| `same_or_ordered` | `same_event` / `distinct_a_first` / `distinct_b_first` / `distinct_order_unknown` |
| `corroborate` | `reinforces` / `independent_but_same` / `not_actually_about_the_same` |
| `where_to_look` | 动作本身（结构化的工具调用，不是自由文本） |
| `scope` / `infer` / `what_to_answer` | 本次的候选集 |

**六类天然封闭，其余用本次候选集。** 所以熵是能算的，不必等 slug 归一化。

### 2.6 一个完整例子（`enough` 大点，SYNX03）

**对的：**

```
category   step:enough
scenario   歧义细胞学与更晚的确诊标本并存，两个候选日期，
           尚未查证同期是否有临床印象
reasoning  conflict_rule.1 与 conflict_rule.2 都 turns_on
           impression_at_ambiguous_cytology，该事实未经查证，任一条都还不能适用
outcome    not_enough
context    {n_searches:1, n_evidence:0, unfiltered_listing_done:false,
            documents_read:[Surgical-Pathology-Document_2022-02-14]}   ← 服务端
grounding  [contract]     used: [rule:conflict_rule.1]
```

**两种典型错法：**

```
scenario  "缺临床印象所以证据不够"                    ✗ 抹掉结论就塌了 —— outcome 混进来了
scenario  "已读 Surgical-Pathology-Document_2022-02-14（419字），搜过 suspicious"
                                                      ✗ 实例细节 —— 这个点永不重复
```

### 2.7 另外两组各一个例子

**A 组 · `standing`**
```
scenario   病理报告用存疑措辞（"suspicious for"）描述恶性
outcome    merely_mentions
reasoning  存疑不是诊断陈述；evidence_rules 要求病理明确断言
```

**C 组 · `is_it_absent`**
```
scenario   已做过一次无过滤全量列表，目标类型的文档一份都没有
outcome    absent_from_corpus                （不是 absent_in_chart）
reasoning  312 份里没有任何肿瘤科或临床印象类文档；
           这不是搜索没搜到，是这类记录不在本 corpus
```

---

## 3 · 一个 run 分解成什么

### 3.1 先把工具按"对世界做了什么"分清

| 工具 | 真正在干什么 |
|---|---|
| `list_documents` `search` `read` | **取信息**——只有这三个真的从病历里拿东西回来 |
| `record_evidence` `note_decision` `submit_answer` | **写下来**——不取任何新信息 |

### 3.2 缺口：看到内容之后的判断没有地方放

一次循环的真实结构：

```
想（要看什么）→ 取信息 → 文本回来 → 想（这文本意味着什么）→ ...
                                     ↑
                              这一步原本没有任何工具
```

"动作即选择"（调 `read(note_id=X)` 已经表达了"我选 X 来读"）只覆盖**取信息之前**的判断。
对照 §1.3 A 组的"什么时候发生"列：**五类里只有 `where_to_look` 是动作之前的，
其余四类全部发生在看到内容之后**——原本一类都没有地方放。

最糟的一个后果：现在只有当模型决定"这是证据"时才会调 `record_evidence`，所以
**读了一份发现无关，那个结论（`standing = neither`）永远不会被记下来**——
而"读了 8 份、7 份判为无关"恰恰是 coverage 的真实分母，也是 Retrieval Prior 的原料。

### 3.3 修正：`record_evidence` 扩展为 `record_finding`

不加新工具，把它扩展成"读完一份文档、对一个字段的结论"：

```
record_finding(note_id, field, standing, asserts, span?, when?, carried_forward?,
               deciding, grounding)
```

- `standing` ∈ `can_establish` / `merely_mentions` / `neither`
- `can_establish` 时 `span` **必填**（要指出凭哪句；服务端照旧自己解析原文，模型贴不进来）
- `neither` 时 `span` 可空——**读完判为无关也要记一条**
- 一条 finding 同时承担 A 组那四类判断

**这个形状仓库里已经有了——就是 `labelling.py` 的 `NoteLabel`**（per-field 的 standing +
断言的值 + 逐字 span + copy-forward）。区别只在：labelling 是离线穷举、**没有局面**；
run 里的这一条**有局面**。记录形状相同，来源不同。

### 3.4 两级，靠位置归属

**大点 = `note_decision`**：需要若干小步骤才能作出的判断。它同时收两头——
`facing`（→ `scenario`）写"上一批小步骤做完之后我知道了什么"，
`decision`（→ `outcome`）写"所以下一步做什么"。

**小点 = 取信息动作自带（动作前）+ `record_finding`（看完后）**。

> **小点密度 = 动作密度。原子粒度由构造保证，不靠模型自觉。**

**归属规则（服务端按 seq 算，不问模型）：**

- 小点 `PART_OF` **紧邻在前**的那次 `note_decision`
- 大点之间 `INFLUENCED`（时间序）
- 提交 → 闸门 → 终局 用 `CAUSED`——**模型只能"影响"，代码才能"导致"**，信任边界画在图里

### 3.5 一个 run 读出来的形状

```
大点1 [where_to_look] 先从文档类型入手，看诊断日期可能住在哪
   └ 小 [where_to_look] 哪类文档可能载有诊断日期
        -> list_documents{}                     观察: 312 份, 19 类
   └ 小 [where_to_look] 病理是最强的锚
        -> read{Surgical-Pathology-Document_2023-04-12}   观察: 419 字, 读完
   └ 小 [standing] 存疑措辞的细胞学对这个字段值多少
        -> record_finding{merely_mentions, span=[295,305)}
大点2 [enough] 这些看下来证据不够，我要确定没有同期的临床印象
   └ 小 [where_to_look] 临床印象会在哪类文档里
        -> search{"malignancy"}                 观察: 1 命中
   └ 小 [where_to_look] 缺席声称前要看全
        -> list_documents{}                     观察: 312 份
大点3 [which_wins] 有 impression 陪着，细胞学定日期
   └ 小 -> submit_answer{FOUND 20230412}
        ──CAUSED──► 闸门 通过 ──CAUSED──► 终局 20230412
```

审计链（`get_causal_chain` 只认 CAUSED / INFLUENCED / PRECEDENT_FOR）**只显示大点**——
`PART_OF` 不在因果边里。这是想要的：**审计读大点，细看才展开小点。**

### 3.6 还没有归宿的一类：搜索结果之后的结论

"这个词没用"、"这类文档里没有"——不是 per-document 的，落不进 `record_finding`。

现在由**下一个动作的 `deciding`** 隐式承载（"上次搜 carcinoma 只找到病理，改搜
malignancy 找临床印象"）。够用，但要承认它是隐式的：结论藏在下一步的理由里，不是独立一条。
**先不加**——等真跑，看这类结论是不是重要到需要独立记录（比如要沉淀"哪些词是死词"）。

---

## 4 · 依据的出处（grounding）

### 4.1 五种出处

| grounding | 依据是 | 可核验性 |
|---|---|---|
| `contract` | 合同明写的条款：Decision Rule / Conflict Rule / evidence rules / 字段定义 | **可核验**——规则 id 服务端可枚举 |
| `card` | 提示里的方法卡片或任务前言 | **可核验**——装配进 prompt 的卡片清单服务端知道 |
| `precedent` | 检索到的先例 | **可核验**——id 必须在本次 run 被返回过 |
| `chart` | 纯粹是病历里读到的事实 | **可核验**——note / evidence 引用已在核验 |
| `own_knowledge` | **模型自带的临床或常识知识；提供的材料里没有** | **不可核验，而这正是要的** |

一个决策可以有多个 grounding。分析关心的是**是否含 `own_knowledge`**。

### 4.2 为什么这一维值钱

1. **抓隐性立法**：`own_knowledge` 反复出现在同一个 `(type, slug)` 上 = 模型在没人批准
   的地方形成了惯例。这是文档 §4.2 说"分歧检测器抓不到"的那类东西，而 grounding 抓得到。
2. **合同成熟度**：`contract` 占比上升、`own_knowledge` 占比下降 = 合同在吸收 agent 的
   知识。这是 D2「B−C 分歧收敛」的另一种形式，而且**零标注**。
3. **编造依据可检测**：声称 `contract` 但引用的 rule id 不在合同里 = 凭空捏造。
   已核实 `STORE.390` 服务端可枚举 `conflict_rules`(4)、
   `discriminating_facts`(`impression_at_ambiguous_cytology`,
   `physician_statement_predating_tissue`)、`decision_rule`、`evidence_rules`。

### 4.3 图上的形状 —— `own_knowledge` 是"没有出边"

```
decision ──cites──►  Rule(conflict_rule.2)        grounding=contract
decision ──cites──►  Card(coverage-judgement)     grounding=card
decision ──uses───►  Evidence(note+span)          grounding=chart
decision ──follows─► Precedent(decision_id)       grounding=precedent
decision             （没有任何 cites 边）         grounding=own_knowledge
```

"哪些判断是模型自己拿的主意"变成一句图查询：**找没有 `cites` 出边的 step 节点。**
不需要读一个字的散文。

### 4.4 服务端核验：永不拒绝，只标记

| 声称 | 检查 | 不通过时 |
|---|---|---|
| `contract` | `used` 里有 `rule:<id>`，且 id 在已加载的 spec 里存在 | 记 `grounding_unverified`，照常放行 |
| `card` | card 名在本次 run 装配的卡片清单里 | 同上 |
| `precedent` | 该 id 本次 run 被返回过 | 同上 |
| `chart` | note / evidence 引用可核验（现有 `_resolve_used` 已做） | 同上 |
| `own_knowledge` | 不检查 | —— |

**永不拒绝**：一个不可核验的引用是**发现**，不是错误。拒绝它只会教模型别说实话。

---

## 5 · 接进现有的工具面

| 工具 | 性质 | 改动 |
|---|---|---|
| `list_documents`<br>`search`<br>`read` | 取信息 | `objective`（现在被 `del` 丢弃）换成必填 `deciding` · `decision_type`（**限 `where_to_look`**）· `grounding`；可选 `because`。**停止丢弃**，进 trace 与账本 |
| **`record_finding`**<br>（原 `record_evidence`） | **看完内容后的判断** | 见 §3.3。`standing` 三值；`span` 在 `can_establish` 时必填、`neither` 时可空；加 `deciding` · `grounding`。承担 A 组另外四类 |
| `note_decision` | 大点 | `decision_type` **限 B/C 组**（+`where_to_look` 例外）；加必填 `grounding`。其余字段不变 |
| `submit_answer` | 终局 | **不变**——闸门管，不是决策叙述 |

**没有新工具。** `used` 的引用词表扩两种（现有 `note:` / `search:` / `evidence:` /
`rule:` / `decision:`）：`card:<name>`、`precedent:<id>`。

### 5.1 服务端每步多算的（都是确定性的）

`level`（由 `decision_type` 推出）· `parent`（紧邻在前的 `note_decision` 的 seq）·
`grounding_unverified`（§4.4）。现有的 `context` 快照与 `_resolve_used` 照旧。

### 5.2 prompt 侧

任务前言改成两级，并说明 grounding：

> 每次取信息说明你这一小步在定什么、凭什么来源；看完内容后用 `record_finding` 记下
> 这份文档对这个字段值多少、断言了什么——**即使判为无关也要记**。一批步骤做完、你得出
> 结论并决定下一步时，叫一次 `note_decision`。**凭自己的临床知识判断没有问题，但要如实
> 标成 `own_knowledge`**——它不会被拒绝，它帮我们看出合同哪里没写。

最后一句是设计的关键：**只有当自认不受惩罚时，自认才有信息量。**

---

## 6 · 图上的完整形状

```
节点
  Decision(step)     category=step:{type}
                     metadata={level, slug, grounding, parent_seq, context, seq, stamps}
  Decision(submit / gate / result)          （现有）
  Evidence           note_id + [start,end) + 逐字原文
  Rule               合同条款 id
  Card               方法卡片名
边
  step   ──PART_OF─────►  step(大点)          小挂大
  step   ──INFLUENCED──►  step(大点→大点)      时间序
  submit ──CAUSED──────►  gate ──CAUSED──► result
  step   ──uses────────►  Evidence
  step   ──cites───────►  Rule | Card
  step   ──follows─────►  Decision(先例)
  step   ──PRECEDENT_FOR► step               裁决之后才连
```

六戳（`case_id` / `run_id` / `step_id` / `spec_hash` / `prompt_hash` / `agent_version`）——
少任何一个，对应那类混淆就无法排除。现在有前四个，缺后两个。

---

## 7 · 这一版不做的

- **PointDefinitionRegistry**（服务端管 slug，模型只能选 ID，未知即 `UNMAPPED_POINT`）——
  冷启动先允许 provisional slug，跑出真实分布再收口
- **先例检索工具**（`recall`）——先例库要人裁决才立得起来，等有量再说
- **搜索结果之后的独立结论**（§3.6）
- **impact 反事实 replay**、**三分诊**、**裁决工作流**
- **词表重打**——`other` 堆积到能看出模式之前不动 type

---

## 8 · 跑起来之后，用这四个数判断设计对不对

1. **同一 type 内 slug 的重复率** —— 从不重复 ⇒ type 太细或 `scenario` 混进了实例细节
   （违反铁律 2）；重复但同 slug 内的决策不可比 ⇒ type 太粗
2. **`other` 里堆了什么** —— 唯一能证伪 taxonomy 的东西
3. **`own_knowledge` 的占比与集中度** —— 集中在哪几个 `(type, slug)` 上，那几个就是
   合同该补的地方；占比随合同版本下降，就是成熟度曲线
4. **`record_finding` 里 `neither` 的比例** —— 读了多少、其中多少判为无关。
   这是 coverage 的真实分母，现在完全测不到
