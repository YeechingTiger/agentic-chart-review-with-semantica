# User Story：病人有没有肺癌

2026-08-26。一个端到端的故事，用来检验 [`DECISION_POINTS_DESIGN.md`](DECISION_POINTS_DESIGN.md)：
同一个病人、同一个问题，合同薄和合同厚各跑一遍，看决策点长什么样，最后看审计能查出什么。

## 0 · 这个故事测什么

选肺癌是因为它压到 `STORE.390` 压不到的地方：

| 要测的 | 靠什么压出来 |
|---|---|
| **弱证据能不能合并成强的** | 患者 P 没有病理，只有影像 + 诊断码 + 治疗三条弱证据 |
| **conflicts 怎么算** | 复杂版给一条"病理赢过影像"的 conflict rule |
| **合同薄 vs 合同厚的差别是可测的** | 同一个 run，`grounding` 的分布完全不同 |
| **审计能不能抓到隐性立法** | 简单版里 agent 必须自己发明一条临床规则 |

**核心主张**：两个版本可能给出**同一个答案**，但**依据的出处完全不同**——
而审计看的正是这个差别，不是答案。

---

## 1 · 患者 P（构造，文档类型取自真实语料）

| 文档 | 内容要点 |
|---|---|
| `Chest-CT-W-Contr_2023-04-06` | 3.1 cm spiculated RUL mass, **highly suspicious for** primary bronchogenic carcinoma |
| `PET-CT-Skull-Base-Thigh_2023-04-07` | hypermetabolic RUL mass, SUVmax 12.4 |
| `Onc-Med-MD-OP-Progress-Note_2023-04-20` | 问题列表 "lung ca (C34.1)"；正文 "will start carbo/pem" |
| `Prescriptions-Filled-RxHub_2023-05-07` | carboplatin filled |
| `Onc-Med-MD-OP-Progress-Note_2023-06-15` | "s/p 2 cycles carbo/pem, tolerating well" |
| `Onc-Med-MD-OP-Progress-Note_2023-07-19` | "s/p 2 cycles carbo/pem" — **同一句，copy-forward** |
| **没有任何病理文档** | 活检做了，报告不在本院 |

这是最难的一档：**答案很可能是 yes，但没有一条证据单独够格。**

---

## 2 · 简单版任务

### 2.1 合同

```yaml
question: 这个病人有没有肺癌？
fields:
  - name: has_lung_cancer
    values: [yes, no, cannot_tell]
```

**没有 evidence rules，没有 conflict rules，没有 corroboration 阶梯。**
合同没说什么算确诊——所以那个判断只能由 agent 自己做。

### 2.2 run 展开（决策点）

```
大点1 [where_to_look]  判断有没有肺癌，最强的锚是病理，先找它
      grounding [own_knowledge]                      ← 合同没说病理最强
  └ 小 [where_to_look] 病理类文档在哪
        -> search{"carcinoma"}                        观察: 3 命中，全是影像
  └ 小 [where_to_look] 换个词
        -> search{"pathology"}                        观察: 0 命中
  └ 小 [standing] 影像说 "highly suspicious for"，这算不算确立
        -> record_finding{Chest-CT-2023-04-06, has_lung_cancer,
                          standing=merely_mentions, asserts=asserted,
                          span=[..]}
        grounding [own_knowledge]                     ← 合同没定义 standing

大点2 [is_it_absent]   没有任何病理文档
      outcome  absent_from_corpus
      grounding [chart]      used [search:carcinoma, search:pathology]
  └ 小 [where_to_look] 缺席声称前要看全
        -> list_documents{}                           观察: 287 份, 16 类，无病理类

大点3 [corroborate]    影像 suspicious + 问题列表 C34.1 + 接受了肺腺癌方案化疗，
                       三条弱证据合起来够不够
      outcome  reinforces
      reasoning "三条独立来源指向同一部位同一时期；临床上这个组合足以认定肺癌"
      grounding [own_knowledge]                       ← ★ 这是关键的一条
      used     [note:Chest-CT-2023-04-06,
                note:Onc-Med-2023-04-20,
                note:Onc-Med-2023-06-15]
  └ 小 [standing] 问题列表里的 C34.1
        -> record_finding{merely_mentions}   grounding [own_knowledge]
  └ 小 [standing] carbo/pem 是肺腺癌方案
        -> record_finding{merely_mentions}   grounding [own_knowledge]

大点4 [enough]  够了
  └ 小 -> submit_answer{has_lung_cancer: yes}
        ──CAUSED──► 闸门 通过 ──CAUSED──► 终局 yes
```

**答案是 yes，看起来很合理。** 但十个判断里有六个的 grounding 是 `own_knowledge`。

---

## 3 · 复杂版任务

### 3.1 合同多了三样

```yaml
question: 这个病人有没有肺癌？由什么证据认定？

evidence_rules:                      # ← 多的第一样：standing 的定义
  can_establish:
    - 病理或细胞病理报告中，病理医师明确断言恶性
  merely_mentions:
    - 影像的 "suspicious for" / "concerning for" 类措辞
    - 问题列表或诊断码
    - 针对该癌种的系统治疗记录
    - 他处诊断的转述

conflict_rules:                      # ← 多的第二样
  - id: conflict_rule.1
    if:   病理断言良性，而影像断言可疑
    then: 病理赢
    turns_on: [pathology_specimen_from_the_same_lesion]

corroboration:                       # ← 多的第三样，正面回答"弱证据能不能合并"
  - id: corroboration.1
    if:   ≥3 条相互独立的 merely_mentions，指向同一解剖部位与同一时期
    then: 可支持 yes，但状态必须标 INFERRED
    independence_test: 同一句话被 copy-forward 到多份文档，算一条不算多条

fields:
  - name: has_lung_cancer
    values: [yes, no, cannot_tell]
  - name: basis
    values: [DOCUMENTED, INFERRED]   # ← 推断出来的 yes 和病理确诊的 yes 不能长得一样
```

### 3.2 run 展开（只列与简单版不同的地方）

```
大点1 [where_to_look]  病理是唯一的 can_establish，先找它
      grounding [contract]   used [rule:evidence_rules.can_establish]   ← 从 own_knowledge 变了
      ...

大点3 [corroborate]    三条 merely_mentions，够不够
      outcome  reinforces
      reasoning "corroboration.1：≥3 条独立的 merely_mentions 指向同一部位同一时期"
      grounding [contract]   used [rule:corroboration.1,
                                   note:Chest-CT-2023-04-06,
                                   note:Onc-Med-2023-04-20,
                                   note:Onc-Med-2023-06-15]
  └ 小 [when_it_happened] 04-20 和 06-15 里 "carbo/pem" 是不是同一句被抄下来的
        -> record_finding{Onc-Med-2023-07-19, standing=neither,
                          when=carried_forward}
        grounding [contract]   used [rule:corroboration.1.independence_test]
                                                       ← ★ 独立性检查，简单版完全没做

大点4 [enough]  够了
  └ 小 -> submit_answer{has_lung_cancer: yes, basis: INFERRED}   ← 多了 basis
```

**答案同样是 yes。但依据的出处从六条 `own_knowledge` 变成了 `contract`，
而且多做了一次独立性检查，多产出了一个 `basis=INFERRED`。**

---

## 4 · 审计：能查出什么

以下每一条都写明：**现象 → 怎么查出来的 → 意味着什么 → 该修哪儿**。

### 4.1 简单版查出的

**① 隐性立法：一条没人批准的临床规则**

- **现象**：大点3 `corroborate` 的 grounding = `own_knowledge`，reasoning 写着
  "临床上这个组合足以认定肺癌"
- **怎么查**：图查询——找 `category=step:corroborate` 且**没有 `cites` 出边**的节点
- **意味着**：模型自己立了一条 corroboration 规则。它可能是对的，但**没人批准过**，
  也没写在任何地方，下一版模型可能立一条不一样的
- **该修哪儿**：合同补 corroboration 阶梯（复杂版补的正是这条）

**② `own_knowledge` 的集中度就是合同的洞**

- **现象**：10 个判断里 6 个 `own_knowledge`，全部集中在 `standing` 和 `corroborate`
- **怎么查**：`decisions(category_prefix="step:") → group by (type, grounding)`
- **意味着**：合同在"什么算证据"和"弱证据怎么合"这两处是空白
- **该修哪儿**：evidence_rules + corroboration —— **审计直接指出了该补哪两节**

**③ 一个推断出来的 yes，和病理确诊的 yes 长得一模一样**

- **现象**：输出只有 `has_lung_cancer: yes`
- **怎么查**：终局节点的字段清单
- **意味着**：下游拿到这个 yes 的人**无法知道它建立在三条弱证据上**
- **该修哪儿**：合同的输出词表加 `basis`（P6 的 DOCUMENTED / INFERRED）

**④ 跨 run 分歧：同样的信息，不同的判断**

- **现象**：简单版跑三次，`corroborate` 点两次判 `reinforces` 一次判 `not_actually_about_the_same`，
  而三次的 `used` 完全相同
- **怎么查**：`precipitate --type corroborate` → 输入重叠 1.0 → **JUDGEMENT 分歧**
- **意味着**：这不是检索问题，是**合同没有规定**——两次都拿到了同样的材料
- **该修哪儿**：Decision Rule（这正是 §4.1① 的独立证据）

### 4.2 复杂版查出的

**⑤ 凭搜索片段做的裁决**

- **现象**：`corroborate` 的 `used` 里 `note:Onc-Med-2023-06-15` 的
  `depth = seen_in_results`——它从没被 `read` 过
- **怎么查**：`_resolve_used` 的 depth 字段，服务端记的事实
- **意味着**：**一条支撑最终答案的证据，agent 只看过搜索片段**
- **该修哪儿**：卡片（"corroborate 之前每条都要读全文"）；如果反复出现，考虑闸门

**⑥ 读了不记**

- **现象**：`context.n_reads = 6`，但 `record_finding` 只有 3 条，`neither` 一条都没有
- **怎么查**：快照计数 vs finding 计数
- **意味着**：读了 6 份、判断蒸发了 3 份。**coverage 的真实分母缺了一半**
- **该修哪儿**：前言（"即使判为无关也要记"）；这个数是 §8 的四个健康指标之一

**⑦ 独立性检查做了，但只做了一处**

- **现象**：`when_it_happened` 只报了一次（07-19 那份），04-20 与 06-15 之间没查
- **怎么查**：`corroborate` 大点下挂的小点里，`when_it_happened` 的条数 vs `used` 的条数
- **意味着**：合同要求独立性检查，agent 做了但没做全——**可能仍把同一句数了两遍**
- **该修哪儿**：这是能力缺口（规则清楚、执行不稳）→ 先例库 + 抽取，不是改合同

### 4.3 两版对比查出的

**⑧ 合同吃掉了多少 agent 的自作主张 —— 零标注的成熟度曲线**

| | 简单版 | 复杂版 |
|---|---|---|
| `own_knowledge` 占比 | 6/10 | 1/12 |
| `contract` 占比 | 0/10 | 8/12 |
| 独立性检查 | 0 次 | 1 次 |
| 输出可区分推断与确诊 | 否 | 是 |

**同一个答案 `yes`，两条完全不同的路。** 这张表不需要任何金标准就能算出来——
它就是 D2「B−C 分歧收敛」的可操作形式：**合同每吸收一条 agent 的自作主张，
`own_knowledge` 掉一格。掉到零，agent 在这个任务上就可以被编译掉了。**

**⑨ 一个反直觉但重要的观察**

简单版和复杂版**答案相同**。如果只看准确率，会得出"复杂合同没有价值"的结论。

但审计看到的是：简单版的 yes 建立在**一条模型自己发明、没人审过、下次可能不一样的规则**上。
**准确率相同，可问责性差一个数量级。** 这正是这套仪器存在的理由——
`RELATED_WORK` 里那句"准确率可以在 C0 遍地时碰巧很高"。

---

## 5 · 这个故事证明了什么、没证明什么

**证明了（在设计层面）：**

- `corroborate` 这个类型是必需的——它承载了"弱证据能不能合并"，而十族里没有它
- `grounding` 能把"合同的洞"定位到**具体是哪一节**（evidence_rules 和 corroboration）
- 审计的对象是**依据**不是答案：两版答案相同，问题只在依据里
- `record_finding` 的 `neither` 计数是 coverage 的真实分母（§4.2⑥ 这条现在完全测不到）

**没证明（只有真跑能定）：**

- 模型会不会**如实**标 `own_knowledge`。如果它倾向于把自己的判断说成 `contract`，
  那就靠 §4.4 的规则 id 核验去抓——但**核验只能抓到虚构的 id，抓不到"引用了真实规则但
  实际没按它判"**。这是这套设计的已知盲区。
- 简单版会不会真的产生跨 run 分歧（④）。也可能模型每次都用同一条自创规则——
  **那就是"规范不清晰 + 执行一致"的危险角落**，分歧检测器抓不到，只有 `own_knowledge`
  的集中度能抓到。这个故事里 ④ 和 ① 是互补的两把钳子。
- 复杂版的 `corroboration.1` 写得对不对。这个故事只证明**有地方写**，不证明**写得对**——
  那要靠金标准和裁决。
