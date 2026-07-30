# 从 AgentLoop 到 ACR：医疗 Chart Review Agent 的运行、评估与持续改进方法论

> 状态：方法论基线  
> 文档日期：2026-07-29  
> 适用项目：`agentic-chart-review`  
> 资料范围：AgentLoop 产品能力与用户指南全覆盖；纯阿里云接入步骤、OpenAPI
> operation、RAM 字段和计费细节只索引，不作为 ACR 设计要求。

实现层的稳定 contracts、Module/Skill/Capability/Stage 边界、独立 Audit 和兼容迁移
见 [`ACR_MODULE_ARCHITECTURE_V2.md`](ACR_MODULE_ARCHITECTURE_V2.md)。

## 0. 结论先行

ACR 不应该把所有“检查 agent 的东西”都叫 eval。一个完整系统至少包含以下七层：

| 层 | 何时运行 | 核心问题 | 能否影响当前答案 |
|---|---|---|---|
| Task contract | 运行前冻结 | 任务在问什么，什么证据和答案才成立 | 是，定义答案边界 |
| In-request controls | 当前请求内 | 这个答案是否被允许提交 | 是，可拒绝、abstain 或转人工 |
| Runtime policies | 当前请求内 | agent 怎样搜索、扩展和分配算力 | 是，但它们是待验证策略，不是天然正确 |
| Observability | 当前请求中记录 | 实际发生了什么 | 否，只记录事实 |
| Audit | 运行中预检或运行后关联 | 是否发生安全、隐私、越权事件 | 预检可阻断；事后事件不改写既有答案 |
| Evaluation & attribution | trace 完成后 | 结果和过程好不好，为什么失败 | 否，只评分、归因和路由 |
| Experiment & optimization | 开发/发布前 | 某个改变是否真的提升质量 | 只决定下一版本是否发布 |

最重要的区分是：

1. **Requirement 不是 evaluator。** 患者隔离、证据准入、字段格式和负向答案的
   proof obligation 是系统执行合同。
2. **Policy 不是 requirement。** 搜哪些词、读多少文档、是否多 agent、是否“看全”
   都是有成本的策略，必须通过实验获得继续存在的资格。
3. **Evaluator type 不是 evaluator category。** `CODE / LLM / AGENT / HUMAN`
   描述“用什么执行评价”；安全、正确性、证据、过程、效率和归因描述“评价什么”。
4. **Gate pass 不是 answer correct。** Gate 只能证明系统定义的义务被执行，不能证明
   搜索词、证据宇宙和临床规则本身正确。
5. **Eval signal 不直接修改 agent。** 它先形成有证据的 failure event 和 repair
   obligation，再修改一个有 owner 的资产，并经过 paired validation。

---

## 1. AgentLoop 到底是什么

AgentLoop 的产品主线不是单独的 Agent-as-a-Judge，而是：

```text
Observability / Trace
        ↓
Audit + Evaluation
        ↓
Dataset / Annotation / Bad Cases
        ↓
Experiment
        ↓
Prompt / Skill / Context assets
        ↓
Production traces
        ↺
```

官方将完整 trajectory 作为基础数据单元，同一份轨迹同时服务于观测、审计、评估、
实验、数据加工和经验挖掘。官方 QuickStart 的实操顺序是创建 AgentSpace、接入
观测、构建 evaluator、创建 evaluation task、导入和标注 dataset、管理 Prompt/Skill、
使用 Memory，最后形成数据飞轮。

对 ACR 有用的是这种**闭环分工**，而不是阿里云控制台、SLS 或 eBPF 的具体实现。

### 1.1 能力地图与 ACR 取舍

| AgentLoop 能力 | 官方作用 | ACR 决定 |
|---|---|---|
| AgentSpace | 团队/业务域的资源、权限和数据边界 | 改造采用：本地 workspace + task bundle + local artifact root |
| Observability | Agent/Tool/Model/Retriever/Memory 的 trace、指标和拓扑 | 直接采用思想：统一 trace ontology |
| Audit | 原始事实 → finding → incident → investigation | 直接采用两级模型；患者边界额外前置为 runtime control |
| Evaluation | Evaluator + Evaluation Task + Analysis | 直接采用三段式结构 |
| Agent-as-a-Judge | evaluator 自带 Prompt、Skill 和工具，读取 trajectory | 改造采用：同患者、只读、能力 broker、不能确认临床 truth |
| Experiment | 对模型、Prompt、Agent、工具配置做重复、可比测试 | 直接采用；增加 chart-observable gold 和 paired patient comparison |
| Dataset | 承载 trace、标注、gold、bad cases、experiment cases | 改造采用：患者数据只在 repo 外本地引用，不复制入共享库 |
| Pipeline | 过滤、去重、采样、聚类、AI 加工、写入数据集 | 直接采用思想：本地 append-only event pipeline |
| Annotation | 人工模板化标注 | 改造采用：registrar/clinician/engineer 的角色化 adjudication |
| Prompt/Skill assets | 版本、diff、label、灰度和运行时加载 | 直接采用；扩展到 spec、tool、runtime policy 和 evaluator |
| Experience Library | 从成功/失败轨迹提炼可复用的“做事方法” | 后续采用：只允许无 PHI、经验证的方法资产 |
| Memory | 保存用户或环境的长期上下文 | 默认不用于患者 chart；禁止跨患者记忆 |
| OpenAPI/RAM/Billing | 云产品集成、权限和成本 | 只索引，不接入 Alibaba Cloud |

### 1.2 AgentLoop 中几个容易混淆的概念

#### Trace 与 Trajectory

- Trace 是一次请求的调用链事实，包含模型、工具、检索、输入输出、耗时和 Token。
- Trajectory 是适合后续评估和经验挖掘的完整 Agent 执行表示。
- ACR 不需要保存不可验证的长篇 CoT；需要保存结构化事件、模型/工具 I/O、evidence
  ledger、plan revision、gate decision、termination 和 hash lineage。

#### Audit 与 Evaluation

- AgentLoop Audit 先提供中性的 Audit Facts：Session、Token、Tool 和行为事实；Risk
  Audit 才把安全或边界信号从低保真 `Finding` 关联为高保真 `Incident`。
- ACR v1 采用 application-level Risk Audit，并把完整 Audit Facts 平台、eBPF 和实体
  调查列为 out of scope。Audit 仍然是独立治理平面，不是 CODE evaluator 的别名。
- Evaluation 面向质量：正确性、证据、任务完成、工具选择、执行效率等。
- “PHI 出现在外部 provider 请求中”是 incident；“答案引用错病理报告”是质量错误。
  二者可能发生在同一 trajectory，但不能用同一结论类型表示。

#### Evaluation 的三个组件

1. **Evaluator**：一个版本化评价标准。
2. **Evaluation Task**：绑定数据源、采样、变量映射、evaluator 和运行策略。
3. **Analysis**：聚合趋势、筛选 bad case、下钻单个 trajectory。

Evaluator 不能同时兼任任务编排、budget、selector 和 certification。ACR v2 使用
`ModuleAsset` 定义 evaluator，`PipelineProfile` 定义条件和依赖，`EvaluationTask`
绑定数据与权限，`CertificationSuite` 独立保存认证标准。旧 `EvaluatorSpec` 仅作为
兼容 adapter。

#### Experience 与 Memory

- Experience 回答“这种任务通常怎样做更好”，例如先看 definitive resection，再处理
  biopsy/resection precedence。
- Memory 回答“这个用户或环境过去是什么情况”。
- Patient chart 不是 Memory。ACR 不得把患者内容写成跨 run、跨患者可召回记忆。
- ACR Experience 只能保存通用方法、适用条件、来源 trajectory hashes、验证指标和
  版本，不能保存患者答案、原文或未经验证的“经验”。

#### AgentLoop 的“online”不等于同一请求内

- “持续/在线 evaluation”指新 trace 产生后自动触发评价，通常是 nearline post-run。
- Experiment 文档中的“online experiment”指由 AgentLoop 服务端调用目标端点；
  “offline”指从用户本地发起。它不是 runtime gate 的同义词。
- ACR 统一使用 `IN_REQUEST`、`POST_RUN_CONTINUOUS` 和
  `OFFLINE_DEVELOPMENT`，避免只写 real-time/online。

---

## 2. ACR 的标准分层

### 2.1 Task contract：任务合同

Task contract 是领域语义，不是 agent 的导航说明，也不是 evaluator。

每个任务包应冻结：

- 字段、value domain、状态和输出 schema；
- patient/entity/time scope；
- evidence eligibility 和 source precedence；
- 正向 witness 要求；
- 负向/absence 主张的可观察范围与 proof obligation；
- abstention 边界；
- conflict rule 和必须由人工决定的边界；
- 每个语义元素的来源、版本、owner 和 sign-off。

跨领域复用的不是一份巨型 spec，而是相同的合同 schema。肿瘤 registry、药物使用、
复发、心血管 phenotype 可以有不同 spec，但共享同一执行和评估框架。

### 2.2 In-request controls：当前请求内的硬控制

Runtime control 必须满足至少一个条件：

- 这是不可协商的安全/权限边界；
- 输出不满足它就逻辑上无效；
- 它是便宜、确定性、可重放的检查；
- 失败后有明确的 reject、abstain 或 human-review 行为。

ACR 应保留的 control：

| Control | 为什么属于 runtime | 失败行为 |
|---|---|---|
| 当前患者范围 | 跨患者证据使答案无效 | 立即拒绝工具调用 |
| Provider/PHI 边界 | 数据发送前才能有效防护 | preflight 阻断 |
| Tool allowlist/read-only | 未记录读取会破坏 evidence ledger | 拒绝调用 |
| Spec、policy 和工具版本冻结 | 无法重放的答案不可比较 | 拒绝启动或标为 unvalidated |
| Evidence admissibility | FOUND 必须有合格 witness | 拒绝提交 |
| Field format/value checks | 结构上无效的答案不能流出 | 拒绝提交 |
| 负向 proof obligation | “没有/证据不足”是全局性主张 | 未闭合则 abstain/review |
| 已发现 conflict/open thread | 不能忽略 chart 自己暴露的矛盾 | 继续调查或人工复核 |
| Hard budget | 防止无限循环和无界花费 | `BUDGET_EXHAUSTED`/review |

Runtime control 不负责判断整个临床答案正确，也不应包含未经实验验证的“最佳搜索
路径”。

### 2.3 Runtime policies：运行时性能策略

Policy 决定 agent 怎样完成合同：

- 初始检索词和 document-type routing；
- 文档阅读顺序；
- sampling frame 和 sample size；
- plan expansion；
- conflict refinement；
- 多 trajectory 的触发条件；
- model、temperature、context management；
- experience retrieval；
- 成本在搜索、阅读和确认之间的分配。

Policy 可以影响当前 run，但不能因为“看起来完备”就被提升为 requirement。每个 policy
必须：

1. 有稳定 `policy_id/version/hash`；
2. 记录预期改善的 failure mode；
3. 能被 feature flag 或 profile 替换；
4. 在同一 task contract 下进行 paired experiment；
5. 失败时可回滚，而不修改 task semantics。

### 2.4 Observability：中性事实层

Observability 只回答“发生了什么”：

- run/session/trace/span IDs；
- model/tool/retrieval/gate/output event；
- 输入输出 hash、版本、耗时、tokens、cost；
- documents listed/searched/read；
- evidence 和 coverage ledger；
- plan revisions、rejections、retries、termination；
- provider boundary 和 patient scope。

Trace 不能因为某个 evaluator 暂时不需要某字段就删掉；同一 trace 将来还要服务 audit、
evaluation、experiment 和 attribution。

### 2.5 Audit：安全证据链

ACR 采用 AgentLoop 的两级模型：

```text
Raw facts
  → Finding（高召回、单点信号）
  → correlation
  → Incident（证据链闭合、可行动）
  → investigation / disposition
```

示例：

- 日志中检测到像姓名的文本：Finding；
- 姓名随 chart 内容进入 external provider 请求：Incident；
- 工具参数出现另一个 patient ID：Incident；
- repo 外本地路径包含患者内容但未越过声明边界：需要调查，不自动等于泄露。

两种防线应并存：

- `IN_REQUEST`：patient scope、provider trust boundary、tool allowlist 等 preflight；
- `POST_RUN_CONTINUOUS`：从完整 trace 关联遗漏的 PHI/data-flow incident。

当前实现只使用 application events；它不宣称能检测未被应用上报的 process/file/
network side effect。`RuntimeEvidenceRef` 是未来 adapter 边界，不是已经实现的 eBPF
能力。

### 2.6 Evaluation：独立质量评价

#### 两条正交分类轴

“如何评”和“评什么”必须分开。

**执行方式：**

| 类型 | 能力 | 权限 |
|---|---|---|
| CODE | 确定性、可重放、便宜 | 可创建 incident 或阻断版本发布 |
| LLM | 单轮、无工具、明确 rubric | 只筛查和排序 |
| AGENT | 读 trajectory、调用声明工具、最小补读 | 只调查、归因和路由 |
| HUMAN | 临床/registry/spec semantic 裁决 | 确认 gold、批准 semantic change |

AgentLoop API 原生类型是 `CODE / LLM / AGENT`；`HUMAN` 是 ACR 为医疗责任和
adjudication 增加的 authority plane。

**评价领域：**

| 大类 | 评价内容 | 典型实现 |
|---|---|---|
| Safety & boundary | PHI、跨患者、外发、权限 | CODE audit + human investigation |
| Outcome & abstention | value/status 是否正确，是否过度断言 | CODE 对 gold；HUMAN 裁决 |
| Evidence & grounding | quote、document standing、source precedence、未读矛盾 | CODE + AGENT |
| Process & control integrity | gate/ledger/termination 是否一致，tool/plan 是否异常 | CODE |
| Reliability & efficiency | provider failure、retry、cost、latency、documents read | CODE |
| Causal attribution | 哪个 defect 解释目标错误，应该修改哪类资产 | AGENT + HUMAN |

`Agent-as-a-Judge` 是执行方式，不是第七个评价领域。

#### CODE 优先原则

同一子问题存在确定性答案时，LLM/AGENT 不得重判：

- quote offset 能否复现：CODE；
- field exact match：CODE；
- 是否跨患者读取：CODE；
- 某段临床文本是否真正支持某一编码：可能需要 AGENT/HUMAN；
- registry 值是否是当前 chart 的 truth：HUMAN。

#### Eval 不应影响已完成答案

Post-run evaluator 可以：

- 标记 run；
- 创建 incident；
- 排入人工复核；
- 阻断 candidate bundle 发布；
- 创建 bad-case/repair obligation。

它不能悄悄改写原 run 的 answer 或把自己的推测写成 confirmed truth。

### 2.7 Causal attribution：从发现缺陷到解释目标错误

发现一个真实 defect，不等于它解释了目标错误。归因必须绑定显式 target event：

```text
Target event
  → reconstruct retrieval → evidence → interpretation → coding → gate → output
  → enumerate rival causes
  → smallest discriminating probe
  → safe counterfactual replay
  → skeptic review
  → attribution gate
```

每个 cause 应标记：

- `relation_to_target = EXPLAINS | CONTRIBUTES | UNRELATED_DEFECT | UNKNOWN`
- `causal_strength = OBSERVED | PLAUSIBLE | COUNTERFACTUAL_SUPPORTED | HUMAN_CONFIRMED`

只有 `EXPLAINS + COUNTERFACTUAL_SUPPORTED` 才能成为强模型归因；临床内容仍需 human
confirmation。

### 2.8 Experiment 与发布

Evaluation 描述当前表现；Experiment 回答“某个改变是否造成改善”。

一个有效 experiment 必须固定：

- patient set 和 chart snapshot；
- task contract/spec hash；
- model/provider 和 seed set；
- budget；
- evaluator bundle；
- truth/adjudication version。

并只改变预先声明的资产：

- spec；
- retrieval/runtime policy；
- Prompt/Skill；
- Tool/check；
- model；
- experience bundle。

结果必须逐病例比较，同时报告 accuracy、abstention、overclaim、evidence validity、
subgroup、cost、latency、calls 和 documents read。平均分提升不能掩盖 critical
regression。

---

## 3. Coverage：必须拆成四件事

“要求 agent 看全”混合了四种不同概念。

### 3.1 Claim obligation：答案欠什么证明

这是 task contract。

- 对正向 `FOUND`：通常一个合格 witness 足够。
- 对负向、不存在或 `EVIDENCE_INSUFFICIENT`：必须声明观察范围，并证明在这个范围内
  没有合格 witness，或者明确哪些 evidence gap 阻止结论。
- 对 source precedence、多个 tumor/entity 或多个时间点：即使是正向答案，也可能要
  证明更高优先级来源不存在或冲突已关闭。

### 3.2 Acquisition policy：怎样获得证明

这是可替换 runtime policy：

- exhaustive read；
- keyword search；
- document-type routing；
- sample misses；
- time-window coverage；
- conflict-triggered expansion。

它们可能提高召回，也可能增加成本、制造 rejection loop 或导致错误 abstention。

### 3.3 Enforcement：何时拒绝答案

这是 runtime control：

- positive witness 不合格：拒绝 `FOUND`；
- negative proof 未闭合：不得输出强负向结论；
- 已发现 conflict/open thread 未关闭：继续调查或转人工；
- 搜索策略已无可行扩展：返回明确 gap，而不是伪造 consensus。

### 3.4 Coverage evaluation：这套做法是否有效

这是 post-run evaluation/experiment：

- gate 是否与 ledger 一致；
- 是否漏掉后来发现的 witness；
- 是否降低 critical overclaim；
- 是否提高 chart-observable exact match；
- 是否增加 abstention、loop、成本和 subgroup regression。

### 3.5 ACR 的默认 coverage 立场

1. 不把“读完所有文档”作为跨任务通用要求。
2. 只对负向/absence 主张和确需全局 precedence 的任务保留 proof obligation。
3. 正向答案在 witness 合格、已发现冲突关闭后允许停止。
4. agent 可以自行决定导航路径；runtime 可以独立抽样 misses，防止 agent 用自己选择的
   文档证明自己搜索充分。
5. 当前 stratified coverage、关键词、sample size 和阈值都是工程假设；在真实
   chart-observable gold 上验证前不能宣称它们提高性能。

### 3.6 Coverage ablation

建议四臂：

| Arm | 行为 |
|---|---|
| A. Witness-only | 正向 witness 后停止；负向直接明确不足，不做广泛 closure |
| B. Negative-proof | 只有即将提交负向结论时才启动 closure |
| C. Current stratified | 当前 stratified search + forced sampling |
| D. Adaptive | 仅在 conflict、entity/time ambiguity、gate risk 时扩大搜索 |

第一轮 10 个真实病例在完成 chart-observable adjudication 前，只能比较：

- calls、cost、documents read、latency；
- rejection loop、provider/runtime degradation；
- evidence admissibility、gate consistency；
- registry disagreement 的变化。

只有被人工确认可从当前 chart 推出的字段才能进入 accuracy、overclaim 和 causal
improvement 结论。10 例仍是探索性 pilot，不能单独批准全局 policy。

---

## 4. Eval 信号怎样改善系统

Eval 输出必须路由到明确资产 owner，而不是统一归因成“模型不够好”。

| 观察到的信号 | 首要区分 | 允许的 repair target |
|---|---|---|
| Gold witness 从未 surfaced | 搜索词错误还是 evidence universe 漏类 | retrieval asset/runtime policy |
| Witness 已读但没采用 | source standing、语义理解还是 entity/time | Skill/Prompt；必要时 spec form |
| 正确答案被 gate 拒绝 | task obligation 错还是 control 实现错 | proof contract 或 deterministic control |
| 错误答案通过 gate | gate 漏检还是 task semantic 错 | answer check/control 或 spec content |
| Registry disagreement | registry 是否 chart-derivable | human adjudication，不能自动 patch |
| 多 runs 分歧 | value、evidence、entity、time 还是 provider | targeted attribution/experiment |
| `SPEC_INSUFFICIENT` loop | 真 spec gap 还是 coverage dead-end 被误路由 | gate/termination routing |
| PHI/cross-patient incident | preflight 是否缺失 | runtime safety control |
| 重试、超时、成本尖峰 | provider、tool、policy 还是 prompt | runtime/tool/model/policy |
| 某成功路径重复有效 | 是否跨病例成立、是否泄漏 PHI | certified experience asset |

### 4.1 Truth mode 决定结论上限

| 模式 | 可用信号 | 允许结果 |
|---|---|---|
| BLIND | trace、spec、detector、多次 run、chart probe | anomaly、hypothesis、test obligation |
| REGISTRY_REFERENCE | 未裁决 registry 值 | disagreement、adjudication obligation |
| GOLD | chart-observable gold 和 witness | confirmed mismatch、contrastive failure packet |
| HUMAN | 角色化签名裁决 | confirmed gold、semantic approval、disposition |

没有 gold 时仍然可以改进 safety、runtime reliability、gate consistency、evidence
admissibility 和成本；不能把 clinical correctness hypothesis 当成 truth。

### 4.2 一次修复的闭环

```text
Trace
  → deterministic screening
  → selected LLM/AGENT evaluation
  → human adjudication when required
  → deterministic bad-case cluster
  → one target event + one repair obligation
  → change one versioned asset
  → paired validation
  → sealed certification
  → canary/shadow
  → production monitoring
```

禁止：

- evaluator 直接修改 production spec；
- 用 majority vote 代替 evidence gate；
- 用模型 confidence 代替 human confirmation；
- 为匹配 registry 教 agent 猜 outside-chart 信息；
- 用同一批 diagnosis cases 无限调参再称为 sealed test。

---

## 5. Framework 开发阶段与 Task 使用阶段

### 5.1 Framework 开发阶段应该完成

这些能力跨所有 chart-review task 共享：

- 统一 trace ontology 和 immutable run manifest；
- patient/provider/tool capability broker；
- task contract、runtime control、runtime policy 的独立 hash；
- `CODE / LLM / AGENT / HUMAN` runner 和权限边界；
- audit `Finding → Incident` correlation；
- evaluator registry、task runner、analysis/event store；
- truth-mode isolation；
- attribution target/counterfactual/skeptic gate；
- experiment、paired comparison、release gate；
- asset lineage：spec、Prompt、Skill、Tool、model、policy、evaluator、experience；
- local-only PHI storage 与 append-only bad-case library；
- evaluator 自身 certification 和 drift monitoring。

Framework 不能硬编码肺癌、STORE 或某一科室的临床知识。

### 5.2 每个新 Task onboarding 时完成

Task package 至少包含：

- extraction spec / field contract；
- evidence source policy 和 precedence；
- entity/time model；
- positive/negative proof obligation；
- retrieval assets；
- task-specific skills；
- task-specific deterministic checks；
- evaluator profile；
- gold/registry/blind 数据策略；
- subgroup 和 critical-error 定义；
- runtime policy experiment；
- human owner 和 sign-off。

### 5.3 Task 生产运行时使用

每例默认只运行：

- frozen task contract；
- cheapest safe runtime policy；
- in-request controls；
- complete trace；
- cheap CODE audit。

只有异常病例才触发：

- conflict refinement；
- unread-evidence evaluator；
- causal attribution；
- human review。

生产 RUN plane 不携带 gold 或 registry reference。

### 5.4 Task 持续改进时使用

- Continuous post-run CODE evaluation：全量或高比例。
- LLM/AGENT evaluation：异常选择和成本受控采样。
- Human adjudication：registry disagreement、临床语义和 semantic patch。
- Periodic experiment：新 model/spec/policy/skill/tool/evaluator/experience 版本。
- Bad-case cluster：按 target/cause/parameter 聚类，而不是按自由文本摘要聚类。

---

## 6. ACR 对 AgentLoop 的采用边界

### 6.1 直接采用

- trajectory-first observability；
- Evaluator / Evaluation Task / Analysis 分工；
- CODE/LLM/AGENT evaluator；
- audit finding/incident 两级模型；
- trace → dataset → annotation → experiment → asset 的数据飞轮；
- Prompt/Skill 的版本、diff、label 和灰度；
- experience 是方法而不是答案。

### 6.2 改造采用

- `HUMAN` 作为独立 authority plane；
- patient-scoped chart tools 和 PHI-local artifact boundary；
- chart-observable gold，不盲从 registry；
- deterministic check 对同一子问题优先；
- agent evaluator 只能 screen/route；
- semantic patch 必须 human sign-off；
- dataset 只保存本地引用和 hash，不复制真实 chart；
- Memory 默认禁用；Experience 必须无 PHI、经验证、可撤销。

### 6.3 不采用

- 不把患者数据发送到 Alibaba Cloud；
- 不依赖 SLS、ARMS、MSE 或 eBPF 才能运行；
- 不保存大量自由文本 CoT；
- 不允许 evaluator 自主上线修改；
- 不允许跨患者 memory；
- 不把在线评分当作当前请求的 clinical gate；
- 不因流程“看起来完整”就强制读全 chart。

---

## 7. 对当前仓库的具体解释

当前 README 中的 “audit layer — this is the product” 实际混合了：

- in-request answer controls；
- coverage acquisition policy；
- trace/manifest observability。

更准确的命名应是：

- `agent.py`：runtime orchestration；
- `answer_gate.py` / `answer_checks.py` / `answer_contract.py`：in-request controls；
- `coverage_planner.py` / `coverage.py`：claim proof 支撑 + runtime coverage policy；
- `trace.py` / `run_manifest.py` / `usage_telemetry.py`：observability；
- `evaluation_modules.py`：post-run deterministic quality evaluators；
- `evaluation_pipeline.py`：typed evaluation control plane；
- `attribution.py`：post-run causal attribution；
- `spec_repair.py`：DEVELOP optimization；
- `conflict_refinement.py`：optional RUN policy。

同一事实可以在两处出现，但作用不同。例如：

- runtime gate 拒绝一个未闭合的负向答案；
- post-run `gate-consistency` evaluator 检查 gate 是否按 ledger 正确执行。

后者是在检查 control 的实现，不是在重新回答患者问题。

---

## 8. 官方文档 coverage matrix

状态说明：

- **深读**：已用于本文方法论。
- **索引**：确认能力边界，但不展开云平台操作细节。
- **采用**：ACR 直接采用核心思想。
- **改造**：采用思想但增加医疗、PHI 或本地边界。
- **不接入**：记录存在，但 ACR 不连接该云服务。

### 8.1 产品、概念与快速入门

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [What is AgentLoop](https://help.aliyun.com/en/document_detail/3033860.html) | 深读/改造 | 总体闭环与 trajectory-first |
| [核心概念](https://help.aliyun.com/zh/document_detail/3042001.html) | 深读/采用 | AgentSpace、Trajectory、Dataset、Pipeline、Evaluation、Experience、Memory |
| [QuickStart 全流程](https://help.aliyun.com/en/document_detail/3033823.html) | 深读/改造 | 闭环顺序和资产关系 |
| [计费说明](https://help.aliyun.com/zh/document_detail/3044490.html) | 索引/不接入 | 证明 evaluation/experiment/data 都有独立成本 |
| [RAM 权限参考](https://help.aliyun.com/en/document_detail/3033852.html) | 索引/改造 | least privilege 思想 |
| [OpenAPI operations](https://help.aliyun.com/en/document_detail/3041792.html) | 索引/不接入 | API 能力边界 |

### 8.2 Observability

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [AI agent observability](https://help.aliyun.com/en/document_detail/3042586.html) | 深读/采用 | capability map |
| [Trace](https://help.aliyun.com/en/document_detail/3042591.html) | 深读/采用 | request/span 结构、回放 |
| [Session analysis](https://help.aliyun.com/en/cms/cloudmonitor-2-0/conversational-analysis-of-ai-agent) | 深读/改造 | run/session 聚合、成本和错误下钻 |
| [Scenario-based analysis](https://help.aliyun.com/zh/document_detail/3042597.html) | 索引/改造 | 多维趋势分析 |
| [AI Agent 接入应用监控](https://help.aliyun.com/zh/document_detail/3046111.html) | 索引/不接入 | 探针与 OpenTelemetry 接入思想 |

具体框架/产品接入页（LangChain、Dify、OpenAI、AgentScope、Hermes、Coding Agents 等）
全部归入 Access Center 索引；它们改变采集方式，不改变 ACR 方法论。

### 8.3 Audit

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [AI Agent 审计](https://help.aliyun.com/zh/document_detail/3045691.html) | 深读/采用 | audit 总体分层 |
| [审计接入指南](https://help.aliyun.com/zh/document_detail/3045692.html) | 深读/改造 | application facts + runtime facts |
| [审计管理](https://help.aliyun.com/zh/document_detail/3045693.html) | 深读/改造 | 独立开关、持续任务 |
| [审计规则说明](https://help.aliyun.com/zh/document_detail/3045694.html) | 深读/采用 | Finding → Incident |
| [风险事件字段说明](https://help.aliyun.com/zh/document_detail/3045695.html) | 索引/改造 | incident schema |
| [风险审计](https://help.aliyun.com/zh/document_detail/3045696.html) | 深读/改造 | 风险排序和调查入口 |
| [实体调查](https://help.aliyun.com/zh/document_detail/3045697.html) | 深读/改造 | patient/provider/tool/entity 关联 |
| [审计事实](https://help.aliyun.com/zh/document_detail/3045698.html) | 深读/采用 | 中性事实不等于风险 |

### 8.4 Evaluation

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [评估概述](https://help.aliyun.com/zh/document_detail/3042179.html) | 深读/采用 | Evaluator/Task/Analysis |
| [评估器](https://help.aliyun.com/zh/document_detail/3042180.html) | 深读/改造 | 预置维度、Prompt、Skill、MCP |
| [评估任务](https://help.aliyun.com/zh/document_detail/3042181.html) | 深读/采用 | trace/log/dataset、采样、持续/历史 |
| [Evaluator API schema](https://help.aliyun.com/zh/document_detail/3045378.html) | 深读/改造 | CODE/LLM/AGENT、变量映射、版本 |

### 8.5 Experiment

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [实验操作指南](https://help.aliyun.com/zh/document_detail/3046601.html) | 深读/改造 | LLM/Agent 实验、在线/离线定义、trajectory 变量 |

离线结果上报、服务注册和控制台操作页归入索引；ACR 采用本地 paired experiment，
不向 AgentLoop 上报患者实验结果。

### 8.6 Data Center、Pipeline 与 Annotation

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [数据集概述](https://help.aliyun.com/zh/document_detail/3042278.html) | 深读/改造 | 结构化数据资产 |
| [数据集控制台接入](https://help.aliyun.com/zh/document_detail/3042280.html) | 深读/改造 | trace/CSV/manual 来源 |
| [数据标注](https://help.aliyun.com/zh/document_detail/3041820.html) | 深读/改造 | human adjudication UI/schema 思想 |
| [Pipeline 用户指南](https://help.aliyun.com/zh/cms/cloudmonitor-2-0/user-guide-for-agentloop-pipeline) | 深读/采用 | 清洗、三级去重、聚类采样、AI 加工 |
| [Pipeline 参考](https://help.aliyun.com/zh/cms/cloudmonitor-2-0/product-feature-documentation) | 索引/不接入 | operators/API/limits |

### 8.7 Agent Assets

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [Agent 资产概述](https://help.aliyun.com/zh/document_detail/3041729.html) | 深读/采用 | 独立版本、status、labels |
| [Prompts 管理](https://help.aliyun.com/zh/document_detail/3041730.html) | 深读/改造 | draft/publish/diff/label/动态加载 |
| [Skills 管理](https://help.aliyun.com/zh/document_detail/3041731.html) | 深读/采用 | SKILL.md、import、version、labels |

### 8.8 Experience 与 Memory

| 文档 | 状态 | ACR 用途 |
|---|---|---|
| [经验库产品介绍](https://help.aliyun.com/zh/document_detail/3047255.html) | 深读/改造 | 方法资产，不是答案 |
| [经验库使用指南](https://help.aliyun.com/zh/document_detail/3047254.html) | 深读/改造 | trajectory mining、持续挖掘、runtime Skill recall |
| [记忆模块控制台接入](https://help.aliyun.com/zh/cms/cloudmonitor-2-0/memory-module-console-operating-guidelines) | 深读/默认不采用 | Facts/Episodic/Summary、检索和生命周期 |
| [QuickStart 的 Memory 集成](https://help.aliyun.com/en/document_detail/3033823.html) | 深读/默认不采用 | Mem0 接口与长期/短期记忆 |

---

## 9. 方法论验收问题

今后新增任何 ACR 机制，都先回答：

1. 它属于 task contract、control、policy、observability、audit、evaluation、experiment
   还是 experience？
2. 它在 `IN_REQUEST`、`POST_RUN_CONTINUOUS` 还是 `OFFLINE_DEVELOPMENT` 运行？
3. 它凭什么有权阻断当前答案？
4. 它评价的维度是什么，执行方式是什么？
5. 它是否依赖 gold；没有 gold 时结论上限是什么？
6. 它发现信号后修改哪个资产，谁是 owner？
7. 它怎样通过 paired experiment 证明收益？
8. 它是否会复制 PHI、跨患者、泄漏 answer key 或积累未经验证的“经验”？
9. 停止条件是什么？
10. 如果移除它，哪个已证明的指标会退化？

如果第 10 个问题没有证据，机制应标为 `EXPERIMENTAL_POLICY`，不能因为“看起来考虑
完备”就成为永久硬要求。
