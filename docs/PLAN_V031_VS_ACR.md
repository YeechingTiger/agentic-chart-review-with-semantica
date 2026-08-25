# v0.3.1 计划 vs acr 实现：逐层对照

2026-08-25。对照对象：四份计划文档（主方法论《agentic_chart_review_complete》→ 决策沉淀
v0.2.4 → 工程计划 v0.3.1 → build brief）与本仓库 `src/acr` 的现状。

仓库侧事实由 8 路并行核实产出（249 次工具调用，全部带 file:line），不靠记忆。
本文标注的行号以核实当日的工作树为准。

---

## 0 · 谱系：这不是两个项目

```
主方法论（"原本的完整 plan"）        Level 0-5 成熟度、三问阶梯、算子、
   │                                acquisition profile、开发/生产双模式
   │ 扩展 §12/§14/§16/§17/§20-23
   ▼
决策沉淀 v0.2.4                     判断账本、决策点等价类、清晰度阶梯
   │ 评审回合五处纠错                C3/C2/CB/C1/C0、三分诊、双车道
   ▼
工程计划 v0.3.1（工程权威）          确定性内核 + bounded LLM + adjudicator
   │                                + Postgres 账本 + 可选图投影
   ▼
CLAUDE_CODE_BUILD_BRIEF（执行入口）  Phase 0+1 先行，垂直切片 = 术后疼痛
```

**acr 是这个谱系早期阶段的实现**（`docs/AGENTLOOP_TO_ACR_METHODOLOGY.md` 是过渡痕迹），
随后独立演化：换了领域（癌症登记）、换了运行时（deepagents）、长出了计划文档没有的
**测量平面**（四臂、镜像对、行为熵、判定类别隔离）。对照的真正问题因此是：
**acr 已经实现了 v0.3.1 的多少、哪里走了相反的路、迁移意味着什么。**

## 1 · 一句话判决

> **两边共享同一套语义学（义务、覆盖、弃答、版本化、fail-closed 词汇），在一个架构点上
> 走了相反的路——组合住在哪：v0.3.1 让代码从原子判断组合最终标签，acr 让模型提交完整
> 答案、代码只做否决与降级。而 acr 的 gate 演化方向（强制→劝告，带测量依据）与 v0.3.1
> 的方向（一切 fail closed）相反——这不是矛盾，是研究仪器与生产系统的姿态差，但迁移时
> 每一条检查都要重新选边。**

一个本次核实才浮出的重要事实：**acr 已经有一个确定性组合器**——
`contract/concordance.py`，模型无关（import-closure 测试强制），Kleene 三值代数
（all_of/any_of/not、equals/in_set/at_least/days_between…），把抽取出的**变量**组合成
五种一致性标签（CONCORDANT / NON_CONCORDANT / NOT_APPLICABLE / NOT_ASSESSABLE /
EXCEPTION_DOCUMENTED），且 `negative_basis` 穿透组合——"证明了的缺席"与"未知"组合
出不同结果（concordance.py:28-34, 802-873）。**即 v0.3.1 的 adjudicator 思想在 acr 的
跨变量层已经存在；v0.3.1 把同一思想推进到变量内部（逐事件、逐判断点）。**

---

## 2 · 逐组件对照

图例：✅ 等价物存在且运行 · 🔶 存在但劝告态/饿着/孤儿 · ⬜ 无 · ⚡ 走了相反的路

### 2.1 合同 / Protocol

| v0.3.1 | acr 现状 | 判定 |
|---|---|---|
| ProtocolCompiler 静态校验，拒绝不完整合同 | `load_spec` 拒绝链完整：pydantic 校验 → `bind_provenance`（answer_checks 一律拒绝——`ANSWER_CHECK_KINDS` 是**空 frozenset**，answer_checks.py:90；日历种类、结局空间、逐条 provenance 诚实规则、未标注的强制元素）→ `check_discriminating_facts`（spec.py:960-1045）。**speclint 在加载期什么都不拒绝**；`extra="allow"`（spec.py:640）使未声明的顶层键静默载入并进 hash | ✅（带两个缺口） |
| CompiledTaskPlan hash | `spec_hash` = sha256(完整 pydantic 模型的 canonical JSON) 截 16 hex；另有去 provenance 的签名 hash | ✅ |
| population / anchor / windows / boundary | `case_context.time_anchorable` 真实强制（case_requirements.py:47-54 → agent.py:1810）。**但**：STORE.1860 的 `anchor.t_ned`/window YAML 只有 specview 渲染器读；`empty_window_policy` 及 `clip_and_judge`（coverage.py:175-228）**实现完整、生产零调用**（spec.py:321 自己承认）；interior_gap gate 检查存在但劝告态 | 🔶 **孤儿代码** |
| sources：mandatory / conditional / **forbidden** | 无此词汇。最近的是 strata `establishes` + for_positive witness；**可采性逐条计算但每条记录 `enforced_by_gate: False`**（coverage.py:641-645）——观察，不过滤 | ⬜/🔶 |
| obligations：`scope_pattern` (WINDOW/EVENT_NEIGHBORHOOD) × `quantifier` (ALL/EXISTS/COUNT…) | **无逐事件量词**。现有的是 chart 级 gate 布尔：required_keywords_all_searched、exclusion_validated（零命中抽样）、Clopper-Pearson elusion 上界。concordance 的 all_of/any_of 是变量级不是事件级 | ⬜ |
| output_states 通用词表 | `result.status` + 四 kind（value/abstain_evidence/abstain_spec/failure），义务按 kind 挂在代码里；`submittable:false` 把状态移出工具 enum。**无 SOURCE_UNAVAILABLE**；无 budget_limited_unknown **状态**（等价物是 termination_reason: SPEND_LIMIT/EXPANSION_LIMIT/MODEL_CALL_LIMIT + negative_basis 路由人工）；无 human_review_required 状态（只有 route_to_human 旗标，无队列） | ✅（三个状态缺席） |
| **output-specific proof obligations**（覆盖门控不分 yes/no 极性） | 义务挂在 kind 上是同一泛化的另一种机制——但**没有任何东西门控一个全称阳性**（"所有事件均达标"式的 yes）。acr 的任务全是单值 GET/SELECT，问题没出现过 | 🔶 **概念同构，极性缺一半** |

### 2.2 内核 / 覆盖 / 停止

| v0.3.1 | acr 现状 | 判定 |
|---|---|---|
| Coverage 状态机（NOT_ENUMERATED…COMPLETE/SOURCE_UNAVAILABLE/FAILED） | `StratumResult` **更富**：N/reviewed/complete、hits_read/**hits_unread**（"自己搜到却从未打开的文档"）、misses_sampled、elusion_upper、draws_invalidated、replacement_draws_required（coverage.py:264-299）。无 SOURCE_UNAVAILABLE | ✅（各有强弱） |
| 覆盖用"文档集×窗口"定义，绝不用关键词（铁律） | **同一结论，实测到达**：required-search 拒绝已删，删除理由是词表实测 87.4% 召回/漏 31.7% 患者（answer_gate.py:106-121 + spec provenance——且该测量**只存在于 provenance 注记里**，没写进任何 run 目录） | ✅⚡ |
| coverage-gated 结论物理阻断 | ⚡ **方向相反**：`evaluate_gate(enforce=False)` 是唯一生产调用形态（answer_gate.py:105），所有层级/elusion/关键词检查全部落 advisories，"PASS with populated advisories"。生产真正拒绝的只有：未列文档就主张缺席（唯一 check_gate 拒绝）+ 提交时的未声明状态 / FOUND 无证据 / 不可能值 / **强制抽样欠账**（列出精确 note_ids，answer_gate.py:292-318）/ SPEC_INSUFFICIENT 报告形状 | ⚡ **enforce=True 代码原样保留，无生产调用者** |
| 预算耗尽 → budget_limited_unknown，不得伪装成 no | 等价且更细：ExpansionBudget 四上限，"honest STUCK/EVIDENCE_INSUFFICIENT——never a silent truncation and never a pass"；negative_basis ∈ {GATE_VALIDATED, AGENT_GAVE_UP, BUDGET_EXHAUSTED, COVERAGE_UNREACHABLE}，**非 GATE_VALIDATED 一律路由人工**（answer_contract.py:373-378） | ✅ |
| STOP_WITH_RULE 引用已满足的停止规则 ID | 无停止规则 ID。运行终止于：接受的答案 / 冻结拒绝循环（REJECTION_LOOP）/ 预算上限，用枚举 termination_reason 标注 | ⬜ |
| 强制抽样（runtime 抽、agent 不得自选） | ✅ 完整且精巧：ForcedSampler 种子化、默认每层 25、帧移动时作废重抽、"census of 4 beats a demanded 25" | ✅ acr 更强 |

### 2.3 答案路径 / Adjudicator

| v0.3.1 | acr 现状 | 判定 |
|---|---|---|
| **最终 label 由确定性 adjudicator 从原子判断组合；LLM 不写标签** | ⚡ **架构分野**：模型经 submit_answer 提交完整答案 → AuditMiddleware 换掉工具回执为 gate 判决 → 接受则原样入 manifest。终局只有**向弃答方向的改写**：NO_ANSWER 默认、outside_notes 强制 SPEC_INSUFFICIENT、`downgrade_a_positive_that_owes_something`（欠义务的未过闸 FOUND 降级为 abstain，值保留为 withheld_value，agent.py:1101-1173）。**没有任何路径从子结果构造阳性标签** | ⚡ |
| （跨变量组合） | **`contract/concordance.py` 存在**——见 §1。v0.3.1 的 adjudicator 思想在这一层已实现 | ✅ |
| 逐事件 result 结构 | 无。一次运行一个答案、每字段单值；evidence 列表和 lesions_considered 是记录不是组合子结果 | ⬜ |
| 模型输出严格校验 | 提交状态 enum 从合同结局空间派生（每次运行生成）；**enum 只在 provider 侧强制**，本地兜底是 gate 的 kind 检查（正是为了堵"未声明状态曾是系统里最宽松的结局"那个洞）。malformed 输出走 nudge-and-loop + 容错解析（`extract_json` 手写花括号扫描器，失败返回 `{"__unparsed__": …}`——为 gpt-5.6-luna 工具通道漏进正文加过 `require=` 参数），**不是** v0.3.1 的"一次 schema-repair 重试、仍败则显式 unknown" | 🔶 |

### 2.4 判断账本 / 记录流

| v0.3.1 / 沉淀文档 | acr 现状 | 判定 |
|---|---|---|
| 两条流：JudgmentEvent（入治理账本）vs OperationalTrace（可观测性） | **一条流、另一种切分**：单一 append-only JSONL 存所有事件（35 种，含全量工具入出与病历文本），双通道标记 **DETERMINISTIC vs SELF_REPORTED**。v0.3.1 按**影响**切（会不会改变结论），acr 按**可信度**切（谁算出来的）。两轴正交，各有价值 | 🔶 互补 |
| PointDefinitionRegistry + 封闭选项 + UNMAPPED_POINT | `rule_catalog` 十种稳定 id（内容身份 + 位置 id×text_sha 指纹）**是注册表的一半**；但轨迹里**没有** (situation, options_considered, chosen) 型判断记录——最近的是 SELF_REPORTED 规则引用和 submit_answer 里的 lesions_considered/reported_lesion 一对。UNMAPPED_POINT 的合同级类似物 = abstain_spec/SPEC_INSUFFICIENT 路由报告 | 🔶 |
| JudgmentRecord（六戳/ExecutionContext） | manifest 覆盖问到的一切**除了**：整体 prompt hash（只有组件资产 hash + code_sha——改 TASK 硬编码串不动任何 prompt hash）和 tenant/site | 🔶 |
| 不存 CoT / PHI 最小化 | 全量轨迹（含病历文本）。**这正是 v0.3.1 §6.4 自己定义的开发环境姿态**（RESEARCH=100%）——不冲突，但生产迁移时是整段重写 | ✅（按 §6.4 读） |
| Postgres append-only + 同事务 outbox + 幂等 | 文件系统 runs/；JSONL 按构造 append-only；**一个真实幂等机制**存在（LocalArtifactStore.append_jsonl 按 event_id 去重，刻意"避免隐藏数据库"）；无事务、无 outbox、无租户 | ⬜/🔶 |
| Semantica 可选投影 / PROV-O | **to_capg() 不存在**——docstring 主张 + CLI 占位符（`--capg` 的函数体是打印 n_events）。MCP server 已删。姿态与"可推迟"一致，但连 Null 投影接口都没有 | ⬜ |

### 2.5 开发平面 / 治理

| v0.3.1 / 沉淀文档 | acr 现状 | 判定 |
|---|---|---|
| 分歧度量：Dirichlet 平滑熵 + 最小样本门槛 | `cluster_behaviors` 熵为**裸 Shannon，无平滑**；gold 可选 ✅ | 🔶 |
| impact 由确定性反事实 replay 实测（翻一个判断重算结局） | **无 flip-replay 引擎**——机械资产（规则、关键词、检索计划、录制搜索）有确定性重放；诊断面的"counterfactual test"是**被验证器把关的 agent 主张**，不是机器执行。结构性原因：组合在模型里，翻一个判断无法确定性重算 | ⬜ **依赖 adjudicator** |
| 三分诊（政策/能力/信息缺口）五分诊 | `diagnose()` 六个 dispositions（含未列入常量表的字面量 "NO_REPAIR_NEEDED"——按常量匹配的代码永远看不见它）：RETRIEVAL_FAILURE ≈ 能力缺口、SPEC_AMBIGUITY ≈ 政策缺口、GOLD_NOT_CHART_OBSERVABLE/GOLD_UNRESOLVED ≈ 信息缺口的 gold 侧 | ✅ 粗粒度版 |
| 硬车道：裁决→版本→受影响重放 | **provenance/签名机制是硬车道的真实实现**：clinician_reviewed 要 reviewed_by/reviewed_on/element_hash_at_review，元素被编辑即签名作废降级 draft（spec.py:419-468, 534-537）——计划文档没细化到这个程度 | ✅ acr 更强 |
| 软车道：nav_prior 自动更新 + held-out 回归门 | Retrieval Prior 有产出率/双数字词成本/版本 ✅；**无发布回归门**——`informed_by()`（held-out 污染检查）零生产调用者；`certified` 状态**不可达**（没有代码路径写它）。对照：spec 修复的 `paired_validate` 强制两臂 run 条件逐项相同（比计划的先验门更严） | 🔶 |
| 先例/在线学习/多智能体 OFF | 完全一致：task 三层禁用；SpecPatchProposal 恒 `may_apply_automatically: False`；负对照 19 个 shuffle 种子从 hash 派生、"accepted from nobody"（反 seed-shopping）；**首次真实 certify 就以负 held-out 增益拒绝了**（assetdev.py:1237-1240） | ✅ |
| 消融阶梯 A0-A5（关键比较 = agent vs **状态机**） | 臂 B/A 在跑（query_only 强制同评分器）；C/D 纯设计；**无状态机臂**。反向：计划阶梯**无查询构造隔离臂**（acr 的 C） | 🔶 互补 |
| 隔离/quarantine | SYNY held-out 存在；但 **.gitignore 曾把 1,589 份新生成文档静默排除在 Git 外——包括六份 held-out SYNY 病历**（只有 _ground_truth.json 进了库），冻结保证被反转直到 2026-08-03 修复（.gitignore:68-84 自记） | 🔶 |

### 2.6 PHI / 安全

| v0.3.1 | acr 现状 | 判定 |
|---|---|---|
| PHI 最小化、canary 测试、威胁模型 | 字节扫描测试与 phi_provider_audit 确认已删；`ACR_PHI_SCAN_PATTERN` **有守卫无引擎**——site.py 编译它、require_person_id_pattern 要求它，但 src/tests/tools **无任何消费者**，所谓 pre-commit 扫描不存在；**至少 10 个文件的注释仍指向已删除的 test_no_phi_in_tree.py**；audit_loop.py 里 _EMAIL/_PHONE/_MRN/_DOB 正则成了无引用残骸 | ⬜ **债** |
| 跨租户/跨病例防护 | 结构性防护（Toolbox 绑定单 chart）；note_id 无患者作用域，下游/gold 从不验证归属；无租户概念 | 🔶 |

---

## 3 · 哲学趋同：独立到达的同一批结论

值得单列，因为它们是"两条时间线是同一个思想"的证据，也是迁移时**不用重新辩论**的部分：

1. **not_found ≠ no；unknown ≠ false** —— acr 的 kind 义务 = 计划的 proof obligations。
2. **覆盖用文档集×窗口定义，不用关键词** —— 计划靠推理立铁律，acr 靠 87.4%/31.7% 实测到达。
3. **protocol_gap 是对不清晰的清晰** —— acr 的 abstain_spec（豁免覆盖、欠路由报告）。
4. **给模型放弃按钮就是在测量按钮** —— acr 的 failure kind 不可提交；计划让内核发射预算态。
5. **每个判断点必须被认领（C0 审计）** —— acr 的 check_discriminating_facts 拒绝 + 提示词
   只许声称被强制的东西（两个测试守护）。
6. **多智能体/在线学习/运行时先例默认关** —— 双方一致，acr 还有实测负对照。
7. **人机同 schema** —— 计划的 decision_maker 词表；acr 的 JUDGED/DETERMINISTIC 不许平均
   + adjudication 进 meta_evaluation。
8. **版本戳齐才可比** —— 计划的 ExecutionContext；acr 的 manifest + arm hash +
   experiment_config_hash（"2026-08-06 前后的 run 不可比"）。

## 4 · 真实差集

### v0.3.1 有、acr 无（迁移要新建的）

按依赖顺序：**逐事件的分析单位与量词**（ALL/EXISTS over events、EVENT_NEIGHBORHOOD、
JOIN）→ **变量内的确定性 adjudicator**（有了它才有 flip-replay 的 impact 实测）→
**typed JudgmentRecord + 服务端 point registry**（封闭选项、UNMAPPED_POINT）→
**覆盖门控的阳性**（全称 yes 的义务）→ **SOURCE_UNAVAILABLE / human_review 队列** →
**生产持久层**（DB/outbox/租户/快照/chart_as_of/证据内容 hash）→ **PHI 制度**。

### acr 有、v0.3.1 无（迁移不能丢的）

**测量平面全部**：镜像对（SYNX03 的 gold 已编码 `discriminating_fact_truth`、
`answer_changes_if_fact_flips`、`expect.naive_answer`——**Complete Answer 的金标准已经在
_ground_truth.json 里**，只是没有叫 gold_evidence 的字段）；四臂之 C（查询构造隔离，计划
阶梯没有）；行为熵 gold-optional；JUDGED/DETERMINISTIC 隔离；`query_only` 同评分器纪律；
**provenance 硬车道**（签名作废机制）；强制抽样 + Clopper-Pearson + 帧作废重抽；
`suspected_recognition_failures`（读到了没引用——**已在记录**）；`hits_unread`（自己搜到
没打开——**已在计算**）；负对照反 seed-shopping;2,190 个测试与 prove_end_to_end。

## 5 · 本会话认识的修正（核实推翻的三条）

1. **"no code reads result['text']"（交接文档）——错。** `keyword_hits_among_drawn`
   读全文做强制抽样判定（answer_gate.py:58-85）；labelling.py 读全文建 prior。
   found/read/cited 三布尔的原料不只在盘上，**一半已经被算出来了**（hits_unread、
   suspected_recognition_failures）。Complete Answer 比预想更近。
2. **"CAPG adapter 存在"（我说的）——错。** to_capg 是 docstring 幽灵 + CLI 占位符。
3. **acr 无确定性组合——错了一半。** concordance.py 是真实的跨变量 adjudicator。

## 6 · 孤儿与幽灵清单（做减法/迁移都要用）

运行时孤儿：`clip_and_judge`+`empty_window_policy`（实现完整仅测试调用）、anchors YAML
（仅 specview 读）、`informed_by()`（零生产调用）、`certified` 状态（不可达）、
`replan_from_trace` 及其读的三种事件（无发射者——**manifest 的 replan 块结构性恒零**，
cli_chart.show 那行永远渲染零）。幽灵引用：`build_manifest`（docstring、agent.py、
evals.py、测试都点名，**函数不存在**——manifest 在 agent.py:1411-1681 内联组装）；
tracer.plan/llm/reflect（定义了、当前运行时从不调用——trace.py 模块文档对现 runtime
不成立）；revise_plan 的注释残留；audit_loop 的 PHI 正则残骸；10 处指向已删 PHI 测试的
注释。已知小缺陷：record_evidence 不对 total_chars 做越界检查（跨界 span 静默截短）、
负 offset 静默钳到 0。

## 7 · 迁移分析：三条路

**(a) 在 acr 上演化到 v0.3.1。** 手术是三刀：组合从提示词迁入 adjudicator（concordance.py
证明模式可行，但变量内组合 = 重写 agent 的答案路径）；enforce=False→True 逐检查重议
（acr 删掉的是 **LLM 判断型**强制、实测净负；v0.3.1 要求的是**确定性校验**强制——两类
不同，教训不迁移）；持久层重写。得：2,190 测试、测量平面、已付学费。失：CLAUDE.md
第一条（不留兼容层）会要求把 §6 的孤儿一并清掉——工作量可观。

**(b) 按 brief 新建仓库，acr 保持为研究仪器。** 干净、快（brief 的 Phase 0+1 无 LLM 内核
在 acr 里没有对应物，本来就要新写）；代价是两套语义学并行漂移——v0.3.1 的 task YAML 和
acr 的 spec YAML 是**同一合同的两种方言**，没有任何机制保证一致。

**(c) 合同为接口的双仓库。** acr 做著合同期探针 + 测量平面（生命周期主张的开发端），
新仓库做 v0.3.1 生产端；两边消费同一份合同工件。这是四份计划文档自己的生命周期观
（§6.4：discovery 产物是 Phase 0 的**输入**）落成的仓库结构——但它成立的前提是**先造
方言转换器**（spec YAML → task YAML），否则"同一份合同"是一句口号。

**我的推荐是 (c) 的一个收缩版**：不先建新仓库——先把 v0.3.1 里**便宜且双向有用**的四件
搬进 acr（typed JudgmentRecord 三字段挂到现有动作 JSON 上、熵加 Dirichlet 平滑、
SOURCE_UNAVAILABLE 状态、mirror gold 正式改名 gold_evidence 并接通四个饿着的消费者），
同时用 brief 起 Phase 0——**只写合同和 fixtures，不写内核**——检验方言转换器可不可写。
内核（Phase 1 之后）等转换器的结论再定 (a) 还是 (b)。

## 8 · 留给决策的问题

1. **组合迁不迁进代码**——这是唯一不可两全的分岔。迁：得到 flip-replay 与逐事件任务；
   失去"测量模型无辅助能力"的基线设定（acr 全部历史测量的前提）。
2. **enforce 逐检查选边**——哪些劝告态检查在生产翻回强制（候选：sample_frames_intact、
   listed_documents 已强制、admissibility？）。
3. **术后疼痛切片换不换成 CRC**——brief 的 fixture 是逐事件任务（acr 缺的算子），CRC 是
   R01 的领域。一个切片没法同时压两个。
4. **PHI 制度**在任何真实数据接触前重建——两条时间线在这一点上没有分歧，只有 acr 的债。
