# ACR 模块架构 v2

> 状态：已实现的兼容演进基线  
> 日期：2026-07-29

## 1. 设计原则

只有能够独立运行、具有 typed input/output、权限、停止条件和 certification
生命周期的单元才是 module。

| 名称 | 含义 | 是否是顶层 module |
|---|---|---|
| Module | 可独立运行和认证的能力 | 是 |
| Skill | Agent evaluator 内部的方法说明 | 否 |
| Capability | 受 scope 和 budget 控制的工具能力 | 否 |
| Stage | 一个 module 内部的执行阶段 | 否 |
| Pipeline | Module 的条件、依赖和数据绑定 | 不是 module |
| Task | 一次运行的数据、truth、预算和权限绑定 | 不是 module |

系统采用四个平面，而不是把所有检查都放进 Eval：

```text
Execution
  → canonical Trajectory
  → Audit              行为事实和治理边界
  → Evaluation         任务质量和因果归因
  → Improvement        修复、paired validation、采用
```

现有 deepagents extraction 保持原位。canonical trajectory 直接从当前
trace/manifest 构造；旧 EvalLoop、重复 evaluator catalog 和兼容 CLI 已移除。

## 2. Stable Kernel

`acr.kernel` 定义跨任务稳定的公共对象：

- `AssetRef`：所有 spec、policy、prompt、skill、tool、evaluator、audit rule 和
  repair strategy 的内容寻址引用。
- `Trajectory`：一次完整 agent execution 的不可变、analysis-ready 记录。
- `TargetRef`：signal 所解释的 run/field/evidence/tool/gate 等明确目标。
- `SignalEnvelope`：Audit、Evaluation、Attribution 和 Repair 输出共享的薄 envelope。

`TrajectoryAdapter` 从 extraction manifest/trace 生成 canonical trajectory。Chart 原文仍留在
repo 外本地 artifact；trajectory event 中的自由文本只保留 hash 和长度，分析模块
按授权读取原始本地引用。

## 3. Module Families

`acr.modules` 只定义五类 module protocol：

1. `RuntimePolicy`：搜索、coverage、扩展和停止策略。
2. `RuntimeControl`：当前请求内的确定性 allow/deny/require。
3. `AuditRule`：truth-blind 行为边界检查。
4. `Evaluator`：CODE/LLM/AGENT/HUMAN 质量评价。
5. `RepairStrategy`：从已确认 signal 路由到有 owner 的 repair obligation。

YAML 只能引用显式注册的 `implementation_id`。系统不允许从 YAML 动态 import
任意 Python 代码。

## 4. Asset、Pipeline、Task 和 Certification

原 `EvaluatorSpec` 同时保存 evaluator、selector、dependency、budget 和 synthetic
fixtures。v2 将其拆开：

- `ModuleAsset`：module 身份、I/O、runner、truth modes、capability request 和最大权限。
- `PipelineProfile`：node、condition、dependency、input binding、capability allowlist、
  budget ceiling 和权限 ceiling。
- `EvaluationTask`：trajectory cohort、truth mode、model、seed、实际 budget 和 grant。
- `CertificationSuite`：must-pass/must-fail fixtures、calibration cohort 和阈值。

Task 的 effective capability 是：

```text
module requested
∩ pipeline allowed
∩ task granted
∩ current patient scope
```

Task 只能缩小 capability、budget 和 authority，不能扩大。

## 5. Audit 与 Evaluation

### Audit

AgentLoop 的 Audit 包含两部分：

- Audit Facts：Session、Turn、Tool、Token 和行为事实。
- Risk Audit：Finding → correlation → Incident → investigation。

ACR v1 只实现 application-level Risk Audit：

- patient boundary
- PHI/provider boundary
- undeclared tool
- local artifact boundary
- trajectory integrity
- hard runtime-control conformance

Audit 不接收 `TruthContext`，不判断 clinical correctness，也不产生 semantic spec
repair。Audit output 独立存入：

```text
<local-root>/audit/findings.jsonl
<local-root>/audit/incidents.jsonl
```

eBPF、进程/文件/网络采集、实体图谱、实时告警和 SIEM 不在 v1 范围。

### Evaluation

`acr.evaluation_pipeline` 的 `EvaluationContext` 将普通 typed channels 与
`TruthContext` 分开。BLIND channel 递归拒绝 gold、answer key 或 registry reference。

`EvaluationResult` 不含 `AuditFinding` 或 `AuditIncident`。安全 signal 可以作为引用
进入 attribution，但不能被 evaluator 重新裁判。

内置 v2 CODE evaluators：

- `evidence-validity`
- `gate-effectiveness`

`causal-attribution@2.0.0` 已登记为 AGENT `ModuleAsset`，具体 tool loop 仍由
`acr attribute` 执行；其中的 targeted probe 覆盖未读证据矛盾检查，不再维护一套
重复的 standalone contradiction runtime。

## 6. Attribution 的内部模块

原 `attribution_modules` 实际表示一个 evaluator 内部的 stages，而不是八个独立
evaluators。正式名称为 `AttributionStage`、`AttributionStageProfile` 和
`AttributionStageRegistry`。

```text
target framing
→ trace reconstruction
→ cause hypothesis
→ targeted probe
→ counterfactual replay
→ skeptic review
→ attribution gate
```

整个序列仍然只产生一个 `AttributionReport`。

## 7. Coverage

Coverage 被拆成：

- `CoveragePolicy`：怎样搜索，属于 RuntimePolicy。
- `CoverageState`：实际读到什么，属于 Trajectory。
- coverage effectiveness：策略是否有效，属于 Evaluation/Experiment。

`acr.runtime_profiles` 提供两个可比较基线：

- `witness-first-baseline`
- `current-stratified-coverage`

它们已经接入 `run_patient`、`acr run|batch|consistency` 和 `acr extract` 的
`--runtime-profile`。默认值仍为 `current-stratified-coverage`。每个 manifest 和
`run_start` event 都保存 profile ref、version 和 content hash。

`witness-first-baseline` 仍执行 patient scope、工具权限、字段格式、answer check、
positive evidence 和 open-thread control；它要求先列出患者文档，但不会执行负例的
forced sampling 或 stratified exclusion gate。它接受的 `EVIDENCE_INSUFFICIENT`
明确记录 `negative_basis=WITNESS_FIRST_BASELINE`，不附加 `coverage_attested`，
也不作为 coverage-validated answer 报告。

任何“必须看全”的性能主张都需要在相同 patient、model、seed 和 budget 下比较
accuracy、critical miss、overclaim、abstention、evidence validity、documents read
和 cost。

## 8. Improvement

`acr.repair_loop` 保证：

- Audit Incident 只路由到 security/control repair。
- Evaluation signal 经 attribution 后路由到 spec/retrieval/skill/gate/runtime owner。
- `REGISTRY_REFERENCE` 只能产生 adjudication/clinician question。
- Semantic repair 必须同时具有 GOLD 和 human adjudication。
- 所有 proposal 仍需 paired validation；有 critical、per-case 或 subgroup regression
  时不得接受。

## 9. Catalog 与 CLI

独立 catalogs：

```text
module_catalog/
pipeline_catalog/
certification_catalog/
```

CLI：

- `acr audit rules|run|summarize|incidents`
- `acr evaluation modules|validate|run|batch|summarize|compare`
- 原 `acr eval` 保持纯 CODE
- `acr attribute` 是唯一 model-using attribution 入口

## 10. 明确不做

- eBPF 和 host/process/network 全栈采集
- 动态 Python plugin import
- 云控制台、dashboard、RBAC、SIEM
- 通用数据湖或向量 dataset 平台
- Memory/Experience 自动注入
- 在线自动修改或发布 semantic spec
- LLM 驱动的安全阻断
