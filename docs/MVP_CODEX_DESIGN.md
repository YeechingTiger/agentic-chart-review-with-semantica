# MVP 设计：codex harness + semantica 账本 + 工具边界观测

2026-08-25。这是决定，不是选项清单。目标：**最小的端到端闭环先跑起来**，其余一切等它绿了再说。

## 0 · 综合判断（为什么这样切）

当前无法往下走的复杂度来自三处：**自己维护 agent 循环**（agent.py ~1,900 行 + 中间件 + 幽灵代码）、
**测量平面长在运行时里**（gate/coverage/planner 互相缠绕、大半劝告态）、**29 个 CLI**。

harness 换成 codex 一刀解决第一处：**循环、上下文管理、重试、计划工具全部归 harness，
我们只剩工具和合同**。而 MCP 是 codex 的原生接口——acr 当年删掉 MCP server 是因为没有消费者，
现在消费者来了。第二处的解法是把观测从运行时里拔出来，钉在工具边界上（§3）。第三处：MVP 只留 3 个命令。

**一条不变的原则从旧系统原样继承：模型提交完整答案，代码只否决。** 组合不进代码（那是 v0.3.1
的路，等 CRC 逐事件任务再议）。**一条 kill condition 原样继承：镜像对必须区分。**

## 1 · 架构（全部）

```
contract (现有 spec.py 加载链，不动)
        │ 渲染为 prompt
        ▼
codex exec ──MCP──► toolserver（5 个工具 + 提交闸门）──► PatientChart（现有 corpus.py，不动）
   │                        │
   │ 事件流(--json)          │ 每次工具调用全量记录
   ▼                        ▼
Layer2 运行日志          Layer1 权威轨迹 (JSONL, append-only)
（存档，永不计分）              │
                            ├──► score.py（exact + gate + 镜像过程检查）
                            └──► ledger.py ──► semantica（判断账本，write-behind）
```

新代码 5 个文件，预算约 1,200 行。复用不动的：`contract/`（加载与拒绝链、outcomes）、
`chartstore/corpus.py`、镜像对语料与 `_ground_truth.json`、`evals.score` 的 exact 部分。
**旧运行时（review/agent.py 及其中间件）不进 MVP 路径**；MVP 全绿后按仓库惯例写 removal 文档整体删除，
在那之前不碰它（不拿能跑的产品换未完成的复杂度）。

## 2 · 组件与预算

| 文件 | 干什么 | 行数预算 |
|---|---|---|
| `mvp/toolserver.py` | stdio MCP server：`list_documents / search / read / record_evidence / submit_answer`。search/read 带**可选** `objective` 字符串参数（模型自述在解决哪个问题——落轨迹，不强制）。submit 时执行**仅三条**拒绝：状态未声明、FOUND 无证据、未列文档就主张缺席。每次调用把 `{ts, run_id, seq, tool, args, result}` 全量追加进 Layer1 | 400 |
| `mvp/runner.py` | 逐患者驱动：`spec.as_prompt_block()` 渲染 → 起 `codex exec --json`（config.toml 指向 Responses 端点，沙箱只读、approvals 关）→ codex 事件流原样存 Layer2。**config 已对 codex-cli 0.149.1 实测核定**：`wire_api = "responses"`（chat 已被移除）；`[mcp_servers.chart] default_tools_approval_mode = "approve"`（默认 `auto` 把无注解工具当写操作，配合 `approval_policy = "never"` 会把每次调用无声拒掉）；MCP 工具在请求里以 namespace 条目出现（`mcp__chart`），function_call 需带独立 `namespace` 字段 | 250 |
| `mvp/fake_model.py` | 本地 Responses-API 假模型：SSE 三帧（created / output_item.done / completed），按脚本回放工具调用轨迹。测试与本沙箱（真实模型端点被出口代理拦截）共用 | 140 |
| `mvp/score.py` | exact 对 `_ground_truth.json`；gate 结果；**镜像过程检查**：从 Layer1 判定"是否有一次 search/read 触及了 discriminating fact 的范围"（用 `rule_gold.discriminating_fact_truth` 现有字段，**不改名不迁 schema**）。计分**只读 Layer1**——用"删掉 Layer2 后照常出分"来证明 | 250 |
| `mvp/ledger.py` | `ReviewLedger` 协议（record_evidence / record_judgment / record_result / chain）+ `SemanticaLedger` + `NullLedger`。semantica==0.6.6 锁版本，进程内 Python API，**不开 REST/MCP/Explorer**。write-behind：账本故障不影响运行结果（用 NullLedger 对照测试钉死） | 250 |
| `mvp/cli.py` | 命令：`compile`（加载合同并拒绝坏的——现有链）、`run`（一个患者）、`ingest`、`chain`；`score` 待 score.py 落地后加入 | 100 |

## 3 · 观测（"trace 怎么做"的答案）

一条原则解决全部问题：

> **凡需要被信任的，都在工具边界上由服务端记录；harness 自己的日志永不承重。**

- **Layer1（权威）**：MCP 边界的 JSONL。模型做过的每件"算数的事"必然经过工具，所以这一层
  按构造完整。DETERMINISTIC 通道。**计分只准读它。**
  细粒度决策点也在这层：`note_decision` 工具（facing/decision/because/options，semantica 的
  局面/凭据/定论框架原样）——**说了什么**是自报内容，但**何时、以什么顺序说的**是服务端事实，
  所以决策点与动作在 seq 上严格穿插可排。永不被闸门拒（拒了就是在测按钮）。
- **Layer2（免费）**：codex `--json` 事件流原样落盘——思考摘要（reasoning item，每回合先于
  该回合的工具调用出现）、token 计数。SELF_REPORTED 通道，只供调试与阅读，永不计分。
  `observe.decision_trace` 把两层合成一条"想→决→做→看→交→判"的可顺读轨迹（`cli trace`），
  对齐靠位置而非时钟：Layer2 的第 k 个 mcp_tool_call 就是 Layer1 的第 k 个 tool_call。
  删掉 Layer2，轨迹只失去 thought 行——其余每步原样（观测独立性照旧）。
- **Layer3（账本）**：从 Layer1 蒸馏出判断（证据记录、提交+闸门判决、终局）写进 semantica：
  evidence 节点 → judgment 节点 → result 节点，因果边相连。三个动词的 MVP 版本——
  审计 = `chain(case)` 一跳到底；对比 = 同 chart 两次 run 的节点 diff；沉淀 = 以后。

这个切法顺带买到**harness 无关性**：换 harness 只换 Layer2，Layer1/3 与计分完全不动——
旧的四臂对比以后想恢复，臂与臂天然同尺。

## 4 · 验收（四条，即 kill condition）

1. **端到端**：27 个合成患者全部经 codex 跑出过闸答案。
2. **镜像对**：SYN0001→`20230412`、SYNX03→`20220309`，且**过程检查对两者给出可区分的判定**
   （SYNX03 要求注意到一份缺席的文档）。区分不了 = 仪器无效 = 停下修，不往下走。
3. **审计链**：任一患者 `chain(case)` 返回 证据→判断→答案 的完整链。
4. **观测独立性**：删除 Layer2 文件后，score 输出逐字节不变。

## 5 · 明确砍掉的（决定，不是遗漏）

四臂阶梯、judge/对抗核验、行为熵命令、coverage planner 与扩张预算、强制抽样、
OpenThread/缺口台账（codex 的 plan 工具落 Layer2 顶替）、skills 渐进披露、improvement 平面、
spec_repair、concordance（CRC 时再启用）、29 个 CLI、provenance 签名机制之外的一切 speclint。
每一样都在 `THE_IDEAL_SYSTEM.md` 的环上有座位，**回来的条件是 MVP 绿了且它有明确消费者**。

## 6 · 顺序

1. toolserver + Layer1 + 三条拒绝 → 用手工 JSON 调用（无 codex）在 SYN0001 上打通
2. codex exec 接入（config TO-VERIFY）→ 27 患者跑通 → 验收 1
3. score + 镜像过程检查 → 验收 2、4
4. semantica ledger → 验收 3
5. 全绿 → 写 removal 文档，删旧运行时路径

一步一验，绿了才走下一步。第 2 步是唯一有外部未知数的地方（codex 配置细节），放在最前面暴露。
