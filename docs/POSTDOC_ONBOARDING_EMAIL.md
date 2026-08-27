# Email to students / postdocs

**Subject:** Agentic Chart Review with Semantica：安装与三本 Notebook 上手说明

各位好，

这个项目用于帮助人类审核 chart-review agent 的决策过程。整体流程是：

```text
Codex chart-review agent
        ↓
local Langtrace raw trace
        ↓
ReAct cycles
        ↓
Luna decision reconstruction
        ↓
Semantica ContextGraph
        ↓
human review / cross-run comparison / policy-impact analysis
```

请按下面步骤开始。

## 1. Clone `main` 并安装

需要 Python 3.12、`uv`、Node/npm 和 Docker Compose。这个 GitHub 仓库是 public，可以
直接 clone。

```bash
git clone --branch main --single-branch https://github.com/YeechingTiger/agentic-chart-review-with-semantica.git
cd agentic-chart-review-with-semantica

uv venv --python 3.12
uv pip install -e ".[ledger,ledger-ui,dev,notebook]"
uv pip install --no-deps semantica==0.6.6

npm install -g @openai/codex@latest
codex --version
```

Semantica 使用 `--no-deps` 是有意的：完整依赖包含本项目不需要的大型 embedding、NLP
和视觉包。

如果要运行人工审核界面，还需安装项目扩展过的 Semantica Explorer：

```bash
tools/install_semantica_decision_review.sh
```

## 2. 启动本地 Langtrace

在项目目录之外 clone 官方 Langtrace：

```bash
cd ..
git clone https://github.com/Scale3-Labs/langtrace.git
cd langtrace
docker compose up -d
```

打开 `http://localhost:3000`，创建一个 Langtrace project，并生成 project API key。请保存
project ID 和 API key。不要对仍需保留的数据运行 `docker compose down -v`，因为它会删除
本地 trace storage。

然后回到项目目录：

```bash
cd ../agentic-chart-review-with-semantica
```

## 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
ACR_API_BASE=https://openrouter.ai/api/v1
ACR_API_KEY=<YOUR_OPENROUTER_KEY>
ACR_MODEL=openai/gpt-5.6-luna

OPENROUTER_API_KEY=<YOUR_OPENROUTER_KEY>

LANGTRACE_API_HOST=http://127.0.0.1:3000
LANGTRACE_PROJECT_ID=<YOUR_LANGTRACE_PROJECT_ID>
LANGTRACE_API_KEY=<YOUR_LANGTRACE_API_KEY>

SEMANTICA_API_KEY=<A_STRONG_LOCAL_ONLY_SECRET>
```

加载环境变量：

```bash
set -a
source .env
set +a
```

请不要把任何 key 放进 notebook cell、CLI 参数、截图或 Git commit，也不要通过邮件回复
key。

## 4. 按顺序阅读和运行三本 Notebook

```bash
.venv/bin/jupyter lab notebooks/
```

请按以下顺序：

1. `01_run_chart_review_experiments.ipynb`：先理解 ReAct-style chart agent 和 7 个工具，
   再运行或复用小型 cohort；默认只看 agent answer 与 synthetic gold。
2. `02_trace_to_decision_chain.ipynb`：把一个真实 run 的 raw trace 当作“长收据”，再跟着
   8 个可以逐一判断对错的问题走到最终答案，并下钻真正值得审核的搜索覆盖步骤。
3. `03_semantica_decision_intelligence.ipynb`：把 ContextGraph 当作“有索引的决策卡片盒”，
   只看一张可读 Decision、similar decisions、同一 evidence 的真实分歧，以及 Policy
   变更后的 re-audit queue。

每本 source notebook 都有对应的 `.executed.ipynb`，其中保留了已经完成的真实
OpenRouter Luna/Terra 运行结果。请先阅读 executed notebooks；只阅读它们不会产生新的模型
费用。在 fresh clone 上，要真正执行 Notebook 2 和 3，需先让 Notebook 1 生成本地 run/ledger。

## 5. 生成新的真实运行（会产生费用）

```bash
export ACR_TUTORIAL_MODE=live
.venv/bin/jupyter lab notebooks/
```

live walkthrough 是 2 cases × 2 task arms × Luna。建议先阅读 executed notebooks，再运行；
请在 OpenRouter 设置合理的 spending limit 并检查 usage dashboard。Luna reconstruction 默认
运行两次，以检查 Decision Episode alignment 是否稳定，因此一次 chart-review run 可能对应
多次模型请求。

## 6. 打开人工审核 UI

Notebook 1 或 CLI 运行结束后，从输出中取得 `<RUN_ID>` 和 `<RUN_DIR>`：

```bash
acr review-ui <RUN_ID> \
  --run-dir "$(pwd)/runs/mvp/<RUN_DIR>" \
  --ledger "$(pwd)/runs/mvp/ledger.json" \
  --port 8877
```

打开 `http://127.0.0.1:8877`。建议按顺序审核：agent 看到了哪些 notes/evidence；为什么
做出当前选择；引用了哪条 guideline/Policy；引用是否支持判断；为什么继续搜索或停止；
最终结论如何由前面的 Decision Episodes 得出；以及可疑步骤对应的 raw Langtrace。

## 7. 验证安装

```bash
.venv/bin/python -m pytest tests/test_mvp_*.py -q
.venv/bin/ruff check src tests tools
acr --help
```

普通测试使用固定 trace 和 stub reconstruction，不产生 OpenRouter 费用。只有明确启用
`ACR_TUTORIAL_MODE=live` 或执行真实 CLI run/reconstruct 时才会产生模型调用。

开始时请先完成三本 executed notebooks 的 walkthrough。之后选择一个 case 运行 `pilot`，
并记录：

- 哪个 Decision Episode 最难判断；
- raw trace 是否支持重构出的 Decision；
- task-only 是否出现未受 guideline 约束的分歧；
- policy-bundle 是否真的引用并正确应用了 Policy；
- 哪个问题应归因于 retrieval、instrumentation、guideline wording 或 model capability。

谢谢。
