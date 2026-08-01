"""把一次运行的审阅链渲染成可读的一串动作，从第一步到停止。

为什么需要它
----------
`chain_report` 报的是链的**健康度**（可解析比率、深度、断点）。这个工具报的是链**本身**：
按顺序，模型做了什么、给出的理由是什么、gate 说了什么、最后为什么停。一个数字告诉你链有
没有断，一串动作才告诉你这条 policy 长什么样。

四个臂跑同一个病人，把它们并排读，policy 的形状就直接可见 —— 这比任何一张卡的描述都准，
因为卡是意图，这是行为。

用法：
    .venv/bin/python tools/render_chain.py <manifest.json> [--max N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from acr.evaluation import evals as E  # noqa: E402
from acr.evaluation.evidence_chain import chain_report  # noqa: E402

#: 每种工具在链里怎么读。名字不是给机器的，是给读的人的。
VERB = {
    "list_documents": "清点", "document_type_summary": "清点类型",
    "search_notes": "检索", "search_documents": "检索", "search": "检索",
    "read_document": "读", "read_documents_batch": "批量读", "read_section": "读段落",
    "submit_answer": "提交", "gate": "闸门",
}


def _arg(ev: dict) -> str:
    a = ev.get("args") or {}
    for k in ("q", "query", "term", "note_id", "doc", "doc_type_contains"):
        if a.get(k):
            return str(a[k])[:44]
    if a.get("date_from") or a.get("date_to"):
        return f"{a.get('date_from', '')}..{a.get('date_to', '')}"
    return ""


def render(path: pathlib.Path, max_rows: int = 40) -> None:
    run = E.RunRecord.from_manifest(str(path))
    rep = chain_report(run)
    m = run.manifest
    ans = m.get("answer") or {}

    print(f"\n{'=' * 78}\n{path.parent.name.split('__')[0]:20s} {path.stem}")
    print(f"{'=' * 78}")
    print(f"答案 {ans.get('status')} {json.dumps(ans.get('value') or {}, ensure_ascii=False)[:60]}"
          f"  | gate_validated={m.get('gate_validated')}")
    print(f"链   {rep['n_links']} 步 | 可解析 {rep['n_grounded']} | 散文 {rep['n_prose_only']}"
          f" | 无来源 {rep['n_unsourced']} | 深度 {rep['max_depth']}")
    print(f"停在 {m.get('termination_reason') or '—'}"
          f" | 步数 {m.get('steps')} | 模型调用 {(m.get('usage') or {}).get('llm_calls')}")
    print("-" * 78)

    links = rep["links"]
    shown = links[:max_rows]
    for ln in shown:
        ev = next((e for e in run.trace if e.get("seq") == ln["seq"]), {})
        tool = ln["tool"].split(".")[-1]
        verb = VERB.get(tool, tool)
        mark = {"GROUNDED": "→", "PROSE_ONLY": "·", "UNSOURCED": " ",
                "UNRESOLVED_REF": "?", "FORWARD_REF": "!"}[ln["status"]]
        ref = f" ←{ln['ref']}" if ln.get("ref") else ""
        why = (ln["why"] or "").replace("\n", " ")[:52]
        print(f"{ln['seq']:>3} {mark} {verb:<10s} {_arg(ev):<46s}{ref}")
        if why:
            print(f"      └ {why}")
    if len(links) > max_rows:
        print(f"    … 另有 {len(links) - max_rows} 步（--max 调整）")
    print("图例：→ 指针可解析   · 只有散文理由   (空) 无理由   ? 解析不到   ! 指向未来")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args()
    for m in a.manifests:
        render(pathlib.Path(m), a.max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
