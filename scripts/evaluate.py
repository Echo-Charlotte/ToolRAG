"""M5 评测脚本：对检索与问答进行量化评测，并生成 Markdown 报告。

用法：
    PYTHONPATH=. .venv/bin/python scripts/evaluate.py
    PYTHONPATH=. .venv/bin/python scripts/evaluate.py --mode hybrid --top-k 5
    PYTHONPATH=. .venv/bin/python scripts/evaluate.py --skip-agent   # 只跑检索指标（不调用 LLM）

流程：
1. 把 eval/corpus/ 下的语料入库（source 前缀 eval://，覆盖式更新）；
2. 对 eval/qa.json 的每道题：
   - 检索指标：Recall@k / Hit@k / MRR / 延迟；
   - 问答指标（可选）：关键词覆盖率（准确率）/ 引用正确率 / 兜底率 / 轮数 / Token / 延迟；
3. 汇总结果写入 eval/report.md。
"""

# ruff: noqa: E402

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete

from app.agent.tool_agent import run_agent
from app.config import settings
from app.db import get_session
from app.eval.dataset import QAItem, load_qa
from app.eval.metrics import (
    citation_accuracy,
    hit_at_k,
    keyword_coverage,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from app.models import Document
from app.rag.ingest import ingest_text
from app.rag.retriever import search

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "eval" / "corpus"
QA_PATH = ROOT / "eval" / "qa.json"
REPORT_PATH = ROOT / "eval" / "report.md"
PREFIX = "eval://"


def _norm(source: str) -> str:
    """把 eval://db.md 归一化为 db.md，便于与评测集里的来源名对比。"""
    return source.split("://", 1)[-1]


def ingest_corpus() -> int:
    """把 eval/corpus 下的语料入库（先清理旧的评测数据）。"""
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"未找到评测语料：{CORPUS_DIR}")
    with get_session() as session:
        session.execute(delete(Document).where(Document.source.like(f"{PREFIX}%")))
    for path in files:
        ingest_text(f"{PREFIX}{path.name}", path.read_text(encoding="utf-8"), {"title": path.stem})
    return len(files)


def eval_retrieval(items: list[QAItem], mode: str, top_k: int) -> list[dict]:
    """对每道题跑检索，计算 Recall@k / Hit@k / MRR 与延迟。"""
    rows: list[dict] = []
    for item in items:
        start = time.perf_counter()
        hits = search(item.question, top_k=top_k, mode=mode)
        latency = time.perf_counter() - start
        retrieved = [_norm(hit["source"]) for hit in hits]
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "retrieved": retrieved,
                "recall": recall_at_k(retrieved, item.expected_sources, top_k),
                "hit": hit_at_k(retrieved, item.expected_sources, top_k),
                "rr": reciprocal_rank(retrieved, item.expected_sources),
                "latency": latency,
            }
        )
    return rows


def eval_agent(items: list[QAItem]) -> list[dict]:
    """对每道题跑完整 Agent 问答，计算答案覆盖、引用正确率、兜底率与延迟。"""
    rows: list[dict] = []
    for item in items:
        start = time.perf_counter()
        result = run_agent(item.question)
        latency = time.perf_counter() - start
        cited = [_norm(citation["source"]) for citation in result.citations]
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "coverage": keyword_coverage(result.answer, item.keywords),
                "citation": citation_accuracy(cited, item.expected_sources),
                "fallback": result.fallback,
                "rounds": result.rounds,
                "tokens": result.usage.total_tokens,
                "latency": latency,
            }
        )
    return rows


def build_report(mode: str, top_k: int, ret_rows: list[dict], agent_rows: list[dict]) -> str:
    lines: list[str] = [
        "# ToolRAG 评测报告（M5）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 检索模式：`{mode}`，top_k = {top_k}",
        f"- Embedding：`{settings.embedding_model}`；对话模型：`{settings.llm_model}`",
        f"- 评测题数：{len(ret_rows)}",
        "",
        "## 1. 检索指标",
        "",
        f"- **Recall@{top_k}**：{mean([r['recall'] for r in ret_rows]):.4f}",
        f"- **Hit@{top_k}**：{mean([1.0 if r['hit'] else 0.0 for r in ret_rows]):.4f}",
        f"- **MRR**：{mean([r['rr'] for r in ret_rows]):.4f}",
        f"- **平均检索延迟**：{mean([r['latency'] for r in ret_rows]) * 1000:.1f} ms",
        "",
        "| ID | 问题 | 命中来源 | Recall | Hit | RR | 延迟(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ret_rows:
        retrieved = ", ".join(row["retrieved"][:3]) or "—"
        lines.append(
            f"| {row['id']} | {row['question']} | {retrieved} | "
            f"{row['recall']:.2f} | {'是' if row['hit'] else '否'} | {row['rr']:.2f} | "
            f"{row['latency'] * 1000:.1f} |"
        )

    if agent_rows:
        lines += [
            "",
            "## 2. 问答指标（Agent）",
            "",
            f"- **答案准确率（关键词覆盖）**：{mean([r['coverage'] for r in agent_rows]):.4f}",
            f"- **引用正确率**：{mean([r['citation'] for r in agent_rows]):.4f}",
            f"- **兜底率**：{mean([1.0 if r['fallback'] else 0.0 for r in agent_rows]):.4f}",
            f"- **平均工具轮数**：{mean([r['rounds'] for r in agent_rows]):.2f}",
            f"- **平均问答延迟**：{mean([r['latency'] for r in agent_rows]):.2f} s",
            f"- **总 Token 用量**：{sum(r['tokens'] for r in agent_rows)}",
            "",
            "| ID | 问题 | 准确率 | 引用正确率 | 兜底 | 轮数 | Token | 延迟(s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in agent_rows:
            lines.append(
                f"| {row['id']} | {row['question']} | {row['coverage']:.2f} | "
                f"{row['citation']:.2f} | {'是' if row['fallback'] else '否'} | {row['rounds']} | "
                f"{row['tokens']} | {row['latency']:.2f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ToolRAG M5 评测")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "keyword", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-agent", action="store_true", help="只跑检索指标，不调用 LLM")
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    count = ingest_corpus()
    items = load_qa(QA_PATH)
    print(f"ingested corpus docs: {count}, qa items: {len(items)}")

    ret_rows = eval_retrieval(items, args.mode, args.top_k)
    print(
        f"retrieval mode={args.mode} top_k={args.top_k} "
        f"recall={mean([r['recall'] for r in ret_rows]):.4f} "
        f"hit={mean([1.0 if r['hit'] else 0.0 for r in ret_rows]):.4f} "
        f"mrr={mean([r['rr'] for r in ret_rows]):.4f}"
    )

    agent_rows: list[dict] = []
    if not args.skip_agent:
        agent_rows = eval_agent(items)
        print(
            f"agent coverage={mean([r['coverage'] for r in agent_rows]):.4f} "
            f"citation_acc={mean([r['citation'] for r in agent_rows]):.4f} "
            f"fallback_rate={mean([1.0 if r['fallback'] else 0.0 for r in agent_rows]):.4f} "
            f"avg_latency={mean([r['latency'] for r in agent_rows]):.2f}s"
        )

    report = build_report(args.mode, args.top_k, ret_rows, agent_rows)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
