"""具体工具实现：文档检索、来源列表、读取全文。"""

from typing import Any

from sqlalchemy import func, select

from app.db import get_session
from app.models import Document
from app.rag.retriever import search


def search_documents(query: str, top_k: int = 5, mode: str = "hybrid") -> dict[str, Any]:
    """在文档库中检索最相关的片段。

    Args:
        query: 自然语言查询。
        top_k: 返回条数（1-20）。
        mode: 检索模式 vector / keyword / hybrid。

    Returns:
        {"query", "mode", "results": [{source, chunk_index, content, score}]}
    """
    top_k = max(1, min(int(top_k), 20))
    hits = search(query, top_k=top_k, mode=mode)
    return {
        "query": query,
        "mode": mode,
        "results": [
            {
                "source": hit["source"],
                "chunk_index": hit["chunk_index"],
                "content": hit["content"],
                "score": hit["score"],
            }
            for hit in hits
        ],
    }


def list_sources() -> dict[str, Any]:
    """列出所有文档来源及各自的切块数量。"""
    stmt = (
        select(Document.source, func.count(Document.id).label("chunks"))
        .group_by(Document.source)
        .order_by(Document.source)
    )
    with get_session() as session:
        rows = session.execute(stmt).all()
    sources = [{"source": source, "chunks": int(count)} for source, count in rows]
    return {"total": len(sources), "sources": sources}


def get_document_content(source: str) -> dict[str, Any]:
    """按 source 取出整篇文档（按 chunk_index 顺序拼接）。"""
    stmt = (
        select(Document.chunk_index, Document.content)
        .where(Document.source == source)
        .order_by(Document.chunk_index)
    )
    with get_session() as session:
        rows = session.execute(stmt).all()
    if not rows:
        return {"source": source, "found": False, "chunks": []}
    return {
        "source": source,
        "found": True,
        "chunks": [
            {"chunk_index": int(index), "content": content} for index, content in rows
        ],
    }
