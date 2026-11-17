"""向量检索：按余弦相似度返回最相关的 chunk。"""

from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db import get_session
from app.models import Document
from app.rag.embeddings import get_embedding_client


def search_similar(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """向量相似度检索。

    Args:
        query: 用户查询文本。
        top_k: 返回条数，默认取配置 retrieval_top_k。

    Returns:
        命中列表，每项含 id/source/chunk_index/content/score/metadata。
        score 为余弦相似度（1 - 余弦距离），越接近 1 越相关。
    """
    top_k = top_k or settings.retrieval_top_k
    query_vector = get_embedding_client().embed_query(query)

    distance = Document.embedding.cosine_distance(query_vector).label("distance")
    stmt = select(Document, distance).order_by(distance).limit(top_k)

    hits: list[dict[str, Any]] = []
    with get_session() as session:
        for doc, dist in session.execute(stmt).all():
            hits.append(
                {
                    "id": doc.id,
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "content": doc.content,
                    "score": round(1.0 - float(dist), 6),
                    "metadata": doc.doc_metadata or {},
                }
            )
    return hits
