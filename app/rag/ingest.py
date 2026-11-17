"""文档入库：切分 → 向量化 → 写入数据库。"""

from typing import Any

from sqlalchemy import delete

from app.config import settings
from app.db import get_session
from app.models import Document
from app.rag.chunking import split_text
from app.rag.embeddings import get_embedding_client


def ingest_text(source: str, text: str, metadata: dict[str, Any] | None = None) -> int:
    """把一篇文档切块、向量化并写入数据库。

    同一 source 重复入库会先删除旧数据（覆盖式更新），避免重复。

    Args:
        source: 文档来源标识，如文件名或 URL。
        text: 文档正文。
        metadata: 附加元数据（如标题、作者），随 chunk 一并保存。

    Returns:
        写入的 chunk 数量；原文为空时返回 0。
    """
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    vectors = get_embedding_client().embed_documents(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError(f"向量数量({len(vectors)})与 chunk 数量({len(chunks)})不一致")

    meta = metadata or {}
    with get_session() as session:
        session.execute(delete(Document).where(Document.source == source))
        session.add_all(
            [
                Document(
                    source=source,
                    chunk_index=index,
                    content=chunk,
                    embedding=vector,
                    doc_metadata=meta,
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
        )
    return len(chunks)
