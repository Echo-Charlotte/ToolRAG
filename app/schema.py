"""数据库结构初始化：启用扩展、建表、建索引，并做轻量迁移。"""

from sqlalchemy import func, select, text, update

from app.db import engine, get_session
from app.models import Base, Document
from app.rag.tokenizer import to_index_text

_HNSW_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw "
    "ON documents USING hnsw (embedding vector_cosine_ops)"
)

# M1 时期创建的表没有 content_tsv，这里做一次幂等的加列
_TSV_COLUMN_SQL = "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsv tsvector"

_TSV_GIN_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS documents_content_tsv_gin "
    "ON documents USING gin (content_tsv)"
)


def init_db() -> None:
    """幂等初始化：可重复执行。"""
    # 1. 启用 pgvector 扩展（必须在建表前）
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 2. 建表（新库直接带上 content_tsv 列；已存在的表不会自动加列）
    Base.metadata.create_all(engine)

    # 3. 建索引 / 补齐旧表的列
    with engine.begin() as conn:
        conn.execute(text(_TSV_COLUMN_SQL))
        conn.execute(text(_HNSW_INDEX_SQL))
        conn.execute(text(_TSV_GIN_INDEX_SQL))

    # 4. 回填历史数据的 content_tsv（新加的列默认是 NULL，会导致关键词检索漏召回）
    _backfill_missing_tsv()


def _backfill_missing_tsv() -> None:
    """为 content_tsv 为空的历史数据补齐关键词索引内容。"""
    with get_session() as session:
        rows = session.execute(
            select(Document.id, Document.content).where(Document.content_tsv.is_(None))
        ).all()
        for doc_id, content in rows:
            session.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(content_tsv=func.to_tsvector("simple", to_index_text(content)))
            )
