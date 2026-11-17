"""数据库结构初始化：启用 vector 扩展、建表、建向量索引。"""

from sqlalchemy import text

from app.db import engine
from app.models import Base

_HNSW_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw "
    "ON documents USING hnsw (embedding vector_cosine_ops)"
)


def init_db() -> None:
    """幂等初始化：可重复执行。"""
    # 1. 启用 pgvector 扩展（必须在建表前）
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 2. 建表（已存在则跳过）
    Base.metadata.create_all(engine)

    # 3. 向量索引：HNSW 适合余弦相似度检索，无需预先训练
    with engine.begin() as conn:
        conn.execute(text(_HNSW_INDEX_SQL))
