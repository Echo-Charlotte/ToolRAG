"""数据模型：文档切块与向量。"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class Document(Base):
    """文档切块表：一行代表原文中的一个 chunk 及其向量。

    注意：属性名用 doc_metadata，映射到数据库列名 metadata
    （metadata 是 SQLAlchemy 声明式基类的保留名，不能直接用）。
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    # 关键词检索用的 tsvector：内容为「中文二元切分后的 token 串」经 to_tsvector('simple') 转换
    # 允许为空是为了兼容 M1 时期写入的旧数据（启动时会自动回填）
    content_tsv: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} source={self.source!r} chunk={self.chunk_index}>"
