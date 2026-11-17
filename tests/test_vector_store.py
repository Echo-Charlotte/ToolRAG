"""向量入库与检索的集成测试。

需要可用的 PostgreSQL + PGVector：不可用时自动跳过。
embedding 用确定性伪向量替换，测试只验证「切分 → 存储 → 向量检索」链路。
"""

import pytest
from sqlalchemy import delete, text

from app.db import engine, get_session
from app.models import Document
from app.rag import ingest as ingest_module
from app.rag import retriever as retriever_module


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="需要可用的 PostgreSQL/PGVector")


class _FakeEmbeddingClient:
    """字符哈希伪向量：相同字符构成相似的文本会得到相近向量。"""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, content: str) -> list[float]:
        vec = [0.0] * self.dim
        for ch in content:
            vec[ord(ch) % self.dim] += 1.0
        norm = sum(value * value for value in vec) ** 0.5 or 1.0
        return [value / norm for value in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, content: str) -> list[float]:
        return self._vector(content)


@pytest.fixture
def fake_embedding(monkeypatch: pytest.MonkeyPatch) -> _FakeEmbeddingClient:
    fake = _FakeEmbeddingClient()
    monkeypatch.setattr(ingest_module, "get_embedding_client", lambda: fake)
    monkeypatch.setattr(retriever_module, "get_embedding_client", lambda: fake)
    return fake


@pytest.fixture
def prepared_db() -> None:
    from app.schema import init_db

    init_db()


@pytest.fixture
def cleanup() -> None:
    yield
    with get_session() as session:
        session.execute(delete(Document).where(Document.source.like("test://%")))


def test_ingest_then_search(
    fake_embedding: _FakeEmbeddingClient, prepared_db: None, cleanup: None
) -> None:
    source = "test://fruits"
    count = ingest_module.ingest_text(source, "苹果是一种水果。香蕉也是水果。", {"title": "水果"})

    assert count >= 1

    hits = retriever_module.search_similar("苹果", top_k=3)
    assert hits
    assert hits[0]["source"] == source
    assert 0.0 <= hits[0]["score"] <= 1.0
    assert hits[0]["metadata"] == {"title": "水果"}


def test_reingest_same_source_overwrites(
    fake_embedding: _FakeEmbeddingClient, prepared_db: None, cleanup: None
) -> None:
    source = "test://overwrite"
    ingest_module.ingest_text(source, "第一版内容。", {})
    ingest_module.ingest_text(source, "第二版内容。", {})

    with get_session() as session:
        total = session.query(Document).filter(Document.source == source).count()
    assert total == 1
