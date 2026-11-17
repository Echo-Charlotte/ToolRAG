"""API 接口测试：用假向量替换真实 embedding，避免依赖外部服务与数据库。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_search_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_hits = [
        {
            "id": 1,
            "source": "doc.md",
            "chunk_index": 0,
            "content": "苹果是一种水果。",
            "score": 0.87,
            "metadata": {"title": "水果"},
        }
    ]
    monkeypatch.setattr("app.api.routes.search_similar", lambda query, top_k: fake_hits)

    resp = client.post("/api/search", json={"query": "苹果", "top_k": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "苹果"
    assert body["hits"][0]["source"] == "doc.md"
    assert body["hits"][0]["score"] == 0.87


def test_search_rejects_empty_query() -> None:
    resp = client.post("/api/search", json={"query": "", "top_k": 1})
    assert resp.status_code == 422


def test_documents_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.routes.ingest_text", lambda source, text, metadata: 3)

    resp = client.post(
        "/api/documents",
        json={"source": "a.txt", "text": "一些内容", "metadata": {}},
    )

    assert resp.status_code == 200
    assert resp.json() == {"source": "a.txt", "chunks": 3}
