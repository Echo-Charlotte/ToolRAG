"""API 接口测试：用假向量替换真实 embedding，避免依赖外部服务与数据库。"""

from types import SimpleNamespace

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
    monkeypatch.setattr(
        "app.api.routes.run_search", lambda query, top_k, mode, fusion, alpha: fake_hits
    )

    resp = client.post("/api/search", json={"query": "苹果", "top_k": 1, "mode": "vector"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "苹果"
    assert body["mode"] == "vector"
    assert body["hits"][0]["source"] == "doc.md"
    assert body["hits"][0]["score"] == 0.87


def test_search_hybrid_returns_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_hits = [
        {
            "id": 1,
            "source": "doc.md",
            "chunk_index": 0,
            "content": "苹果是一种水果。",
            "score": 0.5,
            "metadata": {},
            "vector_score": 0.8,
            "keyword_score": 0.2,
        }
    ]
    monkeypatch.setattr(
        "app.api.routes.run_search", lambda query, top_k, mode, fusion, alpha: fake_hits
    )

    resp = client.post(
        "/api/search", json={"query": "苹果", "mode": "hybrid", "fusion": "weighted"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "hybrid"
    assert body["fusion"] == "weighted"
    assert body["hits"][0]["vector_score"] == 0.8
    assert body["hits"][0]["keyword_score"] == 0.2


def test_search_rejects_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(query: str, top_k: int | None, mode: str, fusion: str | None, alpha: float | None):
        raise ValueError(f"未知检索模式: {mode}")

    monkeypatch.setattr("app.api.routes.run_search", _raise)

    resp = client.post("/api/search", json={"query": "苹果", "mode": "bogus"})

    assert resp.status_code == 400


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


def test_chat_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = SimpleNamespace(
        answer="苹果是一种水果。",
        tool_calls=[
            {
                "name": "search_documents",
                "arguments": {"query": "苹果"},
                "result": {"results": []},
            }
        ],
        rounds=2,
    )
    monkeypatch.setattr("app.api.routes.run_agent", lambda message, history: fake_result)

    resp = client.post("/api/chat", json={"message": "苹果是什么？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "苹果是一种水果。"
    assert body["rounds"] == 2
    assert body["tool_calls"][0]["name"] == "search_documents"


def test_chat_rejects_empty_message() -> None:
    resp = client.post("/api/chat", json={"message": ""})
    assert resp.status_code == 422
