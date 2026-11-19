"""Redis 缓存层与检索缓存测试：用假 Redis 客户端，不依赖真实服务。"""

from typing import Any

import pytest
import redis

from app import cache as cache_module
from app.config import settings
from app.rag import retriever as retriever_module


class _FakeRedis:
    """最小可用的 Redis 替身，只实现缓存层用到的命令。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value
        return True

    def scan_iter(self, match: str | None = None, count: int | None = None) -> list[str]:
        prefix = (match or "").rstrip("*")
        return [key for key in list(self.store) if key.startswith(prefix)]

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(cache_module, "get_redis", lambda: fake)
    return fake


def test_cache_set_get_roundtrip(fake_redis: _FakeRedis) -> None:
    cache_module.cache_set_json("key", [{"a": 1}], ttl=60)
    assert cache_module.cache_get_json("key") == [{"a": 1}]


def test_cache_get_missing_returns_none(fake_redis: _FakeRedis) -> None:
    assert cache_module.cache_get_json("missing") is None


def test_cache_delete_prefix_only_removes_matching(fake_redis: _FakeRedis) -> None:
    cache_module.cache_set_json("toolrag:search:a", 1)
    cache_module.cache_set_json("toolrag:search:b", 2)
    cache_module.cache_set_json("toolrag:other", 3)

    assert cache_module.cache_delete_prefix("toolrag:search:") == 2
    assert cache_module.cache_get_json("toolrag:search:a") is None
    assert cache_module.cache_get_json("toolrag:other") == 3


def test_cache_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cache_enabled", False)
    cache_module.reset_client()
    try:
        assert cache_module.get_redis() is None
        assert cache_module.cache_get_json("key") is None
    finally:
        cache_module.reset_client()


def test_cache_degrades_on_redis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def get(self, key: str) -> Any:
            raise redis.RedisError("boom")

        def set(self, *args: Any, **kwargs: Any) -> None:
            raise redis.RedisError("boom")

        def scan_iter(self, *args: Any, **kwargs: Any) -> Any:
            raise redis.RedisError("boom")

    monkeypatch.setattr(cache_module, "get_redis", lambda: _Boom())

    # Redis 报错时读返回 None、写与删除不抛异常（静默降级）
    assert cache_module.cache_get_json("key") is None
    cache_module.cache_set_json("key", {"a": 1})
    assert cache_module.cache_delete_prefix("key") == 0


def test_search_cached_hits_cache_on_second_call(
    monkeypatch: pytest.MonkeyPatch, fake_redis: _FakeRedis
) -> None:
    calls: list[str] = []

    def fake_search(
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        fusion: str | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(query)
        return [
            {
                "id": 1,
                "source": "d.md",
                "chunk_index": 0,
                "content": "内容",
                "score": 1.0,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(retriever_module, "search", fake_search)

    hits1, cached1 = retriever_module.search_cached("缓存测试", top_k=3, mode="hybrid")
    hits2, cached2 = retriever_module.search_cached("缓存测试", top_k=3, mode="hybrid")

    assert cached1 is False
    assert cached2 is True
    assert hits1 == hits2
    assert calls == ["缓存测试"]  # 第二次直接命中缓存，未再检索


def test_search_cached_key_differs_by_params(
    monkeypatch: pytest.MonkeyPatch, fake_redis: _FakeRedis
) -> None:
    calls: list[str] = []

    def fake_search(
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        fusion: str | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(query)
        return []

    monkeypatch.setattr(retriever_module, "search", fake_search)

    retriever_module.search_cached("q", top_k=3, mode="vector")
    retriever_module.search_cached("q", top_k=5, mode="vector")

    assert len(calls) == 2  # top_k 不同 → 缓存键不同 → 各检索一次


def test_search_cached_rejects_invalid_mode(fake_redis: _FakeRedis) -> None:
    with pytest.raises(ValueError):
        retriever_module.search_cached("q", mode="bogus")


def test_search_cache_key_includes_all_params() -> None:
    key = cache_module.search_cache_key("查询", "hybrid", 5, "rrf", 0.5)
    assert "hybrid" in key
    assert "5" in key
    assert "rrf" in key
    assert "查询" in key
