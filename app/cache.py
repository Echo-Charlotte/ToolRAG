"""Redis 缓存层：检索结果缓存与失效。

设计要点：

- **惰性连接**：首次使用时才建连接，导入期不依赖 Redis 可用；
- **失败兜底**：Redis 不可用时静默降级（读返回 None、写忽略），不影响主流程；
- **短冷却**：连续失败后短暂跳过，避免每次请求都卡在连接超时上。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis

from app.config import settings

logger = logging.getLogger(__name__)

_FAILURE_COOLDOWN_SECONDS = 30.0

_client: redis.Redis | None = None
_cooldown_until: float = 0.0


def reset_client() -> None:
    """重置连接与冷却状态（测试或配置变更后使用）。"""
    global _client, _cooldown_until
    _client = None
    _cooldown_until = 0.0


def get_redis() -> redis.Redis | None:
    """获取 Redis 客户端；缓存被禁用或处于失败冷却期时返回 None。"""
    global _client
    if not settings.cache_enabled:
        return None
    if time.monotonic() < _cooldown_until:
        return None
    if _client is None:
        _client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=settings.cache_timeout,
            socket_timeout=settings.cache_timeout,
        )
    return _client


def _mark_failure(exc: Exception) -> None:
    global _cooldown_until
    _cooldown_until = time.monotonic() + _FAILURE_COOLDOWN_SECONDS
    logger.warning("Redis 不可用，已临时降级（%.0fs）: %s", _FAILURE_COOLDOWN_SECONDS, exc)


def cache_get_json(key: str) -> Any | None:
    """读取 JSON 缓存；未命中、未启用或 Redis 异常时返回 None。"""
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except redis.RedisError as exc:
        _mark_failure(exc)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set_json(key: str, value: Any, ttl: int | None = None) -> None:
    """写入 JSON 缓存（尽力而为，失败不影响主流程）。"""
    client = get_redis()
    if client is None:
        return
    try:
        payload = json.dumps(value, ensure_ascii=False)
    except TypeError as exc:  # 值不可 JSON 序列化时直接放弃缓存
        logger.warning("缓存值无法序列化，跳过写入: %s", exc)
        return
    try:
        client.set(key, payload, ex=ttl or settings.cache_ttl)
    except redis.RedisError as exc:
        _mark_failure(exc)


def cache_delete_prefix(prefix: str) -> int:
    """按前缀删除缓存键，返回删除数量（数据变更后用于失效）。"""
    client = get_redis()
    if client is None:
        return 0
    deleted = 0
    try:
        for key in client.scan_iter(match=f"{prefix}*", count=500):
            deleted += int(client.delete(key))
    except redis.RedisError as exc:
        _mark_failure(exc)
    return deleted


def search_cache_key(
    query: str, mode: str, top_k: int, fusion: str | None, alpha: float | None
) -> str:
    """构造检索缓存键：包含所有会影响结果的参数。"""
    return f"{settings.cache_key_prefix}:search:{mode}:{top_k}:{fusion}:{alpha}:{query}"


def search_cache_prefix() -> str:
    """检索缓存的公共前缀，用于数据变更后统一失效。"""
    return f"{settings.cache_key_prefix}:search:"
