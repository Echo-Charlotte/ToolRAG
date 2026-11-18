"""混合检索的融合排序：RRF 与加权求和。

两路检索的分数**量纲完全不同**：
- 向量检索：余弦相似度，范围约 [0, 1]，差距小（0.6 与 0.7 已是明显差距）；
- 关键词检索：ts_rank，范围不固定（可能 0.06，也可能 5.0），取决于命中词频。

直接相加会被量纲大的那一路主导，所以要靠 RRF（只看排名）或先归一化再加权。
"""

from typing import Any

Hit = dict[str, Any]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]], k: int = 60, top_k: int | None = None
) -> list[Hit]:
    """RRF（Reciprocal Rank Fusion）：`score = Σ 1 / (k + rank)`。

    优点：只依赖排名，不需要归一化，天然免疫两路分数量纲差异。
    `k` 用于削弱头部排名的绝对优势（k 越大，各名次之间差距越平缓）。

    Args:
        ranked_lists: 多个已按相关度降序排好的结果列表。
        k: 平滑常数，常用 60。
        top_k: 返回条数，None 表示全部。

    Returns:
        融合后的结果列表，score 为 RRF 累加分（量级很小，仅用于排序）。
    """
    merged: dict[Any, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit["id"]
            entry = merged.setdefault(key, {**hit, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)

    results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return results[:top_k] if top_k is not None else results


def _min_max_normalize(hits: list[Hit]) -> dict[Any, float]:
    """把一组分数 min-max 归一化到 [0, 1]；只有一个元素时记为 1.0。"""
    if not hits:
        return {}
    scores = [hit["score"] for hit in hits]
    low, high = min(scores), max(scores)
    if high == low:  # 全部相同，无法区分，统一给 1.0
        return {hit["id"]: 1.0 for hit in hits}
    span = high - low
    return {hit["id"]: (hit["score"] - low) / span for hit in hits}


def weighted_fusion(
    vector_hits: list[Hit],
    keyword_hits: list[Hit],
    alpha: float = 0.5,
    top_k: int | None = None,
) -> list[Hit]:
    """加权求和：`score = alpha * 归一化向量分 + (1 - alpha) * 归一化关键词分`。

    只出现在一路结果中的文档，另一路按 0 计。

    Args:
        vector_hits: 向量检索结果（按分数降序）。
        keyword_hits: 关键词检索结果（按分数降序）。
        alpha: 向量分权重，0 表示完全用关键词，1 表示完全用向量。
        top_k: 返回条数，None 表示全部。

    Returns:
        融合后的结果列表，同时保留 vector_score / keyword_score 便于对比。
    """
    vector_normalized = _min_max_normalize(vector_hits)
    keyword_normalized = _min_max_normalize(keyword_hits)

    merged: dict[Any, Hit] = {}
    for hit in vector_hits:
        merged[hit["id"]] = {**hit, "vector_score": hit["score"], "keyword_score": None}
    for hit in keyword_hits:
        entry = merged.setdefault(
            hit["id"], {**hit, "vector_score": None, "keyword_score": None}
        )
        entry["keyword_score"] = hit["score"]

    results: list[Hit] = []
    for doc_id, entry in merged.items():
        score = alpha * vector_normalized.get(doc_id, 0.0) + (1 - alpha) * keyword_normalized.get(
            doc_id, 0.0
        )
        results.append({**entry, "score": round(score, 6)})

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k] if top_k is not None else results
