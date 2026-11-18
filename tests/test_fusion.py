"""融合排序（RRF / 加权求和）单元测试。"""

from typing import Any

from app.rag.fusion import reciprocal_rank_fusion, weighted_fusion


def _hit(doc_id: int, score: float) -> dict[str, Any]:
    return {"id": doc_id, "score": score, "content": f"content-{doc_id}"}


def test_rrf_rewards_documents_in_both_lists() -> None:
    vector_hits = [_hit(1, 0.9), _hit(2, 0.8)]
    keyword_hits = [_hit(2, 5.0), _hit(3, 3.0)]

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=60)

    # 文档 2 两路都命中且名次靠前，RRF 累加分最高
    assert fused[0]["id"] == 2
    assert {hit["id"] for hit in fused} == {1, 2, 3}


def test_rrf_top_k_truncates() -> None:
    fused = reciprocal_rank_fusion([[_hit(1, 1.0), _hit(2, 0.5)]], top_k=1)
    assert len(fused) == 1
    assert fused[0]["id"] == 1


def test_rrf_score_formula() -> None:
    # 单路两条：rank 1 → 1/(60+1)，rank 2 → 1/(60+2)
    fused = reciprocal_rank_fusion([[_hit(1, 1.0), _hit(2, 0.5)]], k=60)
    assert fused[0]["score"] == 1 / 61
    assert fused[1]["score"] == 1 / 62


def test_weighted_fusion_alpha_one_prefers_vector() -> None:
    vector_hits = [_hit(1, 0.9), _hit(2, 0.1)]
    keyword_hits = [_hit(3, 5.0)]

    fused = weighted_fusion(vector_hits, keyword_hits, alpha=1.0)

    assert fused[0]["id"] == 1
    doc3 = next(hit for hit in fused if hit["id"] == 3)
    assert doc3["vector_score"] is None  # 该文档只在关键词结果里
    assert doc3["keyword_score"] == 5.0


def test_weighted_fusion_alpha_zero_prefers_keyword() -> None:
    vector_hits = [_hit(1, 0.9)]
    keyword_hits = [_hit(2, 5.0), _hit(3, 1.0)]

    fused = weighted_fusion(vector_hits, keyword_hits, alpha=0.0)

    assert fused[0]["id"] == 2
    doc1 = next(hit for hit in fused if hit["id"] == 1)
    assert doc1["score"] == 0.0
