"""检索：向量检索、关键词检索与混合检索。"""

from typing import Any

from sqlalchemy import func, select

from app.config import settings
from app.db import get_session
from app.models import Document
from app.rag.embeddings import get_embedding_client
from app.rag.fusion import reciprocal_rank_fusion, weighted_fusion
from app.rag.tokenizer import to_or_tsquery_text


def search_similar(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """向量相似度检索。

    Args:
        query: 用户查询文本。
        top_k: 返回条数，默认取配置 retrieval_top_k。

    Returns:
        命中列表，每项含 id/source/chunk_index/content/score/metadata。
        score 为余弦相似度（1 - 余弦距离），越接近 1 越相关。
    """
    top_k = top_k or settings.retrieval_top_k
    query_vector = get_embedding_client().embed_query(query)

    distance = Document.embedding.cosine_distance(query_vector).label("distance")
    stmt = select(Document, distance).order_by(distance).limit(top_k)

    hits: list[dict[str, Any]] = []
    with get_session() as session:
        for doc, dist in session.execute(stmt).all():
            hits.append(
                {
                    "id": doc.id,
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "content": doc.content,
                    "score": round(1.0 - float(dist), 6),
                    "metadata": doc.doc_metadata or {},
                }
            )
    return hits


def search_keyword(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """关键词全文检索（基于 content_tsv + ts_rank）。

    查询先做中文二元切分，再用 `|` 拼成 OR 形式的 tsquery：
    只要命中任一切片就算候选，最后按 ts_rank 排序。

    Args:
        query: 用户查询文本。
        top_k: 返回条数，默认取配置 retrieval_top_k。

    Returns:
        命中列表，结构同 search_similar，score 为 ts_rank（量级不固定，仅用于内部排序）。
    """
    top_k = top_k or settings.retrieval_top_k
    query_text = to_or_tsquery_text(query)
    if not query_text:  # 查询里没有任何可检索的字符（如纯标点），直接返回空
        return []

    tsquery = func.to_tsquery("simple", query_text)
    rank = func.ts_rank(Document.content_tsv, tsquery).label("rank")
    stmt = (
        select(Document, rank)
        .where(Document.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )

    hits: list[dict[str, Any]] = []
    with get_session() as session:
        for doc, score in session.execute(stmt).all():
            hits.append(
                {
                    "id": doc.id,
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "content": doc.content,
                    "score": round(float(score), 6),
                    "metadata": doc.doc_metadata or {},
                }
            )
    return hits


def search_hybrid(
    query: str,
    top_k: int | None = None,
    fusion: str | None = None,
    alpha: float | None = None,
) -> list[dict[str, Any]]:
    """混合检索：向量 + 关键词两路召回，再融合排序。

    两路各自多召回一些候选（top_k * candidate_multiplier），
    交给融合算法统一排序后再截取 top_k，避免单路漏召回。

    Args:
        query: 用户查询文本。
        top_k: 最终返回条数，默认取配置 retrieval_top_k。
        fusion: 融合策略，rrf | weighted，默认取配置 fusion_strategy。
        alpha: 加权融合中向量分权重，默认取配置 hybrid_alpha。

    Returns:
        融合后的命中列表；weighted 策略会额外带 vector_score / keyword_score。
    """
    top_k = top_k or settings.retrieval_top_k
    fusion = fusion or settings.fusion_strategy
    alpha = settings.hybrid_alpha if alpha is None else alpha
    pool = top_k * settings.candidate_multiplier

    vector_hits = search_similar(query, pool)
    keyword_hits = search_keyword(query, pool)

    if fusion == "weighted":
        return weighted_fusion(vector_hits, keyword_hits, alpha=alpha, top_k=top_k)
    # 默认 RRF
    return reciprocal_rank_fusion([vector_hits, keyword_hits], k=settings.rrf_k, top_k=top_k)


def search(
    query: str,
    top_k: int | None = None,
    mode: str | None = None,
    fusion: str | None = None,
    alpha: float | None = None,
) -> list[dict[str, Any]]:
    """检索调度入口：按 mode 选择向量 / 关键词 / 混合检索。

    Args:
        query: 用户查询文本。
        top_k: 返回条数。
        mode: vector | keyword | hybrid，默认取配置 search_mode。
        fusion: 混合检索的融合策略。
        alpha: 加权融合的向量分权重。

    Raises:
        ValueError: mode 非法。
    """
    mode = mode or settings.search_mode
    if mode == "vector":
        return search_similar(query, top_k)
    if mode == "keyword":
        return search_keyword(query, top_k)
    if mode == "hybrid":
        return search_hybrid(query, top_k, fusion, alpha)
    raise ValueError(f"未知检索模式: {mode}（可选 vector / keyword / hybrid）")
