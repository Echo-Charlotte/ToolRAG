"""评测指标：检索质量与问答质量（纯函数，便于单测）。"""

from collections.abc import Sequence


def recall_at_k(
    retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int
) -> float:
    """Recall@k：前 k 条命中里覆盖了多少「应有来源」。"""
    if not expected_sources:
        return 0.0
    top = set(retrieved_sources[:k])
    hit = top & set(expected_sources)
    return len(hit) / len(set(expected_sources))


def hit_at_k(retrieved_sources: Sequence[str], expected_sources: Sequence[str], k: int) -> bool:
    """Hit@k：前 k 条里是否至少命中一个应有来源。"""
    return bool(set(retrieved_sources[:k]) & set(expected_sources))


def reciprocal_rank(retrieved_sources: Sequence[str], expected_sources: Sequence[str]) -> float:
    """MRR 的单条贡献：第一个命中的排名倒数；未命中为 0。"""
    expected = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    """求均值；空序列返回 0。"""
    return sum(values) / len(values) if values else 0.0


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    """答案准确率（关键词覆盖）：答案里出现的关键词占比。"""
    if not keywords:
        return 1.0
    hit = sum(1 for keyword in keywords if keyword in answer)
    return hit / len(keywords)


def citation_accuracy(cited_sources: Sequence[str], expected_sources: Sequence[str]) -> float:
    """引用正确率：所引用来源中属于「应有来源」的比例；没有引用则为 0。"""
    if not cited_sources:
        return 0.0
    expected = set(expected_sources)
    correct = sum(1 for source in cited_sources if source in expected)
    return correct / len(cited_sources)
