"""M5 评测指标与评测集加载测试（纯逻辑，不依赖外部服务）。"""

from pathlib import Path

from app.eval.dataset import QAItem, load_qa
from app.eval.metrics import (
    citation_accuracy,
    hit_at_k,
    keyword_coverage,
    mean,
    recall_at_k,
    reciprocal_rank,
)

QA_PATH = Path(__file__).resolve().parent.parent / "eval" / "qa.json"


def test_recall_at_k_counts_expected_coverage() -> None:
    assert recall_at_k(["a.md", "b.md"], ["a.md", "c.md"], 2) == 0.5
    assert recall_at_k(["b.md", "a.md"], ["a.md"], 2) == 1.0
    assert recall_at_k(["a.md", "b.md"], ["a.md"], 1) == 1.0


def test_hit_at_k() -> None:
    assert hit_at_k(["x.md", "a.md"], ["a.md"], 2) is True
    assert hit_at_k(["x.md", "y.md"], ["a.md"], 2) is False


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a.md"], ["a.md"]) == 1.0
    assert reciprocal_rank(["x.md", "a.md"], ["a.md"]) == 0.5
    assert reciprocal_rank(["x.md", "y.md"], ["a.md"]) == 0.0


def test_keyword_coverage() -> None:
    assert keyword_coverage("可以用索引来优化", ["索引", "缓存"]) == 0.5
    assert keyword_coverage("任意答案", []) == 1.0


def test_citation_accuracy() -> None:
    assert citation_accuracy(["a.md", "b.md"], ["a.md"]) == 0.5
    assert citation_accuracy([], ["a.md"]) == 0.0


def test_mean_handles_empty() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 3.0]) == 2.0


def test_load_qa_dataset() -> None:
    items = load_qa(QA_PATH)
    assert len(items) >= 10
    assert all(isinstance(item, QAItem) and item.question for item in items)
    assert any(item.expected_sources for item in items)
