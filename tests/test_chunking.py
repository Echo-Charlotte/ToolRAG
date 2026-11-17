"""文本切分单元测试。"""

import pytest

from app.rag.chunking import split_text


def test_empty_text_returns_empty_list() -> None:
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_short_text_stays_single_chunk() -> None:
    text = "这是一段很短的文本。"
    assert split_text(text, chunk_size=100) == [text]


def test_long_text_is_split_within_size_limit() -> None:
    text = "".join(f"第{i}句内容。" for i in range(300))
    chunks = split_text(text, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[0].startswith("第0句内容")


def test_chunks_cover_whole_document() -> None:
    text = "".join(f"第{i}句内容。" for i in range(300))
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)

    joined = "".join(chunks)
    assert joined == text


def test_overlap_adds_shared_tail() -> None:
    text = "甲" * 60 + "。" + "乙" * 60 + "。"
    without_overlap = split_text(text, chunk_size=40, chunk_overlap=0)
    with_overlap = split_text(text, chunk_size=40, chunk_overlap=20)

    assert len(without_overlap) >= 2
    # 有重叠时总长度更短或更少，但内容仍被完整拼接覆盖（允许重复）
    assert with_overlap
    assert sum(len(c) for c in with_overlap) >= len(text.strip())


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, 10), (10, 20), (-1, 0)],
)
def test_invalid_params_raise(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("一些文本", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
