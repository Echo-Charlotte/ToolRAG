"""中文分词工具单元测试。"""

from app.rag.tokenizer import to_index_text, to_or_tsquery_text, tokenize


def test_tokenize_splits_chinese_into_bigrams() -> None:
    assert tokenize("苹果香蕉") == ["苹果", "果香", "香蕉"]


def test_tokenize_single_chinese_char_kept_as_is() -> None:
    assert tokenize("好") == ["好"]


def test_tokenize_keeps_ascii_words_lowercased() -> None:
    assert tokenize("Hello PGVector") == ["hello", "pgvector"]


def test_tokenize_mixes_ascii_and_chinese() -> None:
    # 实现里先收集 ASCII 词，再追加中文二元切片，因此顺序是 ascii 在前
    assert tokenize("使用 PGVector 检索") == ["pgvector", "使用", "检索"]


def test_tokenize_drops_punctuation() -> None:
    assert tokenize("，。！？") == []


def test_to_index_text_joins_with_space() -> None:
    assert to_index_text("苹果香蕉") == "苹果 果香 香蕉"


def test_to_or_tsquery_text_joins_with_pipe() -> None:
    assert to_or_tsquery_text("苹果香蕉") == "苹果 | 果香 | 香蕉"
