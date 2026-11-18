"""中文分词：二元切分（bigram）+ 英文/数字整词保留。

为什么需要它：
PostgreSQL 内置全文检索按空格切词，中文没有空格，整句会被当成**一个 token**，
导致搜「水果」无法命中「苹果是一种常见水果」。
这里把连续中文串切成二元组（苹果是一种常见水果 → 苹果/果是/是一/一种/种常/常见/见水/水果），
再交给 to_tsvector('simple', ...) 建索引。

为什么用 'simple' 配置：
'simple' 只做小写化、不做词干还原（stemming），能原样保留我们的二元 token；
'english' 等配置会把 token 再做词干化，反而破坏中文二元组。
"""

import re

# 连续的中日韩统一表意文字
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
# 英文单词 / 数字 / 下划线，作为整体保留
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """把文本切成 token 列表。

    英文/数字整词保留（如 `ORD-2026` → `ord`、`2026`），中文按二元组切分；
    单字中文（如 `钾`）原样保留。
    """
    tokens = [match.group().lower() for match in _ASCII_WORD.finditer(text)]
    for match in _CJK_RUN.finditer(text):
        run = match.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def to_index_text(text: str) -> str:
    """转成建索引用文本：token 之间以空格分隔，供 to_tsvector 使用。"""
    return " ".join(tokenize(text))


def to_or_tsquery_text(text: str) -> str:
    """转成 tsquery 表达式：token 之间用 `|`（OR）连接。

    这里刻意用 OR 而不是 AND：二元切分会产生「果含」这类跨词组合，
    若要求全部命中，绝大多数正常查询都会零结果。
    OR 匹配后用 ts_rank 排序，命中的 token 越多得分越高。
    """
    return " | ".join(tokenize(text))
