"""文本切分：把长文档切成带重叠的 chunk。

切分策略（面向中文）：
1. 先在句末标点（。！？；等）处尽量断开，避免把一句话劈开；
2. 找不到标点时退化为按固定长度硬切；
3. 相邻 chunk 之间保留 chunk_overlap 个字符的重叠，避免上下文被切断。
"""

import re

# 句末标点，按优先级从高到低尝试
_SENTENCE_ENDINGS = ("。", "！", "？", "；", "\n", ".", "!", "?", ";")


def _prettify(text: str) -> str:
    """归一化空白字符，压缩连续空格与多余空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """把 text 切成若干长度不超过 chunk_size 的片段。

    Args:
        text: 原始文本。
        chunk_size: 每个 chunk 的最大字符数。
        chunk_overlap: 相邻 chunk 的重叠字符数，需小于 chunk_size。

    Returns:
        chunk 列表；若原文为空则返回空列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")

    text = _prettify(text)
    if not text:
        return []

    length = len(text)
    chunks: list[str] = []
    start = 0

    while start < length:
        end = min(start + chunk_size, length)

        # 未到结尾时，尝试回退到最近的句末标点，保证语义完整
        if end < length:
            window = text[start:end]
            for sep in _SENTENCE_ENDINGS:
                pos = window.rfind(sep)
                if pos >= chunk_size // 2:
                    end = start + pos + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        # 下一段起点回退 overlap 个字符；至少前进 1，避免死循环
        start = max(end - chunk_overlap, start + 1)

    return chunks
