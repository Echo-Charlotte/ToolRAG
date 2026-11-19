"""评测集加载与数据模型。"""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class QAItem(BaseModel):
    """一道评测题。"""

    id: str
    question: str
    expected_sources: list[str] = Field(default_factory=list, description="应有的来源文档")
    keywords: list[str] = Field(default_factory=list, description="答案应包含的关键词")


def load_qa(path: str | Path) -> list[QAItem]:
    """从 JSON 文件加载评测集。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAItem(**item) for item in data]
