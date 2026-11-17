"""API 请求 / 响应模型。"""

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """文档入库请求。"""

    source: str = Field(..., min_length=1, description="文档来源标识，如文件名或 URL")
    text: str = Field(..., description="文档正文")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class IngestResponse(BaseModel):
    """文档入库响应。"""

    source: str
    chunks: int = Field(..., description="写入的切块数量")


class SearchRequest(BaseModel):
    """向量检索请求。"""

    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int | None = Field(default=None, ge=1, le=50, description="返回条数，默认取配置值")


class SearchHit(BaseModel):
    """单条检索命中。"""

    id: int
    source: str
    chunk_index: int
    content: str
    score: float = Field(..., description="余弦相似度，越接近 1 越相关")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """向量检索响应。"""

    query: str
    hits: list[SearchHit]
