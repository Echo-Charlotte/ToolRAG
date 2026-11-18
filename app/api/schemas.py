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
    """检索请求。"""

    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int | None = Field(default=None, ge=1, le=50, description="返回条数，默认取配置值")
    mode: str | None = Field(
        default=None, description="检索模式：vector | keyword | hybrid，默认取配置"
    )
    fusion: str | None = Field(
        default=None, description="混合检索的融合策略：rrf | weighted，默认取配置"
    )
    alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="加权融合中向量分的权重，默认取配置"
    )


class SearchHit(BaseModel):
    """单条检索命中。"""

    id: int
    source: str
    chunk_index: int
    content: str
    score: float = Field(..., description="主排序分：向量=余弦相似度，关键词=ts_rank，混合=融合分")
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float | None = Field(default=None, description="加权融合时的原始向量分")
    keyword_score: float | None = Field(default=None, description="加权融合时的原始关键词分")


class SearchResponse(BaseModel):
    """检索响应。"""

    query: str
    mode: str = Field(..., description="实际使用的检索模式")
    fusion: str | None = Field(default=None, description="混合检索实际使用的融合策略")
    hits: list[SearchHit]
