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


class ChatMessage(BaseModel):
    """一条历史消息。"""

    role: str = Field(..., description="角色：user / assistant")
    content: str


class ChatRequest(BaseModel):
    """问答请求。"""

    message: str = Field(..., min_length=1, description="用户问题")
    history: list[ChatMessage] = Field(default_factory=list, description="历史对话，可为空")


class ToolCallTrace(BaseModel):
    """一次工具调用的轨迹。"""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = Field(default=None, description="工具返回结果")


class Citation(BaseModel):
    """引用来源（回答所依据的文档片段）。"""

    source: str = Field(..., description="文档来源标识")
    chunk_index: int | None = Field(default=None, description="命中的切块序号")
    snippet: str = Field(default="", description="片段摘要")


class TokenUsage(BaseModel):
    """对话累计 Token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """问答响应。"""

    answer: str
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list, description="去重后的引用来源")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="累计 Token 用量")
    rounds: int = Field(..., description="实际工具调用轮数")
    fallback: bool = Field(default=False, description="是否为失败兜底回答")
    error: str | None = Field(default=None, description="兜底时的错误原因")
