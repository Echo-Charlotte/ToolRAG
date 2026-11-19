"""HTTP 路由。"""

from fastapi import APIRouter, HTTPException

from app.agent.tool_agent import run_agent
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TokenUsage,
    ToolCallTrace,
)
from app.config import settings
from app.rag.ingest import ingest_text
from app.rag.retriever import search as run_search

router = APIRouter()


@router.get("/health", summary="健康检查")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents", response_model=IngestResponse, summary="文档入库（切分 + 向量化）")
def create_document(req: IngestRequest) -> IngestResponse:
    """把文档切块、向量化后写入 PGVector。同一 source 覆盖式更新。"""
    try:
        count = ingest_text(req.source, req.text, req.metadata)
    except Exception as exc:  # 向量化/数据库异常统一转为 502，便于定位
        raise HTTPException(status_code=502, detail=f"文档入库失败: {exc}") from exc
    if count == 0:
        raise HTTPException(status_code=400, detail="文档内容为空，无法入库")
    return IngestResponse(source=req.source, chunks=count)


@router.post("/search", response_model=SearchResponse, summary="检索（向量 / 关键词 / 混合）")
def search(req: SearchRequest) -> SearchResponse:
    """按 mode 选择检索方式，默认走配置 search_mode（默认 hybrid）。"""
    mode = req.mode or settings.search_mode
    try:
        raw_hits = run_search(req.query, req.top_k, mode, req.fusion, req.alpha)
    except ValueError as exc:  # 非法 mode
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"检索失败: {exc}") from exc
    fusion = (req.fusion or settings.fusion_strategy) if mode == "hybrid" else None
    return SearchResponse(
        query=req.query,
        mode=mode,
        fusion=fusion,
        hits=[SearchHit(**hit) for hit in raw_hits],
    )


@router.post("/chat", response_model=ChatResponse, summary="工具调用问答（Function Calling）")
def chat(req: ChatRequest) -> ChatResponse:
    """让 LLM 自主决定是否调用工具（检索 / 列来源 / 读全文）并作答。

    返回回答、工具调用轨迹、引用来源与 Token 用量；模型不可用时返回兜底回答（不 5xx）。
    """
    history = [msg.model_dump() for msg in req.history]
    try:
        result = run_agent(req.message, history)
    except Exception as exc:  # 最后一道防线：非模型调用异常（如内部 bug）仍转为 502
        raise HTTPException(status_code=502, detail=f"问答失败: {exc}") from exc
    return ChatResponse(
        answer=result.answer,
        tool_calls=[ToolCallTrace(**call) for call in result.tool_calls],
        citations=[Citation(**citation) for citation in result.citations],
        usage=TokenUsage(**result.usage.as_dict()),
        rounds=result.rounds,
        fallback=result.fallback,
        error=result.error,
    )
