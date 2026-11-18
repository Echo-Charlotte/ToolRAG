"""HTTP 路由。"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
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
