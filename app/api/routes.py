"""HTTP 路由。"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.rag.ingest import ingest_text
from app.rag.retriever import search_similar

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


@router.post("/search", response_model=SearchResponse, summary="向量相似度检索")
def search(req: SearchRequest) -> SearchResponse:
    """按余弦相似度检索最相关的 chunk。"""
    try:
        raw_hits = search_similar(req.query, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"检索失败: {exc}") from exc
    return SearchResponse(
        query=req.query,
        hits=[SearchHit(**hit) for hit in raw_hits],
    )
