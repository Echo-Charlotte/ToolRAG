from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.schema import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化数据库结构（幂等）。"""
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.get("/", summary="根路径")
async def root() -> dict[str, str]:
    return {"app": settings.app_name, "env": settings.app_env}
