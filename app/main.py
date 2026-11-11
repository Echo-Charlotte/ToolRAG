from fastapi import FastAPI

from app.api.routes import router
from app.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(router, prefix="/api")


@app.get("/", summary="根路径")
async def root() -> dict[str, str]:
    return {"app": settings.app_name, "env": settings.app_env}
