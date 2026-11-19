FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖先单独安装，利用镜像层缓存：源码改动不会触发重新装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 以非 root 用户运行，降低容器逃逸风险
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# 健康检查：直接打 /api/health，无需额外安装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
