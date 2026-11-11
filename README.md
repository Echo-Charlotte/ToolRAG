# ToolRAG

智能数据查询与文档助手 Agent，支持 **Function Calling 多工具路由** 与 **RAG 混合检索（向量 + 关键词）**，并提供引用溯源、失败兜底与 Token Trace。

## 技术栈

- Python 3.12 / FastAPI / Pydantic v2
- LangChain（多工具路由 + 检索链）
- PostgreSQL + PGVector（向量检索）、Redis（缓存）
- Docker Compose 管理本地依赖服务

## 目录结构

```
ToolRAG/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── config.py        # 配置（pydantic-settings）
│   └── api/
│       └── routes.py    # 路由
├── tests/               # 测试
├── .github/workflows/   # CI
├── docker-compose.yml   # PGVector + Redis
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等
```

### 2. 启动依赖服务（PGVector + Redis）

```bash
docker compose up -d
```

### 3. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 4. 运行服务

```bash
uvicorn app.main:app --reload
```

打开 http://localhost:8000/docs 查看接口文档，健康检查为 `GET /api/health`。

## 代码检查与测试

```bash
ruff check .
pytest -q
```

## 里程碑

| 阶段 | 内容 |
| --- | --- |
| M0 | 项目初始化：骨架、Docker 环境、配置 |
| M1 | 向量检索实现 |
| M2 | 关键词检索 + 混合检索 |
| M3 | Function Calling 多工具路由 |
| M4 | 引用溯源 / 失败兜底 / Token Trace |
| M5 | 评测与优化 |
