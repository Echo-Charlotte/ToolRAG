# ToolRAG

智能数据查询与文档助手 Agent，支持 **Function Calling 多工具路由** 与 **RAG 混合检索（向量 + 关键词）**，并提供引用溯源、失败兜底与 Token Trace。

## 技术栈

- Python 3.12 / FastAPI / Pydantic v2
- LangChain（多工具路由 + 检索链）
- PostgreSQL + PGVector（向量检索）、Redis（缓存）
- Embedding：硅基流动 `BAAI/bge-m3`（OpenAI 兼容接口，维度 1024）
- Docker Compose 管理本地依赖服务

## 目录结构

```
ToolRAG/
├── app/
│   ├── main.py            # FastAPI 入口（含 lifespan 初始化数据库）
│   ├── config.py          # 配置（pydantic-settings）
│   ├── db.py              # SQLAlchemy Engine / Session
│   ├── models.py          # ORM 模型（documents 表 + vector 列）
│   ├── schema.py          # 建表 / 建索引 / 启用 vector 扩展
│   ├── api/
│   │   ├── routes.py      # 路由
│   │   └── schemas.py     # 请求 / 响应模型
│   └── rag/
│       ├── chunking.py    # 文本切分
│       ├── embeddings.py  # Embedding 客户端
│       ├── ingest.py      # 文档入库
│       └── retriever.py   # 向量检索
├── tests/                 # 测试
├── .github/workflows/     # CI
├── docker-compose.yml     # PGVector + Redis
└── .env.example           # 环境变量模板
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 EMBEDDING_API_KEY（硅基流动控制台获取）
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

启动时会自动执行 `init_db()`：启用 `vector` 扩展、建表、建 HNSW 向量索引（幂等）。
打开 http://localhost:8000/docs 查看接口文档。

## API 说明（M1）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/documents` | 文档入库：切分 → 向量化 → 写入 PGVector |
| POST | `/api/search` | 向量相似度检索（余弦） |

### 文档入库

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"source":"fruits.md","text":"苹果是一种常见水果。香蕉富含钾。","metadata":{"title":"水果"}}'
```

响应示例：

```json
{"source": "fruits.md", "chunks": 1}
```

同一 `source` 重复入库会覆盖旧数据。

### 向量检索

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"哪些水果富含钾？","top_k":3}'
```

响应示例：

```json
{
  "query": "哪些水果富含钾？",
  "hits": [
    {
      "id": 2,
      "source": "fruits.md",
      "chunk_index": 0,
      "content": "苹果是一种常见水果。香蕉富含钾。",
      "score": 0.7321,
      "metadata": {"title": "水果"}
    }
  ]
}
```

`score` 为余弦相似度（`1 - 余弦距离`），越接近 1 越相关。

## 代码检查与测试

```bash
ruff check .
pytest -q
```

`tests/test_vector_store.py` 需要可用的 PGVector；若本地未启动数据库会自动跳过。

## 里程碑

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 项目初始化：骨架、Docker 环境、配置 | ✅ |
| M1 | 向量检索：切分、Embedding、PGVector 存储与检索 | ✅ |
| M2 | 关键词检索 + 混合检索 | ⬜ |
| M3 | Function Calling 多工具路由 | ⬜ |
| M4 | 引用溯源 / 失败兜底 / Token Trace | ⬜ |
| M5 | 评测与优化 | ⬜ |
