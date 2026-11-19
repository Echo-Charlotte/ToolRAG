# ToolRAG

智能数据查询与文档助手 Agent，支持 **Function Calling 多工具路由** 与 **RAG 混合检索（向量 + 关键词）**，并提供引用溯源、失败兜底与 Token Trace。

## 技术栈

- Python 3.12 / FastAPI / Pydantic v2
- OpenAI Python SDK（OpenAI 兼容接口，Function Calling 多工具路由）
- PostgreSQL + PGVector（向量检索）、Redis（缓存）
- Embedding：硅基流动 `BAAI/bge-m3`（OpenAI 兼容接口，维度 1024）
- 对话模型：硅基流动 `Qwen/Qwen2.5-7B-Instruct`（OpenAI 兼容接口，支持 Function Calling）
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
│   ├── rag/
│       ├── chunking.py    # 文本切分
│       ├── embeddings.py  # Embedding 客户端
│       ├── tokenizer.py   # 中文二元切分（bigram）
│       ├── ingest.py      # 文档入库（含 content_tsv）
│       ├── retriever.py   # 向量 / 关键词 / 混合检索
│       └── fusion.py      # RRF 与加权融合排序
│   ├── agent/             # Function Calling Agent
│   │   ├── llm.py         # 对话模型客户端
│   │   └── tool_agent.py  # 多轮工具调用循环
│   └── tools/             # LLM 可调用的工具
│       ├── registry.py    # 工具注册表（OpenAI tools schema）
│       └── rag_tools.py   # 检索 / 列来源 / 读全文
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
# LLM_API_KEY 留空即可：默认复用 EMBEDDING_API_KEY（同一硅基流动账号）
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

启动时会自动执行 `init_db()`：启用 `vector` 扩展、建表、建 HNSW 向量索引、建 `content_tsv` 的 GIN 索引，并回填历史数据的关键词索引（全部幂等）。
打开 http://localhost:8000/docs 查看接口文档。

## API 说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/documents` | 文档入库：切分 → 向量化 → 写入 PGVector（同时生成 `content_tsv`） |
| POST | `/api/search` | 检索：支持 `vector` / `keyword` / `hybrid` 三种模式 |
| POST | `/api/chat` | 工具调用问答：LLM 自主决定调用工具后再作答 |

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

### 检索

请求体可选字段：

- `mode`：`vector`（余弦相似度）/ `keyword`（ts_rank 关键词）/ `hybrid`（两路融合），默认取配置 `SEARCH_MODE`（默认 `hybrid`）。
- `fusion`：`hybrid` 模式下的融合策略，`rrf`（默认）或 `weighted`。
- `alpha`：`weighted` 融合中向量分的权重（0~1，默认 0.5）。
- `top_k`：返回条数。

```bash
# 混合检索（默认）
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"哪些水果富含钾？","mode":"hybrid","fusion":"rrf","top_k":3}'

# 纯关键词检索
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"香蕉 钾","mode":"keyword","top_k":3}'
```

响应示例（`weighted` 融合会额外返回 `vector_score` / `keyword_score`）：

```json
{
  "query": "哪些水果富含钾？",
  "mode": "hybrid",
  "fusion": "rrf",
  "hits": [
    {
      "id": 2,
      "source": "fruits.md",
      "chunk_index": 0,
      "content": "苹果是一种常见水果。香蕉富含钾。",
      "score": 0.032787,
      "metadata": {"title": "水果"},
      "vector_score": null,
      "keyword_score": null
    }
  ]
}
```

> `score` 的含义随模式变化：`vector` 为余弦相似度（越接近 1 越相关）；`keyword` 为 `ts_rank`；
> `hybrid` 下 `rrf` 为 RRF 累加分（量级很小，仅用于排序），`weighted` 为归一化加权分（0~1）。

### 中文关键词检索原理

PostgreSQL 内置全文检索按空格切词，中文没有空格会被整句当成一个 token。
因此入库前先用 **二元切分（bigram）** 把中文串切成 `数据 据库` 这样的片段，
再经 `to_tsvector('simple', ...)` 建索引（GIN）；查询同理切成 bigram 后用 `|` 拼成 OR 的 tsquery，
避免 AND 造成的零召回。

### 问答（Function Calling，M3）

`/api/chat` 会把三个工具（`search_documents` / `list_sources` / `get_document_content`）的
schema 一并发给对话模型，由模型自主决定调用哪个工具、调用几次，工具结果以 `tool` 角色回填后
再进行下一轮，直到模型给出最终回答或达到 `AGENT_MAX_ROUNDS` 上限。

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"数据库查询变慢了，可以做哪些优化？请给出依据。"}'
```

响应示例：

```json
{
  "answer": "根据文档，可通过建立索引、优化查询语句、使用缓存等方式优化……（来源：db.md）",
  "tool_calls": [
    {
      "name": "search_documents",
      "arguments": {"query": "数据库查询变慢 优化", "top_k": 5, "mode": "hybrid"},
      "result": {"query": "数据库查询变慢 优化", "mode": "hybrid", "results": []}
    }
  ],
  "rounds": 2
}
```

- `tool_calls`：本轮问答的工具调用轨迹（含参数与返回结果），便于观察模型的路由决策。
- `rounds`：实际工具调用轮数；若达到上限仍未给出结论，`answer` 会显式说明。

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
| M2 | 关键词检索 + 混合检索 | ✅ |
| M3 | Function Calling 多工具路由 | ✅ |
| M4 | 引用溯源 / 失败兜底 / Token Trace | ⬜ |
| M5 | 评测与优化 | ⬜ |
