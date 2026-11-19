from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，统一从环境变量 / .env 读取。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    app_name: str = "ToolRAG"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # LLM（对话模型，M3 使用；默认硅基流动，OpenAI 兼容接口，支持 Function Calling）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Embedding（默认硅基流动，OpenAI 兼容接口）
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # RAG 检索参数
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # 混合检索（M2）
    search_mode: str = "hybrid"  # vector | keyword | hybrid
    fusion_strategy: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    hybrid_alpha: float = 0.5  # 加权融合中向量分的权重
    candidate_multiplier: int = 4  # 混合检索时各路召回的候选池倍数

    # Agent（M3）
    agent_max_rounds: int = 5  # 工具调用最大轮数
    agent_temperature: float = 0.0

    # PostgreSQL + PGVector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "toolrag"
    postgres_password: str = "toolrag"
    postgres_db: str = "toolrag"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # 缓存（M6）
    cache_enabled: bool = True  # 关闭后所有读写直接跳过 Redis
    cache_ttl: int = 300  # 检索结果缓存有效期（秒）
    cache_timeout: float = 0.5  # Redis 连接 / 读写超时（秒），避免阻塞主流程
    cache_key_prefix: str = "toolrag"

    @property
    def effective_llm_api_key(self) -> str:
        """对话模型 API Key；未单独配置时复用 embedding 的 Key（同一硅基流动账号）。"""
        return self.llm_api_key or self.embedding_api_key

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
