"""Embedding 客户端：调用 OpenAI 兼容接口生成文本向量（默认硅基流动 BAAI/bge-m3）。"""

from functools import lru_cache

from openai import OpenAI

from app.config import settings


class EmbeddingClient:
    """封装 embedding 接口，提供批量与单条向量化。"""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，内部按 embedding_batch_size 分批请求。"""
        if not texts:
            return []

        vectors: list[list[float]] = []
        batch_size = max(1, settings.embedding_batch_size)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )
            # 按 index 排序，保证与输入顺序一致
            ordered = sorted(resp.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化（检索时使用）。"""
        return self.embed_documents([text])[0]


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()
