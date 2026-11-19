"""对话模型客户端（OpenAI 兼容接口）。"""

from functools import lru_cache

from openai import OpenAI

from app.config import settings


@lru_cache
def get_chat_client() -> OpenAI:
    return OpenAI(api_key=settings.effective_llm_api_key, base_url=settings.llm_base_url)
