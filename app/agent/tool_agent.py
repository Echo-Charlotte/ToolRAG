"""Function Calling 多工具路由 Agent（含引用溯源、失败兜底与 Token Trace）。

流程：
1. 把用户问题 + 工具 schema 发给 LLM；
2. LLM 若返回 tool_calls，就逐个执行工具并把结果以 tool 角色回填；
3. 重复直到 LLM 不再调用工具（给出最终回答）或达到轮数上限。

在本轮（M4）中额外做到：
- 引用溯源：从工具返回结果里抽取去重后的来源（source + 片段），随回答一并返回；
- 失败兜底：模型调用异常时不抛错，返回可读的兜底回答，并记录错误原因；
- Token Trace：累计各轮对话的 prompt / completion / total tokens。
"""

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.llm import get_chat_client
from app.config import settings
from app.tools.registry import execute_tool, get_tool_specs

SYSTEM_PROMPT = (
    "你是一个智能文档助手 Agent，可以通过调用工具来完成用户请求。\n"
    "可用工具：search_documents（检索文档片段）、list_sources（列出文档来源）、"
    "get_document_content（读取整篇文档）。\n"
    "规则：\n"
    "1. 当问题涉及文档内容时，先调用 search_documents 检索，再基于检索结果回答。\n"
    "2. 回答时请注明信息来源（source）。\n"
    "3. 若检索结果不足以回答，请如实说明，不要编造内容。"
)

# 兜底回答的文案：模型不可用时返回，避免接口直接 5xx
FALLBACK_ANSWER = "抱歉，当前无法连接到模型服务，暂时无法回答该问题。请稍后重试，或检查 LLM 配置。"

# 引用片段的最大长度
_SNIPPET_LIMIT = 120


@dataclass
class TokenUsage:
    """累计的 Token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> None:
        """把一次 LLM 响应的 usage 累加进来（usage 为空则跳过）。"""
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class AgentResult:
    """Agent 一次问答的完整结果。"""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    fallback: bool = False
    error: str | None = None


def _safe_json(raw: str | None) -> dict[str, Any]:
    """把工具调用参数解析成字典，解析失败返回空字典。"""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_assistant_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    """把 LLM 返回的消息整理成可回填 messages 的字典。"""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return payload


def _extract_citations(
    trace: list[dict[str, Any]], answer: str | None = None
) -> list[dict[str, Any]]:
    """从工具调用轨迹里抽取去重后的引用来源。

    支持两类工具结果：
    - search_documents：``results[].{source, chunk_index, content}``
    - get_document_content：``{found, source, chunks[].{chunk_index, content}}``

    Args:
        trace: 工具调用轨迹。
        answer: 最终回答。若提供，则只保留回答中实际提到的来源（更精确的引用溯源）；
            若一个都没匹配到，则退回返回全部已检索来源（说明回答所依据的文档）。

    Returns:
        形如 ``[{"source", "chunk_index", "snippet"}]`` 的列表，按 (source, chunk_index) 去重。
    """
    citations: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    def add(source: Any, chunk_index: Any, content: Any) -> None:
        if source is None or (source, chunk_index) in seen:
            return
        seen.add((source, chunk_index))
        snippet = str(content or "").strip().replace("\n", " ")
        if len(snippet) > _SNIPPET_LIMIT:
            snippet = snippet[:_SNIPPET_LIMIT] + "…"
        citations.append({"source": source, "chunk_index": chunk_index, "snippet": snippet})

    for call in trace:
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        for item in result.get("results") or []:
            if isinstance(item, dict):
                add(item.get("source"), item.get("chunk_index"), item.get("content"))
        if result.get("found") and result.get("source") is not None:
            for chunk in result.get("chunks") or []:
                if isinstance(chunk, dict):
                    add(result["source"], chunk.get("chunk_index"), chunk.get("content"))

    if answer:
        mentioned = [cit for cit in citations if str(cit["source"]) in answer]
        if mentioned:
            return mentioned
    return citations


def run_agent(
    message: str,
    history: list[dict[str, Any]] | None = None,
    client: Any | None = None,
    max_rounds: int | None = None,
) -> AgentResult:
    """执行一轮「工具调用 → 回填 → 再调用」的问答循环。

    Args:
        message: 用户最新问题。
        history: 历史消息（role/content 字典列表），可为空。
        client: 对话模型客户端，默认取全局客户端（便于测试注入假客户端）。
        max_rounds: 最大工具调用轮数，默认取配置 agent_max_rounds。

    Returns:
        AgentResult：最终回答、工具调用轨迹、实际轮数、引用来源、Token 用量。
        模型调用失败时返回兜底回答（``fallback=True``），不抛异常。
    """
    client = client or get_chat_client()
    max_rounds = max_rounds or settings.agent_max_rounds
    tools = get_tool_specs()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": message},
    ]
    trace: list[dict[str, Any]] = []
    usage = TokenUsage()

    for round_index in range(1, max_rounds + 1):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=tools,
                temperature=settings.agent_temperature,
            )
        except Exception as exc:  # 网络/鉴权等异常统一兜底，避免接口 5xx
            return AgentResult(
                answer=FALLBACK_ANSWER,
                tool_calls=trace,
                rounds=round_index,
                citations=_extract_citations(trace, FALLBACK_ANSWER),
                usage=usage,
                fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        usage.add(getattr(response, "usage", None))
        llm_message = response.choices[0].message
        tool_calls = getattr(llm_message, "tool_calls", None)

        if not tool_calls:
            answer = llm_message.content or ""
            return AgentResult(
                answer=answer,
                tool_calls=trace,
                rounds=round_index,
                citations=_extract_citations(trace, answer),
                usage=usage,
            )

        messages.append(_to_assistant_message(llm_message, tool_calls))
        for call in tool_calls:
            name = call.function.name
            arguments = call.function.arguments
            result = execute_tool(name, arguments)
            trace.append({"name": name, "arguments": _safe_json(arguments), "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    return AgentResult(
        answer=f"已达到最大工具调用轮数（{max_rounds}），未能给出最终结论。",
        tool_calls=trace,
        rounds=max_rounds,
        citations=_extract_citations(trace),
        usage=usage,
    )
