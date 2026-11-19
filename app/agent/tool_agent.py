"""Function Calling 多工具路由 Agent。

流程：
1. 把用户问题 + 工具 schema 发给 LLM；
2. LLM 若返回 tool_calls，就逐个执行工具并把结果以 tool 角色回填；
3. 重复直到 LLM 不再调用工具（给出最终回答）或达到轮数上限。
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


@dataclass
class AgentResult:
    """Agent 一次问答的完整结果。"""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0


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
        AgentResult：最终回答、工具调用轨迹、实际轮数。
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

    for round_index in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=tools,
            temperature=settings.agent_temperature,
        )
        llm_message = response.choices[0].message
        tool_calls = getattr(llm_message, "tool_calls", None)

        if not tool_calls:
            return AgentResult(
                answer=llm_message.content or "",
                tool_calls=trace,
                rounds=round_index,
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
    )
