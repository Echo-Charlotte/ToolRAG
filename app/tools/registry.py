"""工具注册表：把工具包装成 OpenAI Function Calling 能识别的 schema。"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.tools.rag_tools import get_document_content, list_sources, search_documents


@dataclass(frozen=True)
class Tool:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_openai_spec(self) -> dict[str, Any]:
        """转成 OpenAI tools 参数里的一项。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_SEARCH_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "自然语言查询"},
        "top_k": {
            "type": "integer",
            "description": "返回条数，1-20，默认 5",
            "minimum": 1,
            "maximum": 20,
        },
        "mode": {
            "type": "string",
            "enum": ["vector", "keyword", "hybrid"],
            "description": "检索模式，默认 hybrid",
        },
    },
    "required": ["query"],
}

_TOOLS: list[Tool] = [
    Tool(
        name="search_documents",
        description="在内部文档库中检索（向量+关键词混合），返回最相关片段及其来源，适合回答文档内容相关问题。",
        parameters=_SEARCH_PARAMS,
        func=search_documents,
    ),
    Tool(
        name="list_sources",
        description="列出文档库中所有文档来源及各自切块数量，用于了解当前有哪些文档。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=list_sources,
    ),
    Tool(
        name="get_document_content",
        description="按 source 获取某篇文档的完整内容（按切块顺序），用于需要通读整篇文档的场景。",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "文档来源标识，如文件名"}
            },
            "required": ["source"],
        },
        func=get_document_content,
    ),
]

TOOL_REGISTRY: dict[str, Tool] = {tool.name: tool for tool in _TOOLS}


def get_tool_specs() -> list[dict[str, Any]]:
    """返回全部工具的 OpenAI tools 规格。"""
    return [tool.to_openai_spec() for tool in _TOOLS]


def execute_tool(name: str, arguments: str | dict[str, Any] | None) -> Any:
    """执行指定工具。

    任何异常都会被转成 ``{"error": ...}`` 返回，保证 LLM 工具循环不会因单次失败中断。

    Args:
        name: 工具名。
        arguments: 工具参数，可以是 JSON 字符串或已解析的字典。

    Returns:
        工具执行结果（通常是 dict）。
    """
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"error": f"未知工具: {name}"}

    if arguments is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"error": f"参数不是合法 JSON: {exc}"}
    else:
        kwargs = arguments

    try:
        return tool.func(**kwargs)
    except TypeError as exc:
        return {"error": f"参数不匹配: {exc}"}
    except Exception as exc:  # noqa: BLE001 - 工具异常需兜底返回
        return {"error": f"工具执行失败: {exc}"}
