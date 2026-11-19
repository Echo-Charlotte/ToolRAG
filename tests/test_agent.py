"""Function Calling Agent 循环测试（用假 LLM 客户端，不依赖网络）。"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent import tool_agent


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeCompletions:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("假客户端脚本已用尽")
        return self._script.pop(0)


class _FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.completions = _FakeCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def patched_tool(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []

    def fake_execute(name: str, arguments: Any) -> dict[str, Any]:
        records.append((name, arguments))
        return {"results": [{"source": "fruits.md", "chunk_index": 0, "content": "苹果是水果。"}]}

    monkeypatch.setattr(tool_agent, "execute_tool", fake_execute)
    return records


def test_agent_executes_tool_then_answers(patched_tool: list) -> None:
    script = [
        _response(
            tool_calls=[_tool_call("call_1", "search_documents", json.dumps({"query": "苹果"}))]
        ),
        _response(content="苹果是一种水果（来源：fruits.md）。"),
    ]
    client = _FakeClient(script)

    result = tool_agent.run_agent("苹果是什么？", client=client)

    assert "苹果" in result.answer
    assert result.rounds == 2
    assert [call["name"] for call in result.tool_calls] == ["search_documents"]
    assert result.tool_calls[0]["arguments"] == {"query": "苹果"}
    # 第二次调用 LLM 时，messages 里应包含回填的 tool 结果
    second_messages = client.completions.calls[1]["messages"]
    assert any(msg.get("role") == "tool" for msg in second_messages)
    assert client.completions.calls[0]["tools"]  # tools 规格已传入


def test_agent_returns_directly_when_no_tool_call(patched_tool: list) -> None:
    client = _FakeClient([_response(content="你好，有什么可以帮你？")])

    result = tool_agent.run_agent("你好", client=client)

    assert result.rounds == 1
    assert result.tool_calls == []
    assert patched_tool == []


def test_agent_stops_at_max_rounds(patched_tool: list) -> None:
    script = [
        _response(tool_calls=[_tool_call("call_x", "list_sources", "{}")]) for _ in range(3)
    ]
    client = _FakeClient(script)

    result = tool_agent.run_agent("循环测试", client=client, max_rounds=3)

    assert result.rounds == 3
    assert "最大工具调用轮数" in result.answer
    assert len(result.tool_calls) == 3


def test_agent_handles_multiple_tools_in_one_round(patched_tool: list) -> None:
    script = [
        _response(
            tool_calls=[
                _tool_call("c1", "search_documents", json.dumps({"query": "苹果"})),
                _tool_call("c2", "list_sources", "{}"),
            ]
        ),
        _response(content="完成"),
    ]
    client = _FakeClient(script)

    result = tool_agent.run_agent("问题", client=client)

    assert [call["name"] for call in result.tool_calls] == ["search_documents", "list_sources"]
    assert result.rounds == 2
