"""工具注册表与工具执行测试（纯逻辑部分，不依赖数据库）。"""

from app.tools.registry import TOOL_REGISTRY, execute_tool, get_tool_specs


def test_registry_contains_all_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "search_documents",
        "list_sources",
        "get_document_content",
    }


def test_tool_specs_shape() -> None:
    specs = get_tool_specs()
    names = [spec["function"]["name"] for spec in specs]
    assert set(names) == set(TOOL_REGISTRY)
    for spec in specs:
        assert spec["type"] == "function"
        assert "parameters" in spec["function"]


def test_execute_unknown_tool_returns_error() -> None:
    assert "error" in execute_tool("no_such_tool", "{}")


def test_execute_tool_handles_bad_json() -> None:
    assert "error" in execute_tool("search_documents", "{not-json")


def test_execute_tool_handles_missing_required_argument() -> None:
    # 缺少必填参数 query，应被兜底为 error 而不是抛异常
    assert "error" in execute_tool("search_documents", "{}")
