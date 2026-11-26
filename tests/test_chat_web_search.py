"""Tests for the chat web-search feature (tools, routing, history hygiene)."""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from open_notebook.graphs import web_tools
from open_notebook.graphs.chat import (
    MAX_WEB_TOOL_ROUNDS,
    _flatten_tool_turn,
    _history_for_model,
    route_after_agent,
)


def _call(i: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": f"c{i}"}],
    )


def _tool(i: int) -> ToolMessage:
    return ToolMessage(content=f"result {i}", tool_call_id=f"c{i}")


# ---- web_tools ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url,ok",
    [
        ("https://example.com/a", True),
        ("http://news.site.id/x?y=1", True),
        ("file:///etc/passwd", False),
        ("ftp://example.com", False),
        ("http://localhost:5055/api", False),
        ("http://127.0.0.1:8000", False),
        ("http://10.0.0.5", False),
        ("http://192.168.1.1", False),
        ("http://172.20.0.2", False),
        ("http://169.254.169.254/latest", False),
        ("http://100.101.203.54:20128", False),
        ("http://surrealdb.internal", False),
        ("http://[::1]/", False),
        ("not a url", False),
    ],
)
def test_is_safe_public_url(url, ok):
    assert web_tools.is_safe_public_url(url) is ok


def test_parse_mcp_body_sse_and_json():
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "hello"}]}}
    assert web_tools._parse_mcp_body("data: " + json.dumps(payload)) == "hello"
    assert web_tools._parse_mcp_body(json.dumps(payload)) == "hello"


def test_parse_mcp_body_errors():
    err = {"jsonrpc": "2.0", "id": 1, "error": {"message": "rate limited"}}
    with pytest.raises(web_tools.WebToolError):
        web_tools._parse_mcp_body(json.dumps(err))
    is_err = {"result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}
    with pytest.raises(web_tools.WebToolError):
        web_tools._parse_mcp_body(json.dumps(is_err))
    with pytest.raises(web_tools.WebToolError):
        web_tools._parse_mcp_body("garbage")


def test_web_search_returns_text_and_trims():
    with patch.object(web_tools, "mcp_call", return_value="A" * 20000) as m:
        out = web_tools.web_search.invoke({"query": "harga gabah", "num_results": 99})
    assert len(out) == web_tools.MAX_RESULT_CHARS
    assert m.call_args[0][1]["numResults"] == 8  # clamped


def test_web_search_failure_is_reported_not_raised():
    with patch.object(web_tools, "mcp_call", side_effect=web_tools.WebToolError("429")):
        out = web_tools.web_search.invoke({"query": "x"})
    assert "failed" in out.lower() and "429" in out


def test_web_fetch_refuses_internal_without_calling_network():
    with patch.object(web_tools, "mcp_call") as m:
        out = web_tools.web_fetch.invoke({"url": "http://127.0.0.1:5055/api/notebooks"})
    assert "Refused" in out
    m.assert_not_called()


# ---- routing -----------------------------------------------------------------


def test_route_ends_when_web_off_or_no_tool_calls():
    assert route_after_agent({"web_search": False, "messages": [HumanMessage(content="q"), _call(1)]}) == END
    assert route_after_agent({"web_search": True, "messages": [HumanMessage(content="q"), AIMessage(content="done")]}) == END


def test_route_runs_tools_then_finalizes_at_cap():
    msgs = [HumanMessage(content="q")]
    for i in range(1, MAX_WEB_TOOL_ROUNDS + 1):
        msgs += [_call(i)]
        assert route_after_agent({"web_search": True, "messages": list(msgs)}) == "tools"
        msgs += [_tool(i)]
    msgs += [_call(99)]
    assert route_after_agent({"web_search": True, "messages": msgs}) == "finalize"


def test_route_counts_only_current_turn():
    old = [HumanMessage(content="old")] + [x for i in range(10) for x in (_call(i), _tool(i))] + [AIMessage(content="ok")]
    cur = old + [HumanMessage(content="new"), _call(50)]
    assert route_after_agent({"web_search": True, "messages": cur}) == "tools"


# ---- history hygiene ---------------------------------------------------------


def test_history_drops_previous_turn_tools_keeps_current():
    msgs = [
        HumanMessage(content="old"),
        _call(1),
        _tool(1),
        AIMessage(content="old answer"),
        HumanMessage(content="new"),
        _call(2),
        _tool(2),
    ]
    kept = _history_for_model(msgs, keep_current_turn_tools=True)
    assert [m.type for m in kept] == ["human", "ai", "human", "ai", "tool"]
    assert kept[1].content == "old answer"
    no_tools = _history_for_model(msgs, keep_current_turn_tools=False)
    assert all(m.type in ("human", "ai") and not getattr(m, "tool_calls", None) for m in no_tools)


def test_flatten_turns_results_into_plain_text():
    msgs = [HumanMessage(content="q"), _call(1), _tool(1)]
    flat = _flatten_tool_turn(msgs)
    assert [m.type for m in flat] == ["human", "ai"]
    assert "result 1" in flat[1].content and not getattr(flat[1], "tool_calls", None)
    assert _flatten_tool_turn([HumanMessage(content="q")]) == [HumanMessage(content="q")]
