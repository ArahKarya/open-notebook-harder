"""Web search / fetch tools for the notebook chat agent.

Backed by the public, keyless Exa MCP endpoint (no API key, no extra RAM).
Tools return plain text; the chat model is told to cite the URLs it used.
Results are only used to answer - nothing is stored in the notebook.
"""

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool
from loguru import logger

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
TIMEOUT_SECONDS = 30
MAX_RESULT_CHARS = 6000
MAX_FETCH_CHARS = 8000


class WebToolError(RuntimeError):
    pass


def _parse_mcp_body(body: str) -> str:
    """First text item of a JSON-RPC ``tools/call`` response (JSON or SSE body)."""
    stripped = body.strip()
    candidates = [stripped] if stripped.startswith("{") else []
    candidates += [
        line[len("data: ") :]
        for line in re.split(r"\r\n|\r|\n", body)
        if line.startswith("data: ")
    ]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if err := data.get("error"):
            raise WebToolError(str(err.get("message") or err))
        result = data.get("result") or {}
        texts = [
            c.get("text", "")
            for c in result.get("content") or []
            if isinstance(c, dict)
        ]
        if result.get("isError"):
            raise WebToolError(" ".join(t for t in texts if t) or "tool call failed")
        text = next((t for t in texts if t), "")
        if text:
            return text
    raise WebToolError("unrecognized or empty response from web search service")


def mcp_call(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "open-notebook-harder",
    }
    try:
        response = httpx.post(
            EXA_MCP_URL, json=payload, headers=headers, timeout=TIMEOUT_SECONDS
        )
    except httpx.HTTPError as exc:
        raise WebToolError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise WebToolError(f"HTTP {response.status_code}: {response.text[:200]}")
    return _parse_mcp_body(response.content.decode("utf-8", errors="replace"))


def is_safe_public_url(url: str) -> bool:
    """Reject non-http(s) and obvious internal targets before asking the fetcher."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return False
    if host in ("localhost", "metadata.google.internal") or host.endswith(
        (".local", ".internal", ".lan")
    ):
        return False
    if re.fullmatch(r"[\d.]+", host):
        parts = [int(p) for p in host.split(".") if p.isdigit()]
        if len(parts) != 4:
            return False
        a, b = parts[0], parts[1]
        if (
            a in (0, 10, 127)
            or (a == 169 and b == 254)
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
            or (a == 100 and 64 <= b <= 127)
        ):
            return False
    if ":" in host:  # IPv6 literal
        return False
    return True


@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the public web for current information.

    Use for recent facts, prices, news, regulations or anything not present in
    the notebook sources. Returns titles, URLs and highlights. Always cite the
    URLs you rely on.
    """
    num = max(1, min(int(num_results), 8))
    try:
        text = mcp_call("web_search_exa", {"query": query, "numResults": num})
    except WebToolError as exc:
        logger.warning(f"web_search failed: {exc}")
        return f"Web search failed: {exc}. Tell the user the web search is unavailable."
    return text[:MAX_RESULT_CHARS]


@tool
def web_fetch(url: str) -> str:
    """Fetch and read the text of one public web page (after web_search)."""
    if not is_safe_public_url(url):
        return "Refused: only public http(s) URLs can be fetched."
    try:
        text = mcp_call("web_fetch_exa", {"urls": [url]})
    except WebToolError as exc:
        logger.warning(f"web_fetch failed for {url}: {exc}")
        return f"Could not fetch {url}: {exc}"
    return text[:MAX_FETCH_CHARS]


WEB_TOOLS = [web_search, web_fetch]
