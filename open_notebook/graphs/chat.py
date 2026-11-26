import asyncio
import sqlite3
from typing import Annotated, Optional

from ai_prompter import Prompter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.domain.notebook import Notebook
from open_notebook.graphs.web_tools import WEB_TOOLS
from open_notebook.exceptions import IncompleteGenerationError, OpenNotebookError
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.error_classifier import classify_error
from open_notebook.utils.text_utils import extract_text_content


class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]
    web_search: Optional[bool]
    force_final: Optional[bool]


def _history_for_model(messages: list, keep_current_turn_tools: bool) -> list:
    """Drop tool plumbing the provider could reject.

    Tool calls / results from earlier turns are never replayed (keeps tokens low
    and avoids ``tool_use`` blocks without a matching tool definition). Within the
    current turn they are kept only when tools are bound to the model.
    """
    last_human = max(
        (i for i, m in enumerate(messages) if m.type == "human"), default=-1
    )
    out = []
    for i, m in enumerate(messages):
        is_tool_msg = m.type == "tool" or (
            isinstance(m, AIMessage) and bool(getattr(m, "tool_calls", None))
        )
        if is_tool_msg and not (keep_current_turn_tools and i > last_human):
            continue
        out.append(m)
    return out


def _flatten_tool_turn(messages: list) -> list:
    """Turn this turn's tool calls/results into plain text (no tool blocks).

    Used for the final answer when no tools are bound: some providers reject
    tool_use/tool_result blocks unless the request also defines tools.
    """
    last_human = max(
        (i for i, m in enumerate(messages) if m.type == "human"), default=-1
    )
    head, turn = messages[: last_human + 1], messages[last_human + 1 :]
    gathered = []
    for m in turn:
        if m.type == "tool":
            gathered.append(str(m.content)[:6000])
    if not gathered:
        return head
    notes = "\n\n---\n\n".join(gathered)
    return head + [
        AIMessage(content="(I searched the web; findings below.)\n\n" + notes)
    ]


def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    try:
        system_prompt = Prompter(prompt_template="chat/system").render(data=state)  # type: ignore[arg-type]
        use_web_tools = bool(state.get("web_search")) and not state.get("force_final")
        # force_final keeps this turn's search results so the answer can use them
        history = _history_for_model(
            state.get("messages", []),
            keep_current_turn_tools=bool(state.get("web_search")),
        )
        if state.get("force_final"):
            history = _flatten_tool_turn(history)
        payload = [SystemMessage(content=system_prompt)] + history
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        # Handle async model provisioning from sync context
        def run_in_new_loop():
            """Run the async function in a new event loop"""
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    provision_langchain_model(
                        str(payload), model_id, "chat", max_tokens=8192
                    )
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an event loop, run in a thread with a new loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                model = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            model = asyncio.run(
                provision_langchain_model(
                    str(payload),
                    model_id,
                    "chat",
                    max_tokens=8192,
                )
            )

        if use_web_tools:
            if not hasattr(model, "bind_tools"):
                raise OpenNotebookError(
                    "The selected model does not support web search tools. "
                    "Pick another model or turn web search off."
                )
            model = model.bind_tools(WEB_TOOLS)
            payload[0] = SystemMessage(
                content=system_prompt
                + "\n\n# WEB SEARCH\nYou can call web_search and web_fetch for "
                "current or external information. Prefer the CONTEXT first. "
                "When you use web results, cite the source URLs in your answer. "
                "Do not claim to have visited pages you did not fetch."
            )
        elif state.get("force_final"):
            payload.append(
                HumanMessage(
                    content="Stop searching. Answer now using the information "
                    "gathered so far, cite the URLs you used, and say what you "
                    "could not verify."
                )
            )

        ai_message = model.invoke(payload)

        if getattr(ai_message, "tool_calls", None):
            # Model wants tools: keep the message as-is, ToolNode runs them next.
            return {"messages": ai_message}

        # Clean thinking content from AI response (e.g., <think>...</think> tags)
        content = extract_text_content(ai_message.content)
        cleaned_content = clean_thinking_content(content)
        if not cleaned_content.strip():
            # Raising inside the node keeps LangGraph from checkpointing a
            # blank AI message that would be replayed as context later.
            raise IncompleteGenerationError(
                "The model returned an empty response. Try again, or pick a "
                "different model if this keeps happening."
            )
        cleaned_message = ai_message.model_copy(update={"content": cleaned_content})

        return {"messages": cleaned_message}
    except OpenNotebookError:
        raise
    except Exception as e:
        error_class, user_message = classify_error(e)
        raise error_class(user_message) from e


conn = sqlite3.connect(
    LANGGRAPH_CHECKPOINT_FILE,
    check_same_thread=False,
)
memory = SqliteSaver(conn)


def finalize_without_tools(state: ThreadState, config: RunnableConfig) -> dict:
    """Tool budget used up: ask the model for a final answer with no tools."""
    return call_model_with_messages({**state, "force_final": True}, config)  # type: ignore[arg-type]


MAX_WEB_TOOL_ROUNDS = 4


def route_after_agent(state: ThreadState) -> str:
    """Run tools only when web search is on and the model asked for them.

    Stops after MAX_WEB_TOOL_ROUNDS tool rounds in the current turn so a model
    that keeps calling tools cannot loop forever.
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if not (
        state.get("web_search")
        and isinstance(last, AIMessage)
        and getattr(last, "tool_calls", None)
    ):
        return END
    rounds = 0
    for msg in reversed(messages):
        if msg.type == "human":
            break
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            rounds += 1
    if rounds > MAX_WEB_TOOL_ROUNDS:
        return "finalize"
    return "tools"


agent_state = StateGraph(ThreadState)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_node("tools", ToolNode(WEB_TOOLS))
agent_state.add_node("finalize", finalize_without_tools)
agent_state.add_edge(START, "agent")
agent_state.add_conditional_edges(
    "agent", route_after_agent, ["tools", "finalize", END]
)
agent_state.add_edge("finalize", END)
agent_state.add_edge("tools", "agent")
graph = agent_state.compile(checkpointer=memory)
