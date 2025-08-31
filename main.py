import logging
import uuid
import os
from typing import TypedDict, Annotated, Sequence, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from llm_factory import get_llm
from tools import get_search_tools
from utils import (
    sanitize_user_input,
    InputValidationError,
    SimpleRateLimiter
)

load_dotenv()

# Initialize rate limiter for production protection
rate_limiter = SimpleRateLimiter(max_requests=100, window_seconds=3600)


class ChatState(TypedDict):
    """State for the chatbot conversation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _should_enable_web_search(message: BaseMessage | str) -> bool:
    """Heuristic to decide if web search should be encouraged.

    Expands beyond explicit phrases to include time-sensitive intents and
    URL presence. This does not force a search; it simply guides the model.
    """
    text = ""
    if isinstance(message, str):
        text = message.lower()
    elif hasattr(message, "content"):
        text = str(message.content).lower()
    else:
        try:
            text = str(message).lower()
        except Exception:
            text = ""

    explicit_triggers = [
        "search online",
        "search the web",
        "look up",
        "browse",
        "do an online search",
        "search online for",
        "please search online",
        "please look up",
        "please search the web",
    ]

    # Heuristics for time-sensitive topics
    time_sensitive_terms = [
        "today",
        "current",
        "latest",
        "recent",
        "news",
        "release date",
        "stock",
        "price",
        "weather",
        "schedule",
        "score",
        "version",
        "update",
        "rumor",
        "rumors",
        "leak",
        "leaks",
        "announced",
        "2023",
        "2024",
        "2025",
    ]

    # Presence of a URL likely means the user wants context from the web
    has_url = ("http://" in text) or ("https://" in text) or ("www." in text)

    if any(trigger in text for trigger in explicit_triggers):
        return True
    if has_url:
        return True
    if any(term in text for term in time_sensitive_terms):
        return True
    return False


def _system_instruction(enable_search_hint: bool) -> SystemMessage:
    """Build a system instruction guiding web_search usage.

    If enable_search_hint is True, nudge the model to consider using the
    web_search tool for time-sensitive or uncertain queries.
    """
    base = (
        "You are a helpful assistant. You have access to a `web_search` tool "
        "that retrieves CURRENT information from the web with brief summaries "
        "and sources. Use it when the user explicitly asks you to search "
        "online, provides a URL, the question is time-sensitive (e.g., news, "
        "prices, schedules, releases), or when you are not reasonably certain "
        "you can answer accurately from general knowledge. Keep queries "
        "concise and specific. If the query is timeless and you are confident, "
        "you may answer without calling the tool."
    )
    if enable_search_hint:
        base += (
            " For this conversation, the user's request appears time-sensitive "
            "or requires up-to-date facts. You must call `web_search` before "
            "answering, verify the information from the results, and cite the "
            "source(s) you used."
        )
    return SystemMessage(content=base)


def chatbot_node(state: ChatState) -> dict:
    """Process messages through the LLM and return updated state.

    Tools (web search) will only be bound when the last user message
    explicitly requests an online search as determined by
    _explicit_online_search_requested.
    """
    try:
        llm = get_llm(model=os.getenv("OPENAI_MODEL"))

        # Build messages with a system instruction to guide tool use
        last_msg = state["messages"][-1]
        sys_msg = _system_instruction(
            enable_search_hint=_should_enable_web_search(last_msg)
        )
        messages_for_llm = [sys_msg] + list(state["messages"])  # type: ignore

        # Bind tools by default so the model can choose to call them
        tools = get_search_tools()
        if tools:
            llm_runnable = llm.bind_tools(tools)
        else:
            llm_runnable = llm

        # Get response from LLM (tools available if configured)
        response = llm_runnable.invoke(messages_for_llm)

        # Fix tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            fixed_tool_calls = []
            for i, tool_call in enumerate(response.tool_calls):
                # Convert to dict if needed
                if hasattr(tool_call, 'dict'):
                    tc_dict = tool_call.dict()
                elif isinstance(tool_call, dict):
                    tc_dict = tool_call.copy()
                else:
                    tc_dict = {"name": str(tool_call), "args": {}}

                # tool call inspected

                # Ensure name and ID exist and are valid
                if not tc_dict.get("name") or tc_dict.get("name") == "":
                    tc_dict["name"] = "web_search"
                if not tc_dict.get("id"):
                    tc_dict["id"] = f"call_{uuid.uuid4().hex[:8]}_{i}"

                fixed_tool_calls.append(tc_dict)

            response.tool_calls = fixed_tool_calls

        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Error in chatbot_node: {e}")
        error_message = AIMessage(content=f"I encountered an error: {str(e)}")
        return {"messages": [error_message]}


def chatbot_streaming_node(state: ChatState) -> dict:
    """Process messages through the LLM with streaming support."""
    try:
        # Log the call with message count without exceeding line length
        msg_count = len(state["messages"])
        # Critical event: track if streaming is invoked
        logging.info("Streaming node called - messages=%d", msg_count)
        # Use the environment-configured model to ensure consistency with
        # non-streaming calls. If OPENAI_MODEL is not set, get_llm will
        # fall back to the default from configuration.
        llm = get_llm(model=os.getenv("OPENAI_MODEL"))

        # Build messages with a system instruction to guide tool use
        last_msg = state["messages"][-1]
        sys_msg = _system_instruction(
            enable_search_hint=_should_enable_web_search(last_msg)
        )
        messages_for_llm = [sys_msg] + list(state["messages"])  # type: ignore

        tools = get_search_tools()  # Get available tools
        llm_with_tools = (
            llm.bind_tools(tools) if tools else llm
        )

        # Stream response from LLM
        full_content = ""
        tool_calls = []

        for chunk in llm_with_tools.stream(messages_for_llm):
            if hasattr(chunk, 'content') and chunk.content:
                full_content += chunk.content
                print(chunk.content, end="", flush=True)

            # Collect tool calls if present
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
                # Avoid long inline f-strings in logs; use structured args
                logging.debug(
                    "Received tool calls in chunk: %s", chunk.tool_calls
                )

        # Fix tool calls by ensuring they all have IDs and names
        if tool_calls:
            fixed_tool_calls = []
            for i, tool_call in enumerate(tool_calls):
                # Convert to dict if needed
                if hasattr(tool_call, 'dict'):
                    tc_dict = tool_call.dict()
                elif isinstance(tool_call, dict):
                    tc_dict = tool_call.copy()
                else:
                    tc_dict = {"name": str(tool_call), "args": {}}

                # Ensure name and ID exist and are valid
                if not tc_dict.get("name") or tc_dict.get("name") == "":
                    tc_dict["name"] = "web_search"
                if not tc_dict.get("id"):
                    tc_dict["id"] = f"call_{uuid.uuid4().hex[:8]}_{i}"

                fixed_tool_calls.append(tc_dict)

            response = AIMessage(
                content=full_content,
                tool_calls=fixed_tool_calls,
            )
        else:
            response = AIMessage(content=full_content)

        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Error in chatbot_streaming_node: {e}")
        error_message = AIMessage(content=f"I encountered an error: {str(e)}")
        return {"messages": [error_message]}


def tools_node(state: ChatState) -> dict:
    """Execute tools based on the last message's tool calls.

    Enhanced error handling and robust argument parsing.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logging.warning("No tool calls found in last message")
        return {"messages": []}

    tools = get_search_tools()
    if not tools:
        error_msg = (
            "No search tools available. Please check your configuration."
        )
        logging.error(error_msg)
        return {
            "messages": [
                ToolMessage(
                    content=error_msg,
                    tool_call_id="search_error",
                    name="web_search",
                )
            ]
        }

    tools_by_name = {tool.name: tool for tool in tools}
    logging.info(f"Available tools: {list(tools_by_name.keys())}")

    tool_messages = []
    for i, tool_call in enumerate(last_message.tool_calls):
        tool_name = "unknown"
        tool_id = f"call_{uuid.uuid4().hex[:8]}_{i}"

        try:
            result = "Tool execution completed"  # Default value

            # Enhanced tool call parsing with validation
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                provided_id = tool_call.get("id")
            else:
                # Handle attribute-based tool calls
                tool_name = getattr(tool_call, 'name', 'unknown')
                tool_args = getattr(tool_call, 'args', {})
                provided_id = getattr(tool_call, 'id', None)

            if provided_id:
                tool_id = provided_id

            logging.info("Executing tool: %s with ID: %s", tool_name, tool_id)

            if tool_name not in tools_by_name:
                available = list(tools_by_name.keys())
                result = (
                    "Tool '%s' not found. Available tools: %s"
                    % (tool_name, available)
                )
                logging.warning(result)
            else:
                tool = tools_by_name[tool_name]

                # Enhanced argument parsing with multiple fallbacks
                query = None
                logging.debug(
                    "Tool args type: %s, value: %s",
                    type(tool_args),
                    tool_args,
                )

                if isinstance(tool_args, dict):
                    # Primary: look for 'query' key
                    query = tool_args.get("query")
                    logging.debug(f"Query from dict 'query': {query}")

                    # Secondary: try other common argument names
                    if not query:
                        for arg_name in ["q", "search_query", "text", "input"]:
                            query = tool_args.get(arg_name)
                            logging.debug(
                                "Query from dict '%s': %s", arg_name, query
                            )
                            if query:
                                break

                    # Tertiary: if args dict has a single key-value pair,
                    # use that value as the query.
                    if not query and len(tool_args) == 1:
                        single_value = next(iter(tool_args.values()))
                        if (
                            isinstance(single_value, str)
                            and single_value.strip()
                        ):
                            query = single_value
                            logging.debug(
                                "Query from single dict value: %s", query
                            )

                elif isinstance(tool_args, str):
                    query = tool_args
                    logging.debug(f"Query from string: {query}")

                elif isinstance(tool_args, list):
                    # Handle case where args is a list
                    if tool_args and isinstance(tool_args[0], str):
                        query = tool_args[0]
                        logging.debug(f"Query from list: {query}")

                else:
                    # Last resort: convert to string
                    query = str(tool_args).strip()
                    logging.debug(
                        "Query from other type (converted): %s", query
                    )

                # Validate the query
                if not query or not query.strip():
                    # Try one more approach: look for the query in the entire
                    # tool call.
                    if isinstance(tool_call, dict):
                        # Check if the query is in the raw tool call somewhere.
                        full_call_str = str(tool_call)
                        # Look for patterns like "query", "search", etc.
                        # followed by meaningful text. If the call has
                        # some content, use the entire call as a fallback.
                        if len(full_call_str) > 10:
                            query = full_call_str

                    if not query or not query.strip():
                        result = (
                            "No valid search query provided. Received args: %s"
                            % (tool_args,)
                        )
                        logging.warning(
                            "Empty or invalid search query provided. Args: %s",
                            tool_args,
                        )
                else:
                    # Keep prints short to satisfy style checks
                    print(
                        "\n🔍 Searching for: '%s'..." % (query,),
                        end="",
                        flush=True,
                    )
                    start_time = __import__("time").time()

                    result = tool.func(query)
 
                    elapsed = __import__("time").time() - start_time
                    logging.info(
                        "Search completed in %.2fs: %s", elapsed, tool_name
                    )
                    print(" ✅ Done (%.1fs)" % (elapsed,), flush=True)

            # Create properly formatted tool message
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)

        except Exception as e:
            error_msg = "Tool execution failed: %s" % (str(e),)
            logging.error(
                "Error in tools_node for %s (ID: %s): %s",
                tool_name,
                tool_id,
                e,
            )

            tool_message = ToolMessage(
                content=error_msg,
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)

    logging.info(
        "Tools node returning %d tool messages", len(tool_messages)
    )
    for i, msg in enumerate(tool_messages):
        logging.debug("Tool message %d: %s...", i, msg.content[:200])

    return {"messages": tool_messages}


def should_continue(state: ChatState) -> Literal["tools", "end"]:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    logging.info(
        "Checking should_continue: last message type = %s", type(last_message)
    )
 
    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logging.info(
            "Found %d tool calls, continuing to tools",
            len(last_message.tool_calls),
        )
        return "tools"
 
    logging.info("No tool calls found, ending workflow")
    return "end"


def create_workflow(streaming: bool = True) -> StateGraph:
    """Create and configure the LangGraph workflow."""
    workflow = StateGraph(ChatState)  # type: ignore

    # Add nodes
    if streaming:
        workflow.add_node("chatbot", chatbot_streaming_node)
    else:
        workflow.add_node("chatbot", chatbot_node)

    # Add custom tools node
    workflow.add_node("tools", tools_node)

    # Set entry point
    workflow.set_entry_point("chatbot")

    # Add conditional edges for tool calling
    workflow.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Add edge from tools back to chatbot
    workflow.add_edge("tools", "chatbot")

    return workflow


def run_chatbot():
    """Main chatbot execution function."""
    try:
        # Create and compile the workflow
        workflow = create_workflow()
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)

        # Configuration for conversation persistence
        config = {"configurable": {"thread_id": "main-conversation"}}

        print("🤖 LangGraph Chatbot initialized!")
        print("🔍 Web search capability enabled - I can search for current")
        print("information when needed.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("-" * 50)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Validate and sanitize user input
                try:
                    # Check rate limits (using a simple identifier)
                    if not rate_limiter.is_allowed("user"):
                        print("❌ Rate limit exceeded. Please wait before "
                              "sending more messages.")
                        continue
                    
                    # Sanitize input for security
                    sanitized_input = sanitize_user_input(user_input)
                    
                    # Create human message with sanitized input
                    human_message = HumanMessage(content=sanitized_input)
                    
                except InputValidationError as e:
                    print(f"❌ Input validation error: {e}")
                    logging.warning("Input validation failed: %s", e)
                    continue

                # Process through the graph
                print("🤖 Assistant: ", end="", flush=True)
 
                _ = app.invoke(
                    {"messages": [human_message]},  # type: ignore
                    config=config,  # type: ignore
                )

                # Add newline after streaming is complete
                # (removed debug marker)
                print()

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Input ended. Goodbye!")
                break
            except Exception as e:
                logging.error("Error in conversation loop: %s", e)
                print("❌ An error occurred: %s" % (e,))
                print("Please try again or type 'quit' to exit.")

    except Exception as e:
        logging.error("Failed to initialize chatbot: %s", e)
        print("❌ Failed to start chatbot: %s" % (e,))
        raise


def main():
    """Main function."""
    # Configure production-appropriate logging
    log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose third-party library logs for production
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("duckduckgo_search").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("langchain_core").setLevel(logging.ERROR)
    logging.getLogger("langchain_openai").setLevel(logging.ERROR)
    logging.getLogger("langchain_community").setLevel(logging.ERROR)
    logging.getLogger("langgraph").setLevel(logging.ERROR)

    try:
        run_chatbot()

    except ValueError as ve:
        print("❌ Configuration error: %s" % (ve,))
        print(
            "Please check your .env file and ensure API keys are "
            "properly set."
        )
        logging.error("Configuration error: %s", ve)

    except Exception as e:
        print("❌ Unexpected error: %s" % (e,))
        logging.error("Unexpected error: %s", e)


if __name__ == "__main__":
    print("=== LangGraph Chatbot ===")
    main()
