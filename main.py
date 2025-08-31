import logging
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from llm_factory import get_llm

load_dotenv()


class ChatState(TypedDict):
    """State for the chatbot conversation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def chatbot_node(state: ChatState) -> dict:
    """Process messages through the LLM and return updated state."""
    try:
        llm = get_llm()
        # Get response from LLM
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Error in chatbot_node: {e}")
        error_message = AIMessage(content=f"I encountered an error: {str(e)}")
        return {"messages": [error_message]}


def chatbot_streaming_node(state: ChatState) -> dict:
    """Process messages through the LLM with streaming support."""
    try:
        llm = get_llm()
        # Stream response from LLM
        full_content = ""
        for chunk in llm.stream(state["messages"]):
            if hasattr(chunk, 'content') and chunk.content:
                full_content += chunk.content
                print(chunk.content, end="", flush=True)
        
        # Create the final AI message with complete content
        response = AIMessage(content=full_content)
        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Error in chatbot_streaming_node: {e}")
        error_message = AIMessage(content=f"I encountered an error: {str(e)}")
        return {"messages": [error_message]}


def create_workflow(streaming: bool = True) -> StateGraph:
    """Create and configure the LangGraph workflow."""
    workflow = StateGraph(ChatState)  # type: ignore
    if streaming:
        workflow.add_node("chatbot", chatbot_streaming_node)
    else:
        workflow.add_node("chatbot", chatbot_node)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)
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

                # Create human message
                human_message = HumanMessage(content=user_input)

                # Process through the graph
                print("🤖 Assistant: ", end="", flush=True)

                app.invoke(
                    {"messages": [human_message]}, config=config) # type: ignore
                
                # Add newline after streaming is complete
                print()

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Input ended. Goodbye!")
                break
            except Exception as e:
                logging.error(f"Error in conversation loop: {e}")
                print(f"❌ An error occurred: {e}")
                print("Please try again or type 'quit' to exit.")

    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        print(f"❌ Failed to start chatbot: {e}")
        raise


def main():
    """Main function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Suppress verbose HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    try:
        run_chatbot()

    except ValueError as ve:
        print(f"❌ Configuration error: {ve}")
        print("Please check your .env file and ensure API keys are properly set.")
        logging.error(f"Configuration error: {ve}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    print("=== LangGraph Chatbot ===")
    main()
